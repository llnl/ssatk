"""Catalogue matching and angles-only attitude/landmark navigation.

Star directions constrain attitude, not translation, in the distant-star
model. A position solution requires bearings to known finite-distance points.
Rotations map camera axes (+z boresight) to GCRF-aligned axes; SI units apply.
These functions accept measured centroids/bearings, not raw camera images.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Backward-compatible import; finite-distance geometry belongs with landmarks.
from .landmarks import position_from_landmark_bearings


@dataclass(frozen=True)
class StarCatalogue:
    """J2000 ICRS astrometry; row IDs refer to the source CSV, not HIP IDs."""

    row_ids: np.ndarray
    names: np.ndarray
    ra_hours: np.ndarray
    dec_deg: np.ndarray
    pmra_mas_yr: np.ndarray
    pmdec_mas_yr: np.ndarray
    magnitude: np.ndarray
    source: str
    sha256: str


def load_star_catalogue(mag_limit=6.5, catalog_path=None):
    """Read SSATK's packaged bright-star table, without a synthetic fallback.

    Input convention: J2000 ICRS RA in hours, Dec in degrees, pmra=mu_alpha*
    and pmdec in mas/Julian year. The shipped extract lacks parallaxes,
    radial velocities and catalogue covariance; these cannot be reconstructed.
    """
    import pandas as pd
    from ssatk.data import data_path

    if not np.isfinite(mag_limit):
        raise ValueError("mag_limit must be finite")
    if catalog_path is None:
        with data_path("bright_stars_mag9.csv") as path:
            raw = Path(path).read_bytes()
        source = "ssa-data-core/bright_stars_mag9.csv"
    else:
        path = Path(catalog_path).expanduser()
        raw, source = path.read_bytes(), path.name
    df = pd.read_csv(BytesIO(raw))
    fields = ["ra", "dec", "pmra", "pmdec", "mag"]
    if not set(fields).issubset(df.columns):
        raise ValueError("catalogue requires ra, dec, pmra, pmdec and mag")
    values = df[fields].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("catalogue contains nonfinite astrometry")
    if not ((df.ra >= 0) & (df.ra < 24) & (df.dec.abs() <= 90)).all():
        raise ValueError("catalogue requires RA in [0,24) hours and Dec in [-90,90] degrees")
    df = df[(df.mag < mag_limit) & (df.mag > -10)]  # exclude the Sol record
    if len(df) < 3:
        raise ValueError("catalogue must contain at least three selected stars")
    names = df["proper"].fillna("").to_numpy(str) if "proper" in df else np.full(len(df), "")
    ids = df.index.to_numpy(int)
    names = np.array([name or f"CSV row {row}" for name, row in zip(names, ids)])
    return StarCatalogue(ids, names, *(df[field].to_numpy(float) for field in fields[:-1]),
                         df.mag.to_numpy(float), source, hashlib.sha256(raw).hexdigest())


def catalogue_directions(catalogue, time, *, observer_position_m=None, observer_velocity_m_s=None):
    """Epoch-propagated unit directions, optionally apparent at a spacecraft.

    Without an observer: astrometric ICRS directions, zero parallax/radial
    velocity. With an observer: Astropy ICRS->GCRS includes aberration and solar
    light deflection using its builtin ephemeris. Both observer position and
    velocity are required in geocentric GCRF-aligned SI axes. This is a
    direction-at-infinity approximation, not a parallax-capable catalogue.
    """
    import astropy.units as u
    from astropy.coordinates import GCRS, SkyCoord
    from astropy.time import Time
    from ssatk.coordinates.star_astrometry import proper_motion_directions

    t = time if isinstance(time, Time) else Time(time, format="gps")
    if not t.isscalar:
        raise ValueError("time must be one epoch")
    if (observer_position_m is None) != (observer_velocity_m_s is None):
        raise ValueError("observer position and velocity must be supplied together")
    years = (t.tdb - Time("J2000").tdb).to_value(u.yr)
    directions = proper_motion_directions(catalogue.ra_hours, catalogue.dec_deg,
        catalogue.pmra_mas_yr, catalogue.pmdec_mas_yr, years)
    c = SkyCoord(x=directions[:, 0], y=directions[:, 1], z=directions[:, 2],
                 representation_type="cartesian", frame="icrs")
    # A unit-spherical coordinate denotes an infinitely distant source. A
    # dimensionless Cartesian ICRS coordinate must not be read as a distance.
    c = SkyCoord(ra=c.spherical.lon, dec=c.spherical.lat, frame="icrs")
    if observer_position_m is not None:
        r, v = np.asarray(observer_position_m, float), np.asarray(observer_velocity_m_s, float)
        if r.shape != (3,) or v.shape != (3,) or not np.isfinite([r, v]).all():
            raise ValueError("observer position/velocity must be finite 3-vectors")
        c = c.transform_to(GCRS(obstime=t, obsgeoloc=r*u.m, obsgeovel=v*u.m/u.s))
    return c.cartesian.xyz.value.T


def _unit_rows(vectors, name):
    vectors = np.asarray(vectors, float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must be a finite (N,3) array")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} must contain nonzero directions")
    return vectors / norms[:, None]


def fit_star_attitude(camera_directions, reference_directions):
    """Wahba/Kabsch equal-weight solution, camera->reference rotation.

    Correspondences must already be known. At least two nonparallel bearings
    are required; degenerate geometry raises instead of fabricating attitude.
    """
    body = _unit_rows(camera_directions, "camera_directions")
    reference = _unit_rows(reference_directions, "reference_directions")
    if body.shape != reference.shape or len(body) < 2:
        raise ValueError("at least two corresponding directions are required")
    if min(np.linalg.svd(body, compute_uv=False)[1], np.linalg.svd(reference, compute_uv=False)[1]) < 1e-6:
        raise ValueError("star directions are parallel or ill-conditioned")
    left, _, right = np.linalg.svd(reference.T @ body)
    return left @ np.diag([1., 1., np.linalg.det(left @ right)]) @ right


def match_star_directions(camera_directions, reference_directions, prior_rotation, gate_rad):
    """One-to-one angular matching with an existing coarse attitude prior.

    This is tracking-mode acquisition, not blind lost-in-space identification.
    No simulated star identity is required. Returns measured and catalogue
    row indices; angular outliers beyond the gate remain unmatched.
    """
    body = _unit_rows(camera_directions, "camera_directions")
    reference = _unit_rows(reference_directions, "reference_directions")
    prior = np.asarray(prior_rotation, float)
    if prior.shape != (3, 3) or not np.isfinite(prior).all() or not np.allclose(prior.T@prior, np.eye(3), atol=1e-10, rtol=0) or np.linalg.det(prior) < 0:
        raise ValueError("prior_rotation must be a proper rotation")
    if not np.isfinite(gate_rad) or not 0 < gate_rad < np.pi:
        raise ValueError("gate_rad must be between zero and pi")
    # Unit-vector chord distance is monotonic in separation, including near 0.
    cost = np.linalg.norm((body @ prior.T)[:, None, :] - reference[None, :, :], axis=2)
    gate = 2*np.sin(gate_rad/2)
    # Private dummy columns allow rejecting outliers instead of forcing them
    # to consume a real catalogue match in the assignment.
    augmented = np.column_stack([np.where(cost < gate, cost, 1e6), np.full((len(body), len(body)), gate)])
    measured, cat = linear_sum_assignment(augmented)
    keep = cat < len(reference)
    return measured[keep], cat[keep]
