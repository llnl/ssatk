"""Independent celestial-frame reference for the exported browser sky."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, SkyCoord, TEME
from astropy.time import Time

from ssatk.plots.build_satellite_viewer import load_star_catalog
from ssatk.plots.starfield import _catalog_path


@pytest.mark.parametrize("year", [2008, 2026])
@pytest.mark.parametrize("frame", ["teme-of-date", "gcrf"])
def test_exported_stars_match_astropy(year, frame):
    """R2: Astropy proper motion and GCRS->TEME chain, within 0.5 arcsec.

    The tolerance includes SSAPy's Earth-orientation approximation and the
    export's seven-decimal direction rounding, not a second frame rotation.
    """
    epoch = datetime(year, 10, 11)
    catalog = pd.read_csv(_catalog_path()).dropna(subset=["ra", "dec", "mag"])
    catalog = catalog[(catalog["mag"] < 2.0) & (catalog["mag"] > -10)]
    inertial = SkyCoord(
        ra=catalog["ra"].to_numpy() * 15 * u.deg, dec=catalog["dec"].to_numpy() * u.deg,
        distance=1e6 * u.pc, pm_ra_cosdec=catalog["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=catalog["pmdec"].to_numpy() * u.mas / u.yr,
        frame="icrs", obstime=Time("J2000"),
    ).apply_space_motion(new_obstime=Time(epoch)).cartesian.xyz.value.T
    reference = inertial if frame == "gcrf" else GCRS(
        CartesianRepresentation(inertial.T * u.m), obstime=Time(epoch)
    ).transform_to(TEME(obstime=Time(epoch))).cartesian.xyz.value.T
    actual = np.asarray(load_star_catalog(mag_limit=2.0, when=epoch, frame=frame)["v"]).reshape(-1, 3)
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    actual /= np.linalg.norm(actual, axis=1)[:, None]
    error_rad = np.arctan2(
        np.linalg.norm(np.cross(actual, reference), axis=1),
        np.sum(actual * reference, axis=1),
    )
    assert np.max(error_rad) < np.deg2rad(0.5 / 3600)
