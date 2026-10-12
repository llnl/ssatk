"""Independent celestial-frame reference for the exported browser sky."""

from datetime import datetime

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, TEME
from astropy.time import Time

from ssatk.plots.build_satellite_viewer import load_star_catalog
from ssatk.plots.starfield import star_directions


@pytest.mark.parametrize("year", [2008, 2026])
def test_exported_stars_match_astropy_teme(year):
    """R2: Astropy's independent GCRS->TEME chain, within 0.5 arcsec.

    The tolerance includes SSAPy's Earth-orientation approximation and the
    export's seven-decimal direction rounding, not a second frame rotation.
    """
    epoch = datetime(year, 10, 11)
    source = star_directions(mag_limit=2.0, when=epoch, frame="gcrf")
    assert source is not None, "The required ssa-data-core star catalog is absent"
    inertial = np.asarray(source[0])
    reference = GCRS(
        CartesianRepresentation(inertial.T * u.m), obstime=Time(epoch)
    ).transform_to(TEME(obstime=Time(epoch))).cartesian.xyz.value.T
    actual = np.asarray(load_star_catalog(mag_limit=2.0, when=epoch)["v"]).reshape(-1, 3)
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    actual /= np.linalg.norm(actual, axis=1)[:, None]
    error_rad = np.arctan2(
        np.linalg.norm(np.cross(actual, reference), axis=1),
        np.sum(actual * reference, axis=1),
    )
    assert np.max(error_rad) < np.deg2rad(0.5 / 3600)
