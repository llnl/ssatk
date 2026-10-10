"""Planet and Sun positions from SSAPy bodies against astropy's ephemeris."""

import numpy as np
import pytest
from astropy.time import Time

from ssatk.environment import SpaceEnvironment
from ssatk.plots.sun_mpl import get_sun_position

EPOCH = Time("2026-10-08T00:00:00", scale="utc")


def _astropy_geocentric(name):
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric

    return (get_body_barycentric(name, EPOCH) - get_body_barycentric("earth", EPOCH)).xyz.to_value(u.m)


@pytest.mark.parametrize("name", ["mars", "jupiter", "venus"])
def test_space_environment_planet_positions_match_astropy(name):
    # R2: astropy built-in ephemeris, geometric geocentric position. SSAPy's
    # DE430 positions agree to 1e-4 of the distance.
    position = SpaceEnvironment().body_position_model(name)(EPOCH.gps)
    expected = _astropy_geocentric(name)
    assert np.linalg.norm(position - expected) <= 1e-4 * np.linalg.norm(expected)


def test_sun_position_for_plots_uses_the_ephemeris():
    # R2: the geometric geocentric Sun from astropy, to 1e-6 of 1 au (150 km).
    # The Meeus fallback the function always fell back to is off by 6.5e-3 (970,000 km).
    expected = _astropy_geocentric("sun")
    position = np.asarray(get_sun_position(EPOCH.gps), dtype=float).reshape(3)
    assert np.linalg.norm(position - expected) <= 1e-6 * np.linalg.norm(expected)
