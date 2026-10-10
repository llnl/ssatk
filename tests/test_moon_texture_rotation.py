import numpy as np
import pytest
from astropy.time import Time
from ssapy import get_body

from ssatk.plots.plotutils import _moon_texture_rotation_rad


@pytest.mark.parametrize("utc", ["2025-01-01T00:00:00", "2026-10-08T00:00:00", "2027-05-15T12:00:00"])
def test_moon_texture_prime_meridian_faces_earth(utc):
    # R1: the Moon is tidally locked, so its 0 deg selenographic meridian points
    # at Earth to within the optical libration in longitude (7.9 deg max). The
    # texture's prime-meridian azimuth must sit within 9 deg of the azimuth of
    # the Moon-to-Earth direction in the GCRF x-y plane. Earth's sidereal angle,
    # used before, bears no relation to it.
    t = Time(utc, scale="utc").gps
    moon = get_body("moon")
    to_earth = -np.asarray(moon.position(t), dtype=float).reshape(3)
    expected = np.arctan2(to_earth[1], to_earth[0])
    angle = float(_moon_texture_rotation_rad(t)[0])
    assert abs(np.degrees((angle - expected + np.pi) % (2 * np.pi) - np.pi)) < 9.0
