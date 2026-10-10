import numpy as np
import pytest
from astropy.time import Time
from ssapy import get_body

from ssatk.orbit_initializer import OrbitInitialize
from ssatk.orbital_mechanics.lagrange_points import lunar_lagrange_points


@pytest.mark.parametrize("utc", ["2025-03-20T12:00:00", "2026-10-08T00:00:00"])
def test_lunar_l4_is_the_leading_equilateral_point(utc):
    # R1: L4 is equidistant from Earth and the Moon at the Earth-Moon distance d
    # and leads the Moon by 60 deg in its orbit plane (1e-9 relative), and
    # matches the toolkit's CR3BP L4 (tested against zero rotating-frame
    # acceleration) to 1e-6 d.
    t = Time(utc, scale="utc")
    orbit = OrbitInitialize.Lunar_L4(t)
    moon = get_body("moon")  # hold the Body: SSAPy closes the ephemeris of a collected temporary
    r_moon = np.asarray(moon.position(t.gps), dtype=float).reshape(3)
    d = np.linalg.norm(r_moon)
    assert np.linalg.norm(orbit.r) == pytest.approx(d, rel=1e-9)
    assert np.linalg.norm(orbit.r - r_moon) == pytest.approx(d, rel=1e-9)
    v_moon = (np.asarray(moon.position(t.gps + 1.0)).reshape(3) - np.asarray(moon.position(t.gps - 1.0)).reshape(3)) / 2.0
    assert np.dot(np.cross(r_moon, orbit.r), np.cross(r_moon, v_moon)) > 0.0  # leading, not trailing
    l4 = np.asarray(lunar_lagrange_points(t)["L4"], dtype=float).reshape(3)
    np.testing.assert_allclose(orbit.r, l4, rtol=0, atol=1e-6 * d)
