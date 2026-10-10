import numpy as np
import pytest
from astropy.time import Time
from ssapy import get_body

from ssatk.orbital_mechanics.lagrange_points import lagrange_points_lunar_fixed_frame

# R1: circular restricted three-body equilibria for the Earth-Moon mass ratio
# mu = 0.01215 lie at 0.8369, 1.1557 and -1.0051 d from the barycentre, i.e.
# 0.8491, 1.1678 and -0.9929 d from Earth, or -0.1509, +0.1678 and -1.9929 d
# from the Moon (d = Earth-Moon distance).


@pytest.mark.parametrize("utc", ["2025-01-01T00:00:00", "2026-10-08T00:00:00"])
def test_lagrange_points_in_the_moon_centred_rotating_frame(utc):
    # R1: in the Moon-centred frame (Earth at -d on X, +Y along the Moon's
    # motion) L4 and L5 are at (-d/2, +/-sqrt(3) d/2, 0), and L1-L3 lie on the X
    # axis at the CR3BP fractions above. 1e-6 of d for L4/L5, 1e-3 of d for
    # L1-L3 (the mass ratio enters at that level).
    t = Time([utc], scale="utc").gps
    moon = get_body("moon")
    d = np.linalg.norm(np.asarray(moon.position(t), dtype=float).reshape(3))
    points = lagrange_points_lunar_fixed_frame(t)
    np.testing.assert_allclose(points["L4"], [-d / 2, np.sqrt(3) * d / 2, 0.0], rtol=0, atol=1e-6 * d)
    np.testing.assert_allclose(points["L5"], [-d / 2, -np.sqrt(3) * d / 2, 0.0], rtol=0, atol=1e-6 * d)
    for name, fraction in {"L1": -0.1509, "L2": 0.1678, "L3": -1.9929}.items():
        x, y, z = points[name] / d
        assert x == pytest.approx(fraction, abs=1e-3), name
        assert abs(y) < 1e-6 and abs(z) < 1e-6, name
