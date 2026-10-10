"""Earth-Moon Lagrange points checked against the CR3BP equilibrium condition."""

import numpy as np
import pytest
import ssapy
from astropy.time import Time

from ssatk.constants import EARTH_MU, MOON_MU
from ssatk.orbital_mechanics import lagrange_points as lp

EPOCHS = Time(["2025-01-01T00:00:00", "2025-03-20T12:00:00", "2026-10-08T00:00:00"], scale="utc").gps


@pytest.mark.parametrize("function", ["lunar_lagrange_points", "lunar_lagrange_points_circular"])
@pytest.mark.parametrize("epoch", EPOCHS)
def test_lagrange_points_are_rotating_frame_equilibria(function, epoch):
    # R1: in the frame rotating with the Earth-Moon line at n^2 = G(M+m)/d^3,
    # gravity from both bodies plus the centrifugal term vanishes at L1-L5.
    moon = ssapy.get_body("moon")
    r_moon = np.asarray(moon.position(epoch), dtype=float).reshape(3)
    d = np.linalg.norm(r_moon)
    barycentre = MOON_MU / (EARTH_MU + MOON_MU) * r_moon
    n_squared = (EARTH_MU + MOON_MU) / d**3
    points = getattr(lp, function)(epoch)

    for name in ("L1", "L2", "L3", "L4", "L5"):
        r = np.asarray(points[name], dtype=float).reshape(3)
        gravity = -EARTH_MU * r / np.linalg.norm(r) ** 3 - MOON_MU * (r - r_moon) / np.linalg.norm(r - r_moon) ** 3
        centrifugal = n_squared * (r - barycentre)
        assert np.linalg.norm(gravity + centrifugal) < 1e-9 * np.linalg.norm(gravity), name


def test_collinear_points_sit_at_published_cr3bp_fractions():
    # R3: for the Earth-Moon mass ratio (mu ~ 0.01215) L1, L2, and L3 sit at about
    # 0.8491, 1.1678, and -0.9929 Earth-Moon distances from Earth.
    epoch = EPOCHS[1]
    moon = ssapy.get_body("moon")
    r_moon = np.asarray(moon.position(epoch), dtype=float).reshape(3)
    d = np.linalg.norm(r_moon)
    points = lp.lunar_lagrange_points(epoch)
    fractions = [np.dot(np.ravel(points[k]), r_moon) / d**2 for k in ("L1", "L2", "L3")]
    np.testing.assert_allclose(fractions, [0.8491, 1.1678, -0.9929], atol=2e-4)
