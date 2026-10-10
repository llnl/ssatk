import sys

import numpy as np
import pytest
from numpy.polynomial import legendre
from ssapy.gravity import HarmonicCoefficients
from ssapy.utils import find_file

from ssatk.constants import EARTH_MU, EARTH_RADIUS

import ssatk.accelerations_orbit  # noqa: F401  (registers the submodule)

zonal = sys.modules["ssatk.accelerations_orbit.accel_earth_harmonics"]

POSITIONS_M = [
    np.array([4200e3, -3100e3, 4800e3]),   # mid-latitude LEO, |r| = 7092 km
    np.array([-6900e3, 1200e3, -2500e3]),  # southern hemisphere LEO
    np.array([0.0, 0.0, 7500e3]),          # over the pole, u = 1
    np.array([26560e3, 0.0, 0.0]),         # GPS radius on the equator, u = 0
]


def _zonal_potential(r, n):
    r_mag = np.linalg.norm(r)
    p_n = legendre.legval(r[2] / r_mag, [0.0] * n + [1.0])
    return -EARTH_MU * zonal.EGM96_ZONAL_J[n] * EARTH_RADIUS**n * p_n / r_mag ** (n + 1)


def _finite_difference_gradient(r, n, step_m=1.0):
    gradient = np.zeros(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step_m
        gradient[axis] = (_zonal_potential(r + offset, n) - _zonal_potential(r - offset, n)) / (2.0 * step_m)
    return gradient


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("r", POSITIONS_M, ids=["leo-north", "leo-south", "pole", "gps-equator"])
def test_zonal_accelerations_match_finite_difference_gradient_of_potential(r, n):
    # R5: a_n = grad U_n. A 1 m central difference of U_n is good to ~1e-14 m/s^2
    # at these radii; the bound is 1e-6 of the term's own size plus 1e-15 m/s^2.
    expected = _finite_difference_gradient(r, n)
    actual = getattr(zonal, f"accel_J{n}")(r)
    tolerance = 1e-6 * np.linalg.norm(expected) + 1e-15
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def test_zonal_coefficients_match_ssapy_egm96_file():
    # R3: unnormalized EGM96 zonals, J_n = -C_n0, from the egm96.egm file SSAPy ships.
    coefficients = HarmonicCoefficients.fromEGM(find_file("egm96", ext=".egm"))
    assert coefficients.radius == EARTH_RADIUS
    assert coefficients.MG == EARTH_MU
    for n in range(2, 9):
        assert zonal.EGM96_ZONAL_J[n] == pytest.approx(-coefficients.CS[n, 0], rel=1e-12, abs=0), n


def test_j2_matches_closed_form_on_the_equator_and_over_the_pole():
    # R1: on the equator a_J2 = -(3/2) J2 mu R^2 / r^4 r_hat; over the pole
    # a_J2 = +3 J2 mu R^2 / r^4 z_hat. Tolerance 1e-12 relative.
    j2 = zonal.EGM96_ZONAL_J[2]
    r = 7000e3
    scale = j2 * EARTH_MU * EARTH_RADIUS**2 / r**4
    np.testing.assert_allclose(zonal.accel_J2([r, 0.0, 0.0]), [-1.5 * scale, 0.0, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(zonal.accel_J2([0.0, 0.0, r]), [0.0, 0.0, 3.0 * scale], rtol=1e-12, atol=1e-20)
