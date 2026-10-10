"""Orbit-acceleration helpers against closed forms and SSAPy's third-body model."""

import numpy as np
import pytest
from astropy.time import Time

from ssatk.accelerations_orbit import (
    accel_equatorial,
    accel_inclination,
    accel_plane,
    accel_point_moon,
    accel_point_sun,
    accel_radial,
    accel_to_circular,
    accel_uniform_earth,
    accel_velocity,
    reset_orbit_status,
)
from ssatk.constants import EARTH_MU, EARTH_RADIUS

EPOCH_GPS_S = Time("2026-10-08T00:00:00", scale="utc").gps
POSITIONS_M = [
    np.array([7000e3, 1000e3, -500e3]),        # LEO
    np.array([0.0, 42164e3, 0.0]),             # GEO
    np.array([-300000e3, 100000e3, 20000e3]),  # cislunar
]


@pytest.mark.parametrize("r", POSITIONS_M, ids=["leo", "geo", "cislunar"])
def test_sun_and_moon_point_masses_match_ssapy_third_body(r):
    # R2: SSAPy AccelThirdBody with its DE ephemeris. The toolkit uses astropy's
    # built-in apparent positions, which differ by light time and aberration;
    # the perturbations agree to 1e-3 relative (measured 1.5e-4 to 6.6e-4).
    from ssapy.body import get_body
    from ssapy.gravity import AccelThirdBody

    for name, ours in (("Sun", accel_point_sun), ("moon", accel_point_moon)):
        expected = AccelThirdBody(get_body(name))(r, np.zeros(3), EPOCH_GPS_S)
        actual = ours(r, EPOCH_GPS_S)
        assert np.linalg.norm(actual - expected) <= 1e-3 * np.linalg.norm(expected), name


def test_uniform_earth_gravity_is_linear_inside_and_inverse_square_outside():
    # R1: a uniform sphere gives g = -mu r / R^3 inside and -mu r / r^3
    # outside; both equal mu / R^2 at the surface. 1e-12 relative.
    for radius in (0.25 * EARTH_RADIUS, 0.9 * EARTH_RADIUS):
        r = np.array([0.0, radius, 0.0])
        np.testing.assert_allclose(accel_uniform_earth(r), -EARTH_MU * r / EARTH_RADIUS**3, rtol=1e-12)
    r = np.array([0.0, 0.0, 2.0 * EARTH_RADIUS])
    np.testing.assert_allclose(accel_uniform_earth(r), [0.0, 0.0, -EARTH_MU / (2.0 * EARTH_RADIUS) ** 2], rtol=1e-12)


def test_steering_helpers_point_along_their_named_directions():
    # R1: at 45 deg latitude on the 0 deg meridian, east is (0, 1, 0), north is
    # (-1, 0, 1)/sqrt 2, radial is (1, 0, 1)/sqrt 2; each helper returns its
    # magnitude along that direction to 1e-12.
    r = 7000e3 * np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    v = np.array([-3000.0, 6500.0, 3000.0])
    magnitude = 1.5e-3
    np.testing.assert_allclose(accel_equatorial(r, magnitude=magnitude), [0.0, magnitude, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        accel_inclination(r, magnitude=-magnitude), -magnitude * np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0), atol=1e-15
    )
    np.testing.assert_allclose(accel_radial(r, magnitude), magnitude * np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), atol=1e-15)
    np.testing.assert_allclose(accel_velocity(v, magnitude), magnitude * v / np.linalg.norm(v), atol=1e-15)
    r_hat = r / np.linalg.norm(r)
    horizontal = v - np.dot(v, r_hat) * r_hat
    np.testing.assert_allclose(accel_plane(r, v, magnitude), magnitude * horizontal / np.linalg.norm(horizontal), atol=1e-15)


def test_circularization_command_points_at_the_circular_velocity():
    # R1: at radius r the target is the horizontal circular speed sqrt(mu/r);
    # the command is the thrust magnitude along v_circ - v, and it stops once
    # |v_circ - v| is within tol (1e-12 relative).
    reset_orbit_status()
    radius = 8000e3
    r = np.array([radius, 0.0, 0.0])
    v = np.array([150.0, 0.85 * np.sqrt(EARTH_MU / radius), 0.0])
    dv = np.array([0.0, np.sqrt(EARTH_MU / radius), 0.0]) - v
    np.testing.assert_allclose(accel_to_circular(r, v, thrust=0.2), 0.2 * dv / np.linalg.norm(dv), rtol=1e-12)
    v_close = np.array([1.0, np.sqrt(EARTH_MU / radius), 0.0])
    np.testing.assert_allclose(accel_to_circular(r, v_close, thrust=0.2, tol=10.0), 0.0, atol=0)
    reset_orbit_status()
