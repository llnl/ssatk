import numpy as np

from ssatk.accelerations_orbit.accel_uniform_earth import accel_uniform_earth
from ssatk.constants import EARTH_MU, EARTH_RADIUS


def test_uniform_earth_gravity_inside_and_outside():
    outside = np.array([2.0 * EARTH_RADIUS, 0.0, 0.0])
    inside = np.array([0.5 * EARTH_RADIUS, 0.0, 0.0])

    np.testing.assert_allclose(accel_uniform_earth(outside), [-EARTH_MU / outside[0] ** 2, 0.0, 0.0])
    np.testing.assert_allclose(accel_uniform_earth(inside), [-EARTH_MU * inside[0] / EARTH_RADIUS**3, 0.0, 0.0])
