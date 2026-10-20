import unittest
from unittest.mock import patch

import numpy as np

from ssatk.constants import EARTH_RADIUS
from ssatk.propagators_orbit.leap_frog import leapfrog


class LeapfrogImpactTests(unittest.TestCase):
    def test_grazing_enter_exit_is_detected_with_physical_speed(self):
        radius = EARTH_RADIUS + 100e3
        r0 = np.array([-20000., radius - 1., 0.])
        v0 = np.array([7500., 0., 0.])
        times = 1.4e9 + np.array([0., 6., 12.])
        with patch('ssatk.propagators_orbit.leap_frog.accel_point_earth',
                   return_value=np.zeros(3)):
            r, v = leapfrog(r0, v0, times)
            self.assertEqual(len(r), 1)
            np.testing.assert_array_equal(r[0], r0)
            r, v, actual = leapfrog(r0, v0, times, return_times=True)
        hit = (20000. - np.sqrt(2 * radius - 1)) / 7500.
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(actual[-1], times[0] + hit, places=6)
        self.assertAlmostEqual(np.linalg.norm(r[-1]), radius, places=6)
        np.testing.assert_array_equal(v[-1], v0)
        np.testing.assert_array_equal(times, 1.4e9 + np.array([0., 6., 12.]))

    def test_impact_velocity_is_synchronized_for_constant_acceleration(self):
        radius = EARTH_RADIUS + 100e3
        with patch('ssatk.propagators_orbit.leap_frog.accel_point_earth',
                   return_value=np.array([-10., 0., 0.])):
            r, v, times = leapfrog([radius + 100, 0, 0], [-10, 0, 0],
                                   [0., 10.], return_times=True)
        hit = (-10 + np.sqrt(2100)) / 10
        self.assertAlmostEqual(times[-1], hit, places=10)
        self.assertAlmostEqual(r[-1, 0], radius, places=6)
        self.assertAlmostEqual(v[-1, 0], -10 - 10 * hit, places=8)

    def test_uniform_grid_roundoff_does_not_accumulate_epoch_drift(self):
        times = 1.4e9 + np.arange(101) * 0.1
        with patch('ssatk.propagators_orbit.leap_frog.accel_point_earth',
                   return_value=np.zeros(3)):
            r, v, actual = leapfrog([7e6, 0, 0], [0, 7500, 0], times,
                                    return_times=True)
        np.testing.assert_array_equal(actual, times)
        np.testing.assert_allclose(r[:, 1], 7500 * (times - times[0]), atol=1e-8, rtol=0)
