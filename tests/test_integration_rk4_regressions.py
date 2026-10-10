import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ssatk.propagators_orbit.rk4 import rk4
from ssatk.propagators_orbit.int_utils import precompute_third_body_positions


rk4_module = importlib.import_module("ssatk.propagators_orbit.rk4")


class RK4InputTests(unittest.TestCase):
    def test_cached_gravity_result_is_not_mutated_by_third_body_addition(self):
        gravity = np.zeros(3)
        with patch.object(rk4_module, 'accel_point_moon', return_value=np.array([1., 0, 0])), \
             patch.object(rk4_module, 'accel_point_sun', return_value=np.zeros(3)):
            r, v = rk4([7e6, 0, 0], [0, 7500, 0], [0, 1], accel_gravity=lambda r: gravity)
        np.testing.assert_array_equal(gravity, np.zeros(3))
        np.testing.assert_allclose(r[-1], [7e6 + 0.5, 7500, 0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(v[-1], [1, 7500, 0], rtol=0, atol=1e-9)


    def test_two_epoch_ephemeris_grid_uses_linear_interpolation(self):
        def position(t):
            return np.array([t, 2 * t, 3 * t])
        with patch('ssapy.get_body', return_value=SimpleNamespace(position=position), create=True):
            interpolant = precompute_third_body_positions([0., 2.], 'moon')
        np.testing.assert_allclose(interpolant([0.5, 1.5]), [[0.5, 1., 1.5], [1.5, 3., 4.5]])
