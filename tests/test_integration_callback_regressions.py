import unittest

import numpy as np

from ssatk.propagators_orbit.high_accuracy import propagate_orbit_state


class CallbackTests(unittest.TestCase):
    def test_velocity_callback_integrates_known_linear_drag_solution(self):
        def drag(r, v):
            return -0.01 * v
        tr = propagate_orbit_state(r0=[7e6, 0, 0], v0=[0, 7500, 0],
                                   t0=1.4e9, times=1.4e9 + np.arange(3),
                                   mu=0, acceleration=drag)
        np.testing.assert_allclose(tr.v[-1], [0, 7500 * np.exp(-0.02), 0], rtol=1e-10, atol=1e-9)
