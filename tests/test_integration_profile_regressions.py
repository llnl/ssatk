import unittest

import numpy as np

from ssatk.propagators_orbit.int_utils import build_profile


class ProfileTests(unittest.TestCase):


    def test_integer_times_have_same_meaning_on_different_grids(self):
        profile = {'start_time': 1800, 'end_time': 2400, 'thrust': 2}
        for times in (np.arange(3601), np.arange(0, 3601, 600)):
            expected = 2 * ((times >= 1800) & (times < 2400))
            np.testing.assert_array_equal(build_profile(profile, times), expected)
