import numpy as np

from ssatk.constants import AU, EARTH_RADIUS
from ssatk.plots.coverage_analysis import compute_eclipse


def test_compute_eclipse_uses_finite_sun_disk_for_partial_shadow():
    r = np.array([
        [-2.0 * EARTH_RADIUS, EARTH_RADIUS, 0.0],
        [2.0 * EARTH_RADIUS, 0.0, 0.0],
    ])
    r_sun = np.array([
        [AU, 0.0, 0.0],
        [AU, 0.0, 0.0],
    ])

    np.testing.assert_array_equal(compute_eclipse(r, r_sun), [True, False])
