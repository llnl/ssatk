
import numpy as np

from ssatk import asteroids


def test_asteroid_size_magnitude_and_filter_conversions():
    H = np.array([17.0, 20.0])
    radius = asteroids.radius_from_H_albedo(H, albedo=0.1)
    np.testing.assert_allclose(asteroids.H_mag(radius, albedo=0.1), H)

    mags = np.full(12, 20.0)
    filters = np.array(list("uuggrriizzyy"))
    types = np.array([0, 1] * 6)
    expected_corrections = np.array([-1.614, -1.927, -0.302, -0.395, 0.172, 0.255, 0.291, 0.455, 0.298, 0.401, 0.303, 0.406])
    np.testing.assert_allclose(asteroids.johnsonV_to_lsst_array(mags, filters, types), mags - expected_corrections)

    ztf_filters = np.array([1, 1, 2, 2, 3, 3])
    ztf_types = np.array([0, 1, 0, 1, 0, 1])
    ztf_expected = np.array([-0.302, -0.395, 0.172, 0.255, 0.291, 0.455])
    np.testing.assert_allclose(asteroids.johnsonV_to_ztf_array(np.full(6, 19.0), ztf_filters, ztf_types), 19.0 - ztf_expected)

    assert np.isclose(asteroids.granvik_low_slope(10.0), 0.3034 * 10.0 - 3.491)
    assert np.isclose(asteroids.granvik_high_slope(20.0), 0.7235 * 20.0 - 13.12)
