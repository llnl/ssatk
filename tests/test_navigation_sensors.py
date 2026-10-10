"""Sensor noise, bias and dropout statistics against their distributions."""

import numpy as np
import pytest
from scipy import stats

from ssatk.navigation import CartesianSensor, GroundStation, GroundStationSensor


def test_cartesian_sensor_noise_bias_and_dropout_statistics():
    # R1: residuals minus bias are N(0, C), so their squared Mahalanobis norm is
    # chi^2(3) (KS p > 0.01), their mean is within 4 standard errors of zero,
    # and the dropout count is Binomial(n, p) (two-sided p > 0.01). 20,000
    # seeded draws.
    covariance = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, -2.0], [0.0, -2.0, 1.0]])
    bias = np.array([10.0, -5.0, 2.0])
    sensor = CartesianSensor(covariance, bias=bias, rng=np.random.default_rng(3), dropout_probability=0.2)
    truth = np.array([7000e3, 0.0, 0.0])
    samples = [sensor.measure(truth, 0.0) for _ in range(20000)]
    values = np.array([s.value for s in samples if s.valid])
    residual = values - truth - bias
    mahalanobis = np.einsum("ij,jk,ik->i", residual, np.linalg.inv(covariance), residual)
    assert stats.kstest(mahalanobis, stats.chi2(3).cdf).pvalue > 0.01
    standard_error = np.sqrt(np.diag(covariance) / len(values))
    assert np.all(np.abs(residual.mean(axis=0)) < 4 * standard_error)
    dropped = sum(not s.valid for s in samples)
    assert stats.binomtest(dropped, 20000, 0.2).pvalue > 0.01


def test_ground_station_range_noise_and_elevation_mask():
    # R1: a target straight overhead at 1000 km is visible with range 1000 km
    # (geodetic up at the equator); range residuals minus bias are N(0, sigma^2)
    # (KS p > 0.01); a target on the far side of Earth is masked and invalid.
    station = GroundStation(lon_deg=0.0, lat_deg=0.0, elevation_m=0.0)
    sensor = GroundStationSensor(station, [[25.0]], measurement="range", bias=3.0, rng=np.random.default_rng(5))
    t = 1.4e9
    from ssatk.coordinates.geodetic import llh_to_gcrf
    from astropy.time import Time

    r_site, _ = llh_to_gcrf(0.0, 0.0, Time(t, format="gps"), height=0.0)
    r_site = np.asarray(r_site, dtype=float).reshape(3)
    overhead = r_site + 1_000e3 * r_site / np.linalg.norm(r_site)
    state = np.r_[overhead, 0.0, 0.0, 0.0]
    observations = [sensor.measure(state, t) for _ in range(2000)]
    assert all(o.visible and o.valid for o in observations)
    assert observations[0].prediction.value == pytest.approx(1_000e3, abs=1.0)
    residual = np.array([o.value - o.prediction.value - 3.0 for o in observations])
    assert stats.kstest(residual / 5.0, "norm").pvalue > 0.01

    far_side = np.r_[-overhead, 0.0, 0.0, 0.0]
    hidden = sensor.measure(far_side, t)
    assert not hidden.visible and not hidden.valid and hidden.value is None
