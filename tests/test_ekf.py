import numpy as np
import pytest

from ssatk.navigation import (
    CartesianMeasurement,
    CartesianOrbitEKF,
    EKFState,
    ExtendedKalmanFilter,
    GroundStation,
    GroundStationMeasurement,
    GroundStationSensor,
    StationPrediction,
    wrap_angle_residual,
)


def test_ekf_predict_update_reduces_position_error():
    state = EKFState(np.array([0.0, 1.0]), np.eye(2), time=0.0)
    ekf = ExtendedKalmanFilter(state)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    ekf.predict(lambda x, _t: transition @ x, transition, np.eye(2) * 1e-6, time=1.0)
    updated = ekf.update([1.2], lambda x: (x[[0]], np.array([[1.0, 0.0]])), np.array([[0.01]]))
    # Independent conditional-Gaussian equations, including the off-diagonal
    # covariance. Merely reducing error would accept an incorrectly scaled gain.
    prior = np.array([[2.000001, 1], [1, 1.000001]])
    gain = prior[:, 0] / (prior[0, 0] + .01)
    np.testing.assert_allclose(updated.x, [1, 1] + .2 * gain, atol=1e-12, rtol=0)
    expected_covariance = prior - np.outer(prior[:, 0], prior[0, :]) / (prior[0, 0] + .01)
    np.testing.assert_allclose(updated.covariance, expected_covariance, atol=1e-12, rtol=0)


def test_wrap_angle_residual_handles_circular_innovation():
    np.testing.assert_allclose(
        wrap_angle_residual([2.0 * np.pi - 1.0e-3], (0,)), [-1.0e-3], atol=1.0e-12, rtol=0
    )


def test_station_observation_removes_known_bias_before_ekf_update(monkeypatch):
    station = GroundStation(0.0, 0.0)

    def predict(self, state, time, measurement):
        return StationPrediction(
            0.0, measurement, state[0], np.array([[1.0, 0, 0, 0, 0, 0]]), True, 0.5
        )

    monkeypatch.setattr(GroundStation, "predict", predict)
    sensor = GroundStationSensor(station, 1.0e-12, bias=2.0, rng=0)
    observation = sensor.measure(np.array([1.0, 0, 0, 0, 0, 0]), 0.0)
    measurement, covariance = observation.as_measurement()
    assert measurement[0] == pytest.approx(1.0, abs=1.0e-5)
    ekf = ExtendedKalmanFilter(EKFState(np.zeros(6), np.eye(6)))
    updated = ekf.update(measurement, GroundStationMeasurement(station, 0.0), covariance)
    assert updated.x[0] == pytest.approx(1.0, abs=1.0e-5)


def test_cartesian_orbit_ekf_predicts_and_updates():
    mu = 3.986004418e14
    state = EKFState(
        [7.0e6, 0.0, 0.0, 0.0, np.sqrt(mu / 7.0e6), 0.0],
        np.eye(6),
    )
    ekf = CartesianOrbitEKF(state)
    predicted = ekf.predict_orbit(time=60.0, mu=mu, max_step=10.0)
    assert predicted.time == pytest.approx(60.0)
    assert predicted.covariance.shape == (6, 6)
    phase = np.sqrt(mu / 7e6**3) * 60
    expected_r = 7e6 * np.array([np.cos(phase), np.sin(phase), 0])
    expected_v = np.sqrt(mu / 7e6) * np.array([-np.sin(phase), np.cos(phase), 0])
    np.testing.assert_allclose(predicted.x[:3], expected_r, atol=.01, rtol=0)
    np.testing.assert_allclose(predicted.x[3:], expected_v, atol=1e-5, rtol=0)
    measurement = predicted.x[:3] + [10.0, 0.0, 0.0]
    prior_x, prior_p = predicted.x.copy(), predicted.covariance.copy()
    innovation_covariance = prior_p[:3, :3] + np.eye(3) * 1e-4
    gain = np.linalg.solve(innovation_covariance, prior_p[:3, :]).T
    updated = ekf.update(
        measurement,
        CartesianMeasurement((0, 1, 2)),
        np.eye(3) * 1.0e-4,
        time=60.0,
    )
    np.testing.assert_allclose(updated.x, prior_x + gain @ (measurement - prior_x[:3]), atol=1e-9, rtol=0)
    np.testing.assert_allclose(updated.covariance, prior_p - gain @ prior_p[:3, :], atol=1e-10, rtol=0)
