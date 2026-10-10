import numpy as np
import pytest

from ssatk.navigation import (
    BatchOrbitFitResult,
    GroundStation,
    StationObservation,
    StationPrediction,
    solve_batch_orbit,
)
from ssatk.propagators_orbit import propagate_orbit_state_with_stm


@pytest.fixture
def synthetic_station(monkeypatch):
    def predict(self, state, time, measurement):
        values = {
            "range": (state[0], np.array([[1.0, 0, 0, 0, 0, 0]])),
            "range_rate": (state[3], np.array([[0, 0, 0, 1.0, 0, 0]])),
            "az_el": (np.array([np.mod(state[1], 2.0 * np.pi), state[4]]), np.array([[0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0]])),
            "ra_dec": (np.array([np.mod(state[2], 2.0 * np.pi), state[5]]), np.array([[0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0]])),
        }
        value, jacobian = values[measurement]
        return StationPrediction(float(time), measurement, value, jacobian, True, 0.5)

    monkeypatch.setattr(GroundStation, "predict", predict)
    return GroundStation(0.0, 0.0)


def _observation(station, measurement, value, time, covariance=1e-8, bias=0.0, valid=True):
    dimension = 1 if measurement in {"range", "range_rate"} else 2
    cov = np.eye(dimension) * covariance
    value = np.asarray(value, dtype=float) if dimension == 2 else float(value)
    bias = np.zeros(2) if dimension == 2 and np.isscalar(bias) else (bias if dimension == 2 else float(bias))
    return station, StationObservation(time, measurement, value + bias, cov, bias, 0.5, True, valid)


def test_batch_fit_recovers_mu_zero_state_and_stm(synthetic_station):
    truth = np.array([10.0, 0.5, 0.3, 1.0, 2.0, 0.2])
    observations = [
        _observation(synthetic_station, "range", truth[0], 0.0),
        _observation(synthetic_station, "range_rate", truth[3], 0.0),
        _observation(synthetic_station, "az_el", [truth[1], truth[4]], 0.0),
        _observation(synthetic_station, "ra_dec", [truth[2] + truth[5], truth[5]], 1.0),
    ]
    result = solve_batch_orbit(
        observations,
        state0=np.zeros(6),
        propagation_kwargs={"mu": 0.0},
    )
    assert isinstance(result, BatchOrbitFitResult)
    np.testing.assert_allclose(result.state0, truth, atol=2e-4)
    assert result.success and result.numerical_rank == 6
    assert result.covariance.shape == (6, 6)
    np.testing.assert_allclose(result.trajectory.stm[1, :3, 3:], np.eye(3), atol=1e-10)
    finite_stm = np.empty((6, 6))
    for column in range(6):
        step = 1e-5
        plus = truth.copy(); plus[column] += step
        minus = truth.copy(); minus[column] -= step
        plus_traj = propagate_orbit_state_with_stm(times=[0.0, 1.0], r0=plus[:3], v0=plus[3:], t0=0.0, mu=0.0)
        minus_traj = propagate_orbit_state_with_stm(times=[0.0, 1.0], r0=minus[:3], v0=minus[3:], t0=0.0, mu=0.0)
        finite_stm[:, column] = np.r_[plus_traj.r[-1], plus_traj.v[-1]] - np.r_[minus_traj.r[-1], minus_traj.v[-1]]
    finite_stm /= 2.0e-5
    np.testing.assert_allclose(result.trajectory.stm[1], finite_stm, atol=1e-8)
    observation = observations[-1][1]
    nominal_state = np.r_[result.trajectory.r[1], result.trajectory.v[1]]
    nominal = synthetic_station.predict(nominal_state, 1.0, observation.measurement)
    finite_measurement = []
    for column in range(6):
        plus = truth.copy(); plus[column] += 1e-5
        minus = truth.copy(); minus[column] -= 1e-5
        plus_traj = propagate_orbit_state_with_stm(times=[0.0, 1.0], r0=plus[:3], v0=plus[3:], t0=0.0, mu=0.0)
        minus_traj = propagate_orbit_state_with_stm(times=[0.0, 1.0], r0=minus[:3], v0=minus[3:], t0=0.0, mu=0.0)
        plus_value = synthetic_station.predict(np.r_[plus_traj.r[-1], plus_traj.v[-1]], 1.0, observation.measurement).value
        minus_value = synthetic_station.predict(np.r_[minus_traj.r[-1], minus_traj.v[-1]], 1.0, observation.measurement).value
        delta = np.asarray(plus_value) - np.asarray(minus_value)
        delta[0] = (delta[0] + np.pi) % (2.0 * np.pi) - np.pi
        finite_measurement.append(delta / 2.0e-5)
    expected_measurement_jacobian = nominal.jacobian @ result.trajectory.stm[1]
    np.testing.assert_allclose(expected_measurement_jacobian, np.asarray(finite_measurement).T, atol=1e-8)
    fisher_jacobian = []
    for station, observation in observations:
        epoch = 0 if observation.time == 0.0 else 1
        state = np.r_[result.trajectory.r[epoch], result.trajectory.v[epoch]]
        prediction = station.predict(state, observation.time, observation.measurement)
        covariance = observation.covariance
        fisher_jacobian.append(np.linalg.solve(np.linalg.cholesky(covariance), prediction.jacobian @ result.trajectory.stm[epoch]))
    _, singular_values, vh = np.linalg.svd(np.vstack(fisher_jacobian), full_matrices=False)
    covariance_oracle = (vh.T / singular_values**2) @ vh
    np.testing.assert_allclose(result.covariance, covariance_oracle, rtol=1e-8, atol=1e-12)
    assert np.linalg.norm(result.weighted_residuals) < 1e-3


def test_batch_fit_handles_bias_and_angular_seam(synthetic_station):
    truth = np.array([10.0, 0.4, 2.0 * np.pi - 0.2, 1.0, 2.0, 0.05])
    observations = [
        _observation(synthetic_station, "range", truth[0], 0.0, bias=2.0),
        _observation(synthetic_station, "range_rate", truth[3], 0.0),
        _observation(synthetic_station, "az_el", [truth[1], truth[4]], 0.0),
        _observation(synthetic_station, "ra_dec", [np.mod(truth[2] + truth[5], 2.0 * np.pi), truth[5]], 1.0),
    ]
    result = solve_batch_orbit(observations, state0=np.zeros(6), propagation_kwargs={"mu": 0.0})
    assert result.success
    assert result.state0[0] == pytest.approx(truth[0], abs=1e-3)
    assert ((result.state0[2] - truth[2] + np.pi) % (2.0 * np.pi) - np.pi) == pytest.approx(0.0, abs=1e-3)
