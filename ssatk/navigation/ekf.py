"""Small, dependency-light extended Kalman filter primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..constants import EARTH_MU
from ..propagators_orbit import propagate_orbit_state_with_stm

Array = np.ndarray


def wrap_angle_residual(residual, indices=(0,), *, period: float = 2.0 * np.pi) -> Array:
    """Wrap selected angular innovations into ``[-period/2, period/2)``."""

    residual = np.asarray(residual, dtype=float).copy()
    if residual.ndim != 1 or not np.all(np.isfinite(residual)):
        raise ValueError("residual must be a finite one-dimensional array.")
    period = float(period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive.")
    indices = tuple(int(index) for index in indices)
    if any(index < 0 or index >= residual.size for index in indices):
        raise ValueError("angle residual indices are out of bounds.")
    residual[list(indices)] = (residual[list(indices)] + period / 2.0) % period - period / 2.0
    return residual


def _matrix(value, shape, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
    return matrix


def _vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return vector


@dataclass(frozen=True)
class EKFState:
    """State estimate and covariance at one filter epoch."""

    x: Array
    covariance: Array
    time: float = 0.0

    def __post_init__(self):
        x = _vector(self.x, np.asarray(self.x).size, "x")
        covariance = _matrix(self.covariance, (x.size, x.size), "covariance")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("covariance must be symmetric.")
        if np.any(np.linalg.eigvalsh(covariance) < -1e-12):
            raise ValueError("covariance must be positive semidefinite.")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "time", float(self.time))


class ExtendedKalmanFilter:
    """Discrete extended Kalman filter with user-supplied dynamics Jacobians.

    The dynamics callback returns the predicted state. Its Jacobian callback
    returns the state-transition matrix for the same step. Measurement models
    return ``(predicted_measurement, measurement_jacobian)``.
    """

    def __init__(self, state: EKFState):
        self.state = state

    def predict(
        self,
        dynamics: Callable[[Array, float], Array],
        jacobian: Array,
        process_noise: Array,
        *,
        time: float | None = None,
    ) -> EKFState:
        n = self.state.x.size
        transition = _matrix(jacobian, (n, n), "jacobian")
        noise = _matrix(process_noise, (n, n), "process_noise")
        predicted = _vector(dynamics(self.state.x.copy(), self.state.time), n, "predicted state")
        covariance = transition @ self.state.covariance @ transition.T + noise
        covariance = 0.5 * (covariance + covariance.T)
        self.state = EKFState(predicted, covariance, self.state.time if time is None else time)
        return self.state

    def update(
        self,
        measurement,
        model: Callable[[Array], tuple[Array, Array]],
        noise: Array,
        *,
        time: float | None = None,
        angle_indices=None,
        residual=None,
    ) -> EKFState:
        prediction, jacobian = model(self.state.x.copy())
        prediction = np.asarray(prediction, dtype=float)
        measurement = np.asarray(measurement, dtype=float)
        if prediction.ndim != 1 or measurement.shape != prediction.shape:
            raise ValueError("measurement and model output must be one-dimensional and equal-sized.")
        m, n = prediction.size, self.state.x.size
        jacobian = _matrix(jacobian, (m, n), "measurement jacobian")
        noise = _matrix(noise, (m, m), "measurement noise")
        innovation = measurement - prediction if residual is None else np.asarray(
            residual(measurement, prediction), dtype=float
        )
        if innovation.shape != prediction.shape or not np.all(np.isfinite(innovation)):
            raise ValueError("measurement residual must be finite and match model output.")
        if angle_indices is None:
            angle_indices = getattr(model, "angle_indices", ())
        innovation = wrap_angle_residual(innovation, angle_indices) if angle_indices else innovation
        innovation_covariance = jacobian @ self.state.covariance @ jacobian.T + noise
        try:
            gain = np.linalg.solve(innovation_covariance, jacobian @ self.state.covariance).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement covariance is singular.") from exc
        updated = self.state.x + gain @ innovation
        identity = np.eye(n)
        residual = identity - gain @ jacobian
        covariance = residual @ self.state.covariance @ residual.T + gain @ noise @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        self.state = EKFState(updated, covariance, self.state.time if time is None else time)
        return self.state


class CartesianOrbitEKF(ExtendedKalmanFilter):
    """EKF adapter for SSATK Cartesian orbit propagation.

    The prediction uses SSATK's translational propagator and its integrated
    state-transition matrix, so arbitrary SSATK acceleration models can be
    used without duplicating filter dynamics.
    """

    def predict_orbit(
        self,
        *,
        time: float,
        mu: float = EARTH_MU,
        acceleration=None,
        process_noise=None,
        fd_step: float = 1.0e-6,
        method: str = "DOP853",
        rtol: float = 1.0e-10,
        atol: float = 1.0e-9,
        max_step: float = np.inf,
    ) -> EKFState:
        """Propagate a six-component Cartesian state to ``time``."""

        if self.state.x.size != 6:
            raise ValueError("CartesianOrbitEKF requires a six-component state.")
        time = float(time)
        if time <= self.state.time:
            raise ValueError("time must be later than the current filter epoch.")
        propagation = propagate_orbit_state_with_stm(
            r0=self.state.x[:3],
            v0=self.state.x[3:],
            t0=self.state.time,
            times=[self.state.time, time],
            mu=mu,
            acceleration=acceleration,
            fd_step=fd_step,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        predicted = np.concatenate((propagation.r[-1], propagation.v[-1]))
        noise = np.zeros((6, 6)) if process_noise is None else process_noise
        return self.predict(
            lambda _state, _time: predicted,
            propagation.stm[-1],
            noise,
            time=time,
        )

class CartesianMeasurement:
    """Direct Cartesian position/velocity measurement model."""

    def __init__(self, indices=(0, 1, 2, 3, 4, 5)):
        self.indices = tuple(int(index) for index in indices)

    def __call__(self, state: Array) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float)
        if state.ndim != 1 or any(index < 0 or index >= state.size for index in self.indices):
            raise ValueError("measurement indices must address a one-dimensional state.")
        jacobian = np.zeros((len(self.indices), state.size))
        jacobian[np.arange(len(self.indices)), self.indices] = 1.0
        return state[list(self.indices)], jacobian


__all__ = [
    "CartesianMeasurement",
    "CartesianOrbitEKF",
    "EKFState",
    "ExtendedKalmanFilter",
    "wrap_angle_residual",
]
