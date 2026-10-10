"""Batch least-squares orbit determination from station observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares

from ..propagators_orbit import propagate_orbit_state_with_stm
from ..propagators_orbit.high_accuracy import OrbitPropagationWithSTM
from .measurements import GroundStation, GroundStationMeasurement
from .sensors import StationObservation

_RESERVED_PROPAGATION_KWARGS = frozenset({"times", "orbit0", "r0", "v0", "t0", "stm0"})
_DEFAULT_STATE_SCALE = np.array([1.0e7, 1.0e7, 1.0e7, 1.0e3, 1.0e3, 1.0e3])
_DEFAULT_STATE_SCALE.flags.writeable = False


@dataclass(frozen=True)
class BatchOrbitFitResult:
    """Result of a scaled batch orbit least-squares fit."""

    state0: np.ndarray
    t0: float
    trajectory: object
    residuals: np.ndarray
    weighted_residuals: np.ndarray
    cost: float
    nfev: int
    message: str
    numerical_rank: int
    covariance: np.ndarray | None
    success: bool

    def __post_init__(self):
        state0 = np.array(self.state0, dtype=float, copy=True)
        if state0.shape != (6,) or not np.all(np.isfinite(state0)):
            raise ValueError("state0 must be a finite six-element vector.")
        state0.flags.writeable = False
        object.__setattr__(self, "state0", state0)
        for name in ("residuals", "weighted_residuals"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite one-dimensional array.")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.covariance is not None:
            covariance = np.array(self.covariance, dtype=float, copy=True)
            if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must be a finite 6x6 matrix.")
            covariance.flags.writeable = False
            object.__setattr__(self, "covariance", covariance)
        trajectory = self.trajectory
        if not hasattr(trajectory, "t"):
            raise ValueError("trajectory must provide t, r, v, and stm arrays.")
        sample_count = np.asarray(trajectory.t).size
        trajectory_arrays = {}
        for name, shape in (("t", None), ("r", (None, 3)), ("v", (None, 3)), ("stm", (None, 6, 6))):
            value = np.array(getattr(trajectory, name), dtype=float, copy=True)
            if shape is None:
                valid = value.ndim == 1 and value.size == sample_count
            else:
                valid = (
                    value.ndim == len(shape)
                    and value.shape[0] == sample_count
                    and all(expected is None or actual == expected for actual, expected in zip(value.shape, shape))
                )
            if not valid or not np.all(np.isfinite(value)):
                raise ValueError(f"trajectory.{name} has an invalid shape or non-finite values.")
            value.flags.writeable = False
            trajectory_arrays[name] = value
        trajectory_copy = OrbitPropagationWithSTM(
            t=trajectory_arrays["t"],
            r=trajectory_arrays["r"],
            v=trajectory_arrays["v"],
            stm=trajectory_arrays["stm"],
            nfev=int(trajectory.nfev),
            message=str(trajectory.message),
        )
        object.__setattr__(self, "trajectory", trajectory_copy)
        object.__setattr__(self, "t0", float(self.t0))
        object.__setattr__(self, "cost", float(self.cost))
        if not np.isfinite(self.cost):
            raise ValueError("cost must be finite.")
        object.__setattr__(self, "nfev", int(self.nfev))
        object.__setattr__(self, "numerical_rank", int(self.numerical_rank))
        if not 0 <= self.numerical_rank <= 6:
            raise ValueError("numerical_rank must be between 0 and 6.")
        object.__setattr__(self, "success", bool(self.success))

@dataclass(frozen=True)
class _PreparedObservation:
    station: GroundStation
    observation: StationObservation
    value: np.ndarray
    covariance: np.ndarray
    whitener: np.ndarray
    epoch_index: int


def _scalar_time(value, name="time") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a scalar GPS time.")
    try:
        array = np.asarray(value.gps if hasattr(value, "gps") else value, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a scalar GPS time.") from None
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar GPS time.")
    return float(array)


def _validate_state(state, name):
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be a finite six-element GCRF SI state.")
    return state


def _whitener(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("observation covariance must be square and positive-definite.")
    if not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T):
        raise ValueError("observation covariance must be square and positive-definite.")
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("observation covariance must be positive-definite.") from exc


def _wrap_first(residual, measurement):
    if measurement in {"az_el", "ra_dec"}:
        residual = residual.copy()
        residual[0] = (residual[0] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def solve_batch_orbit(
    observations: Iterable[tuple[GroundStation, StationObservation]],
    state0,
    *,
    t0=None,
    propagation_kwargs: Mapping | None = None,
    state_scale=None,
    max_nfev: int | None = None,
) -> BatchOrbitFitResult:
    """Fit a six-state GCRF orbit to valid station observations.

    ``state0`` is the six-element initial state at ``t0``. The default state
    scales are ``[1e7, 1e7, 1e7, 1e3, 1e3, 1e3]`` in SI units. Observation covariance is Cholesky-whitened, and the
    returned covariance is in physical SI state units when full rank.
    """

    state0 = _validate_state(state0, "state0")
    propagation_kwargs = {} if propagation_kwargs is None else dict(propagation_kwargs)
    conflicts = _RESERVED_PROPAGATION_KWARGS & propagation_kwargs.keys()
    if conflicts:
        raise ValueError(f"propagation_kwargs cannot override {sorted(conflicts)}.")
    if max_nfev is not None and (isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev <= 0):
        raise ValueError("max_nfev must be a positive integer.")

    prepared = []
    for pair in observations:
        try:
            station, observation = pair
        except (TypeError, ValueError):
            raise ValueError("observations must contain (GroundStation, StationObservation) pairs.") from None
        if not isinstance(station, GroundStation):
            raise TypeError("observations must contain (GroundStation, StationObservation) pairs.")
        if observation is None:
            continue
        if not isinstance(observation, StationObservation):
            raise TypeError("observations must contain (GroundStation, StationObservation) pairs.")
        if not observation.valid or observation.value is None:
            continue
        value, covariance = observation.as_measurement()
        prepared.append((station, observation, value, covariance, _whitener(covariance)))
    if not prepared:
        raise ValueError("at least one valid station observation is required.")
    rows = sum(value.size for _, observation, value, _, _ in prepared)
    if rows < 6:
        raise ValueError("at least six scalar observation residual rows are required.")
    epochs_observed = np.asarray([observation.time for _, observation, _, _, _ in prepared], dtype=float)
    if t0 is None:
        t0 = float(np.min(epochs_observed))
    else:
        t0 = _scalar_time(t0, "t0")
    if t0 > float(np.min(epochs_observed)):
        raise ValueError("t0 must be no later than the earliest observation epoch.")
    epochs = np.unique(np.concatenate(([t0], epochs_observed)))
    if epochs.size < 2:
        raise ValueError("observations must span at least two distinct propagation epochs.")
    epoch_indices = {float(epoch): index for index, epoch in enumerate(epochs)}
    prepared = [
        _PreparedObservation(station, observation, value, covariance, whitener, epoch_indices[float(observation.time)])
        for station, observation, value, covariance, whitener in prepared
    ]

    if state_scale is None:
        state_scale = _DEFAULT_STATE_SCALE
    state_scale = np.asarray(state_scale, dtype=float)
    if state_scale.shape != (6,) or not np.all(np.isfinite(state_scale)) or np.any(state_scale <= 0.0):
        raise ValueError("state_scale must be a finite positive six-element vector.")
    scaled_initial = state0 / state_scale

    def propagate(candidate):
        return propagate_orbit_state_with_stm(
            times=epochs,
            r0=candidate[:3],
            v0=candidate[3:],
            t0=t0,
            **propagation_kwargs,
        )

    def evaluate(scaled_state, *, with_jacobian):
        candidate = np.asarray(scaled_state, dtype=float) * state_scale
        trajectory = propagate(candidate)
        residual_blocks = []
        jacobian_blocks = []
        for item in prepared:
            prediction = GroundStationMeasurement(
                item.station, item.observation.time, item.observation.measurement
            )(np.concatenate((trajectory.r[item.epoch_index], trajectory.v[item.epoch_index])))
            residual = _wrap_first(item.value - prediction[0], item.observation.measurement)
            residual_blocks.append(np.linalg.solve(item.whitener, residual))
            if with_jacobian:
                physical_h = prediction[1] @ trajectory.stm[item.epoch_index]
                jacobian_blocks.append(-np.linalg.solve(item.whitener, physical_h * state_scale))
        weighted = np.concatenate(residual_blocks)
        if not with_jacobian:
            return weighted
        return weighted, np.vstack(jacobian_blocks)

    fit = least_squares(
        lambda scaled: evaluate(scaled, with_jacobian=False),
        scaled_initial,
        jac=lambda scaled: evaluate(scaled, with_jacobian=True)[1],
        **({} if max_nfev is None else {"max_nfev": int(max_nfev)}),
    )
    final_state = fit.x * state_scale
    final_trajectory = propagate(final_state)
    physical_blocks = []
    weighted_blocks = []
    final_jacobian_blocks = []
    for item in prepared:
        state = np.concatenate((final_trajectory.r[item.epoch_index], final_trajectory.v[item.epoch_index]))
        prediction = GroundStationMeasurement(item.station, item.observation.time, item.observation.measurement)(state)
        physical = _wrap_first(item.value - prediction[0], item.observation.measurement)
        physical_blocks.append(physical)
        weighted_blocks.append(np.linalg.solve(item.whitener, physical))
        final_jacobian_blocks.append(
            np.linalg.solve(item.whitener, prediction[1] @ final_trajectory.stm[item.epoch_index])
        )
    physical_residuals = np.concatenate(physical_blocks)
    weighted_residuals = np.concatenate(weighted_blocks)
    final_jacobian_x = np.vstack(final_jacobian_blocks)
    _, singular_values, vh = np.linalg.svd(final_jacobian_x, full_matrices=False)
    tolerance = np.finfo(float).eps * max(final_jacobian_x.shape) * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    covariance = None
    if numerical_rank == 6:
        covariance = (vh.T / singular_values**2) @ vh
    message = str(fit.message)
    success = bool(fit.success and numerical_rank == 6)
    if numerical_rank < 6:
        message = f"{message}; rank deficient (numerical rank {numerical_rank}/6)"
    return BatchOrbitFitResult(
        final_state,
        t0,
        final_trajectory,
        physical_residuals,
        weighted_residuals,
        fit.cost,
        fit.nfev,
        message,
        numerical_rank,
        covariance,
        success,
    )


__all__ = ["BatchOrbitFitResult", "solve_batch_orbit"]
