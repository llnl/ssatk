"""Batch least-squares orbit determination from station observations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import numpy as np
from scipy.optimize import least_squares

from ..propagators_orbit import propagate_orbit_state, propagate_orbit_state_with_stm
from ..propagators_orbit.high_accuracy import OrbitPropagationWithSTM
from .measurements import GroundStation, GroundStationMeasurement, _RESERVED_PROPAGATION_KWARGS, _scalar_time
from .sensors import StationObservation

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
    parameters: Mapping[str, float] = field(default_factory=dict)
    joint_covariance: np.ndarray | None = None
    active_mask: np.ndarray | None = None

    def __post_init__(self):
        state0 = np.array(self.state0, dtype=float, copy=True)
        if state0.shape != (6,) or not np.all(np.isfinite(state0)):
            raise ValueError("state0 must be a finite six-element vector.")
        state0.flags.writeable = False
        object.__setattr__(self, "state0", state0)
        parameters = dict(self.parameters)
        if any(not isinstance(name, str) or not name or name in self.estimate_names[:6]
               or np.ndim(value) != 0 or not np.isfinite(value) for name, value in parameters.items()):
            raise ValueError("parameters must have unique non-state names and finite scalar values.")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        size = 6 + len(parameters)
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
        joint_covariance = self.joint_covariance
        if joint_covariance is None and not parameters:
            joint_covariance = self.covariance
        if joint_covariance is not None:
            joint_covariance = np.array(joint_covariance, dtype=float, copy=True)
            if (joint_covariance.shape != (size, size) or not np.all(np.isfinite(joint_covariance))
                    or np.any(np.diag(joint_covariance) <= 0)):
                raise ValueError("joint_covariance must be finite with positive diagonal and match all estimates.")
            joint_covariance.flags.writeable = False
        object.__setattr__(self, "joint_covariance", joint_covariance)
        active_mask = np.zeros(size, dtype=int) if self.active_mask is None else np.array(self.active_mask, copy=True)
        if active_mask.shape != (size,) or not np.all(np.isin(active_mask, (-1, 0, 1))):
            raise ValueError("active_mask must contain -1, 0 or 1 for each estimate.")
        active_mask.flags.writeable = False
        object.__setattr__(self, "active_mask", active_mask)
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
            solution=getattr(trajectory, "solution", None),
            time_origin=getattr(trajectory, "time_origin", 0.0),
        )
        object.__setattr__(self, "trajectory", trajectory_copy)
        object.__setattr__(self, "t0", float(self.t0))
        object.__setattr__(self, "cost", float(self.cost))
        if not np.isfinite(self.cost):
            raise ValueError("cost must be finite.")
        object.__setattr__(self, "nfev", int(self.nfev))
        object.__setattr__(self, "numerical_rank", int(self.numerical_rank))
        if not 0 <= self.numerical_rank <= size:
            raise ValueError("numerical_rank must be between zero and the number of estimates.")
        object.__setattr__(self, "success", bool(self.success))

    @property
    def estimate_names(self) -> tuple[str, ...]:
        """Covariance/correlation order: Cartesian state, then named parameters."""
        return ("x", "y", "z", "vx", "vy", "vz", *self.parameters)

    @property
    def correlation(self) -> np.ndarray | None:
        """Dimensionless joint correlations, or None when the fit is rank deficient."""
        if self.joint_covariance is None:
            return None
        sigma = np.sqrt(np.diag(self.joint_covariance))
        correlation = self.joint_covariance / np.outer(sigma, sigma)
        correlation.flags.writeable = False
        return correlation

@dataclass(frozen=True)
class _PreparedObservation:
    station: GroundStation
    observation: StationObservation
    value: np.ndarray
    covariance: np.ndarray
    whitener: np.ndarray
    epoch_index: int


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
    propagation_kwargs: Mapping | Callable[[Mapping[str, float]], Mapping] | None = None,
    state_scale=None,
    max_nfev: int | None = None,
    parameters0: Mapping[str, float] | None = None,
    parameter_scale=None,
    parameter_bounds=None,
    parameter_step=None,
    measurement_bias: Callable[[GroundStation, StationObservation, Mapping[str, float]], object] | None = None,
) -> BatchOrbitFitResult:
    """Fit a GCRF orbit and optional named parameters to station observations.

    ``state0`` is the six-element initial state at ``t0``. The default state
    scales are ``[1e7, 1e7, 1e7, 1e3, 1e3, 1e3]`` in SI units. Observation covariance is Cholesky-whitened, and the
    returned covariance is in physical SI state units when full rank.
    Radio observations require an explicit ``t0`` early enough to cover the
    spacecraft signal events and the entire Doppler count interval.

    ``parameters0`` supplies finite scalar guesses in insertion order.
    ``parameter_scale`` supplies positive characteristic magnitudes in that
    order (default one); ``parameter_bounds`` is a pair of lower/upper arrays
    in physical parameter units. ``parameter_step`` optionally supplies positive
    absolute differentiation steps in those units; default steps are cube-root
    machine epsilon times the larger of parameter magnitude and its scale.
    ``propagation_kwargs`` may be a callable of
    the parameter mapping returning existing propagation options, e.g. force
    models with fitted drag/SRP coefficients or finite-burn parameters.
    ``measurement_bias(station, observation, parameters)`` returns an extra
    additive bias in the observable's units, after its known bias correction.
    Parameter derivatives repropagate and resolve signal times using bounded
    finite differences; state derivatives retain the existing STM path.
    Joint covariance is the local inverse Fisher information in physical
    units, not residual-variance scaled. Active bounds are reported separately;
    this Gaussian approximation does not describe truncation at those bounds.
    """

    state0 = _validate_state(state0, "state0")
    parameters0 = {} if parameters0 is None else dict(parameters0)
    names = tuple(parameters0)
    if any(not isinstance(name, str) or not name or name in ("x", "y", "z", "vx", "vy", "vz") for name in names):
        raise ValueError("parameter names must be nonempty strings distinct from the Cartesian state names.")
    initial_parameters = np.asarray(list(parameters0.values()), dtype=float)
    if initial_parameters.shape != (len(names),) or not np.all(np.isfinite(initial_parameters)):
        raise ValueError("parameters0 must contain finite scalar values.")
    parameter_scale = np.ones(len(names)) if parameter_scale is None else np.asarray(parameter_scale, dtype=float)
    if parameter_scale.shape != (len(names),) or not np.all(np.isfinite(parameter_scale)) or np.any(parameter_scale <= 0):
        raise ValueError("parameter_scale must contain one finite positive scale per named parameter.")
    limits = np.tile([[-np.inf], [np.inf]], (1, len(names))) if parameter_bounds is None else np.asarray(parameter_bounds, dtype=float)
    if limits.shape != (2, len(names)) or np.any(np.isnan(limits)) or np.any(limits[0] >= limits[1]):
        raise ValueError("parameter_bounds must contain ordered lower and upper arrays for all named parameters.")
    if np.any(initial_parameters < limits[0]) or np.any(initial_parameters > limits[1]):
        raise ValueError("parameters0 must lie within parameter_bounds.")
    if parameter_step is not None:
        parameter_step = np.asarray(parameter_step, dtype=float)
        if parameter_step.shape != (len(names),) or not np.all(np.isfinite(parameter_step)) or np.any(parameter_step <= 0):
            raise ValueError("parameter_step must contain one finite positive physical step per named parameter.")
    if measurement_bias is not None and not callable(measurement_bias):
        raise TypeError("measurement_bias must be callable.")
    if not callable(propagation_kwargs):
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
    size = 6 + len(names)
    if rows < size:
        raise ValueError("at least one scalar observation residual row per estimated component is required.")
    epochs_observed = np.asarray([observation.time for _, observation, _, _, _ in prepared], dtype=float)
    radio_tracking = any(observation.radio is not None for _, observation, _, _, _ in prepared)
    if radio_tracking and t0 is None:
        raise ValueError("radio observations require an explicit t0 preceding their complete signal/count intervals.")
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
    scale = np.r_[state_scale, parameter_scale]
    scaled_initial = np.r_[state0, initial_parameters] / scale
    lower = np.r_[np.full(6, -np.inf), limits[0]] / scale
    upper = np.r_[np.full(6, np.inf), limits[1]] / scale

    def propagate(candidate, parameters, with_stm):
        kwargs = propagation_kwargs(parameters) if callable(propagation_kwargs) else propagation_kwargs
        if not isinstance(kwargs, Mapping):
            raise TypeError("propagation_kwargs callback must return a mapping of propagation options.")
        conflicts = _RESERVED_PROPAGATION_KWARGS & kwargs.keys()
        if conflicts:
            raise ValueError(f"propagation_kwargs cannot override {sorted(conflicts)}.")
        kwargs = dict(kwargs)
        if not with_stm:
            kwargs.pop("fd_step", None)
            kwargs.setdefault("atol", 1e-9)
            for tolerance in ("rtol", "atol"):
                if np.shape(kwargs.get(tolerance)) == (42,):
                    kwargs[tolerance] = np.asarray(kwargs[tolerance])[:6]
        return (propagate_orbit_state_with_stm if with_stm else propagate_orbit_state)(
            times=epochs,
            r0=candidate[:3],
            v0=candidate[3:6],
            t0=t0,
            **({"dense_output": True} if radio_tracking else {}),
            **kwargs,
        )

    def predict(item, trajectory):
        if item.observation.radio is not None:
            result = item.station.predict_radio(
                trajectory, item.observation.time, item.observation.measurement, item.observation.radio,
            )
            return np.atleast_1d(result.value), result.jacobian
        state = np.concatenate((trajectory.r[item.epoch_index], trajectory.v[item.epoch_index]))
        value, jacobian = GroundStationMeasurement(
            item.station, item.observation.time, item.observation.measurement,
        )(state)
        return value, jacobian @ trajectory.stm[item.epoch_index] if hasattr(trajectory, "stm") else None

    def evaluate(scaled, *, with_stm=False):
        candidate = np.asarray(scaled, dtype=float) * scale
        parameters = MappingProxyType(dict(zip(names, candidate[6:])))
        trajectory = propagate(candidate, parameters, with_stm or radio_tracking)
        physical_blocks = []
        residual_blocks = []
        jacobian_blocks = []
        for item in prepared:
            prediction = predict(item, trajectory)
            bias = np.zeros(item.value.shape) if measurement_bias is None else np.atleast_1d(
                np.asarray(measurement_bias(item.station, item.observation, parameters), dtype=float)
            )
            if bias.shape != item.value.shape or not np.all(np.isfinite(bias)):
                raise ValueError("measurement_bias must return a finite bias matching the observable's dimension.")
            residual = _wrap_first(item.value - prediction[0] - bias, item.observation.measurement)
            physical_blocks.append(residual)
            residual_blocks.append(np.linalg.solve(item.whitener, residual))
            if prediction[1] is not None:
                jacobian_blocks.append(-np.linalg.solve(item.whitener, prediction[1] * state_scale))
        return trajectory, physical_blocks, np.concatenate(residual_blocks), np.vstack(jacobian_blocks) if jacobian_blocks else None

    def jacobian(scaled, nominal=None):
        nominal = evaluate(scaled, with_stm=True) if nominal is None else nominal
        columns = [nominal[3]]
        for column in range(6, size):
            step = (np.cbrt(np.finfo(float).eps) * max(1.0, abs(scaled[column])) if parameter_step is None
                    else parameter_step[column-6] / scale[column])
            plus, minus = scaled.copy(), scaled.copy()
            if scaled[column] - lower[column] < step:
                plus[column] += min(step, upper[column] - scaled[column])
            elif upper[column] - scaled[column] < step:
                minus[column] -= min(step, scaled[column] - lower[column])
            else:
                plus[column] += step
                minus[column] -= step
            denominator = plus[column] - minus[column]
            if denominator == 0:
                raise ValueError("parameter bounds/scale leave no representable finite-difference step.")
            positive = nominal if radio_tracking and plus[column] == scaled[column] else evaluate(plus)
            negative = nominal if radio_tracking and minus[column] == scaled[column] else evaluate(minus)
            columns.append(np.concatenate([
                np.linalg.solve(item.whitener, _wrap_first(p - m, item.observation.measurement)) / denominator
                for item, p, m in zip(prepared, positive[1], negative[1])
            ])[:, None])
        return np.hstack(columns)

    fit = least_squares(
        lambda scaled: evaluate(scaled)[2],
        scaled_initial,
        jac=jacobian,
        bounds=(lower, upper),
        **({} if max_nfev is None else {"max_nfev": int(max_nfev)}),
    )
    final_values = fit.x * scale
    final = evaluate(fit.x, with_stm=True)
    final_jacobian = jacobian(fit.x, final)
    _, singular_values, vh = np.linalg.svd(final_jacobian, full_matrices=False)
    tolerance = np.finfo(float).eps * max(final_jacobian.shape) * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    covariance = None
    if numerical_rank == size:
        covariance = scale[:, None] * ((vh.T / singular_values**2) @ vh) * scale[None, :]
    message = str(fit.message)
    success = bool(fit.success and numerical_rank == size)
    if numerical_rank < size:
        message = f"{message}; rank deficient (numerical rank {numerical_rank}/{size})"
    return BatchOrbitFitResult(
        final_values[:6],
        t0,
        final[0],
        np.concatenate(final[1]),
        final[2],
        0.5 * np.dot(final[2], final[2]),
        fit.nfev,
        message,
        numerical_rank,
        None if covariance is None else covariance[:6, :6],
        success,
        parameters=dict(zip(names, final_values[6:])),
        joint_covariance=covariance,
        active_mask=fit.active_mask,
    )


__all__ = ["BatchOrbitFitResult", "solve_batch_orbit"]
