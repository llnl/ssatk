"""Adaptive high-accuracy translational propagation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from .int_utils import acceleration_adapter

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
AccelerationModel = Callable[..., ArrayLike]


@dataclass(frozen=True)
class OrbitPropagation:
    """Integrated translational state history."""

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    nfev: int
    message: str


@dataclass(frozen=True)
class OrbitPropagationWithSTM:
    """Integrated translational state history and state transition matrices."""

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    stm: np.ndarray
    nfev: int
    message: str


def propagate_orbit_state(
    *,
    times: ArrayLike,
    orbit0=None,
    r0: ArrayLike | None = None,
    v0: ArrayLike | None = None,
    t0: float | None = None,
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | Iterable[AccelerationModel] | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-6,
    max_step: float = np.inf,
) -> OrbitPropagation:
    """Propagate an inertial ``r, v`` state with adaptive ``solve_ivp``.

    This is the preferred SSATK translational propagator when fixed-step RK4 or
    leapfrog is not accurate enough. The default ``DOP853`` method is an
    eighth-order explicit Runge-Kutta method comparable in role to high-order
    adaptive propagators used by SSAPy and GMAT-style workflows.

    ``acceleration`` is an inertial perturbing acceleration in m/s². It may be a
    single callable or an iterable of callables accepting ``(t, r, v)``,
    ``(r, v, t)``, ``(r, t)``, ``(r, v)``, or ``(r)``. Canonical parameter
    names declare argument roles; other names require
    ``acceleration_adapter(model, signature)`` or ``model.acceleration_signature``.
    All callback epochs are absolute GPS seconds.
    """

    times = _times(times)
    r0, v0, t0 = _initial_orbit_state(orbit0=orbit0, r0=r0, v0=v0, t0=t0)
    if times[0] < t0 or times[-1] < t0:
        raise ValueError("times must be at or after the initial epoch t0.")

    models = _models(acceleration)
    y0 = np.concatenate([r0, v0])

    # Integrate in seconds since t0. SciPy bounds the minimum step at
    # 10 * ulp(t), which on absolute GPS seconds is 1.5e-7 s at GPS 1e8 and
    # 2.4e-6 s at GPS 1.4e9; a step-discontinuous acceleration then aborts the
    # run. Acceleration models still receive the absolute epoch.
    t_ref = float(t0)
    sol = solve_ivp(
        lambda t, y: _rhs(t_ref + t, y, mu=mu, models=models),
        (0.0, float(times[-1]) - t_ref),
        y0,
        t_eval=times - t_ref,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"orbit propagation failed: {sol.message}")

    y = sol.y.T
    return OrbitPropagation(
        t=sol.t + t_ref,
        r=y[:, :3],
        v=y[:, 3:],
        nfev=int(sol.nfev),
        message=str(sol.message),
    )


def propagate_orbit_state_with_stm(
    *,
    times: ArrayLike,
    orbit0=None,
    r0: ArrayLike | None = None,
    v0: ArrayLike | None = None,
    t0: float | None = None,
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | Iterable[AccelerationModel] | None = None,
    stm0: ArrayLike | None = None,
    fd_step: float = 1.0e-6,
    method: str = "DOP853",
    rtol: float = 1.0e-10,
    atol: float = 1.0e-9,
    max_step: float = np.inf,
) -> OrbitPropagationWithSTM:
    """Propagate an inertial state and its 6x6 state transition matrix.

    The matrix maps perturbations in the initial ``[r, v]`` state to first
    order perturbations at each output time. The unperturbed central-gravity
    Jacobian is analytic; arbitrary SSATK acceleration models use centered
    finite differences so they require no derivative API.
    """

    times = _times(times)
    r0, v0, t0 = _initial_orbit_state(orbit0=orbit0, r0=r0, v0=v0, t0=t0)
    if times[0] < t0 or times[-1] < t0:
        raise ValueError("times must be at or after the initial epoch t0.")
    if fd_step <= 0.0:
        raise ValueError("fd_step must be positive.")
    initial_stm = np.eye(6) if stm0 is None else np.asarray(stm0, dtype=float)
    if initial_stm.shape != (6, 6):
        raise ValueError("stm0 must have shape (6, 6).")

    models = _models(acceleration)
    y0 = np.concatenate([r0, v0, initial_stm.ravel()])

    # See propagate_orbit_state: integrate in seconds since t0 so the step
    # floor stays at zero-epoch resolution, and hand models the absolute epoch.
    t_ref = float(t0)

    def rhs(t, y):
        epoch = t_ref + t
        state = y[:6]
        matrix = y[6:].reshape(6, 6)
        derivative = _rhs(epoch, state, mu=mu, models=models)
        jacobian = _state_jacobian(
            epoch, state, mu=mu, models=models, relative_step=fd_step
        )
        return np.concatenate([derivative, (jacobian @ matrix).ravel()])

    sol = solve_ivp(
        rhs,
        (0.0, float(times[-1]) - t_ref),
        y0,
        t_eval=times - t_ref,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"orbit STM propagation failed: {sol.message}")

    state = sol.y[:6].T
    return OrbitPropagationWithSTM(
        t=sol.t + t_ref,
        r=state[:, :3],
        v=state[:, 3:],
        stm=sol.y[6:].T.reshape(-1, 6, 6),
        nfev=int(sol.nfev),
        message=str(sol.message),
    )


def _rhs(t: float, y: np.ndarray, *, mu: float, models: tuple[AccelerationModel, ...]) -> np.ndarray:
    r = y[:3]
    v = y[3:]
    radius = np.linalg.norm(r)
    a = np.zeros(3) if mu == 0.0 or radius == 0.0 else -mu * r / radius**3
    for model in models:
        a = a + _vector3(model(t, r, v), "acceleration")
    return np.concatenate([v, a])


def _finite_difference_jacobian(
    t: float,
    y: np.ndarray,
    *,
    mu: float,
    models: tuple[AccelerationModel, ...],
    relative_step: float,
) -> np.ndarray:
    # Finite differences keep arbitrary force models supported; add analytic or
    # complex-step Jacobians only if profiling shows this covariance path needs them.
    jacobian = np.empty((6, 6), dtype=float)
    for column in range(6):
        step = relative_step * max(1.0, abs(float(y[column])))
        plus = y.copy()
        minus = y.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            _rhs(t, plus, mu=mu, models=models) - _rhs(t, minus, mu=mu, models=models)
        ) / (2.0 * step)
    return jacobian


def _state_jacobian(
    t: float,
    y: np.ndarray,
    *,
    mu: float,
    models: tuple[AccelerationModel, ...],
    relative_step: float,
) -> np.ndarray:
    if models:
        return _finite_difference_jacobian(
            t, y, mu=mu, models=models, relative_step=relative_step
        )
    return _kepler_jacobian(y[:3], mu)


def _kepler_jacobian(r: np.ndarray, mu: float) -> np.ndarray:
    jacobian = np.zeros((6, 6), dtype=float)
    jacobian[:3, 3:] = np.eye(3)
    radius = np.linalg.norm(r)
    if mu == 0.0 or radius == 0.0:
        return jacobian
    radius3 = radius**3
    jacobian[3:, :3] = -mu * (
        np.eye(3) / radius3 - 3.0 * np.outer(r, r) / radius**5
    )
    return jacobian


def _call_acceleration(model: AccelerationModel, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _vector3(acceleration_adapter(model)(t, r, v), "acceleration")


def _models(acceleration) -> tuple[AccelerationModel, ...]:
    if acceleration is None:
        return ()
    if callable(acceleration):
        acceleration = (acceleration,)
    return tuple(acceleration_adapter(model) for model in acceleration if model is not None)


def _initial_orbit_state(*, orbit0, r0, v0, t0) -> tuple[np.ndarray, np.ndarray, float]:
    if orbit0 is not None:
        if r0 is not None or v0 is not None:
            raise ValueError("Provide either orbit0 or r0/v0, not both.")
        r0 = orbit0.r
        v0 = orbit0.v
        t0 = getattr(orbit0, "t", 0.0) if t0 is None else t0
    if r0 is None or v0 is None:
        raise ValueError("r0 and v0 are required when orbit0 is not provided.")
    return _vector3(r0, "r0"), _vector3(v0, "v0"), 0.0 if t0 is None else float(t0)


def _times(times: ArrayLike) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _vector3(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return value
