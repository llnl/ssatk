from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU, EARTH_RADIUS
from ..coordinates.attitude import (
    attitude_quaternion_from_frame,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_matrix,
    quaternion_multiply,
    rotate_vector,
)
from ..coordinates.satellite_frames import frame_to_gcrf_matrix
from ._constraints import projected_solver

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
AccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
BodyAccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
NTWAccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
TorqueModel = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike]
MassFlowRateModel = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
WheelTorqueModel = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike]

__all__ = [
    "AccelerationModel",
    "BodyAccelerationModel",
    "MassFlowRateModel",
    "NTWAccelerationModel",
    "SixDOFState",
    "SixDOFTrajectory",
    "Spacecraft",
    "TorqueModel",
    "WheelTorqueModel",
    "altitude_crossing_event",
    "attitude_quaternion_from_frame",
    "gravity_gradient_torque",
    "mass_floor_event",
    "normalize_quaternion",
    "propagate_6dof",
    "propellant_empty_event",
    "quaternion_conjugate",
    "quaternion_from_matrix",
    "quaternion_multiply",
    "radius_crossing_event",
    "rotate_vector",
    "sixdof_rhs",
]


@dataclass(frozen=True)
class SixDOFTrajectory:
    """Integrated 6-DoF trajectory.

    Quaternion convention is ``[w, x, y, z]`` and rotates body-frame vectors
    into the inertial frame. Angular rates are body-frame rad/s.
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    mass: np.ndarray | None = None
    wheel_momentum: np.ndarray | None = None
    nfev: int = 0
    message: str = ""
    status: int = 0
    t_events: tuple[np.ndarray, ...] | None = None
    y_events: tuple[np.ndarray, ...] | None = None
    solution: Callable | None = None
    event_functions: tuple[Callable | None, ...] | None = None

    def spacecraft(
        self,
        index: int = -1,
        *,
        inertia: ArrayLike | None = None,
        mass: float | None = None,
        area: float | None = None,
        cd: float | None = None,
        cr: float | None = None,
        center_of_pressure: ArrayLike | None = None,
        body: object | None = None,
        tank_name: str | None = None,
    ) -> Spacecraft:
        """Return one trajectory sample as a :class:`Spacecraft` state."""

        sample_mass = (
            mass
            if mass is not None
            else None if self.mass is None else float(self.mass[index])
        )
        sample_wheel_momentum = (
            None if self.wheel_momentum is None else self.wheel_momentum[index]
        )
        sample_body = _body_at_state(body, sample_mass, sample_wheel_momentum, tank_name)
        return Spacecraft(
            r=self.r[index],
            v=self.v[index],
            t=float(self.t[index]),
            q=self.q[index],
            omega=self.omega[index],
            wheel_momentum=sample_wheel_momentum,
            inertia=inertia,
            mass=sample_mass,
            area=area,
            cd=cd,
            cr=cr,
            center_of_pressure=center_of_pressure,
            body=sample_body,
        )


@dataclass(frozen=True)
class SixDOFState:
    """Single 6-DoF state using the same convention as ``SixDOFTrajectory``."""

    r: np.ndarray
    v: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    t: float = 0.0
    mass: float | None = None
    wheel_momentum: np.ndarray | None = None


@dataclass(frozen=True)
class Spacecraft:
    """Orbit-like spacecraft state with attitude and optional physical properties.

    ``r`` and ``v`` are inertial/GCRF position and velocity. ``q`` is a
    body-to-inertial quaternion ``[w, x, y, z]``. ``omega`` is the body-frame
    angular-rate vector in rad/s.
    """

    r: ArrayLike
    v: ArrayLike
    t: float = 0.0
    q: ArrayLike = (1.0, 0.0, 0.0, 0.0)
    omega: ArrayLike = (0.0, 0.0, 0.0)
    wheel_momentum: ArrayLike | None = None
    inertia: ArrayLike | None = None
    mass: float | None = None
    area: float | None = None
    cd: float | None = None
    cr: float | None = None
    center_of_pressure: ArrayLike | None = None
    body: object | None = None
    orbit: object | None = None

    def __post_init__(self) -> None:
        body = self.body
        object.__setattr__(self, "r", _as_vector3(self.r, "r"))
        object.__setattr__(self, "v", _as_vector3(self.v, "v"))
        object.__setattr__(self, "q", normalize_quaternion(self.q))
        object.__setattr__(self, "omega", _as_vector3(self.omega, "omega"))
        object.__setattr__(
            self,
            "wheel_momentum",
            _initial_wheel_momentum(self.wheel_momentum, body),
        )
        object.__setattr__(self, "t", float(self.t))
        if body is not None:
            if self.mass is None:
                object.__setattr__(self, "mass", _body_value(body, "current_mass", "mass"))
            if self.inertia is None:
                inertia = _body_value(body, "current_inertia", "inertia", default=None)
                if inertia is not None:
                    object.__setattr__(self, "inertia", inertia)
            if self.area is None:
                area = _body_value(body, "area", "reference_area", default=None)
                if area is not None:
                    object.__setattr__(self, "area", area)
        if self.inertia is not None:
            object.__setattr__(self, "inertia", _inertia_matrix(self.inertia))
        if self.mass is not None and self.mass <= 0.0:
            raise ValueError("mass must be positive when provided.")
        for name in ("area", "cd", "cr"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.center_of_pressure is not None:
            object.__setattr__(
                self,
                "center_of_pressure",
                _as_vector3(self.center_of_pressure, "center_of_pressure"),
            )

    @classmethod
    def from_orbit(
        cls,
        orbit,
        *,
        q: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        omega: ArrayLike = (0.0, 0.0, 0.0),
        wheel_momentum: ArrayLike | None = None,
        inertia: ArrayLike | None = None,
        mass: float | None = None,
        area: float | None = None,
        cd: float | None = None,
        cr: float | None = None,
        center_of_pressure: ArrayLike | None = None,
        body: object | None = None,
        t: float | None = None,
    ) -> Spacecraft:
        """Build a spacecraft state from an SSAPy-style object with ``r/v/t``."""

        return cls(
            r=orbit.r,
            v=orbit.v,
            t=getattr(orbit, "t", 0.0) if t is None else t,
            q=q,
            omega=omega,
            wheel_momentum=wheel_momentum,
            inertia=inertia,
            mass=mass,
            area=area,
            cd=cd,
            cr=cr,
            center_of_pressure=center_of_pressure,
            body=body,
            orbit=orbit,
        )

    def state(self) -> SixDOFState:
        """Return the numerical 6-DoF state."""

        return SixDOFState(
            r=self.r.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            omega=self.omega.copy(),
            t=self.t,
            mass=self.mass,
            wheel_momentum=None if self.wheel_momentum is None else self.wheel_momentum.copy(),
        )

    def to_orbit(self, *, mu: float = EARTH_MU, propkw=None):
        """Return an SSAPy ``Orbit`` carrying this spacecraft's ``r/v/t`` state."""

        from ssapy import Orbit

        return Orbit(self.r.copy(), self.v.copy(), self.t, mu=mu, propkw=propkw)

    def with_body(self, body: object) -> Spacecraft:
        """Return a copy with a spacecraft body design attached."""

        return Spacecraft(
            r=self.r,
            v=self.v,
            t=self.t,
            q=self.q,
            omega=self.omega,
            wheel_momentum=self.wheel_momentum,
            inertia=self.inertia,
            mass=self.mass,
            area=self.area,
            cd=self.cd,
            cr=self.cr,
            center_of_pressure=self.center_of_pressure,
            body=body,
            orbit=self.orbit,
        )

    def propagate(
        self,
        *,
        times: ArrayLike,
        inertia: ArrayLike | None = None,
        body: object | None = None,
        models=None,
        stop_at_dry_mass: bool = False,
        **kwargs,
    ) -> SixDOFTrajectory:
        """Propagate this spacecraft using :func:`propagate_6dof`.

        Models with ``mass_flow_rate`` deplete propagated mass whether they are
        passed through ``models=[...]`` or directly as ``acceleration=...``.
        Tanked bodies coast at dry mass by default; set ``stop_at_dry_mass=True``
        to terminate when propellant is exhausted.
        """

        spacecraft = self.with_body(body) if body is not None else self
        if models is not None:
            models = tuple(models)
            if "acceleration" not in kwargs:
                kwargs["acceleration"] = _sum_model_accelerations(models)
            if "torque" not in kwargs and _any_torque_model(models):
                kwargs["torque"] = _sum_model_torques(models)
            if "mass_flow_rate" not in kwargs and _any_mass_flow_model(models):
                kwargs["mass_flow_rate"] = _sum_model_mass_flow_rates(models)
        if "mass_flow_rate" not in kwargs:
            mass_models = _direct_mass_flow_models(
                kwargs.get("acceleration"),
                kwargs.get("torque"),
                kwargs.get("body_acceleration"),
                kwargs.get("ntw_acceleration"),
            )
            if mass_models:
                kwargs["mass_flow_rate"] = _sum_model_mass_flow_rates(mass_models)
        mass_is_propagated = "mass_flow_rate" in kwargs
        tank_name = _tank_name_for_models(
            kwargs.get("acceleration"), kwargs.get("torque"), kwargs.get("mass_flow_rate")
        )
        inertia = (
            _body_inertia_model(spacecraft, tank_name=tank_name)
            if inertia is None and mass_is_propagated and _supports_body_mass_update(spacecraft.body)
            else spacecraft.inertia if inertia is None else inertia
        )
        if inertia is None:
            raise ValueError("inertia is required to propagate a Spacecraft.")
        if (
            getattr(kwargs.get("torque"), "spacecraft_wheel_torque_model", False)
            and "wheel_torque" not in kwargs
        ):
            kwargs["wheel_torque"] = kwargs.pop("torque")
        if "acceleration" in kwargs:
            kwargs["acceleration"] = _bind_spacecraft_acceleration(kwargs["acceleration"], spacecraft)
        if "torque" in kwargs:
            kwargs["torque"] = _bind_spacecraft_torque(kwargs["torque"], spacecraft)
        body_wheel_axes = _wheel_axes_from_body(spacecraft.body)
        if body_wheel_axes is not None:
            kwargs.setdefault("wheel_axes_body", body_wheel_axes)
            kwargs.setdefault("wheel_momentum_capacity", _wheel_capacity_from_body(spacecraft.body))
        if spacecraft.wheel_momentum is not None:
            kwargs.setdefault("wheel_momentum0", spacecraft.wheel_momentum)
        if models is not None and _any_wheel_torque_model(models) and "wheel_torque" not in kwargs:
            kwargs["wheel_torque"] = _bind_spacecraft_wheel_torque(
                _sum_model_wheel_torques(models),
                spacecraft,
            )
        if "wheel_torque" in kwargs:
            kwargs["wheel_torque"] = _bind_spacecraft_wheel_torque(kwargs["wheel_torque"], spacecraft)
        if "mass_flow_rate" in kwargs:
            kwargs["mass_flow_rate"] = _bind_spacecraft_mass_flow_rate(kwargs["mass_flow_rate"], spacecraft)
            if _supports_body_mass_update(spacecraft.body):
                kwargs["mass_flow_rate"] = _coast_mass_flow_rate(
                    kwargs["mass_flow_rate"], spacecraft, tank_name=tank_name
                )
            kwargs.setdefault("mass0", spacecraft.mass)
        if (
            stop_at_dry_mass
            and "mass_flow_rate" in kwargs
            and _supports_body_mass_update(spacecraft.body)
        ):
            kwargs["events"] = _append_solve_ivp_event(
                kwargs.get("events"),
                propellant_empty_event(spacecraft, tank_name=tank_name),
            )
            for name in ("acceleration", "torque"):
                if name in kwargs:
                    kwargs[name] = _allow_depleted_model(kwargs[name], spacecraft, name)
        if (
            not stop_at_dry_mass
            and "mass_flow_rate" in kwargs
            and _supports_body_mass_update(spacecraft.body)
        ):
            trajectory = _propagate_spacecraft_with_dry_mass_coast(
                spacecraft,
                times,
                inertia,
                kwargs,
            )
        else:
            trajectory = _propagate_spacecraft_once(spacecraft, times, inertia, kwargs)
        if not stop_at_dry_mass and _supports_body_mass_update(spacecraft.body):
            trajectory = _clamp_dry_mass(trajectory, spacecraft.body.dry_mass_total)
        return trajectory


def radius_crossing_event(radius: float, *, terminal: bool = True, direction: float = 0.0):
    """Return a ``solve_ivp`` event for crossing an inertial radius in meters."""

    radius = _positive_float(radius, "radius")

    def event(_t, y):
        return float(np.linalg.norm(np.asarray(y[:3], dtype=float)) - radius)

    event.terminal = bool(terminal)
    event.direction = float(direction)
    return event


def altitude_crossing_event(
    altitude: float,
    *,
    earth_radius: float = EARTH_RADIUS,
    terminal: bool = True,
    direction: float = 0.0,
):
    """Return a ``solve_ivp`` event for crossing altitude above a spherical Earth."""

    return radius_crossing_event(
        _nonnegative_float(altitude, "altitude") + _positive_float(earth_radius, "earth_radius"),
        terminal=terminal,
        direction=direction,
    )


def mass_floor_event(min_mass: float, *, terminal: bool = True, direction: float = -1.0):
    """Return a ``solve_ivp`` event that stops when propagated mass reaches a floor."""

    min_mass = _positive_float(min_mass, "min_mass")

    def event(_t, y):
        y = np.asarray(y, dtype=float)
        if y.size < 14:
            raise ValueError("mass_floor_event requires mass propagation.")
        return float(y[13] - min_mass)

    event.terminal = bool(terminal)
    event.direction = float(direction)
    return event


def propellant_empty_event(
    body_or_spacecraft,
    *,
    terminal: bool = True,
    direction: float = -1.0,
    tank_name: str | None = None,
):
    """Return an event that stops mass at body or selected-tank dry mass."""

    body = getattr(body_or_spacecraft, "body", body_or_spacecraft)
    if getattr(body, "dry_mass_total", None) is None:
        raise ValueError("propellant_empty_event requires a SpacecraftBody or Spacecraft with a body.")
    return mass_floor_event(
        _propellant_mass_floor(body, tank_name), terminal=terminal, direction=direction
    )


def _propellant_mass_floor(body, tank_name=None):
    if tank_name is None:
        return float(body.dry_mass_total)
    matches = [tank for tank in body.tanks if tank.name == tank_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tank named {tank_name!r}.")
    return float(body.current_mass - matches[0].propellant_mass)


def _append_solve_ivp_event(events, event):
    if events is None:
        return event
    if callable(events):
        return (events, event)
    return tuple(events) + (event,)


def _propagate_spacecraft_once(spacecraft, times, inertia, kwargs):
    """Forward a bound spacecraft state to the low-level propagator."""

    return propagate_6dof(
        times=times,
        inertia=inertia,
        r0=spacecraft.r,
        v0=spacecraft.v,
        t0=spacecraft.t,
        q0=spacecraft.q,
        omega0=spacecraft.omega,
        **kwargs,
    )


def _propagate_spacecraft_with_dry_mass_coast(spacecraft, times, inertia, kwargs):
    """Stop at dry mass internally, then continue with propulsive models off."""

    user_events = _event_tuple(kwargs.get("events"))
    tank_name = getattr(kwargs.get("mass_flow_rate"), "tank_name", None)
    coast_kwargs = dict(kwargs)
    coast_kwargs.pop("mass_flow_rate", None)
    coast_kwargs["events"] = user_events or None
    for name in ("ntw_acceleration", "body_acceleration"):
        if _is_propulsive_model(coast_kwargs.get(name)):
            coast_kwargs.pop(name)
    if _propellant_depleted(spacecraft, tank_name):
        return _propagate_spacecraft_once(spacecraft, times, inertia, coast_kwargs)

    dry_event = propellant_empty_event(spacecraft, tank_name=tank_name)
    burn_kwargs = dict(kwargs)
    burn_kwargs["events"] = user_events + (dry_event,)
    for name in ("acceleration", "torque"):
        if name in burn_kwargs:
            burn_kwargs[name] = _allow_depleted_model(burn_kwargs[name], spacecraft, name)
    burn = _propagate_spacecraft_once(spacecraft, times, inertia, burn_kwargs)
    dry_events = () if burn.t_events is None else burn.t_events[-1]
    if not len(dry_events):
        return _drop_event_slot(burn, len(user_events))

    event_t = float(dry_events[0])
    event_y = burn.y_events[-1][0]
    remaining_times = np.asarray(times, dtype=float)
    remaining_times = remaining_times[remaining_times > event_t]
    if not len(remaining_times):
        return _drop_event_slot(burn, len(user_events))

    coast_times = np.concatenate(([event_t], remaining_times))
    coast = _propagate_state_once(event_y, event_t, coast_times, inertia, coast_kwargs)
    return _combine_trajectory_segments(
        _drop_event_slot(burn, len(user_events)),
        coast,
        len(user_events),
    )


def _propagate_state_once(state, t0, times, inertia, kwargs):
    state = np.asarray(state, dtype=float)
    kwargs = dict(kwargs)
    kwargs.pop("mass0", None)
    kwargs.pop("wheel_momentum0", None)
    wheel_axes = kwargs.get("wheel_axes_body")
    n_wheels = 0 if wheel_axes is None else _wheel_axes_matrix(wheel_axes).shape[1]
    mass_state, wheel_start = _state_tail_layout(state.size, n_wheels, None)
    return propagate_6dof(
        times=times,
        inertia=inertia,
        r0=state[:3],
        v0=state[3:6],
        t0=t0,
        q0=state[6:10],
        omega0=state[10:13],
        mass0=None if not mass_state else state[13],
        wheel_momentum0=None if not n_wheels else state[wheel_start:wheel_start + n_wheels],
        **kwargs,
    )


def _event_tuple(events):
    if events is None:
        return ()
    return (events,) if callable(events) else tuple(events)


def _drop_event_slot(trajectory, index):
    if trajectory.t_events is None:
        return trajectory
    return replace(
        trajectory,
        t_events=trajectory.t_events[:index],
        y_events=None if trajectory.y_events is None else trajectory.y_events[:index],
        event_functions=None if trajectory.event_functions is None else trajectory.event_functions[:index],
    )


def _combine_trajectory_segments(first, second, event_count):
    first_slice = slice(None)
    second_slice = slice(1, None)
    t = np.concatenate((first.t[first_slice], second.t[second_slice]))
    r = np.vstack((first.r[first_slice], second.r[second_slice]))
    v = np.vstack((first.v[first_slice], second.v[second_slice]))
    q = np.vstack((first.q[first_slice], second.q[second_slice]))
    omega = np.vstack((first.omega[first_slice], second.omega[second_slice]))
    mass = None
    if first.mass is not None and second.mass is not None:
        mass = np.concatenate((first.mass[first_slice], second.mass[second_slice]))
    wheel_momentum = None
    if first.wheel_momentum is not None and second.wheel_momentum is not None:
        wheel_momentum = np.vstack((first.wheel_momentum[first_slice], second.wheel_momentum[second_slice]))
    t_events = _merge_event_arrays(first.t_events, second.t_events, event_count)
    y_events = _merge_event_arrays(first.y_events, second.y_events, event_count)
    event_functions = _merge_event_functions(first.event_functions, second.event_functions, event_count)
    solution = _piecewise_solution(first.solution, second.solution, first.t[-1])
    return replace(
        first,
        t=t,
        r=r,
        v=v,
        q=q,
        omega=omega,
        mass=mass,
        wheel_momentum=wheel_momentum,
        nfev=first.nfev + second.nfev,
        message="; ".join(item for item in (first.message, second.message) if item),
        status=second.status,
        t_events=t_events,
        y_events=y_events,
        event_functions=event_functions,
        solution=solution,
    )


def _merge_event_arrays(first, second, count):
    if count == 0:
        return None
    first = () if first is None else first
    second = () if second is None else second
    merged = []
    for index in range(count):
        left = np.asarray(first[index])
        right = np.asarray(second[index])
        if not left.size:
            merged.append(right)
        elif not right.size:
            merged.append(left)
        else:
            merged.append(np.concatenate((left, right), axis=0))
    return tuple(merged)


def _merge_event_functions(first, second, count):
    """Preserve callable metadata while joining burn/coast event slots."""
    if count == 0:
        return None
    first = () if first is None else first
    second = () if second is None else second
    return tuple(
        (first[index] if index < len(first) else
         second[index] if index < len(second) else None)
        for index in range(count)
    )


def _piecewise_solution(first, second, split):
    return _piecewise_solution_sequence((first, second), (split,))


def _piecewise_solution_sequence(solutions, breakpoints):
    """Dispatch across N dense solutions with one searchsorted.

    ``solutions[k]`` covers the epochs between ``breakpoints[k - 1]`` and
    ``breakpoints[k]``. A breakpoint itself resolves to the segment that ends
    there, matching :func:`_piecewise_solution`, whose pairwise
    ``values <= split`` sends the split epoch to the earlier segment. An
    impulsive discontinuity therefore still reads as the pre-impulse state.

    Folding :func:`_piecewise_solution` over N segments would nest N-1
    closures, so evaluating M epochs would cost O(N*M) and N Python frames.
    Here one ``np.searchsorted`` and a stable grouping of query indices give
    one call per touched segment without N full-length boolean masks.

    Returns ``None`` when any segment lacks dense output, or when the
    breakpoints are not sorted, rather than returning a solution that would
    silently map epochs to the wrong segment.
    """
    solutions = tuple(solutions)
    if not solutions or any(item is None for item in solutions):
        return None
    if len(solutions) == 1:
        return solutions[0]

    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.shape != (len(solutions) - 1,):
        raise ValueError(
            "breakpoints must hold one epoch per interior segment boundary."
        )
    if np.any(np.diff(breakpoints) < 0.0):
        return None

    def solution(t):
        values = np.asarray(t, dtype=float)
        index = np.searchsorted(breakpoints, values, side="left")
        if values.ndim == 0:
            return solutions[int(index)](t)
        flat = values.reshape(-1)
        index = np.asarray(index).reshape(-1)
        if flat.size == 0:
            # breakpoints[0] is inside the first segment's span by construction.
            probe = np.asarray(solutions[0](float(breakpoints[0])), dtype=float)
            return np.empty((probe.shape[0], 0), dtype=float)
        result = None
        order = np.argsort(index, kind="stable")
        boundaries = np.flatnonzero(np.diff(index[order])) + 1
        for selected in np.split(order, boundaries):
            segment = int(index[selected[0]])
            block = np.asarray(solutions[segment](flat[selected]), dtype=float)
            if result is None:
                result = np.empty((block.shape[0], flat.size), dtype=float)
            result[:, selected] = block
        return result

    return solution


def gravity_gradient_torque(
    r_inertial: ArrayLike,
    q: ArrayLike,
    inertia: ArrayLike,
    mu: float = EARTH_MU,
) -> np.ndarray:
    r_inertial = np.asarray(r_inertial, dtype=float)
    if r_inertial.shape != (3,):
        raise ValueError("r_inertial must be a 3-vector.")
    return _gravity_gradient_torque_prepared(r_inertial, q, _inertia_matrix(inertia), mu)


def _gravity_gradient_torque_prepared(
    r_inertial: np.ndarray,
    q: ArrayLike,
    inertia: np.ndarray,
    mu: float = EARTH_MU,
) -> np.ndarray:
    radius = np.linalg.norm(r_inertial)
    if radius == 0.0:
        return np.zeros(3)
    r_hat_body = rotate_vector(quaternion_conjugate(q), r_inertial / radius)
    return 3.0 * mu / radius**3 * np.cross(r_hat_body, inertia @ r_hat_body)



def sixdof_rhs(
    t: float,
    y: ArrayLike,
    *,
    inertia: ArrayLike,
    inertia_rate=None,
    angular_momentum_flux=None,
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | None = None,
    ntw_acceleration: NTWAccelerationModel | None = None,
    body_acceleration: BodyAccelerationModel | None = None,
    torque: TorqueModel | None = None,
    gravity_gradient: bool = False,
    mass_flow_rate: MassFlowRateModel | None = None,
    wheel_axes_body: ArrayLike | None = None,
    wheel_torque: WheelTorqueModel | None = None,
    wheel_momentum_capacity: ArrayLike | None = None,
    inv_inertia: ArrayLike | None = None,
    mass_state: bool | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    wheel_axes = _wheel_axes_matrix(wheel_axes_body)
    n_wheels = 0 if wheel_axes is None else wheel_axes.shape[1]
    wheel_capacity = _wheel_capacity_vector(wheel_momentum_capacity, n_wheels)
    has_mass, wheel_start = _state_tail_layout(y.size, n_wheels, mass_state)
    r = y[0:3]
    v = y[3:6]
    q = normalize_quaternion(y[6:10])
    omega = y[10:13]
    mass = float(y[13]) if has_mass else None
    wheel_momentum = None if n_wheels == 0 else y[wheel_start:wheel_start + n_wheels]
    mass_floor = None if mass_flow_rate is None else getattr(mass_flow_rate, "mass_floor", None)
    if mass is not None and mass_floor is not None:
        mass = max(mass, float(mass_floor))
    elif mass is not None and mass <= 0.0:
        raise ValueError("mass state must remain positive.")
    inertia_model = inertia
    if inertia_rate is None:
        inertia_rate = getattr(inertia_model, "inertia_rate", None)
    if angular_momentum_flux is None:
        angular_momentum_flux = getattr(inertia_model, "angular_momentum_flux", None)
    if inv_inertia is None:
        inertia = _inertia_at_state(inertia, t, r, v, q, omega, mass)
        inv_inertia = np.linalg.inv(inertia)
    else:
        inertia = np.asarray(inertia, dtype=float)
        inv_inertia = np.asarray(inv_inertia, dtype=float)

    radius = np.linalg.norm(r)
    a = np.zeros(3) if mu == 0.0 or radius == 0.0 else -mu * r / radius**3
    if acceleration is not None:
        a = a + _evaluate_acceleration(acceleration, t, r, v, q, omega, mass, wheel_momentum)
    if ntw_acceleration is not None:
        a = a + frame_to_gcrf_matrix("ntw", r=r, v=v) @ _as_vector3(
            ntw_acceleration(t, r, v, q, omega),
            "ntw_acceleration",
        )
    if body_acceleration is not None:
        a = a + rotate_vector(
            q,
            _as_vector3(body_acceleration(t, r, v, q, omega), "body_acceleration"),
        )

    q_dot = 0.5 * quaternion_multiply(q, [0.0, *omega])
    torque_body = np.zeros(3)
    if gravity_gradient:
        torque_body = torque_body + _gravity_gradient_torque_prepared(r, q, inertia, mu=mu)
    if torque is not None:
        torque_body = torque_body + _evaluate_torque(torque, t, r, v, q, omega, mass, wheel_momentum)
    wheel_torque_scalars = np.zeros(n_wheels)
    wheel_h_body = np.zeros(3)
    if n_wheels:
        wheel_h_body = wheel_axes @ wheel_momentum
        if wheel_torque is not None:
            wheel_torque_scalars = _evaluate_wheel_torque(
                wheel_torque,
                t,
                r,
                v,
                q,
                omega,
                mass,
                wheel_momentum,
                n_wheels,
            )
            wheel_torque_scalars = _apply_wheel_capacity(
                wheel_torque_scalars,
                wheel_momentum,
                wheel_capacity,
            )
            torque_body = torque_body + wheel_axes @ wheel_torque_scalars
    mass_rate = (
        0.0 if mass is None or mass_flow_rate is None else
        -_evaluate_mass_flow_rate(mass_flow_rate, t, r, v, q, omega, mass, wheel_momentum)
    )
    if angular_momentum_flux is not None:
        flux = _as_vector3(_mass_property_value(
            angular_momentum_flux, t, r, v, q, omega, mass, mass_rate
        ), "angular_momentum_flux")
        if not np.all(np.isfinite(flux)):
            raise ValueError("angular_momentum_flux must contain finite values.")
        torque_body = torque_body + flux
    angular_rhs = torque_body - np.cross(omega, inertia @ omega + wheel_h_body)
    if inertia_rate is not None:
        rate = _inertia_rate_matrix(_mass_property_value(
            inertia_rate, t, r, v, q, omega, mass, mass_rate
        ))
        omega_dot = inv_inertia @ (angular_rhs - rate @ omega)
    elif callable(inertia_model) and np.any(omega):
        rate = _inertia_directional_rate(
            inertia_model, t, r, v, q, omega, mass, v, a, q_dot, mass_rate
        )
        angular_mass = _angular_mass_matrix(
            inertia_model, inertia, t, r, v, q, omega, mass
        )
        omega_dot = np.linalg.solve(angular_mass, angular_rhs - rate @ omega)
    else:
        omega_dot = inv_inertia @ angular_rhs

    derivative = [v, a, q_dot, omega_dot]
    if mass is not None:
        derivative.append([mass_rate])
    if n_wheels:
        derivative.append(-wheel_torque_scalars)
    return np.concatenate(derivative)


def propagate_6dof(
    *,
    times: ArrayLike,
    inertia: ArrayLike,
    inertia_rate=None,
    angular_momentum_flux=None,
    orbit0=None,
    r0: ArrayLike | None = None,
    v0: ArrayLike | None = None,
    t0: float | None = None,
    q0: ArrayLike | None = None,
    omega0: ArrayLike | None = None,
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | None = None,
    ntw_acceleration: NTWAccelerationModel | None = None,
    body_acceleration: BodyAccelerationModel | None = None,
    torque: TorqueModel | None = None,
    gravity_gradient: bool = False,
    mass0: float | None = None,
    mass_flow_rate: MassFlowRateModel | None = None,
    wheel_momentum0: ArrayLike | None = None,
    wheel_axes_body: ArrayLike | None = None,
    wheel_torque: WheelTorqueModel | None = None,
    wheel_momentum_capacity: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = "DOP853",
    max_step: float = np.inf,
    first_step: float | None = None,
    events=None,
    dense_output: bool = False,
) -> SixDOFTrajectory:
    """Propagate a coupled translational and rigid-body attitude state.

    ``acceleration`` returns inertial/GCRF m/s². ``ntw_acceleration`` returns
    SSAPy-style ``[N, T, W]`` m/s², where ``T`` follows velocity and ``W``
    follows ``r × v``. ``body_acceleration`` returns body-frame m/s² and is
    rotated into the inertial frame by the current attitude. ``torque`` returns
    body-frame N m. Use ``gravity_gradient=True`` to add the standard
    rigid-body gravity-gradient torque. Without attitude-dependent
    acceleration, attitude does not affect the orbit trajectory.
    ``inertia_rate`` supplies the total body-coordinate derivative of a
    changing inertia tensor as a matrix or callable; ``angular_momentum_flux``
    supplies net inward angular momentum carried by mass flow. For callable
    inertia, the omitted rate is differentiated along the state trajectory,
    including mass, translation, attitude, and angular-rate dependence.
    ``events`` and ``dense_output`` are passed directly to SciPy ``solve_ivp``
    for event-driven propagation segments.
    """

    times = _times(times)
    if wheel_axes_body is None and orbit0 is not None:
        wheel_axes_body = _wheel_axes_from_body(getattr(orbit0, "body", None))
    if wheel_momentum_capacity is None and orbit0 is not None:
        wheel_momentum_capacity = _wheel_capacity_from_body(getattr(orbit0, "body", None))
    state = _initial_state(
        orbit0=orbit0,
        r0=r0,
        v0=v0,
        t0=t0,
        q0=q0,
        omega0=omega0,
        mass0=mass0,
        wheel_momentum0=wheel_momentum0,
    )
    _validate_time_direction(times, state.t)
    if mass_flow_rate is not None and state.mass is None:
        raise ValueError("mass0 or orbit0.mass is required when mass_flow_rate is provided.")
    wheel_axes = _wheel_axes_matrix(wheel_axes_body)
    if wheel_torque is not None and wheel_axes is None:
        raise ValueError("wheel_axes_body or orbit0.body.reaction_wheels is required when wheel_torque is provided.")
    wheel_capacity = _wheel_capacity_vector(
        wheel_momentum_capacity,
        0 if wheel_axes is None else wheel_axes.shape[1],
    )
    wheel_momentum = state.wheel_momentum
    if wheel_axes is not None:
        if wheel_momentum is None:
            wheel_momentum = np.zeros(wheel_axes.shape[1])
        if wheel_momentum.shape != (wheel_axes.shape[1],):
            raise ValueError("wheel_momentum0 must match the number of wheel axes.")
        _validate_initial_wheel_momentum(wheel_momentum, wheel_capacity)
    if callable(inertia):
        inertia_arg = inertia
        inv_inertia = None
    else:
        inertia_arg = _inertia_matrix(inertia)
        inv_inertia = np.linalg.inv(inertia_arg)
    y_parts = [state.r, state.v, state.q, state.omega]
    if state.mass is not None:
        y_parts.append([state.mass])
    if wheel_momentum is not None:
        y_parts.append(wheel_momentum)
    y0 = np.concatenate(y_parts)

    # Integrate in seconds since ``state.t``. SciPy's Runge-Kutta steppers bound
    # the minimum step at ``10 * ulp(t)``, which on absolute GPS seconds is
    # 2.4e-6 s at GPS 1.4e9 and 4.8e-6 s at GPS 3.8e9, against ~5e-322 s at
    # zero. A discontinuity needing a finer step -- tank depletion, a thrust
    # cut-out -- then aborts the run outright. The force, torque and event
    # callbacks still see absolute epochs, and every returned epoch is absolute.
    t_ref = float(state.t)
    times_elapsed = times - t_ref
    events = _event_tuple(events) if events is not None else None
    if wheel_capacity is not None and np.any(np.isfinite(wheel_capacity)):
        project = _wheel_state_projector(
            inertia_arg, wheel_axes, wheel_capacity, state.mass is not None,
            t_ref=t_ref, mass_floor=getattr(mass_flow_rate, "mass_floor", None),
        )
        method = projected_solver(method, project)

    sol = solve_ivp(
        lambda t, y: sixdof_rhs(
            t_ref + t,
            y,
            inertia=inertia_arg,
            inertia_rate=inertia_rate,
            angular_momentum_flux=angular_momentum_flux,
            mu=mu,
            acceleration=acceleration,
            ntw_acceleration=ntw_acceleration,
            body_acceleration=body_acceleration,
            torque=torque,
            gravity_gradient=gravity_gradient,
            mass_flow_rate=mass_flow_rate,
            wheel_axes_body=wheel_axes,
            wheel_torque=wheel_torque,
            wheel_momentum_capacity=wheel_capacity,
            inv_inertia=inv_inertia,
            mass_state=state.mass is not None,
        ),
        (0.0, float(times_elapsed[-1])),
        y0,
        t_eval=times_elapsed,
        rtol=rtol,
        atol=atol,
        method=method,
        max_step=max_step,
        first_step=first_step,
        events=_events_in_elapsed_time(events, t_ref),
        dense_output=dense_output,
    )
    if not sol.success:
        raise RuntimeError(f"6-DoF propagation failed: {sol.message}")

    # With t_eval starting after a terminal event, SciPy returns empty lists.
    y = np.asarray(sol.y, dtype=float).reshape(y0.size, -1).T
    t = np.asarray(sol.t, dtype=float)
    if getattr(sol, "status", 0) == 1 and getattr(sol, "t_events", None) is not None:
        events_with_state = [
            (float(event_times[-1]), y_event[-1])
            for event_times, y_event in zip(sol.t_events, sol.y_events)
            if len(event_times) and len(y_event)
        ]
        if events_with_state:
            # Integration is forward-only; the last root is the stopping point.
            event_t, event_y = max(events_with_state, key=lambda item: item[0])
            resolution = 8.0 * np.spacing(abs(event_t + t_ref))
            if len(t) and abs(event_t - float(t[-1])) <= resolution:
                t[-1] = event_t
                y[-1] = event_y
            else:
                t = np.append(t, event_t)
                y = np.vstack([y, event_y])
    t = t + t_ref
    q = np.array([normalize_quaternion(item) for item in y[:, 6:10]])
    mass = y[:, 13] if state.mass is not None else None
    wheel_start = 14 if state.mass is not None else 13
    wheel_momentum_series = (
        None if wheel_momentum is None else y[:, wheel_start:wheel_start + wheel_momentum.size]
    )
    return SixDOFTrajectory(
        t=t,
        r=y[:, 0:3],
        v=y[:, 3:6],
        q=q,
        omega=y[:, 10:13],
        mass=mass,
        wheel_momentum=wheel_momentum_series,
        nfev=int(getattr(sol, "nfev", 0)),
        message=str(getattr(sol, "message", "")),
        status=int(getattr(sol, "status", 0)),
        t_events=(
            None
            if getattr(sol, "t_events", None) is None
            else tuple(np.asarray(item, dtype=float) + t_ref for item in sol.t_events)
        ),
        y_events=None if getattr(sol, "y_events", None) is None else tuple(sol.y_events),
        solution=_solution_in_absolute_time(getattr(sol, "sol", None), t_ref),
        event_functions=events,
    )


def _initial_state(*, orbit0, r0, v0, t0, q0, omega0, mass0=None, wheel_momentum0=None) -> SixDOFState:
    if orbit0 is not None:
        if r0 is not None or v0 is not None:
            raise ValueError("Provide either orbit0 or r0/v0, not both.")
        r0 = orbit0.r
        v0 = orbit0.v
        t0 = getattr(orbit0, "t", 0.0) if t0 is None else t0
        q0 = getattr(orbit0, "q", q0) if q0 is None else q0
        omega0 = getattr(orbit0, "omega", omega0) if omega0 is None else omega0
        mass0 = getattr(orbit0, "mass", None) if mass0 is None else mass0
        if wheel_momentum0 is None:
            wheel_momentum0 = getattr(orbit0, "wheel_momentum", None)
    if r0 is None or v0 is None:
        raise ValueError("r0 and v0 are required when orbit0 is not provided.")
    if mass0 is not None and mass0 <= 0.0:
        raise ValueError("mass0 must be positive when provided.")
    return SixDOFState(
        r=_as_vector3(r0, "r0"),
        v=_as_vector3(v0, "v0"),
        q=normalize_quaternion((1.0, 0.0, 0.0, 0.0) if q0 is None else q0),
        omega=_as_vector3((0.0, 0.0, 0.0) if omega0 is None else omega0, "omega0"),
        t=0.0 if t0 is None else float(t0),
        mass=None if mass0 is None else float(mass0),
        wheel_momentum=None if wheel_momentum0 is None else _as_vector(wheel_momentum0, "wheel_momentum0"),
    )


def _events_in_elapsed_time(events, t_ref: float):
    """Re-express absolute-epoch event callables in seconds since ``t_ref``."""
    if events is None or t_ref == 0.0:
        return events
    single = callable(events)
    shifted = tuple(
        _event_in_elapsed_time(event, t_ref)
        for event in ((events,) if single else tuple(events))
    )
    return shifted[0] if single else shifted


def _event_in_elapsed_time(event, t_ref: float):
    def shifted(t, y):
        return event(t + t_ref, y)

    for attribute in ("terminal", "direction"):
        if hasattr(event, attribute):
            setattr(shifted, attribute, getattr(event, attribute))
    return shifted


def _rhs_in_elapsed_time(rhs, t_ref: float):
    """Present an absolute-epoch right-hand side to an elapsed-time integrator."""
    if t_ref == 0.0:
        return rhs

    def shifted(t, y):
        return rhs(t_ref + t, y)

    return shifted


def _solution_in_absolute_time(solution, t_ref: float):
    """Wrap a dense solution so callers keep querying it with absolute epochs."""
    if solution is None:
        return solution

    def shifted(t):
        values = np.asarray(t, dtype=float) - t_ref
        if values.ndim == 1 and values.size == 0:
            return np.empty((np.asarray(solution(0.0)).size, 0))
        return solution(values)

    return shifted


def _wheel_state_projector(inertia, axes, capacity, has_mass, *, t_ref=0.0, mass_floor=None):
    wheel_start = 14 if has_mass else 13

    def project(t, y):
        momentum = y[wheel_start:wheel_start + len(capacity)]
        clipped = np.clip(momentum, -capacity, capacity)
        if np.array_equal(clipped, momentum):
            return y
        result = np.array(y, copy=True)
        r, v, q, omega = y[:3], y[3:6], normalize_quaternion(y[6:10]), y[10:13]
        mass = float(y[13]) if has_mass else None
        if mass is not None and mass_floor is not None:
            mass = max(mass, mass_floor)
        epoch = t_ref + t
        matrix = _inertia_at_state(inertia, epoch, r, v, q, omega, mass)
        transfer = axes @ (momentum - clipped)
        target = matrix @ omega + transfer
        corrected = omega + np.linalg.solve(matrix, transfer)
        scale = max(np.linalg.norm(target), np.linalg.norm(matrix @ omega),
                    np.linalg.norm(transfer), np.finfo(float).tiny)
        if callable(inertia):
            for _ in range(8):
                updated = _inertia_at_state(inertia, epoch, r, v, q, corrected, mass)
                residual = updated @ corrected - target
                if np.linalg.norm(residual) <= 32 * np.finfo(float).eps * scale:
                    break
                jacobian = _angular_mass_matrix(inertia, updated, epoch, r, v, q, corrected, mass)
                corrected -= np.linalg.solve(jacobian, residual)
            else:
                raise RuntimeError("wheel projection could not conserve angular momentum for this inertia model.")
        result[10:13] = corrected
        result[wheel_start:wheel_start + len(capacity)] = clipped
        return result

    return project


def _angular_mass_matrix(model, inertia, t, r, v, q, omega, mass):
    """Jacobian of ``I(omega) omega`` for nonlinear inertia projection."""
    matrix = np.array(inertia, copy=True)
    if not callable(model) or not np.any(omega):
        return matrix
    for column in range(3):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(omega[column]))
        plus, minus = omega.copy(), omega.copy()
        plus[column] += step
        minus[column] -= step
        difference = (
            _inertia_at_state(model, t, r, v, q, plus, mass)
            - _inertia_at_state(model, t, r, v, q, minus, mass)
        ) / (plus[column] - minus[column])
        matrix[:, column] += difference @ omega
    return matrix


def _epochs_close(first: float, second: float) -> bool:
    """Compare two epoch values without a scale-dependent relative tolerance.

    ``np.isclose`` defaults to ``rtol=1e-5``, which on absolute GPS seconds is
    a tolerance of hours: 1.4e4 s at GPS 1.4e9 (2024) and 3.8e4 s at GPS 3.8e9
    (2100). Epochs are compared on an absolute floor instead.

    The floor is 1 us, widened to 8 ULP of the larger operand where float64
    cannot resolve that. 1 us is only 4.2 ULP at GPS 1.4e9 and 2.1 ULP at GPS
    3.8e9, and SciPy locates events with ``brentq(xtol=4*EPS, rtol=4*EPS)``,
    whose stopping bound is 1.24 us at GPS 1.4e9 and 3.4 us at GPS 3.8e9 --
    already wider than the fixed floor. Without the ULP term a root SciPy
    resolved as well as it can would be treated as a distinct epoch and
    appended as a duplicate sample.
    """
    tolerance = max(1.0e-6, 8.0 * np.spacing(max(abs(first), abs(second))))
    return bool(np.isclose(first, second, rtol=0.0, atol=tolerance))


def _times(times: ArrayLike) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _validate_time_direction(times: np.ndarray, t0: float) -> None:
    if times[0] < t0 or times[-1] < t0:
        raise ValueError("times must be at or after the initial epoch t0.")


def _as_vector3(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return vector


def _as_vector(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return vector


def _state_tail_layout(size: int, n_wheels: int, mass_state: bool | None) -> tuple[bool, int]:
    if n_wheels < 0:
        raise ValueError("n_wheels must be non-negative.")
    if mass_state is None:
        if n_wheels:
            if size == 13 + n_wheels:
                return False, 13
            if size == 14 + n_wheels:
                return True, 14
        elif size in (13, 14):
            return size == 14, 14
    else:
        expected = 13 + int(bool(mass_state)) + n_wheels
        if size == expected:
            return bool(mass_state), 14 if mass_state else 13
    base = "r, v, q, omega"
    optional = ", optional mass, optional wheel_momentum"
    raise ValueError(f"y has {size} elements; expected {13 + int(bool(mass_state)) + n_wheels} for {base}{optional}.")


def _wheel_axes_matrix(wheel_axes_body: ArrayLike | None) -> np.ndarray | None:
    if wheel_axes_body is None:
        return None
    axes = np.asarray(wheel_axes_body, dtype=float)
    if axes.ndim == 1:
        axes = axes.reshape(3, 1)
    if axes.ndim != 2 or axes.shape[0] != 3:
        raise ValueError("wheel_axes_body must have shape (3, n_wheels).")
    norms = np.linalg.norm(axes, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("wheel_axes_body columns must be non-zero.")
    return axes / norms


def _wheel_axes_from_body(body) -> np.ndarray | None:
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if not wheels:
        return None
    return _wheel_axes_matrix(np.column_stack([_as_vector3(wheel.axis_body, "wheel.axis_body") for wheel in wheels]))


def _wheel_capacity_from_body(body) -> np.ndarray | None:
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if not wheels:
        return None
    capacities = [
        np.inf if getattr(wheel, "momentum_capacity", None) is None else float(wheel.momentum_capacity)
        for wheel in wheels
    ]
    return np.asarray(capacities, dtype=float)


def _wheel_capacity_vector(capacity, n_wheels: int) -> np.ndarray | None:
    if n_wheels == 0:
        return None
    if capacity is None:
        return np.full(n_wheels, np.inf)
    capacity = np.asarray(capacity, dtype=float)
    if capacity.ndim != 1:
        raise ValueError("wheel_momentum_capacity must be a 1-D vector.")
    if capacity.shape != (n_wheels,):
        raise ValueError("wheel_momentum_capacity must match the number of wheel axes.")
    if np.any(capacity <= 0.0) or np.any(np.isnan(capacity)):
        raise ValueError("wheel_momentum_capacity values must be positive.")
    return capacity


def _validate_initial_wheel_momentum(momentum: np.ndarray, capacity: np.ndarray | None) -> None:
    if capacity is not None and np.any(np.abs(momentum) > capacity):
        raise ValueError("wheel_momentum0 exceeds wheel_momentum_capacity.")


def _apply_wheel_capacity(command: np.ndarray, momentum: np.ndarray, capacity: np.ndarray | None) -> np.ndarray:
    if capacity is None:
        return command
    command = command.copy()
    momentum_dot = -command
    over_positive = (momentum >= capacity) & (momentum_dot > 0.0)
    over_negative = (momentum <= -capacity) & (momentum_dot < 0.0)
    command[over_positive | over_negative] = 0.0
    return command


def _initial_wheel_momentum(value, body) -> np.ndarray | None:
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if value is not None:
        momentum = _as_vector(value, "wheel_momentum")
        if wheels and momentum.size != len(wheels):
            raise ValueError("wheel_momentum must match the number of reaction wheels.")
        return momentum
    if not wheels:
        return None
    momentum = []
    for wheel in wheels:
        speed = float(getattr(wheel, "speed", 0.0))
        wheel_inertia = getattr(wheel, "wheel_inertia", None)
        if wheel_inertia is None:
            if speed != 0.0:
                raise ValueError("wheel_inertia is required when a ReactionWheel has non-zero speed.")
            momentum.append(0.0)
        else:
            momentum.append(float(wheel_inertia) * speed)
    return np.asarray(momentum, dtype=float)


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _bind_spacecraft_acceleration(model, spacecraft, *, suppress_depleted=True):
    if getattr(model, "spacecraft_acceleration_model", False) or _is_propulsive_model(model):
        def acceleration(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
            state = _spacecraft_at_state(spacecraft, t, r, v, q, omega, mass, wheel_momentum, getattr(model, "tank_name", None))
            if (
                suppress_depleted
                and _is_propulsive_model(model)
                and not getattr(model, "_depletion_handled", False)
                and _propellant_depleted(state, getattr(model, "tank_name", None))
            ):
                return np.zeros(3)
            if not getattr(model, "spacecraft_acceleration_model", False):
                return _evaluate_acceleration(model, t, r, v, q, omega, mass, wheel_momentum)
            return model(
                spacecraft=state,
                t=t,
                r=r,
                v=v,
                q=q,
                omega=omega,
            )

        acceleration.accepts_mass = True
        acceleration.accepts_wheel_momentum = True
        acceleration._source_model = model
        return acceleration
    return model


def _bind_spacecraft_torque(model, spacecraft, *, suppress_depleted=True):
    if getattr(model, "spacecraft_torque_model", False) or _is_propulsive_model(model):
        evaluator = model.torque if hasattr(model, "torque") else model
        def torque(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
            state = _spacecraft_at_state(spacecraft, t, r, v, q, omega, mass, wheel_momentum, getattr(model, "tank_name", None))
            if (
                suppress_depleted
                and _is_propulsive_model(model)
                and not getattr(model, "_depletion_handled", False)
                and _propellant_depleted(state, getattr(model, "tank_name", None))
            ):
                return np.zeros(3)
            if not getattr(model, "spacecraft_torque_model", False):
                return _evaluate_torque(model, t, r, v, q, omega, mass, wheel_momentum)
            return evaluator(
                spacecraft=state,
                t=t,
                r=r,
                v=v,
                q=q,
                omega=omega,
            )

        torque.accepts_mass = True
        torque.accepts_wheel_momentum = True
        torque._source_model = model
        return torque
    return model


def _bind_spacecraft_wheel_torque(model, spacecraft):
    if getattr(model, "spacecraft_wheel_torque_model", False):
        evaluator = model.wheel_torques if hasattr(model, "wheel_torques") else model

        def wheel_torque(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
            return evaluator(
                spacecraft=_spacecraft_at_state(spacecraft, t, r, v, q, omega, mass, wheel_momentum),
                t=t,
                r=r,
                v=v,
                q=q,
                omega=omega,
            )

        wheel_torque.accepts_mass = True
        wheel_torque.accepts_wheel_momentum = True
        return wheel_torque
    return model


def _bind_spacecraft_mass_flow_rate(model, spacecraft):
    if getattr(model, "spacecraft_mass_flow_model", False) or hasattr(model, "mass_flow_rate"):
        evaluator = model.mass_flow_rate if hasattr(model, "mass_flow_rate") else model

        def mass_flow_rate(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
            state = _spacecraft_at_state(spacecraft, t, r, v, q, omega, mass, wheel_momentum, getattr(model, "tank_name", None))
            return evaluator(
                spacecraft=state,
                t=t,
                r=r,
                v=v,
                q=q,
                omega=omega,
            )

        mass_flow_rate.accepts_mass = True
        mass_flow_rate.accepts_wheel_momentum = True
        return mass_flow_rate
    return model


def _coast_mass_flow_rate(model, spacecraft, *, tank_name=None):
    def mass_flow_rate(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
        return _evaluate_mass_flow_rate(model, t, r, v, q, omega, mass, wheel_momentum)

    mass_flow_rate.accepts_mass = True
    mass_flow_rate.accepts_wheel_momentum = True
    mass_flow_rate.mass_floor = _propellant_mass_floor(spacecraft.body, tank_name)
    mass_flow_rate.tank_name = tank_name
    return mass_flow_rate


def _evaluate_acceleration(model, t, r, v, q, omega, mass=None, wheel_momentum=None):
    if getattr(model, "accepts_wheel_momentum", False):
        return _as_vector3(model(t, r, v, q, omega, mass=mass, wheel_momentum=wheel_momentum), "acceleration")
    if getattr(model, "accepts_mass", False):
        return _as_vector3(model(t, r, v, q, omega, mass=mass), "acceleration")
    return _as_vector3(model(t, r, v, q, omega), "acceleration")


def _evaluate_torque(model, t, r, v, q, omega, mass=None, wheel_momentum=None):
    if getattr(model, "spacecraft_torque_model", False):
        evaluator = model.torque if hasattr(model, "torque") else model
        return _as_vector3(
            evaluator(t=t, r=r, v=v, q=q, omega=omega),
            "torque",
        )
    if getattr(model, "accepts_wheel_momentum", False):
        return _as_vector3(model(t, r, v, q, omega, mass=mass, wheel_momentum=wheel_momentum), "torque")
    if getattr(model, "accepts_mass", False):
        return _as_vector3(model(t, r, v, q, omega, mass=mass), "torque")
    return _as_vector3(model(t, r, v, q, omega), "torque")


def _evaluate_mass_flow_rate(model, t, r, v, q, omega, mass=None, wheel_momentum=None) -> float:
    if getattr(model, "accepts_wheel_momentum", False):
        value = model(t, r, v, q, omega, mass=mass, wheel_momentum=wheel_momentum)
    elif getattr(model, "accepts_mass", False):
        value = model(t, r, v, q, omega, mass=mass)
    else:
        value = model(t, r, v, q, omega)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("mass_flow_rate must be finite.")
    if value < 0.0:
        raise ValueError("mass_flow_rate must be non-negative.")
    return value


def _evaluate_wheel_torque(model, t, r, v, q, omega, mass, wheel_momentum, n_wheels: int) -> np.ndarray:
    if getattr(model, "accepts_wheel_momentum", False):
        value = model(t, r, v, q, omega, mass=mass, wheel_momentum=wheel_momentum)
    elif getattr(model, "accepts_mass", False):
        value = model(t, r, v, q, omega, mass=mass)
    else:
        value = model(t, r, v, q, omega)
    value = _as_vector(value, "wheel_torque")
    if value.shape != (n_wheels,):
        raise ValueError("wheel_torque must return one scalar torque per wheel axis.")
    return value


def _body_value(body, *names: str, default=None):
    for name in names:
        if hasattr(body, name):
            return getattr(body, name)
    return default


def _spacecraft_at_state(spacecraft, t, r, v, q, omega, mass=None, wheel_momentum=None, tank_name=None):
    if spacecraft is None:
        return None
    body = _body_at_state(spacecraft.body, mass, wheel_momentum, tank_name)
    inertia = (
        _body_value(body, "current_inertia", "inertia", default=spacecraft.inertia)
        if body is not None
        else spacecraft.inertia
    )
    return SimpleNamespace(
        r=r,
        v=v,
        t=t,
        q=q,
        omega=omega,
        wheel_momentum=wheel_momentum,
        inertia=inertia,
        mass=spacecraft.mass if mass is None else float(mass),
        area=spacecraft.area,
        cd=spacecraft.cd,
        cr=spacecraft.cr,
        center_of_pressure=spacecraft.center_of_pressure,
        body=body,
        orbit=spacecraft.orbit,
    )


def _body_at_state(body, mass=None, wheel_momentum=None, tank_name=None):
    body = _body_at_mass(body, mass, tank_name)
    return _body_at_wheel_momentum(body, wheel_momentum)


def _body_at_mass(body, mass, tank_name=None):
    if body is None or mass is None or not _supports_body_mass_update(body):
        return body
    try:
        if tank_name is None:
            return body.with_current_mass(float(mass))
        matches = [tank for tank in body.tanks if tank.name == tank_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one tank named {tank_name!r}.")
        tank = matches[0]
        consumed = body.current_mass - float(mass)
        return body.with_tank_propellant_mass(tank_name, max(0.0, tank.propellant_mass - consumed))
    except ValueError as exc:
        if "below dry mass" in str(exc):
            return body.with_propellant_mass(0.0)
        raise


def _supports_body_mass_update(body) -> bool:
    return bool(body is not None and getattr(body, "tanks", ()) and hasattr(body, "with_current_mass"))


def _body_at_wheel_momentum(body, wheel_momentum):
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if body is None or wheel_momentum is None or not wheels or not hasattr(body, "with_reaction_wheels"):
        return body
    wheel_momentum = _as_vector(wheel_momentum, "wheel_momentum")
    if wheel_momentum.shape != (len(wheels),):
        raise ValueError("wheel_momentum must match the number of reaction wheels.")
    updated = []
    for wheel, momentum in zip(wheels, wheel_momentum):
        wheel_inertia = getattr(wheel, "wheel_inertia", None)
        if wheel_inertia is None:
            updated.append(wheel)
        else:
            updated.append(wheel.with_updates(speed=float(momentum) / float(wheel_inertia)))
    return body.with_reaction_wheels(*updated, append=False)


def _body_inertia_model(spacecraft, *, tank_name=None):
    def inertia(t, r, v, q, omega, *, mass=None):
        state_body = _body_at_mass(spacecraft.body, mass, tank_name)
        return _body_value(state_body, "current_inertia", "inertia", default=spacecraft.inertia)

    def inertia_rate(t, r, v, q, omega, *, mass=None, mass_rate=0.0):
        state_body = _body_at_mass(spacecraft.body, mass, tank_name)
        rate = np.zeros((3, 3))
        if mass_rate == 0.0:
            return rate
        original_tanks = tuple(spacecraft.body.tanks)
        total = sum(tank.propellant_mass for tank in original_tanks)
        for original, tank in zip(original_tanks, state_body.tanks):
            fraction = (
                float(tank.name == tank_name) if tank_name is not None else
                original.propellant_mass / total if total else 0.0
            )
            if tank.propellant_mass <= 0.0 and mass_rate < 0.0:
                continue
            offset = tank.position_body - state_body.current_center_of_mass
            rate += mass_rate * fraction * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        return rate

    def flux(t, r, v, q, omega, *, mass=None, mass_rate=0.0):
        return inertia_rate(
            t, r, v, q, omega, mass=mass, mass_rate=mass_rate
        ) @ omega

    inertia.accepts_mass = True
    inertia.inertia_rate = inertia_rate
    inertia.angular_momentum_flux = flux
    return inertia


def _inertia_at_state(inertia, t, r, v, q, omega, mass=None):
    if callable(inertia):
        if getattr(inertia, "accepts_mass", False):
            return _inertia_matrix(inertia(t, r, v, q, omega, mass=mass))
        try:
            return _inertia_matrix(inertia(t, r, v, q, omega, mass=mass))
        except TypeError:
            return _inertia_matrix(inertia(t, r, v, q, omega))
    return _inertia_matrix(inertia)


def _mass_property_value(model, t, r, v, q, omega, mass, mass_rate):
    if not callable(model):
        return np.asarray(model, dtype=float)
    args = (t, r, v, q, omega)
    kwargs = {"mass": mass, "mass_rate": mass_rate}
    try:
        parameters = inspect.signature(model).parameters
    except (TypeError, ValueError):
        return model(*args, **kwargs)
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
        return model(*args, **kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in parameters}
    return model(*args, **accepted)


def _inertia_rate_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-15)):
        raise ValueError("inertia_rate must be a finite symmetric 3x3 matrix.")
    return matrix


def _inertia_directional_rate(
    model, t, r, v, q, omega, mass, r_dot, v_dot, q_dot, mass_rate
):
    relative_rate = max(
        1.0,
        np.linalg.norm(r_dot) / max(1.0, np.linalg.norm(r)),
        np.linalg.norm(v_dot) / max(1.0, np.linalg.norm(v)),
        np.linalg.norm(q_dot),
        0.0 if mass is None else abs(mass_rate) / max(1.0, abs(mass)),
    )
    step = np.cbrt(np.finfo(float).eps) / relative_rate
    after = max(t + step, np.nextafter(t, np.inf))
    before = min(t - step, np.nextafter(t, -np.inf))

    def at(epoch):
        delta = epoch - t
        return _inertia_at_state(
            model,
            epoch,
            r + delta * r_dot,
            v + delta * v_dot,
            normalize_quaternion(q + delta * q_dot),
            omega,
            None if mass is None else mass + delta * mass_rate,
        )

    return (at(after) - at(before)) / (after - before)


def _sum_model_accelerations(models, *, suppress_depleted=True):
    models = tuple(models)
    propulsive = any(_is_propulsive_model(model) for model in models)

    def acceleration(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        total = np.zeros(3)
        for model in models:
            if (suppress_depleted and _is_propulsive_model(model)
                    and _propellant_depleted(spacecraft, getattr(model, "tank_name", None))):
                continue
            if getattr(model, "spacecraft_acceleration_model", False):
                total = total + _as_vector3(
                    model(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega),
                    "acceleration",
                )
            elif callable(model) and not getattr(model, "spacecraft_torque_model", False):
                total = total + _as_vector3(model(t, r, v, q, omega), "acceleration")
        return total

    acceleration.spacecraft_acceleration_model = True
    acceleration.spacecraft_propulsive_model = propulsive
    acceleration._models = models
    acceleration.tank_name = _tank_name_for_models(*models)
    acceleration._depletion_handled = True
    return acceleration


def _sum_model_torques(models, *, suppress_depleted=True):
    models = tuple(models)
    propulsive = any(_is_propulsive_model(model) for model in models)

    def torque(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        total = np.zeros(3)
        for model in models:
            if (suppress_depleted and _is_propulsive_model(model)
                    and _propellant_depleted(spacecraft, getattr(model, "tank_name", None))):
                continue
            if getattr(model, "spacecraft_torque_model", False) and not getattr(model, "spacecraft_wheel_torque_model", False):
                evaluator = model.torque if hasattr(model, "torque") else model
                total = total + _as_vector3(
                    evaluator(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega),
                    "torque",
                )
        return total

    torque.spacecraft_torque_model = True
    torque.spacecraft_propulsive_model = propulsive
    torque.torque = torque
    torque._models = models
    torque._depletion_handled = True
    torque.tank_name = _tank_name_for_models(*models)
    return torque


def _sum_model_wheel_torques(models):
    models = tuple(models)

    def wheel_torques(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        body = getattr(spacecraft, "body", None)
        n_wheels = len(tuple(getattr(body, "reaction_wheels", ())))
        total = np.zeros(n_wheels)
        for model in models:
            if getattr(model, "spacecraft_wheel_torque_model", False):
                evaluator = model.wheel_torques if hasattr(model, "wheel_torques") else model
                commands = _as_vector(
                    evaluator(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega),
                    "wheel_torque",
                )
                if commands.shape != (n_wheels,):
                    raise ValueError("wheel_torque must return one scalar torque per body reaction wheel.")
                total = total + commands
        return total

    wheel_torques.spacecraft_wheel_torque_model = True
    wheel_torques.wheel_torques = wheel_torques
    return wheel_torques


def _sum_model_mass_flow_rates(models):
    models = tuple(models)

    def mass_flow_rate(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        total = 0.0
        for model in models:
            evaluator = getattr(model, "mass_flow_rate", None)
            if evaluator is not None:
                total += float(evaluator(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega))
        return total

    mass_flow_rate.spacecraft_mass_flow_model = True
    mass_flow_rate.tank_name = _tank_name_for_models(*models)
    return mass_flow_rate


def _tank_name_for_models(*models):
    names = set()
    for model in models:
        if model is None:
            continue
        name = getattr(model, "tank_name", None)
        if name is not None:
            names.add(name)
            continue
        nested = getattr(model, "_models", None)
        if nested is not None:
            names.update(
                name
                for name in (_tank_name_for_models(*nested),)
                if name is not None
            )
    if len(names) > 1:
        raise ValueError("a propagation may select at most one named propellant tank.")
    return next(iter(names), None)


def _allow_depleted_model(model, spacecraft, kind):
    source = getattr(model, "_source_model", model)
    models = getattr(source, "_models", None)
    if models is not None:
        source = {
            "acceleration": _sum_model_accelerations,
            "torque": _sum_model_torques,
        }[kind](models, suppress_depleted=False)
    binder = {
        "acceleration": _bind_spacecraft_acceleration,
        "torque": _bind_spacecraft_torque,
    }[kind]
    return binder(source, spacecraft, suppress_depleted=False)


def _any_torque_model(models) -> bool:
    return any(
        getattr(model, "spacecraft_torque_model", False)
        and not getattr(model, "spacecraft_wheel_torque_model", False)
        for model in models
    )


def _any_wheel_torque_model(models) -> bool:
    return any(getattr(model, "spacecraft_wheel_torque_model", False) for model in models)


def _any_mass_flow_model(models) -> bool:
    return any(hasattr(model, "mass_flow_rate") for model in models)


def _is_propulsive_model(model) -> bool:
    return bool(
        getattr(model, "spacecraft_propulsive_model", False)
        or callable(getattr(model, "mass_flow_rate", None))
    )


def _propellant_depleted(spacecraft, tank_name=None) -> bool:
    body = getattr(spacecraft, "body", None)
    mass = getattr(spacecraft, "mass", None)
    return bool(
        body is not None
        and getattr(body, "tanks", ())
        and mass is not None
        and float(mass) <= _propellant_mass_floor(body, tank_name)
    )


def _clamp_dry_mass(trajectory, dry_mass):
    if trajectory.mass is None or not np.any(trajectory.mass < dry_mass):
        return trajectory
    return replace(trajectory, mass=np.maximum(trajectory.mass, dry_mass))


def _direct_mass_flow_models(*models):
    return tuple(model for model in models if hasattr(model, "mass_flow_rate"))


def _model_call_state(args, kwargs):
    kwargs = dict(kwargs)
    spacecraft = kwargs.pop("spacecraft", None)
    if args:
        if len(args) != 5:
            raise TypeError("model calls require t, r, v, q, omega")
        t, r, v, q, omega = args
    else:
        t = kwargs.pop("t")
        r = kwargs.pop("r")
        v = kwargs.pop("v")
        q = kwargs.pop("q")
        omega = kwargs.pop("omega")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    return spacecraft, t, r, v, q, omega


def _inertia_matrix(inertia: ArrayLike) -> np.ndarray:
    inertia = np.asarray(inertia, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix.")
    if not np.allclose(inertia, inertia.T):
        raise ValueError("inertia must be symmetric.")
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("inertia must be positive definite.")
    return inertia
