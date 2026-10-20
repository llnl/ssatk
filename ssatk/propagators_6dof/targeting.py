"""Finite-burn targeting for 6-DoF trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..coordinates.attitude import quaternion_conjugate, quaternion_multiply
from ..coordinates.satellite_frames import frame_to_gcrf_matrix
from .high_accuracy import (
    _combine_trajectories,
    _propagate_spacecraft_segment,
    propagate_spacecraft_segments,
)
from .sixdof import SixDOFState, SixDOFTrajectory, Spacecraft, _body_at_state, _epochs_close

__all__ = [
    "SixDOFMultiSegmentTargetResult",
    "SixDOFMultipleShootingTargetResult",
    "SixDOFTargetResult",
    "solve_6dof_multi_segment_target",
    "solve_6dof_multiple_shooting_target",
    "solve_6dof_target",
]


@dataclass(frozen=True)
class SixDOFTargetResult:
    """Result of a finite-burn 6-DoF single-shooting solve."""

    control: np.ndarray
    residual: np.ndarray
    trajectory: SixDOFTrajectory
    success: bool
    message: str
    nfev: int
    cost: float


@dataclass(frozen=True)
class SixDOFMultiSegmentTargetResult:
    """Result of a bounded multi-segment 6-DoF single-shooting solve."""

    controls: np.ndarray
    residual: np.ndarray
    trajectory: SixDOFTrajectory
    success: bool
    message: str
    nfev: int
    cost: float


@dataclass(frozen=True)
class SixDOFMultipleShootingTargetResult:
    """Result of a bounded 6-DoF multiple-shooting solve.

    ``node_states`` are the pre-impulse start states of each segment. The
    first state is fixed by the input spacecraft; later states are independent
    optimization variables constrained to their preceding segment endpoints.
    ``residual`` is the complete normalized least-squares residual.
    """

    controls: np.ndarray
    node_states: tuple[SixDOFState, ...]
    segment_trajectories: tuple[SixDOFTrajectory, ...]
    trajectory: SixDOFTrajectory
    residual: np.ndarray
    success: bool
    message: str
    nfev: int
    cost: float


class _ControlAcceleration:
    spacecraft_acceleration_model = True

    def __init__(self, control, frame: str, start: float, stop: float):
        self.control = np.asarray(control, dtype=float)
        self.frame = frame
        self.start = float(start)
        self.stop = float(stop)

    def __call__(self, *, t, r, v, q, omega, spacecraft=None):
        if not self.start <= float(t) <= self.stop:
            return np.zeros(3)
        return frame_to_gcrf_matrix(self.frame, r=r, v=v, q=q) @ self.control


def solve_6dof_target(
    spacecraft,
    *,
    times,
    target_r=None,
    target_v=None,
    control0=(0.0, 0.0, 0.0),
    control_scale=(1.0e-4, 1.0e-4, 1.0e-4),
    frame: str = "inertial",
    start: float | None = None,
    stop: float | None = None,
    models: Iterable = (),
    position_scale: float = 1.0e3,
    velocity_scale: float = 1.0,
    bounds=None,
    max_nfev: int = 100,
    propagation_kwargs: Mapping[str, object] | None = None,
) -> SixDOFTargetResult:
    """Solve a finite-burn acceleration control for a terminal state target.

    ``control0`` and the returned ``control`` are three-component accelerations
    in m/s² expressed in ``frame``. The burn is active from ``start`` through
    ``stop`` and is combined with ``models`` during each shooting propagation.
    ``target_r`` and ``target_v`` are inertial Cartesian terminal targets; at
    least one must be supplied. ``bounds`` contains lower and upper acceleration
    vectors and is passed to SciPy's bounded least-squares solver.
    """

    times = _times(times)
    target_r = None if target_r is None else _vector3(target_r, "target_r")
    target_v = None if target_v is None else _vector3(target_v, "target_v")
    if target_r is None and target_v is None:
        raise ValueError("target_r or target_v is required.")
    position_scale = _positive(position_scale, "position_scale")
    velocity_scale = _positive(velocity_scale, "velocity_scale")
    control0 = _vector3(control0, "control0")
    control_scale = _vector3(control_scale, "control_scale")
    if np.any(control_scale <= 0.0):
        raise ValueError("control_scale values must be positive.")
    start = float(times[0] if start is None else start)
    stop = float(times[-1] if stop is None else stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive.")
    lower, upper = _bounds(bounds, control_scale)
    scaled_control0 = control0 / control_scale
    if np.any(scaled_control0 < lower) or np.any(scaled_control0 > upper):
        raise ValueError("control0 must lie within bounds.")

    base_models = tuple(models or ())
    propagation_options = dict(propagation_kwargs or {})

    def evaluate(scaled_control):
        control = np.asarray(scaled_control, dtype=float) * control_scale
        burn = _ControlAcceleration(control, frame, start, stop)
        trajectory = spacecraft.propagate(
            times=times,
            models=(*base_models, burn),
            **propagation_options,
        )
        residual = []
        if target_r is not None:
            residual.extend((trajectory.r[-1] - target_r) / position_scale)
        if target_v is not None:
            residual.extend((trajectory.v[-1] - target_v) / velocity_scale)
        return np.asarray(residual, dtype=float), trajectory

    def residual(scaled_control):
        return evaluate(scaled_control)[0]

    result = least_squares(
        residual,
        scaled_control0,
        bounds=(lower, upper),
        x_scale=1.0,
        max_nfev=int(max_nfev),
    )
    scaled_residual, trajectory = evaluate(result.x)
    physical_residual = []
    if target_r is not None:
        physical_residual.extend(trajectory.r[-1] - target_r)
    if target_v is not None:
        physical_residual.extend(trajectory.v[-1] - target_v)
    return SixDOFTargetResult(
        control=np.asarray(result.x, dtype=float) * control_scale,
        residual=np.asarray(physical_residual, dtype=float),
        trajectory=trajectory,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(0.5 * np.dot(scaled_residual, scaled_residual)),
    )


def solve_6dof_multi_segment_target(
    spacecraft,
    *,
    segments,
    target_r=None,
    target_v=None,
    control0=(0.0, 0.0, 0.0),
    control_scale=(1.0e-4, 1.0e-4, 1.0e-4),
    frame: str = "inertial",
    bounds=None,
    position_scale: float = 1.0e3,
    velocity_scale: float = 1.0,
    constraints=(),
    residual_hook=None,
    max_nfev: int = 100,
    propagation_kwargs: Mapping[str, object] | None = None,
) -> SixDOFMultiSegmentTargetResult:
    """Target a terminal state with bounded controls over several segments.

    Each segment is a mapping with ``times`` plus optional overrides for
    ``models``, ``control0``, ``control_scale``, ``frame``, ``bounds``,
    ``start``, and ``stop``. Remaining keys are passed to
    :func:`propagate_spacecraft_segments`. ``constraints`` may be one callable
    or an iterable of callables; each returns normalized residuals that are
    zero when feasible. ``residual_hook(trajectory, controls)`` can append
    normalized objective residuals.
    """

    target_r = None if target_r is None else _vector3(target_r, "target_r")
    target_v = None if target_v is None else _vector3(target_v, "target_v")
    if target_r is None and target_v is None:
        raise ValueError("target_r or target_v is required.")
    position_scale = _positive(position_scale, "position_scale")
    velocity_scale = _positive(velocity_scale, "velocity_scale")
    max_nfev = int(max_nfev)
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive.")
    specs = _segment_specs(segments, propagation_kwargs)
    controls0 = []
    scales = []
    lower = []
    upper = []
    for spec in specs:
        segment_control0 = _vector3(spec.pop("control0", control0), "control0")
        segment_scale = _vector3(spec.pop("control_scale", control_scale), "control_scale")
        if np.any(segment_scale <= 0.0):
            raise ValueError("control_scale values must be positive.")
        segment_bounds = spec.pop("bounds", bounds)
        segment_lower, segment_upper = _bounds(segment_bounds, segment_scale)
        scaled_control0 = segment_control0 / segment_scale
        if np.any(scaled_control0 < segment_lower) or np.any(scaled_control0 > segment_upper):
            raise ValueError("control0 must lie within bounds.")
        controls0.append(scaled_control0)
        scales.append(segment_scale)
        lower.append(segment_lower)
        upper.append(segment_upper)
    scaled_control0 = np.concatenate(controls0)
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)
    constraint_hooks = (constraints,) if callable(constraints) else tuple(constraints or ())

    def evaluate(scaled_control):
        scaled_control = np.asarray(scaled_control, dtype=float).reshape(len(specs), 3)
        segment_options = []
        for index, (spec, scale) in enumerate(zip(specs, scales)):
            options = dict(spec)
            segment_frame = options.pop("frame", frame)
            segment_start = options.pop("start", None)
            segment_stop = options.pop("stop", None)
            times = np.asarray(options["times"], dtype=float)
            burn = _ControlAcceleration(
                scaled_control[index] * scale,
                segment_frame,
                times[0] if segment_start is None else segment_start,
                times[-1] if segment_stop is None else segment_stop,
            )
            options["models"] = (*tuple(options.get("models") or ()), burn)
            segment_options.append(options)
        trajectory = propagate_spacecraft_segments(spacecraft, segment_options)
        residual = []
        if target_r is not None:
            residual.extend((trajectory.r[-1] - target_r) / position_scale)
        if target_v is not None:
            residual.extend((trajectory.v[-1] - target_v) / velocity_scale)
        for hook in constraint_hooks:
            residual.extend(_hook_residual(hook(trajectory), "constraint"))
        if residual_hook is not None:
            residual.extend(_hook_residual(residual_hook(trajectory, scaled_control * np.asarray(scales)), "residual_hook"))
        return np.asarray(residual, dtype=float), trajectory

    result = least_squares(
        lambda value: evaluate(value)[0],
        scaled_control0,
        bounds=(lower, upper),
        x_scale=1.0,
        max_nfev=max_nfev,
    )
    scaled_residual, trajectory = evaluate(result.x)
    physical_residual = []
    if target_r is not None:
        physical_residual.extend(trajectory.r[-1] - target_r)
    if target_v is not None:
        physical_residual.extend(trajectory.v[-1] - target_v)
    return SixDOFMultiSegmentTargetResult(
        controls=np.asarray(result.x, dtype=float).reshape(len(specs), 3)
        * np.asarray(scales),
        residual=np.asarray(physical_residual, dtype=float),
        trajectory=trajectory,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(0.5 * np.dot(scaled_residual, scaled_residual)),
    )


def solve_6dof_multiple_shooting_target(
    spacecraft,
    *,
    segments,
    target_r=None,
    target_v=None,
    control0=(0.0, 0.0, 0.0),
    control_scale=(1.0e-4, 1.0e-4, 1.0e-4),
    frame: str = "inertial",
    bounds=None,
    position_scale: float = 1.0e3,
    velocity_scale: float = 1.0,
    attitude_scale: float = 1.0,
    angular_rate_scale: float = 1.0,
    mass_scale: float = 1.0,
    wheel_momentum_scale: float = 1.0,
    terminal_residual=None,
    constraints=(),
    residual_hook=None,
    node_residual=None,
    max_nfev: int = 100,
    propagation_kwargs: Mapping[str, object] | None = None,
) -> SixDOFMultipleShootingTargetResult:
    """Target independent 6-DoF segment nodes with multiple shooting.

    Segment starts after the first are independent states, constrained to the
    previous segment's endpoint. Nodes are pre-impulse states, so an
    ``ImpulseManeuver`` on a segment remains an exact discontinuity. Callback
    residuals must already be normalized. ``terminal_residual`` receives the
    final :class:`SixDOFState`; ``node_residual`` receives node states, segment
    trajectories, and physical controls.
    """

    target_r = None if target_r is None else _vector3(target_r, "target_r")
    target_v = None if target_v is None else _vector3(target_v, "target_v")
    constraint_hooks = (constraints,) if callable(constraints) else tuple(constraints or ())
    if target_r is None and target_v is None and terminal_residual is None and not constraint_hooks and residual_hook is None and node_residual is None:
        raise ValueError("a terminal target or residual hook is required.")
    position_scale = _positive(position_scale, "position_scale")
    velocity_scale = _positive(velocity_scale, "velocity_scale")
    attitude_scale = _positive(attitude_scale, "attitude_scale")
    angular_rate_scale = _positive(angular_rate_scale, "angular_rate_scale")
    mass_scale = _positive(mass_scale, "mass_scale")
    wheel_momentum_scale = _positive(wheel_momentum_scale, "wheel_momentum_scale")
    max_nfev = int(max_nfev)
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive.")

    specs = _segment_specs(segments, propagation_kwargs)
    controls0 = []
    control_scales = []
    lower = []
    upper = []
    propagation_specs = []
    for spec in specs:
        options = dict(spec)
        segment_control0 = _vector3(options.pop("control0", control0), "control0")
        segment_scale = _vector3(options.pop("control_scale", control_scale), "control_scale")
        if np.any(segment_scale <= 0.0):
            raise ValueError("control_scale values must be positive.")
        segment_lower, segment_upper = _bounds(options.pop("bounds", bounds), segment_scale)
        scaled_control0 = segment_control0 / segment_scale
        if np.any(scaled_control0 < segment_lower) or np.any(scaled_control0 > segment_upper):
            raise ValueError("control0 must lie within bounds.")
        controls0.append(scaled_control0)
        control_scales.append(segment_scale)
        lower.append(segment_lower)
        upper.append(segment_upper)
        propagation_specs.append(options)
    control_scales = np.asarray(control_scales)

    seed_controls = np.asarray(controls0) * control_scales
    seed_nodes, seed_trajectories, _ = _multiple_shooting_arcs(
        spacecraft, propagation_specs, seed_controls, frame, tracks_mass=spacecraft.mass is not None
    )
    wheel_size = _wheel_state_size(seed_trajectories)
    has_mass = spacecraft.mass is not None
    node_size = 12 + int(has_mass) + wheel_size
    for node in seed_nodes[1:]:
        node_lower, node_upper = _node_bounds(
            node, has_mass, wheel_size, mass_scale, wheel_momentum_scale, spacecraft.body
        )
        lower.append(node_lower)
        upper.append(node_upper)

    initial = np.concatenate([np.concatenate(controls0), np.zeros(node_size * (len(specs) - 1))])
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)

    def evaluate(value):
        value = np.asarray(value, dtype=float)
        scaled_controls = value[: 3 * len(specs)].reshape(len(specs), 3)
        controls = scaled_controls * control_scales
        nodes = [seed_nodes[0]]
        offset = 3 * len(specs)
        for seed in seed_nodes[1:]:
            nodes.append(
                _decode_node(
                    seed, value[offset:offset + node_size], has_mass, wheel_size,
                    position_scale, velocity_scale, attitude_scale, angular_rate_scale,
                    mass_scale, wheel_momentum_scale,
                )
            )
            offset += node_size
        node_spacecraft = tuple(_spacecraft_from_state(spacecraft, state) for state in nodes)
        trajectories, preserves = _multiple_shooting_arcs_from_nodes(
            node_spacecraft, propagation_specs, controls, frame, tracks_mass=has_mass
        )
        trajectory = _combine_trajectories(trajectories, preserves)
        residual = []
        final_state = _trajectory_state(trajectories[-1])
        if target_r is not None:
            residual.extend((final_state.r - target_r) / position_scale)
        if target_v is not None:
            residual.extend((final_state.v - target_v) / velocity_scale)
        if terminal_residual is not None:
            residual.extend(_hook_residual(terminal_residual(final_state), "terminal_residual"))
        for index in range(len(trajectories) - 1):
            residual.extend(
                _continuity_residual(
                    _trajectory_state(trajectories[index]), nodes[index + 1],
                    position_scale, velocity_scale, attitude_scale, angular_rate_scale,
                    mass_scale, wheel_momentum_scale,
                )
            )
        for hook in constraint_hooks:
            residual.extend(_hook_residual(hook(trajectory), "constraint"))
        if residual_hook is not None:
            residual.extend(_hook_residual(residual_hook(trajectory, controls), "residual_hook"))
        if node_residual is not None:
            residual.extend(_hook_residual(node_residual(tuple(nodes), trajectories, controls), "node_residual"))
        return np.asarray(residual, dtype=float), tuple(nodes), tuple(trajectories), trajectory

    result = least_squares(
        lambda value: evaluate(value)[0],
        initial,
        bounds=(lower, upper),
        x_scale=1.0,
        max_nfev=max_nfev,
    )
    residual, nodes, trajectories, trajectory = evaluate(result.x)
    return SixDOFMultipleShootingTargetResult(
        controls=np.asarray(result.x[: 3 * len(specs)]).reshape(len(specs), 3) * control_scales,
        node_states=nodes,
        segment_trajectories=trajectories,
        trajectory=trajectory,
        residual=residual,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(0.5 * np.dot(residual, residual)),
    )


def _multiple_shooting_arcs(spacecraft, specs, controls, frame, *, tracks_mass):
    nodes = []
    trajectories = []
    preserve_boundaries = []
    current = spacecraft
    for spec, control in zip(specs, controls):
        nodes.append(current.state())
        options = _controlled_segment(spec, control, frame)
        if current.wheel_momentum is not None:
            options.setdefault("wheel_momentum0", current.wheel_momentum)
        trajectory, current, has_impulses = _propagate_spacecraft_segment(
            current, options, tracks_mass=tracks_mass
        )
        _require_complete_segment(trajectory, options["times"])
        trajectories.append(trajectory)
        preserve_boundaries.append(has_impulses)
    return tuple(nodes), tuple(trajectories), tuple(preserve_boundaries)


def _multiple_shooting_arcs_from_nodes(nodes, specs, controls, frame, *, tracks_mass):
    trajectories = []
    preserve_boundaries = []
    for node, spec, control in zip(nodes, specs, controls):
        options = _controlled_segment(spec, control, frame)
        if node.wheel_momentum is not None:
            options.setdefault("wheel_momentum0", node.wheel_momentum)
        trajectory, _, has_impulses = _propagate_spacecraft_segment(
            node, options, tracks_mass=tracks_mass
        )
        _require_complete_segment(trajectory, options["times"])
        trajectories.append(trajectory)
        preserve_boundaries.append(has_impulses)
    return tuple(trajectories), tuple(preserve_boundaries)


def _controlled_segment(spec, control, frame):
    options = dict(spec)
    segment_frame = options.pop("frame", frame)
    start = options.pop("start", None)
    stop = options.pop("stop", None)
    times = np.asarray(options["times"], dtype=float)
    start = float(times[0] if start is None else start)
    stop = float(times[-1] if stop is None else stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")
    options["models"] = (*tuple(options.get("models") or ()), _ControlAcceleration(control, segment_frame, start, stop))
    return options


def _require_complete_segment(trajectory, times):
    if trajectory.status != 0 or not _epochs_close(trajectory.t[-1], np.asarray(times, dtype=float)[-1]):
        raise RuntimeError("multiple shooting requires every segment to reach its final epoch.")


def _wheel_state_size(trajectories):
    sizes = {0 if trajectory.wheel_momentum is None else trajectory.wheel_momentum.shape[1] for trajectory in trajectories}
    if len(sizes) != 1:
        raise ValueError("multiple shooting requires the same wheel-state layout in every segment.")
    return sizes.pop()


def _node_bounds(node, has_mass, wheel_size, mass_scale, wheel_momentum_scale, body):
    size = 12 + int(has_mass) + wheel_size
    lower = np.full(size, -np.inf)
    upper = np.full(size, np.inf)
    index = 12
    if has_mass:
        floor = float(getattr(body, "dry_mass_total", 0.0))
        minimum = floor if floor > 0.0 else np.nextafter(0.0, np.inf)
        lower[index] = (minimum - float(node.mass)) / mass_scale
        maximum = getattr(body, "current_mass", None)
        if maximum is not None:
            upper[index] = (float(maximum) - float(node.mass)) / mass_scale
        index += 1
    if wheel_size:
        capacity = _wheel_capacity(body, wheel_size)
        momentum = np.asarray(node.wheel_momentum, dtype=float)
        lower[index:] = (-capacity - momentum) / wheel_momentum_scale
        upper[index:] = (capacity - momentum) / wheel_momentum_scale
    return lower, upper


def _wheel_capacity(body, size):
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if len(wheels) != size:
        return np.full(size, np.inf)
    return np.asarray([
        np.inf if wheel.momentum_capacity is None else float(wheel.momentum_capacity)
        for wheel in wheels
    ])


def _decode_node(
    seed, value, has_mass, wheel_size, position_scale, velocity_scale,
    attitude_scale, angular_rate_scale, mass_scale, wheel_momentum_scale,
):
    index = 0
    r = seed.r + value[index:index + 3] * position_scale
    index += 3
    v = seed.v + value[index:index + 3] * velocity_scale
    index += 3
    q = quaternion_multiply(seed.q, _quaternion_from_rotation_vector(value[index:index + 3] * attitude_scale))
    index += 3
    omega = seed.omega + value[index:index + 3] * angular_rate_scale
    index += 3
    mass = seed.mass
    if has_mass:
        mass = float(seed.mass) + float(value[index]) * mass_scale
        index += 1
    wheel_momentum = seed.wheel_momentum
    if wheel_size:
        wheel_momentum = np.asarray(seed.wheel_momentum) + value[index:index + wheel_size] * wheel_momentum_scale
    return SixDOFState(r=r, v=v, q=q, omega=omega, t=seed.t, mass=mass, wheel_momentum=wheel_momentum)


def _quaternion_from_rotation_vector(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate(([np.cos(0.5 * angle)], np.sin(0.5 * angle) * vector / angle))


def _spacecraft_from_state(template, state):
    body = _body_at_state(template.body, state.mass, state.wheel_momentum)
    inertia = getattr(
        body,
        "current_inertia",
        getattr(body, "inertia", template.inertia),
    )
    return Spacecraft(
        r=state.r, v=state.v, t=state.t, q=state.q, omega=state.omega,
        wheel_momentum=state.wheel_momentum,
        inertia=inertia,
        mass=state.mass,
        area=template.area, cd=template.cd, cr=template.cr,
        center_of_pressure=template.center_of_pressure, body=body, orbit=template.orbit,
    )


def _trajectory_state(trajectory):
    return SixDOFState(
        r=trajectory.r[-1], v=trajectory.v[-1], q=trajectory.q[-1],
        omega=trajectory.omega[-1], t=float(trajectory.t[-1]),
        mass=None if trajectory.mass is None else float(trajectory.mass[-1]),
        wheel_momentum=None if trajectory.wheel_momentum is None else trajectory.wheel_momentum[-1],
    )


def _continuity_residual(
    endpoint, node, position_scale, velocity_scale, attitude_scale,
    angular_rate_scale, mass_scale, wheel_momentum_scale,
):
    residual = [
        *(endpoint.r - node.r) / position_scale,
        *(endpoint.v - node.v) / velocity_scale,
        *_rotation_vector(quaternion_multiply(quaternion_conjugate(endpoint.q), node.q)) / attitude_scale,
        *(endpoint.omega - node.omega) / angular_rate_scale,
    ]
    if endpoint.mass is not None or node.mass is not None:
        if endpoint.mass is None or node.mass is None:
            raise ValueError("mass state layout differs between multiple-shooting segments.")
        residual.append((endpoint.mass - node.mass) / mass_scale)
    if endpoint.wheel_momentum is not None or node.wheel_momentum is not None:
        if endpoint.wheel_momentum is None or node.wheel_momentum is None:
            raise ValueError("wheel state layout differs between multiple-shooting segments.")
        residual.extend((endpoint.wheel_momentum - node.wheel_momentum) / wheel_momentum_scale)
    return np.asarray(residual, dtype=float)


def _rotation_vector(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[1:])
    if sine == 0.0:
        return np.zeros(3)
    return 2.0 * np.arctan2(sine, quaternion[0]) * quaternion[1:] / sine


def _segment_specs(segments, propagation_kwargs):
    specs = [dict(propagation_kwargs or {}, **dict(segment)) for segment in segments]
    if not specs:
        raise ValueError("segments must contain at least one segment.")
    for spec in specs:
        if "times" not in spec:
            raise ValueError("each segment must define times.")
        spec["times"] = _times(spec["times"])
    return specs


def _hook_residual(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = value.reshape(1)
    else:
        value = value.ravel()
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must return finite residuals.")
    return value


def _times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _vector3(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return value


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _bounds(bounds, control_scale):
    if bounds is None:
        return np.full(3, -np.inf), np.full(3, np.inf)
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair.")
    lower = _vector3(bounds[0], "lower bound") / control_scale
    upper = _vector3(bounds[1], "upper bound") / control_scale
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds.")
    return lower, upper
