"""High-accuracy 6-DoF propagation helpers."""

from dataclasses import dataclass

import numpy as np

from ..coordinates.satellite_frames import frame_to_gcrf_matrix
from .sixdof import (
    SixDOFTrajectory,
    Spacecraft,
    _epochs_close,
    _direct_mass_flow_models,
    _inertia_at_state,
    _supports_body_mass_update,
    _tank_name_for_models,
    _piecewise_solution_sequence,
)
from .sixdof import propagate_6dof as _propagate_6dof

__all__ = [
    "ImpulseManeuver",
    "propagate_6dof_high_accuracy",
    "propagate_spacecraft_high_accuracy",
    "propagate_spacecraft_segments",
]


@dataclass(frozen=True)
class ImpulseManeuver:
    """Instantaneous velocity change applied at a segment start epoch."""

    dv: tuple[float, float, float]
    frame: str = "inertial"
    mass_change: float | None = None
    q_reset: tuple[float, float, float, float] | None = None
    omega_reset: tuple[float, float, float] | None = None

    def apply(self, spacecraft: Spacecraft) -> Spacecraft:
        delta_v = np.asarray(self.dv, dtype=float)
        if delta_v.shape != (3,) or not np.all(np.isfinite(delta_v)):
            raise ValueError("dv must be a finite 3-vector.")
        velocity = spacecraft.v + frame_to_gcrf_matrix(
            self.frame, r=spacecraft.r, v=spacecraft.v, q=spacecraft.q
        ) @ delta_v
        mass = spacecraft.mass
        body = spacecraft.body
        if self.mass_change is not None:
            if not np.isfinite(self.mass_change):
                raise ValueError("mass_change must be finite.")
            if mass is None:
                raise ValueError("mass_change requires spacecraft mass.")
            mass = float(mass) + float(self.mass_change)
            if mass <= 0.0:
                raise ValueError("impulse would produce non-positive mass.")
            if body is not None and hasattr(body, "with_current_mass"):
                body = body.with_current_mass(mass)
        inertia = spacecraft.inertia
        if body is not spacecraft.body and hasattr(body, "current_inertia"):
            inertia = None
        return Spacecraft(
            r=spacecraft.r, v=velocity, t=spacecraft.t,
            q=spacecraft.q if self.q_reset is None else self.q_reset,
            omega=spacecraft.omega if self.omega_reset is None else self.omega_reset,
            wheel_momentum=spacecraft.wheel_momentum,
            inertia=inertia, mass=mass, area=spacecraft.area,
            cd=spacecraft.cd, cr=spacecraft.cr,
            center_of_pressure=spacecraft.center_of_pressure,
            body=body, orbit=spacecraft.orbit,
        )


def propagate_6dof_high_accuracy(**kwargs):
    """Call :func:`ssatk.propagators_6dof.propagate_6dof` with high-accuracy defaults."""

    kwargs.setdefault("method", "DOP853")
    kwargs.setdefault("rtol", 1e-10)
    kwargs.setdefault("atol", 1e-9)
    return _propagate_6dof(**kwargs)


def propagate_spacecraft_high_accuracy(
    spacecraft,
    *,
    times,
    models=(),
    environment=None,
    environment_models: bool | str | dict = False,
    ssapy_perturbations: bool | dict = False,
    gravity_gradient: bool = False,
    **kwargs,
):
    """Propagate a :class:`~ssatk.propagators_6dof.Spacecraft` with high-accuracy defaults.

    ``models`` may contain any SSATK acceleration/torque/mass-flow models. Pass
    a ``SpaceEnvironment`` and ``environment_models=True`` to add environment
    backed facet drag and solar-radiation pressure, or pass a preset string
    such as ``"leo"``, ``"earth_orbit"``, ``"cislunar"``, or ``"all"``. Set
    ``ssapy_perturbations=True`` to prepend SSAPy's mature translational
    perturbation stack. Pass dictionaries to customize either option.
    """

    from ..accelerations_6dof import make_ssapy_perturbation_acceleration
    from ..environment import SpaceEnvironment

    model_list = list(models or ())
    environment_flags = {}
    if environment_models:
        if environment is None:
            environment = SpaceEnvironment()
        environment_options = _environment_model_options(environment_models)
        environment_options.setdefault("body", getattr(spacecraft, "body", None))
        environment_model_list = environment.force_models(**environment_options)
        environment_flags = _environment_model_flags(environment_model_list)
        model_list[:0] = environment_model_list
    if ssapy_perturbations:
        options = {} if ssapy_perturbations is True else dict(ssapy_perturbations)
        options.setdefault("spacecraft_kwargs", _spacecraft_physical_kwargs(spacecraft))
        if environment_flags.get("solar_radiation"):
            options.setdefault("include_solar_radiation", False)
        if environment_flags.get("drag"):
            options.setdefault("include_drag", False)
        model_list.insert(0, make_ssapy_perturbation_acceleration(**options))
    kwargs.setdefault("method", "DOP853")
    kwargs.setdefault("rtol", 1e-10)
    kwargs.setdefault("atol", 1e-9)
    return spacecraft.propagate(
        times=times,
        models=model_list,
        gravity_gradient=gravity_gradient,
        **kwargs,
    )


def propagate_spacecraft_segments(spacecraft, segments, **defaults):
    """Propagate consecutive high-accuracy spacecraft segments.

    Each segment is a mapping with ``times`` plus any
    :func:`propagate_spacecraft_high_accuracy` keyword. Segment values override
    ``defaults``. The first time in each segment must equal the current
    spacecraft epoch so gaps are explicit. ``impulses`` optionally contains
    :class:`ImpulseManeuver` objects applied at that exact first epoch.
    """

    segments = [dict(defaults, **dict(segment)) for segment in segments]
    tracks_mass = spacecraft.mass is not None and any(
        "mass_flow_rate" in segment
        or any(getattr(impulse, "mass_change", None) is not None
               for impulse in ((segment.get("impulses"),)
                               if isinstance(segment.get("impulses"), ImpulseManeuver)
                               else (segment.get("impulses") or ())))
        or any(hasattr(model, "mass_flow_rate") for model in (segment.get("models") or ()))
        or bool(_direct_mass_flow_models(*(segment.get(name) for name in
                ("acceleration", "torque", "body_acceleration", "ntw_acceleration"))))
        for segment in segments
    )
    trajectories = []
    preserve_boundaries = []
    current = spacecraft
    for segment in segments:
        trajectory, current, has_impulses = _propagate_spacecraft_segment(
            current, segment, tracks_mass=tracks_mass
        )
        trajectories.append(trajectory)
        preserve_boundaries.append(has_impulses)
        if trajectory.status == 1:
            break
    if not trajectories:
        raise ValueError("segments must contain at least one segment.")
    return _combine_trajectories(trajectories, preserve_boundaries)


def _propagate_spacecraft_segment(spacecraft, segment, *, tracks_mass=False):
    """Propagate one segment from its pre-impulse spacecraft state."""

    options = dict(segment)
    if "times" not in options:
        raise ValueError("each segment must define times.")
    times = np.asarray(options["times"], dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("each segment times must be a 1-D array with at least two entries.")
    if not _epochs_close(times[0], spacecraft.t):
        raise ValueError("each segment must start at the current spacecraft epoch.")
    # Preserve the caller's array while making the shared boundary exact.
    times = times.copy()
    times[0] = spacecraft.t
    options["times"] = times
    impulses = options.pop("impulses", ())
    if isinstance(impulses, ImpulseManeuver):
        impulses = (impulses,)
    current = spacecraft
    for impulse in impulses:
        if not isinstance(impulse, ImpulseManeuver):
            raise TypeError("impulses must contain ImpulseManeuver objects.")
        current = impulse.apply(current)
    if tracks_mass:
        options.setdefault("mass0", current.mass)
    trajectory = propagate_spacecraft_high_accuracy(current, **options)
    final_inertia = (
        None if trajectory.mass is not None and _supports_body_mass_update(current.body)
        else current.inertia
    )
    if options.get("inertia") is not None:
        final_inertia = _inertia_at_state(
            options["inertia"], trajectory.t[-1], trajectory.r[-1], trajectory.v[-1],
            trajectory.q[-1], trajectory.omega[-1],
            current.mass if trajectory.mass is None else trajectory.mass[-1],
        )
    return (
        trajectory,
        trajectory.spacecraft(
            inertia=final_inertia,
            mass=current.mass if trajectory.mass is None else None,
            area=current.area,
            cd=current.cd,
            cr=current.cr,
            center_of_pressure=current.center_of_pressure,
            body=current.body,
            tank_name=_tank_name_for_models(
                *(options.get(name) for name in ("acceleration", "torque", "mass_flow_rate")),
                *(options.get("models") or ()),
            ),
        ),
        bool(impulses),
    )


def _spacecraft_physical_kwargs(spacecraft) -> dict:
    mapping = {"mass": "mass", "area": "area", "cd": "CD", "cr": "CR"}
    return {
        key: float(getattr(spacecraft, attr))
        for attr, key in mapping.items()
        if getattr(spacecraft, attr, None) is not None
    }


def _environment_model_options(environment_models) -> dict:
    if environment_models is True:
        return {"drag": True, "solar_radiation": True}
    if isinstance(environment_models, str):
        return {"preset": environment_models}
    return dict(environment_models)


def _environment_model_flags(models) -> dict[str, bool]:
    from ..accelerations_6dof import (
        SpacecraftAccelDrag,
        SpacecraftAccelSolRad,
        SpacecraftFacetDrag,
        SpacecraftFacetSolRad,
    )

    return {
        "drag": any(isinstance(model, (SpacecraftAccelDrag, SpacecraftFacetDrag)) for model in models),
        "solar_radiation": any(isinstance(model, (SpacecraftAccelSolRad, SpacecraftFacetSolRad)) for model in models),
    }


def _combine_trajectories(trajectories, preserve_boundaries=None) -> SixDOFTrajectory:
    preserve_boundaries = tuple(preserve_boundaries or ())
    slices = [slice(None)]
    slices.extend(
        slice(None) if index < len(preserve_boundaries) and preserve_boundaries[index] else slice(1, None)
        for index in range(1, len(trajectories))
    )
    t = np.concatenate([trajectory.t[index] for trajectory, index in zip(trajectories, slices)])
    r = np.vstack([trajectory.r[index] for trajectory, index in zip(trajectories, slices)])
    v = np.vstack([trajectory.v[index] for trajectory, index in zip(trajectories, slices)])
    q = np.vstack([trajectory.q[index] for trajectory, index in zip(trajectories, slices)])
    omega = np.vstack([trajectory.omega[index] for trajectory, index in zip(trajectories, slices)])
    mass = (
        np.concatenate([trajectory.mass[index] for trajectory, index in zip(trajectories, slices)])
        if all(trajectory.mass is not None for trajectory in trajectories)
        else None
    )
    wheel_momentum = (
        np.vstack([
            trajectory.wheel_momentum[index]
            for trajectory, index in zip(trajectories, slices)
        ])
        if all(trajectory.wheel_momentum is not None for trajectory in trajectories)
        else None
    )
    t_events, y_events, event_functions = _combine_event_results(trajectories)
    # Segments are contiguous, so each interior boundary is the next
    # trajectory's first epoch. Dropping this left dense output unavailable on
    # every segmented run, however each segment was configured.
    solution = _piecewise_solution_sequence(
        [trajectory.solution for trajectory in trajectories],
        [float(trajectory.t[0]) for trajectory in trajectories[1:]],
    )
    return SixDOFTrajectory(
        t=t,
        r=r,
        v=v,
        q=q,
        omega=omega,
        mass=mass,
        wheel_momentum=wheel_momentum,
        nfev=sum(trajectory.nfev for trajectory in trajectories),
        solution=solution,
        message="; ".join(trajectory.message for trajectory in trajectories if trajectory.message),
        status=trajectories[-1].status,
        t_events=t_events or None,
        y_events=y_events or None,
        event_functions=event_functions,
    )


def _combine_event_results(trajectories):
    """Merge event occurrences by callable identity across segments."""
    functions, times, states = [], [], []
    known = {}
    have_states = True
    for trajectory in trajectories:
        for local_index, event_times in enumerate(trajectory.t_events or ()):
            function = (
                None if trajectory.event_functions is None
                else trajectory.event_functions[local_index]
            )
            key = None if function is None else id(function)
            if key is None or key not in known:
                index = len(functions)
                functions.append(function)
                times.append([])
                states.append([])
                if key is not None:
                    known[key] = index
            else:
                index = known[key]
            if trajectory.y_events is None:
                event_states = [None] * len(event_times)
                have_states = False
            else:
                event_states = trajectory.y_events[local_index]
                if len(event_states) != len(event_times):
                    raise ValueError("event epochs and states must have matching lengths")
            for epoch, state in zip(event_times, event_states):
                if (state is not None and times[index] and epoch == times[index][-1]
                        and np.array_equal(state, states[index][-1])):
                    continue
                times[index].append(float(epoch))
                states[index].append(state)
    return (
        tuple(np.asarray(items, dtype=float) for items in times),
        tuple(np.asarray(items, dtype=float) for items in states) if have_states else None,
        tuple(functions) if functions else None,
    )
