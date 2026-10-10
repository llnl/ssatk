"""Optimal transfer search between two orbits for SSAPy.

Where :func:`transfer_ssapy` solves a fixed boundary-value problem (two
states, one time of flight -> the unique connecting two-burn transfer),
:func:`transfer_optimal` searches over the *free* variables of an
orbit-to-orbit transfer -- departure time along orbit 1, time of flight,
arrival phase along orbit 2 (inject/insertion modes), and motion sense -- to find
the transfer that minimizes total, departure, or arrival delta-v, or given a
delta-v budget, the fastest transfer that fits it.

The search uses a coarse porkchop grid of fast impulsive Lambert solves
(Keplerian boundary ephemerides), filters out infeasible candidates
(no zero-revolution solution, or a transfer conic whose perigee dips
below the Earth plus a safety margin), optionally polishes the winner
with a Nelder-Mead local search, and finally plans/propagates the chosen
transfer with :func:`transfer_ssapy` under the full force model -- including
single-burn inject/intercept geometries via ``arrival_burn=False``.

Set ``visualize=True`` for mission-designer curves (porkchop contour and
delta-v vs time-of-flight Pareto front) saved via
``ssatk.plots.figsave``.
"""

import warnings
from collections.abc import Mapping

import numpy as np

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator
from ssapy.compute import rv
from ssapy.constants import EARTH_MU, EARTH_RADIUS

from ssatk.orbital_mechanics.transfer_ssapy_function import transfer_ssapy, solve_lambert
from ssatk.orbital_mechanics._transfer_result import transfer_boundary_states, transfer_state
from ssatk.time_functions._gps import _to_gps_seconds


_OBJECTIVE_ALIASES = {
    "min_dv": "min_dv",
    "dv": "min_dv",
    "delta_v": "min_dv",
    "deltav": "min_dv",
    "minimum_delta_v": "min_dv",
    "min_time": "min_time",
    "time": "min_time",
    "tof": "min_time",
    "minimum_time": "min_time",
}

_DELTA_V_MODE_ALIASES = {
    "total": "total",
    "both": "total",
    "sum": "total",
    "all": "total",
    "two_burn": "total",
    "two_burns": "total",
    "first": "first",
    "departure": "first",
    "depart": "first",
    "dv1": "first",
    "first_burn": "first",
    "departure_burn": "first",
    "last": "last",
    "final": "last",
    "arrival": "last",
    "insertion": "last",
    "dv2": "last",
    "second": "last",
    "last_burn": "last",
    "arrival_burn": "last",
}

_DEPARTURE_MODE_ALIASES = {
    "optimize": "optimize",
    "optimal": "optimize",
    "search": "optimize",
    "anytime": "optimize",
    "any_time": "optimize",
    "whenever": "optimize",
    "leave_whenever": "optimize",
    "free": "optimize",
    "now": "now",
    "leave_now": "now",
    "fixed": "now",
    "fixed_departure": "now",
    "current": "now",
    "state": "now",
}

_STAGE_MODE_ALIASES = {
    "direct": "direct",
    "single": "direct",
    "single_stage": "direct",
    "one_stage": "direct",
    "one_leg": "direct",
    "immediate": "immediate",
    "immediate_stage": "immediate",
    "immediate_staged": "immediate",
    "staged_immediate": "immediate",
    "timed": "timed",
    "stage_timed": "timed",
    "staged": "timed",
    "multi_stage": "timed",
    "multistage": "timed",
    "appropriately_timed": "timed",
    "optimize": "timed",
    "optimized": "timed",
    "leave_whenever": "timed",
    "best": "best",
    "compare": "best",
    "auto": "best",
}

_STAGE_TIMING_ALIASES = {
    "immediate": "immediate",
    "immediate_stage": "immediate",
    "immediate_staged": "immediate",
    "staged_immediate": "immediate",
    "now": "immediate",
    "leave_now": "immediate",
    "fixed": "immediate",
    "fixed_departure": "immediate",
    "no_wait": "immediate",
    "timed": "timed",
    "stage_timed": "timed",
    "staged": "timed",
    "multi_stage": "timed",
    "multistage": "timed",
    "appropriately_timed": "timed",
    "optimized_timing": "timed",
    "optimize": "timed",
    "optimized": "timed",
    "leave_whenever": "timed",
    "wait": "timed",
    "phased": "timed",
}

_ARRIVAL_MODE_ALIASES = {
    "inject": "inject",
    "injection": "inject",
    "rendezvous": "rendezvous",
    "match": "rendezvous",
    "match_state": "rendezvous",
    "match_velocity": "rendezvous",
    "intercept": "intercept",
    "flyby": "intercept",
    "position": "intercept",
    "position_only": "intercept",
    "no_arrival_burn": "intercept",
    "insert": "insertion",
    "insertion": "insertion",
    "orbit_insert": "insertion",
    "orbit_insertion": "insertion",
    "target_orbit": "insertion",
    "free_phase": "insertion",
}

_ROUTE_MODE_ALIASES = {
    **_STAGE_MODE_ALIASES,
    "stage": "timed",
    "stages": "timed",
    "multi": "timed",
    "multi_leg": "timed",
    "multi_stop": "timed",
    "multi_stop_stage": "timed",
    "multi_stop_staged": "timed",
}

_MINIMIZE_ALIASES = {
    **_OBJECTIVE_ALIASES,
    "fuel": "min_dv",
    "propellant": "min_dv",
    "mass": "min_dv",
    "first_burn": "first",
    "departure_burn": "first",
    "depart_burn": "first",
    "last_burn": "last",
    "arrival_burn": "last",
    "insertion_burn": "last",
}

_BOUNDARY_STATE_ALIASES = {
    "initial": ("initial", "departure", "start", "from", "orbit1", "state1", "initial_state"),
    "target": ("target", "arrival", "final", "to", "orbit2", "state2", "target_state"),
}

_TRANSFER_KWARG_KEYS = {
    "propagate",
    "refine",
    "n_samples",
    "rk_step",
    "refine_tol",
    "max_refine_iter",
    "raise_on_budget",
    "prograde",
    "dv_budget",
}

_TRANSFER_PROBLEM_SCHEMA = "ssatk.transfer_problem.v1"


def _normalize_keyword(value, aliases, name):
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{name} must be one of: {valid}") from exc


def _mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping/dict")
    return dict(value)


def _take(mapping, *names, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _update_if_present(overrides, mapping, target, *names):
    value = _take(mapping, *names, default=None)
    if value is not None:
        overrides[target] = value


def _normalize_time_window(value, name):
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must be a two-value time window")
    return (_to_gps_seconds(value[0]), _to_gps_seconds(value[1]))


def _apply_arrival_mode(overrides, value):
    mode = _normalize_keyword(value, _ARRIVAL_MODE_ALIASES, "arrival_mode")
    if mode == "rendezvous":
        overrides.update(rendezvous=True, arrival_burn=True)
    elif mode == "intercept":
        overrides.update(rendezvous=True, arrival_burn=False)
    elif mode == "inject":
        overrides.update(rendezvous=False, arrival_burn=False)
    elif mode == "insertion":
        overrides.update(rendezvous=False, arrival_burn=True)
    overrides["arrival_mode"] = mode


def _parse_objective_section(section):
    overrides = {}
    section = _mapping(section, "objective")
    minimize = _take(section, "minimize", "objective", "mode", "cost", "goal")
    if minimize is not None:
        normalized = _normalize_keyword(minimize, _MINIMIZE_ALIASES, "objective.minimize")
        if normalized in ("first", "last"):
            overrides["objective"] = "min_dv"
            overrides["delta_v_mode"] = normalized
        else:
            overrides["objective"] = normalized
    _update_if_present(overrides, section, "delta_v_mode", "delta_v_mode", "dv_mode", "burn", "burn_mode")
    _update_if_present(overrides, section, "dv_budget", "dv_budget", "delta_v_budget", "budget")
    return overrides


def _parse_boundary_section(section):
    overrides = {}
    section = _mapping(section, "boundary")
    for target, aliases in _BOUNDARY_STATE_ALIASES.items():
        value = _take(section, *aliases)
        if value is not None:
            overrides[target] = value
    for key in ("r1", "v1", "r2", "v2", "t1", "t2"):
        _update_if_present(overrides, section, key, key)
    _update_if_present(overrides, section, "departure_mode", "departure_mode", "leave", "depart")
    _update_if_present(overrides, section, "leave_now", "leave_now")
    _update_if_present(overrides, section, "t_window", "departure_window", "departure_time_window", "t_window", "time_window")
    _update_if_present(overrides, section, "tof_range", "tof_range", "time_of_flight_range", "tof_window")
    _update_if_present(overrides, section, "arrival_window", "arrival_window", "arrival_time_window")
    arrival_mode = _take(section, "arrival_mode", "mode")
    if arrival_mode is not None:
        _apply_arrival_mode(overrides, arrival_mode)
    _update_if_present(overrides, section, "rendezvous", "rendezvous")
    _update_if_present(overrides, section, "arrival_burn", "arrival_burn")
    return overrides


def _parse_constraints_section(section):
    overrides = {}
    section = _mapping(section, "constraints")
    _update_if_present(overrides, section, "tof_range", "tof_range", "time_of_flight_range", "tof_window")
    _update_if_present(overrides, section, "t_window", "departure_window", "departure_time_window", "t_window")
    _update_if_present(overrides, section, "arrival_window", "arrival_window", "arrival_time_window")
    _update_if_present(overrides, section, "dv_budget", "dv_budget", "delta_v_budget", "budget")
    _update_if_present(overrides, section, "max_burns", "max_burns", "burn_limit")
    if "perigee_min" in section:
        overrides["perigee_margin"] = float(section["perigee_min"]) - EARTH_RADIUS
    for key in ("perigee_radius_min", "min_perigee_radius"):
        if key in section:
            overrides["perigee_margin"] = float(section[key]) - EARTH_RADIUS
    _update_if_present(overrides, section, "perigee_margin", "perigee_margin")
    _update_if_present(overrides, section, "perigee_margin", "perigee_altitude_min", "min_perigee_altitude")
    for key in ("burn_duration", "burn_accel", "thrust", "mass", "isp", "accel", "propagator"):
        _update_if_present(overrides, section, key, key)
    return overrides


def _parse_route_section(section):
    overrides = {}
    if section is None:
        return overrides
    if not isinstance(section, Mapping):
        overrides["stage_mode"] = _normalize_keyword(section, _ROUTE_MODE_ALIASES, "route")
        return overrides
    section = dict(section)
    mode = _take(section, "mode", "route", "type")
    if mode is not None:
        overrides["stage_mode"] = _normalize_keyword(mode, _ROUTE_MODE_ALIASES, "route.mode")
    timing = _take(section, "timing", "stage_timing", "wait", "wait_mode")
    if timing is not None:
        overrides["stage_timing"] = _normalize_keyword(timing, _STAGE_TIMING_ALIASES, "route.timing")
    _update_if_present(overrides, section, "n_stage_stops", "n_stage_stops", "stage_stops", "stops", "n_stops")
    _update_if_present(overrides, section, "stage_beam_width", "stage_beam_width", "beam_width")
    _update_if_present(overrides, section, "stage_wait_window", "stage_wait_window", "wait_window")
    _update_if_present(overrides, section, "stage_tof_range", "stage_tof_range", "leg_tof_range", "leg_time_of_flight_range")
    stage_candidates = _mapping(_take(section, "stage_candidates", "candidates", default={}), "route.stage_candidates")
    _update_if_present(overrides, stage_candidates, "stage_radii", "radii", "stage_radii")
    _update_if_present(overrides, stage_candidates, "stage_plane_fractions", "plane_fractions", "stage_plane_fractions")
    _update_if_present(overrides, stage_candidates, "n_stage_phase", "phase_count", "n_phase", "n_stage_phase")
    _update_if_present(overrides, section, "stage_radii", "stage_radii")
    _update_if_present(overrides, section, "stage_plane_fractions", "stage_plane_fractions")
    _update_if_present(overrides, section, "n_stage_phase", "phase_count", "n_stage_phase")
    return overrides


def _parse_solver_section(section):
    overrides = {}
    transfer_kwargs = {}
    section = _mapping(section, "solver")
    for key in ("n_grid", "n_phase", "polish", "visualize", "fig_prefix", "burn_duration", "burn_accel", "thrust", "mass", "isp"):
        _update_if_present(overrides, section, key, key)
    if "samples" in section and "n_samples" not in section:
        transfer_kwargs["n_samples"] = section["samples"]
    for key in _TRANSFER_KWARG_KEYS:
        if key in section:
            transfer_kwargs[key] = section[key]
    if transfer_kwargs:
        overrides["transfer_kwargs"] = transfer_kwargs
    return overrides


def _structured_problem_overrides(*, problem=None, boundary=None, objective=None, constraints=None, route=None, solver=None):
    problem_map = _mapping(problem, "problem") if problem is not None else {}
    overrides = {}
    used = bool(problem_map or boundary is not None or isinstance(objective, Mapping) or constraints is not None or route is not None or solver is not None)

    boundary_section = {}
    boundary_section.update(_mapping(problem_map.get("boundary"), "problem.boundary") if problem_map.get("boundary") is not None else {})
    for key in ("initial", "departure", "start", "from", "orbit1", "target", "arrival", "final", "to", "orbit2", "r1", "v1", "r2", "v2", "t1", "t2", "departure_mode", "arrival_mode", "tof_range", "t_window", "departure_window", "arrival_window"):
        if key in problem_map:
            boundary_section[key] = problem_map[key]
    boundary_section.update(_mapping(boundary, "boundary"))
    overrides.update(_parse_boundary_section(boundary_section))

    objective_section = {}
    if problem_map.get("objective") is not None:
        objective_section.update(_mapping(problem_map.get("objective"), "problem.objective"))
    elif any(key in problem_map for key in ("minimize", "cost", "goal")):
        for key in ("minimize", "cost", "goal", "delta_v_mode", "dv_budget"):
            if key in problem_map:
                objective_section[key] = problem_map[key]
    if isinstance(objective, Mapping):
        objective_section.update(objective)
    overrides.update(_parse_objective_section(objective_section))

    constraints_section = {}
    if problem_map.get("constraints") is not None:
        constraints_section.update(_mapping(problem_map.get("constraints"), "problem.constraints"))
    constraints_section.update(_mapping(constraints, "constraints"))
    overrides.update(_parse_constraints_section(constraints_section))

    route_section = problem_map.get("route", None)
    if route is not None:
        if isinstance(route_section, Mapping) and isinstance(route, Mapping):
            merged_route = dict(route_section)
            merged_route.update(route)
            route_section = merged_route
        else:
            route_section = route
    overrides.update(_parse_route_section(route_section))

    solver_section = {}
    if problem_map.get("solver") is not None:
        solver_section.update(_mapping(problem_map.get("solver"), "problem.solver"))
    solver_section.update(_mapping(solver, "solver"))
    solver_overrides = _parse_solver_section(solver_section)
    transfer_kwargs = dict(overrides.pop("transfer_kwargs", {}))
    transfer_kwargs.update(solver_overrides.pop("transfer_kwargs", {}))
    overrides.update(solver_overrides)
    if transfer_kwargs:
        overrides["transfer_kwargs"] = transfer_kwargs
    return overrides, used


def _set_state_override(current, overrides, name):
    if name not in overrides:
        return current
    value = overrides.pop(name)
    if current is not None:
        raise TypeError(f"Specify {name} either directly or through boundary/problem, not both")
    return value


def _structured_diagnostics(*, arrival_mode, objective, delta_v_mode, stage_mode,
                            stage_timing, departure_mode, t_window, tof_range,
                            arrival_window, dv_budget, perigee_margin, max_burns):
    diagnostics = {
        "problem_schema": _TRANSFER_PROBLEM_SCHEMA,
        "arrival_mode": arrival_mode,
        "route_mode": stage_mode,
        "objective_spec": {
            "objective": objective,
            "delta_v_mode": delta_v_mode,
        },
        "boundary_spec": {
            "departure_mode": departure_mode,
            "departure_window": t_window,
            "time_of_flight_range": tof_range,
            "arrival_window": arrival_window,
        },
        "constraint_spec": {
            "delta_v_budget": dv_budget,
            "perigee_margin": perigee_margin,
            "max_burns": max_burns,
        },
    }
    if stage_timing is not None:
        diagnostics["route_timing"] = stage_timing
    return diagnostics


def _arrival_mode_from_flags(rendezvous, arrival_burn):
    if rendezvous and arrival_burn:
        return "rendezvous"
    if rendezvous and not arrival_burn:
        return "intercept"
    if not rendezvous and arrival_burn:
        return "insertion"
    return "inject"


def _apply_structured_diagnostics(result, diagnostics):
    result["diagnostics"] = dict(result.get("diagnostics", {}))
    result["diagnostics"].update(diagnostics)
    return result


def _validate_max_burns(max_burns, *, stage_mode, n_stage_stops, arrival_burn):
    if max_burns is None:
        return None
    max_burns = int(max_burns)
    final_leg_burns = 2 if arrival_burn else 1
    if max_burns < final_leg_burns:
        raise ValueError(
            "max_burns is too small for the requested arrival mode "
            f"(minimum {final_leg_burns})"
        )
    if stage_mode != "direct":
        required = 2 * int(n_stage_stops) + final_leg_burns
        if required > max_burns and stage_mode != "best":
            raise ValueError(
                f"stage_mode='{stage_mode}' with n_stage_stops={n_stage_stops} "
                f"requires up to {required} burns, exceeding max_burns={max_burns}"
            )
    return max_burns


def _delta_v_metric(delta_v1, delta_v2, mode, arrival_burn=True):
    if mode == "first":
        return float(delta_v1)
    if mode == "last":
        if not arrival_burn:
            raise ValueError("delta_v_mode='last' requires arrival_burn=True")
        return float(delta_v2)
    return float(delta_v1 + (delta_v2 if arrival_burn else 0.0))


def _result_delta_v_metric(result, mode, arrival_burn=True):
    magnitudes = list(result.get("delta_v_magnitudes", []))
    delta_v1 = magnitudes[0] if magnitudes else 0.0
    delta_v2 = magnitudes[1] if len(magnitudes) > 1 else 0.0
    return _delta_v_metric(delta_v1, delta_v2, mode, arrival_burn=arrival_burn)


def _as_orbit(s, mu):
    if isinstance(s, Orbit):
        return Orbit(np.asarray(s.r, float).ravel(),
                     np.asarray(s.v, float).ravel(),
                     _to_gps_seconds(s.t), mu=mu)
    state = transfer_state(state=s, mu=mu)
    return Orbit(state["r"], state["v"], state["t"], mu=mu)


def _period(orbit, mu):
    if orbit.a <= 0:
        raise ValueError(
            "transfer_optimal's default search windows require closed "
            "(elliptical) boundary orbits; supply explicit t_window and "
            "tof_range for hyperbolic states.")
    return 2 * np.pi * np.sqrt(orbit.a ** 3 / mu)


def _ephemeris(orbit, times):
    """Keplerian states of ``orbit`` at ``times`` (input order kept)."""
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    rr, vv = rv(orbit, times[order], propagator=KeplerianPropagator())
    rr, vv = np.atleast_2d(rr), np.atleast_2d(vv)
    out_r = np.empty_like(rr)
    out_v = np.empty_like(vv)
    out_r[order] = rr
    out_v[order] = vv
    return out_r, out_v


def _conic_perigee(r, v, mu):
    """Perigee radius of the conic through state (r, v)."""
    h = np.linalg.norm(np.cross(r, v))
    energy = 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)
    e = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / mu ** 2))
    return h * h / (mu * (1.0 + e))


def _unit(vector, fallback=None):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        if fallback is None:
            raise ValueError("Cannot normalize zero vector")
        return _unit(fallback)
    return vector / norm


def _slerp_unit(start, stop, fraction):
    start = _unit(start, fallback=[0.0, 0.0, 1.0])
    stop = _unit(stop, fallback=start)
    dot = float(np.clip(np.dot(start, stop), -1.0, 1.0))
    if dot < 0.0:
        stop = -stop
        dot = -dot
    if dot > 0.9995:
        return _unit((1.0 - fraction) * start + fraction * stop, fallback=start)
    angle = np.arccos(dot)
    return _unit(
        np.sin((1.0 - fraction) * angle) * start / np.sin(angle)
        + np.sin(fraction * angle) * stop / np.sin(angle),
        fallback=start,
    )


def _basis_from_normal(normal, reference=None):
    normal = _unit(normal, fallback=[0.0, 0.0, 1.0])
    if reference is None:
        reference = np.array([1.0, 0.0, 0.0])
    reference = np.asarray(reference, dtype=float).reshape(3)
    radial = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(radial) <= 1e-12:
        radial = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(radial) <= 1e-12:
        radial = np.cross(normal, [0.0, 1.0, 0.0])
    radial = _unit(radial)
    tangent = _unit(np.cross(normal, radial))
    return radial, tangent


def _circular_state_in_plane(radius, normal, phase, *, t=0.0, mu=EARTH_MU, reference=None):
    radial0, tangent0 = _basis_from_normal(normal, reference=reference)
    radial = np.cos(phase) * radial0 + np.sin(phase) * tangent0
    tangent = -np.sin(phase) * radial0 + np.cos(phase) * tangent0
    return {
        "r": float(radius) * radial,
        "v": np.sqrt(mu / float(radius)) * tangent,
        "t": float(t),
        "label": "staging_orbit",
    }


def _default_stage_radii(initial_state, target_state):
    r0 = np.linalg.norm(initial_state["r"])
    rf = np.linalg.norm(target_state["r"])
    high = max(r0, rf)
    radii = np.array([
        np.sqrt(r0 * rf),
        0.5 * (r0 + rf),
        1.5 * high,
        2.5 * high,
    ])
    radii = radii[np.isfinite(radii) & (radii > 0.0)]
    return sorted({float(np.round(radius, 6)) for radius in radii})


def _stage_candidates(initial_state, target_state, *, stage_radii=None, stage_plane_fractions=None, n_stage_phase=8, mu=EARTH_MU):
    if stage_radii is None:
        stage_radii = _default_stage_radii(initial_state, target_state)
    if stage_plane_fractions is None:
        stage_plane_fractions = (0.0, 0.5, 1.0)
    n_stage_phase = int(n_stage_phase)
    if n_stage_phase < 1:
        raise ValueError("n_stage_phase must be at least 1")

    h0 = np.cross(initial_state["r"], initial_state["v"])
    hf = np.cross(target_state["r"], target_state["v"])
    reference = initial_state["r"]
    for radius in stage_radii:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("stage_radii must be positive")
        for fraction in stage_plane_fractions:
            normal = _slerp_unit(h0, hf, float(fraction))
            for phase in np.linspace(0.0, 2.0 * np.pi, n_stage_phase, endpoint=False):
                yield _circular_state_in_plane(radius, normal, phase, t=initial_state["t"], mu=mu, reference=reference), {
                    "radius": radius,
                    "plane_fraction": float(fraction),
                    "phase": float(phase),
                }


def _stage_info_key(stage_info):
    return (
        round(float(stage_info.get("radius", 0.0)), 6),
        round(float(stage_info.get("plane_fraction", 0.0)), 12),
        round(float(stage_info.get("phase", 0.0)), 12),
    )


def _stage_departure_window(state, stage_timing, stage_wait_window, mu):
    if stage_timing == "immediate":
        return "now", None
    stage_orbit = _as_orbit(state, mu)
    wait = _period(stage_orbit, mu) if stage_wait_window is None else float(stage_wait_window)
    if wait < 0.0:
        raise ValueError("stage_wait_window must be non-negative")
    return "optimize", (float(state["t"]), float(state["t"]) + wait)


def _state_after_leg(leg, *, label="staging_departure"):
    stage_arrive = float(leg["diagnostics"].get("t_arrive", leg["final"]["t"]))
    return {
        "r": leg["final"]["r"],
        "v": leg["final"]["v"],
        "t": stage_arrive,
        "label": label,
    }


def _stage_objective_key(result, objective, dv_budget):
    objective_delta_v = float(result["diagnostics"]["objective_delta_v"])
    elapsed = float(result["tof"])
    if objective == "min_time":
        if dv_budget is None:
            raise ValueError("objective='min_time' requires dv_budget")
        if objective_delta_v > dv_budget:
            return None
        return (elapsed, objective_delta_v)
    return (objective_delta_v, elapsed)


def _stage_metric(legs, mode, arrival_burn=True):
    first = legs[0]["delta_v_magnitudes"][0] if legs and legs[0].get("delta_v_magnitudes") else 0.0
    last_mags = legs[-1].get("delta_v_magnitudes", []) if legs else []
    last = last_mags[-1] if last_mags else 0.0
    total = sum(float(leg.get("delta_v_total", 0.0)) for leg in legs)
    if mode == "first":
        return float(first)
    if mode == "last":
        if not arrival_burn:
            raise ValueError("delta_v_mode='last' requires arrival_burn=True")
        return float(last)
    return float(total)


def _concat_trajectories(legs):
    trajectories = [leg.get("trajectory") for leg in legs]
    if not trajectories or any(traj is None or traj.get("r") is None for traj in trajectories):
        return None
    out = {"frame": trajectories[0].get("frame", "gcrf")}
    for key in ("t", "r", "v"):
        arrays = [traj.get(key) for traj in trajectories]
        if any(array is None for array in arrays):
            out[key] = None
            continue
        stitched = []
        for index, array in enumerate(arrays):
            array = np.asarray(array)
            stitched.append(array[1:] if index and array.shape[0] else array)
        out[key] = np.concatenate(stitched, axis=0)
    return out


def _combine_stage_legs(legs, *, stage_mode, stage_timing, objective, delta_v_mode, arrival_burn, stage_info, direct_result=None):
    burns = [burn for leg in legs for burn in leg.get("burns", [])]
    delta_v_total = float(sum(burn["delta_v_mag"] for burn in burns))
    delta_v_magnitudes = [burn["delta_v_mag"] for burn in burns]
    transfer_orbits = [orbit for leg in legs for orbit in leg.get("transfer_orbits", [])]
    first_depart = legs[0]["diagnostics"].get("t_depart", legs[0]["initial"]["t"])
    final_arrive = legs[-1]["diagnostics"].get("t_arrive", legs[-1]["final"]["t"])
    assumptions = []
    for leg in legs:
        assumptions.extend(leg.get("assumptions", []))
    assumptions.extend(["explicit multi-stage transfer search", f"stage timing: {stage_timing}"])
    combined = dict(legs[-1])
    combined.update(
        method="transfer_optimal_staged",
        initial=legs[0]["initial"],
        target=legs[-1]["target"],
        final=legs[-1]["final"],
        tof=float(final_arrive - first_depart),
        burns=burns,
        delta_v_total=delta_v_total,
        delta_v_vectors=[burn["delta_v"] for burn in burns],
        delta_v_ntw_vectors=[burn["delta_v_ntw"] for burn in burns],
        delta_v_magnitudes=delta_v_magnitudes,
        trajectory=_concat_trajectories(legs),
        transfer_orbits=transfer_orbits,
        success=all(leg.get("success", True) for leg in legs),
        assumptions=list(dict.fromkeys(assumptions)),
    )
    combined["diagnostics"] = dict(combined.get("diagnostics", {}))
    combined["diagnostics"].update(
        objective=objective,
        delta_v_mode=delta_v_mode,
        stage_mode=stage_mode,
        stage_timing=stage_timing,
        stage_count=len(legs),
        leg_count=len(legs),
        stage_stop_count=max(0, len(legs) - 1),
        objective_delta_v=_stage_metric(legs, delta_v_mode, arrival_burn=arrival_burn),
        t_depart=float(first_depart),
        t_arrive=float(final_arrive),
        stage=stage_info,
        legs=[
            {
                "method": leg.get("method"),
                "delta_v_total": leg.get("delta_v_total"),
                "objective_delta_v": leg.get("diagnostics", {}).get("objective_delta_v"),
                "t_depart": leg.get("diagnostics", {}).get("t_depart"),
                "t_arrive": leg.get("diagnostics", {}).get("t_arrive"),
                "tof": leg.get("tof"),
            }
            for leg in legs
        ],
    )
    if direct_result is not None:
        combined["diagnostics"]["direct_delta_v_total"] = direct_result.get("delta_v_total")
        combined["diagnostics"]["direct_objective_delta_v"] = direct_result.get("diagnostics", {}).get("objective_delta_v")
    combined["stage_legs"] = legs
    return combined


def transfer_optimal(
    *args,
    orbit1=None,
    orbit2=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=0.0,
    t2=None,
    problem=None,
    boundary=None,
    objective="min_dv",
    constraints=None,
    route=None,
    solver=None,
    delta_v_mode="total",
    departure_mode="optimize",
    leave_now=None,
    stage_mode="direct",
    stage_radii=None,
    stage_plane_fractions=None,
    n_stage_phase=8,
    n_stage_stops=1,
    stage_beam_width=8,
    stage_timing=None,
    stage_wait_window=None,
    stage_tof_range=None,
    arrival_mode=None,
    rendezvous=True,
    arrival_burn=True,
    t_window=None,
    tof_range=None,
    arrival_window=None,
    n_grid=(32, 32),
    n_phase=24,
    dv_budget=None,
    perigee_margin=100e3,
    max_burns=None,
    polish=True,
    visualize=False,
    fig_prefix="figures/transfer_optimal",
    accel=None,
    propagator=None,
    burn_duration=10.0,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    **transfer_kwargs,
):
    """Find the optimal two-burn (or intercept) transfer between orbits.

    Parameters
    ----------
    initial, target, orbit1, orbit2 : ssapy.orbit.Orbit or tuple
        Departure and target orbits/states.  Epochs may be GPS seconds or
        ``astropy.time.Time``.
    r1, v1, r2, v2 : array_like, optional
        Raw state-vector form.  Equivalent positional form
        ``transfer_optimal(r1, v1, r2, v2, ...)`` is accepted.  These vectors
        define the osculating departure and target orbits; the optimizer still
        searches departure time, time of flight, and optionally target phase.
    problem, boundary, objective, constraints, route, solver : mapping, optional
        Structured transfer-problem interface.  ``problem`` may contain
        ``boundary`` (initial/target states, departure/arrival modes),
        ``objective`` (``minimize`` and burn cost), ``constraints`` (time,
        delta-v, perigee, burn-count, and engine limits), ``route`` (direct,
        immediate staged, timed/multi-stage, or best), and ``solver`` (grid,
        polish, propagation/refinement options).  Separate section keywords
        override matching sections inside ``problem``.
    objective : {"min_dv", "min_time"}
        ``min_dv`` (default) minimizes the objective delta-v within the
        allowed windows.  ``min_time`` minimizes time of flight among
        candidates whose delta-v fits ``dv_budget`` (required).
        Aliases such as ``"delta_v"`` and ``"time"`` are accepted.
    delta_v_mode : {"total", "first", "last"}
        Which burn cost the optimizer minimizes or budgets. ``"total"``
        (aliases ``"both"``/``"sum"``) uses the departure burn plus the
        arrival burn when ``arrival_burn=True``. ``"first"`` (aliases
        ``"departure"``/``"dv1"``) optimizes the departure burn only.
        ``"last"`` (aliases ``"arrival"``/``"dv2"``) optimizes the final
        matching burn only and requires ``arrival_burn=True``.
    departure_mode : {"optimize", "now"}
        ``"optimize"`` (default, aliases ``"whenever"``/``"anytime"``)
        searches departure phase/time along the initial orbit. ``"now"``
        (aliases ``"leave_now"``/``"fixed"``) fixes the departure state to
        the supplied epoch/state.  ``leave_now=True`` is a convenience alias.
    stage_mode : {"direct", "immediate", "timed", "best"}
        ``"direct"`` keeps the original one-leg optimizer. ``"immediate"``
        explicitly searches two-leg staged transfers and starts the second leg
        immediately at the staging orbit. ``"timed"`` lets the second leg wait
        for an optimized departure phase/time from the staging orbit.
        ``"best"`` compares the direct and timed-staged answers.
    stage_radii, stage_plane_fractions, n_stage_phase : optional
        Staging-orbit search grid.  By default, circular staging orbits span
        intermediate/high radii and planes interpolated between the initial and
        target angular-momentum vectors.
    n_stage_stops, stage_beam_width : int, optional
        Number of intermediate staging orbits to search and beam width retained
        between staged legs.  The default ``n_stage_stops=1`` searches a two-leg
        staged transfer; larger values opt into explicit multi-stop staging.
    stage_wait_window, stage_tof_range : optional
        Wait-time search span after reaching the staging orbit and per-leg
        time-of-flight range.  Defaults to one staging-orbit period and
        ``tof_range``, respectively.
    arrival_mode : {"inject", "intercept", "rendezvous", "insertion"}, optional
        User-facing arrival constraint. ``"inject"`` searches a departure burn
        onto a transfer that crosses the target orbit without matching velocity.
        ``"intercept"`` reaches the target object's position at the selected
        arrival time without matching velocity. ``"rendezvous"`` reaches that
        position and matches velocity. ``"insertion"`` matches target-orbit
        velocity with the target phase free, i.e. an orbit-insertion search.
    rendezvous : bool
        Lower-level phase flag used when ``arrival_mode`` is not set. If True
        (default), the arrival state is wherever the *object on orbit 2* is at
        ``t_depart + tof``. If False, the arrival point anywhere along orbit 2
        is a free search variable.
    arrival_burn : bool
        If True (default), the second burn matching the arrival velocity
        is performed and counted.  If False, optimize the *first burn
        only* (intercept/injection/flyby): the spacecraft coasts through the
        target point without matching its velocity.
    t_window : (float, float), optional
        Allowed departure epoch span [GPS s].  Default: one revolution
        of orbit 1 from its epoch.
    tof_range : (float, float), optional
        Allowed time-of-flight span [s].  Default: 2% to 150% of the
        larger orbital period.
    arrival_window : (float, float), optional
        Optional allowed final-arrival epoch span [GPS s].  Candidates outside
        this window are rejected.
    n_grid : (int, int)
        Porkchop grid resolution (departure x time-of-flight).
    n_phase : int
        Arrival-phase samples along orbit 2 for free-phase ``"inject"`` and
        ``"insertion"`` modes.
    dv_budget : float, optional
        Delta-v constraint [m/s]; required for ``objective='min_time'``,
        recorded/warned for ``min_dv`` (via transfer_ssapy).
    perigee_margin : float
        Candidates whose transfer conic dips below
        ``EARTH_RADIUS + perigee_margin`` are rejected [m].
    max_burns : int, optional
        Upper bound on the planned burn count. Direct rendezvous/insertion uses
        two burns, inject/intercept uses one burn, and each explicit staging stop
        adds up to two burns.
    polish : bool
        Refine the best grid cell with a Nelder-Mead local search over
        the continuous variables.
    visualize : bool
        Save mission-designer curves (porkchop + delta-v/TOF Pareto
        front) via ``ssatk.plots.figsave`` under ``fig_prefix``.
    fig_prefix : str
        figsave path prefix for the visualization.
    burn_accel : float, optional
        Burn acceleration magnitude [m/s^2]: the simple alternative to
        ``thrust``/``mass`` (mutually exclusive with them) when the
        thrust-to-mass analysis was done elsewhere.  Sizes burns and
        drives the same feasibility filtering; no propellant estimates.
    thrust, mass, isp : float, optional
        Engine model, passed through to :func:`transfer_ssapy` (thrust
        [N] and mass [kg] together size each burn's duration; isp [s]
        adds propellant estimates).  When given, the porkchop search
        also rejects candidates whose hardware-sized burns would not
        fit inside the time of flight -- so ``min_time`` answers are
        engine-honest, not just budget-honest.
    accel, propagator, burn_duration, **transfer_kwargs
        Passed through to :func:`transfer_ssapy` for the final
        propagated plan (the search itself uses impulsive Keplerian
        Lambert costs; the finishing differential correction absorbs
        finite-burn and force-model differences).
    Returns
    -------
    dict
        Canonical transfer dictionary. ``diagnostics`` includes the search
        objective, arrival mode, chosen arrival phase,
        perigee altitude, delta-v budget, porkchop grid, and Pareto curves.

    Notes
    -----
    * The search is zero-revolution Lambert per leg; long windows still
      explore multi-revolution *phasing* implicitly through the
      departure-time axis, but each transfer arc itself spans < 1 rev.
    * Boundary ephemerides during the search are Keplerian even when a
      perturbed ``accel`` is supplied; over windows of a few days the
      resulting epoch error is absorbed by the final refinement, but for
      strongly perturbed, multi-week windows treat the porkchop as
      approximate.
    * Both motion senses are searched automatically when the two orbits
      counter-rotate; co-rotating geometries search prograde only.
    """
    if orbit1 is not None:
        if initial is not None:
            raise TypeError("Specify either orbit1 or initial, not both")
        initial = orbit1
    if orbit2 is not None:
        if target is not None:
            raise TypeError("Specify either orbit2 or target, not both")
        target = orbit2


    structured_overrides, structured_used = _structured_problem_overrides(
        problem=problem,
        boundary=boundary,
        objective=objective if isinstance(objective, Mapping) else None,
        constraints=constraints,
        route=route,
        solver=solver,
    )
    initial = _set_state_override(initial, structured_overrides, "initial")
    target = _set_state_override(target, structured_overrides, "target")
    r1 = _set_state_override(r1, structured_overrides, "r1")
    v1 = _set_state_override(v1, structured_overrides, "v1")
    r2 = _set_state_override(r2, structured_overrides, "r2")
    v2 = _set_state_override(v2, structured_overrides, "v2")
    t1 = structured_overrides.pop("t1", t1)
    t2 = structured_overrides.pop("t2", t2)
    objective = structured_overrides.pop("objective", objective)
    delta_v_mode = structured_overrides.pop("delta_v_mode", delta_v_mode)
    departure_mode = structured_overrides.pop("departure_mode", departure_mode)
    leave_now = structured_overrides.pop("leave_now", leave_now)
    stage_mode = structured_overrides.pop("stage_mode", stage_mode)
    stage_radii = structured_overrides.pop("stage_radii", stage_radii)
    stage_plane_fractions = structured_overrides.pop("stage_plane_fractions", stage_plane_fractions)
    n_stage_phase = structured_overrides.pop("n_stage_phase", n_stage_phase)
    n_stage_stops = structured_overrides.pop("n_stage_stops", n_stage_stops)
    stage_beam_width = structured_overrides.pop("stage_beam_width", stage_beam_width)
    stage_timing = structured_overrides.pop("stage_timing", stage_timing)
    stage_wait_window = structured_overrides.pop("stage_wait_window", stage_wait_window)
    stage_tof_range = structured_overrides.pop("stage_tof_range", stage_tof_range)
    rendezvous = structured_overrides.pop("rendezvous", rendezvous)
    arrival_burn = structured_overrides.pop("arrival_burn", arrival_burn)
    arrival_mode = structured_overrides.pop("arrival_mode", arrival_mode)
    t_window = structured_overrides.pop("t_window", t_window)
    tof_range = structured_overrides.pop("tof_range", tof_range)
    arrival_window = structured_overrides.pop("arrival_window", arrival_window)
    n_grid = structured_overrides.pop("n_grid", n_grid)
    n_phase = structured_overrides.pop("n_phase", n_phase)
    dv_budget = structured_overrides.pop("dv_budget", dv_budget)
    perigee_margin = structured_overrides.pop("perigee_margin", perigee_margin)
    max_burns = structured_overrides.pop("max_burns", max_burns)
    polish = structured_overrides.pop("polish", polish)
    visualize = structured_overrides.pop("visualize", visualize)
    fig_prefix = structured_overrides.pop("fig_prefix", fig_prefix)
    accel = structured_overrides.pop("accel", accel)
    propagator = structured_overrides.pop("propagator", propagator)
    burn_duration = structured_overrides.pop("burn_duration", burn_duration)
    burn_accel = structured_overrides.pop("burn_accel", burn_accel)
    thrust = structured_overrides.pop("thrust", thrust)
    mass = structured_overrides.pop("mass", mass)
    isp = structured_overrides.pop("isp", isp)
    structured_transfer_kwargs = structured_overrides.pop("transfer_kwargs", {})
    if structured_overrides:
        raise RuntimeError(f"Unhandled structured transfer fields: {sorted(structured_overrides)}")
    merged_transfer_kwargs = dict(structured_transfer_kwargs)
    merged_transfer_kwargs.update(transfer_kwargs)
    transfer_kwargs = merged_transfer_kwargs
    if isinstance(objective, Mapping):
        objective = "min_dv"

    objective = _normalize_keyword(objective, _OBJECTIVE_ALIASES, "objective")
    delta_v_mode = _normalize_keyword(delta_v_mode, _DELTA_V_MODE_ALIASES, "delta_v_mode")
    if leave_now is not None:
        departure_mode = "now" if leave_now else "optimize"
    departure_mode = _normalize_keyword(departure_mode, _DEPARTURE_MODE_ALIASES, "departure_mode")
    stage_mode = _normalize_keyword(stage_mode, _STAGE_MODE_ALIASES, "stage_mode")
    if arrival_mode is not None:
        mode_overrides = {}
        _apply_arrival_mode(mode_overrides, arrival_mode)
        rendezvous = mode_overrides["rendezvous"]
        arrival_burn = mode_overrides["arrival_burn"]
        arrival_mode = mode_overrides["arrival_mode"]
    max_burns = _validate_max_burns(
        max_burns,
        stage_mode=stage_mode,
        n_stage_stops=n_stage_stops,
        arrival_burn=arrival_burn,
    )
    if arrival_mode is None:
        arrival_mode = _arrival_mode_from_flags(rendezvous, arrival_burn)
    if delta_v_mode == "last" and not arrival_burn:
        raise ValueError("delta_v_mode='last' requires arrival_burn=True")

    structured_diag = None
    if structured_used:
        structured_diag = _structured_diagnostics(
            arrival_mode=arrival_mode,
            objective=objective,
            delta_v_mode=delta_v_mode,
            stage_mode=stage_mode,
            stage_timing=stage_timing,
            departure_mode=departure_mode,
            t_window=t_window,
            tof_range=tof_range,
            arrival_window=arrival_window,
            dv_budget=dv_budget,
            perigee_margin=perigee_margin,
            max_burns=max_burns,
        )

    if stage_mode != "direct":
        result = _transfer_optimal_staged(
            *args,
            orbit1=orbit1,
            orbit2=orbit2,
            initial=initial,
            target=target,
            r1=r1,
            v1=v1,
            r2=r2,
            v2=v2,
            t1=t1,
            t2=t2,
            objective=objective,
            delta_v_mode=delta_v_mode,
            departure_mode=departure_mode,
            stage_mode=stage_mode,
            stage_radii=stage_radii,
            stage_plane_fractions=stage_plane_fractions,
            n_stage_phase=n_stage_phase,
            n_stage_stops=n_stage_stops,
            stage_beam_width=stage_beam_width,
            stage_timing=stage_timing,
            stage_wait_window=stage_wait_window,
            stage_tof_range=stage_tof_range,
            rendezvous=rendezvous,
            arrival_burn=arrival_burn,
            t_window=t_window,
            tof_range=tof_range,
            arrival_window=arrival_window,
            n_grid=n_grid,
            n_phase=n_phase,
            dv_budget=dv_budget,
            perigee_margin=perigee_margin,
            max_burns=max_burns,
            polish=polish,
            visualize=visualize,
            fig_prefix=fig_prefix,
            accel=accel,
            propagator=propagator,
            burn_duration=burn_duration,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
            **transfer_kwargs,
        )
        if structured_diag is not None:
            _apply_structured_diagnostics(result, structured_diag)
        return result

    mu = EARTH_MU
    departure_state, arrival_state = transfer_boundary_states(
        *args,
        initial=initial,
        target=target,
        r1=r1,
        v1=v1,
        r2=r2,
        v2=v2,
        t1=t1,
        t2=t2,
        mu=mu,
        name="transfer_optimal",
    )
    o1 = _as_orbit(departure_state, mu)
    o2 = _as_orbit(arrival_state, mu)
    p1, p2 = _period(o1, mu), _period(o2, mu)

    if objective == "min_time" and dv_budget is None:
        raise ValueError("objective='min_time' requires dv_budget")
    if (thrust is None) != (mass is None):
        raise ValueError("thrust and mass must be supplied together.")
    if burn_accel is not None and thrust is not None:
        raise ValueError(
            "Specify either burn_accel or thrust+mass, not both.")
    a_burn = (thrust / mass) if thrust is not None else burn_accel

    t0 = float(o1.t)
    if departure_mode == "now":
        if t_window is not None:
            warnings.warn("departure_mode='now' fixes departure at the supplied state; ignoring t_window.")
        t_window = (t0, t0)
    elif t_window is None:
        t_window = (t0, t0 + p1)
    t_window = (_to_gps_seconds(t_window[0]), _to_gps_seconds(t_window[1]))
    if tof_range is None:
        tof_range = (0.02 * max(p1, p2), 1.5 * max(p1, p2))
    arrival_window = _normalize_time_window(arrival_window, "arrival_window")

    n_dep, n_tof = n_grid
    if departure_mode == "now":
        n_dep = 1
        t_deps = np.array([t0], dtype=float)
    else:
        t_deps = np.linspace(*t_window, n_dep)
    tofs = np.linspace(*tof_range, n_tof)

    # Both senses only if the orbits counter-rotate.
    h1 = np.cross(np.ravel(o1.r), np.ravel(o1.v))
    h2 = np.cross(np.ravel(o2.r), np.ravel(o2.v))
    senses = (True,) if np.dot(h1, h2) >= 0 else (True, False)

    # --- boundary ephemerides (vectorized Keplerian) -------------------
    dep_r, dep_v = _ephemeris(o1, t_deps)
    if rendezvous:
        t_arr_grid = t_deps[:, None] + tofs[None, :]
        arr_r_flat, arr_v_flat = _ephemeris(o2, t_arr_grid.ravel())
        arr_r = arr_r_flat.reshape(n_dep, n_tof, 3)
        arr_v = arr_v_flat.reshape(n_dep, n_tof, 3)
        phases = None
    else:
        phases = np.linspace(0.0, p2, n_phase, endpoint=False)
        ring_r, ring_v = _ephemeris(o2, float(o2.t) + phases)

    r_min = EARTH_RADIUS + perigee_margin

    def candidate_cost(r1, v1, r2, v2, tof):
        """(cost, prograde) for the cheapest feasible sense, else NaN."""
        best, best_sense = (np.nan, np.nan, np.nan), True
        for sense in senses:
            try:
                v1l, v2l = solve_lambert(r1, r2, tof, mu=mu,
                                         prograde=sense, max_iter=60,
                                         tol=1e-6)
            except RuntimeError:
                continue
            if _conic_perigee(r1, v1l, mu) < r_min:
                continue
            dv1 = np.linalg.norm(v1l - v1)
            dv2_actual = np.linalg.norm(v2 - v2l)
            dv2 = dv2_actual if arrival_burn else 0.0
            # Burn-fit filter with headroom: transfer_ssapy enforces
            # burns <= a third of the TOF on the *refined* delta-v,
            # which exceeds this impulsive estimate by the finite-burn
            # steering losses; 25% here reserves that growth.
            if a_burn is not None and (dv1 + dv2) / a_burn >= 0.25 * tof:
                continue                 # burns don't fit this window
            c = _delta_v_metric(dv1, dv2_actual, delta_v_mode, arrival_burn=arrival_burn)
            if not (c >= best[0]):       # also catches best == NaN
                best, best_sense = (c, dv1, dv2), sense
        return best, best_sense

    def arrival_in_window(t_depart, tof_seconds):
        if arrival_window is None:
            return True
        t_arrive = float(t_depart) + float(tof_seconds)
        return arrival_window[0] <= t_arrive <= arrival_window[1]

    # --- porkchop grid ---------------------------------------------------
    if rendezvous:
        cost = np.full((n_dep, n_tof), np.nan)
        dv1g = np.full((n_dep, n_tof), np.nan)
        dv2g = np.full((n_dep, n_tof), np.nan)
        sense_grid = np.ones((n_dep, n_tof), dtype=bool)
        for i in range(n_dep):
            for j in range(n_tof):
                if not arrival_in_window(t_deps[i], tofs[j]):
                    continue
                (cost[i, j], dv1g[i, j], dv2g[i, j]), sense_grid[i, j] = \
                    candidate_cost(dep_r[i], dep_v[i],
                                   arr_r[i, j], arr_v[i, j], tofs[j])
        cost3 = cost[:, :, None]
        dv1g3 = dv1g[:, :, None]
        dv2g3 = dv2g[:, :, None]
    else:
        cost3 = np.full((n_dep, n_tof, n_phase), np.nan)
        dv1g3 = np.full((n_dep, n_tof, n_phase), np.nan)
        dv2g3 = np.full((n_dep, n_tof, n_phase), np.nan)
        sense3 = np.ones((n_dep, n_tof, n_phase), dtype=bool)
        for i in range(n_dep):
            for j in range(n_tof):
                if not arrival_in_window(t_deps[i], tofs[j]):
                    continue
                for k in range(n_phase):
                    (cost3[i, j, k], dv1g3[i, j, k], dv2g3[i, j, k]), \
                        sense3[i, j, k] = candidate_cost(
                            dep_r[i], dep_v[i], ring_r[k], ring_v[k],
                            tofs[j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cost = (np.nanmin(cost3, axis=2)
                    if not np.all(np.isnan(cost3))
                    else np.full((n_dep, n_tof), np.nan))

    feasible_fraction = float(np.mean(np.isfinite(cost3)))
    if not np.any(np.isfinite(cost3)):
        raise RuntimeError(
            "No feasible transfer found on the search grid: every "
            "candidate either lacked a zero-revolution Lambert solution, "
            "dipped its transfer conic below the perigee margin"
            + (", fell outside arrival_window" if arrival_window is not None else "")
            + (", or could not fit the hardware-sized burns (accel "
               f"{a_burn:.4f} m/s^2) into a third of the time of flight"
               if a_burn is not None else "")
            + ". Widen t_window/tof_range, reduce perigee_margin"
            + (", or use a stronger engine / lighter spacecraft"
               if a_burn is not None else "") + ".")

    # For min_time, the budget constrains the impulsive search estimate,
    # but the hardware-sized finite burns add steering losses on top.
    # Retry with a shrunken effective budget until the *refined* plan
    # actually fits (the porkchop grid is reused, so retries are cheap).
    budget_eff = dv_budget
    for _attempt in range(3):
        # --- objective selection on the grid --------------------------------
        if objective == "min_dv":
            flat = np.nanargmin(cost3)
        else:
            ok = np.where(np.isfinite(cost3) & (cost3 <= budget_eff))
            if len(ok[0]) == 0:
                raise ValueError(
                    f"No transfer on the grid fits the {budget_eff:.1f} m/s "
                    f"effective budget (cheapest impulsive candidate: "
                    f"{np.nanmin(cost3):.1f} m/s; requested budget "
                    f"{dv_budget:.1f} m/s with finite-burn losses "
                    "reserved). Increase dv_budget or widen the "
                    "windows.")
            jmin = np.argmin(ok[1])          # smallest tof index
            flat = np.ravel_multi_index(tuple(idx[jmin] for idx in ok),
                                        cost3.shape)
        idx = np.unravel_index(flat, cost3.shape)
        i0, j0 = idx[0], idx[1]
        k0 = idx[2] if not rendezvous else None

        # --- continuous polish (Nelder-Mead) ---------------------------------
        def eval_point(t_dep, tof, phase=None):
            if not arrival_in_window(t_dep, tof):
                return np.nan, True, (None, None, None, None)
            (r1,), (v1,) = _ephemeris(o1, [t_dep])
            if rendezvous:
                (r2,), (v2,) = _ephemeris(o2, [t_dep + tof])
            else:
                (r2,), (v2,) = _ephemeris(o2, [float(o2.t) + phase])
            (c, _, _), sense = candidate_cost(r1, v1, r2, v2, tof)
            return c, sense, (r1, v1, r2, v2)

        x_best = [t_deps[i0], tofs[j0]] + ([] if rendezvous else [phases[k0]])
        if polish:
            from scipy.optimize import minimize

            lo = [t_window[0], tof_range[0]] + ([] if rendezvous else [-np.inf])
            hi = [t_window[1], tof_range[1]] + ([] if rendezvous else [np.inf])

            def penalty(x):
                x = np.clip(x, lo, hi)
                c, _, _ = eval_point(*x)
                if not np.isfinite(c):
                    return 1e12
                if objective == "min_dv":
                    return c
                return x[1] + (0.0 if c <= budget_eff else 1e9 + c)

            res = minimize(penalty, x_best, method="Nelder-Mead",
                           options=dict(maxfev=200, xatol=1.0, fatol=1e-3))
            if np.isfinite(res.fun) and res.fun < 1e9:
                x_best = list(np.clip(res.x, lo, hi))

        t_dep, tof = float(x_best[0]), float(x_best[1])
        phase = float(x_best[2]) % p2 if not rendezvous else None
        c, sense, (r1, v1, r2, v2) = eval_point(*x_best)

        # --- final propagated, refined plan under the full force model ------
        transfer = transfer_ssapy(
            (r1, v1, t_dep), (r2, v2, t_dep + tof),
            accel=accel, propagator=propagator, burn_duration=burn_duration,
            burn_accel=burn_accel, thrust=thrust, mass=mass, isp=isp,
            prograde=sense, arrival_mode=arrival_mode, arrival_burn=arrival_burn,
            dv_budget=(dv_budget if objective == "min_dv" and delta_v_mode == "total" else None),
            **transfer_kwargs)
        objective_delta_v = _result_delta_v_metric(transfer, delta_v_mode, arrival_burn=arrival_burn)
        if objective == "min_dv" or objective_delta_v <= dv_budget:
            break
        budget_eff = budget_eff * dv_budget / objective_delta_v
    if (objective == "min_time" and dv_budget is not None
            and objective_delta_v > dv_budget):
        warnings.warn(
            f"min_time plan requires {objective_delta_v:.1f} m/s in "
            f"delta_v_mode='{delta_v_mode}', "
            f"exceeding the {dv_budget:.1f} m/s budget even after "
            "reserving finite-burn losses; treat this budget as "
            "infeasible for this geometry/engine.")
    if objective == "min_dv" and dv_budget is not None and objective_delta_v > dv_budget:
        warnings.warn(
            f"Transfer requires {objective_delta_v:.1f} m/s in "
            f"delta_v_mode='{delta_v_mode}', exceeding the {dv_budget:.1f} m/s budget.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c_flat = np.moveaxis(cost3, 1, 0).reshape(n_tof, -1)
        d1_flat = np.moveaxis(dv1g3, 1, 0).reshape(n_tof, -1)
        d2_flat = np.moveaxis(dv2g3, 1, 0).reshape(n_tof, -1)
        pareto_dv = np.full(n_tof, np.nan)
        pareto_dv1 = np.full(n_tof, np.nan)
        pareto_dv2 = np.full(n_tof, np.nan)
        for _j in range(n_tof):
            if np.any(np.isfinite(c_flat[_j])):
                _k = np.nanargmin(c_flat[_j])
                pareto_dv[_j] = c_flat[_j, _k]
                pareto_dv1[_j] = d1_flat[_j, _k]
                pareto_dv2[_j] = d2_flat[_j, _k]
    transfer_orbit = transfer["transfer_orbits"][0]
    perigee_altitude = _conic_perigee(
        np.ravel(transfer_orbit.r),
        np.ravel(transfer_orbit.v),
        mu,
    ) - EARTH_RADIUS
    standard = dict(transfer)
    standard["method"] = "transfer_optimal"
    standard["assumptions"] = list(transfer.get("assumptions", []))
    standard["assumptions"].extend([
        "coarse Lambert porkchop search",
        "optional Nelder-Mead polishing",
    ])
    standard["diagnostics"] = dict(transfer.get("diagnostics", {}))
    standard["diagnostics"].update(
        objective=objective,
        delta_v_mode=delta_v_mode,
        departure_mode=departure_mode,
        arrival_mode=arrival_mode,
        objective_delta_v=objective_delta_v,
        rendezvous=rendezvous,
        arrival_burn=arrival_burn,
        arrival_phase=phase,
        prograde=bool(sense),
        t_depart=t_dep,
        t_arrive=t_dep + tof,
        arrival_window=arrival_window,
        perigee_altitude=perigee_altitude,
        dv_budget=dv_budget,
        within_delta_v_budget=(None if dv_budget is None else bool(objective_delta_v <= dv_budget)),
        grid=dict(t_dep=t_deps, tof=tofs, cost=cost,
                  delta_v_mode=delta_v_mode,
                  feasible_fraction=feasible_fraction),
        pareto=dict(tof=tofs, dv=pareto_dv,
                    dv1=pareto_dv1, dv2=pareto_dv2),
    )
    if structured_diag is not None:
        _apply_structured_diagnostics(standard, structured_diag)

    if visualize:
        from ssatk.plots.transfer_designer_curves_plot import (
            transfer_designer_curves_plot)
        transfer_designer_curves_plot(
            standard, save_path=f"{fig_prefix}_designer_curves.jpg")
    return standard


def _transfer_optimal_staged(
    *args,
    orbit1=None,
    orbit2=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=0.0,
    t2=None,
    objective="min_dv",
    delta_v_mode="total",
    departure_mode="optimize",
    stage_mode="timed",
    stage_radii=None,
    stage_plane_fractions=None,
    n_stage_phase=8,
    n_stage_stops=1,
    stage_beam_width=8,
    stage_timing=None,
    stage_wait_window=None,
    stage_tof_range=None,
    rendezvous=True,
    arrival_burn=True,
    t_window=None,
    tof_range=None,
    arrival_window=None,
    n_grid=(32, 32),
    n_phase=24,
    dv_budget=None,
    perigee_margin=100e3,
    max_burns=None,
    polish=True,
    visualize=False,
    fig_prefix="figures/transfer_optimal",
    accel=None,
    propagator=None,
    burn_duration=10.0,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    **transfer_kwargs,
):
    """Search explicit staged transfers through one or more staging orbits."""
    if orbit1 is not None:
        if initial is not None and initial is not orbit1:
            raise TypeError("Specify either orbit1 or initial, not both")
        initial = orbit1
        orbit1 = None
    if orbit2 is not None:
        if target is not None and target is not orbit2:
            raise TypeError("Specify either orbit2 or target, not both")
        target = orbit2
        orbit2 = None

    if stage_timing is None:
        stage_timing = "immediate" if stage_mode == "immediate" else "timed"
    stage_timing = _normalize_keyword(stage_timing, _STAGE_TIMING_ALIASES, "stage_timing")

    n_stage_stops = int(n_stage_stops)
    stage_beam_width = int(stage_beam_width)
    if n_stage_stops < 1:
        raise ValueError("n_stage_stops must be at least 1 for staged transfers")
    if stage_beam_width < 1:
        raise ValueError("stage_beam_width must be at least 1")

    if visualize:
        warnings.warn("visualize=True is currently ignored for staged transfer searches; plot the returned staged result with orbit_plot.")

    mu = EARTH_MU
    initial_state, target_state = transfer_boundary_states(
        *args,
        orbit1=orbit1,
        orbit2=orbit2,
        initial=initial,
        target=target,
        r1=r1,
        v1=v1,
        r2=r2,
        v2=v2,
        t1=t1,
        t2=t2,
        mu=mu,
        name="transfer_optimal",
    )

    direct_result = None
    if stage_mode == "best":
        direct_result = transfer_optimal(
            initial_state,
            target_state,
            objective=objective,
            delta_v_mode=delta_v_mode,
            departure_mode=departure_mode,
            rendezvous=rendezvous,
            arrival_burn=arrival_burn,
            t_window=t_window,
            tof_range=tof_range,
            arrival_window=arrival_window,
            n_grid=n_grid,
            n_phase=n_phase,
            dv_budget=dv_budget,
            perigee_margin=perigee_margin,
            max_burns=max_burns,
            polish=polish,
            visualize=False,
            fig_prefix=fig_prefix,
            accel=accel,
            propagator=propagator,
            burn_duration=burn_duration,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
            stage_mode="direct",
            **transfer_kwargs,
        )
        required_staged_burns = 2 * int(n_stage_stops) + (2 if arrival_burn else 1)
        if max_burns is not None and required_staged_burns > max_burns:
            direct_result = dict(direct_result)
            direct_result["diagnostics"] = dict(direct_result.get("diagnostics", {}))
            direct_result["diagnostics"].update(
                stage_mode="best",
                selected_stage_mode="direct",
                staged_search_skipped=True,
                staged_skip_reason="max_burns",
                staged_required_burns=required_staged_burns,
                max_burns=max_burns,
            )
            return direct_result

    best = None
    best_key = None
    errors = []
    leg_tof_range = stage_tof_range if stage_tof_range is not None else tof_range
    candidates = list(_stage_candidates(
        initial_state,
        target_state,
        stage_radii=stage_radii,
        stage_plane_fractions=stage_plane_fractions,
        n_stage_phase=n_stage_phase,
        mu=mu,
    ))

    partials = [
        {
            "current_state": initial_state,
            "legs": [],
            "stage_infos": [],
            "stage_keys": set(),
        }
    ]

    def run_leg(departure_state, arrival_state, *, leg_delta_v_mode,
                leg_departure_mode, leg_t_window, leg_rendezvous,
                leg_arrival_burn, leg_arrival_window=None):
        return transfer_optimal(
            departure_state,
            arrival_state,
            objective="min_dv",
            delta_v_mode=leg_delta_v_mode,
            departure_mode=leg_departure_mode,
            rendezvous=leg_rendezvous,
            arrival_burn=leg_arrival_burn,
            t_window=leg_t_window,
            tof_range=leg_tof_range,
            arrival_window=leg_arrival_window,
            n_grid=n_grid,
            n_phase=n_phase,
            perigee_margin=perigee_margin,
            polish=polish,
            visualize=False,
            fig_prefix=fig_prefix,
            accel=accel,
            propagator=propagator,
            burn_duration=burn_duration,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
            stage_mode="direct",
            **transfer_kwargs,
        )

    def partial_key(partial):
        legs = partial["legs"]
        if not legs:
            return (0.0, 0.0)
        rank_mode = "first" if delta_v_mode == "first" else "total"
        metric = _stage_metric(legs, rank_mode, arrival_burn=True)
        first_depart = legs[0]["diagnostics"].get("t_depart", legs[0]["initial"]["t"])
        last_arrive = legs[-1]["diagnostics"].get("t_arrive", legs[-1]["final"]["t"])
        elapsed = float(last_arrive) - float(first_depart)
        return (elapsed, metric) if objective == "min_time" else (metric, elapsed)

    for stop_index in range(n_stage_stops):
        next_partials = []
        for partial in partials:
            for stage_state, stage_info in candidates:
                stage_key = _stage_info_key(stage_info)
                if stage_key in partial["stage_keys"]:
                    continue
                try:
                    if partial["legs"]:
                        leg_departure_mode, leg_t_window = _stage_departure_window(
                            partial["current_state"], stage_timing, stage_wait_window, mu)
                    else:
                        leg_departure_mode, leg_t_window = departure_mode, t_window
                    leg_delta_v_mode = "first" if stop_index == 0 and delta_v_mode == "first" else "total"
                    leg = run_leg(
                        partial["current_state"],
                        stage_state,
                        leg_delta_v_mode=leg_delta_v_mode,
                        leg_departure_mode=leg_departure_mode,
                        leg_t_window=leg_t_window,
                        leg_rendezvous=True,
                        leg_arrival_burn=True,
                    )
                    next_partials.append(
                        {
                            "current_state": _state_after_leg(leg),
                            "legs": partial["legs"] + [leg],
                            "stage_infos": partial["stage_infos"] + [stage_info],
                            "stage_keys": partial["stage_keys"] | {stage_key},
                        }
                    )
                except Exception as exc:
                    errors.append(f"stage {stop_index + 1}: {type(exc).__name__}: {exc}")
                    continue
        partials = sorted(next_partials, key=partial_key)[:stage_beam_width]
        if not partials:
            break

    for partial in partials:
        try:
            final_departure_mode, final_t_window = _stage_departure_window(
                partial["current_state"], stage_timing, stage_wait_window, mu)
            final_delta_v_mode = "total" if delta_v_mode == "first" else delta_v_mode
            final_leg = run_leg(
                partial["current_state"],
                target_state,
                leg_delta_v_mode=final_delta_v_mode,
                leg_departure_mode=final_departure_mode,
                leg_t_window=final_t_window,
                leg_rendezvous=rendezvous,
                leg_arrival_burn=arrival_burn,
                leg_arrival_window=arrival_window,
            )
            stage_summary = {
                "stops": partial["stage_infos"],
                "n_stage_stops": n_stage_stops,
                "candidate_count": len(candidates),
                "beam_width": stage_beam_width,
            }
            if len(partial["stage_infos"]) == 1:
                stage_summary.update(partial["stage_infos"][0])
            combined = _combine_stage_legs(
                partial["legs"] + [final_leg],
                stage_mode=stage_mode,
                stage_timing=stage_timing,
                objective=objective,
                delta_v_mode=delta_v_mode,
                arrival_burn=arrival_burn,
                stage_info=stage_summary,
                direct_result=direct_result,
            )
            key = _stage_objective_key(combined, objective, dv_budget)
            if key is None:
                continue
            if best is None or key < best_key:
                best = combined
                best_key = key
        except Exception as exc:
            errors.append(f"final leg: {type(exc).__name__}: {exc}")
            continue

    if best is None:
        if direct_result is not None:
            direct_result = dict(direct_result)
            direct_result["diagnostics"] = dict(direct_result.get("diagnostics", {}))
            direct_result["diagnostics"].update(stage_mode=stage_mode, staged_search_failed=True, staged_errors=errors[:10])
            return direct_result
        detail = "; ".join(errors[:5]) if errors else "no stage candidates were generated"
        raise RuntimeError(f"No feasible staged transfer found ({detail}).")

    if stage_mode == "best" and direct_result is not None:
        direct_key = _stage_objective_key(direct_result, objective, dv_budget)
        if direct_key is not None and direct_key <= best_key:
            direct_result = dict(direct_result)
            direct_result["diagnostics"] = dict(direct_result.get("diagnostics", {}))
            direct_result["diagnostics"].update(
                stage_mode="best",
                selected_stage_mode="direct",
                staged_candidate_delta_v=best["diagnostics"]["objective_delta_v"],
                staged_candidate_tof=best["tof"],
                staged_candidate_stage_stop_count=best["diagnostics"].get("stage_stop_count"),
            )
            return direct_result
        best["diagnostics"]["selected_stage_mode"] = "staged"
    return best
