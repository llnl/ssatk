"""Injection transfers based on fitted endpoint ellipses.

``ellipse_fit`` builds a Keplerian ellipse through two endpoint positions. This
module turns that fitted arc into a transfer-style result by comparing the
spacecraft's initial velocity with the fitted-ellipse departure velocity and,
optionally, comparing the fitted-ellipse arrival velocity with a target velocity.

Use ``ellipse_inject`` when the desired workflow is "burn from this state onto an
elliptic arc that reaches this endpoint". Use ``transfer_to_endpoint`` to route
the same endpoint inputs to another canonical solver.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from ssapy import Orbit

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics._transfer_result import (
    as_time,
    maneuver_burn,
    trajectory_dict,
    transfer_result,
    transfer_state,
)
from ssatk.orbital_mechanics.ellipse_fit import ellipse_fit

__all__ = [
    "ellipse_inject",
    "ellipse_intercept",
    "ellipse_rendezvous",
    "ellipse_insertion",
    "transfer_to_endpoint",
]

_ARRIVAL_MODE_ALIASES = {
    "inject": "inject",
    "intercept": "intercept",
    "rendezvous": "rendezvous",
    "insertion": "insertion",
}

_ENDPOINT_METHOD_ALIASES = {
    "ellipse": "ellipse",
    "ellipse_intercept": "ellipse_intercept",
    "ellipse_rendezvous": "ellipse_rendezvous",
    "ellipse_insertion": "ellipse_insertion",
    "lambert": "lambert",
    "ssapy": "ssapy",
    "optimal": "optimal",
    "hohmann": "hohmann",
    "bielliptic": "bielliptic",
}


def _normalize_keyword(value, aliases, name):
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported {name}={value!r}; choose one of {sorted(aliases)}") from exc


def _is_vector(value):
    try:
        return np.asarray(value, dtype=float).reshape(3).shape == (3,)
    except (TypeError, ValueError):
        return False


def _state_input_has_velocity(value):
    if value is None or np.isscalar(value):
        return False
    if getattr(value, "v", None) is not None:
        return True
    if isinstance(value, Mapping):
        return value.get("v") is not None
    if _is_vector(value):
        return False
    try:
        values = tuple(value)
    except TypeError:
        return False
    return len(values) in (2, 3) and values[1] is not None


def _state_input_has_time(value):
    if value is None or np.isscalar(value) or _is_vector(value):
        return False
    if getattr(value, "t", None) is not None:
        return True
    if isinstance(value, Mapping):
        return value.get("t") is not None
    try:
        values = tuple(value)
    except TypeError:
        return False
    return len(values) == 3 and values[2] is not None


def _arrival_burn_for_mode(arrival_mode):
    if arrival_mode in ("rendezvous", "insertion"):
        return True
    if arrival_mode in ("inject", "intercept"):
        return False
    raise AssertionError(f"Unhandled arrival_mode {arrival_mode!r}")


def _arrival_timing_constraint(arrival_mode):
    return "fixed" if arrival_mode in ("intercept", "rendezvous") else "free"


def _state_with_time(state, time_value):
    updated = dict(state)
    updated["t"] = as_time(time_value, default=state.get("t", 0.0))
    return transfer_state(state=updated)


def _state_from_parts(r, v=None, t=0.0, *, label=None, require_velocity=False, name="state"):
    if r is None:
        raise ValueError(f"{name} requires a position vector")
    if require_velocity and v is None:
        raise ValueError(f"{name} requires both position and velocity")
    return transfer_state(r=r, v=v, t=t, label=label)


def _parse_endpoint_boundary(
    *args,
    orbit1=None,
    orbit2=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=None,
    t2=None,
    tof=None,
):
    """Normalize endpoint inputs while allowing target position-only calls."""
    explicit_target_time = t2 is not None or tof is not None

    if orbit1 is not None:
        if initial is not None:
            raise TypeError("Specify either orbit1 or initial, not both")
        initial = orbit1
    if orbit2 is not None:
        if target is not None:
            raise TypeError("Specify either orbit2 or target, not both")
        target = orbit2

    if args:
        if any(value is not None for value in (initial, target, r1, v1, r2, v2)):
            raise TypeError("Specify endpoint boundary states either positionally or by keyword, not both")
        if len(args) == 2:
            initial, target = args
        elif len(args) == 3 and _is_vector(args[0]) and _is_vector(args[1]) and _is_vector(args[2]):
            r1, v1, r2 = args
        elif len(args) == 4 and all(_is_vector(value) for value in args):
            r1, v1, r2, v2 = args
        else:
            raise TypeError(
                "ellipse_inject expects (initial, target), (r1, v1, r2), "
                "or (r1, v1, r2, v2)"
            )

    initial_time = as_time(t1, default=0.0)
    if initial is None:
        initial_state = _state_from_parts(r1, v1, initial_time, require_velocity=True, name="initial")
    else:
        if _is_vector(initial):
            raise ValueError("initial requires both position and velocity; use r1=... and v1=... for raw vectors")
        initial_state = transfer_state(state=initial)
        if t1 is not None:
            initial_state["t"] = initial_time

    if tof is not None:
        target_time = initial_state["t"] + float(tof)
    elif t2 is not None:
        target_time = as_time(t2, default=initial_state["t"])
    else:
        target_time = None

    target_velocity_inferred = False
    if target is None:
        target_state = _state_from_parts(
            r2,
            v2,
            initial_state["t"] if target_time is None else target_time,
            label="target_endpoint",
            require_velocity=False,
            name="target",
        )
        target_velocity_inferred = v2 is None
    else:
        explicit_target_time = explicit_target_time or _state_input_has_time(target)
        if _is_vector(target):
            target_state = _state_from_parts(
                target,
                None,
                initial_state["t"] if target_time is None else target_time,
                label="target_endpoint",
                require_velocity=False,
                name="target",
            )
        else:
            target_state = transfer_state(state=target)
        target_velocity_inferred = not _state_input_has_velocity(target)
        if target_time is not None:
            target_state["t"] = target_time

    return initial_state, target_state, explicit_target_time, target_velocity_inferred


def ellipse_inject(
    *args,
    orbit1=None,
    orbit2=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=None,
    t2=None,
    tof=None,
    arrival_mode="inject",
    match_arrival_velocity=None,
    a_m=None,
    e=None,
    F2_m=None,
    inc: float = 0.0,
    inc_deg=None,
    n_pts: int = 1000,
    tol=1e-8,
    v_pref_m_s=None,
    pos_tol_m=1.0e3,
    check_positions=True,
    enforce_tof=False,
    tof_tol_s=1.0,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    plot=False,
    save_path=False,
    **save_kwargs,
):
    """Burn onto an ``ellipse_fit`` arc that reaches a target endpoint.

    Parameters
    ----------
    initial, target or positional boundary states
        Accepted forms are ``(initial, target)``, ``(r1, v1, r2)``,
        ``(r1, v1, r2, v2)``, or the corresponding keyword forms. ``target`` may
        be position-only for intercept/injection cases; if a target velocity is
        absent and an arrival burn is requested, a circular velocity is inferred
        by ``transfer_state``.
    arrival_mode : {"inject", "intercept", "rendezvous", "insertion"}, default="inject"
        ``"inject"`` computes only the departure burn needed to enter the fitted
        transfer arc; the endpoint is reached at the arc's natural time of
        flight. ``"intercept"`` reaches a specified endpoint time without
        matching arrival velocity. ``"rendezvous"`` reaches a specified endpoint
        time and matches arrival velocity. ``"insertion"`` matches arrival
        velocity at the arc's natural endpoint time, suitable for free-time orbit
        insertion into the target state/orbit.
    enforce_tof : bool, default=False
        ``ellipse_fit`` determines the time of flight from the fitted ellipse.
        Set this true to reject cases where a supplied ``tof``/``t2`` does not
        match the fitted-ellipse time of flight within ``tof_tol_s``.

    Returns
    -------
    dict
        Canonical SSATK transfer result with extra ``fit`` and ``ellipse_fit``
        entries containing the underlying fitted ellipse dictionary.
    """
    arrival_mode = _normalize_keyword(arrival_mode, _ARRIVAL_MODE_ALIASES, "arrival_mode")
    expected_arrival_burn = _arrival_burn_for_mode(arrival_mode)
    if match_arrival_velocity is not None and bool(match_arrival_velocity) != expected_arrival_burn:
        raise ValueError(
            f"arrival_mode={arrival_mode!r} fixes match_arrival_velocity={expected_arrival_burn!r}; "
            "choose a different arrival_mode instead of overriding the term."
        )
    arrival_burn = expected_arrival_burn

    initial_state, target_state, explicit_target_time, target_velocity_inferred = _parse_endpoint_boundary(
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
        tof=tof,
    )
    timing_constraint = _arrival_timing_constraint(arrival_mode)
    if timing_constraint == "fixed" and not explicit_target_time:
        raise ValueError(
            f"arrival_mode={arrival_mode!r} requires an explicit arrival time; "
            "supply tof, t2, or a target state with t. Use arrival_mode='inject' "
            "or 'insertion' for free-time fitted-ellipse arcs."
        )
    if timing_constraint == "fixed":
        enforce_tof = True

    preferred_velocity = initial_state["v"] if v_pref_m_s is None else v_pref_m_s
    fit = ellipse_fit(
        initial_state["r"],
        target_state["r"],
        a_m=a_m,
        e=e,
        F2_m=F2_m,
        inc=inc,
        inc_deg=inc_deg,
        n_pts=n_pts,
        tol=tol,
        v_pref_m_s=preferred_velocity,
        plot=plot,
        save_path=save_path,
        time_of_departure=initial_state["t"],
        **save_kwargs,
    )

    fit_r = np.asarray(fit["r"], dtype=float)
    fit_v = np.asarray(fit["v"], dtype=float)
    fit_t_rel = np.asarray(fit["t_rel"], dtype=float)
    if fit_r.ndim != 2 or fit_v.ndim != 2 or fit_r.shape != fit_v.shape or fit_r.shape[0] < 2:
        raise RuntimeError("ellipse_fit returned invalid r/v arrays")

    tof_fit = float(fit_t_rel[-1])
    requested_tof = None
    if explicit_target_time:
        requested_tof = float(target_state["t"] - initial_state["t"])
        if enforce_tof and abs(requested_tof - tof_fit) > float(tof_tol_s):
            raise ValueError(
                f"ellipse_fit time of flight is {tof_fit:.6g} s, but the supplied "
                f"endpoint timing requested {requested_tof:.6g} s. Use a fixed-time "
                "Lambert method through transfer_to_endpoint(method='lambert') or "
                "set enforce_tof=False."
            )

    arrival_time = initial_state["t"] + tof_fit
    target_at_arrival = _state_with_time(target_state, arrival_time)
    transfer_initial_state = _state_with_time(initial_state, initial_state["t"])
    transfer_arrival_state = transfer_state(r=fit_r[-1], v=fit_v[-1], t=arrival_time, label="ellipse_endpoint")

    start_residual = float(np.linalg.norm(fit_r[0] - initial_state["r"]))
    end_residual = float(np.linalg.norm(fit_r[-1] - target_state["r"]))
    if check_positions:
        if start_residual > float(pos_tol_m):
            raise ValueError(f"Fitted ellipse starts {start_residual:.3e} m from initial state")
        if end_residual > float(pos_tol_m):
            raise ValueError(f"Fitted ellipse ends {end_residual:.3e} m from target endpoint")

    departure_delta_v = fit_v[0] - initial_state["v"]
    burns = [
        maneuver_burn(
            name="ellipse_injection",
            state=transfer_initial_state,
            delta_v=departure_delta_v,
            t=initial_state["t"],
            t_start=initial_state["t"],
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
            kind="impulsive" if burn_accel is None and thrust is None else "finite_burn",
            notes="Inject from the departure state onto the fitted ellipse.",
        )
    ]

    final_state = transfer_arrival_state
    if arrival_burn:
        arrival_delta_v = target_at_arrival["v"] - fit_v[-1]
        burns.append(
            maneuver_burn(
                name="ellipse_arrival_match",
                state=transfer_arrival_state,
                delta_v=arrival_delta_v,
                t=arrival_time,
                t_end=arrival_time,
                burn_accel=burn_accel,
                thrust=thrust,
                mass=mass,
                isp=isp,
                kind="impulsive" if burn_accel is None and thrust is None else "finite_burn",
                notes="Match the target velocity at the fitted-ellipse endpoint.",
            )
        )
        final_state = target_at_arrival

    trajectory = trajectory_dict(
        t=initial_state["t"] + fit_t_rel,
        r=fit_r,
        v=fit_v,
        frame="gcrf",
    )
    transfer_orbit = Orbit(fit_r[0], fit_v[0], initial_state["t"], mu=EARTH_MU)
    diagnostics = {
        "arrival_mode": arrival_mode,
        "timing_constraint": timing_constraint,
        "arrival_velocity_match": arrival_burn,
        "arrival_burn": arrival_burn,
        "target_velocity_inferred": bool(target_velocity_inferred),
        "requested_tof_s": requested_tof,
        "tof_error_s": None if requested_tof is None else tof_fit - requested_tof,
        "start_position_residual_m": start_residual,
        "end_position_residual_m": end_residual,
        "ellipse": {
            "a_m": float(fit["a"]),
            "e": float(fit["e"]),
            "i_rad": float(fit["i"]),
            "raan_rad": float(fit["raan"]),
            "pa_rad": float(fit["pa"]),
            "rp_m": float(fit["rp"]),
            "ra_m": float(fit["ra"]),
            "period_s": float(fit["period"]),
            "rot_dir": int(fit["rot_dir"]),
        },
    }
    result = transfer_result(
        method="ellipse_inject",
        initial=initial_state,
        target=target_at_arrival,
        final=final_state,
        burns=burns,
        trajectory=trajectory,
        transfer_orbits=[transfer_orbit],
        tof=tof_fit,
        assumptions=[
            "two-body fitted ellipse through endpoint positions",
            "time of flight is set by ellipse_fit geometry, not by Lambert fixed-time constraints",
        ],
        diagnostics=diagnostics,
    )
    result["fit"] = fit
    result["ellipse_fit"] = fit
    return result


def _ellipse_mode_wrapper(args, kwargs, *, arrival_mode, match_arrival_velocity, wrapper_name):
    kwargs = dict(kwargs)
    supplied_arrival_mode = kwargs.pop("arrival_mode", None)
    if supplied_arrival_mode is not None:
        normalized = _normalize_keyword(supplied_arrival_mode, _ARRIVAL_MODE_ALIASES, "arrival_mode")
        if normalized != arrival_mode:
            raise ValueError(
                f"{wrapper_name} fixes arrival_mode={arrival_mode!r}; "
                f"received arrival_mode={supplied_arrival_mode!r}. Use ellipse_inject "
                "directly to select a different arrival mode."
            )
    supplied_match = kwargs.pop("match_arrival_velocity", None)
    if supplied_match is not None and bool(supplied_match) != bool(match_arrival_velocity):
        raise ValueError(
            f"{wrapper_name} fixes match_arrival_velocity={match_arrival_velocity!r}; "
            f"received match_arrival_velocity={supplied_match!r}. Use ellipse_inject "
            "directly to override this behavior."
        )
    result = ellipse_inject(
        *args,
        arrival_mode=arrival_mode,
        match_arrival_velocity=match_arrival_velocity,
        **kwargs,
    )
    result["method"] = wrapper_name
    result["assumptions"].append(f"readable wrapper around ellipse_inject(arrival_mode={arrival_mode!r})")
    return result


def ellipse_intercept(*args, **kwargs):
    """Fixed-time fitted-ellipse endpoint intercept.

    This is the readable wrapper for ``ellipse_inject(...,
    arrival_mode="intercept", match_arrival_velocity=False)``. It reaches a
    specified target position at a specified time, but it does not add a final
    velocity-matching burn.
    """
    return _ellipse_mode_wrapper(
        args,
        kwargs,
        arrival_mode="intercept",
        match_arrival_velocity=False,
        wrapper_name="ellipse_intercept",
    )


def ellipse_rendezvous(*args, **kwargs):
    """Fixed-time fitted-ellipse endpoint rendezvous.

    This is the readable wrapper for ``ellipse_inject(...,
    arrival_mode="rendezvous", match_arrival_velocity=True)``. It computes the
    injection burn onto the fitted ellipse and the final burn required to match
    the target velocity at the specified endpoint time.
    """
    return _ellipse_mode_wrapper(
        args,
        kwargs,
        arrival_mode="rendezvous",
        match_arrival_velocity=True,
        wrapper_name="ellipse_rendezvous",
    )


def ellipse_insertion(*args, **kwargs):
    """Free-time fitted-ellipse endpoint insertion.

    This is the readable wrapper for ``ellipse_inject(...,
    arrival_mode="insertion", match_arrival_velocity=True)``. It injects onto a
    fitted ellipse and matches the target velocity at the arc's natural endpoint
    time, suitable for entering a final orbit without imposing a fixed arrival
    epoch.
    """
    return _ellipse_mode_wrapper(
        args,
        kwargs,
        arrival_mode="insertion",
        match_arrival_velocity=True,
        wrapper_name="ellipse_insertion",
    )


def transfer_to_endpoint(method="ellipse", *args, **kwargs):
    """Dispatch endpoint-transfer inputs to an SSATK transfer method.

    This convenience wrapper keeps one input style while letting users compare
    endpoint-reaching strategies:

    ``method="ellipse"``
        Use :func:`ellipse_inject`; good for a departure injection onto a fitted
        transfer ellipse.
    ``method="ellipse_intercept"`` / ``"ellipse_rendezvous"``
        Use fixed-time endpoint modes. Supply ``tof``, ``t2``, or a target epoch.
    ``method="ellipse_insertion"``
        Use free-time target-orbit insertion on the fitted endpoint ellipse.
    ``method="lambert"`` / ``"ssapy"``
        Use :func:`transfer_ssapy` for a fixed-time Lambert solve; supply
        ``tof`` or ``t2``.
    ``method="optimal"``
        Use :func:`transfer_optimal` for a searched Lambert design space.
    ``method="hohmann"`` / ``"bielliptic"``
        Use analytic circular-orbit transfer approximations.
    """
    normalized = _normalize_keyword(method, _ENDPOINT_METHOD_ALIASES, "method")
    if normalized == "ellipse":
        return ellipse_inject(*args, **kwargs)
    if normalized == "ellipse_intercept":
        return ellipse_intercept(*args, **kwargs)
    if normalized == "ellipse_rendezvous":
        return ellipse_rendezvous(*args, **kwargs)
    if normalized == "ellipse_insertion":
        return ellipse_insertion(*args, **kwargs)
    if normalized in {"lambert", "ssapy"}:
        from ssatk.orbital_mechanics.transfer_ssapy_function import transfer_ssapy

        return transfer_ssapy(*args, **kwargs)
    if normalized == "optimal":
        from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal

        return transfer_optimal(*args, **kwargs)
    if normalized == "hohmann":
        from ssatk.orbital_mechanics.transfer_hohmann import transfer_hohmann

        return transfer_hohmann(*args, **kwargs)
    if normalized == "bielliptic":
        from ssatk.orbital_mechanics.transfer_bielliptic import transfer_bielliptic

        return transfer_bielliptic(*args, **kwargs)
    raise AssertionError(f"Unhandled endpoint method {normalized!r}")
