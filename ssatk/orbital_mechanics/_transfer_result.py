"""Canonical transfer input/output helpers for SSATK maneuver solvers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from ssapy import Orbit
from ssapy.constants import EARTH_MU

from ssatk.constants import G0
from ssatk.coordinates import gcrf_to_ntw
from ssatk.time_functions._gps import _to_gps_seconds

TRANSFER_SCHEMA_VERSION = "ssatk.transfer.v2"


def as_vector(value, name):
    array = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def as_time(value, default=0.0):
    if value is None:
        return float(default)
    return _to_gps_seconds(value)


def circular_state(radius, *, t=0.0, mu=EARTH_MU, plane="xy", prograde=True):
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Circular-orbit radius must be positive")
    if plane.lower() != "xy":
        raise ValueError("Only plane='xy' circular radius inputs are currently supported")
    r = np.array([radius, 0.0, 0.0], dtype=float)
    sign = 1.0 if prograde else -1.0
    v = np.array([0.0, sign * np.sqrt(mu / radius), 0.0], dtype=float)
    return transfer_state(r=r, v=v, t=t, label="circular_radius", mu=mu)


def transfer_state(*, state=None, r=None, v=None, t=None, label=None, mu=EARTH_MU):
    """Normalize an Orbit, mapping, tuple, radius, or explicit r/v/t into a state dict."""
    orbit = None
    if state is not None:
        if isinstance(state, Orbit) or all(hasattr(state, attr) for attr in ("r", "v", "t")):
            orbit = state
            r = state.r if r is None else r
            v = state.v if v is None else v
            t = state.t if t is None else t
        elif isinstance(state, Mapping):
            r = state.get("r", r)
            v = state.get("v", v)
            t = state.get("t", t)
            label = state.get("label", label)
            orbit = state.get("orbit", orbit)
        elif np.isscalar(state):
            return circular_state(state, t=0.0 if t is None else t, mu=mu)
        else:
            values = tuple(state)
            if len(values) == 3:
                r, v, t = values
            elif len(values) == 2:
                r, v = values
            else:
                raise ValueError("State tuples must be (r, v) or (r, v, t)")

    if r is None:
        raise ValueError("State requires a position vector or circular-orbit radius")
    r = as_vector(r, "r")
    if v is None:
        radius = np.linalg.norm(r)
        if radius <= 0.0:
            raise ValueError("Cannot infer circular velocity for zero radius")
        tangent = np.cross([0.0, 0.0, 1.0], r)
        if np.linalg.norm(tangent) <= 1e-12:
            tangent = np.cross([0.0, 1.0, 0.0], r)
        tangent = tangent / np.linalg.norm(tangent)
        v = np.sqrt(mu / radius) * tangent
    v = as_vector(v, "v")
    t = as_time(t, default=0.0)
    return {"label": label, "r": r, "v": v, "t": t, "orbit": orbit}


def _is_state_vector(value):
    if isinstance(value, Orbit) or np.isscalar(value):
        return False
    try:
        return np.asarray(value, dtype=float).reshape(3).shape == (3,)
    except (TypeError, ValueError):
        return False


def transfer_boundary_states(
    *args,
    orbit1=None,
    orbit2=None,
    departure=None,
    arrival=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=0.0,
    t2=None,
    tof=None,
    mu=EARTH_MU,
    name="transfer",
):
    """Normalize two transfer boundary states.

    Accepted forms are ``(initial, target)``, explicit ``initial=`` /
    ``target=`` (or ``departure=`` / ``arrival=``), and raw state vectors as
    ``(r1, v1, r2, v2)`` or keyword ``r1=..., v1=..., r2=..., v2=...``.  Raw
    state-vector epochs default to ``t1=0`` and ``t2=t1`` unless ``t2`` or
    ``tof`` is supplied; fixed-time solvers should still require positive time
    of flight after calling this helper.
    """
    if orbit1 is not None:
        if initial is not None or departure is not None:
            raise TypeError(f"{name} received orbit1 plus initial/departure")
        initial = orbit1
    if orbit2 is not None:
        if target is not None or arrival is not None:
            raise TypeError(f"{name} received orbit2 plus target/arrival")
        target = orbit2

    if args:
        if len(args) == 2:
            if departure is not None or arrival is not None or initial is not None or target is not None:
                raise TypeError(f"{name} received both positional and keyword boundary states")
            departure, arrival = args
        elif len(args) == 4 and all(_is_state_vector(value) for value in args):
            if any(value is not None for value in (departure, arrival, initial, target, r1, v1, r2, v2)):
                raise TypeError(f"{name} received both positional and keyword boundary states")
            r1, v1, r2, v2 = args
        else:
            raise TypeError(
                f"{name} expects boundary states as (initial, target) or raw vectors "
                "(r1, v1, r2, v2)"
            )

    if initial is not None:
        departure = initial
    if target is not None:
        arrival = target

    vector_values = (r1, v1, r2, v2)
    if any(value is not None for value in vector_values):
        if departure is not None or arrival is not None:
            raise TypeError(f"{name} received both boundary states and raw r/v vectors")
        if any(value is None for value in vector_values):
            raise ValueError(f"{name} raw-vector input requires r1, v1, r2, and v2")
        t1 = as_time(t1, default=0.0)
        if t2 is None:
            t2 = t1 + float(tof) if tof is not None else t1
        t2 = as_time(t2, default=t1)
        departure = (r1, v1, t1)
        arrival = (r2, v2, t2)

    if departure is None or arrival is None:
        raise ValueError(
            f"{name} requires boundary states as (initial, target), "
            "initial=/target=, or raw vectors r1, v1, r2, v2"
        )

    departure_state = transfer_state(state=departure, mu=mu)
    arrival_state = transfer_state(state=arrival, mu=mu)
    if tof is not None and arrival_state["t"] <= departure_state["t"]:
        arrival_state["t"] = departure_state["t"] + float(tof)
    return departure_state, arrival_state


def trajectory_dict(*, t=None, r=None, v=None, frame="gcrf"):
    if t is None and r is None and v is None:
        return None
    trajectory = {"frame": frame}
    trajectory["t"] = None if t is None else np.asarray(t, dtype=float)
    trajectory["r"] = None if r is None else np.asarray(r, dtype=float)
    trajectory["v"] = None if v is None else np.asarray(v, dtype=float)
    return trajectory


def _ntw_or_none(delta_v, state):
    if delta_v is None or state is None:
        return None
    try:
        return gcrf_to_ntw(delta_v, state["r"], state["v"])
    except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
        return None


def _propellant_mass(delta_v_mag, mass_kg, isp_s):
    if mass_kg is None or isp_s is None:
        return None
    mass_kg = float(mass_kg)
    isp_s = float(isp_s)
    if mass_kg <= 0.0 or isp_s <= 0.0:
        raise ValueError("mass_kg and isp_s must be positive when supplied")
    return mass_kg * (1.0 - np.exp(-float(delta_v_mag) / (isp_s * G0)))


def maneuver_burn(
    *,
    name,
    state,
    delta_v,
    frame="gcrf",
    t=None,
    t_start=None,
    t_end=None,
    duration=None,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    kind="impulsive",
    notes=None,
):
    """Build a standard burn dictionary from an impulsive delta-v vector."""
    state = transfer_state(state=state)
    delta_v = as_vector(delta_v, "delta_v")
    delta_v_mag = float(np.linalg.norm(delta_v))
    t = state["t"] if t is None else as_time(t)

    if thrust is not None and mass is None:
        raise ValueError("mass is required when thrust is supplied")
    if mass is not None and thrust is None and burn_accel is None:
        raise ValueError("thrust or burn_accel is required when mass is supplied")
    if thrust is not None and burn_accel is not None:
        raise ValueError("Specify either thrust+mass or burn_accel, not both")

    accel_mag = None
    if thrust is not None:
        thrust = float(thrust)
        mass = float(mass)
        if thrust <= 0.0 or mass <= 0.0:
            raise ValueError("thrust and mass must be positive")
        accel_mag = thrust / mass
    elif burn_accel is not None:
        accel_mag = float(burn_accel)
        if accel_mag <= 0.0:
            raise ValueError("burn_accel must be positive")

    if duration is None and accel_mag is not None:
        duration = delta_v_mag / accel_mag if delta_v_mag > 0.0 else 0.0
    if duration is not None:
        duration = float(duration)
        if duration < 0.0:
            raise ValueError("duration must be non-negative")
        if t_start is None and t_end is None:
            t_start = t - 0.5 * duration
            t_end = t + 0.5 * duration
        elif t_start is None:
            t_end = as_time(t_end)
            t_start = t_end - duration
        elif t_end is None:
            t_start = as_time(t_start)
            t_end = t_start + duration
    if t_start is not None:
        t_start = as_time(t_start)
    if t_end is not None:
        t_end = as_time(t_end)
    if duration is None and t_start is not None and t_end is not None:
        duration = t_end - t_start

    direction = np.zeros(3) if delta_v_mag == 0.0 else delta_v / delta_v_mag
    acceleration = None
    if duration is not None and duration > 0.0:
        acceleration = delta_v / duration
    elif accel_mag is not None:
        acceleration = direction * accel_mag

    delta_v_ntw = _ntw_or_none(delta_v, state)
    acceleration_ntw = _ntw_or_none(acceleration, state) if acceleration is not None else None
    return {
        "name": name,
        "kind": kind,
        "frame": frame,
        "state": state,
        "t": t,
        "t_start": t_start,
        "t_end": t_end,
        "duration": duration,
        "delta_v": delta_v,
        "delta_v_ntw": delta_v_ntw,
        "delta_v_mag": delta_v_mag,
        "acceleration": acceleration,
        "acceleration_ntw": acceleration_ntw,
        "acceleration_mag": None if acceleration is None else float(np.linalg.norm(acceleration)),
        "thrust": None if thrust is None else float(thrust),
        "mass": None if mass is None else float(mass),
        "isp": None if isp is None else float(isp),
        "propellant_mass": _propellant_mass(delta_v_mag, mass, isp),
        "notes": notes,
    }


def transfer_result(
    *,
    method,
    initial,
    target,
    final=None,
    burns=None,
    trajectory=None,
    transfer_orbits=None,
    tof=None,
    success=True,
    assumptions=None,
    diagnostics=None,
):
    """Return the canonical transfer result dictionary."""
    initial = transfer_state(state=initial)
    target = transfer_state(state=target)
    final = target if final is None else transfer_state(state=final)
    burns = list(burns or [])
    delta_v_total = float(sum(burn["delta_v_mag"] for burn in burns))
    hardware = {
        "thrust": next((burn["thrust"] for burn in burns if burn.get("thrust") is not None), None),
        "mass": next((burn["mass"] for burn in burns if burn.get("mass") is not None), None),
        "isp": next((burn["isp"] for burn in burns if burn.get("isp") is not None), None),
        "acceleration_mag": next((burn["acceleration_mag"] for burn in burns if burn.get("acceleration_mag") is not None), None),
    }
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "method": method,
        "units": {
            "distance": "m",
            "velocity": "m/s",
            "time": "GPS seconds",
            "delta_v": "m/s",
            "acceleration": "m/s^2",
            "thrust": "N",
            "mass": "kg",
            "isp": "s",
        },
        "initial": initial,
        "target": target,
        "final": final,
        "tof": None if tof is None else float(tof),
        "burns": burns,
        "delta_v_total": delta_v_total,
        "delta_v_vectors": [burn["delta_v"] for burn in burns],
        "delta_v_ntw_vectors": [burn["delta_v_ntw"] for burn in burns],
        "delta_v_magnitudes": [burn["delta_v_mag"] for burn in burns],
        "trajectory": trajectory,
        "transfer_orbits": list(transfer_orbits or []),
        "hardware": hardware,
        "success": bool(success),
        "assumptions": list(assumptions or []),
        "diagnostics": dict(diagnostics or {}),
    }
