"""Shared state parsing for 6-DoF acceleration helpers."""

from __future__ import annotations

import inspect

import numpy as np


def vector3(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return value


def position(value) -> np.ndarray:
    return vector3(value.r if hasattr(value, "r") else value, "r")


def velocity(value, fallback=None) -> np.ndarray:
    if hasattr(value, "v"):
        return vector3(value.v, "v")
    if fallback is None:
        return vector3(value, "v")
    return vector3(fallback, "v")


def time(value, fallback=None):
    return getattr(value, "t", 0.0 if fallback is None else fallback)


def state(value, v=None, t=None):
    if hasattr(value, "r") and hasattr(value, "v"):
        return position(value), velocity(value), time(value, t)
    return position(value), np.zeros(3) if v is None else velocity(v), 0.0 if t is None else t


# accel_add's documented contract: an acceleration callable takes (r) or
# (r, t). These additional names resolve a function's real parameters to
# (r, v, t) roles so call_acceleration can call it with the right argument
# in one shot instead of guessing from positional argument counts.
_ROLES = {"r": "r", "v": "v", "t": "t", "time": "t", "epoch": "t"}


def _role_signature(func):
    """Resolve ``func``'s positional parameters to (r, v, t) roles by name.

    Returns ``None`` when ``func``'s signature cannot be inspected at all
    (rare -- some C-implemented callables), in which case ``call_acceleration``
    falls back to the documented ``(r, t)`` / ``(r)`` call patterns. Raises
    ``TypeError`` when the signature *can* be inspected but has a positional
    parameter that is not named r/v/t (or time/epoch): calling such a model
    positionally would silently hand it the wrong value (this is how
    ``accel_radial(r, magnitude)`` used to receive a GPS epoch as its
    ``magnitude``) rather than require it to be bound (e.g. via
    ``functools.partial``) before being passed to ``accel_add``.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not positional or any(p.kind == p.VAR_POSITIONAL for p in params):
        raise TypeError("acceleration model must have a fixed (r) or (r, t) signature")
    names = [p.name.lstrip("_") for p in positional]
    unknown = [n for n in names if n not in _ROLES]
    if unknown:
        raise TypeError(
            f"acceleration model has unrecognized parameter(s) {unknown}; "
            "bind extra parameters (e.g. via functools.partial) before passing it to accel_add"
        )
    return tuple(_ROLES[n] for n in names)


def call_acceleration(func, r, v=None, t=None):
    signature = _role_signature(func)
    if signature is None:
        # Non-inspectable callable: try the documented (r, t) then (r) call
        # patterns. A TypeError from inside the function itself is expected
        # to propagate -- it isn't caught here because there is nothing left
        # to retry with once the documented patterns are exhausted.
        try:
            return vector3(func(r, t), "acceleration")
        except TypeError:
            return vector3(func(r), "acceleration")
    values = {"r": r, "v": v, "t": t}
    return vector3(func(*(values[role] for role in signature)), "acceleration")
