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
    """Resolve ``func``'s leading positional parameters to (r, v, t) roles by name.

    Returns ``None`` when ``func``'s signature cannot be inspected at all
    (rare -- some C-implemented callables), in which case ``call_acceleration``
    falls back to the documented ``(r, t)`` / ``(r)`` call patterns.

    Parameters are matched by name (r, v, t, time, epoch) in order, stopping
    at the first one that isn't a role name. A model like
    ``accel_equatorial(r, v=None, magnitude=None)`` resolves to ``(r, v)``:
    ``magnitude`` isn't passed at all and simply keeps its default, rather
    than silently receiving ``t``. Raises ``TypeError`` only when that first
    non-role parameter is *required* (no default) -- e.g.
    ``accel_radial(r, magnitude)`` -- since there is then no default for it
    to fall back to and no safe value to guess (this is how it used to
    silently receive a GPS epoch as its ``magnitude``); such a model must be
    bound (e.g. via ``functools.partial``) before being passed to
    ``accel_add``.

    A ``magnitude=None`` default like ``accel_equatorial``'s only looks
    optional here: this function cannot tell "genuinely optional" from
    "required, but defaulted so a role parameter can follow it positionally"
    apart, so it never raises for that case. The callee must still validate
    the value it receives and raise its own ``TypeError`` if left at that
    default -- do not resurrect the fallback this once had, which read the
    real velocity vector out of ``v`` and silently used it as ``magnitude``.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        raise TypeError("acceleration model must have a fixed (r) or (r, t) signature")
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not positional:
        raise TypeError("acceleration model must have a fixed (r) or (r, t) signature")
    roles = []
    for p in positional:
        name = p.name.lstrip("_")
        if name not in _ROLES:
            if p.default is inspect.Parameter.empty:
                raise TypeError(
                    f"acceleration model has a required non-r/v/t parameter {p.name!r}; "
                    "bind it (e.g. via functools.partial) before passing it to accel_add"
                )
            break  # p and any later parameters keep their defaults, unpassed
        roles.append(_ROLES[name])
    return tuple(roles)


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
