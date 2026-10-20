# ssatk/propagators_orbit/int_utils.py

import inspect

import numpy as np
from scipy.interpolate import interp1d

from ..time_functions import to_gps


def acceleration_adapter(model, signature=None):
    """Adapt a declared acceleration callback to ``(t, r, v)`` calls.

    Signature resolution happens once, before integration. Exceptions raised
    inside a callback are therefore never mistaken for an argument mismatch.
    """
    if not callable(model):
        raise TypeError("acceleration model must be callable")
    if signature is None:
        signature = getattr(model, "acceleration_signature", None)
    if signature is None and getattr(model, "spacecraft_acceleration_model", False):
        def spacecraft_model(t, r, v):
            return model(t, r, v, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        spacecraft_model.acceleration_signature = "trv"
        return spacecraft_model
    try:
        parameters = inspect.signature(model)
    except (TypeError, ValueError):
        parameters = None
    if signature is None:
        if parameters is None:
            raise TypeError("non-inspectable model requires acceleration_signature")
        roles = {"r": "r", "v": "v", "t": "t", "time": "t", "epoch": "t"}
        positional = [p for p in parameters.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if (any(p.kind == p.VAR_POSITIONAL for p in parameters.parameters.values())
                or any(p.name.lstrip('_') not in roles for p in positional)):
            raise TypeError("ambiguous model requires acceleration_signature; bind force parameters first")
        signature = ''.join(roles[p.name.lstrip('_')] for p in positional)
    if signature not in ("r", "v", "rt", "tr", "rv", "rvt", "trv"):
        raise ValueError("unsupported acceleration_signature")
    if parameters is not None:
        parameters.bind(*([None] * len(signature)))
    indices = tuple({"t": 0, "r": 1, "v": 2}[role] for role in signature)

    def evaluate(t, r, v):
        values = (t, r, v)
        return model(*(values[index] for index in indices))

    evaluate.acceleration_signature = "trv"
    return evaluate


def precompute_third_body_positions(t, body_name):
    """
    Precompute an interpolated position function for a third body (Moon/Sun/etc.).
    Returns a callable pos(t_query) -> (N,3) position array.
    Use at least two strictly increasing epochs. Interpolation is cubic for
    four or more samples, quadratic for three and linear for two.
    """
    from ssapy import get_body

    t_gps = np.asarray(to_gps(t), dtype=float)
    if (t_gps.ndim != 1 or t_gps.size < 2 or not np.all(np.isfinite(t_gps))
            or np.any(np.diff(t_gps) <= 0)):
        raise ValueError("ephemeris grid must contain at least two finite increasing epochs")
    body = get_body(body_name)
    r_body = np.asarray(body.position(t_gps), dtype=float).T
    if r_body.shape != (len(t_gps), 3) or not np.all(np.isfinite(r_body)):
        raise ValueError("body.position must return finite positions with shape (3, n)")

    interp_funcs = [
        interp1d(t_gps, r_body[:, i], kind=min(3, len(t_gps) - 1), fill_value="extrapolate")
        for i in range(3)
    ]

    def interpolated_position(t_query):
        tq = np.asarray(to_gps(t_query), dtype=float)
        return np.stack([f(tq) for f in interp_funcs], axis=-1)

    return interpolated_position


def build_profile(profile, t_arr):
    """
    Build an (n,) acceleration-magnitude profile aligned to t_arr.

    ``None`` gives zeros and a scalar gives constant acceleration. Numeric lists
    and arrays of length n are samples. Explicit ``{'samples': values}`` and
    ``{'segments': specs}`` remove all sample/segment ambiguity. Bare scalar
    tuples of length 2 or 3 are legacy segments, but are rejected if their length
    also equals n; wrap them explicitly in that case.

    Dictionary segments accept ``start_index``/``end_index`` (nonnegative integer
    indices) OR ``start_time``/``end_time`` (numeric times on t_arr). Legacy
    ``start``/``end`` always mean indices for integers and times for floats;
    boolean, negative-index and mixed explicit/legacy bounds are rejected.
    Bounds select a half-open interval; overlapping segments add. RK4 and
    leapfrog supply elapsed seconds since their first epoch, so their time
    bounds are elapsed seconds, never absolute GPS epochs.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    if t_arr.ndim != 1 or not np.all(np.isfinite(t_arr)) or np.any(np.diff(t_arr) <= 0):
        raise ValueError("profile times must be finite and strictly increasing")
    n = len(t_arr)
    out = np.zeros(n, float)

    if profile is None:
        return out

    if np.isscalar(profile):
        out[:] = _profile_value(profile)
        return out

    explicit_segments = isinstance(profile, dict) and "segments" in profile
    if isinstance(profile, dict) and "samples" in profile:
        if set(profile) != {"samples"}:
            raise ValueError("samples cannot be combined with segment keys")
        return _profile_samples(profile["samples"], n)
    if explicit_segments:
        if set(profile) != {"segments"}:
            raise ValueError("segments cannot be combined with other keys")
        segments = profile["segments"]
        if isinstance(segments, dict):
            segments = [segments]
    elif isinstance(profile, dict):
        segments = [profile]
    elif isinstance(profile, (list, tuple, np.ndarray)):
        if isinstance(profile, np.ndarray) and profile.ndim == 0:
            return build_profile(float(profile), t_arr)
        numeric = all(np.isscalar(item) for item in profile)
        if numeric and isinstance(profile, tuple) and len(profile) in (2, 3):
            if len(profile) == n:
                raise ValueError("ambiguous tuple: use {'samples': ...} or {'segments': [...]}")
            segments = [profile]
        elif numeric or isinstance(profile, np.ndarray):
            return _profile_samples(profile, n)
        else:
            segments = profile
    else:
        raise TypeError("Unsupported profile format")

    for seg in segments:
        if isinstance(seg, dict):
            keys = set(seg)
            bounds = ({"start", "end"}, {"start_index", "end_index"}, {"start_time", "end_time"})
            if keys - set.union(*bounds, {"thrust", "accel"}):
                raise ValueError("unknown profile segment keys")
            active = [mode for mode, names in enumerate(bounds) if keys & names]
            if len(active) > 1:
                raise ValueError("cannot mix index, time and legacy bounds")
            mode = active[0] if active else 0
            start_key, end_key = (("start", "end"), ("start_index", "end_index"), ("start_time", "end_time"))[mode]
            start, end = seg.get(start_key), seg.get(end_key)
            if "thrust" in seg and "accel" in seg:
                raise ValueError("supply thrust or accel, not both")
            thrust = seg.get("thrust", seg.get("accel", 0))
        elif len(seg) == 2:
            start, thrust = seg
            end = None
            mode = 0
        elif len(seg) == 3:
            start, end, thrust = seg
            mode = 0
        else:
            raise ValueError("Segment must be (start, thrust) or (start, end, thrust)")
        start_idx = _profile_bound(start, t_arr, mode, 0)
        end_idx = _profile_bound(end, t_arr, mode, n)
        if start_idx > end_idx or (start is not None and end is not None and mode != 0 and start > end):
            raise ValueError("segment end must not precede start")
        out[start_idx:end_idx] += _profile_value(thrust)

    return out


def _profile_samples(values, n):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a finite vector matching the time grid")
    return values.copy()


def _profile_value(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("profile acceleration must be finite")
    return value


def _profile_bound(value, times, mode, default):
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError("profile bounds must be finite numeric values, not booleans")
    is_index = mode == 1 or (mode == 0 and isinstance(value, (int, np.integer)))
    if is_index:
        if not isinstance(value, (int, np.integer)) or not 0 <= value <= len(times):
            raise ValueError("profile index must be an integer between 0 and the grid length")
        return int(value)
    return int(np.searchsorted(times, value))
