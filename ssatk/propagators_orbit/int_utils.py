# ssatk/propagators_orbit/int_utils.py

import inspect

import numpy as np
from scipy.interpolate import interp1d

from ..time_functions import to_gps


def model_time_breakpoints(models, start, end):
    """Collect declared force/profile boundaries through existing model adapters."""
    points, seen = [], set()
    pending = list(models)
    while pending:
        model = pending.pop()
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        values = getattr(model, "time_breakpoints", ())
        values = values(start, end) if callable(values) else values
        points.extend(np.asarray(values, dtype=float).reshape(-1))
        for name in ("_source_model", "thrust", "throttle"):
            pending.append(getattr(model, name, None))
        for name in ("_models", "accels"):
            pending.extend(getattr(model, name, ()))
    points = np.asarray(points)
    if np.any(np.isnan(points)):
        raise ValueError("model time_breakpoints cannot contain NaN.")
    return np.unique(points[(points > start) & (points < end)])


def solve_ivp_at_breakpoints(fun, t_span, y0, *, breakpoints=(), time_origin=0.0, **options):
    """Run SciPy on consecutive intervals, preserving samples and event results."""
    from scipy.integrate import OdeSolution, solve_ivp

    points = np.asarray(breakpoints, dtype=float)
    points = np.unique(points[(points > t_span[0]) & (points < t_span[1])])
    if not len(points):
        return solve_ivp(fun, t_span, y0, **options)
    requested = options.pop("t_eval", None)
    dense_output = options.pop("dense_output", False)
    solutions = []
    boundaries = np.concatenate(([t_span[0]], points, [t_span[1]]))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        kwargs = dict(options)
        if kwargs.get("first_step") is not None:
            kwargs["first_step"] = min(kwargs["first_step"], right - left)
        # At a discontinuity use the limit inside this interval on both ends.
        inside_left = np.nextafter(time_origin + left, time_origin + right) - time_origin
        inside_right = np.nextafter(time_origin + right, time_origin + left) - time_origin
        def interval_rhs(t, y):
            epoch = np.clip(t, inside_left, inside_right)
            return fun(epoch, y)
        result = solve_ivp(interval_rhs, (left, right), y0, dense_output=True, **kwargs)
        solutions.append(result)
        y0 = result.y[:, -1]
        if result.status != 0:
            break
    dense = OdeSolution(np.concatenate([solutions[0].sol.ts] + [s.sol.ts[1:] for s in solutions[1:]]),
                        [interpolant for s in solutions for interpolant in s.sol.interpolants])
    result = solutions[-1]
    end = result.t[-1]
    if requested is None:
        times = np.concatenate([solutions[0].t] + [s.t[1:] for s in solutions[1:]])
    else:
        requested = np.asarray(requested)
        times = requested[requested <= end]
    if result.t_events is not None:
        event_times, event_states = [], []
        for index in range(len(result.t_events)):
            times_for_event = np.concatenate([s.t_events[index] for s in solutions])
            states_for_event = np.concatenate([np.asarray(s.y_events[index]).reshape(-1, len(y0)) for s in solutions], axis=0)
            unique_times, indices = np.unique(times_for_event, return_index=True)
            event_times.append(unique_times)
            event_states.append(states_for_event[indices])
        result.t_events, result.y_events = event_times, event_states
    result.nfev = sum(s.nfev for s in solutions)
    result.t, result.y = times, dense(times) if len(times) else np.empty((len(y0), 0))
    result.sol = dense if dense_output else None
    return result


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
        spacecraft_model._source_model = model
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
    evaluate._source_model = model
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


def precompute_third_body_positions_exact(t, body_name):
    """Return exact third-body positions evaluated at the supplied epochs.

    Unlike :func:`precompute_third_body_positions`, this helper deliberately
    does not interpolate.  It batches the Astropy ephemeris query while
    retaining the same GCRS conversion used by the point-mass accelerations.
    """
    from astropy.coordinates import GCRS, get_body
    from astropy.time import Time
    from astropy import units as u

    t_gps = np.asarray(to_gps(t), dtype=float)
    if t_gps.ndim != 1 or not np.all(np.isfinite(t_gps)):
        raise ValueError("ephemeris epochs must be a finite one-dimensional array")
    epochs = Time(t_gps, format="gps", scale="utc")
    position = get_body(body_name, epochs).transform_to(GCRS(obstime=epochs))
    result = position.cartesian.xyz.to(u.m).value.T
    if result.shape != (len(t_gps), 3) or not np.all(np.isfinite(result)):
        raise ValueError("body positions must be finite with shape (n, 3)")
    return result


def third_body_acceleration(r, body_position, mu):
    """Evaluate the shared Earth-centered point-mass force lazily."""
    from ..accelerations_6dof.spacecraft import (
        third_body_acceleration as shared_third_body_acceleration,
    )

    return shared_third_body_acceleration(r, body_position, mu)


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
