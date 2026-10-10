"""Mean Exponential Growth factor of Nearby Orbits (MEGNO)."""

import numpy as np


def megno(t, delta, *, return_series: bool = False):
    """
    Mean MEGNO of a tangent (variational) vector sampled along a trajectory.

    MEGNO (Cincotta & Simo 2000) is ``Y(t) = (2/t) int_0^t s (d ln delta/ds) ds``
    and its running mean is ``<Y>(t) = (1/t) int_0^t Y(s) ds``, with ``delta``
    the norm of a deviation vector evolved by the variational equations. For a
    regular (quasi-periodic) orbit ``<Y> -> 2``; for a chaotic orbit with
    largest Lyapunov exponent ``lambda``, ``<Y> -> lambda t / 2``.

    Integrating by parts, ``Y(t) = 2 [ln delta(t) - (1/t) int_0^t ln delta ds]``,
    which needs no derivative of the samples; both integrals use the trapezoid
    rule on the supplied samples, with time measured from ``t[0]``.

    Parameters
    ----------
    t : array-like, shape (N,)
        Strictly increasing sample times.
    delta : array-like, shape (N,) or (N, d)
        Deviation-vector norms, or the deviation vectors themselves (their
        row norms are used). All norms must be positive.
    return_series : bool, optional
        If True, also return the ``Y`` and ``<Y>`` series.

    Returns
    -------
    float or tuple
        ``<Y>`` at the last sample, or ``(<Y>_end, Y, Y_mean)`` with
        ``return_series=True``. ``Y`` and ``Y_mean`` are NaN at ``t[0]``.

    Notes
    -----
    The previous ``megno(r)`` perturbed positions with random noise and
    averaged the log of that noise, so its value had no relation to the
    orbit. The deviation vector has to come from the variational equations
    (e.g. REBOUND ``add_variation``) or from a closely neighbouring trajectory
    renormalised along the way.
    """
    t = np.asarray(t, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 2:
        delta = np.linalg.norm(delta, axis=1)
    if t.ndim != 1 or delta.shape != t.shape or t.size < 3:
        raise ValueError("t and delta must have the same length, at least 3.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing.")
    if np.any(delta <= 0.0) or not np.all(np.isfinite(delta)):
        raise ValueError("delta norms must be positive and finite.")

    s = t - t[0]
    log_delta = np.log(delta)
    steps = np.diff(s)
    cumulative_log = np.concatenate([[0.0], np.cumsum(0.5 * (log_delta[1:] + log_delta[:-1]) * steps)])

    y = np.full_like(s, np.nan)
    y[1:] = 2.0 * (log_delta[1:] - log_delta[0] - (cumulative_log[1:] - log_delta[0] * s[1:]) / s[1:])
    y_filled = np.where(np.isnan(y), 0.0, y)
    cumulative_y = np.concatenate([[0.0], np.cumsum(0.5 * (y_filled[1:] + y_filled[:-1]) * steps)])
    y_mean = np.full_like(s, np.nan)
    y_mean[1:] = cumulative_y[1:] / s[1:]

    if return_series:
        return float(y_mean[-1]), y, y_mean
    return float(y_mean[-1])
