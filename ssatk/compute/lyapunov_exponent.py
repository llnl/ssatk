import numpy as np


def lyapunov_exponent_from_statevectors(
    r,
    v,
    dt=1.0,
    theiler_window=10,
    max_horizon=None,
    fit_window=(0.1, 0.6),
    min_initial_separation=1e-12,
    trim_percentile=None,
    return_diagnostics=True,
):
    """
    Estimate the (largest) Lyapunov exponent from a single trajectory of state vectors
    using the Rosenstein et al. nearest-neighbor divergence method.

    Parameters
    ----------
    r : (N,3) array
        Positions.
    v : (N,3) array
        Velocities.
    dt : float
        Time between samples (in whatever time unit you want the exponent to be 1/unit).
    theiler_window : int
        Exclude nearest neighbors within +/- this many samples to avoid trivial temporal neighbors.
    max_horizon : int or None
        Max number of forward steps k to track divergence. If None, uses as much as possible.
    fit_window : tuple
        Either (t_min_frac, t_max_frac) with 0 <= t_min_frac < t_max_frac <= 1,
        as fractions of the available time range, OR (t_min, t_max) in the same
        time units as dt otherwise.
    min_initial_separation : float
        Pairs with initial separation below this are discarded (avoids log(0) and numerical issues).
    trim_percentile : float or None
        If set (e.g., 90), trims per-k separations above that percentile before averaging
        (can reduce the impact of outliers).
    return_diagnostics : bool
        If True, returns (lambda, t, mean_log_div, diagnostics_dict).
        If False, returns (lambda, t, mean_log_div).

    Returns
    -------
    lle : float
        Estimated largest Lyapunov exponent.
    t : (K,) array
        Time values for the divergence curve.
    mean_log_div : (K,) array
        Mean log separation vs time.
    diagnostics : dict (optional)
        Contains fit indices, intercept, r2, number of pairs used, neighbor indices, etc.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if r.ndim != 2 or v.ndim != 2 or r.shape[1] != 3 or v.shape[1] != 3:
        raise ValueError("r and v must be shaped (N,3).")
    if r.shape[0] != v.shape[0]:
        raise ValueError("r and v must have the same length.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if theiler_window < 0:
        raise ValueError("theiler_window must be >= 0.")

    X = np.concatenate([r, v], axis=1)  # (N,6)
    N = X.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 samples.")

    # --- Find nearest neighbor for each point with a Theiler window ---
    nn = np.full(N, -1, dtype=int)
    nn_dist = np.full(N, np.nan, dtype=float)

    # O(N^2) but memory-light, pure numpy
    for i in range(N):
        d = X - X[i]  # (N,6)
        dist2 = np.einsum("ij,ij->i", d, d)  # squared distance
        dist2[i] = np.inf

        if theiler_window > 0:
            lo = max(0, i - theiler_window)
            hi = min(N, i + theiler_window + 1)
            dist2[lo:hi] = np.inf

        j = int(np.argmin(dist2))
        if np.isfinite(dist2[j]):
            nn[i] = j
            nn_dist[i] = np.sqrt(dist2[j])

    # Keep only valid pairs with a safe initial separation
    valid_i = np.where((nn >= 0) & np.isfinite(nn_dist) & (nn_dist >= min_initial_separation))[0]
    if valid_i.size == 0:
        raise ValueError("No valid neighbor pairs found. Try lowering theiler_window or min_initial_separation.")

    # Horizon K: how far forward we can track for all (i, nn[i])
    if max_horizon is None:
        K = np.max(np.minimum(N - 1 - valid_i, N - 1 - nn[valid_i])) + 1
    else:
        K = int(max_horizon)
        K = max(1, min(K, np.max(np.minimum(N - 1 - valid_i, N - 1 - nn[valid_i])) + 1))

    # --- Build mean log-divergence curve ---
    mean_log_div = np.full(K, np.nan, dtype=float)
    counts = np.zeros(K, dtype=int)

    for k in range(K):
        i_k = valid_i
        j_k = nn[valid_i]

        ok = (i_k + k < N) & (j_k + k < N)
        i2 = i_k[ok] + k
        j2 = j_k[ok] + k
        if i2.size == 0:
            continue

        d = X[i2] - X[j2]
        sep = np.sqrt(np.einsum("ij,ij->i", d, d))

        # Optional trimming of outliers per k
        if trim_percentile is not None:
            p = float(trim_percentile)
            if 0 < p < 100 and sep.size >= 5:
                cutoff = np.percentile(sep, p)
                sep = sep[sep <= cutoff]

        # Avoid log(0)
        sep = sep[sep > 0]
        if sep.size == 0:
            continue

        mean_log_div[k] = np.mean(np.log(sep))
        counts[k] = sep.size

    t = np.arange(K, dtype=float) * dt

    # --- Choose fit region ---
    finite = np.isfinite(mean_log_div)
    if np.count_nonzero(finite) < 3:
        raise ValueError("Not enough finite points in divergence curve to fit a slope.")

    t_f = t[finite]
    y_f = mean_log_div[finite]

    fw0, fw1 = fit_window
    # Values in [0, 1] are fractions of the available range, as documented; a
    # window starting at 0 (e.g. (0.0, 0.1)) is a fraction too.
    if (0.0 <= fw0 < fw1 <= 1.0):
        tmin = t_f.min() + fw0 * (t_f.max() - t_f.min())
        tmax = t_f.min() + fw1 * (t_f.max() - t_f.min())
    else:
        tmin = float(fw0)
        tmax = float(fw1)

    fit_mask = (t_f >= tmin) & (t_f <= tmax)
    if np.count_nonzero(fit_mask) < 3:
        # fallback: fit the first ~30% of available finite points (often the most linear)
        m = max(3, int(0.3 * t_f.size))
        fit_mask = np.zeros_like(t_f, dtype=bool)
        fit_mask[:m] = True

    tt = t_f[fit_mask]
    yy = y_f[fit_mask]

    # Linear fit: yy = a*tt + b
    a, b = np.polyfit(tt, yy, 1)

    # R^2 for diagnostics
    yhat = a * tt + b
    ss_res = np.sum((yy - yhat) ** 2)
    ss_tot = np.sum((yy - np.mean(yy)) ** 2) if yy.size > 1 else np.nan
    r2 = 1.0 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) and ss_tot > 0 else np.nan

    lle = a  # slope is the LLE in 1/dt units

    if not return_diagnostics:
        return lle, t, mean_log_div

    diagnostics = {
        "intercept": b,
        "r2": r2,
        "K": K,
        "num_pairs": int(valid_i.size),
        "counts_per_k": counts,
        "neighbor_index": nn,
        "neighbor_initial_distance": nn_dist,
        "fit_tmin": tmin,
        "fit_tmax": tmax,
        "fit_points": int(tt.size),
    }
    return lle, t, mean_log_div, diagnostics
