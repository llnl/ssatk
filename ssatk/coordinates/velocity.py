import numpy as np
from astropy.time import Time


def v_from_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate velocity from position using numerical differentiation.

    Uses three-point derivatives, including one-sided endpoints, with the
    actual time spacing (which may be nonuniform). Two samples use a secant.

    Parameters
    ----------
    r : np.ndarray
        Position array of shape (N, 3) in meters.
    t : np.ndarray or Time-like
        Time array corresponding to the positions.

    Returns
    -------
    np.ndarray
        Velocity array of shape (N, 3) in meters per second.
    """
    r = np.asarray(r, dtype=float)

    if isinstance(t, Time):
        t = t.gps
    t = np.asarray(t, dtype=float)

    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r must have shape (N, 3)")
    if t.ndim != 1:
        raise ValueError("t must be a 1D array")
    if len(r) != len(t):
        raise ValueError("r and t must have the same length")
    if len(r) < 2:
        raise ValueError("Need at least two samples to compute velocity")

    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be finite and strictly increasing")
    # Remove the constant origin before weighted sums, so a stationary state
    # does not acquire roundoff velocity (and an arbitrary heading).
    return np.gradient(r - r[0], t - t[0], axis=0, edge_order=2 if len(r) > 2 else 1)
