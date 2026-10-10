import numpy as np
from astropy.time import Time


def v_from_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate velocity from position using numerical differentiation.

    Uses centered differences for interior points and one-sided differences
    at the endpoints.

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

    if isinstance(t[0], Time):
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

    v = np.zeros_like(r, dtype=float)

    # Forward difference at the first point
    v[0] = (r[1] - r[0]) / (t[1] - t[0])

    # Centered differences for interior points
    if len(r) > 2:
        v[1:-1] = (r[2:] - r[:-2]) / (t[2:] - t[:-2])[:, np.newaxis]

    # Backward difference at the last point
    v[-1] = (r[-1] - r[-2]) / (t[-1] - t[-2])

    return v
