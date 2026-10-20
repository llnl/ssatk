# ssatk/accelerations_orbit/accel_inclination.py

import numpy as np
from ._state import position


def accel_inclination(r, v=None, magnitude=None):
    """
    Acceleration vector always in the local north/south direction,
    i.e. perpendicular to the radial direction but aimed toward
    the Earth's North pole (or South if magnitude < 0).

    Parameters
    ----------
    r : array_like, shape (3,)
        Position (radial) vector in meters.
    v : array_like, shape (3,)
        (Unused; kept for signature consistency.)
    magnitude : float
        Desired magnitude of the inclination-change acceleration in m/s^2.
        Positive -> northward; negative -> southward.

    Returns
    -------
    a : ndarray, shape (3,)
        Inclination-change acceleration vector in m/s^2.
    """
    if magnitude is None:
        magnitude = v
    r = position(r)
    norm_r = np.linalg.norm(r)
    if norm_r == 0.0 or magnitude == 0.0:
        return np.zeros(3, dtype=float)

    r_hat = r / norm_r
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    north_dir = z_hat - np.dot(z_hat, r_hat) * r_hat
    norm_nd = np.linalg.norm(north_dir)
    if norm_nd == 0.0:
        return np.zeros(3, dtype=float)

    return float(magnitude) * north_dir / norm_nd
