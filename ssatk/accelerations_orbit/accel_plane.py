# ssatk/accelerations_orbit/accel_plane.py

import numpy as np
from ._state import state


def accel_plane(r, v=None, magnitude=None):
    """
    Tangential acceleration in the orbital plane, perpendicular to r̂.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector in meters.
    v : array_like, shape (3,)
        Velocity vector in meters per second.
    magnitude : float
        Desired acceleration magnitude in m/s^2.
        Positive → along the direction of motion;
        negative → opposite direction.

    Returns
    -------
    a : ndarray, shape (3,)
        Tangential acceleration vector in m/s^2.
    """
    if magnitude is None:
        magnitude = v
        v = None
    r, v, _ = state(r, v)

    if magnitude == 0:
        return np.zeros(3, dtype=float)

    norm_r = np.linalg.norm(r)
    if norm_r == 0:
        return np.zeros(3, dtype=float)

    r_hat = r / norm_r

    # Remove radial component of v to get purely tangential direction
    tangential = v - np.dot(v, r_hat) * r_hat
    norm_tan = np.linalg.norm(tangential)
    if norm_tan == 0:
        return np.zeros(3, dtype=float)

    return float(magnitude) * tangential / norm_tan
