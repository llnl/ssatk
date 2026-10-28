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
        Velocity vector in meters per second; a scalar is accepted as
        magnitude shorthand in direct calls.
    magnitude : float
        Desired acceleration magnitude in m/s^2.
        Positive → along the direction of motion;
        negative → opposite direction.
        Required when composing the helper with ``accel_add``.

    Returns
    -------
    a : ndarray, shape (3,)
        Tangential acceleration vector in m/s^2.
    """
    if magnitude is None:
        if v is not None and np.ndim(v) == 0:
            # Support the direct scalar magnitude shorthand.
            magnitude = v
            v = None
        else:
            raise TypeError(
                "accel_plane() requires magnitude; bind it (e.g. via "
                "functools.partial(accel_plane, magnitude=...)) before "
                "passing this model to accel_add"
            )
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
