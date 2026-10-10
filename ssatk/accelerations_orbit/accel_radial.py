import numpy as np
from ._state import position


def accel_radial(r, magnitude):
    """
    Acceleration vector pointing radially outward from Earth's center
    with user-specified magnitude.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector in meters.
    magnitude : float
        Acceleration magnitude in m/s^2.

    Returns
    -------
    a : ndarray, shape (3,)
        Acceleration vector in m/s^2.

    Author: Travis Yeager
    """
    r = position(r)
    norm_r = np.linalg.norm(r)
    if norm_r == 0.0:
        return np.zeros(3)
    return magnitude * r / norm_r
