import numpy as np
from ._state import velocity

def accel_velocity(v, thrust_mag):
    """
    Thrust acceleration in the direction of velocity.

    Parameters
    ----------
    v : array_like, shape (3,)
        Velocity vector in m/s.
    thrust_mag : float
        Thrust magnitude in m/s^2.

    Returns
    -------
    a_thrust : ndarray, shape (3,)
        Thrust acceleration vector.

    Author: Travis Yeager
    """
    v = velocity(v)
    norm_v = np.linalg.norm(v)
    if norm_v == 0.0:
        return np.zeros(3)
    return thrust_mag * v / norm_v
