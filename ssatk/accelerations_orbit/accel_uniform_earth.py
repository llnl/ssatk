# ssatk/accelerations_orbit/accel_uniform_earth.py

import numpy as np
from ..constants import EARTH_RADIUS, EARTH_MU
from ._state import position


def accel_uniform_earth(r):
    """
    Piecewise gravity model:
    - Outside Earth: point-mass gravity ~ -mu r / norm(r)^3
    - Inside Earth (uniform-density sphere): linear gravity ~ -mu r / R^3

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector (m).

    Returns
    -------
    a : ndarray, shape (3,)
        Acceleration (m/s^2).
    """
    x, y, z = position(r)
    r_mag = np.sqrt(x**2 + y**2 + z**2)

    if r_mag < EARTH_RADIUS:
        factor = -EARTH_MU / (EARTH_RADIUS**3)
    else:
        factor = -EARTH_MU / (r_mag**3)

    return np.array([factor * x, factor * y, factor * z])
