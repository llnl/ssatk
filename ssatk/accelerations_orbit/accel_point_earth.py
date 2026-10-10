import numpy as np
from ..constants import EARTH_MU
from ._state import position


def accel_point_earth(r):
    r = position(r)
    x, y, z = r
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    if r_mag == 0.0:
        return np.zeros(3, dtype=float)
    factor = -EARTH_MU / (r_mag**3)
    return np.array([factor * x, factor * y, factor * z])
