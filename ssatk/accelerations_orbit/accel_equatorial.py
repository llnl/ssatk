import numpy as np
from ._state import position


def accel_equatorial(r, v=None, magnitude=None):
    """
    Acceleration vector always in the equatorial (east/west) direction,
    i.e. perpendicular to the Earth's spin axis and radial direction,
    positive -> counter-clockwise (eastward), negative -> clockwise.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position (radial) vector in meters.
    v : array_like, shape (3,)
        (Unused; kept for signature consistency.)
    magnitude : float
        Desired magnitude of the equatorial-plane acceleration in m/s^2.

    Returns
    -------
    a : ndarray, shape (3,)
        Equatorial-plane acceleration vector in m/s^2.
    """
    if magnitude is None:
        magnitude = v
    r = position(r)

    if magnitude == 0:
        return np.zeros(3, dtype=float)

    norm_r = np.linalg.norm(r)
    if norm_r == 0:
        return np.zeros(3, dtype=float)

    r_hat = r / norm_r
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    equ_dir = np.cross(z_hat, r_hat)
    norm_eq = np.linalg.norm(equ_dir)
    if norm_eq == 0:
        return np.zeros(3, dtype=float)

    return float(magnitude) * equ_dir / norm_eq
