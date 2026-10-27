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
        Required: there is no fallback to ``v``. ``accel_add`` maps a
        parameter named ``v`` to the real velocity vector, so a fallback
        here would silently turn that vector into ``magnitude``.

    Returns
    -------
    a : ndarray, shape (3,)
        Equatorial-plane acceleration vector in m/s^2.
    """
    if magnitude is None:
        if v is not None and np.ndim(v) == 0:
            # Preserve the historical accel_equatorial(r, magnitude) form.
            magnitude = v
        else:
            raise TypeError(
                "accel_equatorial() requires magnitude; bind it (e.g. via "
                "functools.partial(accel_equatorial, magnitude=...)) before "
                "passing this model to accel_add"
            )
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
