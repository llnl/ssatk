# ssatk/accelerations_orbit/accel_circularize.py

import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS
from ._state import state

_orbit_achieved = False  # global variable to track circularization


def reset_orbit_status():
    """Reset the internal latch so circularization thrust can be used again."""
    global _orbit_achieved
    _orbit_achieved = False


def accel_to_circular(r, v=None, t=None, *, thrust, tol=10.0, min_altitude=100e3):
    """
    Acceleration command that steers (r, v) toward a circular orbit at radius ``|r|``.

    Designed to be passed into leapfrog(..., accels=[...]) where leapfrog may call
    accelerations as f(r,v,t) (t is accepted but not required here).

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector (m).
    v : array_like, shape (3,)
        Velocity vector (m/s).
    t : optional
        Time (ignored; present for integrator compatibility).
    thrust : float
        Magnitude of available acceleration (m/s^2).
    tol : float, optional
        Convergence tolerance on ``||v_target - v||`` (m/s). Default 10.
    min_altitude : float, optional
        Minimum altitude above Earth's surface to allow control (m). Default 100 km.

    Returns
    -------
    a : ndarray, shape (3,)
        Acceleration command (m/s^2), or zeros if converged/disabled.
    """
    global _orbit_achieved

    if _orbit_achieved:
        return np.zeros(3, dtype=float)

    r, v, t = state(r, v, t)

    r_norm = np.linalg.norm(r)
    if r_norm == 0.0 or thrust == 0.0:
        return np.zeros(3, dtype=float)

    if r_norm < EARTH_RADIUS + float(min_altitude):
        return np.zeros(3, dtype=float)

    # Unit radial direction
    r_hat = r / r_norm

    # Tangential velocity component
    v_r = float(np.dot(v, r_hat))
    v_t_vec = v - v_r * r_hat
    v_t = np.linalg.norm(v_t_vec)

    if v_t > 0.0:
        t_hat = v_t_vec / v_t
    else:
        # Fallback if velocity is purely radial: pick a consistent tangential direction
        t_hat = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
        n = np.linalg.norm(t_hat)
        if n == 0.0:
            return np.zeros(3, dtype=float)
        t_hat /= n

    # Circular speed at current radius
    v_circ = np.sqrt(EARTH_MU / r_norm)

    # Desired circular velocity vector (purely tangential)
    v_target = v_circ * t_hat

    dv = v_target - v
    dv_norm = np.linalg.norm(dv)

    if dv_norm <= float(tol) or dv_norm == 0.0:
        _orbit_achieved = True
        return np.zeros(3, dtype=float)

    return float(thrust) * (dv / dv_norm)
