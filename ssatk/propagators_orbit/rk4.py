import numpy as np

from ..constants import EARTH_MU
from ..time_functions import to_gps

from ..accelerations_orbit.accel_moon import accel_point_moon
from ..accelerations_orbit.accel_sun import accel_point_sun
from ..accelerations_orbit.accel_radial import accel_radial
from ..accelerations_orbit.accel_velocity import accel_velocity
from ..accelerations_orbit.accel_inclination import accel_inclination

from .int_utils import build_profile


def rk4(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    accel_gravity=lambda r: -EARTH_MU * r / np.linalg.norm(r) ** 3,
):
    """
    Fixed-step RK4 propagation with point-mass gravity, lunar and solar
    third-body terms, and optional thrust profiles.

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Initial position (m) and velocity (m/s) in GCRF.
    t : array_like
        Time grid (anything ``to_gps`` accepts).
    radial, velocity, inclination : profile spec or None
        Thrust-acceleration profiles (m/s^2). See ``build_profile``.
    accel_gravity : callable
        ``f(r) -> (3,)`` central-body acceleration (m/s^2).

    Returns
    -------
    r, v : ndarray, shape (n, 3)
        State history on ``t``.

    Notes
    -----
    Two time bases are in play and they are not interchangeable.
    ``accel_point_moon`` and ``accel_point_sun`` resolve an ephemeris from
    the epoch they are handed, so they receive **absolute GPS seconds**.
    Thrust profiles are indexed against **elapsed seconds since ``t[0]``**,
    which is what ``build_profile`` has always been given here.
    """
    t_abs = np.asarray(to_gps(t), dtype=float)
    if t_abs.ndim != 1 or t_abs.size == 0 or not np.all(np.isfinite(t_abs)):
        raise ValueError("t must be a nonempty finite 1-D time grid")
    t_elapsed = t_abs - t_abs[0]
    n_steps = len(t_abs)

    r_th = build_profile(radial, t_elapsed)
    v_th = build_profile(velocity, t_elapsed)
    i_th = build_profile(inclination, t_elapsed)

    r = np.empty((n_steps, 3))
    v = np.empty((n_steps, 3))
    r[0] = np.asarray(r0, float)
    v[0] = np.asarray(v0, float)

    for i in range(n_steps - 1):
        dt = t_abs[i + 1] - t_abs[i]

        def a_total(r_i, v_i, t_i, i_thrust):
            # Callers may reuse a cached/read-only acceleration vector.
            a = np.array(accel_gravity(r_i), dtype=float, copy=True)

            # third-body point-mass terms (Earth-centered inertial)
            a += accel_point_moon(r_i, t_i)
            a += accel_point_sun(r_i, t_i)

            # thrust profiles
            a += accel_radial(r_i, r_th[i_thrust])
            a += accel_velocity(v_i, v_th[i_thrust])
            a += accel_inclination(r_i, v_i, i_th[i_thrust])
            return a

        # RK4 steps
        k1_v = a_total(r[i], v[i], t_abs[i], i)
        k1_r = v[i]

        k2_v = a_total(r[i] + 0.5 * dt * k1_r, v[i] + 0.5 * dt * k1_v, t_abs[i] + 0.5 * dt, i)
        k2_r = v[i] + 0.5 * dt * k1_v

        k3_v = a_total(r[i] + 0.5 * dt * k2_r, v[i] + 0.5 * dt * k2_v, t_abs[i] + 0.5 * dt, i)
        k3_r = v[i] + 0.5 * dt * k2_v

        k4_v = a_total(r[i] + dt * k3_r, v[i] + dt * k3_v, t_abs[i] + dt, i)
        k4_r = v[i] + dt * k3_v

        r[i + 1] = r[i] + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        v[i + 1] = v[i] + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return r, v
