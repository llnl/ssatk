# ssatk/propagators_orbit/leap_frog.py

import numpy as np
from scipy.optimize import brentq

from ..constants import EARTH_RADIUS
from ..time_functions import to_gps

from .int_utils import acceleration_adapter, build_profile  # thrust-profile helper

from ..accelerations_orbit.accel_point_earth import accel_point_earth
from ..accelerations_orbit.accel_radial import accel_radial
from ..accelerations_orbit.accel_velocity import accel_velocity
from ..accelerations_orbit.accel_inclination import accel_inclination


def leapfrog(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    *,
    accels=None,
    stop_altitude_m=100e3,
    verbose=False,
    return_times=False,
):
    """
    Kick-drift-kick integration with point-mass Earth gravity and extra forces.
    The symplectic guarantee applies only to conservative position-only forces;
    prefer the adaptive propagator for velocity-dependent forces.

    Parameters
    ----------
    r0, v0 : array-like (3,)
        Initial position [m] and velocity [m/s] in an inertial frame.
    t : array-like
        Time grid (anything `to_gps` can handle). Must be evenly spaced.
    radial, velocity, inclination : profile spec or None
        Thrust-acceleration profiles (m/s^2). See `build_profile` [104].
        - radial      uses accel_radial(r, magnitude) [65]
        - velocity    uses accel_velocity(v, thrust_mag) [68]
        - inclination uses accel_inclination(r, v, magnitude) [61]
    accels : callable or list[callable] or None
        Optional additional acceleration models to add each step.
        Each function may have signature f(r), f(r,t), f(r,v), or f(r,v,t)
        and must return a (3,) acceleration vector [m/s^2]. ``t`` is the
        absolute GPS epoch, not seconds since ``t[0]``, so ephemeris-backed
        models such as ``accel_point_moon`` can be passed directly.
    stop_altitude_m : float
        Stop integration if ||r|| < EARTH_RADIUS + stop_altitude_m [104].
    verbose : bool
        Print impact message.
    return_times : bool
        Return ``(r, v, times)`` with absolute GPS epochs. This includes the
        synchronized state at a detected sub-step impact. The default ``(r, v)``
        returns only requested grid samples before impact, preserving alignment
        with ``t[:len(r)]``. An initial state already below the limit is retained.

    Returns
    -------
    r, v : ndarray (n,3)
        State history before impact, or full length. See ``return_times``.
    """
    # ---- time arrays ----
    # Absolute GPS epochs go to the ``accels`` callbacks, which may resolve an
    # ephemeris from them; elapsed seconds key the thrust profiles, which is
    # what build_profile has always been given here.
    t_abs = np.array(to_gps(t), dtype=float, copy=True)
    if t_abs.ndim != 1 or not np.all(np.isfinite(t_abs)):
        raise ValueError("t must be a finite 1-D time grid")
    n_steps = len(t_abs)

    if n_steps < 2:
        raise ValueError("t must contain at least 2 time samples")

    t_elapsed = t_abs - t_abs[0]

    dt_vals = np.diff(t_abs)
    if np.any(dt_vals <= 0):
        raise ValueError("t must be strictly increasing")
    if not np.allclose(dt_vals, dt_vals[0], rtol=0.0,
                       atol=4 * np.spacing(max(1.0, np.max(np.abs(t_abs))))):
        raise ValueError("Non-uniform Δt not supported")

    # ---- burn profiles ----
    r_th = build_profile(radial,      t_elapsed)
    v_th = build_profile(velocity,    t_elapsed)
    i_th = build_profile(inclination, t_elapsed)

    # ---- normalize accels -> list ----
    if accels is None:
        accel_list = []
    elif callable(accels):
        accel_list = [acceleration_adapter(accels)]
    else:
        accel_list = [acceleration_adapter(model) for model in accels]

    def _eval_extra_accels(r_i, v_i, t_i):
        """Sum extra accelerations, supporting several common call signatures."""
        if not accel_list:
            return np.zeros(3, dtype=float)

        a = np.zeros(3, dtype=float)
        for f in accel_list:
            value = np.asarray(f(t_i, r_i, v_i), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("acceleration must be a finite 3-vector")
            a += value

        return a

    # ---- state arrays ----
    r = np.empty((n_steps, 3), dtype=float)
    v = np.empty((n_steps, 3), dtype=float)
    r[0] = np.asarray(r0, dtype=float).reshape(3)
    v[0] = np.asarray(v0, dtype=float).reshape(3)
    if not np.all(np.isfinite(r[0])) or not np.all(np.isfinite(v[0])):
        raise ValueError("initial position and velocity must be finite")

    # ---- leap-frog loop ----
    r_stop = float(EARTH_RADIUS + stop_altitude_m)
    if np.isnan(r_stop):
        raise ValueError("stop_altitude_m must not be NaN")
    # A non-positive stopping radius disables the check: ``||r|| >= 0`` can
    # never fall below it. Callers pass a large negative altitude to integrate
    # without an impact surface, so keep that contract and keep the sub-step
    # search, which scales the position polynomial by the radius, unentered.
    check_impact = r_stop > 0

    def result(count):
        return (r[:count], v[:count], t_abs[:count]) if return_times else (r[:count], v[:count])

    for i in range(n_steps - 1):
        # Uniform decimal steps can differ by an epoch ULP after conversion.
        # Use the represented interval so states remain aligned with each row.
        dt = float(dt_vals[i])
        if check_impact and np.linalg.norm(r[i]) < r_stop:
            if verbose:
                print(f"Impact at step {i}, t = {t_elapsed[i]:.2f} s")
            return result(i + 1)

        # first half-kick
        a0 = (
            accel_point_earth(r[i])
            + accel_radial(r[i],            r_th[i])
            + accel_velocity(v[i],          v_th[i])
            + accel_inclination(r[i], v[i], i_th[i])
            + _eval_extra_accels(r[i], v[i], t_abs[i])
        )
        v_half = v[i] + 0.5 * dt * a0

        crossing = (_first_radius_crossing(r[i], v[i], a0, dt, r_stop)
                    if check_impact else None)
        if crossing is not None:
            if verbose:
                print(f"Impact between steps {i} and {i + 1}")
            if not return_times:
                return result(i + 1)
            h = crossing * dt
            half = v[i] + 0.5 * h * a0
            impact_r = r[i] + h * half
            impact_t = t_abs[i] + h
            a_end = (accel_point_earth(impact_r)
                     + accel_radial(impact_r, r_th[i])
                     + accel_velocity(half, v_th[i])
                     + accel_inclination(impact_r, half, i_th[i])
                     + _eval_extra_accels(impact_r, half, impact_t))
            impact_v = half + 0.5 * h * a_end
            if impact_t == t_abs[i]:
                return result(i + 1)
            r[i + 1], v[i + 1], t_abs[i + 1] = impact_r, impact_v, impact_t
            return result(i + 2)

        # drift
        r[i + 1] = r[i] + dt * v_half

        # second half-kick
        a1 = (
            accel_point_earth(r[i + 1])
            + accel_radial(r[i + 1],            r_th[i + 1])
            + accel_velocity(v_half,            v_th[i + 1])
            + accel_inclination(r[i + 1], v_half, i_th[i + 1])
            + _eval_extra_accels(r[i + 1], v_half, t_abs[i + 1])
        )
        v[i + 1] = v_half + 0.5 * dt * a1

    return result(n_steps)


def _first_radius_crossing(r, v, a, dt, radius):
    """First sphere contact on the kick-drift position curve for 0 <= h <= dt.

    Splitting at stationary radii also finds enter/exit pairs with both end
    samples outside. Scale by the radius to condition the polynomial, then
    bracket the first root rather than evaluating a cancellation-prone formula.
    The event accuracy is limited by the fixed-step position approximation.
    """
    p, c, d = r / radius, dt * v / radius, 0.5 * dt**2 * a / radius
    derivative = [np.dot(p, c), np.dot(c, c) + 2 * np.dot(p, d),
                  3 * np.dot(c, d), 2 * np.dot(d, d)]
    roots = np.polynomial.polynomial.polyroots(derivative)
    interior = sorted(float(root.real) for root in roots
                      if abs(root.imag) < 1e-12 and 0 < root.real < 1)

    def distance(s):
        return np.linalg.norm(p + s * c + s**2 * d) - 1

    left = 0.0
    for right in [*interior, 1.0]:
        if distance(right) <= 0:
            return float(brentq(distance, left, right, xtol=1e-14))
        left = right
    return None
