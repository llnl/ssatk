# ssatk/orbital_mechanics/deltav_to_burn.py

import numpy as np
from ssatk.coordinates import ntw_to_gcrf
from ssatk.propagators_orbit import leapfrog

def deltav_to_burn(orbit, times, delta_v_ntw):
    """
    Use a constant NTW acceleration across `times` such that a*duration = delta_v_ntw,
    and compare continuous vs impulsive results.

    Parameters
    ----------
    orbit : ssapy.Orbit
    times : 1D array of seconds (monotonic)
    delta_v_ntw : (3,) NTW delta-v [m/s], ordered [N, T, W]
                  (canonical ssapy convention: N = T x W in-plane/radially
                  outward, T = v_hat tangential, W = r x v out-of-plane).

    Returns
    -------
    dict: keys r_continuous, v_continuous, r_instantaneous,
    v_instantaneous, delta_v_ntw, delta_v_gcrf, t_center, a_ntw
    """
    r0 = np.asarray(orbit.r, float)
    v0 = np.asarray(orbit.v, float)
    t0 = float(orbit.t)

    t = np.asarray(times, float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a 1D array with at least 2 samples")

    # Align start time with orbit.t if needed
    if not np.isclose(t[0], t0, atol=1e-6):
        t = t - (t[0] - t0)

    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError("times must span a positive duration")

    delta_v_ntw = np.asarray(delta_v_ntw, float)
    a_ntw = delta_v_ntw / duration

    # Map canonical NTW [N, T, W] -> leapfrog thrust channels (see burn_to_deltav
    # for the exact/approximate caveats):
    #   N -> radial (+r_hat), T -> tangential (+v_hat), W -> out-of-plane (north).
    a_rad  = a_ntw[0]   # N -> radial (+r_hat)
    a_tan  = a_ntw[1]   # T -> tangential (+v_hat)
    a_incl = a_ntw[2]   # W -> out-of-plane (local north)

    radial_prof      = a_rad
    velocity_prof    = a_tan
    inclination_prof = a_incl

    # Continuous thrust
    r_cont, v_cont = leapfrog(r0, v0, t,
                              radial=radial_prof,
                              velocity=velocity_prof,
                              inclination=inclination_prof)

    # Impulsive approximation at mid-time
    # The impulse is applied at the sample nearest the mid-time; t_center is
    # that sample's time (the exact mid-time is not a grid point for even N).
    k = int(np.argmin(np.abs(t - 0.5 * (t[0] + t[-1]))))
    t_center = float(t[k])

    r_pre, v_pre = leapfrog(r0, v0, t[:k+1],
                            radial=None, velocity=None, inclination=None)
    r_c = r_pre[-1]
    v_c = v_pre[-1]

    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_c, v_c)

    v_after = v_c + delta_v_gcrf
    r_post, v_post = leapfrog(r_c, v_after, t[k:],
                              radial=None, velocity=None, inclination=None)

    r_inst = np.vstack([r_pre[:-1], r_post])
    v_inst = np.vstack([v_pre[:-1], v_post])

    return {
        "r_continuous":     r_cont,
        "v_continuous":     v_cont,
        "r_instantaneous":  r_inst,
        "v_instantaneous":  v_inst,
        "delta_v_ntw":      delta_v_ntw,
        "delta_v_gcrf":     delta_v_gcrf,
        "t_center":         t_center,
        "a_ntw":            a_ntw,
    }
