# ssatk/orbital_mechanics/burn_to_deltav.py

import numpy as np
from ssatk.coordinates import ntw_to_gcrf
from ssatk.propagators_orbit import leapfrog

def burn_to_deltav(orbit, times, burn_ntw):
    """
    Compare continuous finite-duration NTW acceleration vs an instantaneous impulse
    applied at the sample of `times` nearest its mid-time (returned as t_center).

    Parameters
    ----------
    orbit : ssapy.Orbit
    times : 1D array of seconds (monotonic)
    burn_ntw : (3,) constant NTW acceleration [m/s^2], ordered [N, T, W]
               (canonical ssapy convention: N = T x W in-plane/radially outward,
               T = v_hat tangential, W = r x v out-of-plane).

    Returns
    -------
    dict: keys r_continuous, v_continuous, r_instantaneous,
    v_instantaneous, delta_v_ntw, delta_v_gcrf, t_center
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

    # Map canonical NTW [N, T, W] -> leapfrog thrust channels.
    #   N -> radial       (+r_hat)      leapfrog 'radial'
    #   T -> tangential   (+v_hat)      leapfrog 'velocity'
    #   W -> out-of-plane               leapfrog 'inclination'
    # This matches ntw_to_gcrf's [N, T, W] column order, so the continuous
    # (leapfrog) and impulsive (ntw_to_gcrf) branches use the same frame.
    # NOTE: leapfrog's radial channel is exactly +r_hat and its inclination
    # channel is local north; these coincide with canonical N and W only for a
    # circular, equatorial orbit. For inclined/eccentric orbits they differ by
    # the flight-path angle / inclination (a small-angle approximation on the
    # continuous branch; the impulsive branch via ntw_to_gcrf is exact).
    burn_ntw = np.asarray(burn_ntw, float)
    a_rad  = burn_ntw[0]   # N -> radial (+r_hat)
    a_tan  = burn_ntw[1]   # T -> tangential (+v_hat)
    a_incl = burn_ntw[2]   # W -> out-of-plane (local north)

    # Constant profiles across the window
    radial_prof      = a_rad
    velocity_prof    = a_tan
    inclination_prof = a_incl

    # Continuous thrust propagation over the window
    r_cont, v_cont = leapfrog(r0, v0, t,
                              radial=radial_prof,
                              velocity=velocity_prof,
                              inclination=inclination_prof)

    # Impulsive approximation at mid-time:
    # 1) Kepler-only to center
    # The impulse is applied at the sample nearest the mid-time; t_center is
    # that sample's time (the exact mid-time is not a grid point for even N).
    k = int(np.argmin(np.abs(t - 0.5 * (t[0] + t[-1]))))
    t_center = float(t[k])
    r_pre, v_pre = leapfrog(r0, v0, t[:k+1],
                            radial=None, velocity=None, inclination=None)
    r_c = r_pre[-1]
    v_c = v_pre[-1]

    # 2) Equivalent delta-v = a * duration, convert NTW->GCRF at center
    duration = t[-1] - t[0]
    delta_v_ntw = burn_ntw * duration
    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_c, v_c)

    # 3) Apply impulse and Kepler-only forward from center
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
    }
