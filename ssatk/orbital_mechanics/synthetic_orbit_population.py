"""
synthetic_orbit_population.py

Generate a population of synthetic SSAPy orbits by sampling (upsampling) the full
set of classical Keplerian elements, then propagating each orbit to produce
time-series r(t), v(t).

All units are SI:
  - a in meters (m)
  - r in meters (m)
  - v in meters/second (m/s)
  - t in seconds (s) [interpreted by SSAPy as GPS seconds if you care about epochs]
  - mu in m^3/s^2
Angles are radians.

Why Orbit.fromKeplerianElements?
--------------------------------
SSAPy Orbit() is constructed from Cartesian state (r, v, t). For Keplerian
initialization you must use:
    Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t, mu=...)
See the SSAPy orbit module documentation.

Returned time grid:
-------------------
This function returns t_list entries in seconds from t_start (default 0.0).
Internally we pass absolute times (t_start + t) to SSAPy propagation. If you
want a real GPS epoch, pass t_start as GPS seconds (float) or an astropy Time.
"""

import numpy as np

from ssapy import Orbit
from ssapy.propagator import KeplerianPropagator

from ..constants import EARTH_MU


def synthetic_orbit_population(
    *,
    M=40,
    N=7200,
    dt=1.0,
    t_start=0.0,
    mu=EARTH_MU,
    # Nominal (reference) Keplerian elements
    a0=7_000_000.0,                 # m
    e0=1.0e-3,                      # -
    i0=np.deg2rad(55.0),            # rad
    raan0=np.deg2rad(40.0),         # rad
    pa0=np.deg2rad(10.0),           # rad (argument of periapsis)
    nu0=np.deg2rad(0.0),            # rad (true anomaly)
    # 1-sigma spreads (normal) or half-widths (uniform)
    a_sigma=200.0,                  # m
    e_sigma=2.0e-4,                 # -
    i_sigma=np.deg2rad(0.10),       # rad
    raan_sigma=np.deg2rad(0.20),    # rad
    pa_sigma=np.deg2rad(0.20),      # rad
    nu_sigma=np.deg2rad(0.30),      # rad
    distribution="normal",          # "normal" | "uniform"
    include_nominal=True,
    seed=1,
    propagator=None,                # optional SSAPy propagator instance
    e_clip=(0.0, 0.99),
):
    """
    Create a synthetic ensemble (population) of SSAPy Orbit objects and sampled
    r(t), v(t) time series by perturbing all 6 classical Keplerian elements.

    Parameters
    ----------
    M : int
        Number of orbits to generate.
    N : int
        Number of time samples per orbit.
    dt : float
        Time step in seconds.
    t_start : float or astropy.time.Time
        Start epoch passed to SSAPy. If float, interpreted by SSAPy as GPS seconds.
        The returned t_list is always seconds-from-start (0..(N-1)dt).
    mu : float
        Gravitational parameter in m^3/s^2. Default EARTH_MU.
    a0, e0, i0, raan0, pa0, nu0 : float
        Nominal Keplerian elements (SI + radians).
    *_sigma : float
        Perturbation sizes. If distribution="normal", these are 1-sigma standard
        deviations. If "uniform", these are half-widths.
    distribution : {"normal","uniform"}
        Perturbation distribution for each element.
    include_nominal : bool
        If True, orbit 0 is exactly the nominal orbit, and the remaining M-1 are perturbed.
        If False, all M are perturbed.
    seed : int
        RNG seed.
    propagator : ssapy.propagator.Propagator or None
        Propagator used by Orbit.at(). If None, uses KeplerianPropagator().
    e_clip : (float, float)
        Clip eccentricity into [min, max] to avoid invalid values.

    Returns
    -------
    orbits : list[ssapy.Orbit]
        List of length M of initial Orbit objects at t_start.
    r_list : list[np.ndarray]
        List of length M; each entry shape (N,3) in meters.
    v_list : list[np.ndarray]
        List of length M; each entry shape (N,3) in m/s.
    t_list : list[np.ndarray]
        List of length M; each entry shape (N,) in seconds-from-start.
    mu : float
        Returned for convenience (m^3/s^2).
    """
    if M < 1:
        raise ValueError("M must be >= 1.")
    if N < 2:
        raise ValueError("N must be >= 2.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if distribution not in ("normal", "uniform"):
        raise ValueError('distribution must be "normal" or "uniform".')

    rng = np.random.default_rng(seed)
    prop = KeplerianPropagator() if propagator is None else propagator

    # Returned time base is seconds-from-start (SI)
    t_rel = np.arange(N, dtype=float) * float(dt)

    # SSAPy propagation uses absolute times; for float epochs this is just t_start + t_rel
    # (and if you want true GPS seconds, pass t_start accordingly).
    if np.isscalar(t_start):
        t_abs = float(t_start) + t_rel
    else:
        # astropy Time supports addition with seconds via TimeDelta, but we avoid
        # importing astropy here; SSAPy will accept Time, and Orbit.at uses it.
        # We still build a float grid for outputs; for at(), pass t_start + seconds
        # via numpy array and let astropy handle if user supplied Time.
        t_abs = t_start + t_rel  # relies on astropy Time behavior if t_start is Time

    def _draw(scale, size):
        if distribution == "normal":
            return rng.normal(0.0, scale, size=size)
        return rng.uniform(-scale, scale, size=size)

    def _wrap2pi(x):
        return np.mod(x, 2.0 * np.pi)

    # Build sampled element arrays
    if include_nominal:
        n_samp = M - 1
        a_s = a0 + _draw(a_sigma, n_samp)
        e_s = e0 + _draw(e_sigma, n_samp)
        i_s = i0 + _draw(i_sigma, n_samp)
        raan_s = _wrap2pi(raan0 + _draw(raan_sigma, n_samp))
        pa_s = _wrap2pi(pa0 + _draw(pa_sigma, n_samp))
        nu_s = _wrap2pi(nu0 + _draw(nu_sigma, n_samp))

        a_all = np.concatenate([np.array([a0], dtype=float), a_s.astype(float)])
        e_all = np.concatenate([np.array([e0], dtype=float), e_s.astype(float)])
        i_all = np.concatenate([np.array([i0], dtype=float), i_s.astype(float)])
        raan_all = np.concatenate([np.array([_wrap2pi(raan0)], dtype=float), raan_s.astype(float)])
        pa_all = np.concatenate([np.array([_wrap2pi(pa0)], dtype=float), pa_s.astype(float)])
        nu_all = np.concatenate([np.array([_wrap2pi(nu0)], dtype=float), nu_s.astype(float)])
    else:
        a_all = (a0 + _draw(a_sigma, M)).astype(float)
        e_all = (e0 + _draw(e_sigma, M)).astype(float)
        i_all = (i0 + _draw(i_sigma, M)).astype(float)
        raan_all = _wrap2pi(raan0 + _draw(raan_sigma, M)).astype(float)
        pa_all = _wrap2pi(pa0 + _draw(pa_sigma, M)).astype(float)
        nu_all = _wrap2pi(nu0 + _draw(nu_sigma, M)).astype(float)

    # Guardrails
    a_all = np.maximum(a_all, 1.0)  # keep positive
    e_all = np.clip(e_all, float(e_clip[0]), float(e_clip[1]))
    # i can be outside [0,pi] for sampling; wrap to [0,pi] via modulo reflection
    i_all = np.mod(i_all, 2.0 * np.pi)
    i_all = np.where(i_all > np.pi, 2.0 * np.pi - i_all, i_all)

    # Create SSAPy Orbit objects at epoch t_start
    orbits = []
    for k in range(M):
        orb = Orbit.fromKeplerianElements(
            float(a_all[k]),
            float(e_all[k]),
            float(i_all[k]),
            float(pa_all[k]),
            float(raan_all[k]),
            float(nu_all[k]),
            t_start,
            mu=float(mu),
        )
        orbits.append(orb)

    # Propagate each orbit and collect r(t), v(t)
    r_list = []
    v_list = []
    t_list = []
    for k in range(M):
        o_t = orbits[k].at(t_abs, propagator=prop)  # returns a vector Orbit over the times
        r = np.asarray(o_t.r, dtype=float)
        v = np.asarray(o_t.v, dtype=float)

        # Ensure (N,3) shape for downstream stats code
        r = np.reshape(r, (N, 3))
        v = np.reshape(v, (N, 3))

        r_list.append(r)
        v_list.append(v)
        t_list.append(t_rel.copy())

    return orbits, r_list, v_list, t_list, float(mu)
