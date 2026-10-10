# ssatk/orbital_mechanics/all_orbit_quantities.py

import numpy as np
from astropy.time import Time
from ssapy import Orbit
from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.keplerian import true_anomaly  # mean/eccentric -> true anomaly


def all_orbital_quantities(
    r=None,
    v=None,
    a=None,
    e=None,
    i=None,
    pa=None,
    raan=None,
    ta=None,
    ma=None,
    periapsis=None,
    apoapsis=None,
    t=None,
    mu=EARTH_MU,
):
    """
    Compute a consistent set of orbital quantities, aligned with SSAPy conventions.

    SSAPy Keplerian initialization order:
        Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t, mu) [6]

    Priority order if conflicting inputs are provided:
      1) (r, v)
      2) (periapsis, apoapsis)
      3) (a, e)

    Notes:
      - Angles default to 0 if not supplied (i, raan, pa, ta).
      - If ma is provided without ta, we convert ma->ta using ssatk.keplerian.true_anomaly [3].
      - We prefer SSAPy-computed attributes (period, meanMotion, anomalies, etc.) when available [6].
    """
    if t is None:
        t = Time.now().gps
    t = float(t)
    mu = float(mu)

    # ----------------------------
    # Case 1: Cartesian state
    # ----------------------------
    if r is not None and v is not None:
        r = np.asarray(r, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)
        orbit = Orbit(r=r, v=v, t=t, mu=mu)
        return _extract_all_elements_ssapy(orbit)

    # Defaults for angles
    i = 0.0 if i is None else float(i)
    raan = 0.0 if raan is None else float(raan)
    pa = 0.0 if pa is None else float(pa)

    # ----------------------------
    # Case 2: periapsis/apoapsis
    # ----------------------------
    if periapsis is not None and apoapsis is not None:
        periapsis = float(periapsis)
        apoapsis = float(apoapsis)

        a = 0.5 * (periapsis + apoapsis)
        e = (apoapsis - periapsis) / (apoapsis + periapsis)

        ta = _resolve_true_anomaly(ta=ta, ma=ma, e=e)

        orbit = Orbit.fromKeplerianElements(a, e, i, pa, raan, ta, t, mu)
        return _extract_all_elements_ssapy(orbit)

    # ----------------------------
    # Case 3: Keplerian a/e
    # ----------------------------
    if a is not None and e is not None:
        a = float(a)
        e = float(e)

        ta = _resolve_true_anomaly(ta=ta, ma=ma, e=e)

        orbit = Orbit.fromKeplerianElements(a, e, i, pa, raan, ta, t, mu)
        return _extract_all_elements_ssapy(orbit)

    raise ValueError(
        "Insufficient parameters provided. Need either:\n"
        "  1) Position (r) and velocity (v) vectors, or\n"
        "  2) Periapsis and apoapsis distances, or\n"
        "  3) Semi-major axis (a) and eccentricity (e)\n"
        "Angular elements (i, raan, pa, ta) default to 0 if not provided."
    )


def _resolve_true_anomaly(*, ta, ma, e):
    """Pick/compute true anomaly; default to 0. Uses SSATK keplerian.true_anomaly for ma->ta."""
    if ta is not None:
        return float(ta)
    if ma is None:
        return 0.0
    # SSATK keplerian approximation supports eccentricity + mean anomaly.
    ta_val = true_anomaly(eccentricity=float(e), mean_anomaly=float(ma))
    if ta_val is None:
        raise ValueError("Could not compute true anomaly from mean anomaly with provided inputs.")
    return float(ta_val)


def _extract_all_elements_ssapy(orbit: Orbit):
    """
    Return a dictionary that is as consistent with SSAPy as possible:
      - Use SSAPy naming (pa, raan, trueAnomaly/meanAnomaly/eccentricAnomaly, meanMotion, period) [6]
      - Keep r,v,t,mu and add common derived scalars (rp, ra).
    """
    a_val, e_val, i_val, pa_val, raan_val, ta_val = orbit.keplerianElements

    # Prefer SSAPy properties
    out = {
        # epoch/state
        "t": float(orbit.t),
        "mu": float(orbit.mu),
        "r": np.asarray(orbit.r, dtype=float),
        "v": np.asarray(orbit.v, dtype=float),

        # classical Keplerian elements (SSAPy order/names)
        "a": float(a_val),
        "e": float(e_val),
        "i": float(i_val),
        "pa": float(pa_val),
        "raan": float(raan_val),
        "trueAnomaly": float(ta_val),

        # SSAPy anomaly fields (when defined)
        "eccentricAnomaly": _safe_float(getattr(orbit, "eccentricAnomaly", np.nan)),
        "meanAnomaly": _safe_float(getattr(orbit, "meanAnomaly", np.nan)),

        # SSAPy derived fields (when defined)
        "period": _safe_float(getattr(orbit, "period", np.inf)),
        "meanMotion": _safe_float(getattr(orbit, "meanMotion", 0.0)),
        "p": _safe_float(getattr(orbit, "p", np.nan)),
        "angularMomentum": np.asarray(getattr(orbit, "angularMomentum", np.full(3, np.nan)), dtype=float),
        "energy": _safe_float(getattr(orbit, "energy", np.nan)),
        "LRL": np.asarray(getattr(orbit, "LRL", np.full(3, np.nan)), dtype=float),

        # SSAPy periapsis/apoapsis coordinates (vectors)
        "periapsis": np.asarray(getattr(orbit, "periapsis", np.full(3, np.nan)), dtype=float),
        "apoapsis": np.asarray(getattr(orbit, "apoapsis", np.full(3, np.nan)), dtype=float),
    }

    # Common scalar distances (rp/ra) derived from a,e when meaningful
    if np.isfinite(out["a"]) and out["e"] < 1.0:
        out["rp"] = out["a"] * (1.0 - out["e"])
        out["ra"] = out["a"] * (1.0 + out["e"])
    else:
        out["rp"] = np.nan
        out["ra"] = np.inf

    return out


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return np.nan
