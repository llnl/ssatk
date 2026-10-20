import numpy as np
from astropy.time import Time
from ssapy.orbit import Orbit
from ssapy.propagator import SGP4Propagator
from .tle_iter_pairs import tle_iter_pairs

def tle_prop_to_time(t, tle_path, *, validate_checksum=False, truncate=False, return_arrays=False):
    """
    Propagate all TLE pairs in `tle_path` to a common epoch `t` using SGP4.

    Parameters
    ----------
    t : float | str | astropy.time.Time
        Target time. If float, interpreted as GPS seconds.
        If str, interpreted as UTC ISO (e.g., '2025-01-01T00:00:00').
        If astropy Time, uses its .gps.
    tle_path : str | pathlib.Path
        Path to a text file containing 0/1/2 TLE lines.
    validate_checksum : bool, optional
        If True, verify TLE checksums before using pairs.
    truncate : bool, optional
        Passed to SGP4Propagator(truncate=...).
    return_arrays : bool, optional
        If True, also return stacked R (m) and V (m/s) arrays and names.

    Returns
    -------
    orbits : list[Orbit]
        Or list plus (names, R, V) if return_arrays=True.
    """
    # normalize time -> GPS seconds (float)
    if isinstance(t, (int, float, np.floating)):
        t_gps = float(t)
    elif isinstance(t, Time):
        t_gps = float(t.gps)
    elif isinstance(t, str):
        t_gps = float(Time(t, scale="utc").gps)
    else:
        raise TypeError("t must be GPS seconds (float), ISO string, or astropy.time.Time")

    prop = SGP4Propagator(t=t_gps, truncate=truncate)

    orbits_at_t = []
    names = []
    for name, line1, line2 in tle_iter_pairs(tle_path, validate_checksum=validate_checksum):
        try:
            o0 = Orbit.fromTLETuple((line1, line2))
            oT = o0.at(t_gps, propagator=prop)
            orbits_at_t.append(oT)
            names.append(name)
        except Exception:
            # skip malformed TLEs or propagation failures
            continue

    if return_arrays:
        if orbits_at_t:
            R = np.vstack([o.r for o in orbits_at_t])  # meters, GCRF
            V = np.vstack([o.v for o in orbits_at_t])  # m/s, GCRF
        else:
            R = np.empty((0, 3))
            V = np.empty((0, 3))
        return orbits_at_t, names, R, V

    return orbits_at_t
