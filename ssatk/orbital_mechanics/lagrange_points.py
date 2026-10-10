import numpy as np
from ssapy import get_body
from astropy.time import Time

from ..constants import EARTH_MU, MOON_MU
from ..coordinates import gcrf_to_lunar, gcrf_to_lunar_fixed


def moon_normal_vector(t):
    """
    Calculate the normal vector to the Moon's orbital plane at a given time.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Moon's orbital plane normal vector. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    np.ndarray
        The normal vector to the Moon's orbital plane, normalized to unit length.

    Notes
    -----
    - The normal vector is calculated as the cross product of the Moon's position
      vector at time `t` and its position vector one week later (`t + 604800` seconds).
    - The result is normalized to ensure it has unit length.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t, list):
        t = [item.gps if isinstance(item, Time) else item for item in t]
    elif isinstance(t, Time):
        t = t.gps
    moon_body = get_body("moon")
    r = moon_body.position(t).T
    r_random = moon_body.position(np.asarray(t) + 604800).T
    normal = np.cross(r, r_random)
    normal_norm = np.linalg.norm(normal, axis=-1)
    if np.ndim(normal_norm) == 0:
        return normal / normal_norm
    return normal / normal_norm[..., np.newaxis]


def _cr3bp_collinear_fractions(mass_ratio):
    """Collinear equilibria of the circular restricted three-body problem.

    Returns the signed distances of L1, L2, and L3 from the primary along the
    primary-to-secondary line, in units of the primary-secondary separation.
    """
    from scipy.optimize import brentq

    mu = mass_ratio

    def net_acceleration(x):  # rotating frame, barycentric x, primary at -mu
        return (x - (1.0 - mu) * (x + mu) / abs(x + mu) ** 3
                - mu * (x - 1.0 + mu) / abs(x - 1.0 + mu) ** 3)

    eps = 1e-12
    l1 = brentq(net_acceleration, -mu + eps, 1.0 - mu - eps, xtol=1e-15)
    l2 = brentq(net_acceleration, 1.0 - mu + eps, 2.0, xtol=1e-15)
    l3 = brentq(net_acceleration, -2.0, -mu - eps, xtol=1e-15)
    return l1 + mu, l2 + mu, l3 + mu


def _earth_moon_lagrange_points(t):
    """Earth-centred GCRF positions (m) of the instantaneous Earth-Moon L1-L5."""
    if isinstance(t, list):
        t = [item.gps if isinstance(item, Time) else item for item in t]
    elif isinstance(t, Time):
        t = t.gps
    scalar = np.ndim(t) == 0
    t_gps = np.atleast_1d(np.asarray(t, dtype=float))
    moon_body = get_body("moon")  # keep the Body referenced while it is used
    r_moon = np.asarray(moon_body.position(t_gps), dtype=float).reshape(3, -1).T
    v_moon = (np.asarray(moon_body.position(t_gps + 60.0), dtype=float).reshape(3, -1).T
              - np.asarray(moon_body.position(t_gps - 60.0), dtype=float).reshape(3, -1).T) / 120.0
    d = np.linalg.norm(r_moon, axis=-1, keepdims=True)
    x_hat = r_moon / d
    z_hat = np.cross(r_moon, v_moon)
    z_hat /= np.linalg.norm(z_hat, axis=-1, keepdims=True)
    y_hat = np.cross(z_hat, x_hat)  # along the Moon's motion

    f1, f2, f3 = _cr3bp_collinear_fractions(MOON_MU / (EARTH_MU + MOON_MU))
    points = {
        "L1": f1 * d * x_hat,
        "L2": f2 * d * x_hat,
        "L3": f3 * d * x_hat,
        "L4": d * (0.5 * x_hat + np.sqrt(3.0) / 2.0 * y_hat),
        "L5": d * (0.5 * x_hat - np.sqrt(3.0) / 2.0 * y_hat),
    }
    return {key: value[0] if scalar else value for key, value in points.items()}


def lunar_lagrange_points(t):
    """
    Calculate the positions of the lunar Lagrange points (L1, L2, L3, L4, L5)
    in the Earth-Moon system.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Lagrange points. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    dict
        A dictionary containing the positions of the Lagrange points:
        - "L1": Position of L1, between Earth and the Moon
        - "L2": Position of L2, beyond the Moon
        - "L3": Position of L3, on the far side of Earth
        - "L4": Position of L4 (60 degrees ahead of the Moon in its orbit)
        - "L5": Position of L5 (60 degrees behind the Moon in its orbit)

    Notes
    -----
    - The points are the equilibria of the circular restricted three-body
      problem scaled to the instantaneous Earth-Moon distance d. L1, L2, and L3
      lie on the Earth-Moon line at about 0.849 d, 1.168 d, and -0.993 d from
      Earth; L4 and L5 form equilateral triangles with Earth and the Moon in
      the instantaneous orbital plane, leading and trailing the Moon.
    - Positions are Earth-centred GCRF in metres.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return _earth_moon_lagrange_points(t)


def lunar_lagrange_points_circular(t):
    """
    Calculate the positions of the lunar Lagrange points (L1, L2, L3, L4, L5)
    in a circular restricted three-body problem.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Lagrange points. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    dict
        A dictionary containing the positions of the Lagrange points:
        - "L1": Position of L1, between Earth and the Moon
        - "L2": Position of L2, beyond the Moon
        - "L3": Position of L3, on the far side of Earth
        - "L4": Position of L4 (60 degrees ahead of the Moon in its orbit)
        - "L5": Position of L5 (60 degrees behind the Moon in its orbit)

    Notes
    -----
    - The points are the equilibria of the circular restricted three-body
      problem scaled to the instantaneous Earth-Moon distance d. L1, L2, and L3
      lie on the Earth-Moon line at about 0.849 d, 1.168 d, and -0.993 d from
      Earth; L4 and L5 form equilateral triangles with Earth and the Moon in
      the instantaneous orbital plane, leading and trailing the Moon.
    - Positions are Earth-centred GCRF in metres.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return _earth_moon_lagrange_points(t)


def lagrange_points_lunar_frame(t=None):
    """Earth-Moon L1-L5 in the rotating lunar frame of :func:`ssatk.coordinates.gcrf_to_lunar` (Earth-centred), in meters.

    ``t`` is the epoch (GPS seconds or astropy Time); it previously could not be
    set and was always 2025-01-01, which is still the default.
    """
    if t is None:
        t = Time(["2025-1-1"], scale='utc').gps
    elif hasattr(t, "gps"):
        t = np.atleast_1d(t.gps)
    else:
        t = np.atleast_1d(np.asarray(t, dtype=float))
    L = lunar_lagrange_points(t)
    return {
        "L1": np.squeeze(gcrf_to_lunar(L["L1"], t)),
        "L2": np.squeeze(gcrf_to_lunar(L["L2"], t)),
        "L3": np.squeeze(gcrf_to_lunar(L["L3"], t)),
        "L4": np.squeeze(gcrf_to_lunar(L["L4"], t)),
        "L5": np.squeeze(gcrf_to_lunar(L["L5"], t)),
    }


def lagrange_points_lunar_fixed_frame(t=None):
    """Earth-Moon L1-L5 in the Moon-centred Earth-Moon rotating frame of :func:`ssatk.coordinates.gcrf_to_lunar_fixed` (Earth on -X), in meters.

    ``t`` is the epoch (GPS seconds or astropy Time); it previously could not be
    set and was always 2025-01-01, which is still the default.
    """
    if t is None:
        t = Time(["2025-1-1"], scale='utc').gps
    elif hasattr(t, "gps"):
        t = np.atleast_1d(t.gps)
    else:
        t = np.atleast_1d(np.asarray(t, dtype=float))
    L = lunar_lagrange_points(t)
    return {
        "L1": np.squeeze(gcrf_to_lunar_fixed(L["L1"], t)),
        "L2": np.squeeze(gcrf_to_lunar_fixed(L["L2"], t)),
        "L3": np.squeeze(gcrf_to_lunar_fixed(L["L3"], t)),
        "L4": np.squeeze(gcrf_to_lunar_fixed(L["L4"], t)),
        "L5": np.squeeze(gcrf_to_lunar_fixed(L["L5"], t)),
    }
