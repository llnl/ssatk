# ssatk/accelerations_orbit/accel_moon.py

import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS

from ..constants import MOON_MU
from ..time_functions import to_gps
from ._state import state


def accel_point_moon(r: np.ndarray, time=None) -> np.ndarray:
    """
    Earth-centered lunar third-body perturbing acceleration in GCRF.

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.
    time : float | datetime-like | astropy.time.Time | array_like
        Time corresponding to the state (anything supported by to_gps()).

    Returns
    -------
    a_moon : ndarray, shape (3,)
        Perturbing acceleration from the Moon (m/s^2).

    Notes
    -----
    Uses Astropy's active solar-system ephemeris, which defaults to the offline
    built-in model. Select a JPL kernel with
    ``astropy.coordinates.solar_system_ephemeris.set`` when required.
    """
    r, _, time = state(r, np.zeros(3), time)

    # 1) Convert time to GPS seconds since 1980-01-06
    time_gps = to_gps(time)
    t = Time(time_gps, format="gps", scale="utc")

    # 2) Get Moon position in GCRS using Astropy's configured ephemeris.
    moon_gcrs = get_body("moon", t).transform_to(GCRS(obstime=t))

    r_moon = moon_gcrs.cartesian.xyz.to(u.m).value  # (3,)

    # 3) Compute Earth-centered third-body perturbation.
    delta = r_moon - r
    sat_moon_distance = np.linalg.norm(delta)
    earth_moon_distance = np.linalg.norm(r_moon)
    if sat_moon_distance == 0.0 or earth_moon_distance == 0.0:
        return np.zeros(3, dtype=float)

    return MOON_MU * (delta / sat_moon_distance**3 - r_moon / earth_moon_distance**3)
