import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS

from ..constants import SUN_MU
from ..time_functions import to_gps
from ._state import state


def accel_point_sun(r: np.ndarray, time=None) -> np.ndarray:
    """
    Earth-centered solar third-body perturbing acceleration in GCRF.

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.
    time : float or datetime-like or array_like
        The time array element from leapfrog (same type you passed into leapfrog).

    Returns
    -------
    a_sun : ndarray, shape (3,)
        Perturbing acceleration from the Sun (m/s²).

    Notes
    -----
    Uses Astropy's active solar-system ephemeris, which defaults to the offline
    built-in model. Select a JPL kernel with
    ``astropy.coordinates.solar_system_ephemeris.set`` when required.
    """
    r, _, time = state(r, np.zeros(3), time)

    # 1) Convert time to GPS seconds
    time_gps = to_gps(time)
    t = Time(time_gps, format="gps", scale="utc")

    # 2) Get Sun position in GCRS using Astropy's configured ephemeris.
    sun_gcrs = get_body("sun", t).transform_to(GCRS(obstime=t))

    r_sun = sun_gcrs.cartesian.xyz.to(u.m).value

    # 3) Compute Earth-centered third-body perturbation.
    delta = r_sun - r
    sat_sun_distance = np.linalg.norm(delta)
    earth_sun_distance = np.linalg.norm(r_sun)
    if sat_sun_distance == 0.0 or earth_sun_distance == 0.0:
        return np.zeros(3, dtype=float)

    return SUN_MU * (delta / sat_sun_distance**3 - r_sun / earth_sun_distance**3)
