from astropy.time import Time
import numpy as np

from .astropy_llh_to_gcrf import astropy_llh_to_gcrf


def astropy_surface_rv(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc")):
    """
    Get GCRF position and velocity of a surface observer at a specific location and time.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees.
    lat : float
        Geodetic latitude in degrees.
    elevation : float, optional
        Elevation above sea level in meters (default is 0).
    t : Time
        Astropy Time object.

    Returns
    -------
    r : ndarray, shape (3,)
        Position vector in GCRF coordinates (m).
    v : ndarray, shape (3,)
        Velocity vector in GCRF coordinates (m/s).
    """
    # Position in GCRF (meters)
    r_gcrf = astropy_llh_to_gcrf(lon, lat, t, alt=elevation).reshape(3)

    # Earth's angular velocity vector (rad/s)
    omega_earth = np.array([0.0, 0.0, 7.2921150e-5])  # rotation about Z-axis

    # Velocity in GCRF: v = omega × r
    v_gcrf = np.cross(omega_earth, r_gcrf)

    return r_gcrf, v_gcrf
