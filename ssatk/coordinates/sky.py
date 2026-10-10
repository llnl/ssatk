import numpy as np
from ssapy import get_body, groundTrack, rv, SciPyPropagator, AccelKepler, Orbit
from astropy.time import Time

from ..constants import RGEO
from .angle_units import rad0to2pi


def sun_ra_dec(time_):
    """
    Calculate the geocentric Right Ascension and Declination of the Sun.

    Parameters
    ----------
    time_ : int, float, str, or astropy.time.Time
        Time for the Sun position: MJD (UTC) as a number or array, an
        ISO-format string, or an astropy ``Time``.

    Returns
    -------
    tuple of float or numpy.ndarray
        Right Ascension in [0, 2*pi) and Declination of the Sun in radians,
        from SSAPy's geometric GCRF Sun position.
    """
    if isinstance(time_, Time):
        t = time_
    elif isinstance(time_, str):
        t = Time(time_, scale="utc")
    else:
        t = Time(time_, format="mjd", scale="utc")
    sun = get_body("sun")  # keep the Body referenced while its kernel is in use
    r = np.asarray(sun.position(np.atleast_1d(t.gps)), dtype=float).reshape(3, -1)
    ra = np.mod(np.arctan2(r[1], r[0]), 2 * np.pi)
    dec = np.arcsin(r[2] / np.linalg.norm(r, axis=0))
    if np.ndim(t.gps) == 0:
        return float(ra[0]), float(dec[0])
    return ra, dec


def ra_dec(
    r: np.ndarray = None, v: np.ndarray = None,
    x: float = None, y: float = None, z: float = None,
    vx: float = None, vy: float = None, vz: float = None,
    r_earth: np.ndarray = np.array([0, 0, 0]),
    v_earth: np.ndarray = np.array([0, 0, 0]),
    input_unit: str = 'si'
) -> np.ndarray:
    """
    Compute the Right Ascension (RA) and Declination (Dec) of a position and velocity vector,
    relative to Earth's position and velocity.

    Parameters:
    - r (np.ndarray): Position vector of the object in 3D space.
    - v (np.ndarray): Velocity vector of the object.
    - x, y, z (float, optional): Individual Cartesian coordinates of the object.
    - vx, vy, vz (float, optional): Individual velocity components of the object.
    - r_earth (np.ndarray, optional): Earth's position vector. Default is the origin.
    - v_earth (np.ndarray, optional): Earth's velocity vector. Default is zero velocity.
    - input_unit (str, optional): The unit system used. Defaults to SI units.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: RA and Dec in radians.
    """
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    d_earth_mag = np.linalg.norm(r, axis=1)
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    return ra, dec


def gcrf_to_radec(gcrf_coords: np.ndarray) -> np.ndarray:
    """
    Convert GCRF Cartesian coordinates to right ascension and declination.

    Parameters:
    - gcrf_coords (np.ndarray): 3D position vector in GCRF coordinates (x, y, z).

    Returns:
    - Tuple of right ascension and declination in degrees.
    """
    x, y, z = gcrf_coords
    ra = np.arctan2(y, x)
    ra_deg = np.degrees(ra)
    ra_deg = ra_deg % 360
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    dec_deg = np.degrees(dec_rad)
    return ra_deg, dec_deg


def gcrf_to_lat_lon(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert GCRF coordinates to latitude, longitude, and height above Earth's surface.

    Parameters:
    - r (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.

    Returns:
    - Tuple of geodetic longitude (rad), latitude (rad), and height above the WGS84 ellipsoid (m), from ``ssapy.groundTrack``.
    """
    lon, lat, height = groundTrack(r, t)
    return lon, lat, height


def gcrf_to_sim_geo(r_gcrf: np.ndarray, t: np.ndarray, h: float = 10) -> np.ndarray:
    """
    Convert GCRF coordinates to simulated geodetic coordinates (latitude, longitude, height).

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - h (float): Step size in seconds for time propagation.

    Returns:
    - Simulated geodetic coordinates in meters.
    """
    if np.min(np.diff(t.gps)) < h:
        h = np.min(np.diff(t.gps))
    r_gcrf = np.atleast_2d(r_gcrf)
    r_geo, v_geo = rv(Orbit.fromKeplerianElements(*[RGEO, 0, 0, 0, 0, 0], t=t[0]), t, propagator=SciPyPropagator(AccelKepler()))
    angle_geo_to_x = np.arctan2(r_geo[:, 1], r_geo[:, 0])
    c = np.cos(angle_geo_to_x)
    s = np.sin(angle_geo_to_x)
    rotation = np.array([[c, -s, np.zeros_like(c)], [s, c, np.zeros_like(c)], [np.zeros_like(c), np.zeros_like(c), np.ones_like(c)]]).T
    return np.einsum('ijk,ik->ij', rotation, r_gcrf)


def altitude2zenithangle(altitude: float, deg: bool = True) -> float:
    """
    Convert altitude to zenith angle.

    Parameters:
    - altitude (float): The altitude of the object in degrees or radians.
    - deg (bool, optional): If True, the output is in degrees. If False, in radians. Default is True.

    Returns:
    - float: Zenith angle in degrees or radians.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if deg:
        return 90 - altitude
    else:
        return np.pi / 2 - altitude


def zenithangle2altitude(zenith_angle: float, deg: bool = True) -> float:
    """
    Convert zenith angle to altitude.

    Parameters:
    - zenith_angle (float): The zenith angle in degrees or radians.
    - deg (bool, optional): If True, the input is in degrees. If False, in radians. Default is True.

    Returns:
    - float: Altitude in degrees or radians.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if deg:
        return 90 - zenith_angle
    else:
        return np.pi / 2 - zenith_angle
