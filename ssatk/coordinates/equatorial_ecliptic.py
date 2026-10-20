import numpy as np

from ..constants import EARTH_OBLIQUITY_J2000_RAD
from .angle_units import rad0to2pi, deg0to360


# Constants for ecliptic inclination and sine/cosine of it
_ecliptic = EARTH_OBLIQUITY_J2000_RAD
cos_ec = np.cos(_ecliptic)
sin_ec = np.sin(_ecliptic)


def equatorial_xyz_to_ecliptic_xyz(xq: float, yq: float, zq: float) -> float:
    """
    Convert equatorial Cartesian coordinates to ecliptic Cartesian coordinates.

    Parameters:
    - xq, yq, zq (float): Equatorial Cartesian coordinates.

    Returns:
    - float: Ecliptic Cartesian coordinates (xc, yc, zc).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    xc = xq
    yc = cos_ec * yq + sin_ec * zq
    zc = -sin_ec * yq + cos_ec * zq
    return xc, yc, zc


def ecliptic_xyz_to_equatorial_xyz(xc: float, yc: float, zc: float) -> float:
    """
    Convert ecliptic Cartesian coordinates to equatorial Cartesian coordinates.

    Parameters:
    - xc, yc, zc (float): Ecliptic Cartesian coordinates.

    Returns:
    - float: Equatorial Cartesian coordinates (xq, yq, zq).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    xq = xc
    yq = cos_ec * yc - sin_ec * zc
    zq = sin_ec * yc + cos_ec * zc
    return xq, yq, zq


def xyz_to_ecliptic(xc: float, yc: float, zc: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> float:
    """
    Convert a point in space from equatorial coordinates to ecliptic latitude and longitude.

    Parameters:
    - xc, yc, zc (float): Cartesian coordinates of the point in space.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - float: Ecliptic longitude and latitude (in radians or degrees).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    x_ast_to_earth = xc - xe
    y_ast_to_earth = yc - ye
    z_ast_to_earth = zc - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ec_longitude = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    ec_latitude = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ec_longitude), np.degrees(ec_latitude)
    else:
        return ec_longitude, ec_latitude


def xyz_to_equatorial(xq: float, yq: float, zq: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> float:
    """
    Convert a point in space from Cartesian coordinates to equatorial right ascension and declination.

    Parameters:
    - xq, yq, zq (float): Cartesian coordinates of the point.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - float: Right ascension and declination (in radians or degrees).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # RA / DEC calculation - assumes XY plane to be celestial equator, and -x axis to be vernal equinox
    x_ast_to_earth = xq - xe
    y_ast_to_earth = yq - ye
    z_ast_to_earth = zq - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def ecliptic_xyz_to_equatorial(xc: float, yc: float, zc: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> float:
    """
    Convert ecliptic Cartesian coordinates to equatorial right ascension and declination.

    Parameters:
    - xc, yc, zc (float): Ecliptic Cartesian coordinates of the point.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - float: Right ascension and declination (in radians or degrees).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def equatorial_to_ecliptic(right_ascension: float, declination: float, degrees: bool = False) -> float:
    """
    Convert equatorial coordinates (right ascension and declination) to ecliptic coordinates (longitude and latitude).

    Parameters:
    - right_ascension (float): Right ascension in radians or degrees.
    - declination (float): Declination in radians or degrees.
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - float: Ecliptic longitude and latitude (in radians or degrees).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if degrees:
        ra, dec = np.radians(right_ascension), np.radians(declination)
    else:
        ra, dec = right_ascension, declination

    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan2(
        cos_ec * np.cos(dec) * np.sin(ra) + sin_ec * np.sin(dec),
        np.cos(dec) * np.cos(ra),
    )
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude


def ecliptic_to_equatorial(lon: float, lat: float, degrees: bool = False) -> float:
    """
    Convert ecliptic coordinates (longitude and latitude) to equatorial coordinates (right ascension and declination).

    Parameters:
    - lon (float): Ecliptic longitude in radians or degrees.
    - lat (float): Ecliptic latitude in radians or degrees.
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - float: Right ascension and declination (in radians or degrees).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if degrees:
        lon, lat = np.radians(lon), np.radians(lat)

    ra = np.arctan2(
        cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat),
        np.cos(lat) * np.cos(lon),
    )
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return deg0to360(np.degrees(ra)), np.degrees(dec)
    else:
        return rad0to2pi(ra), dec
