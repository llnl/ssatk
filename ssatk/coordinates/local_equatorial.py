import numpy as np
import warnings
from .sky import zenithangle2altitude
from ..time_functions import hms_to_dd, dd_to_hms


def rightascension2hourangle(right_ascension, local_time):
    """
    Convert right ascension and local time to hour angle.

    Parameters:
    - right_ascension (str or float): The right ascension of the object in HH:MM:SS format or decimal degrees.
    - local_time (str or float): The local sidereal time in HH:MM:SS format or decimal hours.

    Returns:
    - str: The corresponding hour angle in HH:MM:SS format.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    ra_deg = hms_to_dd(right_ascension) if isinstance(right_ascension, str) else float(right_ascension)
    lst_deg = hms_to_dd(local_time) if isinstance(local_time, str) else 15.0 * float(local_time)
    # Hour angle = local sidereal time - right ascension, wrapped to [0, 24) h.
    return dd_to_hms((lst_deg - ra_deg) % 360.0)


def equatorial_to_horizontal(
    observer_latitude,
    declination,
    right_ascension=None,
    hour_angle=None,
    local_time=None,
    hms=False
):
    """
    Convert equatorial coordinates (right ascension, declination) to horizontal coordinates (azimuth, altitude).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - declination (float): Declination of the object in degrees.
    - right_ascension (str or float, optional): Right ascension in HH:MM:SS format or decimal degrees.
    - hour_angle (str or float, optional): Hour angle in HH:MM:SS format or decimal degrees.
    - local_time (str or float, optional): Local time in HH:MM:SS format or decimal hours.
    - hms (bool): If True, interpret inputs as HH:MM:SS strings.

    Returns:
    - (float, float): Azimuth and altitude in degrees.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if hour_angle is not None:
        if right_ascension is not None:
            warnings.warn(
                "Both right_ascension and hour_angle parameters are provided; using hour_angle for calculations.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(hour_angle, str):
            hour_angle_dd = hms_to_dd(hour_angle)
        else:
            hour_angle_dd = hour_angle
    elif right_ascension is not None:
        hour_angle_dd = rightascension2hourangle(right_ascension, local_time)
        hour_angle_dd = hms_to_dd(hour_angle_dd)
    else:
        raise ValueError('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle_rad, declination = np.radians(
        [observer_latitude, hour_angle_dd, declination]
    )

    zenith_angle = np.arccos(
        np.sin(observer_latitude) * np.sin(declination) +
        np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle_rad)
    )

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    # Azimuth from north through east. arccos alone cannot tell east from west;
    # the sign of sin(hour angle) does (positive hour angle = west of meridian).
    azimuth = np.mod(np.arctan2(
        -np.cos(declination) * np.sin(hour_angle_rad),
        np.sin(declination) * np.cos(observer_latitude)
        - np.cos(declination) * np.sin(observer_latitude) * np.cos(hour_angle_rad),
    ), 2 * np.pi)
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    """
    Convert horizontal coordinates (azimuth, altitude) to equatorial coordinates (hour angle, declination).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - azimuth (float): Azimuth in degrees.
    - altitude (float): Altitude in degrees.

    Returns:
    - (float, float): Hour angle and declination in degrees.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    altitude_rad, azimuth_rad, latitude_rad = np.radians([altitude, azimuth, observer_latitude])

    zenith_angle_rad = np.pi / 2 - altitude_rad

    declination_rad = np.arcsin(
        np.sin(latitude_rad) * np.cos(zenith_angle_rad) +
        np.cos(latitude_rad) * np.sin(zenith_angle_rad) * np.cos(azimuth_rad)
    )

    cos_hour_angle = (
        (np.cos(zenith_angle_rad) - np.sin(latitude_rad) * np.sin(declination_rad)) /
        (np.cos(latitude_rad) * np.cos(declination_rad))
    )

    hour_angle_rad = np.arccos(np.clip(cos_hour_angle, -1, 1))

    if np.sin(azimuth_rad) > 0:  # east of the meridian: rising, negative hour angle
        hour_angle_rad = 2 * np.pi - hour_angle_rad

    declination, hour_angle = np.degrees([declination_rad, hour_angle_rad])

    return hour_angle, declination
