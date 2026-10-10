from .cartesian import cart2sph_deg
from .angle_units import deg0to360
import numpy as np


def inert2rot(x: float, y: float, xe: float, ye: float, xs: float = 0, ys: float = 0) -> float:
    """
    Rotate inertial coordinates to a rotated frame defined by the position of Earth.

    Parameters:
    - x (float): The x-coordinate in inertial space.
    - y (float): The y-coordinate in inertial space.
    - xe (float): The x-coordinate of Earth's position.
    - ye (float): The y-coordinate of Earth's position.
    - xs (float, optional): The x-coordinate of the Sun's position. Defaults to 0.
    - ys (float, optional): The y-coordinate of the Sun's position. Defaults to 0.

    Returns:
    - Tuple[float, float]: The rotated x and y coordinates.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs), 2) + np.power((y - ys), 2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot


def sim_lonlatrad(x: float, y: float, z: float, xe: float, ye: float, ze: float, xs: float, ys: float, zs: float) -> float:
    """
    Convert Cartesian coordinates (x, y, z) to geodetic longitude, latitude, and radius,
    relative to the Sun's position at the given time.

    Parameters:
    - x, y, z (float): Cartesian coordinates of the point of interest.
    - xe, ye, ze (float): Cartesian coordinates of the observer (Earth).
    - xs, ys, zs (float): Cartesian coordinates of the Sun.

    Returns:
    - float: Longitude, latitude, and radius relative to the Sun in degrees and kilometers.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    # Convert x, y, z to lon, lat, radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    # Correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    return longitude, latitude, radius
