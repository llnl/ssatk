import numpy as np


def cart_to_cyl(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Cartesian ``(x, y, z)`` to cylindrical ``(r, theta, z)``."""

    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z


def cart2sph_deg(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Cartesian ``(x, y, z)`` to ``(azimuth_deg, elevation_deg, radius)``."""

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.degrees(np.arctan2(z, hxy))
    az = np.degrees(np.arctan2(y, x))
    return az, el, r
