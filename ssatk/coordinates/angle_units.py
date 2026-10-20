import numpy as np
from astropy.coordinates import Angle


def dms_to_rad(coords):
    """
    Convert degrees/minutes/seconds (DMS) to radians.

    Parameters
    ----------
    coords : str or sequence of str
        A single DMS string (e.g., '30d30m30s', '45d') or a list/tuple of such strings.

    Returns
    -------
    float or list of float
        Angle(s) in radians.
    """
    if isinstance(coords, (list, tuple)):
        return [Angle(c).radian for c in coords]
    return Angle(coords).radian


def dms_to_deg(coords):
    """
    Convert degrees/minutes/seconds (DMS) to decimal degrees.

    Parameters
    ----------
    coords : str or sequence of str
        A single DMS string or a list/tuple of such strings.

    Returns
    -------
    float or list of float
        Angle(s) in degrees.
    """
    if isinstance(coords, (list, tuple)):
        return [Angle(c).deg for c in coords]
    return Angle(coords).deg


def rad0to2pi(angles):
    """
    Normalize angle(s) in radians to the range [0, 2π).

    Parameters
    ----------
    angles : float or array-like of float
        Angle(s) in radians.

    Returns
    -------
    float or numpy.ndarray
        Normalized angle(s) in radians.
    """
    arr = np.asarray(angles)
    out = arr % (2 * np.pi)
    return out.item() if np.isscalar(angles) else out


def deg0to360(x):
    """
    Normalize angle(s) in degrees to the range [0, 360).

    Parameters
    ----------
    x : float or array-like of float
        Angle(s) in degrees.

    Returns
    -------
    float or numpy.ndarray
        Normalized angle(s) in degrees.
    """
    arr = np.asarray(x)
    out = np.mod(arr, 360.0)
    # Ensure 360 maps to 0 (e.g., input 360 -> 0)
    out = np.where(out == 360.0, 0.0, out)
    return out.item() if np.isscalar(x) else out


def deg0to360array(array_):
    """
    Normalize a list/array of angles (deg) to [0, 360).

    Parameters
    ----------
    array_ : sequence of float or numpy.ndarray

    Returns
    -------
    list of float
        Normalized angles in degrees.
    """
    return list(np.mod(np.asarray(array_, dtype=float), 360.0))


def deg90to90(x):
    """
    Normalize angle(s) in degrees to the range [-90, 90].

    Note
    ----
    Uses the standard wrap formula: ((x + 90) % 180) - 90

    Parameters
    ----------
    x : float or array-like of float
        Angle(s) in degrees.

    Returns
    -------
    float or numpy.ndarray
        Normalized angle(s) in degrees.
    """
    arr = np.asarray(x, dtype=float)
    out = (arr + 90.0) % 180.0 - 90.0
    return out.item() if np.isscalar(x) else out


def deg90to90array(array_):
    """
    Normalize a list/array of angles (deg) to [-90, 90].

    Parameters
    ----------
    array_ : sequence of float or numpy.ndarray

    Returns
    -------
    list of float
        Normalized angles in degrees.
    """
    arr = np.asarray(array_, dtype=float)
    out = (arr + 90.0) % 180.0 - 90.0
    return list(out)
