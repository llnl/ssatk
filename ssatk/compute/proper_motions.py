import numpy as np
import warnings

from ..constants import day_to_second, rad_to_arcsecond

# REBOUND's G = 1 time unit in au and Msun: yr2pi = 1/k day, with the Gaussian
# gravitational constant k = 0.01720209895.
_REBOUND_TIME_UNIT_S = day_to_second / 0.01720209895


def proper_motion(x: np.ndarray, y: np.ndarray, z: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                  xe: float = 0, ye: float = 0, ze: float = 0, vxe: float = 0, vye: float = 0, vze: float = 0,
                  input_unit: str = 'si') -> float:
    """
    Calculate the proper motion of an object in space relative to Earth.

    Parameters:
    x, y, z (np.ndarray): Position coordinates of the object.
    vx, vy, vz (np.ndarray): Velocity components of the object.
    xe, ye, ze (float): Position of Earth.
    vxe, vye, vze (float): Velocity of Earth.
    input_unit (str): 'si' for m and m/s, or 'rebound' for au and au per REBOUND time unit (yr2pi = 1/k day).

    Returns:
    float or None: The total proper motion in arcseconds per second for either input unit, NaN if the object is at the Earth's position, or None for an unrecognized input_unit.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    x_rot = x - xe
    y_rot = y - ye
    z_rot = z - ze
    vx_rot = vx - vxe
    vy_rot = vy - vye
    vz_rot = vz - vze

    d_earth_mag = np.linalg.norm([x_rot, y_rot, z_rot])
    if d_earth_mag == 0:
        return np.nan

    v_ast_earth = np.array([vx_rot, vy_rot, vz_rot])
    los_vector = np.array([x_rot, y_rot, z_rot])

    # Avoid subtracting nearly equal squared speeds for mostly radial motion.
    v_transverse = np.linalg.norm(np.cross(v_ast_earth, los_vector / d_earth_mag))

    if input_unit == 'si':
        return v_transverse / d_earth_mag * rad_to_arcsecond
    elif input_unit == 'rebound':
        return v_transverse / d_earth_mag * rad_to_arcsecond / _REBOUND_TIME_UNIT_S
    else:
        warnings.warn("input_unit must be 'si' or 'rebound'.", UserWarning, stacklevel=2)
        return None


def proper_motion_ra_dec(
    r: np.ndarray = None,
    v: np.ndarray = None,
    x: float = None,
    y: float = None,
    z: float = None,
    vx: float = None,
    vy: float = None,
    vz: float = None,
    r_earth: np.ndarray = np.array([0, 0, 0]),
    v_earth: np.ndarray = np.array([0, 0, 0]),
    input_unit: str = 'si'
) -> np.ndarray:
    """
    Calculate the proper motion in right ascension (RA) and declination (DEC) for a given position and velocity in 3D space.

    Parameters
    ----------
    r : numpy.ndarray, optional
        3D position vector (x, y, z) in SI units (m).
    v : numpy.ndarray, optional
        3D velocity vector (vx, vy, vz) in SI units (m/s).
    x, y, z : float, optional
        Individual position coordinates in meters, used when ``r`` is absent.
    vx, vy, vz : float, optional
        Individual velocity components in m/s, used when ``v`` is absent.
    r_earth, v_earth : numpy.ndarray, optional
        Earth position and velocity vectors; both default to zero.
    input_unit : {"si", "rebound"}, optional
        Unit system of the inputs: ``"si"`` for m and m/s, ``"rebound"`` for
        au and au per REBOUND time unit (yr2pi = 1/k day). Defaults to ``"si"``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(pmra, pmdec)`` in arcseconds per second for either input unit.
        ``pmra`` is the great-circle rate along increasing RA, i.e.
        ``mu_alpha * cos(dec)``.
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

    # Distances to Earth
    d_earth_mag = np.linalg.norm(r, axis=1)

    # RA / DEC calculation
    ra = np.arctan2(r[:, 1], r[:, 0])  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    ra_unit_vector = np.array([-np.sin(ra), np.cos(ra), np.zeros_like(ra)]).T
    dec_unit_vector = -np.array([np.cos(np.pi / 2 - dec) * np.cos(ra), np.cos(np.pi / 2 - dec) * np.sin(ra), -np.sin(np.pi / 2 - dec)]).T
    pmra = (np.einsum('ij,ij->i', v, ra_unit_vector)) / d_earth_mag * rad_to_arcsecond
    pmdec = (np.einsum('ij,ij->i', v, dec_unit_vector)) / d_earth_mag * rad_to_arcsecond

    if input_unit == 'si':
        return pmra, pmdec
    elif input_unit == 'rebound':
        # arcsec per REBOUND time unit -> arcsec per second
        return pmra / _REBOUND_TIME_UNIT_S, pmdec / _REBOUND_TIME_UNIT_S
    else:
        warnings.warn("input_unit must be 'si' or 'rebound'.", UserWarning, stacklevel=2)
        return
