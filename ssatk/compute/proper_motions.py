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
    float or np.ndarray: The total proper motion in arcseconds per second for either input
        unit (a scalar for scalar inputs, or an array vectorized along the sample axis of
        array-like inputs), NaN where the object is at the Earth's position, or None for an
        unrecognized input_unit.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot = np.broadcast_arrays(
        np.asarray(x, dtype=float) - xe, np.asarray(y, dtype=float) - ye, np.asarray(z, dtype=float) - ze,
        np.asarray(vx, dtype=float) - vxe, np.asarray(vy, dtype=float) - vye, np.asarray(vz, dtype=float) - vze,
    )
    scalar_input = x_rot.ndim == 0

    los_vector = np.stack([x_rot, y_rot, z_rot], axis=-1)
    v_ast_earth = np.stack([vx_rot, vy_rot, vz_rot], axis=-1)

    d_earth_mag = np.linalg.norm(los_vector, axis=-1)
    zero = d_earth_mag == 0

    with np.errstate(invalid="ignore", divide="ignore"):
        safe_d_earth_mag = np.where(zero, 1.0, d_earth_mag)
        # Avoid subtracting nearly equal squared speeds for mostly radial motion.
        v_transverse = np.linalg.norm(
            np.cross(v_ast_earth, los_vector / safe_d_earth_mag[..., None]), axis=-1)
        pm = np.where(zero, np.nan, v_transverse / safe_d_earth_mag)

    if input_unit == 'si':
        pm = pm * rad_to_arcsecond
    elif input_unit == 'rebound':
        pm = pm * rad_to_arcsecond / _REBOUND_TIME_UNIT_S
    else:
        warnings.warn("input_unit must be 'si' or 'rebound'.", UserWarning, stacklevel=2)
        return None

    return float(pm) if scalar_input else pm


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
    r : numpy.ndarray, optional
        3D position vector (x, y, z) in SI units (m), shape ``(3,)`` for a
        single state or ``(N, 3)`` for a batch.
    v : numpy.ndarray, optional
        3D velocity vector (vx, vy, vz) in SI units (m/s), matching the
        shape of ``r``.
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
    tuple of float or numpy.ndarray
        ``(pmra, pmdec)`` in arcseconds per second for either input unit:
        scalars for a single ``(3,)``/``x, y, z`` state, arrays of shape
        ``(N,)`` for an ``(N, 3)`` batch. ``pmra`` is the great-circle rate
        along increasing RA, i.e. ``mu_alpha * cos(dec)``.
    """
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    single_state = r.ndim == 1
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)

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

    if single_state:
        pmra, pmdec = pmra[0], pmdec[0]

    if input_unit == 'si':
        return pmra, pmdec
    elif input_unit == 'rebound':
        # arcsec per REBOUND time unit -> arcsec per second
        return pmra / _REBOUND_TIME_UNIT_S, pmdec / _REBOUND_TIME_UNIT_S
    else:
        warnings.warn("input_unit must be 'si' or 'rebound'.", UserWarning, stacklevel=2)
        return
