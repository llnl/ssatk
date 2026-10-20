# ssatk/accelerations_orbit/accel_earth_harmonics.py
"""Point-mass Earth gravity plus the J2-J8 zonal harmonics.

The zonal potential of degree n is

    U_n = -mu J_n R^n P_n(u) / r^(n+1),    u = z / r,

with P_n the Legendre polynomial, and its gradient is

    a_n = mu J_n R^n / r^(n+3) * [((n+1) P_n(u) + u P_n'(u)) r_vec - P_n'(u) r z_hat].

The coefficients are the unnormalized EGM96 zonals (J_n = -C_n0), which match
the EGM96 file shipped with SSAPy and the EGM96/WGS84 ``EARTH_MU`` and
``EARTH_RADIUS`` in ``ssatk.constants``. The z axis of ``r`` is taken
to be Earth's rotation axis; no precession, nutation, or polar motion is
applied.
"""

import numpy as np
from numpy.polynomial import legendre as _legendre

from ..constants import EARTH_MU, EARTH_RADIUS
from ._state import position

# Unnormalized EGM96 zonal coefficients J_n = -C_n0.
EGM96_ZONAL_J = {
    2: 1.0826266835531513e-3,
    3: -2.5326564853322355e-6,
    4: -1.619621591367e-6,
    5: -2.2729608286869828e-7,
    6: 5.406812391070849e-7,
    7: -3.523599084182364e-7,
    8: -2.0479946698535123e-7,
}


def _check_r_safe(r2: float) -> None:
    """Raise error if r is below Earth's surface."""
    r_mag = float(np.sqrt(r2))
    if r_mag < EARTH_RADIUS:
        raise ValueError(f"r magnitude ({r_mag:.2f} m) is below Earth's surface.")


def _accel_zonal(r, n: int) -> np.ndarray:
    """Acceleration (m/s^2) from the degree-n zonal term at position r (m)."""
    r = position(r)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    u = r[2] / r_mag

    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    p_n = _legendre.legval(u, coeffs)
    dp_n = _legendre.legval(u, _legendre.legder(coeffs))

    factor = EARTH_MU * EGM96_ZONAL_J[n] * EARTH_RADIUS**n / r_mag ** (n + 3)
    radial = ((n + 1) * p_n + u * dp_n) * r
    axial = np.array([0.0, 0.0, dp_n * r_mag])
    return factor * (radial - axial)


def accel_J2(r: np.ndarray) -> np.ndarray:
    """J2 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 2)


def accel_J3(r: np.ndarray) -> np.ndarray:
    """J3 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 3)


def accel_J4(r: np.ndarray) -> np.ndarray:
    """J4 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 4)


def accel_J5(r: np.ndarray) -> np.ndarray:
    """J5 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 5)


def accel_J6(r: np.ndarray) -> np.ndarray:
    """J6 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 6)


def accel_J7(r: np.ndarray) -> np.ndarray:
    """J7 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 7)


def accel_J8(r: np.ndarray) -> np.ndarray:
    """J8 zonal acceleration (m/s^2) at position ``r`` (m)."""
    return _accel_zonal(r, 8)


def accel_earth_harmonics(r: np.ndarray) -> np.ndarray:
    """
    Central gravity + J2..J8 zonal harmonics.
    """
    r = position(r)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    a_central = -EARTH_MU * r / (r_mag**3)
    return a_central + sum(_accel_zonal(r, n) for n in range(2, 9))
