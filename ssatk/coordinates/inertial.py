import erfa
import numpy as np


def j2000_to_gcrf(pos_j2000, obstime=None):
    """
    Convert n x 3 array of J2000 (EME2000) positions to GCRF coordinates.

    EME2000 is the mean equator and equinox of J2000.0, a fixed frame. It
    differs from the GCRF only by the constant IAU 2006 frame bias, a
    23 mas rotation (IERS Conventions 2010, eqs. 5.20-5.21:
    xi0 = -16.6170 mas, eta0 = -6.8192 mas, d_alpha0 = -14.6 mas), so
    ``r_GCRF = B^T r_J2000``. No precession or nutation is involved.

    Parameters:
    -----------
    pos_j2000 : ndarray
        n x 3 array of x, y, z positions in J2000 coordinates (meters).
    obstime : optional
        Ignored. Kept for backward compatibility: the transformation does
        not depend on time.

    Returns:
    --------
    pos_gcrf : ndarray
        n x 3 array of x, y, z positions in GCRF coordinates (meters).

    Raises:
    -------
    ValueError
        If pos_j2000 is not an n x 3 array.
    """
    pos_j2000 = np.asarray(pos_j2000, dtype=float)
    if pos_j2000.ndim != 2 or pos_j2000.shape[1] != 3:
        raise ValueError("Input must be an n x 3 array of positions.")

    # erfa.bp06 returns the frame bias B (GCRS -> mean J2000) as its first matrix.
    frame_bias, _precession, _bias_precession = erfa.bp06(2451545.0, 0.0)
    return pos_j2000 @ frame_bias
