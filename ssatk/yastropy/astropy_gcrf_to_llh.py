import erfa
import numpy as np

_WGS84 = 1  # erfa ellipsoid identifier for gc2gd


def astropy_gcrf_to_llh(r_gcrf, t, *, allow_predicted=True):
    """
    Convert GCRF Cartesian coordinates to geodetic longitude, latitude, altitude.

    Parameters:
    -----------
    r_gcrf : ndarray, shape (N, 3)
        GCRF Cartesian coordinates in meters.
    t : Time
        UTC astropy time for the conversion; scalar, or one per row of ``r_gcrf``.
    allow_predicted : bool, optional
        Forwarded to the packaged EOP table; default ``True`` (see
        ``ssatk.coordinates.earth_fixed``'s module docstring).

    Returns:
    --------
    tuple : (longitude, latitude, altitude)
        Longitude, latitude in degrees, altitude in meters above the WGS84
        ellipsoid.

    Despite the "astropy" name, this shares SSATK's IAU 2006/2000A route
    (:mod:`ssatk.coordinates._earth_rotation`) rather than calling Astropy's
    GCRS/ITRS transform directly, so it uses SSATK's packaged EOP data
    instead of Astropy's own (network-backed) IERS table.
    """
    # Deferred: ssatk.coordinates imports this package (for astropy_surface_rv),
    # so importing it at module load time would be circular.
    from ..coordinates._earth_rotation import gcrs_to_itrs_matrix
    from ..coordinates.earth_fixed import _time_array
    from ..time_functions import to_gps

    r_gcrf = np.asarray(r_gcrf, dtype=float)
    gps = _time_array(to_gps(t), len(r_gcrf))
    rotation = gcrs_to_itrs_matrix(gps, allow_predicted=allow_predicted)
    r_itrf = np.einsum("nij,nj->ni", rotation, r_gcrf)

    longitude, latitude, altitude = erfa.gc2gd(_WGS84, r_itrf)
    return np.degrees(longitude), np.degrees(latitude), altitude
