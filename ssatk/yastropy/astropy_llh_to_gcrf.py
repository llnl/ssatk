import erfa
import numpy as np

from ..time_functions import to_gps

_WGS84 = 1  # erfa ellipsoid identifier for gd2gc


def astropy_llh_to_gcrf(lon, lat, t, alt=0, *, allow_predicted=True):
    """
    Convert geodetic latitude, longitude, altitude to GCRF Cartesian coordinates.

    Parameters:
    -----------
    lon, lat : float or array-like
        Longitude and latitude in degrees (same length).
    alt : float or array-like
        Altitude in meters above the WGS84 ellipsoid.
    t : Time
        UTC astropy time for the conversion; scalar, or one per ``lon``/``lat``.
    allow_predicted : bool, optional
        Forwarded to the packaged EOP table; default ``True`` (see
        ``ssatk.coordinates.earth_fixed``'s module docstring).

    Returns:
    --------
    ndarray, shape (N, 3)
        GCRF Cartesian coordinates in meters.

    Despite the "astropy" name, this shares SSATK's IAU 2006/2000A route
    (:mod:`ssatk.coordinates._earth_rotation`) rather than calling Astropy's
    GCRS/ITRS transform directly, so it uses SSATK's packaged EOP data
    instead of Astropy's own (network-backed) IERS table. An epoch outside
    that table holds the nearest end record's values and warns once with
    :class:`~ssatk.environment_eop.EarthOrientationExtrapolationWarning`.
    """
    # Deferred: ssatk.coordinates imports this package (for astropy_surface_rv),
    # so importing it at module load time would be circular.
    from ..coordinates._earth_rotation import gcrs_to_itrs_matrix
    from ..coordinates.earth_fixed import _time_array

    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    alt = np.broadcast_to(np.asarray(alt, dtype=float), lon.shape)
    r_itrf = np.asarray(
        erfa.gd2gc(_WGS84, np.radians(lon), np.radians(lat), alt), dtype=float
    )

    gps = _time_array(to_gps(t), len(r_itrf))
    rotation = gcrs_to_itrs_matrix(gps, allow_predicted=allow_predicted, extrapolate=True)
    return np.einsum("nji,nj->ni", rotation, r_itrf)
