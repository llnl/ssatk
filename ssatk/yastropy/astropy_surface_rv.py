from astropy.time import Time


def astropy_surface_rv(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc")):
    """
    Get GCRF position and velocity of a surface observer at a specific location and time.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees.
    lat : float
        Geodetic latitude in degrees.
    elevation : float, optional
        Elevation above sea level in meters (default is 0).
    t : Time
        Astropy Time object.

    Returns
    -------
    r : ndarray, shape (3,)
        Position vector in GCRF coordinates (m).
    v : ndarray, shape (3,)
        Velocity vector in GCRF coordinates (m/s).

    Uses :func:`ssatk.coordinates.geodetic.llh_to_gcrf`: a WGS84 site fixed
    in ITRF, transformed with the shared IAU 2006/2000A rotation and its
    time derivative. Epochs outside the packaged EOP table use the nearest
    record and emit an EarthOrientationExtrapolationWarning.
    """
    # Geodetic imports this module, so resolve the shared route at call time.
    from ..coordinates.geodetic import llh_to_gcrf

    return llh_to_gcrf(lon, lat, t, height=elevation)
