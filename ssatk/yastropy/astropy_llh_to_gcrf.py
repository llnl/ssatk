from astropy.coordinates import GCRS, ITRS
from astropy import units as u
import numpy as np


def astropy_llh_to_gcrf(lon, lat, t, alt=0):
    """
    Convert latitude, longitude, altitude to GCRF Cartesian coordinates.
    
    Parameters:
    -----------
    lon, lat : float
        Latitude and longitude in degrees
    alt : float
        Altitude in meters above ellipsoid
    t : Time
        UTC astropy time for the conversion
    
    Returns:
    --------
    tuple : (x, y, z)
        GCRF Cartesian coordinates in meters
    """
    from astropy.coordinates import EarthLocation
    
    # Create EarthLocation
    earth_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)
    
    # Convert to ITRS
    itrs_coord = ITRS(earth_loc.geocentric, obstime=t)
    
    # Transform to GCRS
    gcrs_coord = itrs_coord.transform_to(GCRS(obstime=t))
    
    # Extract Cartesian coordinates
    x = gcrs_coord.cartesian.x.to(u.m).value
    y = gcrs_coord.cartesian.y.to(u.m).value
    z = gcrs_coord.cartesian.z.to(u.m).value
    
    return np.column_stack((x, y, z))
