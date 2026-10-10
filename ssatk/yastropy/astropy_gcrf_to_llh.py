from astropy.coordinates import GCRS, ITRS
from astropy.coordinates.representation import CartesianRepresentation
from astropy import units as u


def astropy_gcrf_to_llh(r_gcrf, t):
    """
    Convert GCRF Cartesian coordinates to latitude, longitude, and altitude.
    
    Parameters:
    -----------
    gcrf_x, gcrf_y, gcrf_z : float
        GCRF Cartesian coordinates in meters
    t : Time
        UTC astrpy time for the conversion
    
    Returns:
    --------
    tuple : (latitude, longitude, altitude)
        Longitude, Latitude in degrees, altitude in meters above ellipsoid
    """
    
    x = r_gcrf[:, 0] * u.m
    y = r_gcrf[:, 1] * u.m
    z = r_gcrf[:, 2] * u.m

    # Create CartesianRepresentation object
    cart_repr = CartesianRepresentation(
        x=x,
        y=y,
        z=z
    )
    
    # Create GCRS coordinate object using the representation
    gcrs_coord = GCRS(cart_repr, obstime=t)
    
    # Transform to ITRS (Earth-fixed)
    itrs_coord = gcrs_coord.transform_to(ITRS(obstime=t))
    
    # Convert to geodetic coordinates
    geodetic = itrs_coord.earth_location
    
    # Extract longitude, latitude, and altitude
    latitude = geodetic.lat.degree
    longitude = geodetic.lon.degree
    altitude = geodetic.height.to(u.m).value
    
    return longitude, latitude, altitude
