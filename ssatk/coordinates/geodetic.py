import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
from astropy.time import Time
from ssapy import groundTrack
from ssapy.orbit import EarthObserver, Orbit

from ..constants import EARTH_MU, EARTH_RADIUS
from ..time_functions import to_gps
from ..yastropy import astropy_surface_rv


def gcrf_to_llh(r_gcrf, t):
    """Convert GCRF position(s) to geodetic ``(lon_deg, lat_deg, height_m)``."""
    r_input = np.asarray(r_gcrf)
    r = np.atleast_2d(r_input)
    if r.shape[1] != 3:
        raise ValueError("r_gcrf must be shape (3,) or (N,3)")

    cart = CartesianRepresentation(
        x=r[:, 0] * u.m,
        y=r[:, 1] * u.m,
        z=r[:, 2] * u.m,
    )
    itrs = GCRS(cart, obstime=t).transform_to(ITRS(obstime=t))
    xyz = itrs.cartesian.xyz.to(u.m).value.T
    loc = EarthLocation(x=xyz[..., 0] * u.m, y=xyz[..., 1] * u.m, z=xyz[..., 2] * u.m)

    lon = loc.lon.to_value(u.deg)
    lat = loc.lat.to_value(u.deg)
    height = loc.height.to_value(u.m)
    if r_input.ndim == 1:
        return lon.item(), lat.item(), height.item()
    return lon, lat, height


def gcrf_to_lonlat(r_gcrf: np.ndarray, t: np.ndarray):
    """Convert GCRF position(s) to ``(lon_deg, lat_deg, height_m)`` using SSAPy."""
    t = np.atleast_1d(t)
    r_gcrf = np.atleast_2d(r_gcrf)
    lon, lat, height = groundTrack(r_gcrf, to_gps(t), format="geodetic")
    return np.degrees(lon), np.degrees(lat), height


def llh_to_gcrf(lon: float, lat: float, t: Time, height: float = 0.0):
    """Convert geodetic ``(lon_deg, lat_deg, height_m)`` to GCRF position/velocity."""
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=height * u.m)
    gcrs = loc.get_gcrs(obstime=t)
    r_gcrf = gcrs.cartesian.xyz.to(u.m).value
    v_gcrf = gcrs.cartesian.differentials["s"].d_xyz.to(u.m / u.s).value
    return r_gcrf, v_gcrf


def surface_rv(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc")):
    """Return GCRF position and velocity of a surface point from Astropy."""
    return astropy_surface_rv(lon=lon, lat=lat, elevation=elevation, t=t)


def surface_rv_ssapy(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc"), fast=False):
    """Return GCRF position and velocity of a surface point from SSAPy."""
    observer = EarthObserver(lon=lon, lat=lat, elevation=elevation, fast=fast)
    return observer.getRV(to_gps(t))


def bbox_min(lons, lats):
    """
    Compute the minimal bounding box for points given by lats/lons.
    Returns ``(lat_min, lat_max, lon_left, lon_right, lon_span_deg)``.
    """
    if lons is None or lats is None:
        raise ValueError("lons and lats must be provided")
    lons = list(lons)
    lats = list(lats)
    if len(lons) != len(lats) or len(lons) == 0:
        raise ValueError("lons and lats must be same nonzero length")

    lat_min = min(lats)
    lat_max = max(lats)
    angles = sorted(((x % 360) + 360) % 360 for x in lons)
    if len(angles) == 1:
        lon = ((angles[0] + 180.0) % 360.0) - 180.0
        return lat_min, lat_max, lon, lon, 0.0

    gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % 360.0 for i in range(len(angles))]
    gap_index = max(range(len(gaps)), key=lambda i: gaps[i])
    span = 360.0 - gaps[gap_index]
    left = angles[(gap_index + 1) % len(angles)]
    right = left + span
    norm180 = lambda x: ((x + 180.0) % 360.0) - 180.0
    return lat_min, lat_max, norm180(left), norm180(right), span


def lonlat_distance(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Great-circle distance in meters between two points on a sphere.

    All four angles are in radians, in the order ``(lat1, lat2, lon1, lon2)``.
    Uses the haversine formula on a sphere of radius ``EARTH_RADIUS``
    (6378.137 km, the WGS84 equatorial radius), so it is not a geodesic on
    the ellipsoid.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * EARTH_RADIUS


def lonlat_perigee(lon, lat, t, alt=1000e3, e=0, i=None, EARTH_MU=EARTH_MU):
    """Create an orbit with perigee over geodetic ``(lon, lat)`` at epoch ``t``."""
    if i is None:
        i = lat

    r_gcrf, _ = astropy_surface_rv(lon, lat, t=t)
    rp = EARTH_RADIUS + alt
    r_hat = r_gcrf / np.linalg.norm(r_gcrf)
    r_peri = rp * r_hat

    def rodrigues(u, k, theta):
        return (
            u * np.cos(theta)
            + np.cross(k, u) * np.sin(theta)
            + k * (np.dot(k, u)) * (1 - np.cos(theta))
        )

    h_hat = rodrigues(np.array([0.0, 0.0, 1.0]), r_hat, np.deg2rad(i))
    h_hat /= np.linalg.norm(h_hat)
    v_hat = np.cross(h_hat, r_hat)
    v_hat /= np.linalg.norm(v_hat)
    v_peri = v_hat * np.sqrt(EARTH_MU * (1 + e) / rp)

    return Orbit(r=r_peri, v=v_peri, t=t, mu=EARTH_MU)
