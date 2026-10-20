"""Idealized Walker-delta constellations built from SSAPy orbit states."""
import numpy as np
from astropy.time import Time
from ssapy import Orbit
from ..constants import EARTH_RADIUS, EARTH_MU


def walker_delta(*, planes=72, satellites_per_plane=22, altitude_m=550e3,
                 inclination_deg=53., phasing=1, epoch="2025-01-01T00:00:00"):
    """Return a vector SSAPy Orbit containing plane-major circular GCRF states.

    A design constellation, not operational Starlink TLEs. RAANs span 360 deg;
    the phase step between adjacent planes is ``360*phasing/total`` degrees.
    Propagate the returned objects with :func:`ssapy.rv`. Altitude is above
    the SSATK spherical Earth radius, and epoch strings are UTC.
    """
    for name, value in (("planes", planes), ("satellites_per_plane", satellites_per_plane)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(phasing, bool) or int(phasing) != phasing or not 0 <= phasing < planes:
        raise ValueError("phasing must be an integer in [0, planes)")
    if not np.isfinite(altitude_m) or altitude_m <= 0:
        raise ValueError("altitude_m must be finite and positive")
    if not np.isfinite(inclination_deg) or not 0 <= inclination_deg <= 180:
        raise ValueError("inclination_deg must lie in [0, 180]")
    epoch = epoch if isinstance(epoch, Time) else Time(epoch, scale="utc")
    if not epoch.isscalar or not np.isfinite(epoch.gps):
        raise ValueError("epoch must be a finite scalar time")
    total = int(planes*satellites_per_plane)
    states = [Orbit.fromKeplerianElements(
        a=EARTH_RADIUS+altitude_m, e=0., i=np.deg2rad(inclination_deg), pa=0.,
        raan=2*np.pi*plane/planes,
        trueAnomaly=2*np.pi*(slot/satellites_per_plane+phasing*plane/total),
        t=epoch.gps, mu=EARTH_MU,
    ) for plane in range(int(planes)) for slot in range(int(satellites_per_plane))]
    return Orbit(r=np.array([state.r for state in states]),
                 v=np.array([state.v for state in states]), t=epoch.gps, mu=EARTH_MU)
