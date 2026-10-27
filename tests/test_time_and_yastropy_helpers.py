import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
from ssatk.data import data_path
from ssatk.yastropy.astropy_gcrf_to_llh import astropy_gcrf_to_llh
from ssatk.yastropy.astropy_llh_to_gcrf import astropy_llh_to_gcrf
from ssatk.yastropy.astropy_surface_rv import astropy_surface_rv


def test_astropy_llh_gcrf_roundtrip_and_surface_velocity():
    time = Time("2025-01-01T00:00:00", scale="utc")
    r_gcrf = astropy_llh_to_gcrf(lon=[0.0], lat=[0.0], alt=0.0, t=time)

    assert r_gcrf.shape == (1, 3)
    lon, lat, alt = astropy_gcrf_to_llh(r_gcrf, time)
    np.testing.assert_allclose(lon, [0.0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(lat, [0.0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(alt, [0.0], atol=1e-6, rtol=0)

    r_surface, v_surface = astropy_surface_rv(lon=0.0, lat=0.0, elevation=0.0, t=time)
    assert r_surface.shape == (3,)
    assert v_surface.shape == (3,)
    assert np.isclose(np.linalg.norm(r_surface), 6_378_137.0, rtol=0, atol=1e-6)

    # R2: Astropy's independent transform, scoped to the packaged EOP.
    # Use the core route's 2e-4 m/s budget for finite-difference roundoff
    # and Astropy's omission of the celestial-pole offsets.
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        reference = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg).get_gcrs(time)
    v_ref = reference.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s)
    np.testing.assert_allclose(v_surface, v_ref, rtol=0, atol=2e-4)
