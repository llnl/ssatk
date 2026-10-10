import numpy as np
from astropy.time import Time

from ssatk.yastropy.astropy_gcrf_to_llh import astropy_gcrf_to_llh
from ssatk.yastropy.astropy_llh_to_gcrf import astropy_llh_to_gcrf
from ssatk.yastropy.astropy_surface_rv import astropy_surface_rv


def test_astropy_llh_gcrf_roundtrip_and_surface_velocity():
    time = Time("2025-01-01T00:00:00", scale="utc")
    r_gcrf = astropy_llh_to_gcrf(lon=[0.0], lat=[0.0], alt=0.0, t=time)

    assert r_gcrf.shape == (1, 3)
    lon, lat, alt = astropy_gcrf_to_llh(r_gcrf, time)
    np.testing.assert_allclose(lon, [0.0], atol=1e-9)
    np.testing.assert_allclose(lat, [0.0], atol=1e-9)
    np.testing.assert_allclose(alt, [0.0], atol=1e-6)

    r_surface, v_surface = astropy_surface_rv(lon=0.0, lat=0.0, elevation=0.0, t=time)
    assert r_surface.shape == (3,)
    assert v_surface.shape == (3,)
    assert np.isclose(np.linalg.norm(r_surface), 6_378_137.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(v_surface, np.cross([0.0, 0.0, 7.2921150e-5], r_surface))
