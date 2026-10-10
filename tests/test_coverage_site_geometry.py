import numpy as np
import pytest

from ssatk.plots.coverage_analysis import elevation_from_site, site_ecef

A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563


def _wgs84_ecef(lat_deg, lon_deg, h):
    phi, lam = np.radians(lat_deg), np.radians(lon_deg)
    e2 = F_WGS84 * (2.0 - F_WGS84)
    n = A_WGS84 / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    return np.array([(n + h) * np.cos(phi) * np.cos(lam), (n + h) * np.cos(phi) * np.sin(lam), (n * (1.0 - e2) + h) * np.sin(phi)])


@pytest.mark.parametrize("lat, lon, h", [(37.68, -121.77, 200.0), (-33.0, 151.2, 50.0), (78.2, 15.4, 500.0), (0.0, 0.0, 0.0)])
def test_site_position_and_look_angles_use_the_wgs84_ellipsoid(lat, lon, h):
    # R1: WGS84 geodetic -> ECEF, x = (N + h) cos(phi) cos(lam), z = (N (1 - e^2) + h) sin(phi),
    # to 1 mm. A target along the geodetic normal is at elevation 90 deg; targets
    # along local east and north are at elevation 0 and azimuth 90 / 0 deg (1e-6 deg).
    site = site_ecef(lat, lon, h)
    np.testing.assert_allclose(site, _wgs84_ecef(lat, lon, h), rtol=0, atol=1e-3)

    phi, lam = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.cross(up, east)
    targets = np.array([site + 500e3 * up, site + 1000e3 * east, site + 1000e3 * north])
    el, az, dist = elevation_from_site(targets, site)
    assert el[0] == pytest.approx(90.0, abs=1e-6)
    np.testing.assert_allclose(el[1:], 0.0, atol=1e-6)
    assert az[1] == pytest.approx(90.0, abs=1e-6)
    assert min(az[2], 360.0 - az[2]) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(dist, [500e3, 1000e3, 1000e3], rtol=1e-12)
