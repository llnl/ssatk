import numpy as np
import pytest
from astropy.time import Time

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.gamma_and_heading import (
    calc_gamma_and_heading,
    calc_gamma_and_heading_itrf,
    calc_heading_itrf,
)


def _local_enu(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    return east, north, up


def _velocity(lon_deg, lat_deg, gamma_deg, heading_deg, speed=7500.0):
    east, north, up = _local_enu(lon_deg, lat_deg)
    gamma, heading = np.radians(gamma_deg), np.radians(heading_deg)
    horizontal = np.cos(gamma) * (np.sin(heading) * east + np.cos(heading) * north)
    return speed * (horizontal + np.sin(gamma) * up)


CASES = [  # (lon, lat, gamma, heading) in degrees
    (0.0, 0.0, 0.0, 90.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 60.0),
    (90.0, 0.0, 0.0, 90.0),
    (-121.7, 37.7, 0.0, 180.0),
    (150.0, -45.0, 0.0, 270.0),
    (30.0, 45.0, 12.0, 315.0),
    (-60.0, -20.0, -25.0, 135.0),
]


@pytest.mark.parametrize("lon, lat, gamma, heading", CASES)
def test_heading_matches_local_east_north_construction(lon, lat, gamma, heading):
    # R1: velocity built from its local east/north/up components; heading is
    # clockwise from north, recovered to 1e-9 deg.
    _east, _north, up = _local_enu(lon, lat)
    r = 7000e3 * up
    result = calc_heading_itrf(r[None, :], _velocity(lon, lat, gamma, heading)[None, :])
    assert result[0] == pytest.approx(heading, abs=1e-9)


@pytest.mark.parametrize("lon, lat, gamma, heading", CASES)
def test_gamma_is_positive_when_climbing_on_a_straight_track(lon, lat, gamma, heading):
    # R1: for straight-line ITRF motion the forward difference at the first
    # sample is exact, so gamma (sin gamma = r_hat . v_hat) and heading at that
    # sample are recovered to 1e-9 deg.
    _east, _north, up = _local_enu(lon, lat)
    r0 = 7000e3 * up
    v = _velocity(lon, lat, gamma, heading)
    t = np.arange(3, dtype=float)
    gamma_out, heading_out = calc_gamma_and_heading_itrf(r0 + t[:, None] * v, t)
    assert gamma_out[0] == pytest.approx(gamma, abs=1e-9)
    assert heading_out[0] == pytest.approx(heading, abs=1e-9)


def test_prograde_equatorial_circular_orbit_heads_east_at_zero_gamma():
    # R1/R4: a circular GCRF orbit has no radial velocity, and the Earth-fixed
    # velocity v - w x r keeps it horizontal, so gamma = 0 within 1e-3 deg at
    # interior samples. A prograde orbit in the GCRF equator heads east within
    # 0.5 deg, the tilt between the GCRF and ITRF equators in 2026.
    radius = 7000e3
    n = np.sqrt(EARTH_MU / radius**3)
    t_gps = Time("2026-10-08T00:00:00", scale="utc").gps + np.arange(0.0, 600.0, 10.0)
    phase = n * (t_gps - t_gps[0])
    r = radius * np.column_stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)])

    gamma, heading = calc_gamma_and_heading(r, t_gps)

    np.testing.assert_allclose(gamma[1:-1], 0.0, atol=1e-3)
    np.testing.assert_allclose(heading[1:-1], 90.0, atol=0.5)
