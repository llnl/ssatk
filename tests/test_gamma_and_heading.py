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
    # all samples including endpoints. A prograde orbit in the GCRF equator heads east within
    # 0.5 deg, the tilt between the GCRF and ITRF equators in 2026.
    radius = 7000e3
    n = np.sqrt(EARTH_MU / radius**3)
    t_gps = Time("2026-10-08T00:00:00", scale="utc").gps + np.arange(0.0, 600.0, 10.0)
    phase = n * (t_gps - t_gps[0])
    r = radius * np.column_stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)])

    gamma, heading = calc_gamma_and_heading(r, t_gps)

    np.testing.assert_allclose(gamma, 0.0, atol=1e-3, rtol=0)
    np.testing.assert_allclose(heading, 90.0, atol=0.5, rtol=0)


@pytest.mark.parametrize("lon,lat", [(0, 0), (45, 45), (-120, -30)])
def test_radial_and_stationary_motion_have_no_compass_heading(lon, lat):
    """R1: radial velocity has gamma +/-90 and no horizontal azimuth.

    An oblique Cartesian radial vector exercises projection roundoff; angles
    allow 1e-10 degrees. Zero velocity leaves both angles undefined.
    """
    east, _, up = _local_enu(lon, lat)
    r = np.tile(7e6 * up, (4, 1))
    v = 7500 * np.array([up, -up, np.zeros(3), east])
    gamma, heading = calc_gamma_and_heading_itrf(r, np.arange(4), v=v)
    np.testing.assert_allclose(gamma[[0, 1, 3]], [90, -90, 0], atol=1e-10, rtol=0)
    assert np.isnan(gamma[2]) and np.isnan(heading[:3]).all()
    assert heading[3] == pytest.approx(90, abs=1e-10)


def test_heading_is_undefined_at_both_geographic_poles():
    """R1: longitude/north azimuth is singular at either pole, 1e-12 deg gamma."""
    r = 7e6 * np.array([[0, 0, 1], [0, 0, -1], [np.cos(np.pi / 2), 0, 1]])
    gamma, heading = calc_gamma_and_heading_itrf(
        r, [0, 1, 2], v=np.tile([7500, 0, 0], (3, 1))
    )
    assert np.isnan(heading).all()
    np.testing.assert_allclose(gamma, 0, atol=1e-12, rtol=0)


def test_irregular_time_derivatives_match_an_analytic_quadratic():
    """R1: d/dt [R+t²,2t²,-3t²]=[2t,4t,-6t], all endpoints, 5e-9 m/s."""
    from ssatk.coordinates import v_from_r

    t = np.array([0.0, 1.0, 4.0, 10.0])
    r = np.column_stack((7e6 + t * t, 2 * t * t, -3 * t * t))
    expected = t[:, None] * [2, 4, -6]
    np.testing.assert_allclose(
        v_from_r(r, Time(Time("2025-01-01").gps + t, format="gps")),
        expected,
        atol=5e-9,
        rtol=0,
    )
    np.testing.assert_allclose(
        v_from_r([[7e6, 0, 0], [7e6 + 20, 40, 60]], [0, 2]),
        [[10, 20, 30], [10, 20, 30]],
        atol=1e-12,
        rtol=0,
    )
    stationary = np.tile([7e6, 2e6, 3e6], (4, 1))
    np.testing.assert_array_equal(v_from_r(stationary, t), np.zeros((4, 3)))
    gamma, heading = calc_gamma_and_heading_itrf(stationary, t)
    assert np.isnan(gamma).all() and np.isnan(heading).all()


@pytest.mark.parametrize("times", [[0, 1, 1], [0, 2, 1]])
def test_velocity_rejects_epochs_without_a_forward_time_derivative(times):
    """C: repeated/reversed epochs cannot define the documented derivative."""
    from ssatk.coordinates import v_from_r

    with pytest.raises(ValueError, match="strictly increasing"):
        v_from_r(np.ones((3, 3)), times)


def test_supplied_states_match_astropy_spherical_velocity_angles():
    """R2: Astropy GCRS->ITRS spherical differentials, 5e-4 deg.

    Widely spaced, independent states avoid making the reference a chord
    derivative. The allowance covers SSAPy IAU 1976/80 versus Astropy 2006/2000A.
    """
    import astropy.units as u
    from astropy.coordinates import (
        GCRS,
        ITRS,
        CartesianRepresentation,
        CartesianDifferential,
        SphericalRepresentation,
        SphericalDifferential,
    )
    from ssatk.orbital_mechanics.gamma_and_heading import calc_gamma

    times = Time(["2025-01-01", "2025-01-01T01:00:00", "2025-06-21"])
    r = np.array([[7e6, 1e6, -5e5], [-4e6, 5e6, 3e6], [1e6, -7e6, 2e6]])
    v = np.array([[-1000, 7300, 1500], [-3000, -4000, 5000], [6500, 2000, -2500]])
    rep = CartesianRepresentation(
        r.T * u.m, differentials=CartesianDifferential(v.T * u.m / u.s)
    )
    spherical = (
        GCRS(rep, obstime=times)
        .transform_to(ITRS(obstime=times))
        .represent_as(SphericalRepresentation, SphericalDifferential)
    )
    rates = spherical.differentials["s"]
    east = (spherical.distance * np.cos(spherical.lat) * rates.d_lon).to_value(
        u.m / u.s, equivalencies=u.dimensionless_angles()
    )
    north = (spherical.distance * rates.d_lat).to_value(
        u.m / u.s, equivalencies=u.dimensionless_angles()
    )
    expected_gamma = np.degrees(
        np.arctan2(rates.d_distance.to_value(u.m / u.s), np.hypot(east, north))
    )
    expected_heading = np.degrees(np.arctan2(east, north)) % 360
    gamma, heading = calc_gamma_and_heading(r, times, v=v)
    np.testing.assert_allclose(gamma, expected_gamma, atol=5e-4, rtol=0)
    np.testing.assert_allclose(
        (heading - expected_heading + 180) % 360 - 180, 0, atol=5e-4, rtol=0
    )
    np.testing.assert_allclose(
        calc_gamma(r, times, v=v), expected_gamma, atol=5e-4, rtol=0
    )
