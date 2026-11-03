"""Stage-3 references: WGS84, Astropy, rotation invariants and 6-DoF kinematics."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers

from ssatk.coordinates.attitude import rotate_vector
from ssatk.coordinates.earth_fixed import gcrf_to_itrf_eop, itrf_to_gcrf_eop
from ssatk.data import data_path
from ssatk.launch.preflight import launch_initial_state
from ssatk.launch.sites import launch_pads
from ssatk.propagators_6dof.sixdof import sixdof_rhs

SITE = "Kennedy Space Center LC-39A"
EPOCH = "2020-05-30T19:22:00"


@pytest.mark.parametrize("latitude_deg", [-90.0, 0.0, 28.6082, 90.0])
@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 210.0])
def test_pad_position_vertical_and_heading_match_wgs84(monkeypatch, latitude_deg, heading_deg):
    """R1: WGS84 ellipsoid within 1 nm; analytic ENU/heading within 2e-15 rad."""
    longitude_deg, height_m = -80.6041, 123.0
    monkeypatch.setitem(launch_pads, "synthetic pad", {"latitude": latitude_deg, "longitude": longitude_deg})
    state = launch_initial_state("synthetic pad", EPOCH, heading_deg=heading_deg, height_m=height_m)
    latitude, longitude, heading = np.deg2rad([latitude_deg, longitude_deg, heading_deg])
    equatorial_radius, flattening = 6378137.0, 1 / 298.257223563
    eccentricity_squared = flattening * (2 - flattening)
    normal_radius = equatorial_radius / np.sqrt(1 - eccentricity_squared * np.sin(latitude)**2)
    position_longitude = np.deg2rad(longitude_deg % 360.0)
    expected_position = [
        (normal_radius + height_m) * np.cos(latitude) * np.cos(position_longitude),
        (normal_radius + height_m) * np.cos(latitude) * np.sin(position_longitude),
        (normal_radius * (1 - eccentricity_squared) + height_m) * np.sin(latitude),
    ]
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array([-np.sin(latitude)*np.cos(longitude), -np.sin(latitude)*np.sin(longitude), np.cos(latitude)])
    up = np.array([np.cos(latitude)*np.cos(longitude), np.cos(latitude)*np.sin(longitude), np.sin(latitude)])
    body_to_itrf = np.asarray(state["enu_to_itrf"]) @ state["body_to_enu"]
    expected_forward = np.cos(heading) * north + np.sin(heading) * east
    np.testing.assert_allclose(state["itrf_position_m"], expected_position, atol=1e-9, rtol=0)
    np.testing.assert_allclose(body_to_itrf[:, 0], expected_forward, atol=2e-15, rtol=0)
    np.testing.assert_allclose(body_to_itrf[:, 2], up, atol=2e-15, rtol=0)
    np.testing.assert_allclose(body_to_itrf[:, 1], np.cross(up, expected_forward), atol=2e-15, rtol=0)
    assert state["height_datum"] == "WGS84 ellipsoidal"
    assert state["terrain_height_m"] is None
    assert state["geocentric_radius_m"] == pytest.approx(np.linalg.norm(expected_position), abs=1e-9)


@pytest.mark.parametrize("epoch_utc", [EPOCH, "2024-04-08T18:00:00", "2026-07-20T00:00:00"])
def test_initialization_matches_independent_astropy_pad_and_rotation(epoch_utc):
    """R2: independent Astropy IAU 2006 route, 1 m / 1 mm/s / 1e-7 rad budgets.

    Astropy selects merged IERS-B estimates and omits dX/dY; SSATK selects
    IERS-A columns and applies dX/dY. These are model/data comparison budgets.
    """
    state = launch_initial_state(SITE, epoch_utc, heading_deg=90.0)
    epoch = Time(epoch_utc, scale="utc")
    with data_path("environment/eop/finals2000A.all") as path:
        reference_eop = iers.IERS_A.open(str(path))
    location = EarthLocation.from_geodetic(state["longitude_deg"]*u.deg,
                                         state["latitude_deg"]*u.deg, state["height_m"]*u.m)
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(reference_eop):
        position, velocity = location.get_gcrs_posvel(epoch)
        axes = ITRS(CartesianRepresentation(np.eye(3)*u.m), obstime=epoch).transform_to(GCRS(obstime=epoch))
    np.testing.assert_allclose(state["gcrf_position_m"], position.xyz.to_value(u.m), atol=1.0, rtol=0)
    np.testing.assert_allclose(state["gcrf_velocity_m_s"], velocity.xyz.to_value(u.m/u.s), atol=1e-3, rtol=0)
    np.testing.assert_allclose(state["itrf_to_gcrf"], axes.cartesian.xyz.value, atol=1e-7, rtol=0)


def test_frame_and_quaternion_inverse_rotations_preserve_vectors():
    """R4: proper rotations, inverse/composition and quaternion equivalence, 3e-15."""
    state = launch_initial_state(SITE, EPOCH, heading_deg=37.0)
    vector = np.array([0.2, -0.7, 0.5])
    for name in ("enu_to_itrf", "itrf_to_gcrf", "enu_to_gcrf", "body_to_enu", "body_to_gcrf"):
        matrix = np.asarray(state[name])
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=3e-15, rtol=0)
        np.testing.assert_allclose(matrix.T @ (matrix @ vector), vector, atol=3e-15, rtol=0)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=3e-15)
    expected = np.asarray(state["itrf_to_gcrf"]) @ state["enu_to_itrf"] @ state["body_to_enu"] @ vector
    np.testing.assert_allclose(rotate_vector(state["q_body_to_gcrf"], vector), expected, atol=3e-15, rtol=0)


def test_moving_point_transform_recovers_rotating_pad_local_state():
    """R4: inverse state transforms and pad-relative ENU, 5 nm / 1 pm/s."""
    state = launch_initial_state(SITE, EPOCH, heading_deg=90.0)
    enu_to_itrf = np.asarray(state["enu_to_itrf"])
    local_position = np.array([500.0, -20.0, 80.0])
    local_velocity = np.array([1.0, 2.0, -0.5])
    position = np.asarray(state["itrf_position_m"]) + enu_to_itrf @ local_position
    velocity = enu_to_itrf @ local_velocity
    inertial_position, inertial_velocity = itrf_to_gcrf_eop([position], EPOCH, v=[velocity])
    recovered_position, recovered_velocity = gcrf_to_itrf_eop(inertial_position, EPOCH, v=inertial_velocity)
    np.testing.assert_allclose(enu_to_itrf.T @ (recovered_position[0] - state["itrf_position_m"]),
                               local_position, atol=5e-9, rtol=0)
    np.testing.assert_allclose(enu_to_itrf.T @ recovered_velocity[0], local_velocity, atol=1e-12, rtol=0)


def test_pad_stationarity_and_body_rate_match_finite_difference_and_sixdof():
    """R1/R5: Earth spin within 2e-11 rad/s; full-model derivative within 1 mm/s.

    Compare full matrix differences at a five-second half-width, including
    precession, nutation, polar motion and interpolated UT1.
    """
    epoch = Time(EPOCH, scale="utc")
    before, state, after = [launch_initial_state(SITE, epoch + offset*u.s, heading_deg=35.0)
                            for offset in (-5.0, 0.0, 5.0)]
    matrix = np.asarray(state["body_to_gcrf"])
    derivative = (np.asarray(after["body_to_gcrf"]) - before["body_to_gcrf"]) / 10.0
    skew = matrix.T @ derivative
    expected_omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    np.testing.assert_allclose(state["omega_body_rad_s"], expected_omega, atol=2e-11, rtol=0)
    assert np.linalg.norm(state["omega_body_rad_s"]) == pytest.approx(7.292115e-5, abs=2e-11)
    expected_velocity = (np.asarray(after["gcrf_position_m"]) - before["gcrf_position_m"]) / 10.0
    np.testing.assert_allclose(state["gcrf_velocity_m_s"], expected_velocity, atol=1e-3, rtol=0)
    samples = (before, state, after)
    position, velocity = gcrf_to_itrf_eop(
        [sample["gcrf_position_m"] for sample in samples],
        [sample["epoch_gps_s"] for sample in samples],
        v=[sample["gcrf_velocity_m_s"] for sample in samples],
    )
    np.testing.assert_allclose(position, [state["itrf_position_m"]]*3, atol=5e-9, rtol=0)
    np.testing.assert_allclose(velocity, np.zeros((3, 3)), atol=1e-12, rtol=0)
    quaternions = [np.asarray(sample["q_body_to_gcrf"]) for sample in (before, state, after)]
    for quaternion in (quaternions[0], quaternions[2]):
        if quaternion @ quaternions[1] < 0.0:
            quaternion *= -1.0
    state_vector = np.concatenate([state["gcrf_position_m"], state["gcrf_velocity_m_s"],
                                   state["q_body_to_gcrf"], state["omega_body_rad_s"]])
    rhs = sixdof_rhs(state["epoch_gps_s"], state_vector, inertia=np.eye(3), mu=0.0)
    np.testing.assert_allclose(rhs[6:10], (quaternions[2]-quaternions[0])/10.0, atol=2e-11, rtol=0)


def test_leap_second_time_boundary_and_rotation_match_iers_a_and_erfa():
    """R1/R2: UTC leap maps to consecutive GPS/TAI/TT seconds (0.5 ns).

    Independent raw IERS-A UT1 interpolation plus ERFA c2t06a, 5e-9 rad
    rotation budget for the celestial-pole corrections c2t06a omits.
    """
    import erfa

    epochs = Time(["2016-12-31T23:59:59", "2016-12-31T23:59:60", "2017-01-01T00:00:00"], scale="utc")
    with data_path("environment/eop/finals2000A.all") as path:
        reference = iers.IERS_A.open(str(path))
    left_index = np.flatnonzero(reference["MJD"].value == 57753)[0]
    right_index = left_index + 1
    start = Time("2016-12-31T00:00:00", scale="utc").gps
    end = Time("2017-01-01T00:00:00", scale="utc").gps
    tai = Time(["2017-01-01T00:00:35", "2017-01-01T00:00:36", "2017-01-01T00:00:37"], scale="tai")
    samples = []
    for index, epoch in enumerate(epochs):
        fraction = (epoch.gps - start) / (end - start)
        dut1_left = reference["UT1_UTC_A"][left_index].to_value(u.s)
        dut1_right = reference["UT1_UTC_A"][right_index].to_value(u.s)
        epoch.delta_ut1_utc = dut1_left + fraction * (dut1_right - 1.0 - dut1_left) if index < 2 else dut1_right
        xp = ((1-fraction)*reference["PM_x_A"][left_index] + fraction*reference["PM_x_A"][right_index]).to_value(u.rad)
        yp = ((1-fraction)*reference["PM_y_A"][left_index] + fraction*reference["PM_y_A"][right_index]).to_value(u.rad)
        expected = erfa.c2t06a(epoch.tt.jd1, epoch.tt.jd2, epoch.ut1.jd1, epoch.ut1.jd2, xp, yp)
        state = launch_initial_state(SITE, epoch, heading_deg=0.0)
        np.testing.assert_allclose(state["itrf_to_gcrf"], expected.T, atol=5e-9, rtol=0)
        assert abs((epoch.tai - tai[index]).to_value(u.s)) < 5e-10
        assert state["epoch_tai"] == tai[index].isot
        assert state["epoch_tt"] == tai[index].tt.isot
        assert state["epoch_ut1"] == epoch.ut1.isot
        samples.append(state)
    np.testing.assert_allclose(np.diff([sample["epoch_gps_s"] for sample in samples]), [1.0, 1.0], atol=0, rtol=0)
    for before, after in zip(samples, samples[1:]):
        angular_step = np.asarray(after["itrf_to_gcrf"]) @ np.asarray(before["itrf_to_gcrf"]).T
        assert np.arctan2(angular_step[1, 0], angular_step[0, 0]) == pytest.approx(7.292115e-5, abs=2e-10)
