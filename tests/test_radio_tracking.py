"""Vacuum tracking references: analytic light time, count Doppler and derivatives."""

from dataclasses import replace

import numpy as np
import pytest

from ssatk.constants import c
from ssatk.navigation import measurements
from ssatk.navigation import (
    GroundStation, GroundStationMeasurement, GroundStationSensor,
    RadioTrackingConfig, solve_batch_orbit,
)
from ssatk.propagators_orbit import propagate_orbit_state_with_stm


@pytest.fixture
def stationary_origin(monkeypatch):
    monkeypatch.setattr(measurements, "_earth_state", lambda time, offset, kernel: (np.zeros(3), np.zeros(3)))


@pytest.fixture
def fixed_station(monkeypatch, stationary_origin):
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (np.zeros(3), np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: tuple(np.eye(3)))
    return GroundStation(0, 0, min_elevation_rad=-np.pi / 2)


def _trajectory(state, t0, end, **kwargs):
    return propagate_orbit_state_with_stm(
        r0=state[:3], v0=state[3:], t0=t0, times=[t0, end], mu=0,
        dense_output=True, rtol=1e-12, atol=1e-9, **kwargs,
    )


def test_radial_light_time_delay_and_doppler_closed_form(fixed_station, gps_epoch):
    """R1: constant radial speed, 1 micrometre range and 1 microhertz Doppler."""
    radius, speed, elapsed = 20_000_000.0, 3000.0, 600.0
    radio = RadioTrackingConfig(7.2e9, 10.0, 880 / 749, 2e-6)
    state = np.array([radius, 0, 0, speed, 0, 0])
    trajectory = _trajectory(state, gps_epoch, gps_epoch + elapsed)
    range_prediction = fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_range", radio)
    expected = (radius + speed * elapsed) * c / (c + speed) + (c - speed) * radio.transponder_delay_s / 2
    assert range_prediction.value == pytest.approx(expected, abs=1e-6)
    expected_jacobian = np.array([[c / (c + speed), 0, 0,
                                  c * (elapsed * c - radius) / (c + speed)**2 - radio.transponder_delay_s / 2, 0, 0]])
    np.testing.assert_allclose(range_prediction.jacobian, expected_jacobian, atol=1e-8, rtol=0)
    doppler = fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_doppler", radio)
    expected_doppler = -2 * radio.turnaround_ratio * radio.uplink_frequency_hz * speed / (c + speed)
    assert doppler.value == pytest.approx(expected_doppler, abs=1e-6)
    expected_h = np.zeros((1, 6))
    expected_h[0, 3] = -2 * radio.turnaround_ratio * radio.uplink_frequency_hz * c / (c + speed)**2
    np.testing.assert_allclose(doppler.jacobian, expected_h, atol=1e-8, rtol=0)


def test_accelerating_count_doppler_closed_form(fixed_station, gps_epoch):
    """R1: quadratic radial motion, 1 micrometre range and 1 microhertz Doppler."""
    radius, speed, acceleration = 20_000_000.0, 1000.0, 2.0
    radio = RadioTrackingConfig(7.2e9, 60.0, 880 / 749, 0.01)
    elapsed = 300.0
    trajectory = _trajectory(
        np.array([radius, 0, 0, speed, 0, 0]), gps_epoch, gps_epoch + elapsed,
        acceleration=lambda t, r, v: np.array([acceleration, 0, 0]),
    )

    def reference_range(t):
        position = radius + speed * t + 0.5 * acceleration * t**2
        b = c + speed + acceleration * t
        down_delay = 2 * position / (b + np.sqrt(b**2 - 2 * acceleration * position))
        reception = t - down_delay - radio.transponder_delay_s
        up_range = radius + speed * reception + 0.5 * acceleration * reception**2
        return (c * (down_delay + radio.transponder_delay_s) + up_range) / 2

    expected_range = reference_range(elapsed)
    expected_doppler = -2 * radio.turnaround_ratio * radio.uplink_frequency_hz / (c * radio.count_interval_s) * (
        expected_range - reference_range(elapsed - radio.count_interval_s)
    )
    assert fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_range", radio).value == pytest.approx(expected_range, abs=1e-6)
    assert fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_doppler", radio).value == pytest.approx(expected_doppler, abs=1e-6)


@pytest.mark.parametrize("measurement", ["two_way_range", "two_way_doppler"])
def test_radio_initial_state_jacobian_finite_difference(monkeypatch, gps_epoch, measurement, stationary_origin):
    """R5: full moving-station signal-event derivatives, 2e-5 per SI component."""
    position = np.array([1e6, -2e6, 3e6])
    velocity = np.array([100.0, -200.0, 50.0])
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (position + velocity * (time - gps_epoch), velocity))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: tuple(np.eye(3)))
    station = GroundStation(0, 0, min_elevation_rad=-np.pi / 2)
    radio = RadioTrackingConfig(7.2e9, 10.0, 880 / 749, 0.005)
    model = GroundStationMeasurement(station, gps_epoch + 600, measurement, radio, gps_epoch,
                                    {"mu": 0, "rtol": 1e-12, "atol": 1e-9})
    state = np.array([20e6, 5e6, -4e6, -1000, 2000, 500.0])
    _, analytic = model(state)
    finite = np.empty((1, 6))
    for index, step in enumerate([1.0] * 3 + [0.01] * 3):
        plus, minus = state.copy(), state.copy()
        plus[index] += step
        minus[index] -= step
        finite[:, index] = (model(plus)[0] - model(minus)[0]) / (2 * step)
    np.testing.assert_allclose(analytic, finite, atol=2e-5, rtol=0)


def test_sensor_and_batch_against_analytic_free_flight(monkeypatch, gps_epoch, stationary_origin):
    """R1: three-station free flight, 1 mm position and 1 micrometre/s velocity."""
    stations = [GroundStation(index, 0, min_elevation_rad=-np.pi / 2) for index in range(3)]
    sites = np.array([[0, 0, 0], [0, 8e6, 0], [0, 0, 8e6]])
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (sites[int(self.lon_deg)], np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: tuple(np.eye(3)))
    truth = np.array([20e6, 5e6, 3e6, -1000.0, 2000.0, 500.0])
    radio = RadioTrackingConfig(7.2e9, 10.0, 880 / 749, 2e-6)
    trajectory = _trajectory(truth, gps_epoch, gps_epoch + 600)

    def range_reference(site, elapsed):
        relative = truth[:3] + truth[3:] * elapsed - site
        projection = relative @ truth[3:]
        delay = (relative @ relative) / (np.sqrt(projection**2 + (c**2 - truth[3:] @ truth[3:]) * (relative @ relative)) + projection)
        reception = truth[:3] + truth[3:] * (elapsed - delay - radio.transponder_delay_s)
        return 0.5 * (c * (delay + radio.transponder_delay_s) + np.linalg.norm(reception - site))

    observations = []
    for station, site in zip(stations, sites):
        for elapsed in (100.0, 300.0, 600.0):
            for measurement in ("two_way_range", "two_way_doppler"):
                reference = range_reference(site, elapsed)
                if measurement == "two_way_doppler":
                    reference = -2 * radio.turnaround_ratio * radio.uplink_frequency_hz / (c * radio.count_interval_s) * (
                        reference - range_reference(site, elapsed - radio.count_interval_s)
                    )
                sensor = GroundStationSensor(station, 0.0, measurement=measurement, radio=radio, bias=2.0, rng=0)
                observation = sensor.measure(trajectory, gps_epoch + elapsed)
                assert observation.as_measurement()[0][0] == pytest.approx(reference, abs=1e-6)
                observations.append((station, replace(observation, value=reference + 2, covariance=np.array([[1.0]]))))
    fit = solve_batch_orbit(
        observations, state0=truth + [100, -200, 300, 1, -2, 3], t0=gps_epoch,
        propagation_kwargs={"mu": 0, "rtol": 1e-12, "atol": 1e-9}, max_nfev=20,
    )
    np.testing.assert_allclose(fit.state0[:3], truth[:3], atol=1e-3, rtol=0)
    np.testing.assert_allclose(fit.state0[3:], truth[3:], atol=1e-6, rtol=0)
    assert fit.success and fit.numerical_rank == 6


def test_translating_earth_origin_closed_form(fixed_station, monkeypatch, gps_epoch):
    """R1: common radial origin velocity, 1 micrometre and 1 microhertz."""
    origin_speed, radius, speed, elapsed = 30_000.0, 20e6, 3000.0, 600.0
    origin_velocity = np.array([origin_speed, 0, 0])
    monkeypatch.setattr(measurements, "_earth_state", lambda time, offset, kernel: (
        origin_velocity * (time - gps_epoch + offset), origin_velocity,
    ))
    radio = RadioTrackingConfig(7.2e9, 10, 880 / 749, 0.005)
    trajectory = _trajectory(np.array([radius, 0, 0, speed, 0, 0]), gps_epoch, gps_epoch + elapsed)
    down_delay = (radius + speed * elapsed) / (c + speed + origin_speed)
    receive_elapsed = elapsed - down_delay - radio.transponder_delay_s
    up_delay = (radius + speed * receive_elapsed) / (c - origin_speed)
    expected_range = c * (down_delay + radio.transponder_delay_s + up_delay) / 2
    expected_doppler = -2 * radio.turnaround_ratio * radio.uplink_frequency_hz * c * speed / (
        (c + speed + origin_speed) * (c - origin_speed)
    )
    assert fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_range", radio).value == pytest.approx(expected_range, abs=1e-6)
    assert fixed_station.predict_radio(trajectory, gps_epoch + elapsed, "two_way_doppler", radio).value == pytest.approx(expected_doppler, abs=1e-6)


def test_earth_origin_against_erfa_epv00():
    """R2: heliocentric ERFA epv00, 20 km position and 0.03 m/s velocity.

    ERFA documents 11.2 km and 5 mm/s worst-case errors against DE405 for
    1900--2100. Subtracting the Sun aligns the DE430 and DE405 origins: their
    solar-system barycenters differ because the planetary ephemerides differ.
    """
    import erfa
    from astropy.time import Time
    from ssapy import get_body

    epoch, offset = 1.4e9, -10.001
    sun = get_body("sun")
    time = Time(epoch, offset, format="gps", scale="tai").tdb
    try:
        position, velocity = measurements._earth_state(epoch, offset, sun.position.kernel)
        sun_r, sun_v = sun.position.kernel[0, 10].compute_and_differentiate(time.jd1, time.jd2)
    finally:
        sun.position.close()
    reference, _ = erfa.epv00(time.jd1, time.jd2)
    assert np.linalg.norm(position - sun_r * 1000 - reference["p"] * 149597870700) < 20e3
    assert np.linalg.norm(velocity - sun_v * (1000 / 86400) - reference["v"] * (149597870700 / 86400)) < 0.03


def test_dense_signal_offsets_against_linear_motion(gps_epoch):
    """R1: between-grid state/STM including a 1 ns offset, 1 micrometre."""
    state0 = np.array([20e6, 5e6, 3e6, 3000, 2000, 500.0])
    trajectory = _trajectory(state0, gps_epoch, gps_epoch + 600)
    elapsed = 100.0 - 1e-9
    state, stm = trajectory.at(gps_epoch + 100, offset_s=-1e-9)
    np.testing.assert_allclose(state[:3], state0[:3] + state0[3:] * elapsed, atol=1e-6, rtol=0)
    expected = np.eye(6)
    expected[:3, 3:] = np.eye(3) * elapsed
    np.testing.assert_allclose(stm, expected, atol=1e-10, rtol=0)
