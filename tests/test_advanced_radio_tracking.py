"""Analytic coherent-link, plasma, clock and relativistic tracking references."""

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.time import Time

from ssatk.constants import c, EARTH_MU
from ssatk.navigation import GroundStation, GroundStationMeasurement, RadioTrackingConfig, solve_batch_orbit
from ssatk.io import read_tdm
from ssatk.navigation import measurements
from ssatk.propagators_orbit import propagate_orbit_state_with_stm


@pytest.fixture
def fixed_sites(monkeypatch):
    monkeypatch.setattr(measurements, "_earth_state", lambda time, offset, kernel: (np.zeros(3), np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (np.array([self.lon_deg * 1e6, self.lat_deg * 1e6, 0.]), np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: (np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.])))
    return GroundStation(0, 0, min_elevation_rad=-np.pi / 2)


def trajectory(state, t0, end):
    return propagate_orbit_state_with_stm(r0=state[:3], v0=state[3:], t0=t0, times=[t0, end],
                                         mu=0, dense_output=True, rtol=1e-12, atol=1e-9)


def transmit_event(state, receiver, transmitter, elapsed, delay=0):
    """40-digit quadratic events, retaining sub-picosecond count differences."""
    with localcontext() as context:
        context.prec = 40
        values = [Decimal(str(value)) for value in state]
        elapsed, delay, light_speed = Decimal(str(elapsed)), Decimal(str(delay)), Decimal(str(c))
        displacement = [values[i] + values[i+3]*elapsed - Decimal(str(receiver[i])) for i in range(3)]
        projection = sum(displacement[i]*values[i+3] for i in range(3))
        coefficient = light_speed**2 - sum(value**2 for value in values[3:])
        down = (-projection + (projection**2 + coefficient * sum(value**2 for value in displacement)).sqrt()) / coefficient
        spacecraft_receive = elapsed - down - delay
        distance_squared = sum((values[i] + values[i+3]*spacecraft_receive - Decimal(str(transmitter[i])))**2 for i in range(3))
        return spacecraft_receive - distance_squared.sqrt() / light_speed


def ramp_doppler(config, t0, first, last):
    """Independent 40-digit integrated piecewise quadratic carrier phase."""
    with localcontext() as context:
        context.prec = 40
        cycles = Decimal(0)
        for start, stop, frequency, rate in config.frequency_ramps:
            a, b = max(first, start - t0), min(last, stop - t0)
            if b > a:
                x, y = Decimal(str(a)) - Decimal(str(start - t0)), Decimal(str(b)) - Decimal(str(start - t0))
                cycles += Decimal(str(frequency)) * (y - x) + Decimal(str(rate)) * (y*y - x*x) / 2
        return float(Decimal(str(config.turnaround_ratio)) * cycles / Decimal(str(config.count_interval_s)) - Decimal(str(config.reference_frequency)))


@pytest.mark.parametrize("kind", ["three_way_range", "three_way_doppler", "differenced_range", "differenced_doppler"])
def test_links_against_quadratic_signal_events(fixed_sites, gps_epoch, kind):
    """R1/R5: quadratic light time; 1 um range, 5 uHz Doppler, 3e-5 SI Jacobian."""
    state = np.array([20e6, 5e6, 3e6, 3000., -1000., 100.])
    transmitter = GroundStation(-2, 1, min_elevation_rad=-np.pi / 2)
    receiver = GroundStation(1, -3, min_elevation_rad=-np.pi / 2)
    other = GroundStation(4, 2, min_elevation_rad=-np.pi / 2)
    radio = RadioTrackingConfig(7.2e9, 10., .8, .002, transmitter=transmitter)
    if kind.startswith("differenced"):
        radio = replace(radio, reference_receiver=other)
    elapsed = 600.
    prediction = receiver.predict_radio(trajectory(state, gps_epoch, gps_epoch + elapsed), gps_epoch + elapsed, kind, radio)

    def oracle(candidate):
        def link(site):
            position = np.array([site.lon_deg * 1e6, site.lat_deg * 1e6, 0.])
            emit = transmit_event(candidate, position, np.array([-2e6, 1e6, 0.]), elapsed, radio.transponder_delay_s)
            if kind.endswith("range"):
                return Decimal(str(c)) * (Decimal(str(elapsed)) - emit) / 2
            start = transmit_event(candidate, position, np.array([-2e6, 1e6, 0.]), elapsed - 10., radio.transponder_delay_s)
            return Decimal(str(radio.turnaround_ratio)) * Decimal(str(radio.uplink_frequency_hz)) * ((emit - start) - 10) / 10
        return float(link(receiver) - link(other) if kind.startswith("differenced") else link(receiver))

    assert prediction.value == pytest.approx(oracle(state), rel=0, abs=1e-6 if kind.endswith("range") else 5e-6)
    derivative = np.empty(6)
    for column, step in enumerate([10., 10., 10., .01, .01, .01]):
        delta = np.zeros(6)
        delta[column] = step
        derivative[column] = (oracle(state + delta) - oracle(state - delta)) / (2 * step)
    np.testing.assert_allclose(prediction.jacobian[0], derivative, rtol=0, atol=3e-5)


def test_ramps_crossing_frequency_step_against_decimal_phase(fixed_sites, gps_epoch):
    """R1: 40-digit piecewise phase over a ramp boundary; 5 uHz Doppler."""
    state = np.array([20e6, 0, 0, 3000., 0, 0])
    radio = RadioTrackingConfig(7.2e9, 10., .8, frequency_ramps=[
        [gps_epoch, gps_epoch + 595, 7.2e9, 40.],
        [gps_epoch + 595, gps_epoch + 1000, 7.2e9 + 10e3, -20.],
    ])
    elapsed = 600.
    end = transmit_event(state, np.zeros(3), np.zeros(3), elapsed)
    start = transmit_event(state, np.zeros(3), np.zeros(3), elapsed - 10.)
    expected = ramp_doppler(radio, gps_epoch, start, end)
    prediction = fixed_sites.predict_radio(trajectory(state, gps_epoch, gps_epoch + elapsed), gps_epoch + elapsed, "two_way_doppler", radio)
    assert prediction.value == pytest.approx(expected, rel=0, abs=5e-6)
    assert radio.uplink_phase_excess(gps_epoch, 590, 600) == pytest.approx(168250., rel=0, abs=1e-8)


def test_clock_equipment_and_proper_delays_against_affine_clocks(fixed_sites, gps_epoch):
    """R1: affine station clocks and constant proper rate; 1 um and 5 uHz."""
    receiver = replace(fixed_sites, clock_offset_s=.003, clock_drift=2e-6, clock_epoch=gps_epoch, receive_delay_s=.002)
    transmitter = GroundStation(1, 0, min_elevation_rad=-np.pi/2, clock_offset_s=-.002,
                                clock_drift=-1e-6, clock_epoch=gps_epoch, transmit_delay_s=.001)
    radio = RadioTrackingConfig(7.2e9, 10., .8, .005, transmitter=transmitter,
                                transponder_delay_is_proper=True, transponder_clock=lambda time, state: .999)
    state = np.array([20e6, 0, 0, 3000., 0, 0])
    elapsed = 600.
    def emit(tag):
        receive = (Decimal(str(tag)) - Decimal('.003')) / Decimal('1.000002')
        antenna = receive - Decimal('.002')
        event = transmit_event(state, np.zeros(3), np.array([1e6, 0, 0]), antenna, Decimal('.005') / Decimal('.999'))
        event -= Decimal('.001')
        return event - Decimal('.002') - Decimal('.000001') * event
    expected_range = float(Decimal(str(c)) * (Decimal(str(elapsed)) - emit(elapsed)) / 2)
    expected_doppler = float(Decimal('.8') * Decimal('7.2e9') * ((emit(elapsed) - emit(elapsed-10)) - 10) / 10)
    traj = trajectory(state, gps_epoch, gps_epoch + elapsed)
    assert receiver.predict_radio(traj, gps_epoch + elapsed, "three_way_range", radio).value == pytest.approx(expected_range, rel=0, abs=1e-6)
    assert receiver.predict_radio(traj, gps_epoch + elapsed, "three_way_doppler", radio).value == pytest.approx(expected_doppler, rel=0, abs=5e-6)


def test_media_against_saastamoinen_and_plasma_dispersion(fixed_sites):
    """R1/R3: Saastamoinen/plasma paths, 1 pm; clock-scaled static range, 1 um."""
    station = replace(fixed_sites, meteorology=(1000., 290., 10.), slant_tec=1e17)
    zenith = .0022768 * 1000 / (1 - .00266) + .002277 * (1255 / 290 + .05) * 10
    plasma = 40.3e17 / (7.2e9)**2
    assert station.atmospheric_delay(0., np.pi/6, 7.2e9) == pytest.approx(2*zenith + plasma, rel=0, abs=1e-12)
    assert station.atmospheric_delay(0., np.pi/6, 7.2e9, phase=True) == pytest.approx(2*zenith - plasma, rel=0, abs=1e-12)

    t0, distance, drift, frequency, tec = 1.4e9, 20e6, 1e-4, 1e9, 1e18
    station = replace(fixed_sites, clock_drift=drift, clock_epoch=t0, slant_tec=tec)
    state = np.array([distance, 0., 0., 0., 0., 0.])
    radio = RadioTrackingConfig(frequency, 10., 1.)
    prediction = station.predict_radio(trajectory(state, t0, t0+100), t0+60, "two_way_range", radio)
    expected = distance*(1+drift) + 40.3*tec/frequency**2/(1+drift)
    assert prediction.value == pytest.approx(expected, rel=0, abs=1e-6)


def test_shapiro_and_geocentric_proper_clock_against_static_reference(fixed_sites, gps_epoch):
    """R1/R3: radial GR logarithm and IAU L_G proper rate; 1 um range."""
    station = GroundStation(7, 0, min_elevation_rad=-np.pi/2)
    state = np.array([20e6, 0, 0, 0., 0, 0])
    radio = RadioTrackingConfig(7.2e9, 10., 1., .001, shapiro_bodies=("earth",), transponder_delay_is_proper=True)
    rate = 1 + 6.969290134e-10 - EARTH_MU / (20e6 * c**2)
    expected = 13e6 + 2 * EARTH_MU / c**2 * np.log(20 / 7) + c * .001 / (2 * rate)
    prediction = station.predict_radio(trajectory(state, gps_epoch, gps_epoch + 100), gps_epoch + 100, "two_way_range", radio)
    assert prediction.value == pytest.approx(expected, rel=0, abs=1e-6)


def test_solar_shapiro_against_static_logarithm(fixed_sites, gps_epoch, monkeypatch):
    """R1: prescribed solar mass/location, static radial GR path; 1 um range."""
    class Kernel:
        def __getitem__(self, key):
            assert key == (0, 10)
            return SimpleNamespace(compute=lambda jd1, jd2: np.array([-1e8, 0., 0.]))
    sun = SimpleNamespace(mu=1.3e20, position=SimpleNamespace(kernel=Kernel(), close=lambda: None))
    monkeypatch.setattr(measurements, "get_body", lambda name: sun)
    station = GroundStation(7, 0, min_elevation_rad=-np.pi/2)
    state = np.array([20e6, 0, 0, 0., 0, 0])
    radio = RadioTrackingConfig(7.2e9, 10., shapiro_bodies=("sun",))
    expected = 13e6 + 2 * sun.mu / c**2 * np.log((1e11+20e6)/(1e11+7e6))
    prediction = station.predict_radio(trajectory(state, gps_epoch, gps_epoch+100), gps_epoch+100, "two_way_range", radio)
    assert prediction.value == pytest.approx(expected, rel=0, abs=1e-6)


def test_media_in_signal_events_and_carrier_phase(fixed_sites, gps_epoch):
    """R1: linearly changing slant path with opposite carrier sign; 1 um/1 uHz."""
    state = np.array([20e6, 0, 0, 0., 0, 0])
    station = replace(fixed_sites, media_delay=lambda time, elevation, frequency, phase: (-1 if phase else 1) * (2 + .001 * (time - gps_epoch)))
    radio = RadioTrackingConfig(7.2e9, 10.)
    elapsed = 100.
    # Static craft: downlink excess uses receive time, uplink uses solved transmit time.
    down = (20e6 + 2 + .001 * elapsed) / c
    transmit = (elapsed - down - (20e6 + 2) / c) / (1 + .001 / c)
    expected_range = c * (elapsed - transmit) / 2
    expected_doppler = radio.uplink_frequency_hz * .002 / (c - .001)
    traj = trajectory(state, gps_epoch, gps_epoch + elapsed)
    assert station.predict_radio(traj, gps_epoch + elapsed, "two_way_range", radio).value == pytest.approx(expected_range, rel=0, abs=1e-6)
    assert station.predict_radio(traj, gps_epoch + elapsed, "two_way_doppler", radio).value == pytest.approx(expected_doppler, rel=0, abs=1e-6)


def test_ekf_adapter_covers_corrected_receive_epoch(fixed_sites, gps_epoch):
    """R1: negative receive clock offset, static orbit; 1 um range."""
    station = replace(fixed_sites, clock_offset_s=-.2)
    state = np.array([20e6, 0, 0, 0., 0, 0])
    model = GroundStationMeasurement(station, gps_epoch + 100, "two_way_range", RadioTrackingConfig(7.2e9, 10),
                                      t0=gps_epoch, propagation_kwargs={"mu": 0})
    value, _ = model(state)
    assert value[0] == pytest.approx(20e6, rel=0, abs=1e-6)


@pytest.mark.parametrize("ramped", [False, True])
def test_three_way_tdm_joint_fit_against_quadratic_motion(fixed_sites, ramped):
    """R1/R5: independent signal events, bias and Fisher matrix; 1 mm/1 um/s.

    Covariance differences normalized by reference sigmas are <= 1e-3
    (dimensionless); derivatives use independent high-precision phase/events.
    """
    t0 = float(Time("2024-01-01T00:00:00").gps)
    state = np.array([20e6, 5e6, 3e6, 3000., -1000., 100.])
    stations = {"TX": GroundStation(-2, 1, min_elevation_rad=-np.pi/2),
                "RX0": GroundStation(1, -3, min_elevation_rad=-np.pi/2),
                "RX1": GroundStation(4, 2, min_elevation_rad=-np.pi/2),
                "RX2": GroundStation(-3, -2, min_elevation_rad=-np.pi/2)}
    config = RadioTrackingConfig(7.2e9, 10., .8, frequency_ramps=[
        [t0, t0+295, 7.2e9, 40.], [t0+295, t0+900, 7.2e9+10e3, -20.],
    ] if ramped else None)
    noise = np.tile([1., .01], 9)
    def oracle(candidate):
        result = []
        for name in ("RX0", "RX1", "RX2"):
            site = stations[name]
            position = np.array([site.lon_deg*1e6, site.lat_deg*1e6, 0.])
            for elapsed in (100., 300., 600.):
                end = transmit_event(candidate[:6], position, np.array([-2e6, 1e6, 0.]), elapsed)
                start = transmit_event(candidate[:6], position, np.array([-2e6, 1e6, 0.]), elapsed-10.)
                distance = float(Decimal(str(c)) * (Decimal(str(elapsed))-end) / 2) + candidate[6]
                doppler = (ramp_doppler(config, t0, start, end) if ramped else
                           float(Decimal('.8') * Decimal('7.2e9') * (end-start-10) / 10))
                result.extend((distance, doppler))
        return np.asarray(result)
    truth = np.r_[state, 3.]
    expected = oracle(truth)
    lines = ["CCSDS_TDM_VERS = 2.0", "CREATION_DATE = 2024-01-01T00:00:00", "ORIGINATOR = QUADRATIC_REFERENCE"]
    row = 0
    for name in ("RX0", "RX1", "RX2"):
        lines.extend(["META_START", "TIME_SYSTEM = UTC", "PARTICIPANT_1 = TX", "PARTICIPANT_2 = SPACECRAFT",
                      f"PARTICIPANT_3 = {name}", "MODE = SEQUENTIAL", "PATH = 1,2,3", "RANGE_UNITS = km",
                      "INTEGRATION_INTERVAL = 10", "INTEGRATION_REF = END",
                      f"FREQ_OFFSET = {config.reference_frequency}", "META_STOP", "DATA_START"])
        for elapsed in (100., 300., 600.):
            epoch = Time(t0+elapsed, format="gps")
            epoch.precision = 9
            lines.append(f"RANGE = {epoch.utc.isot} {expected[row]/500:.17g}")
            lines.append(f"RECEIVE_FREQ_1 = {epoch.utc.isot} {expected[row+1]:.17g}")
            row += 2
        lines.append("DATA_STOP")
    pairs = read_tdm('\n'.join(lines)).to_station_observations(stations, radio={"TX": config},
        covariance={"three_way_range": 1., "three_way_doppler": .0001}, range_convention="round_trip")
    fit = solve_batch_orbit(pairs, state+[100., -200., 300., 1., -2., 3.], t0=t0,
        parameters0={"range_bias_m": 0.}, parameter_scale=[10.], parameter_step=[.1],
        measurement_bias=lambda station, observation, p: p["range_bias_m"] if observation.measurement.endswith("range") else 0.,
        propagation_kwargs={"mu": 0, "rtol": 1e-12, "atol": 1e-9}, max_nfev=30)
    assert fit.success and fit.numerical_rank == 7
    np.testing.assert_allclose(fit.state0[:3], state[:3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(fit.state0[3:], state[3:], rtol=0, atol=1e-6)
    assert fit.parameters["range_bias_m"] == pytest.approx(3., rel=0, abs=1e-3)
    jacobian = np.empty((18, 7))
    for column, step in enumerate([10., 10., 10., .01, .01, .01, .1]):
        delta = np.zeros(7)
        delta[column] = step
        jacobian[:, column] = (oracle(truth+delta)-oracle(truth-delta)) / (2*step) / noise
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    sigma = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose((fit.joint_covariance-covariance)/np.outer(sigma, sigma), 0., rtol=0, atol=1e-3)
