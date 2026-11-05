"""CCSDS 503.0-B-2 conventions and independent vacuum navigation references."""

from io import StringIO
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from astropy.time import Time

from ssatk.constants import c
from ssatk.io import read_tdm
from ssatk.navigation import GroundStation, RadioTrackingConfig, solve_batch_orbit
from ssatk.navigation import measurements


def _message(metadata, records, xml=False):
    header = {"CREATION_DATE": "2024-01-01T00:00:00", "ORIGINATOR": "ANALYTIC_REFERENCE"}
    if xml:
        root = ET.Element("tdm", {"id": "CCSDS_TDM_VERS", "version": "2.0", "xmlns": "urn:ccsds:schema:ndmxml"})
        head = ET.SubElement(root, "header")
        for key, value in header.items():
            ET.SubElement(head, key).text = value
        body = ET.SubElement(root, "body")
        for meta, data in zip(metadata, records):
            segment = ET.SubElement(body, "segment")
            section = ET.SubElement(segment, "metadata")
            for key, value in meta.items():
                ET.SubElement(section, key).text = str(value)
            section = ET.SubElement(segment, "data")
            for key, epoch, value in data:
                item = ET.SubElement(section, "observation")
                ET.SubElement(item, "EPOCH").text = str(epoch)
                ET.SubElement(item, key).text = str(value)
        return ET.tostring(root, encoding="unicode")
    lines = ["CCSDS_TDM_VERS = 2.0", *(f"{key} = {value}" for key, value in header.items())]
    for meta, data in zip(metadata, records):
        lines.extend(["META_START", *(f"{key} = {value}" for key, value in meta.items()), "META_STOP", "DATA_START"])
        lines.extend(f"{key} = {epoch} {value}" for key, epoch, value in data)
        lines.append("DATA_STOP")
    return "\n".join(lines)


def _metadata(**changes):
    return {
        "TIME_SYSTEM": "UTC", "PARTICIPANT_1": "STATION", "PARTICIPANT_2": "SPACECRAFT",
        "MODE": "SEQUENTIAL", "PATH": "1,2,1", "RANGE_UNITS": "km",
        "INTEGRATION_INTERVAL": "10", "INTEGRATION_REF": "END", **changes,
    }


def _convert(message, **kwargs):
    return message.to_station_observations(
        {"STATION": GroundStation(0, 0)}, radio={"STATION": RadioTrackingConfig(7.2e9, 1, 880 / 749)},
        covariance={"two_way_range": 4, "two_way_doppler": 9}, **kwargs,
    )


@pytest.mark.parametrize("xml", [False, True])
@pytest.mark.parametrize("convention", ["round_trip", "half_round_trip"])
@pytest.mark.parametrize("units", ["km", "s"])
def test_range_and_rate_against_ccsds_scaling(xml, convention, units):
    """R3: CCSDS §§3.5.2.3/7 full/half path; 1 nm range, 1e-10 Hz shift."""
    # An ICD may use either full or half path; neither is a TDM default.
    path_factor = 2 if convention == "round_trip" else 1
    raw_range = 20000 * path_factor / (1000 if units == "km" else c)
    raw_rate = 30 * path_factor / 1000 + 0.002
    meta = _metadata(RANGE_UNITS=units, TURNAROUND_NUMERATOR="880", TURNAROUND_DENOMINATOR="749",
                     CORRECTION_DOPPLER="-0.002", CORRECTIONS_APPLIED="NO")
    records = [("RANGE", "2024-01-01T00:00:00", raw_range), ("DOPPLER_INTEGRATED", "2024-01-01T00:00:00", raw_rate)]
    message = read_tdm(_message([meta], [records], xml))
    pairs = _convert(message, range_convention=convention, doppler_convention=convention)
    assert pairs[0][1].value == pytest.approx(20000, abs=1e-9)
    expected_shift = -2 * (7.2e9 * 880 / 749) * 30 / c
    assert pairs[1][1].value == pytest.approx(expected_shift, abs=1e-10)
    assert pairs[1][1].radio.count_interval_s == 10
    np.testing.assert_allclose(pairs[0][1].covariance, [[4]], rtol=0, atol=0)
    assert pairs[0][1].elevation_rad is None and pairs[0][1].visible is None


@pytest.mark.parametrize("xml", [False, True])
@pytest.mark.parametrize("reference,offset", [("START", 10), ("MIDDLE", 5), ("END", 0)])
def test_received_frequency_count_tags_and_corrections(xml, reference, offset):
    """R3: CCSDS §3.5.2.8 and table 3-3; 1 microsecond epoch, 2 microhertz."""
    carrier = 7.2e9 * 880 / 749
    meta = _metadata(INTEGRATION_REF=reference, FREQ_OFFSET=str(carrier),
                     CORRECTION_RECEIVE="3", CORRECTIONS_APPLIED="NO")
    records = [("TRANSMIT_FREQ_1", "2017-001T00:00:00", "7200000000 [Hz]"),
               ("TRANSMIT_FREQ_RATE_1", "2017-001T00:00:00", "0 [Hz/s]"),
               ("RECEIVE_FREQ_1", "2017-001T00:00:00", "-23")]
    message = read_tdm(_message([meta], [records], xml))
    observation = _convert(message)[0][1]
    assert observation.time == pytest.approx(1167264018 + offset, abs=1e-6)
    assert observation.value == pytest.approx(-20, abs=2e-6)
    # Reconstruct the corrected carrier according to CCSDS, then reverse the
    # standard's positive-recession shift to SSATK's received-minus-reference.
    applied_meta = dict(meta, CORRECTIONS_APPLIED="YES")
    applied = read_tdm(_message([applied_meta], [[("RECEIVE_FREQ", "2017-001T00:00:00", "-20")]], xml))
    assert _convert(applied)[0][1].value == pytest.approx(observation.value, abs=2e-6)
    assert float(message.segments[0].observations[0].epoch.gps) == pytest.approx(1167264018, abs=1e-6)


@pytest.mark.parametrize("system,epoch", [("UTC", "2017-01-01T00:00:00"), ("GPS", "2017-001T00:00:18"), ("TAI", "2017-001T00:00:37")])
def test_times_and_range_corrections_against_leap_second_reference(system, epoch):
    """R3: GPS=TAI-19 s, UTC offset 37 s on 2017-01-01; 1 us, 1 nm."""
    meta = _metadata(TIME_SYSTEM=system, CORRECTION_RANGE="-0.006", CORRECTIONS_APPLIED="NO", DATA_QUALITY="DEGRADED")
    message = read_tdm(StringIO(_message([meta], [[("RANGE", epoch, "40.006")]])))
    observation = _convert(message, range_convention="round_trip", bias={"two_way_range": 2})[0][1]
    assert observation.time == pytest.approx(1167264018, abs=1e-6)
    assert observation.value == pytest.approx(20000, abs=1e-9)
    assert observation.bias == pytest.approx(2, abs=0)
    assert observation.valid is False
    assert message.segments[0].metadata["TIME_SYSTEM"] == system


def _range_reference(state, site, elapsed, delay):
    relative = state[:3] + state[3:] * elapsed - site
    projection = relative @ state[3:]
    light_time = (relative @ relative) / (
        np.sqrt(projection**2 + (c**2 - state[3:] @ state[3:]) * (relative @ relative)) + projection
    )
    reception = state[:3] + state[3:] * (elapsed - light_time - delay) - site
    return (c * (light_time + delay) + np.sqrt(reception @ reception)) / 2


@pytest.mark.parametrize("xml", [False, True])
def test_file_to_orbit_and_covariance_against_analytic_vacuum_links(monkeypatch, tmp_path, xml):
    """R1/R5: exact free flight and complex-step Fisher covariance.

    Budgets: 1 mm position, 1 micrometre/s velocity, 1e-6 relative covariance;
    residuals 1 micrometre range and 1 microhertz Doppler.
    """
    t0 = 1.4e9
    sites = np.array([[0, 0, 0], [0, 8e6, 0], [0, 0, 8e6]])
    monkeypatch.setattr(measurements, "_earth_state", lambda *args: (np.zeros(3), np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (sites[int(self.lon_deg)], np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda *args: tuple(np.eye(3)))
    truth = np.array([20e6, 5e6, 3e6, -1000, 2000, 500])
    config = RadioTrackingConfig(7.2e9, 10, 880 / 749, 2e-6)
    carrier = config.uplink_frequency_hz * config.turnaround_ratio

    def expected(state, site, elapsed, kind):
        end = _range_reference(state, site, elapsed, config.transponder_delay_s)
        if kind == "two_way_range":
            return end
        start = _range_reference(state, site, elapsed - config.count_interval_s, config.transponder_delay_s)
        return -2 * carrier * (end - start) / (c * config.count_interval_s)

    metadata, records, oracle = [], [], []
    stations = {f"STATION_{index}": GroundStation(index, 0, min_elevation_rad=-np.pi / 2) for index in range(3)}
    for index, site in enumerate(sites):
        metadata.append(_metadata(PARTICIPANT_1=f"STATION_{index}", INTEGRATION_REF="MIDDLE", FREQ_OFFSET=str(carrier)))
        data = []
        for elapsed in (100, 300, 600):
            for key, kind in (("RANGE", "two_way_range"), ("RECEIVE_FREQ_1", "two_way_doppler")):
                value = expected(truth, site, elapsed, kind)
                epoch = Time(t0 + elapsed - (5 if kind == "two_way_doppler" else 0), format="gps")
                epoch.precision = 9
                data.append((key, epoch.utc.isot, value / 500 if kind == "two_way_range" else value))
                oracle.append((site, elapsed, kind, value))
        records.append(data)
    path = tmp_path / ("analytic.xml" if xml else "analytic.tdm")
    path.write_text(_message(metadata, records, xml), encoding="utf-8")
    message = read_tdm(path)
    pairs = message.to_station_observations(
        stations, radio={name: config for name in stations},
        covariance={"two_way_range": 1, "two_way_doppler": 1}, range_convention="round_trip",
    )
    for (_, observation), (_, _, kind, value) in zip(pairs, oracle):
        assert observation.value == pytest.approx(value, abs=1e-6)
        assert observation.measurement == kind
    fit = solve_batch_orbit(pairs, truth + [100, -200, 300, 1, -2, 3], t0=t0,
                            propagation_kwargs={"mu": 0, "rtol": 1e-12, "atol": 1e-9}, max_nfev=20)
    assert fit.success and fit.numerical_rank == 6
    np.testing.assert_allclose(fit.state0[:3], truth[:3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(fit.state0[3:], truth[3:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit.residuals, 0, rtol=0, atol=1e-6)
    jacobian = np.empty((len(oracle), 6))
    for column in range(6):
        perturbed = truth.astype(complex)
        perturbed[column] += 1e-6j
        jacobian[:, column] = [expected(perturbed, site, elapsed, kind).imag / 1e-6 for site, elapsed, kind, _ in oracle]
    fisher_covariance = np.linalg.inv(jacobian.T @ jacobian)
    np.testing.assert_allclose(fit.covariance, fisher_covariance, rtol=1e-6, atol=0)


@pytest.mark.parametrize("changes,keyword,value", [
    ({"PATH": "1,2,3", "PARTICIPANT_3": "OTHER_STATION"}, "RANGE", "40"),
    ({"RANGE_MODULUS": "1"}, "RANGE", "0.1"),
    ({"RANGE_UNITS": "RU"}, "RANGE", "40"),
    ({"TIMETAG_REF": "TRANSMIT"}, "RANGE", "40"),
    ({}, "TRANSMIT_FREQ_RATE_1", "1"),
    ({}, "TROPO_DRY", "2"),
    ({"TRANSMIT_DELAY_1": "0.000077"}, "RANGE", "40"),
])
def test_incompatible_link_physics_is_rejected(changes, keyword, value):
    """R3: reject ambiguity, media, ramps and link events outside the vacuum model."""
    message = read_tdm(_message([_metadata(**changes)], [[(keyword, "2024-01-01T00:00:00", value)]]))
    with pytest.raises(ValueError):
        _convert(message, range_convention="round_trip")
