"""CCSDS TDM 1.0/2.0 KVN and XML ingestion for coherent radio navigation.

The parsed message retains metadata, comments, original epochs and values.
Conversion supports coherent two-/three-way and differenced radar links,
bounded ramps, cumulative Doppler counts and explicit media/calibration data.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

import numpy as np
from astropy import units as u
from astropy.time import Time

from ssatk.constants import c
from ssatk.io.ccsds_cdm import _split_source, _time
from ssatk.navigation import GroundStation, RadioTrackingConfig, StationObservation

__all__ = ["TDMObservation", "TDMSegment", "TrackingDataMessage", "read_tdm", "read_ramp_table"]

_CONVENTIONS = {"round_trip": 0.5, "half_round_trip": 1.0}
_DOPPLER = {"DOPPLER_INTEGRATED", "DOPPLER_COUNT", "RECEIVE_FREQ", "RECEIVE_FREQ_1"}
_MEASUREMENTS = _DOPPLER | {"RANGE"}


def read_ramp_table(source: str | Path | TextIO, *, station_id: str, spacecraft_id: str,
                    stop_time, time_system: str = "TAI", epoch_format: str = "mjd") -> np.ndarray:
    """Read GMAT seven-column ramps as [start GPS, stop GPS, Hz, Hz/s].

    Rows are epoch, station ID, spacecraft ID, uplink band, ramp type,
    frequency and rate. Select one explicit link; type 1..5 ramps are valid.
    Epoch format is conventional MJD (JD-2400000.5) or ``gmat_mjd``
    (JD-2430000.0). Specify the source time scale; A1 is not silently TAI.
    Final coverage ends at explicit ``stop_time`` (Astropy Time or GPS seconds).
    """
    if not isinstance(station_id, str) or not station_id or not isinstance(spacecraft_id, str) or not spacecraft_id:
        raise ValueError("station_id and spacecraft_id must be nonempty strings.")
    if time_system not in {"TAI", "UTC", "TT", "TDB", "GPS"} or epoch_format not in {"mjd", "gmat_mjd"}:
        raise ValueError("ramp time_system and epoch_format must identify the actual source convention.")
    text = source if isinstance(source, str) and ("\n" in source or len(source.split()) == 7) else _split_source(source)
    selected = []
    band = None
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"ramp row {number} requires seven columns.")
        if fields[1:3] != [station_id, spacecraft_id]:
            continue
        if not re.fullmatch(r"[1-3]", fields[3]) or not re.fullmatch(r"[1-5]", fields[4]):
            raise ValueError(f"ramp row {number} has an unsupported band or invalid ramp type.")
        if band is not None and band != fields[3]:
            raise ValueError("one ramp table cannot change uplink band.")
        band = fields[3]
        epoch = _number(fields[0], "ramp epoch")
        jd_offset = 2400000.5 if epoch_format == "mjd" else 2430000.0
        if time_system == "GPS":
            time = Time(jd_offset, epoch, format="jd", scale="tai") + 19 * u.s
        else:
            time = Time(jd_offset, epoch, format="jd", scale=time_system.lower())
        selected.append((float(time.gps), _number(fields[5], "ramp frequency", "Hz"), _number(fields[6], "ramp rate", "Hz/s")))
    if not selected:
        raise ValueError("ramp table contains no records for the selected link.")
    stop = float(stop_time.gps) if isinstance(stop_time, Time) else float(stop_time)
    rows = [[epoch, selected[index + 1][0] if index + 1 < len(selected) else stop, frequency, rate]
            for index, (epoch, frequency, rate) in enumerate(selected)]
    return RadioTrackingConfig(selected[0][1], 1., frequency_ramps=rows).frequency_ramps


@dataclass(frozen=True)
class TDMObservation:
    """A tracking record with its original epoch and lexical value/unit."""

    keyword: str
    epoch: Time
    value: str


@dataclass(frozen=True)
class TDMSegment:
    """Tracking records sharing immutable metadata and ordered comments."""

    metadata: Mapping[str, str]
    observations: tuple[TDMObservation, ...]
    metadata_comments: tuple[str, ...] = ()
    data_comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackingDataMessage:
    """Parsed tracking message; no station locations or noise are inferred."""

    header: Mapping[str, str]
    segments: tuple[TDMSegment, ...]
    comments: tuple[str, ...] = ()

    def to_station_observations(
        self,
        stations: Mapping[str, GroundStation],
        *,
        radio: Mapping[str, RadioTrackingConfig],
        covariance: Mapping[str, float],
        range_convention: str | None = None,
        doppler_convention: str | None = None,
        doppler_count_convention: str | None = None,
        bias: Mapping[str, float] | None = None,
    ) -> list[tuple[GroundStation, StationObservation]]:
        """Convert records for ``solve_batch_orbit`` or EKF updates.

        ``stations`` and ``radio`` are keyed by the ground participant's ID.
        ``covariance`` and optional ``bias`` use the internal measurement names
        ``two_way_range`` (m², m) and ``two_way_doppler`` (Hz², Hz).
        Range and integrated range-rate conventions must explicitly be
        ``round_trip`` or ``half_round_trip``, as specified by the source ICD.
        Frequency records need no path scaling. The file supplies the Doppler
        count interval; explicit radio settings supply missing hardware data.

        Scalar corrections marked NO are added before conversion. START and
        MIDDLE count tags move to the receive-interval end. Degraded records
        remain invalid; elevation and visibility remain unknown (None).
        SINGLE_DIFF data (PATH_2 minus PATH_1) becomes primary-minus-reference.
        Ramps and equipment/media inputs require explicit coverage/calibration.
        Counter beat signs come from ``doppler_count_convention`` and the ICD.
        """
        output = []
        target = None
        bias = {} if bias is None else bias
        supplied_stations = stations
        stations, media_targets = _tdm_media(self.segments, stations)
        for segment in self.segments:
            meta = segment.metadata
            if {item.keyword for item in segment.observations} <= {"STEC", "TROPO_DRY", "TROPO_WET"}:
                continue
            if meta.get("MODE") not in {"SEQUENTIAL", "SINGLE_DIFF"} or meta.get("TIMETAG_REF", "RECEIVE") != "RECEIVE":
                raise ValueError("TDM navigation requires SEQUENTIAL/SINGLE_DIFF mode and RECEIVE time tags.")
            differenced = meta["MODE"] == "SINGLE_DIFF"
            if differenced and "RECEIVE_BAND" not in meta:
                raise ValueError("SINGLE_DIFF data requires RECEIVE_BAND identifying the differenced carriers.")
            if (differenced and "PATH" in meta) or (not differenced and ("PATH_1" in meta or "PATH_2" in meta)):
                raise ValueError("TDM PATH fields must agree with MODE.")
            paths = [[part.strip() for part in meta.get(key, "").split(",")]
                     for key in (("PATH_1", "PATH_2") if differenced else ("PATH",))]
            links = [_tdm_link(segment, path, stations, radio, supplied_stations) for path in paths]
            spacecraft_id, station, config = links[0]
            if target is not None and spacecraft_id != target:
                raise ValueError("TDM navigation must describe one spacecraft distinct from its stations.")
            target = spacecraft_id
            if media_targets and media_targets != {target}:
                raise ValueError("STEC segments must refer to the same spacecraft as the tracking paths.")
            if differenced:
                if links[1][0] != spacecraft_id:
                    raise ValueError("differenced TDM paths must track the same spacecraft.")
                config = replace(config, reference_receiver=links[1][1], reference_radio=links[1][2])
            prefix = "differenced" if differenced else ("two_way" if paths[0][0] == paths[0][2] else "three_way")
            kinds = {item.keyword for item in segment.observations}
            auxiliary = {f"{name}_{path[0]}" for path in paths for name in ("TRANSMIT_FREQ", "TRANSMIT_FREQ_RATE")}
            unsupported = kinds - _MEASUREMENTS - auxiliary
            if unsupported:
                raise ValueError(f"unsupported TDM navigation observables: {sorted(unsupported)}")
            has_doppler = bool(kinds & (_DOPPLER - {"DOPPLER_COUNT"}))
            offset = 0.0
            if has_doppler:
                interval = _number(meta.get("INTEGRATION_INTERVAL", ""), "INTEGRATION_INTERVAL", "s")
                reference = meta.get("INTEGRATION_REF")
                if interval <= 0 or reference not in {"START", "MIDDLE", "END"}:
                    raise ValueError("TDM Doppler requires positive INTEGRATION_INTERVAL and START/MIDDLE/END INTEGRATION_REF.")
                config = replace(config, count_interval_s=interval)
                if config.reference_radio is not None:
                    config = replace(config, reference_radio=replace(config.reference_radio, count_interval_s=interval))
                offset = {"START": interval, "MIDDLE": interval / 2, "END": 0.0}[reference]
            quality = meta.get("DATA_QUALITY", "RAW")
            if quality not in {"RAW", "VALIDATED", "DEGRADED"}:
                raise ValueError("unsupported TDM DATA_QUALITY.")
            corrections = {key: value for key, value in meta.items() if key.startswith("CORRECTION_")}
            applied = meta.get("CORRECTIONS_APPLIED")
            if (corrections or applied is not None) and applied not in {"YES", "NO"}:
                raise ValueError("TDM corrections require CORRECTIONS_APPLIED=YES or NO.")
            range_units = meta.get("RANGE_UNITS", "km")
            counts = _tdm_counts(segment, doppler_count_convention) if "DOPPLER_COUNT" in kinds else {}
            for key, value in corrections.items():
                units = {"CORRECTION_RANGE": range_units, "CORRECTION_DOPPLER": "km/s", "CORRECTION_RECEIVE": "Hz"}.get(key)
                correction_value = _number(value, key, units)
                if units is None and correction_value != 0:
                    raise ValueError(f"unsupported nonzero TDM correction {key}.")
            for item in segment.observations:
                key = item.keyword
                if key in auxiliary:
                    continue
                observation_config = config
                if key == "RANGE":
                    if range_units not in {"km", "s"} or meta.get("RANGE_MODE", "CONSTANT") not in {"CONSTANT", "COHERENT"}:
                        raise ValueError("TDM navigation supports unambiguous range in km or s, without RU conversion.")
                    if _number(meta.get("RANGE_MODULUS", "0"), "RANGE_MODULUS", range_units) != 0:
                        raise ValueError("ambiguous TDM range is unsupported.")
                    scale = _path_scale(range_convention, "range_convention") * (1000.0 if range_units == "km" else c)
                    units, measurement, correction = range_units, f"{prefix}_range", "CORRECTION_RANGE"
                elif key == "DOPPLER_INTEGRATED":
                    if config.frequency_ramps is not None or (config.reference_radio is not None and
                        (config.reference_radio.frequency_ramps is not None or
                         config.reference_radio.uplink_frequency_hz * config.reference_radio.turnaround_ratio != config.uplink_frequency_hz * config.turnaround_ratio)):
                        raise ValueError("integrated range rate requires a single constant carrier; use received frequency or Doppler counts for ramps.")
                    if config.reference_frequency != config.uplink_frequency_hz * config.turnaround_ratio or (config.reference_radio is not None and
                        config.reference_radio.reference_frequency != config.reference_radio.uplink_frequency_hz * config.reference_radio.turnaround_ratio):
                        raise ValueError("integrated range rate requires the coherent carrier reference; use received frequency for custom references.")
                    scale = -2 * _path_scale(doppler_convention, "doppler_convention") * config.uplink_frequency_hz * config.turnaround_ratio * 1000 / c
                    units, measurement, correction = "km/s", f"{prefix}_doppler", "CORRECTION_DOPPLER"
                elif key == "DOPPLER_COUNT":
                    epoch_key = float(item.epoch.gps)
                    if epoch_key not in counts:
                        continue
                    count_value, duration = counts[epoch_key]
                    observation_config = replace(config, count_interval_s=duration)
                    if config.reference_radio is not None:
                        observation_config = replace(observation_config, reference_radio=replace(config.reference_radio, count_interval_s=duration))
                    scale = 1.0
                    units, measurement, correction = "Hz", f"{prefix}_doppler", "CORRECTION_RECEIVE"
                else:
                    scale = 1.0
                    units, measurement, correction = "Hz", f"{prefix}_doppler", "CORRECTION_RECEIVE"
                value = count_value if key == "DOPPLER_COUNT" else _number(item.value, key, units)
                if applied == "NO":
                    value += _number(meta.get(correction, "0"), correction, units)
                value *= scale
                if key == "RANGE" and value < 0 and not differenced:
                    raise ValueError("sequential TDM range must be nonnegative after correction.")
                if key.startswith("RECEIVE_FREQ"):
                    reference = config.reference_frequency
                    if differenced:
                        reference -= config.reference_radio.reference_frequency
                    offset_frequency = _number(meta.get("FREQ_OFFSET", "0"), "FREQ_OFFSET", "Hz")
                    if differenced:
                        value = -value
                        offset_frequency = -offset_frequency
                    # Cancel carrier offsets before adding the much smaller Doppler shift.
                    value += offset_frequency - reference
                elif differenced:
                    value = -value
                epoch = item.epoch + (offset if key in _DOPPLER and key != "DOPPLER_COUNT" else 0.0) * u.s
                try:
                    variance = float(covariance[measurement])
                    observation_bias = float(bias.get(measurement, 0.0))
                except KeyError as exc:
                    raise ValueError(f"missing TDM measurement covariance for {measurement}.") from exc
                if not np.isfinite(variance) or variance <= 0 or not np.isfinite(observation_bias):
                    raise ValueError("TDM measurement variance must be finite and positive; bias must be finite.")
                output.append((station, StationObservation(
                    float(epoch.gps), measurement, value, np.array([[variance]]), observation_bias,
                    None, None, valid=quality != "DEGRADED", radio=observation_config,
                )))
        if not output:
            raise ValueError("TDM contains no supported navigation observations.")
        return output


def _path_scale(convention, name):
    if convention not in _CONVENTIONS:
        raise ValueError(f"{name} must explicitly be 'round_trip' or 'half_round_trip' according to the source ICD.")
    return _CONVENTIONS[convention]


def _tdm_counts(segment, convention):
    if convention not in {"received_minus_reference", "reference_minus_received"}:
        raise ValueError("Doppler counts require an explicit received/reference beat-sign convention from the ICD.")
    meta = segment.metadata
    if meta.get("DOPPLER_COUNT_ROLLOVER", "NO") != "NO":
        raise ValueError("counter rollover requires source accumulator calibration and cannot be reconstructed implicitly.")
    scale_text = meta.get("DOPPLER_COUNT_SCALE", "1")
    if not re.fullmatch(r"[1-9]\d*", scale_text):
        raise ValueError("DOPPLER_COUNT_SCALE must be a positive integer.")
    scale = int(scale_text)
    bias = _number(meta.get("DOPPLER_COUNT_BIAS", "0"), "DOPPLER_COUNT_BIAS", "Hz")
    if bias < 0:
        raise ValueError("DOPPLER_COUNT_BIAS must be nonnegative.")
    samples = [item for item in segment.observations if item.keyword == "DOPPLER_COUNT"]
    if len(samples) < 2:
        raise ValueError("cumulative Doppler counts require at least two counter samples.")
    output = {}
    previous = None
    for item in samples:
        if not re.fullmatch(r"\d+", item.value.strip()):
            raise ValueError("DOPPLER_COUNT values must be nonnegative integers.")
        count = int(item.value)
        if previous is not None:
            duration = float((item.epoch - previous[0]).to_value(u.s))
            value = ((count - previous[1]) / duration - bias) / scale
            if convention == "reference_minus_received":
                value = -value
            output[float(item.epoch.gps)] = value, duration
        previous = item.epoch, count
    return output


def _tdm_media(segments, stations):
    stations = dict(stations)
    tables = {}
    targets = set()
    for segment in segments:
        kinds = {item.keyword for item in segment.observations}
        if not kinds or not kinds <= {"STEC", "TROPO_DRY", "TROPO_WET"}:
            continue
        participants = [value for key, value in segment.metadata.items() if re.fullmatch(r"PARTICIPANT_[1-5]", key)]
        station_ids = [value for value in participants if value in stations]
        if len(station_ids) != 1 or segment.metadata.get("INTERPOLATION") != "LINEAR":
            raise ValueError("media segments require one identified station and explicit LINEAR interpolation.")
        if "INTERPOLATION_DEGREE" in segment.metadata and segment.metadata["INTERPOLATION_DEGREE"] != "1":
            raise ValueError("LINEAR media interpolation requires degree 1.")
        if "STEC" in kinds:
            spacecraft = set(participants) - set(station_ids)
            if len(spacecraft) != 1 or segment.metadata.get("MODE") != "SEQUENTIAL":
                raise ValueError("STEC segments require one spacecraft and one station in SEQUENTIAL mode.")
            targets.update(spacecraft)
        station_id = station_ids[0]
        for item in segment.observations:
            values = tables.setdefault(station_id, {}).setdefault(item.keyword, [])
            epoch = float(item.epoch.gps)
            units = "TECU" if item.keyword == "STEC" else "m"
            value = _number(item.value, item.keyword, units)
            if value < 0 or (values and epoch <= values[-1][0]):
                raise ValueError("media samples must be nonnegative and chronological across segments.")
            values.append((epoch, value * (1e16 if item.keyword == "STEC" else 1)))

    def interpolate(samples, time):
        epochs, values = np.asarray(samples).T
        if len(samples) < 2 or time < epochs[0] or time > epochs[-1]:
            raise ValueError("media interpolation requires two samples and coverage of every signal event.")
        return float(np.interp(time, epochs, values))

    for station_id, channels in tables.items():
        station = stations[station_id]
        if any(len(samples) < 2 for samples in channels.values()):
            raise ValueError("each media channel requires at least two interpolation samples.")
        if "STEC" in channels:
            if callable(station.slant_tec) or station.slant_tec != 0:
                raise ValueError("TDM STEC and station TEC calibration would be applied twice.")
            station = replace(station, slant_tec=lambda time, elevation, samples=channels["STEC"]: interpolate(samples, time))
        if "TROPO_DRY" in channels or "TROPO_WET" in channels:
            if not {"TROPO_DRY", "TROPO_WET"} <= channels.keys() or station.troposphere_mapping is None:
                raise ValueError("TDM troposphere requires dry/wet samples and an explicit ICD elevation mapping.")
            if station.meteorology is not None or station.zenith_troposphere is not None:
                raise ValueError("TDM zenith delays and station troposphere calibration would be applied twice.")
            station = replace(station, zenith_troposphere=lambda time, dry=channels["TROPO_DRY"], wet=channels["TROPO_WET"]:
                              (interpolate(dry, time), interpolate(wet, time)))
        stations[station_id] = station
    return stations, targets


def _number(text, key, units=None):
    value = str(text).strip()
    match = re.fullmatch(r"([^\[\]]+?)(?:\s*\[([^\[\]]+)\])?", value)
    if match is None or (match[2] is not None and match[2].strip() != units):
        raise ValueError(f"invalid TDM value/units for {key}: {text!r}")
    try:
        result = float(match[1])
    except ValueError as exc:
        raise ValueError(f"invalid numeric TDM value for {key}: {text!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"TDM {key} must be finite.")
    return result


def _tdm_link(segment, path, stations, radio, supplied_stations):
    meta = segment.metadata
    if len(path) != 3 or path[0] == path[1] or path[1] == path[2] or not all(re.fullmatch(r"[1-5]", part) for part in path):
        raise ValueError("TDM navigation requires station-spacecraft-station paths.")
    try:
        transmit_id, spacecraft_id, receive_id = [meta[f"PARTICIPANT_{part}"] for part in path]
        transmitter, receiver, config = stations[transmit_id], stations[receive_id], radio[transmit_id]
    except KeyError as exc:
        raise ValueError(f"missing TDM participant or station/radio configuration: {exc}") from exc
    if spacecraft_id in {transmit_id, receive_id}:
        raise ValueError("TDM spacecraft must be distinct from its stations.")
    if path[0] != path[2] and transmit_id == receive_id:
        raise ValueError("three-way TDM participants must identify distinct transmitting and receiving stations.")
    if not isinstance(transmitter, GroundStation) or not isinstance(receiver, GroundStation) or not isinstance(config, RadioTrackingConfig):
        raise TypeError("stations and radio must contain GroundStation and RadioTrackingConfig objects.")
    if config.reference_receiver is not None or (config.transmitter is not None and config.transmitter != supplied_stations[transmit_id]):
        raise ValueError("supplied radio path differs from TDM participants.")
    _check_radio_metadata(meta, config, path, transmitter, receiver)
    config = _tdm_ramps(segment, path[0], config)
    if path[0] != path[2] or config.transmitter is not None:
        config = replace(config, transmitter=transmitter)
    return spacecraft_id, receiver, config


def _tdm_ramps(segment, participant, config):
    frequencies = [item for item in segment.observations if item.keyword == f"TRANSMIT_FREQ_{participant}"]
    rates = {float(item.epoch.gps): _number(item.value, item.keyword, "Hz/s") for item in segment.observations
             if item.keyword == f"TRANSMIT_FREQ_RATE_{participant}"}
    if rates and not frequencies:
        raise ValueError("frequency-rate records require matching frequency records.")
    if rates and set(rates) != {float(item.epoch.gps) for item in frequencies}:
        raise ValueError("frequency and rate records must have matching epochs.")
    if not frequencies:
        return config
    values = [(float(item.epoch.gps), _number(item.value, item.keyword, "Hz"), rates.get(float(item.epoch.gps), 0.)) for item in frequencies]
    if config.frequency_ramps is None and all(frequency == config.uplink_frequency_hz and rate == 0 for _, frequency, rate in values):
        return config
    if config.frequency_ramps is None:
        if not rates:
            raise ValueError("varying uplink records require explicit frequency rates or a supplied ramp table.")
        if "STOP_TIME" not in segment.metadata:
            raise ValueError("TDM ramp records require an explicit STOP_TIME for final coverage.")
        stop = float(_time(segment.metadata["STOP_TIME"], "UTC").gps)
        rows = [[epoch, values[index + 1][0] if index + 1 < len(values) else stop, frequency, rate]
                for index, (epoch, frequency, rate) in enumerate(values)]
        return replace(config, frequency_ramps=rows)
    for epoch, frequency, rate in values:
        if not np.isclose(config.uplink_frequency(epoch), frequency, rtol=0, atol=1e-6):
            raise ValueError("TDM transmit frequencies disagree with the supplied ramp table.")
        index = min(np.searchsorted(config.frequency_ramps[:, 0], epoch, side="right") - 1, len(config.frequency_ramps) - 1)
        if rates and rate != config.frequency_ramps[index, 3]:
            raise ValueError("TDM transmit rates disagree with the supplied ramp table.")
    return config


def _check_radio_metadata(meta, config, path, transmitter, receiver):
    numerator, denominator = meta.get("TURNAROUND_NUMERATOR"), meta.get("TURNAROUND_DENOMINATOR")
    if numerator is not None or denominator is not None:
        if numerator is None or denominator is None or not re.fullmatch(r"[1-9]\d*", numerator) or not re.fullmatch(r"[1-9]\d*", denominator):
            raise ValueError("TDM turnaround ratio requires two positive integers.")
        if int(numerator) / int(denominator) != config.turnaround_ratio:
            raise ValueError("TDM turnaround ratio differs from the supplied radio configuration.")
    delays = {f"TRANSMIT_DELAY_{path[0]}": transmitter.transmit_delay_s,
              f"RECEIVE_DELAY_{path[2]}": receiver.receive_delay_s}
    for key, expected in delays.items():
        if key in meta and _number(meta[key], key, "s") != expected:
            raise ValueError("TDM equipment delays must match the explicit station calibration.")
    spacecraft_delays = (f"TRANSMIT_DELAY_{path[1]}", f"RECEIVE_DELAY_{path[1]}")
    if any(key in meta for key in spacecraft_delays):
        delay = sum(_number(meta.get(key, "0"), key, "s") for key in spacecraft_delays)
        if delay != config.transponder_delay_s:
            raise ValueError("TDM spacecraft equipment delays must match the explicit transponder calibration.")


def _fields(items):
    fields, comments = {}, []
    for key, value in items:
        if key == "COMMENT":
            comments.append(value)
        elif key in fields:
            raise ValueError(f"duplicate TDM field {key}.")
        elif not value:
            raise ValueError(f"empty TDM field {key}.")
        else:
            fields[key] = value
    return MappingProxyType(fields), tuple(comments)


def _segment(metadata, records, data_comments):
    meta, comments = _fields(metadata)
    if "TIME_SYSTEM" not in meta or not any(re.fullmatch(r"PARTICIPANT_[1-5]", key) for key in meta):
        raise ValueError("TDM segment requires TIME_SYSTEM and PARTICIPANT_n metadata.")
    observations, last = [], {}
    bounds = {key: _time(meta[key], "UTC") for key in ("START_TIME", "STOP_TIME") if key in meta}
    if "START_TIME" in bounds and "STOP_TIME" in bounds and bounds["STOP_TIME"] < bounds["START_TIME"]:
        raise ValueError("TDM STOP_TIME precedes START_TIME.")
    for key, epoch, value in records:
        time = _time(epoch, meta["TIME_SYSTEM"])
        if key in last and time <= last[key]:
            raise ValueError(f"TDM {key} epochs must be unique and chronological within a segment.")
        if ("START_TIME" in bounds and time < bounds["START_TIME"]) or ("STOP_TIME" in bounds and time > bounds["STOP_TIME"]):
            raise ValueError("TDM observation epoch is outside START_TIME/STOP_TIME.")
        last[key] = time
        observations.append(TDMObservation(key, time, value))
    if "DATA_TYPES" in meta:
        declared = {key.strip() for key in meta["DATA_TYPES"].split(",")}
        if set(last) - declared:
            raise ValueError("TDM data contains an observable absent from DATA_TYPES.")
    return TDMSegment(meta, tuple(observations), comments, tuple(data_comments))


def _read_kvn(text):
    header, segments, metadata, records, comments = [], [], [], [], []
    section = "header"
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line in {"META_START", "META_STOP", "DATA_START", "DATA_STOP"}:
            if line == "META_START" and section in {"header", "between"}:
                section, metadata, records, comments = "metadata", [], [], []
            elif line == "META_STOP" and section == "metadata":
                section = "before_data"
            elif line == "DATA_START" and section == "before_data":
                section = "data"
            elif line == "DATA_STOP" and section == "data":
                segments.append(_segment(metadata, records, comments))
                section = "between"
            else:
                raise ValueError(f"unexpected TDM {line} at line {number}.")
            continue
        if line == "COMMENT" or line.startswith("COMMENT "):
            key, value = "COMMENT", line[7:].strip()
        elif "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
        else:
            raise ValueError(f"invalid TDM KVN line {number}: {line!r}")
        if not re.fullmatch(r"[A-Z][A-Z_0-9]*", key):
            raise ValueError(f"invalid TDM keyword at line {number}.")
        if section == "header":
            header.append((key, value))
        elif section == "metadata":
            metadata.append((key, value))
        elif section == "data":
            if key == "COMMENT":
                comments.append(value)
            else:
                parts = value.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"TDM tracking record needs epoch and value at line {number}.")
                records.append((key, *parts))
        else:
            raise ValueError(f"unexpected TDM content at line {number}.")
    if section != "between":
        raise ValueError("TDM is missing a complete metadata/data segment.")
    fields, comments = _fields(header)
    return TrackingDataMessage(fields, tuple(segments), comments)


def _read_xml(text):
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("TDM XML must not contain DTDs or entity declarations.")
    root = ET.fromstring(text)

    def local(element):
        return element.tag.rsplit("}", 1)[-1]

    def children(element, names):
        fields = {}
        for child in element:
            key = local(child)
            if key not in names or key in fields:
                raise ValueError(f"unexpected or duplicate TDM XML element {key}.")
            fields[key] = child
        if set(fields) != set(names):
            raise ValueError(f"TDM XML requires {sorted(names)}.")
        return fields

    def scalar(element):
        if len(element) or set(element.attrib) - {"units"}:
            raise ValueError("TDM XML scalar contains unexpected children or attributes.")
        value = (element.text or "").strip()
        return value + (f" [{element.attrib['units']}]" if "units" in element.attrib else "")

    if local(root) != "tdm" or root.get("id") != "CCSDS_TDM_VERS":
        raise ValueError("TDM XML requires a tdm root with id='CCSDS_TDM_VERS'.")
    sections = children(root, {"header", "body"})
    header, comments = _fields([(local(item), scalar(item)) for item in sections["header"]])
    header = MappingProxyType({"CCSDS_TDM_VERS": root.get("version", ""), **header})
    segments = []
    for element in sections["body"]:
        if local(element) != "segment":
            raise ValueError("TDM XML body must contain segments.")
        parts = children(element, {"metadata", "data"})
        metadata = [(local(item), scalar(item)) for item in parts["metadata"]]
        records, data_comments = [], []
        for item in parts["data"]:
            if local(item) == "COMMENT":
                data_comments.append(scalar(item))
            elif local(item) == "observation" and len(item) == 2:
                epochs = [child for child in item if local(child) == "EPOCH"]
                values = [child for child in item if local(child) != "EPOCH"]
                if len(epochs) != 1 or len(values) != 1:
                    raise ValueError("TDM XML observation requires one EPOCH and one observable.")
                records.append((local(values[0]), scalar(epochs[0]), scalar(values[0])))
            else:
                raise ValueError("unexpected TDM XML data element.")
        segments.append(_segment(metadata, records, data_comments))
    return TrackingDataMessage(header, tuple(segments), comments)


def read_tdm(source: str | Path | TextIO) -> TrackingDataMessage:
    """Read a TDM path, text stream, or KVN/XML text, without inferring physics.

    Scalar records retain their lexical values; unsupported navigation data can
    be inspected but are rejected by ``to_station_observations``. Time systems
    use the existing CCSDS calendar/ordinal-date conversion (SCLK unsupported).
    """
    text = source if isinstance(source, str) and source.lstrip("\ufeff\r\n\t ").startswith(("<", "CCSDS_TDM_VERS")) else _split_source(source)
    text = text.lstrip("\ufeff\r\n\t ")
    message = _read_xml(text) if text.startswith("<") else _read_kvn(text)
    if message.header.get("CCSDS_TDM_VERS") not in {"1.0", "2.0"}:
        raise ValueError("supported CCSDS TDM versions are 1.0 and 2.0.")
    if not message.header.get("ORIGINATOR") or not message.header.get("CREATION_DATE") or not message.segments:
        raise ValueError("TDM requires CREATION_DATE, ORIGINATOR and at least one segment.")
    _time(message.header["CREATION_DATE"], "UTC")
    return message
