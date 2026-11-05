"""CCSDS TDM 1.0/2.0 KVN and XML ingestion for coherent radio navigation.

The parsed message retains metadata, comments, original epochs and values.
Conversion supports unambiguous two-way range, integrated range rate and
received frequency for constant-frequency, single-station vacuum links.
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

__all__ = ["TDMObservation", "TDMSegment", "TrackingDataMessage", "read_tdm"]

_CONVENTIONS = {"round_trip": 0.5, "half_round_trip": 1.0}
_DOPPLER = {"DOPPLER_INTEGRATED", "RECEIVE_FREQ", "RECEIVE_FREQ_1"}
_MEASUREMENTS = _DOPPLER | {"RANGE"}


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
        Unsupported observables, ramps, delays and link modes raise ValueError.
        """
        output = []
        target = None
        bias = {} if bias is None else bias
        for segment in self.segments:
            meta = segment.metadata
            if meta.get("MODE") != "SEQUENTIAL" or meta.get("TIMETAG_REF", "RECEIVE") != "RECEIVE" or "PATH_1" in meta or "PATH_2" in meta:
                raise ValueError("TDM navigation requires SEQUENTIAL mode and RECEIVE time tags.")
            path = meta.get("PATH", "").split(",")
            path = [part.strip() for part in path]
            if len(path) != 3 or path[0] != path[2] or path[0] == path[1] or not all(
                re.fullmatch(r"[1-5]", part) for part in path
            ):
                raise ValueError("TDM navigation requires a single-station two-way PATH.")
            try:
                station_id = meta[f"PARTICIPANT_{path[0]}"]
                spacecraft_id = meta[f"PARTICIPANT_{path[1]}"]
                station, config = stations[station_id], radio[station_id]
            except KeyError as exc:
                raise ValueError(f"missing TDM participant or station/radio configuration: {exc}") from exc
            if station_id == spacecraft_id or (target is not None and spacecraft_id != target):
                raise ValueError("TDM navigation must describe one spacecraft distinct from its stations.")
            target = spacecraft_id
            if not isinstance(station, GroundStation) or not isinstance(config, RadioTrackingConfig):
                raise TypeError("stations and radio must contain GroundStation and RadioTrackingConfig objects.")
            kinds = {item.keyword for item in segment.observations}
            auxiliary = {f"TRANSMIT_FREQ_{path[0]}", f"TRANSMIT_FREQ_RATE_{path[0]}"}
            unsupported = kinds - _MEASUREMENTS - auxiliary
            if unsupported:
                raise ValueError(f"unsupported TDM navigation observables: {sorted(unsupported)}")
            _check_radio_metadata(meta, config)
            for item in segment.observations:
                if item.keyword in auxiliary:
                    expected = config.uplink_frequency_hz if "RATE" not in item.keyword else 0.0
                    units = "Hz" if "RATE" not in item.keyword else "Hz/s"
                    if _number(item.value, item.keyword, units) != expected:
                        raise ValueError("TDM transmit frequencies must match the constant uplink; ramps are unsupported.")
            has_doppler = bool(kinds & _DOPPLER)
            offset = 0.0
            if has_doppler:
                interval = _number(meta.get("INTEGRATION_INTERVAL", ""), "INTEGRATION_INTERVAL", "s")
                reference = meta.get("INTEGRATION_REF")
                if interval <= 0 or reference not in {"START", "MIDDLE", "END"}:
                    raise ValueError("TDM Doppler requires positive INTEGRATION_INTERVAL and START/MIDDLE/END INTEGRATION_REF.")
                config = replace(config, count_interval_s=interval)
                offset = {"START": interval, "MIDDLE": interval / 2, "END": 0.0}[reference]
            quality = meta.get("DATA_QUALITY", "RAW")
            if quality not in {"RAW", "VALIDATED", "DEGRADED"}:
                raise ValueError("unsupported TDM DATA_QUALITY.")
            corrections = {key: value for key, value in meta.items() if key.startswith("CORRECTION_")}
            applied = meta.get("CORRECTIONS_APPLIED")
            if (corrections or applied is not None) and applied not in {"YES", "NO"}:
                raise ValueError("TDM corrections require CORRECTIONS_APPLIED=YES or NO.")
            range_units = meta.get("RANGE_UNITS", "km")
            for key, value in corrections.items():
                units = {"CORRECTION_RANGE": range_units, "CORRECTION_DOPPLER": "km/s", "CORRECTION_RECEIVE": "Hz"}.get(key)
                correction_value = _number(value, key, units)
                if units is None and correction_value != 0:
                    raise ValueError(f"unsupported nonzero TDM correction {key}.")
            for item in segment.observations:
                key = item.keyword
                if key in auxiliary:
                    continue
                if key == "RANGE":
                    if range_units not in {"km", "s"} or meta.get("RANGE_MODE", "CONSTANT") not in {"CONSTANT", "COHERENT"}:
                        raise ValueError("TDM navigation supports unambiguous range in km or s, without RU conversion.")
                    if _number(meta.get("RANGE_MODULUS", "0"), "RANGE_MODULUS", range_units) != 0:
                        raise ValueError("ambiguous TDM range is unsupported.")
                    scale = _path_scale(range_convention, "range_convention") * (1000.0 if range_units == "km" else c)
                    units, measurement, correction = range_units, "two_way_range", "CORRECTION_RANGE"
                elif key == "DOPPLER_INTEGRATED":
                    scale = -2 * _path_scale(doppler_convention, "doppler_convention") * config.uplink_frequency_hz * config.turnaround_ratio * 1000 / c
                    units, measurement, correction = "km/s", "two_way_doppler", "CORRECTION_DOPPLER"
                else:
                    scale = 1.0
                    units, measurement, correction = "Hz", "two_way_doppler", "CORRECTION_RECEIVE"
                value = _number(item.value, key, units)
                if applied == "NO":
                    value += _number(meta.get(correction, "0"), correction, units)
                value *= scale
                if key == "RANGE" and value < 0:
                    raise ValueError("two-way TDM range must be nonnegative after correction.")
                if key.startswith("RECEIVE_FREQ"):
                    value += _number(meta.get("FREQ_OFFSET", "0"), "FREQ_OFFSET", "Hz") - config.uplink_frequency_hz * config.turnaround_ratio
                epoch = item.epoch + (offset if key in _DOPPLER else 0.0) * u.s
                try:
                    variance = float(covariance[measurement])
                    observation_bias = float(bias.get(measurement, 0.0))
                except KeyError as exc:
                    raise ValueError(f"missing TDM measurement covariance for {measurement}.") from exc
                if not np.isfinite(variance) or variance <= 0 or not np.isfinite(observation_bias):
                    raise ValueError("TDM measurement variance must be finite and positive; bias must be finite.")
                output.append((station, StationObservation(
                    float(epoch.gps), measurement, value, np.array([[variance]]), observation_bias,
                    None, None, valid=quality != "DEGRADED", radio=config,
                )))
        if not output:
            raise ValueError("TDM contains no supported navigation observations.")
        return output


def _path_scale(convention, name):
    if convention not in _CONVENTIONS:
        raise ValueError(f"{name} must explicitly be 'round_trip' or 'half_round_trip' according to the source ICD.")
    return _CONVENTIONS[convention]


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


def _check_radio_metadata(meta, config):
    numerator, denominator = meta.get("TURNAROUND_NUMERATOR"), meta.get("TURNAROUND_DENOMINATOR")
    if numerator is not None or denominator is not None:
        if numerator is None or denominator is None or not re.fullmatch(r"[1-9]\d*", numerator) or not re.fullmatch(r"[1-9]\d*", denominator):
            raise ValueError("TDM turnaround ratio requires two positive integers.")
        if int(numerator) / int(denominator) != config.turnaround_ratio:
            raise ValueError("TDM turnaround ratio differs from the supplied radio configuration.")
    for key, value in meta.items():
        if key.startswith(("TRANSMIT_DELAY_", "RECEIVE_DELAY_")) and _number(value, key, "s") != 0:
            raise ValueError("nonzero TDM equipment delays require calibration beyond the supported radio model.")


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
