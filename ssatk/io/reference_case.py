"""Portable trajectory exports for independent propagation comparisons."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time

__all__ = [
    "ReferenceCase",
    "ReferenceCaseFiles",
    "ReferenceCaseSegment",
    "compare_reference_case",
    "read_oem",
    "read_reference_case",
    "write_reference_case",
]

_SUPPORTED_COVARIANCE_FRAMES = frozenset(
    {"GCRF", "ICRF", "EME2000", "ITRF", "RTN", "RIC", "RSW", "QSW", "TNW", "VNC", "LVLH", "TEME"}
)


@dataclass(frozen=True)
class ReferenceCaseFiles:
    """Files written for a portable SSATK reference case."""

    metadata_path: Path
    ephemeris_path: Path


@dataclass(frozen=True)
class ReferenceCase:
    """Trajectory and metadata read from an SSATK reference case."""

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    metadata: Mapping[str, object]
    segments: tuple[ReferenceCaseSegment, ...] = ()
    covariance_t: np.ndarray | None = None
    covariance: np.ndarray | None = None
    covariance_reference_frame: str | None = None
    acceleration_t: np.ndarray | None = None
    acceleration: np.ndarray | None = None


@dataclass(frozen=True)
class ReferenceCaseSegment:
    """One CCSDS OEM segment with SI states, acceleration, and KVN metadata."""

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    metadata: Mapping[str, object]
    comments: tuple[str, ...] = ()
    covariance_t: np.ndarray | None = None
    covariance: np.ndarray | None = None
    covariance_reference_frame: str | None = None
    acceleration_t: np.ndarray | None = None
    acceleration: np.ndarray | None = None


def read_reference_case(source) -> ReferenceCase:
    """Read an SSATK JSON/OEM reference case into SI Cartesian arrays.

    ``source`` may be either the sidecar JSON or the CCSDS OEM file. If only
    the OEM is available, ``t`` starts at zero because its original time
    origin is stored only in the JSON sidecar.
    """

    source = Path(source)
    if source.suffix.lower() == ".json":
        metadata_path = source
        metadata = json.loads(source.read_text(encoding="utf-8"))
        ephemeris_path = source.parent / metadata["files"]["ephemeris"]
    elif source.suffix.lower() == ".oem":
        ephemeris_path = source
        metadata_path = source.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
    else:
        raise ValueError("source must be an SSATK .json or CCSDS .oem file.")
    if not ephemeris_path.exists():
        raise FileNotFoundError(ephemeris_path)
    oem = read_oem(ephemeris_path)
    time_origin = float(metadata.get("time_origin_s", 0.0))
    epoch_offset = 0.0
    if metadata.get("epoch"):
        sidecar_epoch = _oem_epoch(metadata["epoch"], "UTC")
        oem_epoch = _oem_epoch(oem.metadata["epoch"], oem.metadata["TIME_SYSTEM"])
        epoch_offset = float((oem_epoch - sidecar_epoch).sec)
    times = time_origin + epoch_offset + oem.t
    if metadata.get("sample_count") not in (None, len(oem.t)):
        raise ValueError("OEM sample count does not match reference-case metadata.")
    covariance_count = metadata.get("covariance_count")
    actual_covariance_count = 0 if oem.covariance_t is None else len(oem.covariance_t)
    if covariance_count not in (None, actual_covariance_count):
        raise ValueError("OEM covariance count does not match reference-case metadata.")
    covariance_frame = metadata.get("covariance_reference_frame")
    if (
        covariance_frame is not None
        and oem.covariance_reference_frame is not None
        and str(covariance_frame).strip().upper() != oem.covariance_reference_frame
    ):
        raise ValueError("OEM covariance reference frame does not match reference-case metadata.")
    covariance_epochs = metadata.get("covariance_epochs_s")
    if covariance_epochs is not None:
        if oem.covariance_t is None:
            raise ValueError("reference-case metadata declares covariance epochs but OEM has none.")
        covariance_epochs = np.asarray(covariance_epochs, dtype=float)
        if covariance_epochs.shape != oem.covariance_t.shape or not np.allclose(
            covariance_epochs, oem.covariance_t, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("OEM covariance epochs do not match reference-case metadata.")
    acceleration_count = metadata.get("acceleration_count")
    actual_acceleration_count = 0 if oem.acceleration_t is None else len(oem.acceleration_t)
    if acceleration_count not in (None, actual_acceleration_count):
        raise ValueError("OEM acceleration count does not match reference-case metadata.")
    acceleration_units = metadata.get("acceleration_units")
    if acceleration_units is not None:
        if oem.acceleration_t is None:
            raise ValueError("reference-case metadata declares acceleration units but OEM has none.")
        if str(acceleration_units).strip().lower() != "m/s^2":
            raise ValueError("reference-case acceleration units must be m/s^2.")
    acceleration_order = metadata.get("acceleration_order")
    if acceleration_order is not None and list(acceleration_order) != ["ax", "ay", "az"]:
        raise ValueError("reference-case acceleration order must be [ax, ay, az].")
    acceleration_epochs = metadata.get("acceleration_epochs_s")
    if acceleration_epochs is not None:
        if oem.acceleration_t is None:
            raise ValueError("reference-case metadata declares acceleration epochs but OEM has none.")
        acceleration_epochs = np.asarray(acceleration_epochs, dtype=float)
        if acceleration_epochs.shape != oem.acceleration_t.shape or not np.allclose(
            acceleration_epochs, oem.acceleration_t, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("OEM acceleration epochs do not match reference-case metadata.")
    acceleration_segment_counts = metadata.get("acceleration_segment_counts")
    if acceleration_segment_counts is not None:
        actual_segment_counts = [
            0 if segment.acceleration_t is None else len(segment.acceleration_t)
            for segment in oem.segments
        ]
        if acceleration_segment_counts != actual_segment_counts:
            raise ValueError("OEM acceleration segment counts do not match reference-case metadata.")
    if not np.all(np.diff(times) > 0.0) and len(times) > 1:
        raise ValueError("OEM epochs must be strictly increasing.")
    covariance_t = None if oem.covariance_t is None else oem.covariance_t + time_origin + epoch_offset
    return ReferenceCase(
        t=times,
        r=oem.r,
        v=oem.v,
        metadata={**oem.metadata, **metadata},
        segments=tuple(
            replace(
                segment,
                t=segment.t + time_origin + epoch_offset,
                covariance_t=None if segment.covariance_t is None else segment.covariance_t + time_origin + epoch_offset,
                acceleration_t=None if segment.acceleration_t is None else segment.acceleration_t + time_origin + epoch_offset,
            )
            for segment in oem.segments
        ),
        covariance_t=covariance_t,
        covariance=oem.covariance,
        covariance_reference_frame=oem.covariance_reference_frame,
        acceleration_t=None if oem.acceleration_t is None else oem.acceleration_t + time_origin + epoch_offset,
        acceleration=oem.acceleration,
    )


def read_oem(source) -> ReferenceCase:
    """Read a CCSDS OEM 2.0 KVN message into SI state and optional acceleration arrays."""

    text = _oem_text_source(source)
    header, parsed_segments = _parse_oem(text)
    if header.get("CCSDS_OEM_VERS") != "2.0":
        raise ValueError("only CCSDS OEM version 2.0 is supported.")
    for key in ("CREATION_DATE", "ORIGINATOR"):
        if not header.get(key):
            raise ValueError(f"OEM header is missing required {key}.")
    _oem_epoch(header["CREATION_DATE"], "UTC")
    if not parsed_segments:
        raise ValueError("CCSDS OEM contains no state segments.")

    common = None
    all_epochs = []
    segments = []
    for metadata, comments, records, covariance_records in parsed_segments:
        segment = _parse_oem_segment(metadata, comments, records, covariance_records)
        if common is None:
            common = {
                key: metadata[key].strip().upper()
                for key in ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM")
            }
        elif any(metadata[key].strip().upper() != common[key] for key in common):
            raise ValueError("OEM segments must describe the same object, center, frame, and time system.")
        all_epochs.extend(segment[0])
        segments.append(segment)

    first_epoch = all_epochs[0]
    times = np.asarray([float((epoch - first_epoch).sec) for epoch in all_epochs])
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("OEM epochs must be strictly increasing across segments.")
    offset = 0
    segment_objects = []
    positions = []
    velocities = []
    covariance_times = []
    covariances = []
    covariance_frames = set()
    acceleration_arrays = []
    acceleration_times = []
    for metadata, comments, (epochs, states, covariance_epochs, covariance_values, covariance_frame, accelerations) in zip(
        (item[0] for item in parsed_segments),
        (item[1] for item in parsed_segments),
        segments,
    ):
        count = len(states)
        segment_times = times[offset:offset + count]
        offset += count
        segment_objects.append(
            ReferenceCaseSegment(
                t=segment_times,
                r=states[:, :3] * 1.0e3,
                v=states[:, 3:] * 1.0e3,
                metadata=metadata,
                comments=comments,
                covariance_t=None if not covariance_epochs else np.asarray(
                    [float((epoch - first_epoch).sec) for epoch in covariance_epochs]
                ),
                covariance=None if covariance_values is None else covariance_values * 1.0e6,
                covariance_reference_frame=covariance_frame,
                acceleration_t=None if accelerations is None else segment_times.copy(),
                acceleration=None if accelerations is None else accelerations * 1.0e3,
            )
        )
        positions.append(states[:, :3])
        velocities.append(states[:, 3:])
        if covariance_epochs:
            covariance_times.extend(float((epoch - first_epoch).sec) for epoch in covariance_epochs)
            covariances.append(covariance_values)
            covariance_frames.add(covariance_frame)
        if accelerations is not None:
            acceleration_arrays.append(accelerations)
            acceleration_times.extend(segment_times)
    if len(covariance_times) > 1 and not np.all(np.diff(covariance_times) > 0.0):
        raise ValueError("OEM covariance epochs must be strictly increasing across segments.")

    header_comments = tuple(header.pop("_COMMENTS", ()))
    aggregate = dict(header)
    aggregate.update(parsed_segments[0][0])
    aggregate.update({
        "format": "ccsds-oem",
        "schema_version": "2.0",
        "segment_count": len(segments),
        "epoch": parsed_segments[0][2][0][0],
    })
    aggregate.update({
        "center_name": parsed_segments[0][0]["CENTER_NAME"],
        "reference_frame": parsed_segments[0][0]["REF_FRAME"],
        "time_system": parsed_segments[0][0]["TIME_SYSTEM"],
    })
    aggregate["comments"] = header_comments
    if covariance_frames:
        aggregate["covariance_count"] = len(covariance_times)
        aggregate["covariance_reference_frame"] = (
            next(iter(covariance_frames)) if len(covariance_frames) == 1 else "MIXED"
        )
    if acceleration_arrays:
        aggregate.update({
            "acceleration_count": int(sum(len(values) for values in acceleration_arrays)),
            "acceleration_units": "m/s^2",
            "acceleration_epochs_s": acceleration_times,
            "acceleration_segment_counts": [
                0 if segment.acceleration_t is None else len(segment.acceleration_t)
                for segment in segment_objects
            ],
        })
    return ReferenceCase(
        t=times,
        r=np.vstack(positions) * 1.0e3,
        v=np.vstack(velocities) * 1.0e3,
        metadata=aggregate,
        segments=tuple(segment_objects),
        covariance_t=None if not covariance_times else np.asarray(covariance_times),
        covariance=None if not covariances else np.vstack(covariances) * 1.0e6,
        covariance_reference_frame=(
            None
            if not covariance_frames
            else next(iter(covariance_frames)) if len(covariance_frames) == 1 else "MIXED"
        ),
        acceleration_t=None if not acceleration_arrays else np.asarray(acceleration_times),
        acceleration=None if not acceleration_arrays else np.vstack(acceleration_arrays) * 1.0e3,
    )


def compare_reference_case(trajectory, reference, *, time_tolerance=1e-9):
    """Compare a trajectory with reference states at the reference epochs."""
    actual_t, actual_r, actual_v = _trajectory_arrays(trajectory)
    reference = read_reference_case(reference) if isinstance(reference, (str, Path)) else reference
    if not all(hasattr(reference, name) for name in ("t", "r", "v")):
        raise TypeError("reference must be a ReferenceCase or .json/.oem path.")
    reference_t, reference_r, reference_v = _trajectory_arrays(reference)
    time_tolerance = float(time_tolerance)
    if not np.isfinite(time_tolerance) or time_tolerance < 0.0:
        raise ValueError("time_tolerance must be non-negative.")
    for name in ("reference_frame", "center_name"):
        actual_metadata = getattr(trajectory, "metadata", {})
        reference_metadata = getattr(reference, "metadata", {})
        actual_value = actual_metadata.get(name) if isinstance(actual_metadata, Mapping) else None
        reference_value = reference_metadata.get(name) if isinstance(reference_metadata, Mapping) else None
        if (
            actual_value is not None
            and reference_value is not None
            and str(actual_value).strip().upper() != str(reference_value).strip().upper()
        ):
            raise ValueError(f"trajectory and reference {name} metadata do not match.")
    if reference_t[0] < actual_t[0] - time_tolerance or reference_t[-1] > actual_t[-1] + time_tolerance:
        raise ValueError("reference epochs must overlap the trajectory time span.")
    sample_t = np.clip(reference_t, actual_t[0], actual_t[-1])
    positions = np.array([np.interp(sample_t, actual_t, actual_r[:, i]) for i in range(3)]).T
    velocities = np.array([np.interp(sample_t, actual_t, actual_v[:, i]) for i in range(3)]).T
    position_residual = np.linalg.norm(positions - reference_r, axis=1)
    velocity_residual = np.linalg.norm(velocities - reference_v, axis=1)
    return {
        "max_position_m": float(position_residual.max()),
        "rms_position_m": float(np.sqrt(np.mean(position_residual**2))),
        "final_position_m": float(position_residual[-1]),
        "max_velocity_m_s": float(velocity_residual.max()),
        "rms_velocity_m_s": float(np.sqrt(np.mean(velocity_residual**2))),
        "final_velocity_m_s": float(velocity_residual[-1]),
        "sample_count": int(reference_t.size),
    }


def write_reference_case(
    trajectory,
    output_dir,
    *,
    epoch,
    case_name: str = "ssatk_reference_case",
    center_name: str = "Earth",
    reference_frame: str = "GCRF",
    time_system: str = "UTC",
    force_models: Iterable[str] = (),
    constants: Mapping[str, object] | None = None,
    integrator: Mapping[str, object] | None = None,
    precision: int = 17,
    covariance: np.ndarray | None = None,
    covariance_t: np.ndarray | None = None,
    covariance_reference_frame: str | None = None,
    acceleration: np.ndarray | None = None,
    acceleration_t: np.ndarray | None = None,
    overwrite: bool = False,
) -> ReferenceCaseFiles:
    """Write a trajectory as CCSDS OEM 2.0 plus a JSON case description.

    ``trajectory`` must expose ``t``, ``r``, and ``v`` arrays. Times are
    interpreted as seconds from the first sample; OEM positions and velocities
    are written in km and km/s, while the JSON sidecar records SI states too.
    Optional ``acceleration`` values use SI units and shape ``(N, 3)``; they
    are written as the optional three acceleration components on each OEM
    state record. ``acceleration_t`` must match the state epochs.
    Optional ``covariance`` values use SI state units and shape ``(M, 6, 6)``;
    ``covariance_t`` is measured from the first state sample and
    ``covariance_reference_frame`` is preserved without frame rotation.
    The sidecar is intentionally explicit so another propagator can reproduce
    the case without inferring frames, constants, or tolerances from a plot.
    """

    times, positions, velocities = _trajectory_arrays(trajectory)
    epoch_utc = _epoch_utc(epoch)
    case_name = _case_name(case_name)
    if not isinstance(time_system, str) or time_system.upper() != "UTC":
        raise ValueError("write_reference_case currently supports only UTC.")
    if not isinstance(reference_frame, str) or not reference_frame.strip():
        raise ValueError("reference_frame must be a non-empty string.")
    if not isinstance(center_name, str) or not center_name.strip():
        raise ValueError("center_name must be a non-empty string.")
    if not isinstance(precision, int) or not 1 <= precision <= 17:
        raise ValueError("precision must be an integer from 1 through 17.")
    if covariance is None:
        covariance = getattr(trajectory, "covariance", None)
    if acceleration is None:
        acceleration = getattr(trajectory, "acceleration", None)
    elapsed = times - times[0]
    covariance, covariance_t, covariance_reference_frame = _normalize_oem_covariance(
        covariance,
        covariance_t,
        covariance_reference_frame or getattr(trajectory, "covariance_reference_frame", None),
        elapsed,
        reference_frame,
    )
    acceleration, acceleration_t = _normalize_oem_acceleration(acceleration, acceleration_t, elapsed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{case_name}.json"
    ephemeris_path = output_dir / f"{case_name}.oem"
    if not overwrite and (metadata_path.exists() or ephemeris_path.exists()):
        raise FileExistsError(f"reference case already exists: {case_name}")

    models = [force_models] if isinstance(force_models, str) else [str(model) for model in force_models]
    metadata = {
        "format": "ssatk-reference-case",
        "schema_version": 1,
        "case_name": case_name,
        "epoch": _format_epoch(epoch_utc),
        "time_system": "UTC",
        "center_name": center_name,
        "reference_frame": reference_frame,
        "state_units": {"time": "s", "position": "m", "velocity": "m/s"},
        "state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "time_origin_s": float(times[0]),
        "time_span_s": float(elapsed[-1]),
        "sample_count": int(times.size),
        "force_models": models,
        "constants": dict(constants or {}),
        "integrator": dict(integrator or {}),
        "numeric_precision_digits": precision,
        "initial_state": np.concatenate((positions[0], velocities[0])).tolist(),
        "final_state": np.concatenate((positions[-1], velocities[-1])).tolist(),
        "files": {"ephemeris": ephemeris_path.name},
    }
    if covariance is not None:
        metadata.update({
            "covariance_count": int(covariance.shape[0]),
            "covariance_reference_frame": covariance_reference_frame,
            "covariance_units": "km^2 for the canonical [km, km/s] state covariance",
            "covariance_epochs_s": covariance_t.tolist(),
        })
    if acceleration is not None:
        metadata.update({
            "acceleration_count": int(acceleration.shape[0]),
            "acceleration_units": "m/s^2",
            "acceleration_epochs_s": acceleration_t.tolist(),
            "acceleration_order": ["ax", "ay", "az"],
            "acceleration_segment_counts": [int(acceleration.shape[0])],
        })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ephemeris_path.write_text(
        _oem_text(
            case_name,
            epoch_utc,
            elapsed,
            positions,
            velocities,
            center_name=center_name,
            reference_frame=reference_frame,
            precision=precision,
            covariance=covariance,
            covariance_t=covariance_t,
            covariance_reference_frame=covariance_reference_frame,
            acceleration=acceleration,
        ),
        encoding="utf-8",
    )
    return ReferenceCaseFiles(metadata_path, ephemeris_path)


def _oem_text_source(source):
    if hasattr(source, "read"):
        value = source.read()
        return value.decode() if isinstance(value, bytes) else str(value)
    if isinstance(source, Path) or hasattr(source, "__fspath__"):
        return Path(source).read_text(encoding="utf-8")
    if isinstance(source, str):
        if "\n" in source or "\r" in source or source.lstrip().startswith("CCSDS_"):
            return source
        return Path(source).read_text(encoding="utf-8")
    raise TypeError("source must be CCSDS OEM text, a path, or a text stream.")


def _oem_key_value(text):
    if "=" not in text:
        raise ValueError(f"invalid CCSDS OEM key/value line: {text!r}")
    key, value = text.split("=", 1)
    key = key.strip().upper()
    if not key or not value.strip():
        raise ValueError(f"invalid CCSDS OEM key/value line: {text!r}")
    return key, value.strip()


def _oem_comment(text):
    value = text[len("COMMENT"):].strip()
    return value[1:].strip() if value.startswith("=") else value


def _parse_oem(text):
    header = {}
    header_comments = []
    parsed = []
    metadata = None
    comments = []
    records = []
    covariance_records = []
    covariance_metadata = None
    covariance_lines = []
    covariance_comments = []
    section = "header"
    data_open = False

    def finish():
        nonlocal metadata, comments, records, covariance_records, section, data_open
        if metadata is None:
            return
        if covariance_metadata is not None:
            raise ValueError("CCSDS OEM COVARIANCE_START is missing COVARIANCE_STOP.")
        if not records:
            raise ValueError("CCSDS OEM segment contains no state records.")
        parsed.append((metadata, tuple(comments), records, covariance_records))
        metadata = None
        comments = []
        records = []
        covariance_records = []
        section = "header"
        data_open = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        marker = line.upper()
        if marker == "COVARIANCE_START":
            if metadata is None or section == "meta" or covariance_metadata is not None:
                raise ValueError("CCSDS OEM COVARIANCE_START is out of order.")
            covariance_metadata = {}
            covariance_lines = []
            covariance_comments = []
            continue
        if marker == "COVARIANCE_STOP":
            if covariance_metadata is None:
                raise ValueError("CCSDS OEM COVARIANCE_STOP is out of order.")
            covariance_metadata["_LINES"] = tuple(covariance_lines)
            covariance_metadata["_COMMENTS"] = tuple(covariance_comments)
            covariance_records.append(covariance_metadata)
            covariance_metadata = None
            covariance_lines = []
            covariance_comments = []
            continue
        if marker == "META_START":
            if section == "meta":
                raise ValueError("duplicate CCSDS OEM META_START.")
            if metadata is not None:
                if data_open:
                    raise ValueError("CCSDS OEM DATA_START is missing DATA_STOP.")
                finish()
            metadata = {}
            comments = []
            records = []
            section = "meta"
            continue
        if marker == "META_STOP":
            if section != "meta" or metadata is None:
                raise ValueError("CCSDS OEM META_STOP is out of order.")
            section = "data"
            continue
        if marker == "DATA_START":
            if metadata is None or section == "meta" or data_open:
                raise ValueError("CCSDS OEM DATA_START is out of order.")
            data_open = True
            section = "data"
            continue
        if marker == "DATA_STOP":
            if metadata is None or section != "data" or not data_open:
                raise ValueError("CCSDS OEM DATA_STOP is out of order.")
            finish()
            continue
        if marker.startswith("COMMENT"):
            if covariance_metadata is not None:
                covariance_comments.append(_oem_comment(line))
            elif metadata is None:
                header_comments.append(_oem_comment(line))
            else:
                comments.append(_oem_comment(line))
            continue

        if covariance_metadata is not None:
            if "=" not in line:
                covariance_lines.append(line)
                continue
            key, value = _oem_key_value(line)
            if key in covariance_metadata:
                raise ValueError(f"duplicate CCSDS OEM covariance key {key}.")
            covariance_metadata[key] = value
            continue

        if section == "data":
            fields = line.replace("D", "E").replace("d", "e").split()
            if len(fields) not in (7, 10):
                raise ValueError(
                    "CCSDS OEM state records must contain one epoch and six or nine components."
                )
            try:
                values = np.asarray(fields[1:], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CCSDS OEM state record: {line!r}") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError("CCSDS OEM state and acceleration records must be finite.")
            records.append((fields[0], values[:6], None if len(values) == 6 else values[6:]))
            continue

        key, value = _oem_key_value(line)
        target = header if metadata is None else metadata
        if key in target:
            raise ValueError(f"duplicate CCSDS OEM key {key}.")
        target[key] = value

    if section == "meta":
        raise ValueError("CCSDS OEM META_START is missing META_STOP.")
    if covariance_metadata is not None:
        raise ValueError("CCSDS OEM COVARIANCE_START is missing COVARIANCE_STOP.")
    if metadata is not None:
        if data_open:
            raise ValueError("CCSDS OEM DATA_START is missing DATA_STOP.")
        finish()
    header["_COMMENTS"] = tuple(header_comments)
    return header, parsed


def _parse_oem_segment(metadata, comments, records, covariance_records=()):
    required = ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM", "START_TIME", "STOP_TIME")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError("OEM segment is missing required " + ", ".join(missing) + ".")
    scale = metadata["TIME_SYSTEM"].upper()
    start = _oem_epoch(metadata["START_TIME"], scale)
    stop = _oem_epoch(metadata["STOP_TIME"], scale)
    if stop < start:
        raise ValueError("OEM STOP_TIME must not precede START_TIME.")
    epochs = []
    states = []
    accelerations = []
    acceleration_presence = None
    for epoch_text, state, acceleration in records:
        epoch = _oem_epoch(epoch_text, scale)
        if epoch < start or epoch > stop:
            raise ValueError("OEM state epoch is outside the segment START_TIME/STOP_TIME span.")
        if epochs and epoch <= epochs[-1]:
            raise ValueError("OEM state epochs must be strictly increasing within a segment.")
        has_acceleration = acceleration is not None
        if acceleration_presence is not None and has_acceleration != acceleration_presence:
            raise ValueError("OEM acceleration records must be present for every state epoch in a segment.")
        acceleration_presence = has_acceleration
        epochs.append(epoch)
        states.append(state)
        if acceleration is not None:
            accelerations.append(acceleration)
    covariance_epochs = []
    covariance_values = []
    covariance_frame = None
    for covariance_record in covariance_records:
        epoch, covariance, frame = _parse_oem_covariance(
            covariance_record, scale, metadata["REF_FRAME"]
        )
        if epoch < start or epoch > stop:
            raise ValueError("OEM covariance epoch is outside the segment START_TIME/STOP_TIME span.")
        if covariance_epochs and epoch <= covariance_epochs[-1]:
            raise ValueError("OEM covariance epochs must be strictly increasing within a segment.")
        if covariance_frame is not None and frame != covariance_frame:
            raise ValueError("OEM covariance reference frame must be constant within a segment.")
        covariance_epochs.append(epoch)
        covariance_values.append(covariance)
        covariance_frame = frame
    return (
        epochs,
        np.vstack(states),
        covariance_epochs,
        None if not covariance_values else np.stack(covariance_values),
        covariance_frame,
        None if acceleration_presence is not True else np.vstack(accelerations),
    )


def _parse_oem_covariance(record, scale, default_frame):
    if not record.get("EPOCH"):
        raise ValueError("OEM covariance block is missing EPOCH.")
    frame = str(record.get("COV_REF_FRAME", default_frame)).strip().upper()
    if frame not in _SUPPORTED_COVARIANCE_FRAMES:
        raise ValueError(f"unsupported OEM covariance reference frame {frame!r}.")
    lines = tuple(record.get("_LINES", ()))
    unknown = set(record) - {"EPOCH", "COV_REF_FRAME", "_LINES", "_COMMENTS"}
    if unknown:
        raise ValueError("unsupported OEM covariance fields: " + ", ".join(sorted(unknown)) + ".")
    if len(lines) != 6 or any(
        len(line.replace("D", "E").replace("d", "e").split()) != row + 1
        for row, line in enumerate(lines)
    ):
        raise ValueError("OEM KVN covariance blocks must contain six triangular rows.")
    covariance = np.zeros((6, 6), dtype=float)
    try:
        for row, line in enumerate(lines):
            row_values = [float(value) for value in line.replace("D", "E").replace("d", "e").split()]
            for column, value in enumerate(row_values):
                covariance[row, column] = covariance[column, row] = value
    except (TypeError, ValueError) as exc:
        raise ValueError("OEM covariance entries must be finite numbers.") from exc
    if not np.all(np.isfinite(covariance)):
        raise ValueError("OEM covariance entries must be finite numbers.")
    scale_value = max(1.0, float(np.max(np.abs(covariance))))
    if np.min(np.linalg.eigvalsh(covariance)) < -1.0e-12 * scale_value:
        raise ValueError("OEM covariance must be positive semidefinite.")
    return _oem_epoch(record["EPOCH"], scale), covariance, frame


def _oem_epoch(value, scale):
    scale = str(scale).upper()
    scales = {"UTC": "utc", "TAI": "tai", "TT": "tt", "TDB": "tdb", "TCB": "tcb", "TCG": "tcg", "UT1": "ut1"}
    if scale not in scales and scale != "GPS":
        raise ValueError(f"unsupported OEM time system {scale!r}.")
    text = str(value).strip().removesuffix("Z")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*time is after end of day.*")
            if scale == "GPS":
                result = Time(_oem_time_text(text), format=_oem_time_format(text), scale="tai") + 19.0 * u.s
            else:
                result = Time(_oem_time_text(text), format=_oem_time_format(text), scale=scales[scale])
    except (TypeError, ValueError, Warning) as exc:
        raise ValueError(f"invalid OEM {scale} epoch {value!r}.") from exc
    if not np.all(np.isfinite(np.asarray(result.jd))):
        raise ValueError(f"invalid OEM {scale} epoch {value!r}.")
    return result


def _oem_time_text(text):
    if len(text) > 8 and text[4] == "-" and text[8] == "T":
        return f"{text[:4]}:{text[5:8]}:{text[9:]}"
    return text


def _oem_time_format(text):
    return "yday" if len(text) > 8 and text[4] == "-" and text[8] == "T" else "isot"


def _trajectory_arrays(trajectory):
    try:
        times = np.asarray(trajectory.t, dtype=float)
        positions = np.asarray(trajectory.r, dtype=float)
        velocities = np.asarray(trajectory.v, dtype=float)
    except AttributeError as exc:
        raise TypeError("trajectory must expose t, r, and v arrays.") from exc
    if times.ndim != 1 or times.size == 0:
        raise ValueError("trajectory.t must be a non-empty 1-D array.")
    if positions.shape != (times.size, 3) or velocities.shape != (times.size, 3):
        raise ValueError("trajectory r and v arrays must each have shape (N, 3).")
    if not np.all(np.isfinite(np.concatenate((times, positions.ravel(), velocities.ravel())))):
        raise ValueError("trajectory arrays must contain only finite values.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("trajectory.t must be strictly increasing.")
    return times, positions, velocities


def _normalize_oem_covariance(covariance, covariance_t, covariance_reference_frame, elapsed, default_frame):
    if covariance is None:
        if covariance_t is not None or covariance_reference_frame is not None:
            raise ValueError("covariance_t and covariance_reference_frame require covariance.")
        return None, None, None
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 3 or covariance.shape[1:] != (6, 6) or covariance.shape[0] == 0:
        raise ValueError("covariance must have shape (M, 6, 6) with M > 0.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values.")
    for matrix in covariance:
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12 * scale):
            raise ValueError("covariance matrices must be symmetric.")
        if np.min(np.linalg.eigvalsh(matrix)) < -1.0e-12 * scale:
            raise ValueError("covariance matrices must be positive semidefinite.")
    if covariance_t is None:
        if covariance.shape[0] != elapsed.size:
            raise ValueError("covariance_t is required when covariance does not have one sample per state epoch.")
        covariance_t = elapsed.copy()
    else:
        covariance_t = np.asarray(covariance_t, dtype=float)
        if covariance_t.shape != (covariance.shape[0],):
            raise ValueError("covariance_t must have one epoch per covariance matrix.")
    if not np.all(np.isfinite(covariance_t)) or (
        covariance_t.size > 1 and not np.all(np.diff(covariance_t) > 0.0)
    ):
        raise ValueError("covariance_t must be finite and strictly increasing.")
    if covariance_t[0] < elapsed[0] or covariance_t[-1] > elapsed[-1]:
        raise ValueError("covariance_t must lie within the state time span.")
    frame = str(covariance_reference_frame or default_frame).strip().upper()
    if frame not in _SUPPORTED_COVARIANCE_FRAMES:
        raise ValueError(f"unsupported OEM covariance reference frame {frame!r}.")
    return covariance, covariance_t, frame


def _normalize_oem_acceleration(acceleration, acceleration_t, elapsed):
    if acceleration is None:
        if acceleration_t is not None:
            raise ValueError("acceleration_t requires acceleration.")
        return None, None
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.shape != (elapsed.size, 3):
        raise ValueError("acceleration must have shape (N, 3) matching the state epochs.")
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("acceleration must contain only finite values.")
    if acceleration_t is None:
        acceleration_t = elapsed.copy()
    else:
        acceleration_t = np.asarray(acceleration_t, dtype=float)
        if acceleration_t.shape != (elapsed.size,):
            raise ValueError("acceleration_t must have one epoch per acceleration vector.")
        if not np.all(np.isfinite(acceleration_t)) or (
            acceleration_t.size > 1 and not np.all(np.diff(acceleration_t) > 0.0)
        ):
            raise ValueError("acceleration_t must be finite and strictly increasing.")
        if not np.allclose(acceleration_t, elapsed, rtol=0.0, atol=1.0e-12):
            raise ValueError("acceleration_t must match the state epochs.")
    return acceleration, acceleration_t


def _epoch_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    else:
        raise TypeError("epoch must be an ISO-8601 string or timezone-aware datetime.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("epoch must include a timezone, such as the UTC suffix 'Z'.")
    return parsed.astimezone(timezone.utc)


def _format_epoch(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _case_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("case_name must contain only letters, digits, '_', '.', or '-'.")
    return value


def _number(value: float, precision: int) -> str:
    return f"{float(value):.{precision}e}"


def _oem_text(
    case_name,
    epoch,
    elapsed,
    positions,
    velocities,
    *,
    center_name,
    reference_frame,
    precision,
    covariance=None,
    covariance_t=None,
    covariance_reference_frame=None,
    acceleration=None,
) -> str:
    start = _format_oem_epoch(epoch)
    stop = _format_oem_epoch(epoch + timedelta(seconds=float(elapsed[-1])))
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {start}",
        "ORIGINATOR = SSATK",
        "META_START",
        (
            "COMMENT SSATK reference ephemeris; positions in km, velocities in km/s, "
            "and accelerations in km/s^2."
            if acceleration is not None
            else "COMMENT SSATK reference ephemeris; positions in km and velocities in km/s."
        ),
        f"OBJECT_NAME = {case_name}",
        "OBJECT_ID = UNKNOWN",
        f"CENTER_NAME = {center_name}",
        f"REF_FRAME = {reference_frame}",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {start}",
        f"STOP_TIME = {stop}",
        f"USEABLE_START_TIME = {start}",
        f"USEABLE_STOP_TIME = {stop}",
        "META_STOP",
    ]
    for index, (offset, position, velocity) in enumerate(zip(elapsed, positions, velocities)):
        sample_epoch = _format_oem_epoch(epoch + timedelta(seconds=float(offset)))
        state = [_number(value * 1.0e-3, precision) for value in (*position, *velocity)]
        if acceleration is not None:
            state.extend(_number(value * 1.0e-3, precision) for value in acceleration[index])
        lines.append(" ".join([sample_epoch, *state]))
    if covariance is not None:
        for offset, matrix in zip(covariance_t, covariance):
            sample_epoch = _format_oem_epoch(epoch + timedelta(seconds=float(offset)))
            lines.extend((
                "COVARIANCE_START",
                f"EPOCH = {sample_epoch}",
                f"COV_REF_FRAME = {covariance_reference_frame}",
            ))
            matrix_km = matrix * 1.0e-6
            for row in range(6):
                lines.append(" ".join(_number(matrix_km[row, column], precision) for column in range(row + 1)))
            lines.append("COVARIANCE_STOP")
    lines.append("")
    return "\n".join(lines)


def _format_oem_epoch(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "")
