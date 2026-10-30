"""Persistence helpers for sampled SSA event results."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import numpy as np

from ssatk._paths import output_root, safe_relative_parts


_BASE_COLUMNS = (
    "time_gps_s",
    "event_index",
    "object_id_a",
    "object_id_b",
    "valid",
    "trigger",
)
_VECTOR_COMPONENTS = ("r", "t", "n")


def _target_path(path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    parts = safe_relative_parts(path)
    if not parts:
        raise ValueError("relative output path must include a filename")
    return output_root() / Path(*parts)


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    )
    return handle, Path(handle.name)


def _json_id(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(
        "event object IDs must be JSON primitives (None, str, bool, int, "
        "or finite float)"
    )


def _json_ids(ids):
    return [_json_id(value) for value in ids]


def _window_dict(window, object_ids):
    first, second = window.pair
    return {
        "object_ids": [
            _json_id(object_ids[first]),
            _json_id(object_ids[second]),
        ],
        "start": window.start,
        "end": window.end,
        "trigger_start": window.trigger_start,
        "trigger_end": window.trigger_end,
        "trigger_count": window.trigger_count,
    }


def write_event_summary(result, path):
    """Write a JSON summary of a :class:`PairAnalysisResult` atomically."""
    object_ids = _json_ids(result.object_ids)
    windows = [
        _window_dict(window, object_ids) for window in result.windows
    ]
    payload = {
        "schema": "ssatk.ssa.event_summary.v1",
        "metadata": {
            "start_time_gps_s": result.start_time,
            "end_time_gps_s": result.end_time,
            "sample_count": result.sample_count,
        },
        "completeness": bool(result.complete),
        "counts": {
            "objects": len(object_ids),
            "windows": len(windows),
            "trigger_samples": sum(
                window["trigger_count"] for window in windows
            ),
        },
        "object_ids": object_ids,
        "windows": windows,
        "diagnostics": result.diagnostics,
        "provenance": result.provenance,
        "config": asdict(result.config),
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    target = _target_path(path)
    handle, temporary = _atomic_path(target)
    try:
        with handle:
            handle.write(text)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _metric_columns(metrics, row_count):
    columns = []
    for name in sorted(metrics):
        value = np.asarray(metrics[name])
        if value.ndim == 1:
            if len(value) != row_count:
                raise ValueError(f"metric {name!r} has the wrong row count")
            columns.append((name, value))
        elif value.ndim == 2:
            if value.shape[0] != row_count:
                raise ValueError(f"metric {name!r} has the wrong row count")
            if value.shape[1] != 3:
                raise ValueError(
                    f"vector metric {name!r} must have three components"
                )
            for component_index, component in enumerate(_VECTOR_COMPONENTS):
                columns.append(
                    (f"{name}_{component}", value[:, component_index])
                )
        else:
            raise ValueError(f"metric {name!r} must be scalar or vector per time")
    return columns


def read_event_summary(path):
    """Reload window results for ``iter_event_metrics`` with fingerprint checks.

    The source and observer are still required for replay. Loading a summary
    does not authenticate its provenance; replay verifies the saved fingerprints.
    """
    from ssatk.ssa.event_windows import EventWindow
    from ssatk.ssa.workflows import PairAnalysisConfig, PairAnalysisResult

    payload = json.loads(_target_path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "ssatk.ssa.event_summary.v1":
        raise ValueError("Unsupported SSA event summary schema.")
    ids = tuple(_json_ids(payload["object_ids"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Event object IDs must be unique.")
    indices = {value: index for index, value in enumerate(ids)}
    windows = []
    for item in payload["windows"]:
        fields = dict(item)
        pair = fields.pop("object_ids")
        if len(pair) != 2 or any(value not in indices for value in pair):
            raise ValueError("Event window references unknown object IDs.")
        windows.append(EventWindow(pair=tuple(indices[value] for value in pair), **fields))
    metadata = payload["metadata"]
    start, end = metadata["start_time_gps_s"], metadata["end_time_gps_s"]
    count = metadata["sample_count"]
    if not np.all(np.isfinite([start, end])) or start > end:
        raise ValueError("Summary times must be finite and ordered.")
    if type(count) is not int or count < 0 or type(payload["completeness"]) is not bool:
        raise ValueError("Summary sample count and completeness are invalid.")
    for window in windows:
        if not start <= window.start <= window.trigger_start <= window.trigger_end <= window.end <= end:
            raise ValueError("Window times lie outside the summary or padding.")
    expected_counts = {"objects": len(ids), "windows": len(windows),
                       "trigger_samples": sum(window.trigger_count for window in windows)}
    if payload["counts"] != expected_counts:
        raise ValueError("Summary counts disagree with its windows.")
    config = PairAnalysisConfig(**payload["config"])
    provenance = payload["provenance"]
    for key in ("sample_fingerprint_sha256", "declared_times_sha256"):
        value = provenance.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Summary lacks a valid {key} for replay.")
    if provenance.get("config") != asdict(config):
        raise ValueError("Summary configuration differs from its provenance.")
    if type(provenance.get("declared_sample_count")) is not int or provenance["declared_sample_count"] < count:
        raise ValueError("Summary declared sample count is invalid.")
    return PairAnalysisResult(ids, tuple(windows), config, start, end, count,
                              payload["completeness"], payload["diagnostics"], provenance)


def _csv_value(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, Real) and not math.isfinite(value):
        return ""
    return value


def write_event_metrics(batches, path):
    """Stream event metric batches to an atomically replaced CSV file."""
    target = _target_path(path)
    handle, temporary = _atomic_path(target)
    columns = None
    try:
        with handle:
            writer = csv.writer(handle, lineterminator="\n")
            for batch in batches:
                times = np.asarray(batch.times)
                valid = np.asarray(batch.valid)
                trigger = np.asarray(batch.trigger)
                if times.ndim != 1:
                    raise ValueError("times must be one-dimensional")
                if (
                    not np.issubdtype(times.dtype, np.number)
                    or np.issubdtype(times.dtype, np.complexfloating)
                ):
                    raise ValueError("times must be real numeric GPS seconds")
                times = times.astype(float, copy=False)
                if not np.all(np.isfinite(times)):
                    raise ValueError("times must be finite GPS seconds")
                row_count = len(times)
                for name, flags in (("valid", valid), ("trigger", trigger)):
                    if flags.ndim != 1 or flags.shape != (row_count,):
                        raise ValueError(
                            f"{name} must be one-dimensional and match batch times"
                        )
                    if flags.dtype != np.dtype(bool):
                        raise ValueError(f"{name} must contain boolean values")
                ids = _json_ids(batch.object_ids)
                if len(ids) != 2:
                    raise ValueError("event batches must contain two object IDs")
                metric_columns = _metric_columns(batch.metrics, row_count)
                names = tuple(column[0] for column in metric_columns)
                if columns is None:
                    columns = names
                    writer.writerow((*_BASE_COLUMNS, *columns))
                elif columns != names:
                    raise ValueError("event metric batches have different schemas")
                for index in range(row_count):
                    values = [
                        times[index],
                        batch.window_index,
                        ids[0],
                        ids[1],
                        valid[index],
                        trigger[index],
                    ]
                    values.extend(column[1][index] for column in metric_columns)
                    writer.writerow([_csv_value(value) for value in values])
            if columns is None:
                writer.writerow(_BASE_COLUMNS)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = ["write_event_summary", "read_event_summary", "write_event_metrics"]
