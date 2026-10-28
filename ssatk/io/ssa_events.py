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
                row_count = len(times)
                if times.ndim != 1:
                    raise ValueError("times must be one-dimensional")
                if valid.shape != (row_count,) or trigger.shape != (row_count,):
                    raise ValueError("valid and trigger must match batch times")
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


__all__ = ["write_event_summary", "write_event_metrics"]
