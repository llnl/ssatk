"""Cross-event eclipse discovery validation for V19.3.

The validator compares a selected discovery backend against an independent
NASA/GSFC catalog corpus.  The corpus is never consulted by the solver while
searching; it is read only after discovery, which prevents circular
validation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence
import json
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_event_search import DiscoveredEclipse, discover_eclipses
from ssapy_toolkit.compute.eclipse_reference_corpus import ReferenceEclipseRecord, index_by_key, records


@dataclass(frozen=True)
class ValidationThresholds:
    greatest_time_s: float = 15.0
    magnitude: float = 0.0012
    lunar_duration_min: float = 0.35
    missing_events: int = 0
    unexpected_events: int = 0
    type_mismatches: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventResidual:
    key: str
    mode: str
    reference_type: str
    predicted_type: str | None
    greatest_residual_s: float | None
    magnitude_metric: str
    reference_magnitude: float | None
    predicted_magnitude: float | None
    magnitude_residual: float | None
    penumbral_duration_residual_min: float | None = None
    partial_duration_residual_min: float | None = None
    total_duration_residual_min: float | None = None
    type_match: bool = False
    present: bool = False
    passes: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _lunar_metric(discovered: DiscoveredEclipse, reference: ReferenceEclipseRecord) -> tuple[str, float | None, float | None]:
    metadata = dict(discovered.metadata)
    if reference.eclipse_type == "penumbral":
        return (
            "penumbral_magnitude",
            reference.penumbral_magnitude,
            _finite_optional(metadata.get("penumbral_magnitude")),
        )
    return (
        "umbral_magnitude",
        reference.umbral_magnitude,
        _finite_optional(metadata.get("umbral_magnitude", discovered.magnitude)),
    )


def _finite_optional(value) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _duration_minutes(contacts: Mapping[str, float], start: str, end: str) -> float | None:
    if start not in contacts or end not in contacts:
        return None
    return (float(contacts[end]) - float(contacts[start])) * 1440.0


def compare_event(discovered: DiscoveredEclipse | None, reference: ReferenceEclipseRecord,
                  thresholds: ValidationThresholds) -> EventResidual:
    if discovered is None:
        return EventResidual(
            key=reference.key, mode=reference.mode, reference_type=reference.eclipse_type,
            predicted_type=None, greatest_residual_s=None,
            magnitude_metric="solar_magnitude" if reference.mode == "solar" else "catalog_magnitude",
            reference_magnitude=reference.magnitude, predicted_magnitude=None,
            magnitude_residual=None, type_match=False, present=False, passes=False,
        )
    time_residual = (float(discovered.greatest_jd_utc) - reference.greatest_jd_utc) * 86400.0
    if reference.mode == "solar":
        metric = "solar_magnitude"
        ref_mag = float(reference.magnitude)
        pred_mag = _finite_optional(discovered.magnitude)
    else:
        metric, ref_mag, pred_mag = _lunar_metric(discovered, reference)
    mag_residual = None if ref_mag is None or pred_mag is None else pred_mag - ref_mag

    pen_res = par_res = tot_res = None
    duration_ok = True
    if reference.mode == "lunar":
        contacts = discovered.contacts_jd_utc
        predicted = _duration_minutes(contacts, "P1", "P4")
        if predicted is not None and reference.penumbral_duration_min is not None:
            pen_res = predicted - reference.penumbral_duration_min
            duration_ok &= abs(pen_res) <= thresholds.lunar_duration_min
        predicted = _duration_minutes(contacts, "U1", "U4")
        if predicted is not None and reference.partial_duration_min is not None:
            par_res = predicted - reference.partial_duration_min
            duration_ok &= abs(par_res) <= thresholds.lunar_duration_min
        predicted = _duration_minutes(contacts, "U2", "U3")
        if predicted is not None and reference.total_duration_min is not None:
            tot_res = predicted - reference.total_duration_min
            duration_ok &= abs(tot_res) <= thresholds.lunar_duration_min

    type_match = discovered.eclipse_type == reference.eclipse_type
    pass_mag = mag_residual is not None and abs(mag_residual) <= thresholds.magnitude
    passes = (
        type_match
        and abs(time_residual) <= thresholds.greatest_time_s
        and pass_mag
        and duration_ok
    )
    return EventResidual(
        key=reference.key, mode=reference.mode, reference_type=reference.eclipse_type,
        predicted_type=discovered.eclipse_type, greatest_residual_s=float(time_residual),
        magnitude_metric=metric, reference_magnitude=ref_mag,
        predicted_magnitude=pred_mag, magnitude_residual=mag_residual,
        penumbral_duration_residual_min=pen_res,
        partial_duration_residual_min=par_res,
        total_duration_residual_min=tot_res,
        type_match=type_match, present=True, passes=bool(passes),
    )


def validate_reference_corpus(
    *,
    start: str = "2021-01-01",
    end: str = "2031-01-01",
    mode: str = "all",
    backend: str = "swisseph",
    thresholds: ValidationThresholds | None = None,
) -> dict[str, object]:
    """Validate an independent discovery backend against the NASA corpus."""
    limits = thresholds or ValidationThresholds()
    reference = records(mode)
    predicted = discover_eclipses(start, end, kind=mode, backend=backend)
    predicted_by_key = {item.key: item for item in predicted}
    reference_by_key = {item.key: item for item in reference}
    residuals = [compare_event(predicted_by_key.get(item.key), item, limits) for item in reference]
    unexpected = sorted(set(predicted_by_key) - set(reference_by_key))
    missing = [item.key for item in residuals if not item.present]
    type_mismatches = [item.key for item in residuals if item.present and not item.type_match]

    times = np.asarray([item.greatest_residual_s for item in residuals if item.greatest_residual_s is not None], dtype=float)
    mags = np.asarray([item.magnitude_residual for item in residuals if item.magnitude_residual is not None], dtype=float)
    durations = np.asarray([
        value
        for item in residuals
        for value in (
            item.penumbral_duration_residual_min,
            item.partial_duration_residual_min,
            item.total_duration_residual_min,
        )
        if value is not None
    ], dtype=float)

    summary = {
        "reference_count": len(reference),
        "predicted_count": len(predicted),
        "matched_count": sum(item.present for item in residuals),
        "passing_event_count": sum(item.passes for item in residuals),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "type_mismatch_count": len(type_mismatches),
        "greatest_time_abs_max_s": _safe_max_abs(times),
        "greatest_time_rms_s": _safe_rms(times),
        "magnitude_abs_max": _safe_max_abs(mags),
        "magnitude_rms": _safe_rms(mags),
        "lunar_duration_abs_max_min": _safe_max_abs(durations),
        "lunar_duration_rms_min": _safe_rms(durations),
    }
    release_pass = (
        len(missing) <= limits.missing_events
        and len(unexpected) <= limits.unexpected_events
        and len(type_mismatches) <= limits.type_mismatches
        and all(item.passes for item in residuals)
    )
    return {
        "$schema": "ssapy-toolkit.eclipse.multi-event-validation/1.9.3",
        "title": "V19.3 cross-event eclipse discovery validation",
        "scope": {"start": start, "end": end, "mode": mode},
        "validation_backend": backend,
        "independence_statement": (
            "Reference catalog values are read only after discovery; they do not seed, fit, or alter the selected backend solution."
        ),
        "time_comparison": (
            "NASA dynamical greatest time minus published Delta T compared with the backend UTC/UT greatest instant; residual includes sub-second UTC-UT1 qualification."
        ),
        "thresholds": limits.to_dict(),
        "summary": summary,
        "missing_events": missing,
        "unexpected_events": unexpected,
        "type_mismatches": type_mismatches,
        "events": [item.to_dict() for item in residuals],
        "passes_release_gate": bool(release_pass),
    }


def _safe_max_abs(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(np.abs(values)))


def _safe_rms(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.sqrt(np.mean(values * values)))


def write_validation_report(path: str | Path, **kwargs) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = validate_reference_corpus(**kwargs)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "ValidationThresholds", "EventResidual", "compare_event",
    "validate_reference_corpus", "write_validation_report",
]
