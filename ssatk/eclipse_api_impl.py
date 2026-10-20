"""Public API for SSATK eclipse events and products.

Events discovered by :mod:`ssatk.compute.eclipse_event_search` use the same
``ReferenceEvent`` record as the two bundled reference eclipses, so every
product renders either.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import json

import numpy as np

from ssatk.compute.eclipse_raytrace import bundle_penetrations, tangent_residuals, trace_event_rays
from ssatk.compute.eclipse_reference_events import (
    LUNAR_2025, SOLAR_2024, ReferenceDefinition, ReferenceEvent, jd_to_datetime,
)
from ssatk.compute.eclipse_state import build_event as _build_event, event_moon_body_to_gcrf
from ssatk.coordinates.eclipse_observer_geometry import ObserverConfig, observer_state, refine_solar_contacts

SCHEMA_ID = "ssatk.eclipse.event/1"
ProductName = Literal["cinematic", "plotly", "scientific-suite", "observer-suite"]


@dataclass(frozen=True)
class EventRequest:
    """One bundled eclipse to build.

    ``sample_count`` is the minimum number of states; exact contact states
    are always inserted, so the event may contain more.
    """

    kind: str = "solar"
    backend: str = "ssapy"
    sample_count: int = 121
    solar_scope: str = "global"
    observer: ObserverConfig | None = None

    def __post_init__(self):
        if self.solar_scope not in {"global", "local"}:
            raise ValueError("solar_scope must be 'global' (P1-P4) or 'local' (observer C1-C4)")


@dataclass(frozen=True)
class RenderRequest:
    product: ProductName | str
    output: str | Path
    event: EventRequest | ReferenceEvent = field(default_factory=EventRequest)
    quality: str = "ultra"
    animate: bool = False
    playback_seconds: float | None = None
    photometry: str = "quadratic-visible"
    timezone_name: str = "UTC"


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "metrics": _json_safe(self.metrics),
            "warnings": list(self.warnings),
        }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    return value


def build_event(request: EventRequest = EventRequest()) -> ReferenceEvent:
    """Build one event from ``request``."""
    event = _build_event(
        request.kind,
        backend=request.backend,
        n_frames=int(request.sample_count),
        solar_scope=request.solar_scope,
        observer=request.observer,
    )
    metadata = dict(event.metadata)
    metadata["requested_minimum_state_count"] = int(request.sample_count)
    if request.observer is not None:
        metadata["observer_model"] = request.observer.to_dict()
    return replace(event, metadata=metadata)


def event_to_dict(event: ReferenceEvent) -> dict[str, Any]:
    return {
        "$schema": SCHEMA_ID,
        "definition": {
            "key": event.definition.key,
            "mode": event.definition.mode,
            "title": event.definition.title,
            "source_label": event.definition.source_label,
            "contacts_jd_utc": {k: float(v) for k, v in event.definition.contacts_jd.items()},
            "greatest_jd_utc": float(event.definition.greatest_jd),
            "notes": list(event.definition.notes),
        },
        "jd_utc": np.asarray(event.jd, dtype=float).tolist(),
        "moon_km": np.asarray(event.moon_km, dtype=float).tolist(),
        "sun_km": np.asarray(event.sun_km, dtype=float).tolist(),
        "center_visibility": np.asarray(event.center_visibility, dtype=float).tolist(),
        "separation_deg": np.asarray(event.separation_deg, dtype=float).tolist(),
        "frame": event.frame,
        "backend": event.backend,
        "metadata": _json_safe(dict(event.metadata)),
    }


def event_from_dict(payload: dict[str, Any]) -> ReferenceEvent:
    if payload.get("$schema") != SCHEMA_ID:
        raise ValueError(f"unsupported event schema {payload.get('$schema')!r}")
    record = dict(payload["definition"])
    builtin = {SOLAR_2024.key: SOLAR_2024, LUNAR_2025.key: LUNAR_2025}
    if record["key"] in builtin:
        definition = builtin[record["key"]]
    else:
        contacts_jd = {str(k): float(v) for k, v in dict(record["contacts_jd_utc"]).items()}
        greatest_jd = float(record["greatest_jd_utc"])
        contacts_jd.setdefault("MAX", greatest_jd)
        definition = ReferenceDefinition(
            key=str(record["key"]),
            mode=str(record["mode"]),
            title=str(record.get("title", record["key"])),
            source_label=str(record.get("source_label", "serialized event")),
            contacts_utc={k: jd_to_datetime(v) for k, v in contacts_jd.items()},
            greatest_utc=jd_to_datetime(greatest_jd),
            notes=tuple(str(v) for v in record.get("notes", ())),
        )
    jd = np.asarray(payload["jd_utc"], dtype=float)
    moon = np.asarray(payload["moon_km"], dtype=float)
    sun = np.asarray(payload["sun_km"], dtype=float)
    if moon.shape != (len(jd), 3) or sun.shape != (len(jd), 3):
        raise ValueError("serialized body states must have shape (N, 3)")
    return ReferenceEvent(
        definition=definition,
        jd=jd,
        moon_km=moon,
        sun_km=sun,
        frame=str(payload["frame"]),
        backend=str(payload["backend"]),
        center_visibility=np.asarray(payload["center_visibility"], dtype=float),
        separation_deg=np.asarray(payload["separation_deg"], dtype=float),
        metadata=dict(payload.get("metadata", {})),
    )


def export_event_state(event: ReferenceEvent, output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event_to_dict(event), indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def load_event_state(path: str | Path) -> ReferenceEvent:
    return event_from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def validate_event(
    event: ReferenceEvent,
    *,
    observer: ObserverConfig | None = None,
    ray_azimuth: int = 24,
) -> ValidationResult:
    """Consistency checks on an event: timeline, states, lunar attitude, and ray paths."""
    jd = np.asarray(event.jd, dtype=float)
    moon = np.asarray(event.moon_km, dtype=float)
    sun = np.asarray(event.sun_km, dtype=float)
    checks = {
        "nonempty_timeline": len(jd) > 0,
        "strictly_increasing_time": bool(len(jd) == 1 or np.all(np.diff(jd) > 0.0)),
        "finite_body_states": bool(np.all(np.isfinite(moon)) and np.all(np.isfinite(sun))),
        "body_state_shapes": moon.shape == (len(jd), 3) and sun.shape == (len(jd), 3),
        "visibility_bounds": bool(np.all((event.center_visibility >= 0.0) & (event.center_visibility <= 1.0))),
    }
    attitude = np.asarray(event_moon_body_to_gcrf(event, float(event.greatest_jd)), dtype=float)
    orth_error = float(np.max(np.abs(attitude.T @ attitude - np.eye(3))))
    determinant = float(np.linalg.det(attitude))
    checks["proper_lunar_attitude"] = orth_error < 1e-10 and determinant > 0.999999999

    bundle = trace_event_rays(event, float(event.greatest_jd), n_azimuth=max(8, int(ray_azimuth)))
    penetrations = bundle_penetrations(bundle)
    checks["no_earth_ray_penetration"] = penetrations["earth"] == 0
    checks["no_moon_ray_penetration"] = penetrations["moon"] == 0
    metrics: dict[str, Any] = {
        "event_key": event.definition.key,
        "backend": event.metadata.get("state_source", event.backend),
        "frame_backend": event.metadata.get("frame_backend", event.frame),
        "moon_orientation_backend": event.metadata.get("moon_orientation_backend", "not recorded"),
        "state_count": int(len(jd)),
        "moon_attitude_orthogonality_error": orth_error,
        "moon_attitude_determinant": determinant,
        "ray_penetrations": penetrations,
        "tangent_residuals": tangent_residuals(bundle),
    }
    warnings: list[str] = []
    if observer is not None and event.mode == "solar":
        metrics["observer_at_greatest"] = observer_state(event, event.greatest_jd, observer).to_dict()
        try:
            contacts = refine_solar_contacts(event, observer)
        except (RuntimeError, ValueError) as exc:
            checks["ordered_local_contacts"] = False
            warnings.append(f"local contact refinement failed: {type(exc).__name__}: {exc}")
        else:
            metrics["refined_contacts_jd_utc"] = contacts
            order = ("C1", "C2", "MAX", "C3", "C4") if {"C2", "C3"}.issubset(contacts) else ("C1", "MAX", "C4")
            metrics["observer_eclipse_class"] = "total-or-annular" if len(order) == 5 else "partial"
            checks["ordered_local_contacts"] = all(contacts[a] < contacts[b] for a, b in zip(order, order[1:]))
    return ValidationResult(passed=all(checks.values()), checks=checks, metrics=metrics, warnings=tuple(warnings))


def render_product(request: RenderRequest) -> str | dict[str, object]:
    """Build the requested event once and render one product from it."""
    product = str(request.product).strip().lower()
    event = request.event if isinstance(request.event, ReferenceEvent) else build_event(request.event)
    backend = str(event.metadata.get("state_source", event.backend))
    solar_scope = str(event.metadata.get("solar_scope", "global"))
    observer = ObserverConfig.from_value(event.metadata.get("observer_model"))

    if product == "cinematic":
        from ssatk.plots.eclipse_cinematic_light_3d import generate_cinematic_view
        return generate_cinematic_view(
            event, request.output, animate=request.animate, quality=request.quality,
            n_frames=len(event.jd), backend=backend, solar_scope=solar_scope,
            playback_seconds=request.playback_seconds, observer_config=observer,
            photometry=request.photometry,
        )
    if product == "plotly":
        from ssatk.plots.eclipse_ultra_public_3d import generate_public_ultra_view
        return generate_public_ultra_view(
            event, request.output, quality=request.quality, backend=backend,
            solar_scope=solar_scope, observer=observer,
        )
    if product == "scientific-suite":
        from ssatk.plots.eclipse_scientific_suite import generate_selected_scientific_suite
        return generate_selected_scientific_suite(
            event, request.output, animate=request.animate, quality=request.quality,
            playback_seconds=request.playback_seconds, photometry=request.photometry,
        )
    if product == "observer-suite":
        if event.mode != "solar":
            raise ValueError("observer-suite supports solar eclipses")
        if observer is None:
            raise ValueError("observer-suite requires EventRequest.observer")
        from ssatk.plots.eclipse_observer_products import generate_solar_observer_products
        return generate_solar_observer_products(
            event, observer, request.output, timezone_name=request.timezone_name,
            include_interactive=request.animate, quality=request.quality,
        )
    raise ValueError("product must be 'cinematic', 'plotly', 'scientific-suite', or 'observer-suite'")
