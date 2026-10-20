"""Canonical public API for SSAPy-Toolkit eclipse products.

V22.2 keeps this module the only supported public entry point.  Arbitrary
events discovered by :mod:`event_search` use the same immutable event schema as
the two bundled reference eclipses.  Legacy renderers are quarantined under
``ssapy_toolkit.eclipse_legacy`` and are no longer resolved from the package root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json

import numpy as np

from ssapy_toolkit.io.eclipse_asset_resolver import audit_assets
from ssapy_toolkit.compute.eclipse_state import build_event as _build_event, event_moon_body_to_gcrf
from ssapy_toolkit.coordinates.eclipse_lunar_geometry import LunarLimbProfile
from ssapy_toolkit.coordinates.eclipse_observer_geometry import ObserverConfig, observer_state, refine_solar_contacts
from ssapy_toolkit.compute.eclipse_reference_events import (
    LUNAR_2025, SOLAR_2024, ReferenceDefinition, ReferenceEvent, jd_to_datetime
)
from ssapy_toolkit.compute.eclipse_raytrace import bundle_penetrations, tangent_residuals, trace_event_rays

SCHEMA_ID = "ssapy-toolkit.eclipse.event/2.2"
BackendName = Literal["reference", "auto", "ssapy-core", "ssapy", "swisseph"]
ProductName = Literal["cinematic", "plotly", "scientific-suite", "observer-suite"]


@dataclass(frozen=True)
class EventRequest:
    """Canonical event request.

    ``sample_count`` is a backward-compatible name for the *minimum* number
    of solver states requested. Exact contact states are always inserted, so
    the resolved event may contain more states. GUI code should display both
    ``requested_minimum_state_count`` and ``resolved_state_count``.
    """

    kind: str = "solar"
    backend: BackendName | str = "auto"
    sample_count: int = 121
    solar_scope: str = "global"
    observer: ObserverConfig | None = None
    lunar_limb_profile: LunarLimbProfile | None = None

    @property
    def minimum_sample_count(self) -> int:
        return int(self.sample_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": "ssapy-toolkit.eclipse.request/2.2",
            "kind": str(self.kind),
            "backend": str(self.backend),
            "sample_count": int(self.sample_count),
            "minimum_sample_count": int(self.minimum_sample_count),
            "sample_count_semantics": "minimum; exact contacts may increase resolved_state_count",
            "solar_scope": str(self.solar_scope),
            "observer": None if self.observer is None else self.observer.to_dict(),
            "lunar_limb_profile": (
                None if self.lunar_limb_profile is None else self.lunar_limb_profile.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventRequest":
        observer = ObserverConfig.from_value(payload.get("observer"))
        limb_payload = payload.get("lunar_limb_profile")
        limb = None
        if isinstance(limb_payload, dict) and limb_payload.get("position_angle_deg") is not None:
            radius_values = limb_payload.get("radius_km", limb_payload.get("radius_km_values"))
            if radius_values is None:
                raise ValueError("serialized lunar limb requires radius_km")
            limb = LunarLimbProfile(
                np.asarray(limb_payload["position_angle_deg"], dtype=float),
                np.asarray(radius_values, dtype=float),
                source=str(limb_payload.get("source", "serialized lunar limb")),
            )
        count = payload.get("minimum_sample_count", payload.get("sample_count", 121))
        return cls(
            kind=str(payload.get("kind", "solar")),
            backend=str(payload.get("backend", "auto")),
            sample_count=int(count),
            solar_scope=str(payload.get("solar_scope", "global")),
            observer=observer,
            lunar_limb_profile=limb,
        )


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

    def to_dict(self) -> dict[str, Any]:
        event_payload = (
            {"type": "event", "value": event_to_dict(self.event)}
            if isinstance(self.event, ReferenceEvent)
            else {"type": "request", "value": self.event.to_dict()}
        )
        return {
            "$schema": "ssapy-toolkit.eclipse.render-request/2.2",
            "product": str(self.product),
            "output": str(self.output),
            "event": event_payload,
            "quality": str(self.quality),
            "animate": bool(self.animate),
            "playback_seconds": self.playback_seconds,
            "photometry": str(self.photometry),
            "timezone_name": str(self.timezone_name),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RenderRequest":
        record = dict(payload.get("event", {}))
        value = dict(record.get("value", {}))
        event = event_from_dict(value) if record.get("type") == "event" else EventRequest.from_dict(value)
        return cls(
            product=str(payload["product"]),
            output=payload["output"],
            event=event,
            quality=str(payload.get("quality", "ultra")),
            animate=bool(payload.get("animate", False)),
            playback_seconds=payload.get("playback_seconds"),
            photometry=str(payload.get("photometry", "quadratic-visible")),
            timezone_name=str(payload.get("timezone_name", "UTC")),
        )


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ssapy-toolkit.eclipse.validation/2.2",
            "passed": self.passed,
            "checks": self.checks,
            "metrics": _json_safe(self.metrics),
            "warnings": list(self.warnings),
        }


def _definition_for_key(key: str):
    normalized = str(key).lower()
    if normalized in {"solar", "solar_2024", SOLAR_2024.key}:
        return SOLAR_2024
    if normalized in {"lunar", "lunar_2025", LUNAR_2025.key}:
        return LUNAR_2025
    raise ValueError(f"unsupported built-in eclipse definition {key!r}")


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


def build_event(request: EventRequest | str = EventRequest(), **overrides) -> ReferenceEvent:
    """Build one immutable event through the canonical request schema."""
    if isinstance(request, EventRequest):
        options = {
            "kind": request.kind,
            "backend": request.backend,
            "sample_count": request.sample_count,
            "solar_scope": request.solar_scope,
            "observer": request.observer,
            "lunar_limb_profile": request.lunar_limb_profile,
        }
        options.update(overrides)
        observer = options.pop("observer", None)
        limb = options.pop("lunar_limb_profile", None)
        event = _build_event(
            options.pop("kind"),
            backend=options.pop("backend"),
            n_frames=int(options.pop("sample_count")),
            solar_scope=options.pop("solar_scope"),
            observer=observer,
            **options,
        )
    else:
        observer = overrides.pop("observer", None)
        limb = overrides.pop("lunar_limb_profile", None)
        if "sample_count" in overrides and "n_frames" not in overrides:
            overrides["n_frames"] = int(overrides.pop("sample_count"))
        event = _build_event(request, observer=observer, **overrides)
    metadata = dict(event.metadata)
    metadata["canonical_schema"] = SCHEMA_ID
    metadata["asset_provenance"] = audit_assets(policy="data-first", portable=True)
    requested_minimum = int(
        request.minimum_sample_count if isinstance(request, EventRequest)
        else overrides.get("n_frames", len(event.jd))
    )
    metadata["requested_minimum_state_count"] = requested_minimum
    metadata["resolved_state_count"] = int(len(event.jd))
    metadata["sample_count_semantics"] = "minimum; exact contact states are inserted"
    if observer is not None:
        metadata["observer_config"] = observer.to_dict() if hasattr(observer, "to_dict") else _json_safe(observer)
    if limb is not None:
        metadata["lunar_limb_profile"] = limb.to_dict()
    from dataclasses import replace
    return replace(event, metadata=metadata)


def event_to_dict(event: ReferenceEvent) -> dict[str, Any]:
    return {
        "$schema": SCHEMA_ID,
        "sampling": {
            "requested_minimum_state_count": event.metadata.get("requested_minimum_state_count"),
            "resolved_state_count": int(len(event.jd)),
            "sample_count_semantics": event.metadata.get(
                "sample_count_semantics", "minimum; exact contact states are inserted"
            ),
        },
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
        "moon_gcrf_km": np.asarray(event.moon_km, dtype=float).tolist(),
        "sun_gcrf_km": np.asarray(event.sun_km, dtype=float).tolist(),
        "center_visibility": np.asarray(event.center_visibility, dtype=float).tolist(),
        "separation_deg": np.asarray(event.separation_deg, dtype=float).tolist(),
        "frame": event.frame,
        "backend": event.backend,
        "metadata": _json_safe(dict(event.metadata)),
    }


def event_from_dict(payload: dict[str, Any]) -> ReferenceEvent:
    schema = payload.get("$schema")
    if schema not in {SCHEMA_ID, "ssapy-toolkit.eclipse.event/2.0.1", "ssapy-toolkit.eclipse.event/2.0", "ssapy-toolkit.eclipse.event/1.9", "ssapy-toolkit.eclipse.event/1.8", "ssapy-toolkit.eclipse.event/1.7"}:
        raise ValueError(f"unsupported event schema {schema!r}")
    definition_payload = dict(payload["definition"])
    try:
        definition = _definition_for_key(definition_payload["key"])
    except ValueError:
        contacts_jd = {
            str(k): float(v) for k, v in dict(definition_payload.get("contacts_jd_utc", {})).items()
        }
        if not contacts_jd:
            metadata_record = dict(payload.get("metadata", {})).get("event_search_record", {})
            contacts_jd = {
                str(k): float(v) for k, v in dict(metadata_record.get("contacts_jd_utc", {})).items()
            }
        greatest_jd = float(definition_payload.get(
            "greatest_jd_utc", contacts_jd.get("MAX", np.asarray(payload["jd_utc"], dtype=float)[0])
        ))
        if "MAX" not in contacts_jd:
            contacts_jd["MAX"] = greatest_jd
        definition = ReferenceDefinition(
            key=str(definition_payload["key"]),
            mode=str(definition_payload["mode"]),
            title=str(definition_payload.get("title", definition_payload["key"])),
            source_label=str(definition_payload.get("source_label", "serialized dynamic event")),
            contacts_utc={k: jd_to_datetime(v) for k, v in contacts_jd.items()},
            greatest_utc=jd_to_datetime(greatest_jd),
            notes=tuple(str(v) for v in definition_payload.get("notes", ())),
        )
    jd = np.asarray(payload["jd_utc"], dtype=float)
    moon = np.asarray(payload["moon_gcrf_km"], dtype=float)
    sun = np.asarray(payload["sun_gcrf_km"], dtype=float)
    visible = np.asarray(payload["center_visibility"], dtype=float)
    separation = np.asarray(payload["separation_deg"], dtype=float)
    if moon.shape != (len(jd), 3) or sun.shape != (len(jd), 3):
        raise ValueError("serialized body states must have shape (N, 3)")
    return ReferenceEvent(
        definition=definition,
        jd=jd,
        moon_km=moon,
        sun_km=sun,
        frame=str(payload["frame"]),
        backend=str(payload["backend"]),
        center_visibility=visible,
        separation_deg=separation,
        metadata={
            **dict(payload.get("metadata", {})),
            **({
                "requested_minimum_state_count": dict(payload.get("sampling", {})).get("requested_minimum_state_count"),
                "resolved_state_count": int(len(jd)),
                "sample_count_semantics": dict(payload.get("sampling", {})).get(
                    "sample_count_semantics", "minimum; exact contact states are inserted"
                ),
            } if payload.get("sampling") else {}),
        },
    )


def export_event_state(event: ReferenceEvent, output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event_to_dict(event), indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def load_event_state(path: str | Path) -> ReferenceEvent:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return event_from_dict(payload)


def validate_event(
    event: ReferenceEvent,
    *,
    observer: ObserverConfig | None = None,
    lunar_limb_profile: LunarLimbProfile | None = None,
    ray_azimuth: int = 24,
) -> ValidationResult:
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
    residuals = tangent_residuals(bundle)
    checks["no_earth_ray_penetration"] = penetrations["earth"] == 0
    checks["no_moon_ray_penetration"] = penetrations["moon"] == 0

    metrics: dict[str, Any] = {
        "event_key": event.definition.key,
        "backend": event.metadata.get("state_source", event.backend),
        "frame_backend": event.metadata.get("frame_backend", event.frame),
        "moon_orientation_backend": event.metadata.get("moon_orientation_backend", "not recorded"),
        "requested_minimum_state_count": event.metadata.get("requested_minimum_state_count"),
        "resolved_state_count": int(len(jd)),
        "sample_count_semantics": event.metadata.get(
            "sample_count_semantics", "minimum; exact contacts may increase resolved state count"
        ),
        "sample_count": int(len(jd)),
        "moon_attitude_orthogonality_error": orth_error,
        "moon_attitude_determinant": determinant,
        "ray_penetrations": penetrations,
        "tangent_residuals": residuals,
    }
    warnings: list[str] = []

    if observer is not None and event.mode == "solar":
        obs = observer_state(event, event.greatest_jd, observer, limb_profile=lunar_limb_profile)
        metrics["observer_at_greatest"] = obs.to_dict()
        try:
            contacts = refine_solar_contacts(event, observer, limb_profile=lunar_limb_profile)
            metrics["refined_contacts_jd_utc"] = contacts
            metrics["observer_eclipse_class"] = (
                "total-or-annular" if {"C2", "C3"}.issubset(contacts) else "partial"
            )
            if {"C2", "C3"}.issubset(contacts):
                checks["ordered_local_contacts"] = (
                    contacts["C1"] < contacts["C2"] < contacts["MAX"]
                    < contacts["C3"] < contacts["C4"]
                )
            else:
                checks["ordered_local_contacts"] = contacts["C1"] < contacts["MAX"] < contacts["C4"]
        except Exception as exc:
            checks["ordered_local_contacts"] = False
            warnings.append(f"local contact refinement failed: {type(exc).__name__}: {exc}")

    return ValidationResult(
        passed=all(checks.values()), checks=checks, metrics=metrics, warnings=tuple(warnings)
    )


def render_product(request: RenderRequest) -> str | dict[str, object]:
    """Render one product from one immutable selected event.

    V22.2 deliberately builds the event once and passes that exact object to
    every renderer.  In particular, ``scientific-suite`` no longer discards a
    discovered/serialized event and substitutes the two bundled references.
    """
    product = str(request.product).strip().lower()
    event = request.event if isinstance(request.event, ReferenceEvent) else build_event(request.event)
    backend = str(event.metadata.get("state_source", event.backend))
    solar_scope = str(event.metadata.get("solar_scope", "global"))
    observer = ObserverConfig.from_value(event.metadata.get("observer_model"))

    if product == "cinematic":
        from ssapy_toolkit.plots.eclipse_cinematic_light_3d import generate_cinematic_view
        return generate_cinematic_view(
            event, request.output,
            animate=request.animate,
            quality=request.quality,
            n_frames=len(event.jd),
            backend=backend,
            solar_scope=solar_scope,
            playback_seconds=request.playback_seconds,
            observer_config=observer,
            photometry=request.photometry,
        )
    if product == "plotly":
        from ssapy_toolkit.plots.eclipse_ultra_public_3d import generate_public_ultra_view
        return generate_public_ultra_view(
            event, request.output, quality=request.quality, backend=backend,
            solar_scope=solar_scope, observer=observer,
        )
    if product == "scientific-suite":
        from ssapy_toolkit.plots.eclipse_scientific_suite import generate_selected_scientific_suite
        return generate_selected_scientific_suite(
            event, request.output, animate=request.animate,
            quality=request.quality, playback_seconds=request.playback_seconds,
            photometry=request.photometry,
        )
    if product == "observer-suite":
        if event.mode != "solar":
            raise ValueError("observer-suite currently supports solar eclipses")
        if observer is None:
            raise ValueError("observer-suite requires EventRequest.observer")
        from ssapy_toolkit.plots.eclipse_observer_products import generate_solar_observer_products
        return generate_solar_observer_products(
            event, observer, request.output, timezone_name=request.timezone_name,
            include_interactive=request.animate, quality=request.quality,
        )
    raise ValueError("product must be 'cinematic', 'plotly', 'scientific-suite', or 'observer-suite'")
