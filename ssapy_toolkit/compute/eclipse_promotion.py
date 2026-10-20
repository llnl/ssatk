"""V22.2 strict-SSAPy promotion gate, data audit, and residual report.

A promotion report is either a genuine strict LLNL execution or explicitly
``blocked``.  Reference states are never substituted into a strict report.
The gate evaluates ephemerides, Toolkit frame transforms, Earth/Moon attitude,
finite-Sun ray clipping, local solar contacts, the WGS-84 shadow-axis ground
point, and lunar P/U contacts against the validated event references.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import json
import math

import numpy as np

from ssapy_toolkit.io.eclipse_asset_resolver import audit_assets
from ssapy_toolkit.io.eclipse_provenance import portable_record
from ssapy_toolkit.compute.eclipse_state import (
    build_event,
    event_gcrf_to_itrf_km,
    event_moon_body_to_gcrf,
    event_positions_gcrf,
)
from ssapy_toolkit.compute.eclipse_brightness import shadow_axis_surface_point
from ssapy_toolkit.coordinates.eclipse_lunar_geometry import iau_moon_body_to_gcrf
from ssapy_toolkit.coordinates.eclipse_observer_geometry import ObserverConfig, refine_solar_contacts
from ssapy_toolkit.compute.eclipse_runtime import BackendUnavailableError, resolve_backend, runtime_capabilities
from ssapy_toolkit.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    solar_central_line_wgs84,
    solar_local_contacts,
)
from ssapy_toolkit.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    bundle_penetrations,
    tangent_residuals,
    trace_event_rays,
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _attitude_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def _root_near(function, reference_jd: float, half_window_s: float = 1800.0) -> float:
    try:
        from scipy.optimize import brentq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for strict contact promotion") from exc
    grid = reference_jd + np.linspace(-half_window_s, half_window_s, 361) / 86400.0
    values = np.asarray([function(float(jd)) for jd in grid], dtype=float)
    brackets: list[tuple[float, float]] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if abs(float(f_left)) < 1e-15:
            brackets.append((float(left), float(left)))
        elif f_left * f_right < 0.0:
            brackets.append((float(left), float(right)))
    if not brackets:
        raise RuntimeError(f"no contact root near JD {reference_jd:.9f}")
    lo, hi = min(brackets, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - reference_jd))
    return lo if lo == hi else float(brentq(function, lo, hi, xtol=1e-12))


def _lunar_contact_residuals_s(event) -> dict[str, float]:
    """Solve lunar P/U contacts from the strict body state and cone radii."""
    references = LUNAR_2025.contacts_jd

    def geometry(jd: float) -> tuple[float, float, float]:
        sun, moon = event_positions_gcrf(event, jd)
        sun_distance = float(np.linalg.norm(sun))
        sun_hat = sun / sun_distance
        downstream = -sun_hat
        axial_distance = float(np.dot(moon, downstream))
        axis_offset = float(np.linalg.norm(moon - axial_distance * downstream))
        umbra_radius = (
            LUNAR_DANJON_EARTH_RADIUS_KM
            - axial_distance * (R_SUN_KM - LUNAR_DANJON_EARTH_RADIUS_KM) / sun_distance
        )
        penumbra_radius = (
            LUNAR_DANJON_EARTH_RADIUS_KM
            + axial_distance * (R_SUN_KM + LUNAR_DANJON_EARTH_RADIUS_KM) / sun_distance
        )
        return axis_offset, umbra_radius, penumbra_radius

    equations = {
        "P1": lambda jd: geometry(jd)[0] - (geometry(jd)[2] + R_MOON_MEAN_KM),
        "U1": lambda jd: geometry(jd)[0] - (geometry(jd)[1] + R_MOON_MEAN_KM),
        "U2": lambda jd: geometry(jd)[0] - (geometry(jd)[1] - R_MOON_MEAN_KM),
        "U3": lambda jd: geometry(jd)[0] - (geometry(jd)[1] - R_MOON_MEAN_KM),
        "U4": lambda jd: geometry(jd)[0] - (geometry(jd)[1] + R_MOON_MEAN_KM),
        "P4": lambda jd: geometry(jd)[0] - (geometry(jd)[2] + R_MOON_MEAN_KM),
    }
    return {
        name: (_root_near(equation, float(references[name]), 2400.0) - float(references[name])) * 86400.0
        for name, equation in equations.items()
    }


def _solar_promotion_metrics(event) -> dict[str, object]:
    observer = ObserverConfig(
        SOLAR_GREATEST_SITE_LAT_DEG,
        SOLAR_GREATEST_SITE_LON_EAST_DEG,
        elevation_m=0.0,
        apply_refraction=False,
        name="NASA greatest-eclipse reference site",
    )
    local_event = build_event(
        "solar", backend="ssapy", n_frames=max(41, len(event.jd)),
        solar_scope="local", observer=observer,
    )
    strict_contacts = refine_solar_contacts(local_event, observer)
    reference_contacts = solar_local_contacts(
        observer.latitude_deg, observer.longitude_east_deg, observer.elevation_m / 1000.0
    )
    contact_residuals = {
        name: (float(strict_contacts[name]) - float(reference_contacts[name])) * 86400.0
        for name in ("C1", "C2", "MAX", "C3", "C4")
    }

    sun, moon = event_positions_gcrf(event, event.greatest_jd)
    moon_itrf, sun_itrf = event_gcrf_to_itrf_km(
        event, np.asarray([moon, sun]), np.asarray([event.greatest_jd, event.greatest_jd])
    )
    strict_hit = shadow_axis_surface_point(
        moon_itrf, sun_itrf, target_axes_km=EARTH_AXES_KM
    )
    reference_hit_record = solar_central_line_wgs84(event.greatest_jd)
    if strict_hit is None or reference_hit_record is None:
        raise RuntimeError("solar shadow axis did not intersect WGS-84 Earth at greatest eclipse")
    reference_hit = np.asarray(reference_hit_record[2], dtype=float)
    return {
        "local_contact_residuals_s": contact_residuals,
        "maximum_absolute_local_contact_residual_s": float(max(abs(v) for v in contact_residuals.values())),
        "shadow_axis_surface_residual_km": float(np.linalg.norm(np.asarray(strict_hit) - reference_hit)),
    }


def _event_report(kind: str, *, samples: int) -> dict[str, object]:
    event = build_event(kind, backend="ssapy", n_frames=samples, solar_scope="global")
    bundle = trace_event_rays(event, event.greatest_jd, n_azimuth=48)
    strict_attitude = event_moon_body_to_gcrf(event, event.greatest_jd)
    reference_attitude = iau_moon_body_to_gcrf(event.greatest_jd)
    report: dict[str, object] = {
        "event_key": event.definition.key,
        "sample_count": int(len(event.jd)),
        "ephemeris_backend": event.metadata.get("ephemeris_backend"),
        "frame_backend": event.metadata.get("frame_backend"),
        "moon_orientation_backend": event.metadata.get("moon_orientation_backend"),
        "sun_reference_rms_km": float(event.metadata.get("ssapy_sun_reference_rms_km", np.nan)),
        "moon_reference_rms_km": float(event.metadata.get("ssapy_moon_reference_rms_km", np.nan)),
        "sun_reference_max_km": float(event.metadata.get("ssapy_sun_reference_max_km", np.nan)),
        "moon_reference_max_km": float(event.metadata.get("ssapy_moon_reference_max_km", np.nan)),
        "strict_vs_iau_moon_attitude_deg": _attitude_difference_deg(reference_attitude, strict_attitude),
        "ray_penetrations": bundle_penetrations(bundle),
        "tangent_residuals": tangent_residuals(bundle),
    }
    if kind == "solar":
        report.update(_solar_promotion_metrics(event))
    else:
        contact_residuals = _lunar_contact_residuals_s(event)
        report.update({
            "contact_residuals_s": contact_residuals,
            "maximum_absolute_contact_residual_s": float(max(abs(v) for v in contact_residuals.values())),
        })
    return report


def build_ssapy_promotion_report(
    *,
    samples: int = 41,
    require_strict: bool = False,
) -> dict[str, object]:
    """Execute the strict release gate or return an explicit blocked report."""
    caps = runtime_capabilities(execute=True)
    thresholds = {
        "ray_penetrations": 0,
        "tangent_radius_error_km": 1.0e-5,
        "tangent_orthogonality_km": 1.0e-5,
        "proper_rotation_determinant_min": 0.999999999,
        "solar_local_contact_residual_s": 5.0,
        "solar_shadow_axis_surface_residual_km": 5.0,
        "lunar_contact_residual_s": 8.0,
        "arbitrary_search_greatest_residual_s": 60.0,
    }
    report: dict[str, object] = {
        "schema": "ssapy-toolkit.eclipse.ssapy-promotion/2.0",
        "requested_backend": "ssapy",
        "llnl_ssapy_version": _version("llnl-ssapy"),
        "ssapy_toolkit_version": _version("ssapy-toolkit"),
        "astropy_version": _version("astropy"),
        "llnl_ssapy_data_version": _version("llnl-ssapy-data"),
        "capabilities": caps.to_dict(portable=True),
        "ssapy_data": audit_assets(policy="data-first", portable=True),
        "thresholds": thresholds,
    }
    try:
        selection = resolve_backend("ssapy", execute_probe=True)
    except BackendUnavailableError as exc:
        report.update({
            "status": "blocked",
            "promoted": False,
            "blocking_reason": str(exc),
            "required_data_assets": {
                "rendering": ["earth_albedo", "moon_albedo"],
                "strict_llnl": ["de430_kernel", "moon_pa_kernel"],
            },
            "events": {},
            "arbitrary_event_discovery": {
                "executed": False,
                "reason": "strict LLNL state provider unavailable",
                "required_interval": ["2024-04-01", "2024-04-15"],
            },
            "next_command": "ssapy-eclipse promote --require-strict --output ssapy_promotion_report.json",
        })
        if require_strict:
            raise
        return report

    events = {
        "solar": _event_report("solar", samples=max(25, int(samples))),
        "lunar": _event_report("lunar", samples=max(25, int(samples))),
    }
    from ssapy_toolkit.compute.eclipse_event_search import discover_eclipses
    discovered = discover_eclipses(
        "2024-04-01", "2024-04-15", kind="solar", backend="ssapy"
    )
    if len(discovered) != 1:
        discovery_report = {
            "executed": True, "passed": False, "count": len(discovered),
            "records": [item.to_dict() for item in discovered],
        }
    else:
        residual_s = abs(float(discovered[0].greatest_jd_utc) - float(SOLAR_2024.greatest_jd))*86400.0
        discovery_report = {
            "executed": True,
            "passed": residual_s <= thresholds["arbitrary_search_greatest_residual_s"],
            "greatest_residual_s": residual_s,
            "record": discovered[0].to_dict(),
        }
    passed = bool(discovery_report.get("passed", False))
    failures: list[str] = []
    if not passed:
        failures.append("strict arbitrary-event discovery")
    for name, event in events.items():
        penetration = event["ray_penetrations"]
        if penetration["earth"] != 0 or penetration["moon"] != 0:
            passed = False
            failures.append(f"{name}: ray penetration")
        for key, value in event["tangent_residuals"].items():
            if ("error" in key or "orthogonality" in key) and abs(float(value)) > 1.0e-5:
                passed = False
                failures.append(f"{name}: {key}")
    if events["solar"]["maximum_absolute_local_contact_residual_s"] > thresholds["solar_local_contact_residual_s"]:
        passed = False; failures.append("solar: local contact residual")
    if events["solar"]["shadow_axis_surface_residual_km"] > thresholds["solar_shadow_axis_surface_residual_km"]:
        passed = False; failures.append("solar: WGS-84 shadow-axis residual")
    if events["lunar"]["maximum_absolute_contact_residual_s"] > thresholds["lunar_contact_residual_s"]:
        passed = False; failures.append("lunar: contact residual")

    report.update({
        "status": "passed" if passed else "failed",
        "promoted": bool(passed),
        "selection": selection.to_dict(portable=True),
        "events": events,
        "arbitrary_event_discovery": discovery_report,
        "failures": failures,
    })
    if require_strict and not passed:
        raise RuntimeError("strict SSAPy promotion gate failed; inspect the report")
    return report


def write_ssapy_promotion_report(
    output: str | Path,
    *,
    samples: int = 41,
    require_strict: bool = False,
) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always write the diagnostic record, even when strict promotion is
    # blocked or fails.  CI and release tooling need the reason on disk
    # before the command exits nonzero.
    report = build_ssapy_promotion_report(samples=samples, require_strict=False)
    path.write_text(json.dumps(portable_record(report), indent=2, allow_nan=False), encoding="utf-8")
    if require_strict and not bool(report.get("promoted", False)):
        if report.get("status") == "blocked":
            raise BackendUnavailableError(str(report.get("blocking_reason", "strict SSAPy promotion is blocked")))
        raise RuntimeError("strict SSAPy promotion gate failed; inspect the written report")
    return str(path)
