"""Runtime audit for LLNL SSAPy-driven eclipse rendering."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ssapy_toolkit.io.eclipse_asset_resolver import audit_assets
from ssapy_toolkit.io.eclipse_provenance import portable_record
from ssapy_toolkit.compute.eclipse_brightness import ephemeris_positions
from ssapy_toolkit.compute.eclipse_runtime import (
    BackendUnavailableError,
    resolve_backend,
    runtime_capabilities,
)
from ssapy_toolkit.compute.eclipse_state import build_event, event_moon_body_to_gcrf, event_positions_gcrf
from ssapy_toolkit.coordinates.eclipse_lunar_attitude import iau_moon_attitude
from ssapy_toolkit.compute.eclipse_reference_events import (
    LUNAR_2025, SOLAR_2024, ReferenceDefinition, build_reference_event,
)


def audit_definition(definition: ReferenceDefinition) -> dict[str, object]:
    reference_event = build_reference_event(
        definition, jd=[definition.greatest_jd],
    )
    reference_sun, reference_moon = event_positions_gcrf(
        reference_event, definition.greatest_jd,
    )
    result: dict[str, object] = {
        "event": definition.key,
        "greatest_jd_utc": float(definition.greatest_jd),
        "reference_sun_distance_km": float(np.linalg.norm(reference_sun)),
        "reference_moon_distance_km": float(np.linalg.norm(reference_moon)),
        "ssapy_ephemeris_executed": False,
        "ssapy_moon_orientation_executed": False,
    }
    caps = runtime_capabilities(execute=True)
    if caps.ssapy_ephemeris_healthy:
        ephem = ephemeris_positions([definition.greatest_jd], backend="ssapy")
        result.update({
            "ssapy_ephemeris_executed": True,
            "ssapy_backend_label": ephem.backend,
            "ssapy_sun_distance_km": float(np.linalg.norm(ephem.sun_gcrf_km[0])),
            "ssapy_moon_distance_km": float(np.linalg.norm(ephem.moon_gcrf_km[0])),
            "ssapy_sun_reference_delta_km": float(
                np.linalg.norm(ephem.sun_gcrf_km[0]-reference_sun)
            ),
            "ssapy_moon_reference_delta_km": float(
                np.linalg.norm(ephem.moon_gcrf_km[0]-reference_moon)
            ),
        })
    if caps.ssapy_orientation_healthy:
        try:
            strict_event = build_event(
                definition, backend="ssapy", jd=[definition.greatest_jd]
            )
            strict_matrix = np.asarray(
                event_moon_body_to_gcrf(strict_event, definition.greatest_jd), dtype=float
            )
            reference_matrix = iau_moon_attitude(definition.greatest_jd).body_to_gcrf
            relative = reference_matrix.T @ strict_matrix
            angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1.0)/2.0, -1.0, 1.0))))
            result.update({
                "ssapy_moon_orientation_executed": True,
                "ssapy_moon_orientation_source": caps.ssapy_orientation_source,
                "ssapy_vs_iau_attitude_deg": angle,
                "ssapy_moon_orientation_determinant": float(np.linalg.det(strict_matrix)),
                "ssapy_moon_orientation_orthogonality_error": float(
                    np.max(np.abs(strict_matrix.T @ strict_matrix-np.eye(3)))
                ),
            })
        except Exception as exc:
            result["ssapy_moon_orientation_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _selection_result(mode: str) -> dict[str, object]:
    try:
        return {"ok": True, **resolve_backend(mode).to_dict(portable=True)}
    except (BackendUnavailableError, ImportError) as exc:
        return {"ok": False, "requested": mode, "error": f"{type(exc).__name__}: {exc}"}


def capability_audit() -> dict[str, object]:
    caps = runtime_capabilities(execute=True)
    return {
        "purpose": (
            "V22.2 runtime audit. The SSAPy body callables are exercised with "
            "numeric GPS seconds; strict modes own both ephemerides and frame "
            "transforms and fail instead of silently falling back."
        ),
        "runtime": caps.to_dict(portable=True),
        "ssapy_data": audit_assets(policy="data-first", portable=True),
        "selection_matrix": {
            mode: _selection_result(mode)
            for mode in ("reference", "auto", "ssapy-core", "ssapy")
        },
        "events": [audit_definition(SOLAR_2024), audit_definition(LUNAR_2025)],
    }


def write_capability_audit(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(portable_record(capability_audit()), indent=2), encoding="utf-8")
    return str(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="ssapy_capability_audit.json")
    args = parser.parse_args()
    print(write_capability_audit(args.output))
