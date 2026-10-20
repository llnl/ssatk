"""Destination-system acceptance for a genuine LLNL SSAPy eclipse release.

This module is intentionally separate from the portable reference build.  It
runs only through ``backend='ssapy'`` and therefore cannot silently substitute
Swiss Ephemeris, the analytical preview, Astropy-only frames, or the bundled
reference reconstruction.  The report is always written, including when the
host is missing the LLNL runtime or its ephemeris/orientation data.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Mapping

from ssatk.compute.eclipse_event_search import discover_eclipses, write_discovery_catalog
from ssatk.compute.eclipse_capability_audit import capability_audit
from ssatk.compute.eclipse_promotion import build_ssapy_promotion_report


def _write_markdown(report: Mapping[str, object], path: Path) -> None:
    status = str(report.get("status", "unknown"))
    lines = [
        "# Strict LLNL SSAPy eclipse acceptance",
        "",
        f"Status: **{status}**",
        "",
        "This gate requires LLNL SSAPy ephemerides and SSATK frame",
        "transformations for every event state. It never permits a fallback.",
        "",
    ]
    reason = report.get("blocking_reason")
    if reason:
        lines.extend(["## Blocking reason", "", str(reason), ""])
    products = report.get("products", {})
    if products:
        lines.extend(["## Products", ""])
        for key, value in dict(products).items():
            lines.append(f"- **{key}:** `{value}`")
        lines.append("")
    lines.extend([
        "## Required destination command",
        "",
        "```bash",
        "ssatk-eclipse strict-acceptance --require-strict --render --output-dir strict_ssapy_acceptance",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_strict_acceptance(
    output_dir: str | Path,
    *,
    samples: int = 61,
    render: bool = False,
    quality: str = "high",
    require_strict: bool = False,
) -> dict[str, object]:
    """Execute and seal the strict LLNL acceptance record.

    When the LLNL runtime is missing the function returns a complete blocked
    report unless ``require_strict`` is true.  When available it validates the
    two reference events, performs an arbitrary 2026 search with the same
    strict provider, serializes all states, and optionally renders peak HTMLs.
    """
    from ssatk.eclipse_api_impl import (
        EventRequest,
        RenderRequest,
        build_event,
        export_event_state,
        render_product,
        validate_event,
    )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    capability = capability_audit()
    promotion = build_ssapy_promotion_report(samples=max(25, int(samples)), require_strict=False)
    report: dict[str, object] = {
        "$schema": "ssapy-toolkit.eclipse.strict-acceptance/2.0",
        "requested_backend": "ssapy",
        "capability_audit": capability,
        "promotion_gate": promotion,
        "products": {},
    }
    if not bool(promotion.get("promoted", False)):
        report.update({
            "status": "blocked" if promotion.get("status") == "blocked" else "failed",
            "passed": False,
            "blocking_reason": promotion.get("blocking_reason", "strict promotion gate did not pass"),
        })
    else:
        products: dict[str, str] = {}
        validations: dict[str, object] = {}
        for kind in ("solar", "lunar"):
            event = build_event(EventRequest(
                kind=kind, backend="ssapy", sample_count=max(25, int(samples)), solar_scope="global"
            ))
            validation = validate_event(event)
            validations[kind] = validation.to_dict()
            products[f"{kind}_event"] = export_event_state(event, out / f"{kind}_strict_ssapy_event_v22_2.json")
            if render:
                products[f"{kind}_peak"] = str(render_product(RenderRequest(
                    product="cinematic", output=out / f"{kind}_strict_ssapy_peak_v22_2.html",
                    event=event, quality=quality, animate=False,
                )))
        records = discover_eclipses(
            "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z",
            kind="all", backend="ssapy",
        )
        products["arbitrary_2026_catalog"] = write_discovery_catalog(
            records, out / "strict_ssapy_2026_catalog_v22_2.json"
        )
        passed = all(bool(value.get("passed")) for value in validations.values()) and bool(records)
        report.update({
            "status": "passed" if passed else "failed",
            "passed": bool(passed),
            "validations": validations,
            "arbitrary_event_count": len(records),
            "products": products,
        })
    json_path = out / "STRICT_SSAPY_ACCEPTANCE.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _write_markdown(report, out / "STRICT_SSAPY_ACCEPTANCE.md")
    if require_strict and not bool(report.get("passed", False)):
        raise RuntimeError(str(report.get("blocking_reason", "strict SSAPy acceptance failed")))
    return report
