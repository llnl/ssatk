"""Run the optional external-tool checks and write one JSON report."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssatk import __version__
from ssatk.constants import EARTH_MU

Metric = tuple[str, str, float]

_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "orekit_two_body",
        "tool": "Orekit 10.3.1 KeplerianPropagator",
        "model": "Earth two-body Keplerian propagation",
        "runner": "demos.benchmarks.demo_orekit_benchmark.main",
        "constants": {"earth_mu_m3_s2": EARTH_MU},
        "settings": {"duration_s": 3_600.0, "step_s": 60.0, "solver": "SSATK DOP853"},
        "metrics": (
            ("max_position_error_m", "m", 2.0e-2),
            ("max_velocity_error_m_s", "m/s", 2.0e-5),
        ),
    },
    {
        "id": "gmat_two_body",
        "tool": "GMAT R2026a RungeKutta89",
        "model": "Earth degree/order-0 JGM2 point mass",
        "runner": "demos.benchmarks.demo_gmat_benchmark.main",
        "constants": {"gmat_jgm2_mu_m3_s2": 3.986004415e14},
        "settings": {"duration_s": 600.0, "step_s": 60.0, "solver": "SSATK DOP853"},
        "metrics": (
            ("max_position_error_m", "m", 2.0e-2),
            ("max_velocity_error_m_s", "m/s", 2.0e-5),
        ),
    },
    {
        "id": "basilisk_6dof",
        "tool": "Basilisk Spacecraft + ExtForceTorque",
        "model": "six-DoF zero-gravity constant body force and torque",
        "runner": "demos.benchmarks.demo_basilisk_6dof.run",
        "constants": {"mu_m3_s2": 0.0, "mass_kg": 12.0, "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]},
        "settings": {"duration_s": 20.0, "step_s": 0.5, "solver": "SSATK DOP853"},
        "metrics": (
            ("max_position_error_m", "m", 1.0e-5),
            ("max_velocity_error_m_s", "m/s", 1.0e-7),
            ("max_quaternion_error", "1", 1.0e-8),
            ("max_body_rate_error_rad_s", "rad/s", 1.0e-9),
        ),
    },
)


def _runner(case: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    module_name, function_name = case["runner"].rsplit(".", 1)
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def _call(case: dict[str, Any], *, fast: bool, output_dir: Path) -> dict[str, Any]:
    runner = _runner(case)
    if case["id"] == "basilisk_6dof":
        return runner(output_dir=output_dir, fast=fast)
    return runner(make_figures=False, fast=fast, verbose=False, allow_install=False)


def _acceptance(result: dict[str, Any], metrics: tuple[Metric, ...]) -> list[dict[str, Any]]:
    acceptance = []
    for key, unit, tolerance in metrics:
        value = result.get(key)
        numeric = float(value) if isinstance(value, (int, float)) else math.nan
        acceptance.append(
            {
                "name": key,
                "value": numeric if math.isfinite(numeric) else None,
                "unit": unit,
                "tolerance": tolerance,
                "pass": math.isfinite(numeric) and numeric <= tolerance,
                "reference": "zero residual",
            }
        )
    return acceptance


def _metadata(case: dict[str, Any], *, fast: bool) -> dict[str, Any]:
    durations = {"orekit_two_body": 3_600.0, "gmat_two_body": 600.0, "basilisk_6dof": 20.0}
    if not fast:
        durations.update({"orekit_two_body": 43_200.0, "gmat_two_body": 14_400.0, "basilisk_6dof": 120.0})
    return {
        **{key: case[key] for key in ("id", "tool", "model", "runner", "constants")},
        "settings": {**case["settings"], "duration_s": durations[case["id"]]},
        "tolerances": {key: {"value": tolerance, "unit": unit} for key, unit, tolerance in case["metrics"]},
    }


def run_external_validation(*, output_path: Path | None = None, fast: bool = True) -> dict[str, Any]:
    """Run fixed external cases and optionally write a deterministic JSON report."""
    cases = []
    with tempfile.TemporaryDirectory(prefix="ssatk-external-validation-") as temporary:
        output_dir = Path(temporary)
        for specification in _CASES:
            try:
                result = _call(specification, fast=fast, output_dir=output_dir)
            except Exception as error:  # noqa: BLE001 - retain failing case context in the report.
                case = {
                    **_metadata(specification, fast=fast),
                    "status": "error",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            else:
                skipped = bool(result.get("skipped"))
                acceptance = [] if skipped else _acceptance(result, specification["metrics"])
                passed = not skipped and all(metric["pass"] for metric in acceptance)
                case = {
                    **_metadata(specification, fast=fast),
                    "status": "skipped" if skipped else ("passed" if passed else "failed"),
                    "result": result,
                    "acceptance": acceptance,
                }
                if skipped:
                    case["skip"] = {"reason": result.get("reason", "external runtime unavailable"), "allowed": True}
            cases.append(case)

    summary = {
        "passed": sum(case["status"] == "passed" for case in cases),
        "failed": sum(case["status"] == "failed" for case in cases),
        "skipped": sum(case["status"] == "skipped" for case in cases),
        "errors": sum(case["status"] == "error" for case in cases),
        "total": len(cases),
    }
    report = {
        "schema_version": 1,
        "deterministic": True,
        "ssatk_version": __version__,
        "configuration": {"fast": fast, "allow_install": False, "make_figures": False},
        "provenance": {
            "reference": "Existing SSATK external benchmark runners; residuals are compared at runner-reported epochs.",
            "source_checkout": True,
        },
        "summary": summary,
        "cases": cases,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "ssatk_output" / "reports" / "external_validation.json")
    parser.add_argument("--full", action="store_true", help="Use the longer fixed benchmark durations.")
    parser.add_argument("--require-external", action="store_true", help="Fail if any external runtime is skipped.")
    args = parser.parse_args(argv)
    report = run_external_validation(output_path=args.output, fast=not args.full)
    print(json.dumps(report, indent=2, sort_keys=True))
    summary = report["summary"]
    return int(bool(summary["failed"] or summary["errors"] or (args.require_external and summary["skipped"])))


if __name__ == "__main__":
    raise SystemExit(main())
