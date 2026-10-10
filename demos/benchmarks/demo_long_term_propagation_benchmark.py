"""Long-term matched-regime comparisons against GMAT and Orekit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demos.benchmarks import demo_gmat_benchmark, demo_orekit_benchmark
from demos.benchmarks.benchmark_report import write_benchmark_report
from ssatk.constants import EARTH_MU
from ssatk.io.ssatk_data import ssatk_data
from ssatk.plots.figpath import figpath
from ssatk.propagators_orbit import propagate_orbit_state

UNDER_PYTEST = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
GALLERY_CATEGORY = "benchmarks"
GMAT_MU = demo_gmat_benchmark.GMAT_JGM2_MU_M3_S2
MU_DELTA = GMAT_MU - EARTH_MU
MU_RELATIVE_DELTA = MU_DELTA / EARTH_MU


def _mu_note() -> str:
    return (
        f"Δμ (GMAT JGM2 − SSATK) = {MU_DELTA:.3e} m³/s² "
        f"({MU_RELATIVE_DELTA:.3e} relative)"
    )

CASES = (
    {"name": "leo", "label": "LEO", "radius_m": 7_000_000.0, "duration_s": 7 * 86_400.0, "step_s": 300.0},
    {"name": "geo", "label": "GEO", "radius_m": 42_164_169.0, "duration_s": 30 * 86_400.0, "step_s": 900.0},
    {
        "name": "cislunar_radius",
        "label": "Cislunar radius",
        "radius_m": 384_400_000.0,
        "duration_s": 60 * 86_400.0,
        "step_s": 1_800.0,
    },
)


def _compare(rows: np.ndarray, *, scale: float, mu: float = EARTH_MU) -> tuple[np.ndarray, dict[str, float]]:
    rows = np.asarray(rows, dtype=float).reshape((-1, 7))
    times = rows[:, 0]
    reference = rows[:, 1:] * scale
    ssatk = propagate_orbit_state(
        r0=reference[0, :3],
        v0=reference[0, 3:],
        times=times,
        mu=mu,
        rtol=1e-12,
        atol=1e-9,
    )
    dr = np.linalg.norm(ssatk.r - reference[:, :3], axis=1)
    dv = np.linalg.norm(ssatk.v - reference[:, 3:], axis=1)
    return np.column_stack((times, dr, dv)), {
        "duration_s": float(times[-1]),
        "sample_count": int(times.size),
        "rms_position_error_m": float(np.sqrt(np.mean(dr**2))),
        "max_position_error_m": float(np.max(dr)),
        "rms_velocity_error_m_s": float(np.sqrt(np.mean(dv**2))),
        "max_velocity_error_m_s": float(np.max(dv)),
    }


def _write_case_plot(case: dict, residuals: dict[str, np.ndarray], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hours = {name: values[:, 0] / 3_600.0 for name, values in residuals.items()}
    plots = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, values in residuals.items():
        axes[0].plot(hours[name], values[:, 1], label=name)
        axes[1].plot(hours[name], values[:, 2], label=name)
    axes[0].set_ylabel("Position residual [m]")
    axes[1].set_ylabel("Velocity residual [m/s]")
    axes[1].set_xlabel("Elapsed time [hr]")
    axes[0].set_title(f"SSATK long-term residuals: {case['label']}\n{_mu_note()}")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    path = out_dir / f"long_term_{case['name']}_residuals.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plots.append(str(path))
    return plots


def _write_summary_plot(cases: list[dict], out_dir: Path) -> str:
    labels = [case["label"] for case in cases]
    tools = sorted({tool for case in cases for tool in case["tools"]})
    x = np.arange(len(labels))
    width = 0.8 / max(len(tools), 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index, tool in enumerate(tools):
        values_r = [case["tools"].get(tool, {}).get("rms_position_error_m", np.nan) for case in cases]
        values_v = [case["tools"].get(tool, {}).get("rms_velocity_error_m_s", np.nan) for case in cases]
        offset = (index - (len(tools) - 1) / 2) * width
        axes[0].bar(x + offset, values_r, width, label=tool)
        axes[1].bar(x + offset, values_v, width, label=tool)
    axes[0].set_ylabel("RMS position [m]")
    axes[1].set_ylabel("RMS velocity [m/s]")
    axes[1].set_xticks(x, labels)
    axes[0].set_title("SSATK long-term propagation comparison")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.text(0.5, 0.01, _mu_note(), ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.14)
    path = out_dir / "long_term_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main(make_figures=None, fast=None, verbose=None, allow_install=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST
    if allow_install is None:
        allow_install = not UNDER_PYTEST

    gmat = demo_gmat_benchmark._find_gmat()
    gmat_ready = gmat is not None and demo_gmat_benchmark._ensure_container_image(allow_install=allow_install)
    figure_dir = Path(figpath("benchmarks"))
    results = []
    for case_template in CASES:
        case = dict(case_template)
        if fast:
            case["duration_s"] = min(case["duration_s"], 86_400.0)
        velocity_m_s = float(np.sqrt(EARTH_MU / case["radius_m"]))
        case["expected_sample_count"] = round(case["duration_s"] / case["step_s"]) + 1
        residuals = {}
        tool_results = {}

        if gmat_ready:
            state_path = Path(ssatk_data(f"data/benchmarks/long_term/{case['name']}_gmat_states.csv"))
            rows = demo_gmat_benchmark._run_gmat(
                root=gmat[0],
                executable=gmat[1],
                state_path=state_path,
                radius_m=case["radius_m"],
                velocity_m_s=velocity_m_s,
                duration_s=case["duration_s"],
                step_s=case["step_s"],
            )
            residuals["GMAT"], tool_results["GMAT"] = _compare(
                rows,
                scale=np.full(6, 1_000.0),
                mu=GMAT_MU,
            )
            tool_results["GMAT"]["state_path"] = str(state_path)

        orekit = demo_orekit_benchmark._run_orekit(
            radius=case["radius_m"],
            duration=case["duration_s"],
            step=case["step_s"],
            allow_install=allow_install,
        )
        if orekit is not None:
            state_path = Path(ssatk_data(f"data/benchmarks/long_term/{case['name']}_orekit_states.csv"))
            np.savetxt(state_path, orekit, delimiter=",", fmt="%.17g")
            residuals["Orekit"], tool_results["Orekit"] = _compare(orekit, scale=1.0)
            tool_results["Orekit"]["state_path"] = str(state_path)

        if not tool_results:
            continue
        case["tools"] = tool_results
        if make_figures:
            case["plots"] = _write_case_plot(case, residuals, figure_dir)
        results.append(case)

    if not results:
        return {"skipped": True, "reason": "GMAT and Orekit runtimes unavailable"}
    summary = {
        "benchmark": "SSATK long-term propagation comparison",
        "force_model": "Earth-centered degree/order-0 point mass",
        "mu_m3_s2": {
            "ssatk": float(EARTH_MU),
            "gmat_jgm2": float(GMAT_MU),
            "gmat_minus_ssatk": float(MU_DELTA),
            "relative_difference": float(MU_RELATIVE_DELTA),
        },
        "cislunar_definition": "Earth-centered two-body case at lunar orbital radius; not an Earth-Moon-Sun model",
        "ssatk_method": "DOP853, rtol=1e-12, atol=1e-9",
        "cases": results,
    }
    if make_figures:
        summary["summary_plot"] = _write_summary_plot(results, figure_dir)
    result_path = Path(ssatk_data("data/benchmarks/long_term_propagation_results.json"))
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["results_path"] = str(result_path)
    if make_figures:
        summary["report_path"] = write_benchmark_report()
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True, allow_install=True)
