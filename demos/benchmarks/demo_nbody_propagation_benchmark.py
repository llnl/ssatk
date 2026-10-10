"""Compare SSATK, GMAT, and Orekit with progressively richer point masses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ssapy.body import get_body
from ssapy.gravity import AccelThirdBody

from demos.benchmarks import demo_gmat_benchmark, demo_orekit_benchmark
from demos.benchmarks.benchmark_report import write_benchmark_report
from ssatk.io.ssatk_data import ssatk_data
from ssatk.plots.figpath import figpath
from ssatk.propagators_orbit import propagate_orbit_state

UNDER_PYTEST = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
EPOCH_UTC = "2026-01-01T12:00:00"
GMAT_MU = demo_gmat_benchmark.GMAT_JGM2_MU_M3_S2
MODES = {
    "earth_moon_sun": ("Earth–Moon–Sun", ("Luna", "Sun"), ("moon", "Sun")),
    "solar_system": (
        "Earth + Moon + Sun + planets",
        ("Luna", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"),
        ("moon", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"),
    ),
}
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


def _gmat_script(*, radius_m: float, duration_s: float, step_s: float, point_masses: tuple[str, ...]) -> str:
    steps = round(float(duration_s) / float(step_s))
    if steps < 1 or not np.isclose(steps * float(step_s), float(duration_s)):
        raise ValueError("duration_s must be a positive integer multiple of step_s")
    velocity_m_s = float(np.sqrt(GMAT_MU / radius_m))
    reports = ["Report StateReport Sat.ElapsedSecs Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ;"]
    for _ in range(steps):
        reports.extend(
            (
                f"Propagate NBodyProp(Sat) {{Sat.ElapsedSecs = {float(step_s):.17g}}};",
                "Report StateReport Sat.ElapsedSecs Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ;",
            )
        )
    masses = ", ".join(point_masses)
    return f"""% SSATK/GMAT n-body comparison case.
    GMAT SolarSystem.EphemerisSource = 'SPICE';
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '01 Jan 2026 12:00:00.000';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = {radius_m / 1000.0:.17g};
Sat.Y = 0;
Sat.Z = 0;
Sat.VX = 0;
Sat.VY = {velocity_m_s / 1000.0:.17g};
Sat.VZ = 0;

Create ForceModel NBodyFM;
NBodyFM.CentralBody = Earth;
NBodyFM.PrimaryBodies = {{Earth}};
NBodyFM.PointMasses = {{{masses}}};
NBodyFM.Drag = None;
NBodyFM.SRP = Off;
NBodyFM.RelativisticCorrection = Off;
NBodyFM.ErrorControl = RSSStep;
NBodyFM.GravityField.Earth.Degree = 0;
NBodyFM.GravityField.Earth.Order = 0;
NBodyFM.GravityField.Earth.PotentialFile = 'JGM2.cof';
NBodyFM.GravityField.Earth.TideModel = 'None';

Create Propagator NBodyProp;
NBodyProp.FM = NBodyFM;
NBodyProp.Type = RungeKutta89;
NBodyProp.InitialStepSize = 10;
NBodyProp.Accuracy = 1e-13;
NBodyProp.MinStep = 0.001;
NBodyProp.MaxStep = {float(step_s):.17g};

Create ReportFile StateReport;
StateReport.Filename = '/benchmark/states.csv';
StateReport.WriteHeaders = false;
StateReport.FixedWidth = false;
StateReport.Delimiter = ',';
StateReport.Precision = 17;

BeginMissionSequence;
{os.linesep.join(reports)}
"""


def _run_gmat(*, root: Path, executable: str, script: str, state_path: Path, expected_samples: int) -> np.ndarray:
    return demo_gmat_benchmark._run_gmat_script(
        root=root,
        executable=executable,
        state_path=state_path,
        script=script,
        expected_samples=expected_samples,
    )


def _orekit_data_dir() -> Path | None:
    configured = os.environ.get("OREKIT_DATA_DIR")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        (
            Path("/p/lustre1/yeager7/orekit-data-de430-clean"),
            Path(__file__).resolve().parents[3] / "orekit-data",
            Path.home() / "workdir" / "orekit-data",
        )
    )
    return next((path for path in candidates if path.is_dir()), None)


def _run_orekit(*, mode: str, radius: float, duration: float, step: float, allow_install: bool):
    data_dir = _orekit_data_dir()
    jar = demo_orekit_benchmark._orekit_jar(allow_install=allow_install)
    if (
        data_dir is None
        or jar is None
        or shutil.which("mvn") is None
        or shutil.which("javac") is None
        or shutil.which("java") is None
    ):
        return None
    with tempfile.TemporaryDirectory(prefix="ssatk-orekit-nbody-") as temp_name:
        temp = Path(temp_name)
        classpath_file = temp / "classpath.txt"
        subprocess.run(
            ["mvn", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={classpath_file}"],
            cwd=demo_orekit_benchmark.OREKIT_DIR,
            check=True,
        )
        dependencies = classpath_file.read_text(encoding="utf-8").strip()
        classpath = os.pathsep.join((str(jar), dependencies))
        subprocess.run(
            [
                "javac",
                "-source",
                "8",
                "-target",
                "8",
                "-cp",
                classpath,
                "-d",
                str(temp),
                str(demo_orekit_benchmark.OREKIT_DIR / "OrekitNBody.java"),
            ],
            check=True,
        )
        completed = subprocess.run(
            [
                "java",
                "-cp",
                os.pathsep.join((str(temp), classpath)),
                "OrekitNBody",
                str(data_dir),
                mode,
                str(GMAT_MU),
                str(radius),
                str(duration),
                str(step),
                EPOCH_UTC,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    rows = np.loadtxt(completed.stdout.splitlines()[1:], delimiter=",")
    return rows.reshape((-1, 7))


def _ssatk_rows(*, mode: str, radius: float, duration: float, step: float) -> np.ndarray:
    epoch_gps = float(Time(EPOCH_UTC, scale="utc").gps)
    elapsed = np.arange(0.0, duration + 0.5 * step, step)
    bodies = MODES[mode][2]
    acceleration = [AccelThirdBody(get_body(name)) for name in bodies]
    state = propagate_orbit_state(
        r0=np.array([radius, 0.0, 0.0]),
        v0=np.array([0.0, np.sqrt(GMAT_MU / radius), 0.0]),
        times=epoch_gps + elapsed,
        t0=epoch_gps,
        mu=GMAT_MU,
        acceleration=acceleration,
        rtol=1e-12,
        atol=1e-9,
    )
    return np.column_stack((elapsed, state.r, state.v))


def _compare(reference_rows: np.ndarray, ssatk_rows: np.ndarray, *, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    reference = np.asarray(reference_rows, dtype=float).reshape((-1, 7))
    ssatk = np.asarray(ssatk_rows, dtype=float).reshape((-1, 7))
    if reference.shape != ssatk.shape or not np.allclose(reference[:, 0], ssatk[:, 0], atol=1e-6):
        raise RuntimeError("n-body benchmark epochs do not match")
    dr = np.linalg.norm(ssatk[:, 1:4] - reference[:, 1:4] * scale, axis=1)
    dv = np.linalg.norm(ssatk[:, 4:7] - reference[:, 4:7] * scale, axis=1)
    residuals = np.column_stack((reference[:, 0], dr, dv))
    return residuals, {
        "duration_s": float(reference[-1, 0]),
        "sample_count": int(reference.shape[0]),
        "rms_position_error_m": float(np.sqrt(np.mean(dr**2))),
        "max_position_error_m": float(np.max(dr)),
        "rms_velocity_error_m_s": float(np.sqrt(np.mean(dv**2))),
        "max_velocity_error_m_s": float(np.max(dv)),
    }


def _write_case_plot(case: dict, residuals: dict[str, np.ndarray], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, values in residuals.items():
        hours = values[:, 0] / 3_600.0
        axes[0].plot(hours, values[:, 1], label=name)
        axes[1].plot(hours, values[:, 2], label=name)
    axes[0].set_ylabel("Position residual [m]")
    axes[1].set_ylabel("Velocity residual [m/s]")
    axes[1].set_xlabel("Elapsed time [hr]")
    axes[0].set_title(
        f"SSATK n-body residuals: {case['label']} — {case['mode_label']}\n"
        "Earth μ = GMAT JGM2; SSAPy / GMAT / Orekit DE430"
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    path = out_dir / f"nbody_{case['mode']}_{case['name']}_residuals.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _write_summary_plot(cases: list[dict], out_dir: Path) -> str:
    mode_labels = {"earth_moon_sun": "E–M–S", "solar_system": "All planets"}
    labels = [f"{case['label']}\n{mode_labels[case['mode']]}" for case in cases]
    tools = sorted({tool for case in cases for tool in case["tools"]})
    x = np.arange(len(labels))
    width = 0.8 / max(len(tools), 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for index, tool in enumerate(tools):
        position = [case["tools"].get(tool, {}).get("rms_position_error_m", np.nan) for case in cases]
        velocity = [case["tools"].get(tool, {}).get("rms_velocity_error_m_s", np.nan) for case in cases]
        offset = (index - (len(tools) - 1) / 2) * width
        axes[0].bar(x + offset, position, width, label=tool)
        axes[1].bar(x + offset, velocity, width, label=tool)
    axes[0].set_ylabel("RMS position [m]")
    axes[1].set_ylabel("RMS velocity [m/s]")
    axes[1].set_xticks(x, labels)
    axes[0].set_title("SSATK higher n-body propagation comparison")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.text(
        0.5,
        0.01,
        "Earth μ = GMAT JGM2; SSAPy / GMAT / Orekit DE430",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18)
    path = out_dir / "nbody_summary.png"
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
    gmat_ready = (
        gmat is not None
        and demo_gmat_benchmark._ensure_container_image(allow_install=allow_install)
        and demo_gmat_benchmark._de430_available()
    )
    figure_dir = Path(figpath("benchmarks"))
    results = []
    for mode, (mode_label, point_masses, _) in MODES.items():
        for case_template in CASES:
            case = dict(case_template)
            if fast:
                case["duration_s"] = min(case["duration_s"], 86_400.0)
            case.update({"mode": mode, "mode_label": mode_label})
            expected_samples = round(case["duration_s"] / case["step_s"]) + 1
            ssatk = _ssatk_rows(mode=mode, radius=case["radius_m"], duration=case["duration_s"], step=case["step_s"])
            residuals = {}
            tool_results = {}

            if gmat_ready and mode == "earth_moon_sun":
                state_path = Path(ssatk_data(f"data/benchmarks/nbody/{mode}_{case['name']}_gmat_states.csv"))
                rows = _run_gmat(
                    root=gmat[0],
                    executable=gmat[1],
                    script=_gmat_script(
                        radius_m=case["radius_m"],
                        duration_s=case["duration_s"],
                        step_s=case["step_s"],
                        point_masses=point_masses,
                    ),
                    state_path=state_path,
                    expected_samples=expected_samples,
                )
                residuals["GMAT"] , tool_results["GMAT"] = _compare(rows, ssatk, scale=1_000.0)
                tool_results["GMAT"]["state_path"] = str(state_path)

            orekit = _run_orekit(
                mode=mode,
                radius=case["radius_m"],
                duration=case["duration_s"],
                step=case["step_s"],
                allow_install=allow_install,
            )
            if orekit is not None:
                state_path = Path(ssatk_data(f"data/benchmarks/nbody/{mode}_{case['name']}_orekit_states.csv"))
                np.savetxt(state_path, orekit, delimiter=",", fmt="%.17g")
                residuals["Orekit"], tool_results["Orekit"] = _compare(orekit, ssatk, scale=1.0)
                tool_results["Orekit"]["state_path"] = str(state_path)

            if not tool_results:
                continue
            case["expected_sample_count"] = expected_samples
            case["tools"] = tool_results
            if make_figures:
                case["plot"] = _write_case_plot(case, residuals, figure_dir)
            results.append(case)

    if not results:
        return {"skipped": True, "reason": "GMAT and Orekit n-body runtimes unavailable"}
    summary = {
        "benchmark": "SSATK higher n-body propagation comparison",
        "epoch_utc": EPOCH_UTC,
        "earth_mu_m3_s2": float(GMAT_MU),
        "force_models": "Earth degree/order-0 point mass plus listed third-body point masses",
        "ephemeris_sources": {"ssatk": "DE430", "gmat": "DE430", "orekit": "DE430"},
        "ssatk_method": "DOP853, rtol=1e-12, atol=1e-9, SSAPy AccelThirdBody",
        "cases": results,
    }
    if make_figures:
        summary["summary_plot"] = _write_summary_plot(results, figure_dir)
    result_path = Path(ssatk_data("data/benchmarks/nbody_propagation_results.json"))
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
