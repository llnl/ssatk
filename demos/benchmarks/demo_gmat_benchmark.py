"""Compare SSATK point-mass propagation with GMAT's R2026a console."""

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

from ssatk.constants import EARTH_MU
from ssatk.io.ssatk_data import ssatk_data
from ssatk.plots.figpath import figpath
from ssatk.propagators_orbit import propagate_orbit_state

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "benchmarks"
GMAT_IMAGE = "ubuntu:24.04"
# POTFIELD value in GMAT's data/gravity/earth/JGM2.cof.
GMAT_JGM2_MU_M3_S2 = 3.986004415e14
GMAT_DE430_BSP = Path(os.environ.get("GMAT_DE430_BSP", "/p/lustre1/yeager7/gmat-data-de430/de430.bsp"))


def _de430_available() -> bool:
    return GMAT_DE430_BSP.is_file()


def _find_gmat() -> tuple[Path, str] | None:
    configured = os.environ.get("GMAT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / "GMAT",
            Path.home() / "workdir" / "GMAT",
        ]
    )
    for root in candidates:
        for executable in ("GmatConsole-R2026a", "GmatConsole"):
            if (root / "bin" / executable).is_file():
                return root, executable
    return None


def _ensure_container_image(*, allow_install: bool) -> bool:
    if shutil.which("podman") is None:
        return False
    exists = subprocess.run(
        ["podman", "image", "exists", GMAT_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode == 0:
        return True
    if not allow_install:
        return False
    pulled = subprocess.run(
        ["podman", "pull", GMAT_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return pulled.returncode == 0


def _script(*, radius_m: float, velocity_m_s: float, duration_s: float, step_s: float) -> str:
    steps = round(float(duration_s) / float(step_s))
    if steps < 1 or not np.isclose(steps * float(step_s), float(duration_s)):
        raise ValueError("duration_s must be a positive integer multiple of step_s")
    reports = [
        "Report StateReport Sat.ElapsedSecs Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ;"
    ]
    for _ in range(steps):
        reports.extend(
            (
                f"Propagate TwoBodyProp(Sat) {{Sat.ElapsedSecs = {float(step_s):.17g}}};",
                "Report StateReport Sat.ElapsedSecs Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ;",
            )
        )
    return f"""% SSATK/GMAT two-body comparison case.
Create Spacecraft Sat;
Sat.DateFormat = A1ModJulian;
Sat.Epoch = 21545.0;
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = {radius_m / 1000.0:.17g};
Sat.Y = 0;
Sat.Z = 0;
Sat.VX = 0;
Sat.VY = {velocity_m_s / 1000.0:.17g};
Sat.VZ = 0;

Create ForceModel TwoBodyFM;
TwoBodyFM.CentralBody = Earth;
TwoBodyFM.PrimaryBodies = {{Earth}};
TwoBodyFM.GravityField.Earth.Degree = 0;
TwoBodyFM.GravityField.Earth.Order = 0;
TwoBodyFM.GravityField.Earth.PotentialFile = 'JGM2.cof';
TwoBodyFM.GravityField.Earth.TideModel = 'None';

Create Propagator TwoBodyProp;
TwoBodyProp.FM = TwoBodyFM;
TwoBodyProp.Type = RungeKutta89;
TwoBodyProp.InitialStepSize = 10;
TwoBodyProp.Accuracy = 1e-13;
TwoBodyProp.MinStep = 0.001;
TwoBodyProp.MaxStep = {float(step_s):.17g};

Create ReportFile StateReport;
StateReport.Filename = '/benchmark/states.csv';
StateReport.WriteHeaders = false;
StateReport.FixedWidth = false;
StateReport.Delimiter = ',';
StateReport.Precision = 17;

BeginMissionSequence;
{os.linesep.join(reports)}
"""


def _run_gmat(
    *,
    root: Path,
    executable: str,
    state_path: Path,
    radius_m: float,
    velocity_m_s: float,
    duration_s: float,
    step_s: float,
) -> np.ndarray:
    return _run_gmat_script(
        root=root,
        executable=executable,
        state_path=state_path,
        script=_script(
            radius_m=radius_m,
            velocity_m_s=velocity_m_s,
            duration_s=duration_s,
            step_s=step_s,
        ),
        expected_samples=round(float(duration_s) / float(step_s)) + 1,
    )


def _run_gmat_script(
    *,
    root: Path,
    executable: str,
    state_path: Path,
    script: str,
    expected_samples: int,
) -> np.ndarray:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gmat-benchmark-", dir=state_path.parent) as temp_name:
        temp = Path(temp_name)
        (temp / "case.script").write_text(script, encoding="utf-8")
        command = [
            "podman",
            "run",
            "--rm",
            "--userns=keep-id",
            "--volume",
            f"{root}:/GMAT:rw",
            "--volume",
            f"{GMAT_DE430_BSP}:/GMAT/data/planetary_ephem/spk/DE421AllPlanets.bsp:ro",
            "--volume",
            f"{temp}:/benchmark:rw",
            GMAT_IMAGE,
            "bash",
            "-lc",
            (
                "cd /GMAT/bin && export LD_LIBRARY_PATH=/GMAT/bin:/GMAT/lib && "
                f"./{executable} --run /benchmark/case.script "
                "--logfile /benchmark/gmat_console.log"
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("GMAT benchmark exceeded 180 seconds") from exc
        if completed.returncode != 0:
            output = (completed.stdout + "\n" + completed.stderr)[-4000:]
            raise RuntimeError(f"GMAT benchmark failed with exit code {completed.returncode}\n{output}")

        generated = temp / "states.csv"
        if not generated.is_file():
            raise RuntimeError("GMAT completed without producing states.csv")
        shutil.copyfile(generated, state_path)
        rows = np.loadtxt(generated, delimiter=",")
    rows = np.asarray(rows, dtype=float).reshape((-1, 7))
    if rows.shape[0] < 2 or not np.all(np.diff(rows[:, 0]) > 0.0):
        raise RuntimeError(f"GMAT state report is invalid: shape={rows.shape}")
    if rows.shape[0] != expected_samples:
        raise RuntimeError(
            f"GMAT state report has {rows.shape[0]} samples; expected {expected_samples}"
        )
    return rows


def main(make_figures=None, fast=None, verbose=None, allow_install=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST
    if allow_install is None:
        allow_install = not UNDER_PYTEST

    gmat = _find_gmat()
    if gmat is None:
        return {"skipped": True, "reason": "GMAT R2026a installation unavailable"}
    if not _ensure_container_image(allow_install=allow_install):
        return {"skipped": True, "reason": "Podman or Ubuntu 24.04 image unavailable"}

    radius_m = 7_000_000.0
    velocity_m_s = float(np.sqrt(EARTH_MU / radius_m))
    duration_s = 14_400.0 if not fast else 600.0
    step_s = 60.0
    state_path = Path(ssatk_data("data/benchmarks/gmat_two_body_states.csv"))
    rows = _run_gmat(
        root=gmat[0],
        executable=gmat[1],
        state_path=state_path,
        radius_m=radius_m,
        velocity_m_s=velocity_m_s,
        duration_s=duration_s,
        step_s=step_s,
    )

    times = rows[:, 0]
    reference = rows[:, 1:]
    ssatk = propagate_orbit_state(
        r0=reference[0, :3] * 1_000.0,
        v0=reference[0, 3:] * 1_000.0,
        times=times,
        mu=GMAT_JGM2_MU_M3_S2,
        rtol=1e-12,
        atol=1e-9,
    )
    dr = np.linalg.norm(ssatk.r - reference[:, :3] * 1_000.0, axis=1)
    dv = np.linalg.norm(ssatk.v - reference[:, 3:] * 1_000.0, axis=1)
    result = {
        "skipped": False,
        "tool": "GMAT R2026a RungeKutta89",
        "container": GMAT_IMAGE,
        "force_model": "Earth degree/order 0 JGM2 point mass",
        "duration_s": float(times[-1]),
        "sample_count": int(times.size),
        "state_path": str(state_path),
        "rms_position_error_m": float(np.sqrt(np.mean(dr**2))),
        "max_position_error_m": float(np.max(dr)),
        "rms_velocity_error_m_s": float(np.sqrt(np.mean(dv**2))),
        "max_velocity_error_m_s": float(np.max(dv)),
    }

    if make_figures:
        out_dir = Path(figpath("benchmarks"))
        out_dir.mkdir(parents=True, exist_ok=True)
        hours = times / 3_600.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dr, label="|Δr|")
        ax.set(xlabel="Elapsed time [hr]", ylabel="Position residual [m]", title="SSATK vs GMAT two-body position")
        ax.grid(True, alpha=0.3)
        ax.legend()
        position_path = out_dir / "gmat_two_body_position_error.png"
        fig.savefig(position_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dv, label="|Δv|")
        ax.set(xlabel="Elapsed time [hr]", ylabel="Velocity residual [m/s]", title="SSATK vs GMAT two-body velocity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        velocity_path = out_dir / "gmat_two_body_velocity_error.png"
        fig.savefig(velocity_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        result["position_error_plot"] = str(position_path)
        result["velocity_error_plot"] = str(velocity_path)

    result_path = Path(ssatk_data("data/benchmarks/gmat_two_body_results.json"))
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["results_path"] = str(result_path)
    if verbose:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True, allow_install=True)
