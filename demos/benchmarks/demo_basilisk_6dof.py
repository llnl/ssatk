"""Compare SSATK coupled 6-DoF propagation with Basilisk."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import constant_body_thrust, constant_body_torque
from ssatk.propagators_6dof import propagate_6dof

GALLERY_INCLUDE = False
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
OUTPUT_DIR = Path.home() / "ssatk_output" / "data" / "benchmarks"


def _basilisk_available() -> bool:
    try:
        import Basilisk  # noqa: F401
    except ImportError:
        return False
    return True


def _run_basilisk(times: np.ndarray, *, mass: float, inertia: np.ndarray, force: np.ndarray, torque: np.ndarray):
    from Basilisk.simulation import extForceTorque, spacecraft
    from Basilisk.utilities import SimulationBaseClass, macros, simHelpers

    step = float(times[1] - times[0])
    sim = SimulationBaseClass.SimBaseClass()
    task_name = "dynamicsTask"
    process = sim.CreateNewProcess("dynamicsProcess")
    process.addTask(sim.CreateNewTask(task_name, macros.sec2nano(step)))

    body = spacecraft.Spacecraft()
    body.ModelTag = "sixDofExternalValidation"
    body.hub.mHub = mass
    body.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(inertia.ravel())
    body.hub.r_CN_NInit = [7.0e6, 0.0, 0.0]
    body.hub.v_CN_NInit = [0.0, 0.0, 0.0]
    body.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    body.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    sim.AddModelToTask(task_name, body)

    force_torque = extForceTorque.ExtForceTorque()
    body.addDynamicEffector(force_torque)
    sim.AddModelToTask(task_name, force_torque)
    force_torque.extForce_B = force.tolist()
    force_torque.extTorquePntB_B = torque.tolist()
    recorder = body.scStateOutMsg.recorder()
    sim.AddModelToTask(task_name, recorder)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float(times[-1])))
    sim.ExecuteSimulation()
    return {
        "time_s": np.asarray(recorder.times(), dtype=float) * 1e-9,
        "r_m": np.asarray(recorder.r_BN_N),
        "v_m_s": np.asarray(recorder.v_BN_N),
        "sigma": np.asarray(recorder.sigma_BN),
        "omega_rad_s": np.asarray(recorder.omega_BN_B),
    }


def run(*, output_dir: Path = OUTPUT_DIR, fast: bool = UNDER_PYTEST) -> dict:
    """Run the reproducible Basilisk comparison and write one JSON summary."""
    if not _basilisk_available():
        return {"skipped": True, "reason": "Basilisk package unavailable"}

    from Basilisk.utilities import RigidBodyKinematics
    from ssatk.coordinates.attitude import quaternion_from_matrix

    def mrp_to_quaternion(sigma):
        return quaternion_from_matrix(np.asarray(RigidBodyKinematics.MRP2C(np.asarray(sigma).ravel())).T)

    duration = 20.0 if fast else 120.0
    times = np.linspace(0.0, duration, int(duration / 0.5) + 1)
    mass = 12.0
    inertia = np.diag([2.0, 3.0, 4.0])
    # Axial commands exercise the attitude-dependent rotation while avoiding
    # Basilisk's legacy SWIG handling of non-scalar vector payloads.
    force = np.array([0.0, 0.0, 0.006])
    torque = np.array([0.0, 0.0, 0.003])
    reference = _run_basilisk(times, mass=mass, inertia=inertia, force=force, torque=torque)
    trajectory = propagate_6dof(
        times=times,
        r0=reference["r_m"][0],
        v0=reference["v_m_s"][0],
        q0=mrp_to_quaternion(reference["sigma"][0]),
        omega0=reference["omega_rad_s"][0],
        inertia=inertia,
        mu=0.0,
        body_acceleration=constant_body_thrust(force, mass),
        torque=constant_body_torque(torque),
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05,
    )
    # Basilisk's spacecraft output uses MRPs; convert its samples before comparing.
    q_ref = np.asarray([mrp_to_quaternion(sigma) for sigma in reference["sigma"]])
    dr = np.linalg.norm(trajectory.r - reference["r_m"], axis=1)
    dv = np.linalg.norm(trajectory.v - reference["v_m_s"], axis=1)
    dq = np.minimum(np.linalg.norm(trajectory.q - q_ref, axis=1), np.linalg.norm(trajectory.q + q_ref, axis=1))
    domega = np.linalg.norm(trajectory.omega - reference["omega_rad_s"], axis=1)
    result = {
        "skipped": False,
        "tool": "Basilisk Spacecraft + ExtForceTorque",
        "force_model": "constant body-frame force and torque; zero gravity",
        "duration_s": duration,
        "sample_count": int(times.size),
        "max_position_error_m": float(np.max(dr)),
        "max_velocity_error_m_s": float(np.max(dv)),
        "max_quaternion_error": float(np.max(dq)),
        "max_body_rate_error_rad_s": float(np.max(domega)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "basilisk_6dof_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(fast=False), indent=2))
