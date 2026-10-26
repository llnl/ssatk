"""Run an isolated variable-mass motor burn using installed SSAPy-Data inputs."""

from __future__ import annotations

import json

import numpy as np

from ssatk.accelerations_6dof.thrust import (
    THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR,
    SpacecraftManeuverAccel,
    load_packaged_thrust_curve,
    load_packaged_thrust_curve_metadata,
)
from ssatk.constants import STANDARD_GRAVITY
from ssatk.launch.preflight import verify_data_resources
from ssatk.propagators_6dof import Spacecraft

GALLERY_INCLUDE = False
MOTOR_CURVE = "estes_c6_rasp_5f4294d20002e90000000833"


def main():
    resource_paths = [
        f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/index.csv",
        f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/{MOTOR_CURVE}.csv",
        f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/{MOTOR_CURVE}.json",
    ]
    resources = verify_data_resources([
        {"path": path} for path in resource_paths
    ])
    metadata = load_packaged_thrust_curve_metadata(MOTOR_CURVE, collection="thrustcurve_org_pd")
    curve = load_packaged_thrust_curve(MOTOR_CURVE, collection="thrustcurve_org_pd")
    motor = metadata["motor"]
    initial_mass = float(motor["totalWeightG"]) / 1000.0
    propellant_mass = float(motor["propWeightG"]) / 1000.0
    if not np.isfinite([initial_mass, propellant_mass]).all() or not 0 < propellant_mass < initial_mass:
        raise ValueError("motor masses must be finite and satisfy 0 < propellant < total mass")
    if not np.isfinite(curve.total_impulse) or curve.total_impulse <= 0.0:
        raise ValueError("motor curve must supply a positive finite impulse")

    effective_isp = curve.total_impulse / (propellant_mass * STANDARD_GRAVITY)
    acceleration = SpacecraftManeuverAccel(curve, frame="gcrf", isp=effective_isp)
    spacecraft = Spacecraft(r=[1.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], mass=initial_mass, inertia=np.eye(3))
    trajectory = spacecraft.propagate(
        times=curve.times, acceleration=acceleration, mu=0.0,
        max_step=float(np.min(np.diff(curve.times))) / 2.0,
    )
    dry_mass = initial_mass - propellant_mass
    reference_delta_v = effective_isp * STANDARD_GRAVITY * np.log(initial_mass / dry_mass)
    delta_v = float(trajectory.v[-1, 0])
    result = {
        "scope": "ISOLATED MOTOR BURN; NOT AN EARTH LAUNCH OR FLIGHT QUALIFICATION",
        "qualification_status": "NOT RUN",
        "assumptions": [
            "Motor only; no payload, gravity, atmosphere, winds, torque or staging.",
            "Fixed inertial +X thrust; identity inertia is unused without rotational dynamics.",
            "Constant effective Isp inferred from sampled impulse and catalog propellant mass.",
            "Propellant mass flow is proportional to thrust; no measured depletion history.",
        ],
        "metadata": metadata,
        "resource_checks": resources,
        "sampled_impulse_n_s": curve.total_impulse,
        "catalog_impulse_n_s": float(motor["totImpulseNs"]),
        "impulse_difference_n_s": curve.total_impulse - float(motor["totImpulseNs"]),
        "effective_isp_s": effective_isp,
        "initial_mass_kg": initial_mass,
        "dry_mass_kg": dry_mass,
        "final_mass_kg": float(trajectory.mass[-1]),
        "mass_residual_kg": float(trajectory.mass[-1]) - dry_mass,
        "delta_v_m_s": delta_v,
        "rocket_equation_delta_v_m_s": float(reference_delta_v),
        "delta_v_residual_m_s": delta_v - float(reference_delta_v),
        "elapsed_time_s": (trajectory.t - trajectory.t[0]).tolist(),
        "mass_kg": trajectory.mass.tolist(),
        "position_m": trajectory.r.tolist(),
        "velocity_m_s": trajectory.v.tolist(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
