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
from ssatk.data import data_package_available
from ssatk.launch.preflight import verify_data_resources
from ssatk.propagators_6dof import Spacecraft

GALLERY_INCLUDE = False
MOTOR_CURVE = "estes_c5_rasp_5f4294d20002e900000008c0"
# SHA-256 of the installed ssa-data-propulsion 0.0.1 curve CSV and sidecar
# JSON; both files are unchanged in 0.0.2. The collection index.csv is not
# pinned here because ssa-data-propulsion 0.0.2 removes curves from it
# (including the earlier Estes C6 curve this demo used), which would change
# its hash; it is still checked against the installed wheel's own manifest.
MOTOR_CURVE_CSV_SHA256 = "edf41ecb5ee8cc4a0106c2292d42b70aecfa4928332aaa197d6b321ff36fc600"
MOTOR_CURVE_JSON_SHA256 = "f04581656dddb710b041013d7c00ef3ddf2bf6e2c5557e8da469fd3911c6e38a"


def main():
    if not data_package_available("ssa_data_propulsion"):
        message = "ssa-data-propulsion is not installed; install the 'propulsion' extra to run this demo."
        print(f"Skipped: {message}")
        return {"skipped": True, "reason": message}

    resources = verify_data_resources([
        {"path": f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/index.csv"},
        {"path": f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/{MOTOR_CURVE}.csv", "sha256": MOTOR_CURVE_CSV_SHA256},
        {"path": f"{THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}/{MOTOR_CURVE}.json", "sha256": MOTOR_CURVE_JSON_SHA256},
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

    # Inferred, not measured: there is no independent thrust-stand Isp for
    # this curve, only the sampled impulse and the catalog propellant mass.
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
            "Constant effective Isp inferred (not measured) from sampled impulse and catalog propellant mass.",
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
