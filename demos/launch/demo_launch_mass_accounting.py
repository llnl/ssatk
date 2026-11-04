"""Synthetic Step-4 stage/tank accounting; no flight qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from ssatk.launch.vehicles import assemble_stage_stack
from ssatk.satellites import Component, SpacecraftBody, Tank, point_mass_inertia

GALLERY_INCLUDE = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten.")
    args = parser.parse_args(argv)
    payload = SpacecraftBody(name="payload", mass=5., inertia=np.diag([2.,3.,4.]),
                             center_of_mass=[0.,0.,4.])
    stage = SpacecraftBody(name="first", mass=7., inertia=np.diag([4.,5.,6.]), tanks=(
        Tank(name="first-fuel", propellant_mass=20., residual_mass=1., dry_mass=2.,
             position_body=[0.,0.,1.], cylinder_radius=.4, propellant_density=100.,
             propellant_base_body=[0.,0.,0.]),))
    upper = SpacecraftBody(name="upper", mass=3., inertia=np.diag([1.,2.,3.]),
                           center_of_mass=[0.,0.,3.], tanks=(
        Tank(name="upper-fuel", propellant_mass=4., residual_mass=.2,
             position_body=[0.,0.,3.]),))
    fairing = Component(name="fairing", mass=2., position_body=[0.,0.,5.])
    samples = []
    for inventory in np.linspace(20.,1.,20):
        current_stage = stage.with_tank_propellant_mass("first-fuel", float(inventory))
        stack = assemble_stage_stack(payload, [current_stage, upper], structures=[fairing])
        samples.append({"first_inventory_kg": float(inventory), "mass_kg": stack.current_mass,
                        "center_body_m": stack.current_center_of_mass.tolist(),
                        "inertia_body_kg_m2": stack.current_inertia.tolist(),
                        "minimum_inertia_eigenvalue_kg_m2": float(np.linalg.eigvalsh(stack.current_inertia).min())})
    retained = assemble_stage_stack(payload, [upper])
    detached = assemble_stage_stack(current_stage, structures=[fairing])
    mass_residual = stack.current_mass - retained.current_mass - detached.current_mass
    first_moment_residual = (
        stack.current_mass * stack.current_center_of_mass
        - retained.current_mass * retained.current_center_of_mass
        - detached.current_mass * detached.current_center_of_mass
    )

    def inertia_about_origin(body):
        center = body.current_center_of_mass
        return body.current_inertia + point_mass_inertia(body.current_mass, center)

    inertia_residual = (
        inertia_about_origin(stack)
        - inertia_about_origin(retained)
        - inertia_about_origin(detached)
    )
    assert abs(mass_residual) < 1e-12, "detachment mass imbalance in kg"
    assert np.allclose(first_moment_residual, 0.0, atol=1e-12, rtol=0.0), "detachment first-moment imbalance in kg m"
    assert np.allclose(inertia_residual, 0.0, atol=1e-12, rtol=0.0), "detachment inertia imbalance in kg m^2"
    assert all(sample["minimum_inertia_eigenvalue_kg_m2"] > 0 for sample in samples)
    report = {"scope": "STEP 4 MASS ACCOUNTING ONLY", "qualification_status": "NOT RUN",
              "source": "Synthetic specified geometry, not engineering or flight data",
              "model": "Constant-density axial cylinder, flat free surface, no slosh",
              "samples": samples, "detachment_mass_residual_kg": mass_residual,
              "detachment_first_moment_residual_kg_m": first_moment_residual.tolist(),
              "detachment_inertia_residual_kg_m2": inertia_residual.tolist(),
              "retained_mass_kg": retained.current_mass,
              "detached_mass_kg": detached.current_mass}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
