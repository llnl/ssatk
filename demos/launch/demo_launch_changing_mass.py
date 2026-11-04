"""Verify a synthetic steady axial drain; no ascent or mission qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import SpacecraftManeuverAccel
from ssatk.constants import STANDARD_GRAVITY
from ssatk.coordinates.attitude import rotate_vector
from ssatk.launch.preflight import collect_software_baseline
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank

GALLERY_INCLUDE = False


class AxialDrainManeuver(SpacecraftManeuverAccel):
    """Steady, pressure-matched axial exhaust with axial spin and no swirl.

    Only the declared example geometry and continuous burn segment are covered.
    The instantaneous center velocity is integrated; thrust already includes
    relative exhaust momentum. See docs/launch_baseline.rst for the balance.
    """

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None):
        if spacecraft is None or np.any(np.asarray(omega)[:2] != 0.0):
            raise ValueError("The axial reference requires a spacecraft and purely axial spin.")
        acceleration = super().acceleration(t=t, r=r, v=v, q=q, omega=omega, spacecraft=spacecraft)
        mass_rate = -self.mass_flow_rate(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega)
        center_rate = mass_rate * spacecraft.body.center_of_mass_mass_derivative(self.tank_name)
        return acceleration - rotate_vector(q, 2.0 * mass_rate * center_rate / spacecraft.mass)


def run_reference():
    """Return state and balance evidence for a two-second steady burn."""
    body = SpacecraftBody(name="synthetic axial vehicle", mass=10.,
                          center_of_mass=[0., 0., 2.], inertia=np.diag([4., 4., 3.]),
                          tanks=(Tank(name="feed", propellant_mass=20., residual_mass=1.,
                                      cylinder_radius=.4, propellant_density=100.,
                                      propellant_base_body=[0., 0., 0.]),))
    burn = AxialDrainManeuver(100., direction=[0., 0., 1.], frame="body",
                             isp=100. / STANDARD_GRAVITY, tank_name="feed")
    nozzle_body = body.tanks[0].propellant_base_body
    mass_rate = -1.
    transport0 = mass_rate * (nozzle_body - body.current_center_of_mass)
    craft = Spacecraft(r=[0., 0., 0.], v=transport0 / body.current_mass,
                       omega=[0., 0., .2], body=body)
    trajectory = craft.propagate(times=np.linspace(0., 2., 21), models=[burn], mu=0.,
                                 angular_momentum_flux=np.zeros(3), rtol=1e-11, atol=1e-12)
    samples = []
    for index, time in enumerate(trajectory.t):
        state = trajectory.spacecraft(index, body=body, tank_name="feed")
        center = state.body.current_center_of_mass
        center_rate = mass_rate * state.body.center_of_mass_mass_derivative("feed")
        transport = mass_rate * (nozzle_body - center)
        samples.append({"elapsed_s": float(time), "mass_kg": state.mass,
                        "position_gcrf_m": state.r.tolist(), "velocity_gcrf_m_s": state.v.tolist(),
                        "q_body_to_gcrf": state.q.tolist(), "omega_body_rad_s": state.omega.tolist(),
                        "center_body_m": center.tolist(), "center_rate_body_m_s": center_rate.tolist(),
                        "inertia_body_kg_m2": state.inertia.tolist(),
                        "linear_momentum_gcrf_kg_m_s": (state.mass * state.v - transport).tolist(),
                        "exhaust_velocity_gcrf_m_s": (state.v - center_rate - [0., 0., 100.]).tolist(),
                        "angular_momentum_gcrf_kg_m2_s": rotate_vector(state.q, state.inertia @ state.omega).tolist()})
    momentum = np.array([sample["angular_momentum_gcrf_kg_m2_s"] for sample in samples])
    impulse = burn.thrust * float(trajectory.t[-1])
    consumed = body.current_mass - float(trajectory.mass[-1])
    residual = float(np.max(np.abs(momentum - momentum[0])))
    assert abs(consumed - 2.) < 1e-9, "mass balance in kg"
    assert abs(impulse - consumed * 100.) < 1e-8, "impulse balance in N s"
    assert residual < 1e-9, "angular momentum balance in kg m2/s"
    return {"scope": "STEP 5 AXIAL CHANGING-MASS REFERENCE ONLY", "status": "PASS",
            "qualification_status": "NOT RUN", "source": "Synthetic specified geometry, not flight data",
            "model": "Steady axial plug flow, axial spin, pressure-matched point nozzle, no swirl or slosh",
            "software": collect_software_baseline(),
            "inputs": {"nozzle_body_m": nozzle_body.tolist(), "signed_mass_rate_kg_s": mass_rate,
                       "thrust_N": burn.thrust, "relative_exhaust_speed_m_s": 100.,
                       "dry_mass_kg": body.mass, "dry_center_body_m": body.center_of_mass.tolist(),
                       "dry_inertia_kg_m2": body.inertia.tolist(),
                       "tank": {"name": body.tanks[0].name, "initial_propellant_kg": body.tanks[0].propellant_mass,
                                "residual_kg": body.tanks[0].residual_mass,
                                "radius_m": body.tanks[0].cylinder_radius,
                                "density_kg_m3": body.tanks[0].propellant_density,
                                "base_body_m": body.tanks[0].propellant_base_body.tolist(),
                                "axis_body": body.tanks[0].propellant_axis_body.tolist()}},
            "samples": samples, "consumed_mass_kg": consumed, "thrust_impulse_N_s": impulse,
            "maximum_angular_momentum_residual_kg_m2_s": residual}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten.")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
