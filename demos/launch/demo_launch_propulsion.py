"""Retain a synthetic fixed-nozzle propulsion reference, not flight qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import SpacecraftManeuverAccel, ThrustCurve, integrated_thrust_impulse
from ssatk.launch.preflight import collect_software_baseline
from ssatk.propagators_6dof import Spacecraft, propagate_spacecraft_segments
from ssatk.satellites import SpacecraftBody, Tank

GALLERY_INCLUDE = False


def run_reference():
    """Check one documented two-second burn between explicit coast segments."""
    body = SpacecraftBody(name="synthetic nozzle reference", mass=10., inertia=np.eye(3),
                          tanks=(Tank(name="feed", propellant_mass=5., residual_mass=1.),
                                 Tank(name="reserve", propellant_mass=2., residual_mass=.5)))
    craft = Spacecraft(r=[0., 0., 0.], v=[0., 0., 0.], body=body)
    thrust = ThrustCurve([1., 3.], [10., 10.], bounds_error=True)
    flow = ThrustCurve([1., 3.], [.1, .1], bounds_error=True)
    burn = SpacecraftManeuverAccel(thrust, propellant_flow_rate=flow, tank_name="feed",
                                  frame="gcrf", direction=[1., 0., 0.], start=1., stop=3.,
                                  throttle=.5, throttle_range=(.3, 1.),
                                  reference_pressure=100000., ambient_pressure=50000.,
                                  nozzle_exit_area=1e-5, pressure_range=(0., 100000.))
    trajectory = propagate_spacecraft_segments(craft, [
        {"times": np.linspace(0., 1., 11)},
        {"times": np.linspace(1., 3., 21), "models": [burn]},
        {"times": np.linspace(3., 4., 11)},
    ], mu=0., rtol=1e-11, atol=1e-12)
    burning_time = np.clip(trajectory.t - 1., 0., 2.)
    reference_mass = 17. - .05*burning_time
    reference_velocity = 110.*np.log(17./reference_mass)
    mass_error = float(np.max(np.abs(trajectory.mass - reference_mass)))
    velocity_error = float(np.max(np.abs(trajectory.v[:, 0] - reference_velocity)))
    impulse = integrated_thrust_impulse(lambda t: np.linalg.norm(burn.thrust_vector(
        t, craft.r, craft.v, craft.q, craft.omega, craft)), 1., 3.)
    assert abs(impulse - 11.) < 1e-9, "fixed-nozzle impulse in N s"
    assert mass_error < 1e-9, "constant flow mass balance in kg"
    assert velocity_error < 1e-9, "rocket equation in m/s"
    final = trajectory.spacecraft(body=body, tank_name="feed")
    np.testing.assert_allclose([tank.propellant_mass for tank in final.body.tanks],
                               [4.9, 2.], atol=1e-9, rtol=0)
    return {
        "scope": "STEP 6 SYNTHETIC FIXED-NOZZLE REFERENCE", "status": "PASS",
        "qualification_status": "NOT RUN", "software": collect_software_baseline(),
        "source": "Specified mathematical reference in demo_launch_propulsion.py; no measured motor data",
        "assumptions": ["Linear thrust and flow throttle scaling, with zero as shutdown",
                        "Fixed nozzle geometry and flow, F=F_ref+(p_ref-p_ambient)*A_exit",
                        "Reference thrust already includes pressure at p_ref; correction applied once",
                        "Force through center of mass; point tanks; no exhaust torque, gravity or atmosphere",
                        "Ignition and cutoff are segment boundaries; no ignition/shutdown state resets"],
        "inputs": {"time_s": thrust.times.tolist(), "reference_thrust_n": thrust.thrusts.tolist(),
                   "reference_flow_kg_s": flow.thrusts.tolist(), "tank_name": "feed",
                   "reference_pressure_pa": 100000., "ambient_pressure_pa": 50000.,
                   "nozzle_exit_area_m2": 1e-5, "pressure_range_pa": [0., 100000.],
                   "throttle": .5, "throttle_range": [.3, 1.], "interpolation": "linear; no extrapolation",
                   "initial_mass_kg": 17., "initial_tank_mass_kg": [5., 2.],
                   "residual_tank_mass_kg": [1., .5]},
        "solver": {"method": "DOP853", "rtol": 1e-11, "atol": 1e-12},
        "impulse_n_s": impulse, "consumed_propellant_kg": 17. - float(trajectory.mass[-1]),
        "final_tank_mass_kg": [tank.propellant_mass for tank in final.body.tanks],
        "maximum_mass_error_kg": mass_error, "maximum_velocity_error_m_s": velocity_error,
        "tolerances": {"mass_kg": 1e-9, "velocity_m_s": 1e-9},
        "elapsed_s": trajectory.t.tolist(), "mass_kg": trajectory.mass.tolist(),
        "position_gcrf_m": trajectory.r.tolist(), "velocity_gcrf_m_s": trajectory.v.tolist(),
    }


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
