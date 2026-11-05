"""Calibrate a synthetic motor/sensor and retain a deliberately failed holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import SpacecraftManeuverAccel
from ssatk.coordinates.attitude import rotate_vector
from ssatk.launch import calibrate_launch_measurements
from ssatk.propagators_6dof import Spacecraft

GALLERY_INCLUDE = False


def reference_case(name, *, epoch=1.4e9, mass=17., offset=0.):
    """Closed-form constant-flow rocket velocity plus a rotating sensor lever arm."""
    time = epoch+np.arange(.5, 3.01, .5)
    sample = np.r_[time[0]-.25, time]+.02
    elapsed = sample-epoch
    flow, thrust, bias, spin = .1, 10., .04, .2
    arm = np.array([.3, .4, 0.])
    velocity = thrust/flow*np.log(mass/(mass-flow*elapsed))
    velocity -= spin*(arm[0]*np.sin(spin*elapsed)+arm[1]*np.cos(spin*elapsed))
    observed = (velocity[1:]+velocity[:-1])/2+bias+offset
    covariance = .01**2 * .2**np.abs(np.subtract.outer(np.arange(time.size), np.arange(time.size)))
    source = "Specified mathematical case in demo_launch_calibration.py; not measured hardware"
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = {name: {"source": source, "sha256": fingerprint,
                     "uncertainty": "Prescribed exact input; Gaussian sensor covariance supplied separately"}
              for name in ("thrust", "mass_properties", "atmosphere", "telemetry")}
    return {
        "id": name, "flight_id": name, "evidence": "synthetic", "observable": "sensor_velocity_x",
        "unit": "m/s", "time_gps_s": time.tolist(), "observed": observed.tolist(),
        "covariance": covariance.tolist(), "inputs": inputs,
        "independence": "Separate specified flights with independent sensor errors",
        "model_source": "Constant-flow rocket equation; spherical fixed inertia, no gravity or atmosphere",
        "model": {"epoch_gps_s": epoch, "initial_mass_kg": mass, "flow_kg_s": flow,
                  "spin_rad_s": spin, "rtol": 1e-11, "atol": 1e-12},
        "sensor": {"position_body_m": arm.tolist(), "frame": "GCRF x velocity",
                   "filter": "Two-sample arithmetic mean; first support sample 0.25 s before first tag",
                   "time_alignment": "Physical sample GPS time = telemetry tag + 0.02 s",
                   "support_gps_s": [epoch, float(sample[-1])]},
        "requirements": {"max_abs_residual": .05, "max_abs_bias": .02, "min_coverage": .9,
                         "source": "Declared synthetic demonstration budgets; not mission acceptance limits"},
    }


def predict(parameters, case):
    """Reuse 6-DoF propagation, then apply the specified sensor position and filter."""
    model = case["model"]
    epoch = model["epoch_gps_s"]
    time = np.asarray(case["time_gps_s"])
    samples = np.r_[time[0]-.25, time]+.02
    craft = Spacecraft(r=[7e6, 0., 0.], v=[0., 0., 0.], t=epoch,
                       mass=model["initial_mass_kg"], inertia=np.eye(3), omega=[0., 0., model["spin_rad_s"]])
    burn = SpacecraftManeuverAccel(parameters["thrust_n"], frame="gcrf", direction=[1., 0., 0.],
                                  propellant_flow_rate=model["flow_kg_s"], start=epoch, stop=samples[-1])
    trajectory = craft.propagate(times=np.r_[epoch, samples], models=[burn], mu=0.,
                                 rtol=model["rtol"], atol=model["atol"])
    arm = np.asarray(case["sensor"]["position_body_m"])
    velocity = np.array([v[0]+rotate_vector(q, np.cross(omega, arm))[0]
                         for v, q, omega in zip(trajectory.v[1:], trajectory.q[1:], trajectory.omega[1:])])
    return (velocity[1:]+velocity[:-1])/2+parameters["velocity_bias_m_s"]


def run_reference(*, epoch=1.4e9):
    """Fit only the first flight; evaluate nominal and biased flights unchanged."""
    return calibrate_launch_measurements(
        [reference_case("calibration", epoch=epoch)],
        [reference_case("held_out", epoch=epoch+10, mass=19.),
         reference_case("failed_sensor", epoch=epoch+20, mass=23., offset=.3)],
        predict, {"thrust_n": 9., "velocity_bias_m_s": 0.},
        parameter_units=["N", "m/s"], parameter_scale=[10., .1],
        parameter_bounds=([1., -.5], [20., .5]), coverage_probability=.95,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten")
    args = parser.parse_args(argv)
    report = run_reference()
    encoded = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
