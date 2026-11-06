"""Export or customize synthetic motor/sensor data and calibrate held-out cases."""

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
                   "clock_offset_s": .02, "first_sample_lead_s": .25, "filter_weights": [.5, .5],
                   "support_gps_s": [epoch, float(sample[-1])]},
        "requirements": {"max_abs_residual": .05, "max_abs_bias": .02, "min_coverage": .9,
                         "source": "Declared synthetic demonstration budgets; not mission acceptance limits"},
    }


def predict(parameters, case):
    """Reuse 6-DoF propagation, then apply the specified sensor position and filter."""
    model = case["model"]
    sensor = case["sensor"]
    if (case["observable"], case["unit"], sensor["frame"]) != ("sensor_velocity_x", "m/s", "GCRF x velocity"):
        raise ValueError("this sample predictor supports sensor_velocity_x in GCRF x velocity, m/s")
    weights = np.asarray(sensor["filter_weights"], dtype=float)
    offset, lead = float(sensor["clock_offset_s"]), float(sensor["first_sample_lead_s"])
    if (weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
            or not np.isfinite([offset, lead]).all() or lead <= 0):
        raise ValueError("sensor needs two nonnegative filter weights summing to one, a finite clock offset and positive lead")
    epoch = model["epoch_gps_s"]
    time = np.asarray(case["time_gps_s"])
    samples = np.r_[time[0]-lead, time]+offset
    if (samples[0] <= epoch or np.any(np.diff(samples) <= 0)
            or samples[0] < sensor["support_gps_s"][0] or samples[-1] > sensor["support_gps_s"][1]):
        raise ValueError("corrected sensor samples must increase after the model epoch and lie inside support_gps_s")
    craft = Spacecraft(r=[7e6, 0., 0.], v=[0., 0., 0.], t=epoch,
                       mass=model["initial_mass_kg"], inertia=np.eye(3), omega=[0., 0., model["spin_rad_s"]])
    burn = SpacecraftManeuverAccel(parameters["thrust_n"], frame="gcrf", direction=[1., 0., 0.],
                                  propellant_flow_rate=model["flow_kg_s"], start=epoch, stop=samples[-1])
    trajectory = craft.propagate(times=np.r_[epoch, samples], models=[burn], mu=0.,
                                 rtol=model["rtol"], atol=model["atol"])
    arm = np.asarray(sensor["position_body_m"])
    velocity = np.array([v[0]+rotate_vector(q, np.cross(omega, arm))[0]
                         for v, q, omega in zip(trajectory.v[1:], trajectory.q[1:], trajectory.omega[1:])])
    return weights[0]*velocity[:-1]+weights[1]*velocity[1:]+parameters["velocity_bias_m_s"]


def sample_dataset(*, epoch=1.4e9):
    """Editable, deterministic synthetic observations and fit settings; no measurements."""
    return {
        "schema": "ssatk.launch_calibration_dataset.v1",
        "description": "SYNTHETIC: 10 N thrust and 0.04 m/s sensor bias; failed_sensor has an additional 0.3 m/s fault. Not flight measurements.",
        "calibration_cases": [reference_case("calibration", epoch=epoch)],
        "validation_cases": [reference_case("held_out", epoch=epoch+10, mass=19.),
                             reference_case("failed_sensor", epoch=epoch+20, mass=23., offset=.3)],
        "parameters0": {"thrust_n": 9., "velocity_bias_m_s": 0.},
        "parameter_units": ["N", "m/s"], "parameter_scale": [10., .1],
        "parameter_bounds": [[1., -.5], [20., .5]], "coverage_probability": .95,
    }


def run_reference(*, epoch=1.4e9, dataset=None):
    """Fit the designated calibration cases, then evaluate frozen holdouts."""
    options = dict(sample_dataset(epoch=epoch) if dataset is None else dataset)
    if options.pop("schema", None) != "ssatk.launch_calibration_dataset.v1":
        raise ValueError("expected schema ssatk.launch_calibration_dataset.v1")
    options.pop("description", None)
    units = {"thrust_n": "N", "velocity_bias_m_s": "m/s"}
    if (set(options["parameters0"]) != set(units)
            or options["parameter_units"] != [units[name] for name in options["parameters0"]]):
        raise ValueError("the sample predictor requires thrust_n in N and velocity_bias_m_s in m/s")
    return calibrate_launch_measurements(predict=predict, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--write-sample", type=Path, help="Write a new editable synthetic dataset JSON without fitting")
    inputs.add_argument("--dataset", type=Path, help="Read edited dataset JSON; defaults to the built-in synthetic sample")
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten")
    args = parser.parse_args(argv)
    if args.write_sample is not None and args.output is not None:
        parser.error("--write-sample cannot be combined with --output")
    destination = args.write_sample if args.write_sample is not None else args.output
    if destination is not None and destination.exists():
        parser.error(f"refusing to overwrite {destination}")
    dataset = None
    if args.dataset is not None:
        try:
            dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    report = sample_dataset() if args.write_sample is not None else run_reference(dataset=dataset)
    encoded = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if destination is None:
        print(encoded, end="")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    if args.write_sample is not None:
        print(f"Wrote synthetic dataset to {destination}")
        return 0
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
