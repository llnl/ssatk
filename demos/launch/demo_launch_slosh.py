"""Fit one slosh analogue to synthetic evidence and predict a held-out motion."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ssatk.launch.preflight import collect_software_baseline
from ssatk.propagators_6dof import SloshMode, propagate_6dof_extended

GALLERY_INCLUDE = False


def _axial_reference(times, parameters, displacement, force=0., bus_mass=10.):
    """Closed-form two-mass spring/damper with zero initial velocities (SI)."""
    mass, frequency, damping = parameters
    ratio = 1 + mass/bus_mass
    decay = ratio*damping*frequency
    damped = np.sqrt(ratio*frequency**2-decay**2)
    equilibrium = -force/(bus_mass*ratio*frequency**2)
    amplitude = displacement-equilibrium
    relative = equilibrium + amplitude*np.exp(-decay*times)*(
        np.cos(damped*times) + decay/damped*np.sin(damped*times))
    relative_rate = -amplitude*np.exp(-decay*times)*ratio*frequency**2/damped*np.sin(damped*times)
    bus_position = force*times**2/(2*(bus_mass+mass)) + mass/(bus_mass+mass)*(displacement-relative)
    bus_velocity = force*times/(bus_mass+mass) - mass/(bus_mass+mass)*relative_rate
    return np.column_stack((bus_position, bus_velocity, relative, relative_rate))


def _simulate(times, parameters, displacement, force=0., *, method="DOP853", rtol=1e-10, max_step=.1):
    mass, frequency, damping = parameters
    result = propagate_6dof_extended(
        times=times, r0=[0, 0, 0], v0=[0, 0, 0], mu=0, inertia=np.eye(3),
        bus_mass=10., mass0=10.+mass, coupled_slosh=True,
        slosh=SloshMode([1, 0, 0], mass, frequency, damping,
                        displacement0=displacement, displacement_limit=.05),
        acceleration=lambda t, r, v, q, omega: np.array([force/10., 0, 0]),
        method=method, rtol=rtol, atol=1e-12, max_step=max_step)
    return np.column_stack((result.trajectory.r[:, 0], result.trajectory.v[:, 0], result.slosh))


def run_reference():
    """R1: independent two-mass solution; synthetic identification, not flight data."""
    truth = np.array([2., 3., .06])  # kg, rad/s, dimensionless; specified synthetic analogue.
    calibration_times = np.linspace(0., 4., 33)
    calibration = _axial_reference(calibration_times, truth, .025)
    resolution = 1e-4  # m: declared synthetic observable scale, not instrument accuracy.

    def residual(parameters):
        predicted = _simulate(calibration_times, parameters, .025)
        return ((predicted[:, [0, 2]] - calibration[:, [0, 2]])/resolution).ravel()

    fit = least_squares(residual, [1.5, 2.7, .08], bounds=([.5, 1., .01], [4., 5., .3]),
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    assert fit.success, fit.message
    sensitivity = fit.jac * fit.x
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)
    assert np.linalg.matrix_rank(sensitivity) == 3, "parameters are not locally identifiable"
    # Relative motion alone identifies only effective frequency and damping.
    relative_sensitivity = sensitivity[1::2]
    relative_singular_values = np.linalg.svd(relative_sensitivity, compute_uv=False)
    assert relative_singular_values[-1]/relative_singular_values[0] < 1e-5

    times = np.linspace(0., 6., 121)
    force, displacement = .12, -.015
    reference = _axial_reference(times, truth, displacement, force)
    prediction = _simulate(times, fit.x, displacement, force)
    refined = _simulate(times, fit.x, displacement, force, method="Radau", rtol=1e-12, max_step=.025)
    budgets = np.array([2e-7, 2e-7, 2e-7, 2e-7])  # m, m/s, m, m/s.
    errors = np.max(np.abs(prediction-reference), axis=0)
    refinement = np.max(np.abs(refined-prediction), axis=0)
    assert np.all(errors < budgets), errors
    assert np.all(refinement < budgets), refinement
    rigid_position = force*times**2/(2*(10.+truth[0]))
    rigid_rms = float(np.sqrt(np.mean((rigid_position-reference[:, 0])**2)))
    coupled_rms = float(np.sqrt(np.mean((prediction[:, 0]-reference[:, 0])**2)))
    assert rigid_rms > resolution and coupled_rms < resolution, "no resolved prediction improvement"
    return {
        "scope": "STEP 12 SYNTHETIC COUPLED SLOSH INCREMENT", "status": "PASS",
        "qualification_status": "NOT RUN", "measured_validation_status": "UNAVAILABLE",
        "software": collect_software_baseline(),
        "source": "Closed-form two-mass linear spring/damper, independently prescribed synthetic parameters",
        "model": {"bus_mass_kg": 10., "bus_inertia_kg_m2": np.eye(3),
                  "axis_body": [1., 0., 0.], "lever_arm_body_m": [0., 0., 0.],
                  "displacement_limit_m": .05, "mu_m3_s2": 0., "initial_bus_r_v": [0.]*6,
                  "q_body_to_gcrf": [1., 0., 0., 0.], "omega_body_rad_s": [0.]*3},
        "parameter_names": ["modal_mass_kg", "clamped_frequency_rad_s", "clamped_damping_ratio"],
        "synthetic_parameters": truth, "fitted_parameters": fit.x,
        "calibration": {"time_s": calibration_times, "initial_displacement_m": .025,
                        "force_n": 0., "columns": ["bus_position_m", "relative_displacement_m"],
                        "observations": calibration[:, [0, 2]], "observable_scale_m": resolution,
                        "max_residual_m": float(np.max(np.abs(fit.fun))*resolution),
                        "scaled_sensitivity_singular_values": singular_values,
                        "condition_number": float(singular_values[0]/singular_values[-1]),
                        "relative_only_singular_values": relative_singular_values,
                        "fit_bounds": [[.5, 1., .01], [4., 5., .3]], "fit_start": [1.5, 2.7, .08]},
        "held_out": {"time_s": times, "initial_displacement_m": displacement, "force_n": force,
                     "columns": ["bus_position_m", "bus_velocity_m_s", "relative_m", "relative_m_s"],
                     "reference": reference, "prediction": prediction, "budgets": budgets,
                     "max_errors": errors, "refinement_errors": refinement,
                     "rigid_bus_position_rms_m": rigid_rms, "coupled_bus_position_rms_m": coupled_rms},
        "solver": {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12, "max_step_s": .1},
        "refinement_solver": {"method": "Radau", "rtol": 1e-12, "atol": 1e-12, "max_step_s": .025},
        "selection": "Coupled model resolves this synthetic observable; retain rigid flight baseline pending measured evidence",
        "limits": ["Constant masses and clamped modal coefficients; bounded linear spring/damper analogue",
                   "No draining tanks, fluid free-surface prediction, flexibility, recovery or deployment loads",
                   "Local identifiability requires both bus and relative motion; not a global uniqueness or uncertainty claim",
                   "No measured calibration, held-out flight validation or mission qualification"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file, never overwritten")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(), indent=2, allow_nan=False, default=lambda value: value.tolist())+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
