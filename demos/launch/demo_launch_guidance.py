"""Synthetic measured-attitude tracking through a delayed finite-bandwidth gimbal."""

import argparse
import json
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import SpacecraftAttitudePD, SpacecraftManeuverAccel
from ssatk.coordinates.attitude import quaternion_conjugate, quaternion_from_matrix, rotate_vector
from ssatk.launch import LaunchActuator, LaunchControl, LaunchPhase, propagate_launch
from ssatk.launch.preflight import collect_software_baseline
from ssatk.navigation.sensors import CartesianSensor
from ssatk.navigation.star_tracker import fit_star_attitude
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank

GALLERY_INCLUDE = False


def run_reference(*, solver=None):
    """Pitch tracking with synthetic direction/gyro measurements, not telemetry."""
    duration, force, arm, target_angle = 15., 100., 2., .12
    target_q = np.array([np.cos(target_angle/2), 0., np.sin(target_angle/2), 0.])
    covariance = np.diag([1e-10]*12)
    bias = np.r_[np.zeros(9), 0., .001, 0.]
    solver = {**dict(mu=0., max_step=.1, rtol=1e-9, atol=1e-11), **(solver or {})}
    results = []
    for times in ([0., duration], np.linspace(0., duration, 151)):
        sensor = CartesianSensor(covariance, bias=bias, rng=np.random.default_rng(9))
        actuator = LaunchActuator(initial=[0.], lower=[-.06], upper=[.06], rate=[.15], lag=[.1],
                                  units=["rad"], source="declared synthetic pitch-gimbal model")
        pd = SpacecraftAttitudePD(kp=40., kd=30.)

        def measure(truth):
            directions = np.array([rotate_vector(quaternion_conjugate(truth.q), row) for row in np.eye(3)])
            sample = sensor.measure(np.r_[directions.ravel(), truth.omega], truth.t)
            return {"value": sample.value, "valid": sample.valid, "covariance": sample.covariance}

        def estimate(time, packets, previous):
            for packet in packets:
                measurement = packet["measurement"]
                if measurement["valid"]:
                    values = measurement["value"]-bias
                    rotation = fit_star_attitude(values[:9].reshape(3,3), np.eye(3))
                    previous = {"q": quaternion_from_matrix(rotation), "omega": values[9:],
                                "sample_time_gps_s": packet["sample_time_gps_s"],
                                "measurement_covariance": measurement["covariance"]}
            return previous

        def controller(time, estimated, reference):
            pd.q_target = reference["q"]
            torque = pd.torque(t=time, r=np.zeros(3), v=np.zeros(3),
                              q=estimated["q"], omega=estimated["omega"])
            # A nozzle below the COM produces negative pitch torque for +gimbal.
            return [-np.arcsin(np.clip(torque[1]/(arm*force), -1., 1.))]

        initial_q = np.array([np.cos(-.1/2),0.,np.sin(-.1/2),0.])
        control = LaunchControl(sensor=measure, estimator=estimate,
            guidance=lambda t,e:{"q": target_q.copy()}, controller=controller, actuator=actuator,
            estimate={"q": initial_q.copy(), "omega": np.zeros(3), "sample_time_gps_s": 0.,
                      "measurement_covariance": covariance.copy()},
            sensor_period=.1, control_period=.1, sensor_latency=.02, command_latency=.02)

        def thrust(time, *args):
            angle = actuator.value(time)[0]
            return force*np.array([np.sin(angle),0.,np.cos(angle)])

        body = SpacecraftBody(name="synthetic pitch plant", mass=100., inertia=np.diag([20.,20.,10.]),
                              tanks=(Tank(name="feed", propellant_mass=5., position_body=[0.,0.,0.]),))
        craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], q=initial_q, body=body)
        burn = SpacecraftManeuverAccel(thrust, frame="body", position_body=[0.,0.,-arm], isp=300., tank_name="feed")
        result = propagate_launch(craft, times, [LaunchPhase("powered", (burn,))], initial_phase="powered",
            ground_condition=lambda t,y:y[0]-6e6, control=control, **solver)
        error = abs(2*np.arctan2(result.state.q[2], result.state.q[0])-target_angle)
        assert error < .002, "synthetic terminal pitch tracking in rad"
        assert abs(result.state.omega[1]) < .002, "synthetic terminal pitch rate in rad/s"
        expected_mass = 105.-duration*force/(300.*9.80665)
        assert abs(result.state.mass-expected_mass) < 1e-8, "constant-Isp inventory in kg"
        actual = np.array([h["actual"][0] for h in control.history])
        assert np.max(np.abs(actual)) <= .06, "gimbal hard stop in rad"
        results.append((result, control, error))
    coarse, result = results[0][0], results[1][0]
    sampling_error = float(np.max(np.abs(coarse.state.q-result.state.q)))
    assert sampling_error < 1e-10, "quaternion output-grid independence"
    assert [h["time_gps_s"] for h in results[0][1].history] == [h["time_gps_s"] for h in results[1][1].history]
    return {"scope": "STEP 9 SYNTHETIC CLOSED-LOOP PITCH REFERENCE", "status": "PASS",
            "qualification_status": "NOT RUN", "software": collect_software_baseline(),
            "inputs": {"epoch_gps_s": 0., "duration_s": duration, "force_n": force, "arm_m": arm,
                       "target_pitch_rad": target_angle, "initial_pitch_rad": -.1, "isp_s": 300.,
                       "sensor_period_s": .1, "control_period_s": .1, "sensor_latency_s": .02,
                       "command_latency_s": .02, "gimbal_bounds_rad": [-.06,.06],
                       "gimbal_rate_rad_s": .15, "gimbal_lag_s": .1, "kp": 40., "kd": 30.,
                       "sensor_covariance": covariance, "sensor_bias": bias, "seed": 9, "solver": solver},
            "assumptions": ["Synthetic orthogonal inertial direction observations plus a body-rate gyro at COM",
                            "Independent Gaussian direction-component and gyro noise; declared bias removed by estimator",
                            "Equal-weight Wahba attitude fit and held last measurement, not an attitude EKF",
                            "Only pitch is controlled; no gravity, weather, Earth rotation, slosh or flexibility",
                            "External body wrench with imposed point-inventory removal at COM; B=0 and fixed dry inertia",
                            "No nozzle/internal-fluid transport inferred; zero relative fluid angular momentum",
                            "Finite-bandwidth gimbal orientation only; no nozzle reaction inertia or measured calibration",
                            "Prior supplied explicitly; sensor alone sees truth; estimator covariance is measurement covariance"],
            "terminal_pitch_error_rad": results[1][2], "terminal_pitch_rate_rad_s": float(result.state.omega[1]),
            "mass_error_kg": float(result.state.mass-expected_mass), "output_grid_quaternion_error": sampling_error,
            "tolerances": {"pitch_rad": .002, "pitch_rate_rad_s": .002, "mass_kg": 1e-8, "quaternion": 1e-10},
            "control": results[1][1].history, "events": result.events,
            "truth": [{"phase": phase, "time_gps_s": arc.t, "position_gcrf_m": arc.r,
                       "velocity_gcrf_m_s": arc.v, "q_body_to_gcrf": arc.q,
                       "omega_body_rad_s": arc.omega, "mass_kg": arc.mass,
                       "nfev": arc.nfev, "integration_diagnostics": arc.integration_diagnostics}
                      for phase,arc in result.arcs]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file, never overwritten")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(), indent=2, allow_nan=False, default=lambda v:v.tolist())+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
