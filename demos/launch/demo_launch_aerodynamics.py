"""Retain bounded synthetic launch-weather and load references, not flight data."""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.time import Time

from ssatk.coordinates.attitude import quaternion_from_matrix
from ssatk.coordinates.earth_fixed import itrf_to_gcrf_eop
from ssatk.launch import LaunchWeatherProfile, LaunchAtmosphere, AerodynamicCoefficientTable, LaunchAerodynamics
from ssatk.launch.preflight import collect_software_baseline
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody

GALLERY_INCLUDE = False


def run_reference():
    """Check loads at three prescribed states against ideal-gas and body-axis formulas."""
    epoch = float(Time("2020-05-30T19:22:00", scale="utc").gps)
    gas = dict(gas_constant=287.05, gamma=1.4, sutherland=(1.716e-5, 273.15, 110.4))
    coverage = dict(valid_time_gps=(epoch-60, epoch+60), latitude_range=(-1, 1), longitude_range=(-1, 1))
    lower = LaunchWeatherProfile([0, 2000], [100000, 80000], [300, 280], [[5, 0, 0], [7, 0, 0]],
                                 source="Synthetic lower p=100000-10h, T=300-.01h, east wind=5+.001h", **gas, **coverage)
    upper = LaunchWeatherProfile([1000, 3000], [90000, 80000], [280, 276], [[10, 0, 0], [10, 0, 0]],
                                 source="Synthetic sampled upper p=95000-5h, T=282-.002h, east wind=10", **gas, **coverage)
    atmosphere = LaunchAtmosphere(lower, upper=upper, transition=(1000, 2000))
    angles = np.array([-.2, 0, .2])
    force = np.array([[-.3, 0, -2*angle] for angle in angles])
    moment = np.array([[0, -.1*angle, 0] for angle in angles])
    table = AerodynamicCoefficientTable({"alpha": angles}, force, moment,
                                        source="Synthetic C_F=(-.3,0,-2 alpha), C_M=(0,-.1 alpha,0); Mach/beta/command independent")
    coefficient_to_body = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    model = LaunchAerodynamics(atmosphere, table, reference_area=2, reference_lengths=[1, 3, 1],
                               reynolds_length=3, reference_position_body=[0, 0, 0],
                               coefficient_to_body=coefficient_to_body)
    body = SpacecraftBody(name="synthetic aerodynamic reference", mass=100, inertia=np.diag([20, 30, 40]))
    records, force_errors, moment_errors = [], [], []
    for height in (500., 1500., 2500.):
        position = itrf_to_gcrf_eop([[6378137.+height, 0, 0]], [epoch])[0]
        # At lon=lat=0, ENU columns are +Y, +Z, +X in ITRF.
        body_axes = itrf_to_gcrf_eop(np.eye(3)[[1, 2, 0]], [epoch]*3).T
        q = quaternion_from_matrix(body_axes)
        air = atmosphere.sample(epoch, position)
        angle, speed = .1, 100.
        air_body = coefficient_to_body @ (speed*np.array([np.cos(angle), 0, np.sin(angle)]))
        craft = Spacecraft(r=position, v=air["atmosphere_velocity_gcrf_m_s"] + body_axes @ air_body,
                           t=epoch, q=q, body=body)
        result = model.loads(t=craft.t, r=craft.r, v=craft.v, q=craft.q, omega=craft.omega, spacecraft=craft)
        weight = np.clip((height-1000)/1000, 0, 1)
        pressure = (1-weight)*(100000-10*height) + weight*(95000-5*height)
        temperature = (1-weight)*(300-.01*height) + weight*(282-.002*height)
        dynamic_pressure = .5*pressure/(287.05*temperature)*speed**2
        expected_force = dynamic_pressure*2*np.array([.2, 0, -.3])
        expected_moment = dynamic_pressure*2*np.array([0, -.03, 0])
        force_errors.append(float(np.max(np.abs(result["force_body_n"]-expected_force))))
        moment_errors.append(float(np.max(np.abs(result["torque_center_body_n_m"]-expected_moment))))
        records.append({"position_gcrf_m": craft.r, "velocity_gcrf_m_s": craft.v,
                        "q_body_to_gcrf": craft.q, "omega_body_rad_s": craft.omega, "loads": result})
    assert max(force_errors) < 1e-6, "closed-form force in N"
    assert max(moment_errors) < 1e-6, "closed-form moment in N m"
    return {"scope": "STEP 7 SYNTHETIC WEATHER AND LOAD REFERENCE", "status": "PASS",
            "qualification_status": "NOT RUN", "software": collect_software_baseline(),
            "inputs": {"epoch_gps_s": epoch, "gas": gas, "coverage": coverage,
                       "lower": {"height_m": [0, 2000], "pressure_pa": [100000, 80000],
                                 "temperature_k": [300, 280], "wind_enu_m_s": [[5, 0, 0], [7, 0, 0]]},
                       "upper": {"height_m": [1000, 3000], "pressure_pa": [90000, 80000],
                                 "temperature_k": [280, 276], "wind_enu_m_s": [[10, 0, 0], [10, 0, 0]]},
                       "transition_m": [1000, 2000], "alpha_knots_rad": angles,
                       "force_coefficients": force, "moment_coefficients": moment,
                       "reference_area_m2": 2, "reference_lengths_m": [1, 3, 1],
                       "reynolds_length_m": 3, "reference_position_body_m": [0, 0, 0],
                       "coefficient_to_body": coefficient_to_body,
                       "mass_kg": 100, "inertia_kg_m2": [20, 30, 40]},
            "assumptions": ["Synthetic weather and coefficients; no measured launch inputs",
                            "Common ideal dry air and Sutherland viscosity; linear p/T/wind overlap",
                            "Steady coefficients; prescribed states, no trajectory propagation",
                            "No damping, plume, rarefaction, slosh, events or mission qualification"],
            "maximum_force_error_n": max(force_errors), "maximum_moment_error_n_m": max(moment_errors),
            "tolerances": {"force_n": 1e-6, "moment_n_m": 1e-6}, "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten.")
    args = parser.parse_args(argv)
    record = run_reference()
    encoded = json.dumps(record, indent=2, allow_nan=False,
                         default=lambda value: value.tolist()) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
