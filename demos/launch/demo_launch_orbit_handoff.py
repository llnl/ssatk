"""Synthetic upper-stage ascent, exact cutoff and native SSAPy orbital coast."""

import argparse
import json
from pathlib import Path

import numpy as np
import ssapy
from astropy.time import Time

from ssatk.accelerations_6dof import SpacecraftManeuverAccel, make_ssapy_perturbation_acceleration
from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.coordinates import gcrf_to_llh
from ssatk.coordinates.satellite_frames import transform_from_gcrf
from ssatk.launch import LaunchEvent, LaunchPhase, propagate_launch
from ssatk.launch.preflight import collect_software_baseline
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank

GALLERY_INCLUDE = False


def run_reference(*, perturbed=False):
    """Compare launch/coast states with native SSAPy under identical forces.

    This is a specified terminal-ascent segment, not a pad-to-orbit mission.
    Point propellant at the fixed COM gives B=0 and constant dry inertia.
    """
    epoch = float(Time("2026-07-20T00:00:00", scale="utc").gps)
    duration, thrust, isp = 40.125, 1000., 300.
    body = SpacecraftBody(name="synthetic upper stage", mass=150., inertia=np.diag([40., 50., 60.]),
                          reference_area=1., tanks=(Tank(name="upper", propellant_mass=50., residual_mass=5.),))
    initial = Spacecraft(r=[6.8e6, 0., 0.], v=[200., 7500., 900.], t=epoch, body=body,
                         area=2.5, cd=2.2, cr=1.3, center_of_pressure=[0., 0., .2])
    options = dict(earth_degree=2, earth_order=0, include_moon=True, include_sun=True,
                   include_planets=False, include_drag=True, include_solar_radiation=True,
                   include_earth_radiation=True)
    perturbations = make_ssapy_perturbation_acceleration(**options).accels if perturbed else ()
    burn = SpacecraftManeuverAccel(thrust, frame="inertial", direction=[0., 1., 0.], isp=isp, tank_name="upper")
    phases = (
        LaunchPhase("powered", (*perturbations, burn),
                    (LaunchEvent("cutoff", lambda t, y: t-epoch-duration, "coast", 1),)),
        LaunchPhase("coast", perturbations),
    )
    solver = dict(mu=EARTH_MU, rtol=1e-11, atol=1e-9, max_step=2.)
    def clearance(t, y):
        return np.linalg.norm(y[:3])-EARTH_RADIUS

    # A scheduled cutoff is an exact endpoint, avoiding a quantized time-only root.
    ascent_times = epoch+np.r_[np.arange(0., 40., 5.), duration]
    ascent = propagate_launch(initial, ascent_times, phases,
                              initial_phase="powered", ground_condition=clearance, **solver)
    insertion = ascent.state
    orbit = insertion.to_orbit(mu=EARTH_MU)
    # The native coast needs central gravity; the launch runner already adds it.
    native_terms = [ssapy.AccelKepler(mu=EARTH_MU), *(model.accel for model in perturbations)]
    acceleration = ssapy.AccelSum(native_terms)
    coast_solver = dict(method="DOP853", rtol=1e-11, atol=1e-9, max_step=2.)
    propagator = ssapy.SciPyPropagator(acceleration, ode_kwargs=coast_solver)
    times = insertion.t+np.arange(0., 601., 30.)
    r, v = ssapy.rv(orbit, times, propagator=propagator)
    overlap = propagate_launch(insertion, times, (LaunchPhase("coast", perturbations),),
                               initial_phase="coast", ground_condition=clearance, **solver)
    arc = overlap.arcs[0][1]
    position_error = float(np.max(np.linalg.norm(arc.r-r, axis=1)))
    velocity_error = float(np.max(np.linalg.norm(arc.v-v, axis=1)))
    assert position_error < 1e-3 and velocity_error < 1e-6, "matched-force overlap in m and m/s"
    assert orbit.t == insertion.t == epoch+duration
    np.testing.assert_array_equal(orbit.r, insertion.r)
    np.testing.assert_array_equal(orbit.v, insertion.v)
    np.testing.assert_array_equal(r[0], insertion.r)
    np.testing.assert_array_equal(v[0], insertion.v)
    expected_mass = 200.-thrust*duration/(isp*9.80665)
    assert abs(orbit.propkw["mass"]-expected_mass) < 1e-10, "constant-Isp mass balance in kg"
    assert orbit.propkw == dict(mass=insertion.mass, area=2.5, CD=2.2, CR=1.3)
    np.testing.assert_array_equal(insertion.center_of_pressure, initial.center_of_pressure)
    cutoff = ascent.events[0]
    np.testing.assert_array_equal(cutoff["pre_state"]["velocity_gcrf_m_s"], insertion.v)
    state_accel = -EARTH_MU*insertion.r/np.linalg.norm(insertion.r)**3
    for model in perturbations:
        state_accel += model(insertion)
    assembly_error = float(np.linalg.norm(acceleration(orbit.r, orbit.v, orbit.t, **orbit.propkw)-state_accel))
    assert assembly_error < 1e-12, "single-count force assembly in m/s^2"
    analytic = None
    if not perturbed:
        analytic_r, analytic_v = ssapy.rv(orbit, times, propagator=ssapy.KeplerianPropagator())
        analytic = dict(position_m=float(np.max(np.linalg.norm(r-analytic_r, axis=1))),
                        velocity_m_s=float(np.max(np.linalg.norm(v-analytic_v, axis=1))))
        assert analytic["position_m"] < 1e-3 and analytic["velocity_m_s"] < 1e-6

    # Replace only the identical boundary sample; no interpolation of the handoff.
    history_t = np.concatenate([*(a.t[:-1] for _, a in ascent.arcs), times])
    history_r = np.concatenate([*(a.r[:-1] for _, a in ascent.arcs), r])
    history_v = np.concatenate([*(a.v[:-1] for _, a in ascent.arcs), v])
    history_mass = np.concatenate([*(a.mass[:-1] for _, a in ascent.arcs), np.full(len(times), insertion.mass)])
    longitude, latitude, height = gcrf_to_llh(history_r, history_t, allow_predicted=False)
    return {
        "scope": "STEP 11 SYNTHETIC UPPER-STAGE ASCENT TO COAST", "status": "PASS",
        "qualification_status": "NOT RUN", "software": collect_software_baseline(),
        "inputs": dict(epoch_gps_s=epoch, position_gcrf_m=initial.r, velocity_gcrf_m_s=initial.v,
                       initial_mass_kg=200., thrust_n=thrust, isp_s=isp, burn_duration_s=duration,
                       ascent_solver=solver, coast_solver=coast_solver),
        "force_configuration": {"central_mu_m3_s2": EARTH_MU,
                                "perturbations": options if perturbed else {},
                                "native_terms": [type(term).__name__ for term in native_terms],
                                "coast_properties": orbit.propkw},
        "insertion": {**ascent.events[-1]["post_state"], "frame": "GCRF", "mu_m3_s2": orbit.mu,
                      "element_type": "osculating", "semimajor_axis_m": float(orbit.a),
                      "eccentricity": float(orbit.e), "inclination_rad": float(orbit.i),
                      "equinoctial_a_hx_hy_ex_ey_lv": np.asarray(orbit.equinoctialElements),
                      "specific_energy_m2_s2": float(orbit.energy),
                      "specific_angular_momentum_m2_s": orbit.angularMomentum,
                      "periapsis_radius_m": float(orbit.a*(1-orbit.e)),
                      "apoapsis_radius_m": float(orbit.a*(1+orbit.e))},
        "comparison": dict(overlap_position_m=position_error, overlap_velocity_m_s=velocity_error,
                           acceleration_assembly_m_s2=assembly_error, kepler_reference=analytic,
                           position_budget_m=1e-3, velocity_budget_m_s=1e-6),
        "overlap_residuals": {"time_gps_s": times, "position_gcrf_m": arc.r-r, "velocity_gcrf_m_s": arc.v-v,
                              "position_rtn_m": np.array([transform_from_gcrf(dr, "rtn", r=ri, v=vi)
                                                          for dr, ri, vi in zip(arc.r-r, r, v)]),
                              "velocity_rtn_m_s": np.array([transform_from_gcrf(dv, "rtn", r=ri, v=vi)
                                                            for dv, ri, vi in zip(arc.v-v, r, v)])},
        "events": ascent.events,
        "history": dict(time_gps_s=history_t, position_gcrf_m=history_r, velocity_gcrf_m_s=history_v,
                        mass_kg=history_mass, longitude_deg=longitude, latitude_deg=latitude, height_wgs84_m=height),
        "limits": ["Specified upper-stage initial state; no pad-to-orbit mission or flight calibration",
                   "Fixed-COM point inventory, prescribed inertial thrust, no fluid angular momentum",
                   "Coast is translational with constant mass/area/CD/CR; retained body and attitude are not evolved by Orbit",
                   "Optional native gravity/drag/radiation laws use their own declared frame/environment approximations",
                   "Ground track uses the shared packaged-EOP IAU 2006/2000A and WGS84 route",
                   "Osculating elements are not mean elements or an SGP4/TLE product"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file, never overwritten")
    parser.add_argument("--perturbed", action="store_true", help="Include degree-2 zonal gravity, Sun/Moon, drag and radiation")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(perturbed=args.perturbed), indent=2, allow_nan=False,
                         default=lambda value: value.tolist())+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
