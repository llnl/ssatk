"""Synthetic pad/rail, burnout, separation, restart and insertion-event ledger."""

import argparse
import json
from pathlib import Path

import numpy as np

from demos.launch.demo_launch_changing_mass import AxialDrainManeuver
from ssatk.coordinates.attitude import quaternion_from_matrix, rotate_vector
from ssatk.launch import LaunchEvent, LaunchPhase, LaunchGuide, propagate_launch, separate_launch_body
from ssatk.launch.preflight import collect_software_baseline
from ssatk.launch.vehicles import assemble_stage_stack
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank

GALLERY_INCLUDE = False


def run_reference():
    """Two constant-Isp burns and conservative staging against the rocket equation."""
    radius, exhaust_speed = 6378137., 300.*9.80665
    upper = SpacecraftBody(name="synthetic upper", mass=10., inertia=np.diag([20.,20.,5.]),
        center_of_mass=[0.,0.,2.], tanks=(Tank(name="upper", propellant_mass=5., residual_mass=1., position_body=[0.,0.,2.]),))
    lower = SpacecraftBody(name="synthetic lower", mass=20., inertia=np.diag([40.,40.,10.]),
        center_of_mass=[0.,0.,-1.], tanks=(Tank(name="lower", propellant_mass=50., residual_mass=2., position_body=[0.,0.,-1.]),))
    rotation = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
    stack = assemble_stage_stack(upper, [lower])
    state = Spacecraft(r=[radius,0.,0.], v=[0.,0.,0.], q=quaternion_from_matrix(rotation), body=stack)
    def frame(t):
        return [radius,0.,0.], np.zeros(3), np.zeros(3), rotation, np.zeros(3), np.zeros(3)
    pad, rail = LaunchGuide(frame, length=0.), LaunchGuide(frame, length=10.)
    first = AxialDrainManeuver(10000., frame="body", direction=[0.,0.,1.], isp=300., tank_name="lower")
    second = AxialDrainManeuver(1000., frame="body", direction=[0.,0.,1.], isp=300., tank_name="upper")
    def first_transport(craft):
        return rotate_vector(craft.q, -10000./exhaust_speed*(lower.tanks[0].position_body-craft.body.current_center_of_mass))
    def second_transport(craft):
        return rotate_vector(craft.q, -1000./exhaust_speed*(upper.tanks[0].position_body-craft.body.current_center_of_mass))
    def split(craft):
        remaining = next(t.propellant_mass for t in craft.body.tanks if t.name == "lower")
        return separate_launch_body(craft, (upper, lower.with_tank_propellant_mass("lower", remaining)))
    phases = [
        LaunchPhase("pad", events=(LaunchEvent("liftoff_release", lambda t,y:t-1., "rail", 1),), constraint=pad),
        LaunchPhase("rail", (first,), (LaunchEvent("rail_exit", rail.exit_condition, "first_powered", 1),),
                    constraint=rail, momentum_transport=first_transport),
        LaunchPhase("first_powered", (first,), depletion_phase="burnout", momentum_transport=first_transport),
        LaunchPhase("burnout", events=(LaunchEvent("burnout", lambda t,y:0., "separation"),)),
        LaunchPhase("separation", events=(LaunchEvent("stage_separation", lambda t,y:0., "restart_coast", reset=split),)),
        LaunchPhase("restart_coast", events=(LaunchEvent("restart", lambda t,y:t-20., "second_powered", 1),)),
        LaunchPhase("second_powered", (second,), (LaunchEvent("cutoff", lambda t,y:t-30., "coast", 1),),
                    momentum_transport=second_transport),
        LaunchPhase("coast", events=(LaunchEvent("insertion_boundary", lambda t,y:y[0]-radius-100000., "inserted", 1),)),
        LaunchPhase("inserted")]
    solver = dict(mu=0., rtol=1e-10, atol=1e-11, max_step=.25)
    results = [propagate_launch(state, grid, phases, initial_phase="pad",
                  ground_condition=lambda t,y:y[0]-radius, **solver)
               for grid in ([0.,70.], np.linspace(0.,70.,141))]
    coarse, result = results
    expected = exhaust_speed*(np.log(85./37.)+np.log(15./(15.-10.*1000./exhaust_speed)))
    velocity_error = float(abs(result.state.v[0]-expected))
    event_time_error = float(max(abs(a["time_gps_s"]-b["time_gps_s"])
                                for a,b in zip(coarse.events, result.events)))
    assert velocity_error < 1e-6, "two-burn rocket equation in m/s"
    assert event_time_error < 1e-7, "output-grid event-time independence in s"
    assert [e["event"] for e in coarse.events] == [e["event"] for e in result.events]
    assert result.state.r[0] > radius+100000., "ascent crosses the orbital burnup boundary"
    truth = [{"phase": phase, "time_gps_s": arc.t, "position_gcrf_m": arc.r,
              "velocity_gcrf_m_s": arc.v, "q_body_to_gcrf": arc.q,
              "omega_body_rad_s": arc.omega, "mass_kg": arc.mass}
             for phase,arc in result.arcs]
    return {"scope": "STEP 8 SYNTHETIC PHASE AND RESET REFERENCE", "status": "PASS",
            "qualification_status": "NOT RUN", "software": collect_software_baseline(),
            "inputs": {"epoch_gps_s": 0., "radius_m": radius, "initial_mass_kg": 85.,
                       "first_thrust_n": 10000., "second_thrust_n": 1000., "isp_s": 300.,
                       "release_time_s": 1., "rail_length_m": 10., "restart_time_s": 20.,
                       "cutoff_time_s": 30., "insertion_boundary_height_m": 100000.,
                       "duration_s": 70., "solver": solver},
            "assumptions": ["Synthetic rigid bodies and spherical ground; no gravity, atmosphere or Earth rotation",
                            "Ideal hold-down and frictionless rail; prescribed force direction",
                            "Axial point inventory exits at its body-fixed tank position; explicit M*v-B transport",
                            "Step 5 centroid acceleration correction; zero axial exhaust moment; cutoff before separation",
                            "Insertion labels a 100 km boundary, not an achieved orbit or qualified mission"],
            "rocket_equation_velocity_m_s": expected, "velocity_error_m_s": velocity_error,
            "output_grid_event_time_error_s": event_time_error,
            "tolerances": {"velocity_m_s": 1e-6, "event_time_s": 1e-7},
            "termination": result.termination, "events": result.events, "truth": truth}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file, never overwritten")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(), indent=2, allow_nan=False,
                         default=lambda value: value.tolist()) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
