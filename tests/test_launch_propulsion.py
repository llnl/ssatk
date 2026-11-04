"""Step-6 independent flow against the analytic constant-flow rocket equation."""

import numpy as np
import pytest

from ssatk.accelerations_6dof import SpacecraftManeuverAccel, ThrustCurve
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank
from ssatk.constants import STANDARD_GRAVITY
from ssatk.accelerations_6dof.thrust import load_thrust_curve_csv
from ssatk.propagators_6dof import propagate_spacecraft_segments


def test_documented_flow_matches_mass_balance_and_rocket_equation():
    """R1: 1e-9 kg inventory, 1e-9 m/s velocity; specified synthetic motor."""
    body = SpacecraftBody(name="reference", mass=10., inertia=np.eye(3), tanks=(
        Tank(name="feed", propellant_mass=5., residual_mass=1.),
        Tank(name="reserve", propellant_mass=2., residual_mass=.5)))
    craft = Spacecraft(r=[0., 0., 0.], v=[0., 0., 0.], body=body)
    burn = SpacecraftManeuverAccel(10., frame="gcrf", direction=[1., 0., 0.],
                                  propellant_flow_rate=ThrustCurve([0., 2.], [.1, .1]),
                                  tank_name="feed", start=0., stop=2.)
    times = np.linspace(0., 2., 21)
    trajectory = craft.propagate(times=times, models=[burn], mu=0., rtol=1e-11, atol=1e-12)
    expected_mass = 17. - .1*times
    np.testing.assert_allclose(trajectory.mass, expected_mass, atol=1e-9, rtol=0)
    np.testing.assert_allclose(trajectory.v[:, 0], 100.*np.log(17./expected_mass), atol=1e-9, rtol=0)
    final = trajectory.spacecraft(body=body, tank_name="feed")
    np.testing.assert_allclose([tank.propellant_mass for tank in final.body.tanks],
                               [4.8, 2.], atol=1e-9, rtol=0)


@pytest.mark.parametrize("inferred_flow", [False, True])
def test_pressure_and_throttle_match_reference_nozzle_rocket_equation(inferred_flow):
    """R1: fixed nozzle F=F_ref+(p_ref-p)*A; 1e-9 N/kg/m s-1 absolute."""
    flow = {"isp": 100./STANDARD_GRAVITY} if inferred_flow else {"propellant_flow_rate": .1}
    burn = SpacecraftManeuverAccel(10., frame="gcrf", direction=[1., 0., 0.],
                                  throttle=.5, throttle_range=(.3, 1.),
                                  reference_pressure=100000., ambient_pressure=50000.,
                                  nozzle_exit_area=1e-5, pressure_range=(0., 100000.), **flow)
    craft = Spacecraft(r=[0., 0., 0.], v=[0., 0., 0.], mass=17., inertia=np.eye(3))
    np.testing.assert_allclose(burn(craft)*17., [5.5, 0., 0.], atol=1e-9, rtol=0)
    np.testing.assert_allclose(burn.mass_flow_rate(craft), .05, atol=1e-9, rtol=0)
    times = np.linspace(0., 2., 21)
    trajectory = craft.propagate(times=times, models=[burn], mu=0., rtol=1e-11, atol=1e-12)
    mass = 17. - .05*times
    np.testing.assert_allclose(trajectory.mass, mass, atol=1e-9, rtol=0)
    np.testing.assert_allclose(trajectory.v[:, 0], 110.*np.log(17./mass), atol=1e-9, rtol=0)
    burn.ambient_pressure = 100000.
    np.testing.assert_allclose(burn(craft)*17., [5., 0., 0.], atol=1e-9, rtol=0)
    np.testing.assert_allclose(burn.mass_flow_rate(craft), .05, atol=1e-9, rtol=0)
    burn.ambient_pressure = 100001.
    with pytest.raises(ValueError, match="pressure_range"):
        burn(craft)
    burn.throttle = .2
    with pytest.raises(ValueError, match="throttle_range"):
        burn.mass_flow_rate(craft)
    burn.throttle = 0.
    np.testing.assert_allclose(burn(craft), [0., 0., 0.], atol=0., rtol=0)
    np.testing.assert_allclose(burn.mass_flow_rate(craft), 0., atol=0., rtol=0)


def test_csv_histories_interpolate_against_triangle_and_reject_extrapolation(tmp_path):
    """R1: triangle impulse 20 N s and flow integral .2 kg, tolerance 1e-12."""
    path = tmp_path / "specified_motor.csv"
    path.write_text("time_s,thrust_n,mass_flow_kg_s\n0,0,0\n2,10,.1\n4,0,0\n", encoding="utf-8")
    thrust = load_thrust_curve_csv(path, bounds_error=True)
    flow = load_thrust_curve_csv(path, thrust_column="mass_flow_kg_s", bounds_error=True)
    np.testing.assert_allclose(thrust.total_impulse, 20., atol=1e-12, rtol=0)
    np.testing.assert_allclose(flow.total_impulse, .2, atol=1e-12, rtol=0)
    np.testing.assert_allclose([thrust(1.), flow(3.)], [5., .05], atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="interpolation span"):
        thrust(4.01)


def test_ignition_cutoff_restart_segments_match_analytic_impulse_and_inventory():
    """R1/R4: 30 N s over two burns; 1e-9 kg and m/s constant-flow solution."""
    craft = Spacecraft(r=[0., 0., 0.], v=[0., 0., 0.], mass=17., inertia=np.eye(3))
    segments = []
    for start, stop, firing in [(0., 1., False), (1., 3., True), (3., 5., False), (5., 6., True)]:
        models = []
        if firing:
            models = [SpacecraftManeuverAccel(ThrustCurve([start, stop], [10., 10.], bounds_error=True),
                      frame="gcrf", direction=[1., 0., 0.], start=start, stop=stop,
                      propellant_flow_rate=ThrustCurve([start, stop], [.1, .1], bounds_error=True))]
        segments.append({"times": np.linspace(start, stop, 11), "models": models})
    trajectory = propagate_spacecraft_segments(craft, segments, mu=0., rtol=1e-11, atol=1e-12)
    burning_time = np.clip(trajectory.t - 1., 0., 2.) + np.clip(trajectory.t - 5., 0., 1.)
    mass = 17. - .1*burning_time
    np.testing.assert_allclose(trajectory.mass, mass, atol=1e-9, rtol=0)
    np.testing.assert_allclose(trajectory.v[:, 0], 100.*np.log(17./mass), atol=1e-9, rtol=0)
    from demos.launch.demo_launch_propulsion import run_reference

    record = run_reference()
    np.testing.assert_allclose(record["impulse_n_s"], 11., atol=1e-9, rtol=0)
    np.testing.assert_allclose(record["consumed_propellant_kg"], .1, atol=1e-9, rtol=0)
