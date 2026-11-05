"""Continuous actuator and sampled launch control against analytic references."""

import numpy as np
from scipy.integrate import quad

from ssatk.accelerations_6dof import SpacecraftManeuverAccel
from ssatk.coordinates.attitude import quaternion_from_matrix
from ssatk.launch import LaunchActuator, LaunchControl, LaunchEvent, LaunchPhase, propagate_launch
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank
from ssatk.navigation.sensors import CartesianSensor


def test_slew_limited_lag_matches_piecewise_closed_form():
    """Clipped targets and analytic ramp/exponential branches: 2e-14 rad."""
    a = LaunchActuator(initial=[0., 0., 0.], lower=-3., upper=3., rate=2.,
                       lag=[.5, .5, 0.], units=["rad"]*3, source="synthetic")
    a.command(0., [1., 4., -5.])
    for t in [0., .1, 1., 1.25, 2., 10., .5]:
        expected = [1.-np.exp(-2*t), 2*t if t <= 1 else 3.-np.exp(-2*(t-1)), -min(2*t, 3.)]
        np.testing.assert_allclose(a.value(t), expected, rtol=0, atol=2e-14)
    a.command(2., [-3., -3., 3.])
    before = np.array([1.-np.exp(-4.), 3.-np.exp(-2.), -3.])
    np.testing.assert_allclose(a.value(2.), before, rtol=0, atol=2e-14)
    np.testing.assert_allclose(a.value(2.1), before+[-.2, -.2, .2], rtol=0, atol=2e-14)


def test_delayed_sampled_torque_matches_analytic_spin_and_quaternion():
    """Delayed exponential torque on isotropic I=2: 2e-9 rad/s and quaternion."""
    results = []
    for grid in ([0., 2.], np.linspace(0., 2., 41)):
        actuator = LaunchActuator(initial=[0.], lower=-2., upper=2., rate=100., lag=.3,
                                  units=["N m"], source="synthetic")
        def sensor(truth):
            # A known constant measurement bias drives an observable torque;
            # the estimator has no access to the true angular state.
            return {"value": 1., "sample_position_m": truth.r.copy()}
        def estimate(t, packets, prior):
            return {"value": packets[-1]["measurement"]["value"]}
        control = LaunchControl(sensor=sensor, estimator=estimate,
            guidance=lambda t,e:e["value"], controller=lambda t,e,g:[g], actuator=actuator,
            estimate={"value": 0.}, sensor_period=.5, control_period=.25,
            sensor_latency=.2, command_latency=.1)
        craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.],
                           body=SpacecraftBody(name="synthetic", mass=10., inertia=2*np.eye(3)))
        result = propagate_launch(craft, grid, [LaunchPhase("flight", propagation={
            "torque": lambda t,*args:np.array([0.,0.,actuator.value(t)[0]])})],
            initial_phase="flight", ground_condition=lambda t,y:y[0]-6e6,
            control=control, mu=0., rtol=1e-11, atol=1e-12, max_step=.05)
        dt, lag = 2.-.35, .3
        rate = (dt-lag*(1.-np.exp(-dt/lag)))/2
        angle = (.5*dt**2-lag*dt+lag**2*(1.-np.exp(-dt/lag)))/2
        np.testing.assert_allclose(result.state.omega, [0.,0.,rate], rtol=0, atol=2e-9)
        np.testing.assert_allclose(result.state.q, [np.cos(angle/2),0.,0.,np.sin(angle/2)], rtol=0, atol=2e-9)
        # Independent integral of torque also constrains angular impulse (N m s).
        impulse = quad(lambda t:1.-np.exp(-(t-.35)/lag), .35, 2., epsabs=1e-12)[0]
        np.testing.assert_allclose(2*result.state.omega[2], impulse, rtol=0, atol=4e-9)
        results.append((result, control))
    np.testing.assert_allclose(results[0][0].state.q, results[1][0].state.q, rtol=0, atol=1e-12)
    np.testing.assert_allclose([h["time_gps_s"] for h in results[0][1].history],
                               [h["time_gps_s"] for h in results[1][1].history], rtol=0, atol=0.)


def test_off_center_thrust_matches_rotated_force_and_cross_product():
    """Independent rotated force and lever arm at two tank masses: 1e-12 N m."""
    q = quaternion_from_matrix(np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]))
    for inventory in (5., 2.):
        body = SpacecraftBody(name="synthetic", mass=10., inertia=np.eye(3), tanks=(
            Tank(name="feed", propellant_mass=inventory, position_body=[0.,0.,2.]),))
        craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], q=q, body=body)
        burn = SpacecraftManeuverAccel([3.,4.,5.], frame="gcrf", position_body=[0.,0.,-1.], isp=100.)
        force_body = np.array([4.,-3.,5.])
        arm = np.array([0.,0.,-1.-2*inventory/(10+inventory)])
        np.testing.assert_allclose(burn.torque(craft), np.cross(arm, force_body), rtol=0, atol=1e-12)
        np.testing.assert_allclose(burn(craft), np.array([3.,4.,5.])/(10+inventory), rtol=0, atol=1e-12)
        np.testing.assert_allclose(burn.mass_flow_rate(craft), np.sqrt(50)/(100*9.80665), rtol=0, atol=1e-15)
        # With the engine off, moment is zero even where an orbit frame is undefined.
        inactive = SpacecraftManeuverAccel(5., frame="ntw", position_body=[0.,0.,-1.], start=1.)
        np.testing.assert_allclose(inactive.torque(craft), np.zeros(3), rtol=0, atol=0.)


def test_same_epoch_phase_resets_precede_sensor_and_preserve_transport():
    """Constant-velocity reset chain and scheduled transport jump: 1e-12 m/s."""
    actuator = LaunchActuator(initial=[0.], lower=-2., upper=2., rate=2., lag=0.,
                              units=["kg m/s"], source="synthetic transport boundary")
    control = LaunchControl(sensor=lambda s:{"velocity": s.v.copy()},
        estimator=lambda t,p,e:p[-1]["measurement"], guidance=lambda t,e:1.,
        controller=lambda t,e,g:[g], actuator=actuator, estimate={}, sensor_period=.5, control_period=.5)
    def reset(craft):
        craft.r[1] += 2.
        return craft
    craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], body=SpacecraftBody(name="synthetic", mass=10., inertia=np.eye(3)))
    phases = [LaunchPhase("a", events=(LaunchEvent("first",lambda t,y:0.,"b"),)),
              LaunchPhase("b", events=(LaunchEvent("second",lambda t,y:0.,"c",reset=reset),)),
              LaunchPhase("c", momentum_transport=lambda s:[actuator.target[0],0.,0.])]
    result = propagate_launch(craft, [0.,1.], phases, initial_phase="a",
                             ground_condition=lambda t,y:y[0]-6e6, control=control, mu=0.)
    np.testing.assert_allclose(result.state.v, [.1,0.,0.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.state.r, [7e6+.1,2.,0.], rtol=0, atol=1e-8)
    np.testing.assert_allclose(control.history[0]["acquired"][0]["measurement"]["velocity"], [0.,0.,0.], rtol=0, atol=0.)
    assert control.history[0]["phase"] == "c"
    assert [e["event"] for e in result.events[:3]] == ["first", "second", "__control"]
    # An initial scheduled jump also preserves the caller's source state.
    initial = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], body=craft.body)
    direct_actuator = LaunchActuator(initial=[0.], lower=-2., upper=2., rate=2., lag=0.,
                                     units=["kg m/s"], source="synthetic transport boundary")
    direct_control = LaunchControl(sensor=lambda s:{}, estimator=lambda t,p,e:e,
        guidance=lambda t,e:1., controller=lambda t,e,g:[g], actuator=direct_actuator,
        estimate={}, sensor_period=.5, control_period=.5)
    direct = propagate_launch(initial, [0.,1.], [LaunchPhase("flight",
        momentum_transport=lambda s:[direct_actuator.target[0],0.,0.])], initial_phase="flight",
        ground_condition=lambda t,y:y[0]-6e6, control=direct_control, mu=0.)
    np.testing.assert_allclose(initial.v, np.zeros(3), rtol=0, atol=0.)
    np.testing.assert_allclose(direct.state.v, [.1,0.,0.], rtol=0, atol=1e-12)


def test_absolute_gps_measurements_match_biased_constant_velocity_reference():
    """Sampled position with known bias and delayed delivery: 1e-7 m."""
    epoch = 1_400_000_000.
    bias = np.array([.1,-.2,.3])
    sensor = CartesianSensor(np.zeros((3,3)), bias=bias, rng=np.random.default_rng(4))
    actuator = LaunchActuator(initial=[0.], lower=-1., upper=1., rate=1., lag=.1,
                              units=["N m"], source="synthetic")
    def measure(s):
        return {"value": sensor.measure(s.r, s.t).value}
    control = LaunchControl(sensor=measure, estimator=lambda t,p,e:p[-1]["measurement"],
        guidance=lambda t,e:0., controller=lambda t,e,g:[g], actuator=actuator,
        estimate={"value": np.zeros(3)}, sensor_period=.1, control_period=.3,
        sensor_latency=.02, command_latency=.03)
    r0, v = np.array([7e6,0.,0.]), np.array([1.,2.,3.])
    craft = Spacecraft(r=r0, v=v, t=epoch, body=SpacecraftBody(name="synthetic",mass=10.,inertia=np.eye(3)))
    result = propagate_launch(craft, [epoch,epoch+1.], [LaunchPhase("flight")],
        initial_phase="flight", ground_condition=lambda t,y:y[0]-6e6, control=control, mu=0., max_step=.1)
    for entry in control.history:
        for p in entry["acquired"]:
            expected = r0+v*(p["sample_time_gps_s"]-epoch)+bias
            np.testing.assert_allclose(p["measurement"]["value"], expected, rtol=0, atol=1e-7)
        for p in entry["delivered"]:
            np.testing.assert_allclose(entry["time_gps_s"], p["delivery_time_gps_s"], rtol=0, atol=0.)
    np.testing.assert_allclose(result.state.r, r0+v, rtol=0, atol=1e-7)


def test_nominal_gimbal_tracking_matches_commanded_equilibrium():
    """Known zero-rate pitch equilibrium: 2e-3 rad/rad s-1, mass 1e-8 kg."""
    from demos.launch.demo_launch_guidance import run_reference
    record = run_reference()
    final = record["truth"][-1]
    pitch = 2*np.arctan2(final["q_body_to_gcrf"][-1,2], final["q_body_to_gcrf"][-1,0])
    np.testing.assert_allclose(pitch, .12, rtol=0, atol=.002)
    np.testing.assert_allclose(final["omega_body_rad_s"][-1], np.zeros(3), rtol=0, atol=.002)
    np.testing.assert_allclose(final["mass_kg"][-1], 105.-1500/(300*9.80665), rtol=0, atol=1e-8)
