"""Launch events against closed forms, rigid-body kinematics and conservation."""

import numpy as np
import pytest
from scipy.integrate import quad

from demos.launch.demo_launch_changing_mass import AxialDrainManeuver
from ssatk.accelerations_6dof import SpacecraftManeuverAccel
from ssatk.coordinates.attitude import quaternion_from_matrix, rotate_vector
from ssatk.launch.events import (LaunchEvent, LaunchPhase, LaunchGuide, LaunchReset,
    launch_ground_clearance, propagate_launch, separate_launch_body)
from ssatk.launch.vehicles import assemble_stage_stack
from ssatk.propagators_6dof import Spacecraft, sixdof_rhs
from ssatk.satellites import SpacecraftBody, Tank


def body(mass=10., **kwargs):
    return SpacecraftBody(name="synthetic", mass=mass, inertia=np.diag([2., 3., 4.]), **kwargs)


def test_two_child_separation_matches_rigid_motion_and_impulse():
    """Closed-form rotated child r/v and torque impulse: 1e-9 m, m/s, rad/s."""
    a = body(10., center_of_mass=[0., 0., 2.])
    b = body(20., center_of_mass=[0., 0., -1.])
    stack = assemble_stage_stack(a, [b])
    q = quaternion_from_matrix(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    craft = Spacecraft(r=[7e6, 20., 30.], v=[100., 200., 300.], q=q,
                       omega=[.2, -.3, .4], body=stack)
    # Equal/opposite impulses at a common point have zero system moment.
    impulse = np.array([3., 4., 5.])
    reset = separate_launch_body(craft, [a, b], impulses_body=[impulse, -impulse],
                                 positions_body=[[0., 0., 0.], [0., 0., 0.]])
    for child, part, j in zip((reset.state, *reset.children), (a, b), (impulse, -impulse)):
        offset = part.current_center_of_mass-stack.current_center_of_mass
        np.testing.assert_allclose(child.r, craft.r+rotate_vector(q, offset), atol=1e-9, rtol=0)
        np.testing.assert_allclose(child.v, craft.v+rotate_vector(q, np.cross(craft.omega, offset)+j/part.current_mass), atol=1e-9, rtol=0)
        expected = craft.omega+np.linalg.solve(part.current_inertia, np.cross(-part.current_center_of_mass, j))
        np.testing.assert_allclose(child.omega, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(sum(c.mass*c.v for c in (reset.state, *reset.children)),
                               craft.mass*craft.v, atol=1e-9, rtol=0)


def test_centroid_rate_mapping_matches_linear_momentum():
    """Moving reference-point velocities: 1e-10 m/s and kg m/s."""
    a, b = body(10., center_of_mass=[0., 0., 2.]), body(20., center_of_mass=[0., 0., -1.])
    craft = Spacecraft(r=[7e6, 0., 0.], v=[10., 20., 30.], body=assemble_stage_stack(a, [b]))
    rates = np.array([[0., 0., .03], [0., 0., -.015]])
    reset = separate_launch_body(craft, [a, b], center_rates_body=rates)
    for child, rate in zip((reset.state, *reset.children), rates):
        np.testing.assert_allclose(child.v, craft.v+rate, atol=1e-10, rtol=0)
    np.testing.assert_allclose(sum(c.mass*c.v for c in (reset.state, *reset.children)),
                               craft.mass*craft.v, atol=1e-10, rtol=0)


def test_draining_separation_matches_material_momentum_and_reference_rates():
    """Axial cylinder split: 1e-10 m/s and kg m/s; spin 1e-10 kg m2/s."""
    capacity = 100.*np.pi*.4**2
    upper = body(10., center_of_mass=[0., 0., 2.])
    lower = body(20., center_of_mass=[0., 0., -1.], tanks=(
        Tank(name="feed", propellant_mass=5., residual_mass=1., cylinder_radius=.4,
             propellant_density=100., propellant_base_body=[0., 0., -2.]),))
    stack = assemble_stage_stack(upper, [lower])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    craft = Spacecraft(r=[7e6, 0., 0.], v=[10., 20., 30.], q=quaternion_from_matrix(rotation),
                       omega=[0., 0., .2], body=stack)
    # Removing a layer at z=-2+m/k differentiates the first mass moment.
    surface = -2.+5./capacity
    parent_rate = np.array([0., 0., -(surface-stack.current_center_of_mass[2])/35.])
    rates = np.array([[0., 0., 0.], [0., 0., -(surface-lower.current_center_of_mass[2])/25.]])
    parent_b = -(np.array([0., 0., -2.])-stack.current_center_of_mass)
    child_b = np.array([np.zeros(3), -(np.array([0., 0., -2.])-lower.current_center_of_mass)])
    def split(state):
        return separate_launch_body(state, [upper, lower], center_rates_body=rates,
            parent_center_rate_body=parent_rate, momentum_transport_body=parent_b,
            child_momentum_transports_body=child_b)
    phases = [LaunchPhase("draining", events=(LaunchEvent("split", lambda t,y:0., "retained", reset=split),),
                          momentum_transport=rotation@parent_b), LaunchPhase("retained")]
    result = propagate_launch(craft, [0., 1.], phases, initial_phase="draining",
                              ground_condition=lambda t,y:y[0]-6e6, mu=0.)
    reset = split(craft)
    children = (reset.state, *reset.children)
    for child, rate in zip(children, rates):
        np.testing.assert_allclose(child.v, craft.v+rotation@(rate-parent_rate), atol=1e-10, rtol=0)
    reference = 35.*craft.v-rotation@parent_b
    measured = sum(c.mass*c.v-rotation@b for c, b in zip(children, child_b))
    np.testing.assert_allclose(measured, reference, atol=1e-10, rtol=0)
    spin = sum(rotation@(c.inertia@c.omega) for c in children)
    np.testing.assert_allclose(spin, rotation@(craft.inertia@craft.omega), atol=1e-10, rtol=0)
    event = result.events[0]
    np.testing.assert_allclose(event["children"][0]["material_momentum_gcrf_kg_m_s"],
                               children[1].mass*children[1].v-rotation@child_b[1], atol=1e-10, rtol=0)


def test_transport_enters_angular_momentum_about_common_origin():
    """Equal material bulk velocities conserve P/H: 1e-10 kg m/s and kg m2/s."""
    upper, lower = body(10., center_of_mass=[0., 0., 2.]), body(20., center_of_mass=[0., 0., -1.])
    craft = Spacecraft(r=[7e6, 0., 0.], v=[0., 0., 0.], body=assemble_stage_stack(upper, [lower]))
    transports = np.array([[4., 0., 0.], [-1., 0., 0.]])
    def split(state):
        children = tuple(Spacecraft(r=state.r+b.current_center_of_mass,
            v=np.array([-.1, 0., 0.])+transport/b.current_mass, body=b, t=state.t)
            for b, transport in zip((upper, lower), transports))
        return LaunchReset(children[0], children[1:], child_momentum_transports=(transports[1],))
    phases = [LaunchPhase("parent", events=(LaunchEvent("split", lambda t,y:0., "child", reset=split),),
                          momentum_transport=[3., 0., 0.]), LaunchPhase("child", momentum_transport=transports[0])]
    result = propagate_launch(craft, [0., 1.], phases, initial_phase="parent",
                              ground_condition=lambda t,y:y[0]-6e6, mu=0.)
    event = result.events[0]
    states = (event["post_state"], *event["children"])
    momentum = sum(s["material_momentum_gcrf_kg_m_s"] for s in states)
    angular = sum(np.cross(s["position_gcrf_m"]-craft.r, s["material_momentum_gcrf_kg_m_s"]) for s in states)
    np.testing.assert_allclose(momentum, [-3., 0., 0.], atol=1e-10, rtol=0)
    np.testing.assert_allclose(angular, [0., 0., 0.], atol=1e-10, rtol=0)


@pytest.mark.parametrize("samples", [2, 81])
def test_axial_ignition_cutoff_restart_and_depletion_against_material_balance(samples):
    """Cylinder closed forms: 1e-8 m/s, kg m/s; spin 1e-9 kg m2/s; roots 1e-7 s."""
    capacity = 100.*np.pi*.4**2
    constant = 20.+100./(2.*capacity)
    craft = Spacecraft(r=[7e6, 0., 0.], v=[0., 0., 0.], omega=[0., 0., .2],
        body=SpacecraftBody(name="axial", mass=10., center_of_mass=[0., 0., 2.],
            inertia=np.diag([4., 4., 3.]), tanks=(
                Tank(name="feed", propellant_mass=20., residual_mass=1., cylinder_radius=.4,
                     propellant_density=100., propellant_base_body=[0., 0., 0.]),)))
    burn = AxialDrainManeuver(100., direction=[0., 0., 1.], frame="body",
                             propellant_flow_rate=1., tank_name="feed")
    def transport(state):
        # One-sided s=-1 at the boundary, including the incoming depleted state.
        return rotate_vector(state.q, state.body.current_center_of_mass)
    phases = [LaunchPhase("coast0", events=(LaunchEvent("ignition", lambda t,y:t-1., "burn1", 1),)),
              LaunchPhase("burn1", (burn,), (LaunchEvent("cutoff", lambda t,y:t-3., "coast1", 1),),
                          momentum_transport=transport),
              LaunchPhase("coast1", events=(LaunchEvent("restart", lambda t,y:t-4., "burn2", 1),)),
              LaunchPhase("burn2", (burn,), depletion_phase="coast2", momentum_transport=transport),
              LaunchPhase("coast2")]
    result = propagate_launch(craft, np.linspace(0., 22., samples), phases, initial_phase="coast0",
        ground_condition=lambda t,y:y[0]-6e6, mu=0., angular_momentum_flux=np.zeros(3),
        rtol=1e-11, atol=1e-12, max_step=.2)
    np.testing.assert_allclose([e["time_gps_s"] for e in result.events], [1., 3., 4., 21., 22.], atol=1e-7, rtol=0)
    coast_velocity = 0.
    for start, end, ignition, cutoff in ((30., 28., result.events[0], result.events[1]),
                                        (28., 11., result.events[2], result.events[3])):
        def center(mass):
            return mass/(2.*capacity)-10./capacity+constant/mass
        initial_v = coast_velocity+center(start)/start
        def velocity(mass):
            return initial_v+(-100.+1./capacity)*np.log(mass/start)+constant*(1./mass**2-1./start**2)
        expected_pre = velocity(end)
        coast_velocity = expected_pre-center(end)/end
        np.testing.assert_allclose(ignition["post_state"]["velocity_gcrf_m_s"], [0., 0., initial_v], atol=1e-8, rtol=0)
        np.testing.assert_allclose(cutoff["pre_state"]["velocity_gcrf_m_s"], [0., 0., expected_pre], atol=1e-8, rtol=0)
        np.testing.assert_allclose(cutoff["post_state"]["velocity_gcrf_m_s"], [0., 0., coast_velocity], atol=1e-8, rtol=0)
        # Independent inward exhaust momentum quadrature, s=-1, u=-100 m/s.
        reference = quad(lambda m: -(velocity(m)+(1./(2.*capacity)-constant/m**2)-100.),
                         end, start, epsabs=1e-11, epsrel=1e-13)[0]
        measured = (cutoff["pre_state"]["material_momentum_gcrf_kg_m_s"]
                    - ignition["post_state"]["material_momentum_gcrf_kg_m_s"])
        np.testing.assert_allclose(measured, [0., 0., reference], atol=1e-8, rtol=0)
        for snapshot in (ignition["post_state"], cutoff["pre_state"]):
            mass = snapshot["mass_kg"]
            center_rate = -(1./(2.*capacity)-constant/mass**2)
            material = mass*(snapshot["velocity_gcrf_m_s"][2]-center_rate)-(mass-10.)/capacity
            np.testing.assert_allclose(snapshot["material_momentum_gcrf_kg_m_s"], [0., 0., material], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.state.v, [0., 0., coast_velocity], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.state.inertia@result.state.omega, craft.inertia@craft.omega, atol=1e-9, rtol=0)
    for event in result.events:
        np.testing.assert_allclose(event["pre_state"]["material_momentum_gcrf_kg_m_s"],
                                   event["post_state"]["material_momentum_gcrf_kg_m_s"], atol=1e-8, rtol=0)


def test_mutating_reset_preserves_pre_state_against_impulse_reference():
    """One N s on 10 kg: 0.1 m/s, 1e-12 m/s and kg m/s absolute."""
    craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], body=body())
    def impulse(state):
        state.v[0] += .1
        return LaunchReset(state, impulse_gcrf=[1.,0.,0.])
    phases = [LaunchPhase("a", events=(LaunchEvent("impulse", lambda t,y:t-1., None, 1, reset=impulse),))]
    result = propagate_launch(craft, [0.,2.], phases, initial_phase="a",
                              ground_condition=lambda t,y:y[0]-6e6, mu=0.)
    np.testing.assert_allclose(result.state.v, [.1,0.,0.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(craft.v, [0.,0.,0.], atol=1e-12, rtol=0)
    event = result.events[0]
    np.testing.assert_allclose(event["pre_state"]["velocity_gcrf_m_s"], [0.,0.,0.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(event["conservation"]["residuals"]["linear_momentum_kg_m_s"], [0.,0.,0.], atol=1e-12, rtol=0)


@pytest.mark.parametrize("samples", [2, 51])
def test_root_depletion_separation_restart_against_rocket_equation(samples):
    """Depletion 1e-7 s; two-stage rocket equation 1e-7 m/s; residual 1e-8 kg."""
    upper = body(10., tanks=(Tank(name="upper", propellant_mass=2., residual_mass=.5),))
    lower = body(20., tanks=(Tank(name="lower", propellant_mass=3., residual_mass=1.),))
    craft = Spacecraft(r=[7e6, 0., 0.], v=[0., 0., 0.], body=assemble_stage_stack(upper, [lower]))
    # c=100 m/s and mass flow = 1 kg/s.
    first = SpacecraftManeuverAccel(100., frame="gcrf", direction=[1., 0., 0.], isp=100./9.80665, tank_name="lower")
    second = SpacecraftManeuverAccel(100., frame="gcrf", direction=[1., 0., 0.], isp=100./9.80665, tank_name="upper")
    def split(state):
        depleted = lower.with_tank_propellant_mass("lower", state.body.tanks[1].propellant_mass)
        return separate_launch_body(state, [upper, depleted])
    modes = [LaunchPhase("first", (first,), depletion_phase="staging"),
             LaunchPhase("staging", events=(LaunchEvent("separation", lambda t,y: 0., "restart", reset=split),)),
             LaunchPhase("restart", events=(LaunchEvent("ignition", lambda t,y: (t-3.)*y[13]/12., "second", direction=1),)),
             LaunchPhase("second", (second,), depletion_phase="coast"), LaunchPhase("coast")]
    result = propagate_launch(craft, np.linspace(0., 6., samples), modes, initial_phase="first",
                              ground_condition=lambda t,y: y[0]-6e6, mu=0., max_step=.2, rtol=1e-11, atol=1e-12)
    expected = 100.*(np.log(35./33.)+np.log(12./10.5))
    np.testing.assert_allclose(result.state.v, [expected,0.,0.], atol=1e-7, rtol=0)
    np.testing.assert_allclose([e["time_gps_s"] for e in result.events], [2.,2.,3.,4.5,6.], atol=1e-7, rtol=0)
    assert [e["event"] for e in result.events] == ["__depletion:first", "separation", "ignition", "__depletion:second", "__duration"]
    np.testing.assert_allclose(result.state.body.tanks[0].propellant_mass, .5, atol=1e-8, rtol=0)
    assert len(result.events) == len({e["event"] for e in result.events})


def test_simultaneous_roots_use_priority_with_same_pre_root():
    """Constant velocity roots at 2 s independent of output grid: 1e-10 s/m."""
    craft = Spacecraft(r=[7e6, 0., 0.], v=[1., 0., 0.], body=body())
    events = (LaunchEvent("cutoff", lambda t,y: t-2., "coast", 1, 0),
              LaunchEvent("insertion", lambda t,y: y[0]-7000002., "inserted", 1, 1, 1e-8))
    phases = [LaunchPhase("powered", events=events), LaunchPhase("coast"), LaunchPhase("inserted")]
    results = [propagate_launch(craft, grid, phases, initial_phase="powered",
                 ground_condition=lambda t,y: y[0]-6e6, mu=0., max_step=.25)
               for grid in ([0.,4.], np.linspace(0.,4.,21))]
    for result in results:
        np.testing.assert_allclose([e["time_gps_s"] for e in result.events], [2.,2.,4.], atol=1e-8, rtol=0)
        assert [e["event"] for e in result.events] == ["cutoff", "insertion", "__duration"]
    np.testing.assert_allclose(results[0].state.r, results[1].state.r, atol=1e-10, rtol=0)


def test_pad_reactions_and_rail_exit_against_constant_acceleration():
    """Hold force and torque: 1e-12 N/N m; rail exit 1e-8 s, m, m/s."""
    rotation = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
    def frame(t):
        return [7e6,0.,0.], np.zeros(3), np.zeros(3), rotation, np.zeros(3), np.zeros(3)
    pad, rail = LaunchGuide(frame, length=0.), LaunchGuide(frame, length=4.)
    craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], q=quaternion_from_matrix(rotation), body=body())
    def acceleration(t,r,v,q,omega): return np.array([2.,3.,4.])
    def torque(t,r,v,q,omega): return np.array([1.,2.,3.])
    y = np.r_[craft.r,craft.v,craft.q,craft.omega,craft.mass]
    free = sixdof_rhs(0., y, inertia=craft.inertia, mu=0., acceleration=acceleration, torque=torque)
    reactions = pad.reactions(0., y, free, mass=10., inertia=craft.inertia)
    np.testing.assert_allclose(reactions["force_gcrf_n"], [-20.,-30.,-40.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(reactions["torque_body_n_m"], [-1.,-2.,-3.], atol=1e-12, rtol=0)
    options = dict(mu=0., acceleration=acceleration, torque=torque)
    phases = [LaunchPhase("pad", events=(LaunchEvent("liftoff", lambda t,y:t-1., "rail",1),), constraint=pad),
              LaunchPhase("rail", events=(LaunchEvent("rail_exit", rail.exit_condition, "free",1),), constraint=rail),
              LaunchPhase("free")]
    result = propagate_launch(craft, [0.,5.], phases, initial_phase="pad",
                              ground_condition=lambda t,y:y[0]-7e6, max_step=.25, **options)
    exit_state = result.events[1]["pre_state"]
    np.testing.assert_allclose(result.events[1]["time_gps_s"], 3., atol=1e-8, rtol=0)
    np.testing.assert_allclose(exit_state["position_gcrf_m"], [7000004.,0.,0.], atol=1e-8, rtol=0)
    np.testing.assert_allclose(exit_state["velocity_gcrf_m_s"], [4.,0.,0.], atol=1e-8, rtol=0)


def test_rotating_rail_reactions_against_centripetal_and_coriolis():
    """Analytic rotating line: 1e-10 N, 1e-9 m, 1e-10 m/s."""
    rate = .2
    def frame(t):
        c,s = np.cos(rate*t), np.sin(rate*t)
        rotation = np.array([[-s,0.,c], [c,0.,s], [0.,1.,0.]])
        return np.zeros(3), np.zeros(3), np.zeros(3), rotation, [0.,rate,0.], np.zeros(3)
    rail = LaunchGuide(frame, length=20.)
    rotation = frame(0.)[3]
    craft = Spacecraft(r=[2.,0.,0.], v=[3.,.4,0.], q=quaternion_from_matrix(rotation), omega=[0.,rate,0.], body=body())
    y = np.r_[craft.r,craft.v,craft.q,craft.omega,craft.mass]
    # Free acceleration -Omega² r keeps constant along-rail speed; support supplies Coriolis.
    free = sixdof_rhs(0., y, inertia=craft.inertia, mu=0., acceleration=lambda *args: np.array([-.08,0.,0.]))
    reaction = rail.reactions(0., y, free, mass=10., inertia=craft.inertia)
    np.testing.assert_allclose(reaction["force_gcrf_n"], [0.,12.,0.], atol=1e-10, rtol=0)
    def acceleration(t,r,v,q,omega):
        return -rate**2*r
    trajectory = craft.propagate(times=[0.,1.], mu=0., acceleration=acceleration,
                                 constraint=rail, max_step=.1, rtol=1e-11, atol=1e-12)
    axis = np.array([np.cos(rate), np.sin(rate),0.])
    np.testing.assert_allclose(trajectory.r[-1], 5.*axis, atol=1e-9, rtol=0)
    np.testing.assert_allclose(trajectory.v[-1], 3.*axis+5.*rate*np.array([-np.sin(rate),np.cos(rate),0.]), atol=1e-10, rtol=0)


def test_pad_release_from_rest_matches_free_constant_acceleration():
    """Direct liftoff after one-second hold: 1e-8 m, m/s and s."""
    pad = LaunchGuide(lambda t:([7e6,0.,0.], np.zeros(3), np.zeros(3),
                               np.eye(3), np.zeros(3), np.zeros(3)), length=0.)
    craft = Spacecraft(r=[7e6,0.,0.], v=[0.,0.,0.], body=body())
    phases = [LaunchPhase("pad", events=(LaunchEvent("release", lambda t,y:t-1., "free",1),), constraint=pad),
              LaunchPhase("free")]
    result = propagate_launch(craft, [0.,5.], phases, initial_phase="pad",
        ground_condition=lambda t,y:y[0]-7e6, mu=0., acceleration=lambda *args:np.array([2.,0.,0.]), max_step=.2)
    np.testing.assert_allclose(result.state.r, [7000016.,0.,0.], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.state.v, [8.,0.,0.], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.events[0]["time_gps_s"], 1., atol=1e-8, rtol=0)
    assert result.termination == "__duration"


def test_return_to_rail_base_preserves_root_state_and_records_contact():
    """Ballistic rail contact: sqrt(3) s, -2 sqrt(3) m/s, 1e-7 absolute."""
    rotation = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
    rail = LaunchGuide(lambda t:([7e6,0.,0.], np.zeros(3), np.zeros(3),
                                rotation, np.zeros(3), np.zeros(3)), length=10.)
    craft = Spacecraft(r=[7000003.,0.,0.], v=[0.,0.,0.], q=quaternion_from_matrix(rotation), body=body())
    result = propagate_launch(craft, [0.,5.], [LaunchPhase("rail", constraint=rail)], initial_phase="rail",
        ground_condition=lambda t,y:y[0]-6e6, mu=0., acceleration=lambda *args:np.array([-2.,0.,0.]), max_step=.1)
    np.testing.assert_allclose(result.state.t, np.sqrt(3.), atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.state.r, [7e6,0.,0.], atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.state.v, [-2.*np.sqrt(3.),0.,0.], atol=1e-7, rtol=0)
    assert result.termination == "__rail_bottom"
    assert result.events[-1]["conservation"]["status"] == "PASS"


def test_ground_contact_and_100km_crossing_have_separate_launch_policy():
    """Ballistic roots and constant velocity through 100 km: 1e-7 s and m."""
    craft = Spacecraft(r=[6378137.+10.,0.,0.], v=[0.,0.,0.], body=body())
    result = propagate_launch(craft, [0.,10.], [LaunchPhase("fall")], initial_phase="fall",
        ground_condition=lambda t,y:y[0]-6378137., mu=0., acceleration=lambda *args:np.array([-2.,0.,0.]), max_step=.2)
    np.testing.assert_allclose(result.state.t, np.sqrt(10.), atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.state.r[0], 6378137., atol=1e-7, rtol=0)
    assert result.termination == "__ground"
    craft = Spacecraft(r=[6378137.+10.,0.,0.], v=[2000.,0.,0.], body=body())
    phases = [LaunchPhase("ascent", events=(LaunchEvent("100km", lambda t,y:y[0]-6478137., "above", 1),)), LaunchPhase("above")]
    result = propagate_launch(craft, [0.,60.], phases, initial_phase="ascent",
        ground_condition=lambda t,y:y[0]-6378137., mu=0., max_step=1., acceleration=lambda *args:np.array([2.,0.,0.]))
    crossing = (np.sqrt(2000.**2+4.*99990.)-2000.)/2.
    np.testing.assert_allclose(result.events[0]["time_gps_s"], crossing, atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.state.r[0], 6501747., atol=1e-7, rtol=0)


def test_wgs84_clearance_against_equatorial_closed_form():
    """WGS84 a+h at the equator transformed by independent Astropy: 1e-5 m."""
    from astropy import units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from astropy.time import Time
    from ssatk.data import data_path
    from astropy.utils import iers
    epoch = Time("2020-05-30T19:22:00", scale="utc")
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        point = ITRS(CartesianRepresentation([6378137.+123.,0.,0.]*u.m), obstime=epoch).transform_to(GCRS(obstime=epoch))
    y = np.r_[point.cartesian.xyz.to_value(u.m), np.zeros(10)]
    np.testing.assert_allclose(launch_ground_clearance(23.)(float(epoch.gps), y), 100., atol=1e-5, rtol=0)


@pytest.mark.parametrize("acceleration", [-2., 2.])
def test_surface_roundoff_preserves_liftoff_and_fall(acceleration):
    """R1: signed 1 nm pad residual; constant acceleration, 1e-7 m/s and m."""
    ground = 7e6
    craft = Spacecraft(r=[ground-1e-9, 0., 0.], v=[0., 0., 0.], body=body())
    result = propagate_launch(craft, [0., 1.], [LaunchPhase("flight")], initial_phase="flight",
        ground_condition=lambda t,y:y[0]-ground, mu=0., max_step=.1,
        acceleration=lambda *args:np.array([acceleration, 0., 0.]))
    if acceleration < 0:
        assert result.termination == "__ground"
        np.testing.assert_allclose(result.state.v, [0., 0., 0.], atol=1e-7, rtol=0)
    else:
        assert result.termination == "__duration"
        np.testing.assert_allclose(result.state.r, [ground+1., 0., 0.], atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.state.v, [2., 0., 0.], atol=1e-7, rtol=0)


@pytest.mark.parametrize("initial_position, expected_time", [(0., 2.), (3., 0.)])
def test_flight_limit_stops_at_crossing_or_initial_violation(initial_position, expected_time):
    """R1: constant 1 m/s motion with 2 m limit; epoch/position within 1e-8 s/m."""
    craft = Spacecraft(r=[7e6+initial_position, 0., 0.], v=[1., 0., 0.], body=body())
    limit = LaunchEvent("position_limit", lambda t,y:7e6+2.-y[0], None, -1, trigger_below=True)
    result = propagate_launch(craft, [0., 4.], [LaunchPhase("flight", events=(limit,))],
        initial_phase="flight", ground_condition=lambda t,y:y[0]-6e6, mu=0., max_step=.1)
    assert result.termination == "position_limit"
    np.testing.assert_allclose(result.state.t, expected_time, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.state.r[0], 7e6+initial_position+expected_time, atol=1e-8, rtol=0)
