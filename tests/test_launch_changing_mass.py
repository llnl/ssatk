"""Step-5 closed forms, finite differences and open-system momentum balances."""

import numpy as np
import pytest
from scipy.integrate import quad

from demos.launch.demo_launch_changing_mass import run_reference
from ssatk.accelerations_6dof import thruster_force_torque
from ssatk.coordinates.attitude import rotate_vector
from ssatk.propagators_6dof import propagate_6dof
from ssatk.satellites import SpacecraftBody, Tank, Thruster


@pytest.mark.parametrize("tank_name", [None, "feed"])
def test_center_rate_matches_finite_difference_of_mass_and_first_moment(tank_name):
    """Central differences at 1e-4 kg: 2e-10 m/kg and 2e-10 m absolute."""
    body = SpacecraftBody(name="dry", mass=10., inertia=np.eye(3),
                          center_of_mass=[.2, .3, 2.], tanks=(
        Tank(name="feed", propellant_mass=20., residual_mass=1., cylinder_radius=.4,
             propellant_density=100., propellant_base_body=[1., 0., 0.],
             propellant_axis_body=[1., 2., 3.]),
        Tank(name="reserve", propellant_mass=4., residual_mass=.5, position_body=[-1., 2., 0.])))
    step = 1e-4
    if tank_name is None:
        plus, minus = (body.with_propellant_mass(body.propellant_mass + sign*step) for sign in (1., -1.))
    else:
        plus, minus = (body.with_tank_propellant_mass(tank_name, 20. + sign*step) for sign in (1., -1.))
    reference = (plus.current_center_of_mass - minus.current_center_of_mass) / (2.*step)
    np.testing.assert_allclose(body.center_of_mass_mass_derivative(tank_name), reference, atol=2e-10, rtol=0)
    first_moment_reference = (plus.tanks[0].mass_first_moment - minus.tanks[0].mass_first_moment)
    first_moment_reference /= plus.tanks[0].propellant_mass - minus.tanks[0].propellant_mass
    np.testing.assert_allclose(body.tanks[0].mass_first_moment_derivative, first_moment_reference, atol=2e-10, rtol=0)


def test_fixed_mass_free_symmetric_top_matches_euler_solution_and_conservation():
    """Euler closed form: 1e-10 rad/s; momentum 1e-9 kg m2/s; energy 1e-10 J."""
    times = np.linspace(0., 5., 51)
    inertia = np.diag([2., 2., 3.])
    initial = [.2, 0., .3]
    trajectory = propagate_6dof(r0=[0., 0., 0.], v0=[1., 2., 3.], times=times,
                                omega0=initial, inertia=inertia, mu=0., rtol=1e-11)
    frequency = (3. - 2.) / 2. * .3
    expected = np.column_stack([.2*np.cos(frequency*times), .2*np.sin(frequency*times),
                                np.full(times.size, .3)])
    np.testing.assert_allclose(trajectory.omega, expected, atol=1e-10, rtol=0)
    momentum = np.array([rotate_vector(q, inertia @ omega) for q, omega in zip(trajectory.q, trajectory.omega)])
    np.testing.assert_allclose(momentum, np.tile(inertia @ initial, (len(times), 1)), atol=1e-9, rtol=0)
    energy = .5*np.einsum("ni,ij,nj->n", trajectory.omega, inertia, trajectory.omega)
    np.testing.assert_allclose(energy, .5*np.dot(initial, inertia @ initial), atol=1e-10, rtol=0)
    np.testing.assert_allclose(trajectory.r, times[:, None]*[1., 2., 3.], atol=1e-10, rtol=0)


def test_point_nozzle_flux_matches_analytic_spin_and_signed_momentum_transport():
    """I=kM and co-rotating point exhaust: 1e-10 rad/s, 1e-9 kg m2/s."""
    times = np.linspace(0., 2., 21)
    coefficient, nozzle_radius, mass0, mass_rate, spin0 = 2., 1., 10., -1., .2

    def inertia(t, r, v, q, omega, *, mass):
        return coefficient * mass * np.eye(3)

    def rate(t, r, v, q, omega, *, mass, mass_rate):
        return coefficient * mass_rate * np.eye(3)

    def flux(t, r, v, q, omega, *, mass, mass_rate):
        return [0., 0., mass_rate * nozzle_radius**2 * omega[2]]

    trajectory = propagate_6dof(r0=[0., 0., 0.], v0=[0., 0., 0.], times=times,
                                omega0=[0., 0., spin0], mass0=mass0,
                                mass_flow_rate=lambda *args: -mass_rate, inertia=inertia,
                                inertia_rate=rate, angular_momentum_flux=flux, mu=0., rtol=1e-11)
    exponent = nozzle_radius**2 / coefficient - 1.
    expected = spin0 * ((mass0 + mass_rate*times) / mass0)**exponent
    np.testing.assert_allclose(trajectory.omega[:, 2], expected, atol=1e-10, rtol=0)
    expelled = quad(lambda t: mass_rate * nozzle_radius**2 * spin0 *
                    ((mass0 + mass_rate*t) / mass0)**exponent, 0., times[-1], epsabs=1e-12)[0]
    np.testing.assert_allclose(coefficient*trajectory.mass[-1]*trajectory.omega[-1, 2]
                               - coefficient*mass0*spin0, expelled, atol=1e-9, rtol=0)


@pytest.fixture(scope="module")
def axial_record():
    return run_reference()


def test_axial_drain_matches_closed_form_centroid_motion_and_spin(axial_record):
    """Rational cylindrical CoM and log solution: 1e-9 m/m s-1, 1e-10 rad/s/q."""
    samples = axial_record["samples"]
    times = np.array([sample["elapsed_s"] for sample in samples])
    mass = 30. - times
    capacity = 100.*np.pi*.4**2
    constant = 10.*2. + 10.**2/(2.*capacity)
    center = mass/(2.*capacity) - 10./capacity + constant/mass
    np.testing.assert_allclose([sample["center_body_m"][2] for sample in samples], center, atol=1e-11, rtol=0)
    velocity0 = center[0]/30.
    logarithm = np.log(mass/30.)
    velocity = velocity0 + (-100. + 1./capacity)*logarithm + constant*(1./mass**2 - 1./30.**2)
    position = (velocity0*times + (100. - 1./capacity)*(mass*logarithm - mass + 30.)
                + constant*(1./mass - 1./30.) - constant*times/30.**2)
    np.testing.assert_allclose([sample["velocity_gcrf_m_s"][2] for sample in samples], velocity, atol=1e-9, rtol=0)
    np.testing.assert_allclose([sample["position_gcrf_m"][2] for sample in samples], position, atol=1e-9, rtol=0)
    inertia = 3. + (mass - 10.)*.4**2/2.
    spin = .2*inertia[0]/inertia
    angle = .2*inertia[0]/(-.4**2/2.) * np.log(inertia/inertia[0])
    quaternion = np.column_stack([np.cos(angle/2.), np.zeros(times.size), np.zeros(times.size), np.sin(angle/2.)])
    np.testing.assert_allclose([sample["omega_body_rad_s"][2] for sample in samples], spin, atol=1e-10, rtol=0)
    np.testing.assert_allclose([sample["q_body_to_gcrf"] for sample in samples], quaternion, atol=1e-10, rtol=0)


def test_axial_drain_closes_mass_impulse_and_expelled_linear_momentum(axial_record):
    """Dry/plug momentum sum and Gauss exhaust integral: 1e-8 kg m/s; mass 1e-9 kg; impulse 1e-8 N s."""
    capacity = 100.*np.pi*.4**2
    constant = 20. + 100./(2.*capacity)
    center0 = 30./(2.*capacity) - 10./capacity + constant/30.

    def inward_momentum_rate(t):
        mass = 30. - t
        center_rate = -(1./(2.*capacity) - constant/mass**2)
        velocity = (center0/30. + (-100. + 1./capacity)*np.log(mass/30.)
                    + constant*(1./mass**2 - 1./30.**2))
        return -(velocity - center_rate - 100.)

    reference = quad(inward_momentum_rate, 0., 2., epsabs=1e-11, epsrel=1e-13)[0]
    samples = axial_record["samples"]
    mass = 30. - np.array([sample["elapsed_s"] for sample in samples])
    center_rate = -(1./(2.*capacity) - constant/mass**2)
    body_velocity = np.array([sample["velocity_gcrf_m_s"] for sample in samples])
    body_velocity[:, 2] -= center_rate
    retained = mass[:, None] * body_velocity
    retained[:, 2] -= (mass - 10.) / capacity
    np.testing.assert_allclose([sample["linear_momentum_gcrf_kg_m_s"] for sample in samples],
                               retained, atol=1e-8, rtol=0)
    measured = np.array(samples[-1]["linear_momentum_gcrf_kg_m_s"]) - samples[0]["linear_momentum_gcrf_kg_m_s"]
    np.testing.assert_allclose(measured, [0., 0., reference], atol=1e-8, rtol=0)
    np.testing.assert_allclose([sample["mass_kg"] for sample in samples],
                               30. - np.array([sample["elapsed_s"] for sample in samples]), atol=1e-9, rtol=0)
    np.testing.assert_allclose(axial_record["thrust_impulse_N_s"],
                               100.*axial_record["consumed_mass_kg"], atol=1e-8, rtol=0)


def test_thrust_moment_uses_moving_center_against_analytic_lever_arm():
    """Cylinder first moment and point nozzle cross product: 1e-12 N m."""
    capacity = 100.*np.pi*.4**2
    thruster = Thruster(thrust=20., direction_body=[1., 0., 0.], position_body=[0., 0., -1.])
    for propellant in (20., 10., 1.):
        body = SpacecraftBody(name="dry", mass=10., inertia=np.eye(3), center_of_mass=[0., 0., 2.],
                              tanks=(Tank(name="feed", propellant_mass=propellant, residual_mass=1.,
                                          cylinder_radius=.4, propellant_density=100.,
                                          propellant_base_body=[0., 0., 0.]),))
        center = (20. + propellant**2/(2.*capacity)) / (10. + propellant)
        force, torque = thruster_force_torque([thruster], center_of_mass=body.current_center_of_mass)
        np.testing.assert_allclose(force, [20., 0., 0.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(torque, [0., -20.*(1. + center), 0.], atol=1e-12, rtol=0)
