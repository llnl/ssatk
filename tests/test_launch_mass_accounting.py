"""Step-4 references: volume quadrature, finite differences and conservation."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from ssatk.launch.vehicles import assemble_stage_stack
from ssatk.satellites import Component, SpacecraftBody, Tank
from ssatk.propagators_6dof import Spacecraft
from ssatk.accelerations_6dof import SpacecraftManeuverAccel
from ssatk.propagators_6dof.sixdof import _body_inertia_model


def cylinder(mass):
    return Tank(name="first", propellant_mass=mass, residual_mass=min(1.0, mass),
                dry_mass=2.0, position_body=[0.5, -0.2, 1.0],
                inertia=np.diag([.2, .3, .4]),
                cylinder_radius=0.4, propellant_density=100.0,
                propellant_base_body=[0.1, 0.2, -1.0],
                propellant_axis_body=[1.0, 2.0, 3.0])


@pytest.mark.parametrize("mass", [0.0, 1.0, 8.0, 20.0])
def test_cylinder_stack_matches_independent_volume_quadrature(mass):
    """Gauss volume quadrature: 1e-11 kg, m and kg m² absolute."""
    tank = cylinder(mass)
    payload = SpacecraftBody(name="payload", mass=5.0, inertia=np.diag([2., 3., 4.]),
                             center_of_mass=[-1., 0., 2.])
    stage = SpacecraftBody(name="stage", mass=7.0, inertia=np.diag([4., 5., 6.]),
                           center_of_mass=[0., 1., -2.], tanks=(tank,))
    body = assemble_stage_stack(payload, [stage])
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    e1 = np.cross(axis, [0., 0., 1.]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    nodes, weights = leggauss(5)
    height = mass / (100. * np.pi * 0.4**2)
    points = [payload.center_of_mass, stage.center_of_mass, tank.position_body]
    masses = [5., 7., 2.]
    for nr, wr in zip(nodes, weights):
        radius = (nr + 1.) * .2
        for nz, wz in zip(nodes, weights):
            z = (nz + 1.) * height / 2.
            for theta in np.arange(16) * (2. * np.pi / 16):
                points.append(np.array([.1, .2, -1.]) + axis*z
                              + radius*(e1*np.cos(theta) + e2*np.sin(theta)))
                masses.append(100. * radius * .2 * wr * height/2. * wz * 2.*np.pi/16.)
    points, masses = np.array(points), np.array(masses)
    center = np.sum(masses[:, None]*points, axis=0) / masses.sum()
    offsets = points - center
    inertia = np.diag([6.2, 8.3, 10.4]) + sum(
        m*(np.dot(d, d)*np.eye(3) - np.outer(d, d)) for m, d in zip(masses, offsets))
    np.testing.assert_allclose(body.current_mass, masses.sum(), atol=1e-11, rtol=0)
    np.testing.assert_allclose(body.current_center_of_mass, center, atol=1e-11, rtol=0)
    np.testing.assert_allclose(body.current_inertia, inertia, atol=1e-11, rtol=0)
    assert np.linalg.eigvalsh(body.current_inertia).min() > 0


def test_depletion_derivative_and_continuity_against_finite_difference():
    """Differences: 2e-8 m²; continuity 1e-7 m, 2e-6 kg m² for 1e-7 kg."""
    body = SpacecraftBody(name="stage", mass=7., inertia=np.eye(3), tanks=(cylinder(20.),))
    spacecraft = Spacecraft(r=[7e6, 0, 0], v=[0, 0, 0], body=body)
    derivative = _body_inertia_model(spacecraft, tank_name="first").inertia_rate
    for mass in (2., 8., 19.):
        step = 1e-4
        plus = body.with_tank_propellant_mass("first", mass+step)
        minus = body.with_tank_propellant_mass("first", mass-step)
        expected = (plus.current_inertia-minus.current_inertia)/(2*step)
        actual = derivative(0, None, None, None, None, mass=9.+mass, mass_rate=1.)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
        current = body.with_tank_propellant_mass("first", mass)
        neighbor = body.with_tank_propellant_mass("first", mass+1e-7)
        np.testing.assert_allclose(current.current_center_of_mass, neighbor.current_center_of_mass, atol=1e-7, rtol=0)
        np.testing.assert_allclose(current.current_inertia, neighbor.current_inertia, atol=2e-6, rtol=0)
    assert body.tanks[0].propellant_mass == 20.  # Trial states leave the source intact.


def test_detachment_conserves_mass_first_moment_and_full_tensor():
    """Retained + detached about common origin: 1e-12 kg, kg m, kg m²."""
    payload = SpacecraftBody(name="payload", mass=5., inertia=np.eye(3))
    stage = SpacecraftBody(name="first", mass=7., inertia=np.eye(3), tanks=(cylinder(8.),))
    fairing = Component(name="fairing", mass=3., position_body=[1., -2., 4.])
    before = assemble_stage_stack(payload, [stage], structures=[fairing])
    retained = assemble_stage_stack(payload)
    assert before.current_mass == retained.current_mass + stage.current_mass + fairing.mass
    np.testing.assert_allclose(before.current_mass*before.current_center_of_mass,
        retained.current_mass*retained.current_center_of_mass
        + stage.current_mass*stage.current_center_of_mass + fairing.mass*fairing.position_body,
        atol=1e-12, rtol=0)
    def about_origin(body):
        c = body.current_center_of_mass
        return body.current_inertia + body.current_mass*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
    np.testing.assert_allclose(about_origin(before),
        about_origin(retained)+about_origin(stage)+fairing.inertia_about(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cylindrical_reserve", [False, True])
def test_named_tank_burn_preserves_residual_and_other_stage_inventory(cylindrical_reserve):
    """Rocket equation and mass conservation: 1e-8 m/s and kg absolute."""
    body = SpacecraftBody(name="stage", mass=10., inertia=np.eye(3), tanks=(
        Tank(name="feed", propellant_mass=.02, residual_mass=.005),
        Tank(name="reserve", propellant_mass=1., residual_mass=.1,
             **(dict(cylinder_radius=.4, propellant_density=100.,
                     propellant_base_body=[0., 0., 2.]) if cylindrical_reserve else {}))))
    spacecraft = Spacecraft(r=[7e6, 0, 0], v=[0, 0, 0], body=body)
    burn = SpacecraftManeuverAccel(10., frame="gcrf", direction=[1,0,0], isp=100., tank_name="feed")
    trajectory = spacecraft.propagate(times=[0., 4.], models=[burn], mu=0.)
    final = trajectory.spacecraft(body=body, tank_name="feed")
    np.testing.assert_allclose([tank.propellant_mass for tank in final.body.tanks], [.005, 1.], atol=1e-8, rtol=0)
    expected = 100.*9.80665*np.log(body.current_mass/(body.current_mass-.015))
    np.testing.assert_allclose(trajectory.v[-1], [expected,0,0], atol=1e-8, rtol=0)


def test_proportional_depletion_preserves_each_unusable_residual():
    """Mass conservation at the combined floor: 1e-8 kg and m/s absolute."""
    body = SpacecraftBody(name="stage", mass=10., inertia=np.eye(3), tanks=(
        Tank(name="a", propellant_mass=.02, residual_mass=.005),
        Tank(name="b", propellant_mass=.04, residual_mass=.01),
        Tank(name="residual", propellant_mass=1., residual_mass=1.,
             cylinder_radius=.4, propellant_density=100., propellant_base_body=[0., 0., 2.])))
    spacecraft = Spacecraft(r=[7e6,0,0], v=[0,0,0], body=body)
    burn = SpacecraftManeuverAccel(10., frame="gcrf", direction=[1,0,0], isp=100.)
    trajectory = spacecraft.propagate(times=[0., 8.], models=[burn], mu=0.)
    final = trajectory.spacecraft(body=body)
    np.testing.assert_allclose([t.propellant_mass for t in final.body.tanks], [.005,.01,1.], atol=1e-8, rtol=0)
    expected = 100.*9.80665*np.log(11.06/11.015)
    np.testing.assert_allclose(trajectory.v[-1], [expected,0,0], atol=1e-8, rtol=0)
