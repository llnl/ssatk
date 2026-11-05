"""R1/R4: two-mass mechanics, momentum and dissipated energy for coupled slosh."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from ssatk.coordinates.attitude import rotate_vector
from ssatk.propagators_6dof import SloshMode, propagate_6dof_extended, propagate_6dof_extended_variational


@pytest.mark.parametrize("method", ["DOP853", "RK45", "Radau"])
def test_axial_two_mass_motion_matches_closed_form(method):
    """R1: force on bus only, 2e-9 m and m/s against damped two-mass solution."""
    from demos.launch.demo_launch_slosh import _axial_reference

    times = np.linspace(0., 3., 61)
    result = propagate_6dof_extended(
        times=times, r0=[0, 0, 0], v0=[0, 0, 0], mu=0, inertia=np.eye(3),
        q0=[np.sqrt(.5), 0, 0, np.sqrt(.5)], bus_mass=10., coupled_slosh=True,
        slosh=SloshMode([1, 0, 0], 2., 3., .06, displacement0=.025, displacement_limit=.05),
        acceleration=lambda t, r, v, q, omega: [0., .012, 0.],
        method=method, rtol=1e-11, atol=1e-13)
    expected = _axial_reference(times, [2., 3., .06], .025, .12)
    actual = np.column_stack((result.trajectory.r[:, 1], result.trajectory.v[:, 1], result.slosh))
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
    np.testing.assert_allclose(result.trajectory.r[:, [0, 2]], 0., atol=2e-12, rtol=0)


@pytest.mark.parametrize("damping", [0., .12])
def test_rotating_offset_modes_conserve_momentum_and_account_for_heat(damping):
    """R4: Cartesian particles + rigid bus; 1e-9 kg m/s, kg m2/s, J; heat 1e-7 J."""
    times = np.linspace(0., 2.5, 2501)
    inertia = np.diag([4., 5., 6.])
    modes = [SloshMode([1, 0, 0], 2., 3., damping, [0, .4, 0], .02, .01, .1),
             SloshMode([0, 1, 0], 1., 4., damping, [0, 0, .3], -.01, 0., .1)]
    result = propagate_6dof_extended(
        times=times, r0=[0, 0, 0], v0=[.01, -.02, .03], omega0=[.08, -.06, .1],
        inertia=inertia, mu=0, bus_mass=10., coupled_slosh=True, slosh=modes,
        rtol=1e-11, atol=1e-13, max_step=.025)
    track = result.trajectory
    momentum = 10.*track.v
    angular = np.cross(track.r, momentum) + np.array([
        rotate_vector(q, inertia @ omega) for q, omega in zip(track.q, track.omega)])
    energy = 5.*np.sum(track.v**2, axis=1) + .5*np.einsum("ni,ij,nj->n", track.omega, inertia, track.omega)
    loss_rate = np.zeros(len(times))
    for i, mode in enumerate(modes):
        x, rate = result.slosh[:, :, i].T
        arm = mode.lever_arm_body + x[:, None]*mode.axis_body
        position = track.r + np.array([rotate_vector(q, a) for q, a in zip(track.q, arm)])
        relative_velocity = np.cross(track.omega, arm) + rate[:, None]*mode.axis_body
        velocity = track.v + np.array([rotate_vector(q, u) for q, u in zip(track.q, relative_velocity)])
        momentum += mode.mass*velocity
        angular += np.cross(position, mode.mass*velocity)
        energy += .5*mode.mass*(np.sum(velocity**2, axis=1) + mode.natural_frequency**2*x**2)
        loss_rate += 2*mode.damping_ratio*mode.natural_frequency*mode.mass*rate**2
    np.testing.assert_allclose(momentum-momentum[0], 0., atol=1e-9, rtol=0)
    np.testing.assert_allclose(angular-angular[0], 0., atol=1e-9, rtol=0)
    heat = cumulative_trapezoid(loss_rate, times, initial=0.)
    # Halving the independent quadrature spacing bounds its contribution.
    coarse_heat = cumulative_trapezoid(loss_rate[::2], times[::2], initial=0.)
    np.testing.assert_allclose(heat[::2], coarse_heat, atol=1e-7, rtol=0)
    np.testing.assert_allclose(energy+heat, energy[0], atol=1e-7 if damping else 1e-9, rtol=0)
    if damping:
        assert energy[-1] < energy[0] and heat[-1] > 0.


def test_common_free_fall_does_not_excite_a_centered_mode():
    """R1: circular Kepler orbit, 1e-5 m, 1e-8 m/s; mode remains zero to 1e-12 m."""
    from ssatk.constants import EARTH_MU

    radius = 7e6
    rate = np.sqrt(EARTH_MU/radius**3)
    times = np.linspace(0., 60., 31)
    result = propagate_6dof_extended(
        times=times, r0=[radius, 0, 0], v0=[0, radius*rate, 0], inertia=np.eye(3),
        bus_mass=10., coupled_slosh=True,
        slosh=SloshMode([1, 0, 0], 2., 3., displacement_limit=.05),
        rtol=1e-12, atol=1e-13)
    expected_r = radius*np.column_stack((np.cos(rate*times), np.sin(rate*times), np.zeros(len(times))))
    expected_v = radius*rate*np.column_stack((-np.sin(rate*times), np.cos(rate*times), np.zeros(len(times))))
    np.testing.assert_allclose(result.trajectory.r, expected_r, atol=1e-5, rtol=0)
    np.testing.assert_allclose(result.trajectory.v, expected_v, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.slosh, 0., atol=1e-12, rtol=0)


def test_coupled_stm_matches_two_mass_initial_displacement_derivative():
    """R1/R5: analytic displacement column, 2e-8 in SI derivative units."""
    result = propagate_6dof_extended_variational(
        times=[0., .4], r0=[0, 0, 0], v0=[0, 0, 0], inertia=np.eye(3), mu=0,
        bus_mass=10., mass0=12., coupled_slosh=True,
        slosh=SloshMode([1, 0, 0], 2., 3., displacement0=.02, displacement_limit=.05),
        rtol=1e-11, atol=1e-13)
    frequency = 3*np.sqrt(1.2)
    cosine, sine = np.cos(frequency*.4), np.sin(frequency*.4)
    expected = [(1-cosine)/6, frequency*sine/6, cosine, -frequency*sine]
    np.testing.assert_allclose(result.stm[-1, [0, 3, 14, 15], 14], expected, atol=2e-8, rtol=0)


def test_identification_and_held_out_prediction_match_synthetic_parameters():
    """R1: known 2 kg, 3 rad/s, .06 damping recovered within 2e-6 relative."""
    from demos.launch.demo_launch_slosh import run_reference

    record = run_reference()
    np.testing.assert_allclose(record["fitted_parameters"], [2., 3., .06], atol=0, rtol=2e-6)
    np.testing.assert_allclose(record["held_out"]["prediction"], record["held_out"]["reference"],
                               atol=2e-7, rtol=0)  # m and m/s; independent closed form.
