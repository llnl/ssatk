"""R1/C: optical fixes and their orbit-filter adapter against analytic geometry."""

from dataclasses import replace

import numpy as np
import pytest
from astropy.time import Time

from ssatk.navigation import (
    CartesianOrbitEKF, EKFState, solve_optical_navigation, update_orbit_from_optical_fix,
)


def _axis_fix(position, time=0.):
    # Three orthogonal rays to known points 1, 2, 3 km along positive axes.
    return solve_optical_navigation(
        np.eye(3), np.eye(3), np.asarray(position)+np.diag([1000., 2000., 3000.]),
        np.eye(3), time=time, prior_camera_to_gcrf=np.eye(3), match_gate_rad=.01)


def test_optical_fix_recovers_known_rotation_calibration_and_intersecting_rays():
    # R1: camera->GCRF is a literal +90deg z rotation. Landmark->star camera
    # calibration is a separate +90deg x rotation; confusing either transpose
    # yields a wrong position. One deliberately offset star is withheld and a
    # backward outlier cannot take a catalogue assignment.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    calibration = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    epsilon = .003
    stars = np.array([[np.sin(epsilon), 0., np.cos(epsilon)], [0., -1., 0.],
                      [1., 0., 0.], [0., 0., -1.]])
    position = np.array([7e6, -200., 300.])
    epoch = Time('2025-01-01T00:00:00', scale='utc')
    fix = solve_optical_navigation(
        stars, np.eye(3), position+np.diag([1000., 2000., 3000.]), rotation @ calibration,
        time=epoch, prior_camera_to_gcrf=rotation, match_gate_rad=.01,
        fit_star_mask=np.array([False, True, True, True]),
        landmark_camera_to_star_camera=calibration)
    np.testing.assert_allclose(fix.camera_to_gcrf, rotation, atol=1e-15, rtol=0)
    np.testing.assert_allclose(fix.position_gcrf_m, position, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(fix.star_indices, [0, 1, 2])
    np.testing.assert_array_equal(fix.catalogue_indices, [2, 0, 1])
    np.testing.assert_array_equal(fix.fit_mask, [False, True, True])
    np.testing.assert_allclose(fix.star_residual_rad, [epsilon, 0, 0], atol=1e-15, rtol=0)
    np.testing.assert_allclose(fix.landmark_residual_rad, 0, atol=1e-14, rtol=0)
    assert fix.time == epoch.gps and not fix.position_gcrf_m.flags.writeable


def test_repeated_optical_fixes_match_linear_gaussian_state_and_covariance():
    # R1: zero force gives r(t)=r0+v0*t and a known exact transition matrix.
    # A batch Gaussian information solution is an independent oracle for the
    # sequential update, including inferred velocity and r/v cross covariance.
    epoch = 1.4e9
    truth = np.array([7e6, -2000., 3000., 1., 2., -3.])
    prior = truth + [100., -50., 20., 2., -1., 3.]
    p0 = np.diag([400., 900., 1600., 4., 9., 16.])
    noise = np.diag([4., 9., 16.])
    navigator = CartesianOrbitEKF(EKFState(prior, p0, epoch))
    precision, information = np.linalg.inv(p0), np.linalg.solve(p0, prior)
    for dt in (0., 10., 30., 60.):
        h = np.column_stack([np.eye(3), dt*np.eye(3)])
        observed = truth[:3] + dt*truth[3:]
        state = update_orbit_from_optical_fix(navigator, _axis_fix(observed, epoch+dt), noise,
                                             propagation_kwargs={'mu': 0.})
        precision += h.T @ np.linalg.solve(noise, h)
        information += h.T @ np.linalg.solve(noise, observed)
        covariance0 = np.linalg.inv(precision)
        transition = np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        expected = transition @ np.linalg.solve(precision, information)
        np.testing.assert_allclose(state.x, expected, atol=2e-7, rtol=0)
        np.testing.assert_allclose(state.covariance, transition @ covariance0 @ transition.T,
                                   atol=1e-8, rtol=0)
        assert state.time == epoch+dt
    assert np.linalg.norm(state.x[3:]-truth[3:]) < .03


def test_filtered_optical_state_can_seed_a_known_station_keeping_impulse():
    # R1: a prescribed inertial velocity correction exactly cancels a known
    # free-flight velocity offset. Exercise the real six-DoF maneuver API with
    # an estimated state, not a demo import or a reconstructed truth state.
    from ssatk.propagators_6dof import Spacecraft, ImpulseManeuver, propagate_spacecraft_segments

    navigator = CartesianOrbitEKF(EKFState([7e6, 0, 0, 1., 2., 3.], np.eye(6), time=20.))
    state = update_orbit_from_optical_fix(navigator, _axis_fix([7e6, 0, 0], 20.), np.eye(3))
    spacecraft = Spacecraft(r=state.x[:3], v=state.x[3:], t=state.time, mass=100., inertia=np.eye(3))
    target_v = np.array([1., 2., 2.5])
    result = propagate_spacecraft_segments(spacecraft, [{
        'times': [state.time, state.time+10.], 'mu': 0.,
        'impulses': ImpulseManeuver(target_v-state.x[3:], frame='inertial'),
    }])
    np.testing.assert_allclose(result.v[-1], target_v, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.r[-1], state.x[:3]+10*target_v, atol=1e-8, rtol=0)


@pytest.mark.parametrize('problem', ['old_epoch', 'singular_covariance', 'asymmetric_covariance', 'time_override', 'prediction_failure'])
def test_invalid_filter_update_leaves_the_prior_unchanged(problem):
    # C: malformed or stale navigation must not partially advance guidance state.
    navigator = CartesianOrbitEKF(EKFState([7e6, 0, 0, 0, 7500., 0], np.eye(6), 10.))
    original = navigator.state
    fix, covariance, kwargs = _axis_fix([7e6, 0, 0], 20.), np.eye(3), {}
    if problem == 'old_epoch':
        fix = replace(fix, time=9.)
    elif problem == 'singular_covariance':
        covariance[0, 0] = 0.
    elif problem == 'asymmetric_covariance':
        covariance[0, 1] = 1.
    elif problem == 'time_override':
        kwargs['time'] = 30.
    elif problem == 'prediction_failure':
        kwargs['method'] = 'not-an-integrator'
    with pytest.raises(ValueError):
        update_orbit_from_optical_fix(navigator, fix, covariance, propagation_kwargs=kwargs)
    assert navigator.state is original


@pytest.mark.parametrize('problem', ['parallel_stars', 'parallel_landmarks', 'bad_calibration', 'empty_fit', 'non_boolean_mask', 'array_epoch'])
def test_unobservable_or_malformed_optical_inputs_fail_explicitly(problem):
    stars, references, bearings = np.eye(3), np.eye(3), np.eye(3)
    kwargs = dict(time=0., prior_camera_to_gcrf=np.eye(3), match_gate_rad=.1)
    if problem == 'parallel_stars':
        stars = references = np.tile([0., 0., 1.], (3, 1))
    elif problem == 'parallel_landmarks':
        bearings = np.tile([0., 0., 1.], (3, 1))
    elif problem == 'bad_calibration':
        kwargs['landmark_camera_to_star_camera'] = np.diag([1., 1., -1.])
    elif problem == 'empty_fit':
        kwargs['fit_star_mask'] = np.zeros(3, dtype=bool)
    elif problem == 'non_boolean_mask':
        kwargs['fit_star_mask'] = np.ones(3, dtype=int)
    elif problem == 'array_epoch':
        kwargs['time'] = [0., 1.]
    with pytest.raises(ValueError):
        solve_optical_navigation(stars, references, np.eye(3)*1000, bearings, **kwargs)
