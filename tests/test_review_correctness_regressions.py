"""Numerical regressions for the October 2026 SSATK correctness review.

References are closed forms, independently integrated trajectories, SSAPy
conventions, finite differences, or rotation/covariance invariants. Tolerances
are stated alongside each assertion. All quantities use SI units.
"""

import importlib
import warnings

import numpy as np
import pytest
import astropy.units as u
from astropy.utils import iers
from scipy.integrate import solve_ivp
from scipy.stats import ncx2

import ssatk


from ssapy import Orbit, get_body
from ssapy.accel import AccelSolRad
from ssapy.utils import rv_to_ntw
from ssatk.constants import EARTH_MU
from ssatk.coordinates.attitude import attitude_quaternion_from_frame
from ssatk.coordinates.lunar import gcrf_to_lunar, gcrf_to_lunar_fixed
from ssatk.coordinates.satellite_frames import (
    frame_to_gcrf_matrix, gcrf_to_ntw, nadir_velocity_to_gcrf_matrix,
)
from ssatk.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssatk.orbital_mechanics.transfer_ssapy_function import solve_lambert
from ssatk.propagators_6dof import propagate_6dof_covariance, propagate_6dof_variational
from ssatk.propagators_orbit import leapfrog, propagate_orbit_state
from ssatk.time_functions import get_times
from ssatk.ssa import probability_of_collision
from ssatk.ssapy_wrappers.quick_int import quickint


def test_collision_probability_resolves_a_narrow_offset_gaussian():
    """R=10 m, sigma=1 cm, mean 5 m from origin: probability is nearly 1."""
    sigma, radius = 0.01, 10.0
    mean = 5.0 * np.array([np.cos(1.0), np.sin(1.0)])
    expected = ncx2.cdf((radius / sigma)**2, 2, mean @ mean / sigma**2)
    assert expected == pytest.approx(1.0, abs=1e-12)
    actual = probability_of_collision(mean, sigma**2 * np.eye(2), radius)
    assert actual == pytest.approx(expected, abs=1e-8)


def _force_leaves(acceleration):
    if hasattr(acceleration, "accels"):
        for child in acceleration.accels:
            yield from _force_leaves(child)
    else:
        yield acceleration


@pytest.mark.parametrize("name,cache", [
    ("best_prop", "_accel_best_cache"),
    ("best_gravity_prop", "_accel_best_gravity_cache"),
])
def test_force_factory_respects_each_spacecraft_area(name, cache, monkeypatch):
    """Real SSAPy SRP must scale by 10 when area increases by 10 at fixed mass."""
    module = importlib.import_module("ssatk.ssapy_wrappers.ssapy_props")
    monkeypatch.setattr(module, cache, None)
    factory = getattr(module, name)
    first = factory(dict(mass=100.0, area=1.0, CR=1.3, CD=2.2))
    second = factory(dict(mass=100.0, area=10.0, CR=1.3, CD=2.2))
    a = next(x for x in _force_leaves(first.accel) if isinstance(x, AccelSolRad))
    b = next(x for x in _force_leaves(second.accel) if isinstance(x, AccelSolRad))
    sun = get_body("sun")
    r = np.asarray(sun.position(1.4e9))
    r = 7e6 * r / np.linalg.norm(r)  # sunlit Earth orbit
    ratio = np.linalg.norm(b(r, np.zeros(3), 1.4e9)) / np.linalg.norm(a(r, np.zeros(3), 1.4e9))
    assert ratio == pytest.approx(10.0, rel=1e-12, abs=0)


@pytest.mark.parametrize("frame,local_up", [
    ("enu", [0.0, 0.0, 1.0]),
    ("ned", [0.0, 0.0, -1.0]),
    ("sez", [0.0, 0.0, 1.0]),
])
def test_local_frame_dispatcher_returns_inertial_axes(frame, local_up):
    """R2: at the equator geodetic up is parallel to Astropy's GCRS station
    position, Astropy configured with SSATK's packaged finals2000A.all so
    both sides use the same Earth-orientation data.

    Previously compared against ``ssapy.EarthObserver`` (IAU 1976/1980);
    ``frame_to_gcrf_matrix``'s local-tangent frames share SSATK's single IAU
    2006/2000A route (``ssatk.coordinates.eci_to_ecf_matrix``), so that
    comparison measured a ~9.3e-8 (unit-vector) model difference, not a bug.
    The measured agreement here is ~1.1e-9 (about 7 mm at Earth's radius).
    """
    from ssatk.data import data_path

    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        from astropy.coordinates import EarthLocation
        from astropy.time import Time

        t = Time(1.4e9, format="gps")
        location = EarthLocation(lon=0.0 * u.deg, lat=0.0 * u.deg, height=0.0 * u.m)
        position = location.get_gcrs(obstime=t).cartesian.xyz.to_value(u.m)
    expected = position / np.linalg.norm(position)
    actual = frame_to_gcrf_matrix(frame, lat=0.0, lon=0.0, t=1.4e9) @ local_up
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)


def test_nadir_frame_is_a_rotation_for_an_eccentric_state():
    r, v = np.array([7e6, 0.0, 0.0]), np.array([1000.0, 7500.0, 0.0])
    matrix = nadir_velocity_to_gcrf_matrix(r, v)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)
    # Along-track, negative orbit normal, and nadir, respectively. Orthogonality
    # alone would also accept an identity matrix or a different frame.
    expected = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12, rtol=0)
    from scipy.spatial.transform import Rotation
    q = attitude_quaternion_from_frame("nadir_velocity", r=r, v=v)
    np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
                               expected, atol=1e-12, rtol=0)


def test_lambert_recovers_an_independently_integrated_hyperbolic_arc():
    """A valid 300 s arc stays above its 7000 km periapsis throughout."""
    state0 = np.array([7e6, 0.0, 0.0, 0.0, 20000.0, 0.0])

    def rhs(t, y):
        return np.r_[y[3:], -EARTH_MU * y[:3] / np.linalg.norm(y[:3])**3]

    truth = solve_ivp(rhs, (0.0, 300.0), state0, method="DOP853", rtol=1e-12, atol=1e-8)
    assert truth.success
    assert np.min(np.linalg.norm(truth.y[:3], axis=0)) >= 7e6 - 1e-6
    independent = Orbit(state0[:3], state0[3:], 0.0).at(300.0)
    np.testing.assert_allclose(truth.y[:3, -1], independent.r, atol=1e-3, rtol=0)
    departure, arrival = solve_lambert(state0[:3], truth.y[:3, -1], 300.0)
    np.testing.assert_allclose(departure, state0[3:], atol=1e-3, rtol=0)
    np.testing.assert_allclose(arrival, truth.y[3:, -1], atol=1e-3, rtol=0)


def _attitude_options():
    return dict(inertia=np.eye(3), r0=[7e6, 0, 0], v0=[0, 0, 0], t0=0.0,
                mu=0.0, q0=[1, 0, 0, 0], omega0=[0, 0, 0.5], rtol=1e-11, atol=1e-13)


def _free_body_error_stm(t, rate=0.5):
    """Closed-form right/body attitude error: theta_dot=-omega x theta+dw.

    An isotropic, torque-free body has constant angular velocity. Integrating
    the planar sine/cosine solution gives both attitude blocks; translation
    is free flight. No production Jacobian or quaternion conversion is used.
    """
    c, s = np.cos(rate * t), np.sin(rate * t)
    phi = np.eye(12)
    phi[:3, 3:6] = t * np.eye(3)
    phi[6:9, 6:9] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    phi[6:9, 9:12] = [[s / rate, (1 - c) / rate, 0],
                       [(c - 1) / rate, s / rate, 0], [0, 0, t]]
    return phi


def test_variational_stm_includes_angular_momentum_flux_derivative():
    """For I=1 and angular flux=-omega, d omega(t)/d omega0 = exp(-t) I."""
    def flux(t, r, v, q, omega, *, mass=None, mass_rate=0.0):
        return -omega

    result = propagate_6dof_variational(times=[0.0, 1.0], angular_momentum_flux=flux, **_attitude_options())
    np.testing.assert_allclose(result.trajectory.omega[-1], [0, 0, 0.5 * np.exp(-1)], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.stm[-1, 10:13, 10:13], np.exp(-1) * np.eye(3), atol=1e-8, rtol=0)


def test_attitude_covariance_is_independent_of_output_sampling():
    all_samples = propagate_6dof_variational(times=[0.0, 1.0, 2.0], **_attitude_options())
    late_samples = propagate_6dof_variational(times=[1.0, 2.0], **_attitude_options())
    np.testing.assert_allclose(all_samples.stm[-1], late_samples.stm[-1], atol=1e-10, rtol=0)
    initial_covariance = np.diag([0.0]*6 + [1e-4, 4e-4, 9e-4] + [0.0]*3)
    first = propagate_6dof_covariance(all_samples, initial_covariance)[-1]
    second = propagate_6dof_covariance(late_samples, initial_covariance)[-1]
    phi = _free_body_error_stm(2.0)
    expected = phi @ initial_covariance @ phi.T
    for result, covariance in ((all_samples, first), (late_samples, second)):
        np.testing.assert_allclose(result.attitude_error_stm[-1], phi, atol=1e-9, rtol=0)
        np.testing.assert_allclose(covariance, expected, atol=1e-12, rtol=0)


def _normal_burn_reference(orbit, target):
    """Independent Radau integration of a 1 m/s^2 positive normal burn.

    The stopping condition uses cos(i), separately from the production angle
    event. Tightening tolerances by 10 must change the reference by <0.1 mm.
    """
    def rhs(t, y):
        r, v = y[:3], y[3:]
        h = np.cross(r, v)
        return np.r_[v, -EARTH_MU * r / np.linalg.norm(r)**3 + h / np.linalg.norm(h)]

    def event(t, y):
        h = np.cross(y[:3], y[3:])
        return h[2] / np.linalg.norm(h) - np.cos(target)

    event.terminal = True
    solutions = [solve_ivp(rhs, (0, 300), np.r_[orbit.r, orbit.v], method="Radau",
                           rtol=tol, atol=tol, events=event) for tol in (1e-11, 1e-12)]
    for solution in solutions:
        assert solution.success and len(solution.t_events[0]) == 1
    np.testing.assert_allclose(solutions[0].y[:, -1], solutions[1].y[:, -1], atol=1e-4, rtol=0)
    return solutions[1].t[-1], solutions[1].y[:, -1]


def test_inclination_target_is_invariant_under_node_rotation():
    def solve(raan):
        orbit = Orbit.fromKeplerianElements(7e6, 0, np.radians(30), 0, raan, 0, t=0)
        result = transfer_inclination_continuous(orbit.r, orbit.v, i_target=np.radians(31),
                                                a_thrust=1.0, max_time=300.0)
        time, state = _normal_burn_reference(orbit, np.radians(31))
        assert result["tof"] == pytest.approx(time, abs=1e-5)
        np.testing.assert_allclose(result["final"]["r"], state[:3], atol=1e-2, rtol=0)
        np.testing.assert_allclose(result["final"]["v"], state[3:], atol=1e-5, rtol=0)
        return result

    reference = solve(np.pi)
    h = np.cross(reference["final"]["r"], reference["final"]["v"])
    assert np.arccos(h[2] / np.linalg.norm(h)) == pytest.approx(np.radians(31), abs=1e-10)
    actual = solve(0.0)
    assert actual["tof"] == pytest.approx(reference["tof"], abs=1e-5)


@pytest.mark.parametrize("function", [gcrf_to_lunar, gcrf_to_lunar_fixed])
def test_lunar_state_transform_uses_the_supplied_velocity(function):
    """At fixed r,t a rotation must preserve the norm of a velocity difference."""
    epochs = 1.4e9 + np.array([0.0, 60.0, 120.0])
    positions = np.tile([7e6, 0.0, 0.0], (3, 1))
    velocity0 = np.zeros_like(positions)
    velocity1 = np.tile([100.0, 0.0, 0.0], (3, 1))
    _, first = function(positions, epochs, v=velocity0)
    _, second = function(positions, epochs, v=velocity1)
    np.testing.assert_allclose(np.linalg.norm(second - first, axis=1), 100.0, atol=1e-8, rtol=0)


def test_quickint_preserves_the_input_orbit_epoch():
    orbit = Orbit([7e6, 0, 0], [0, np.sqrt(EARTH_MU / 7e6), 0], 1.4e9)
    # Limit to one second, avoiding an unnecessary full-period calculation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, _, times = quickint(orbit, max_period_days=1.0 / 86400.0)
    assert float(times[0].gps) == pytest.approx(float(orbit.t), abs=1e-6, rel=0)


def test_leapfrog_accepts_an_astropy_grid_from_get_times():
    """A 4.8 ps conversion discrepancy must not invalidate a 1 s time step."""
    time = get_times(2.0, freq=1.0)
    np.testing.assert_allclose(time.gps, [0.0, 1.0, 2.0], atol=1e-10, rtol=0)
    speed = np.sqrt(EARTH_MU / 7e6)
    r0, v0 = [7e6, 0, 0], [0, speed, 0]
    reference_r, reference_v = leapfrog(r0, v0, [0.0, 1.0, 2.0])
    actual_r, actual_v = leapfrog(r0, v0, time)
    np.testing.assert_allclose(actual_r, reference_r, atol=1e-5, rtol=0)
    np.testing.assert_allclose(actual_v, reference_v, atol=1e-8, rtol=0)
    phase = speed / 7e6 * np.arange(3)
    expected_r = 7e6 * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(3)))
    expected_v = speed * np.column_stack((-np.sin(phase), np.cos(phase), np.zeros(3)))
    # Velocity-Verlet's O(dt^2) global error at dt=1 s, over only 2 s.
    np.testing.assert_allclose(actual_r, expected_r, atol=0.01, rtol=0)
    np.testing.assert_allclose(actual_v, expected_v, atol=2e-5, rtol=0)


def test_control_ntw_matches_ssapy_for_a_non_circular_state():
    r, v = np.array([7e6, 2e6, 1e6]), np.array([-1000.0, 7000.0, 2000.0])
    vector = np.array([2.0, -3.0, 5.0])
    np.testing.assert_allclose(gcrf_to_ntw(vector, r, v), rv_to_ntw(r, v, r + vector), atol=1e-12, rtol=0)


def test_control_translational_propagation_matches_ssapy():
    orbit = Orbit([7e6, 2e6, 1e6], [-1000.0, 7000.0, 2000.0], 1.4e9)
    times = 1.4e9 + np.linspace(0.0, 1200.0, 5)
    actual = propagate_orbit_state(orbit0=orbit, times=times, rtol=1e-11, atol=1e-8)
    expected = orbit.at(times)
    np.testing.assert_allclose(actual.r, expected.r, atol=1e-2, rtol=0)
    np.testing.assert_allclose(actual.v, expected.v, atol=1e-5, rtol=0)


def test_thrust_impulse_numpy_compatibility():
    """Integral of a 2 N force over 5 s is 10 N s, also on NumPy 1.26."""
    from ssatk.accelerations_6dof import integrated_thrust_impulse
    impulse = integrated_thrust_impulse(lambda t: 2.0, 0.0, 5.0)
    assert impulse == pytest.approx(10.0, abs=1e-12)


@pytest.fixture(autouse=True)
def offline_earth_orientation():
    with iers.conf.set_temp("auto_download", False), \
         iers.conf.set_temp("auto_max_age", None), \
         iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        yield


@pytest.mark.parametrize("distance,sigma", [(0, .001), (5, .01), (9.99, .01),
                                            (10, .01), (10.01, .01), (15, .01),
                                            (5, 2), (0, 100)])
@pytest.mark.parametrize("angle", [0.0, .2, 1.0, 2.7])
def test_collision_probability_matches_noncentral_chi_square(distance, sigma, angle):
    """R1/R4: isotropic disk probability, including the boundary, within 1e-8."""
    radius = 10.0
    mean = distance * np.array([np.cos(angle), np.sin(angle)])
    expected = ncx2.cdf((radius / sigma)**2, 2, (distance / sigma)**2)
    actual = probability_of_collision(mean, sigma**2 * np.eye(2), radius)
    assert actual == pytest.approx(expected, abs=1e-8, rel=0)


def test_anisotropic_collision_probability_matches_independent_polar_cubature():
    """R2/R4: fixed polar Gaussian cubature and rotation invariance, 1e-10."""
    from numpy.polynomial.legendre import leggauss

    radius, mean, covariance = 3.0, np.array([1.0, 1.0]), np.diag([.2**2, 1.2**2])
    def cubature(n):
        radial_nodes, radial_weights = leggauss(n)
        angular_nodes, angular_weights = leggauss(4 * n)
        distance = radius * (radial_nodes + 1.0) / 2.0
        theta = np.pi * (angular_nodes + 1.0)
        points = distance[:, None, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
        delta = points - mean
        exponent = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
        density = np.exp(-.5 * exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))
        return np.sum(density * distance[:, None] * radial_weights[:, None]
                      * angular_weights[None, :]) * radius * np.pi / 2.0

    expected = cubature(128)
    assert cubature(64) == pytest.approx(expected, abs=2e-12, rel=0)
    for angle in (0.0, .73, 2.1):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        actual = probability_of_collision(rotation @ mean, rotation @ covariance @ rotation.T, radius)
        assert actual == pytest.approx(expected, abs=1e-10, rel=0)


def test_nearly_rank_one_collision_probability_matches_one_dimensional_limit():
    """R1: a 1e-5 m transverse width changes the 1-D limit by <1e-10."""
    from scipy.special import ndtr

    expected = ndtr(1.5) - ndtr(-1.5)
    actual = probability_of_collision([0.0, 0.0], np.diag([1e-10, 4.0]), 3.0)
    assert actual == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("function", [gcrf_to_lunar, gcrf_to_lunar_fixed])
def test_lunar_velocity_matches_position_derivative_at_an_isolated_epoch(function):
    """R5: derivative of a known linear GCRF trajectory, tolerance 1e-4 m/s."""
    epoch, dt = 1.4e9, .25
    r, v = np.array([7e6, 2e6, -1e6]), np.array([-1000.0, 7000.0, 2000.0])
    _, actual = function(r, epoch, v=v)
    expected = (function(r + dt * v, epoch + dt) - function(r - dt * v, epoch - dt)) / (2 * dt)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    epochs = epoch + np.array([0, 86400, 172800])
    positions, velocities = np.tile(r, (3, 1)), np.tile(v, (3, 1))
    _, sparse = function(positions, epochs, v=velocities)
    np.testing.assert_allclose(sparse[0], actual, atol=1e-10, rtol=0)


def test_moon_state_is_stationary_at_origin_of_moon_centred_frame():
    """R4: the frame origin has exactly zero relative position and velocity."""
    from ssatk.coordinates.lunar import get_lunar_rv

    epoch = 1.4e9
    r, v = get_lunar_rv(epoch)
    position, velocity = gcrf_to_lunar_fixed(r, epoch, v=v)
    np.testing.assert_allclose(position, 0.0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(velocity, 0.0, atol=1e-8, rtol=0)


def test_variational_stm_includes_inertia_rate():
    """R1: I(t)=exp(t) I and Idot=I(t) conserve I omega without external torque."""
    options = _attitude_options()
    options["inertia"] = lambda t, r, v, q, omega, **kwargs: np.exp(t) * np.eye(3)
    result = propagate_6dof_variational(times=[0.0, 1.0],
                                        inertia_rate=lambda t, r, v, q, omega, **kwargs: np.exp(t) * np.eye(3), **options)
    np.testing.assert_allclose(result.trajectory.omega[-1], [0, 0, .5 * np.exp(-1)], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.stm[-1, 10:13, 10:13], np.exp(-1) * np.eye(3), atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.exp(result.trajectory.t)[:, None] * result.trajectory.omega,
                               [[0, 0, .5], [0, 0, .5]], atol=1e-10, rtol=0)


def test_extended_attitude_covariance_is_independent_of_output_sampling():
    """R4: changing output times cannot change terminal covariance (1e-10)."""
    from ssatk.propagators_6dof import FlexibleMode, propagate_6dof_extended_variational

    options = dict(_attitude_options(), flexible=FlexibleMode([1, 0, 0], 1.0, .1))
    complete = propagate_6dof_extended_variational(times=[0, 1, 2], **options)
    late = propagate_6dof_extended_variational(times=[1, 2], **options)
    covariance = np.diag([0.0]*6 + [1e-4, 4e-4, 9e-4] + [0.0]*5)
    phi = np.eye(14)
    phi[:12, :12] = _free_body_error_stm(2.0)
    expected = phi @ covariance @ phi.T
    for result in (complete, late):
        np.testing.assert_allclose(propagate_6dof_covariance(result, covariance)[-1],
                                   expected, atol=1e-12, rtol=0)


def test_chained_attitude_stm_preserves_its_original_reference_quaternion():
    """R4: composing two one-second segments matches a two-second STM, 1e-9."""
    options = _attitude_options()
    whole = propagate_6dof_variational(times=[0, 1, 2], **options)
    options.update(t0=1.0, q0=whole.trajectory.q[1], omega0=whole.trajectory.omega[1])
    second = propagate_6dof_variational(times=[1, 2], stm0=whole.stm[1],
                                        stm_reference_q=whole.reference_q, **options)
    np.testing.assert_allclose(second.attitude_error_stm[-1], whole.attitude_error_stm[-1], atol=1e-9, rtol=0)
    np.testing.assert_allclose(second.attitude_error_stm[-1], _free_body_error_stm(2.0), atol=1e-9, rtol=0)


@pytest.mark.parametrize("raan", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_inclination_target_detects_decreasing_inclination(raan):
    """R4: a descending-node normal burn reaches 29 deg for any RAAN, 1e-10 rad."""
    orbit = Orbit.fromKeplerianElements(7e6, 0, np.radians(30), 0, raan, np.pi, t=0)
    result = transfer_inclination_continuous(orbit.r, orbit.v, i_target=np.radians(29),
                                            a_thrust=1.0, max_time=300.0)
    h = np.cross(result["final"]["r"], result["final"]["v"])
    assert np.arccos(h[2] / np.linalg.norm(h)) == pytest.approx(np.radians(29), abs=1e-10, rel=0)
    time, state = _normal_burn_reference(orbit, np.radians(29))
    assert result["tof"] == pytest.approx(time, abs=1e-5)
    np.testing.assert_allclose(result["final"]["r"], state[:3], atol=1e-2, rtol=0)
    np.testing.assert_allclose(result["final"]["v"], state[3:], atol=1e-5, rtol=0)


@pytest.mark.parametrize("function,moon_centred", [(gcrf_to_lunar, False), (gcrf_to_lunar_fixed, True)])
def test_lunar_frame_matches_analytic_circular_ephemeris(function, moon_centred, monkeypatch):
    """R1: exact rotating-frame kinematics for a prescribed circular Moon.

    This validates the coordinate transform, not the accuracy of a lunar
    ephemeris. Finite-difference roundoff and truncation allow 1e-6 m/s.
    """
    module = importlib.import_module("ssatk.coordinates.lunar")
    distance, rate, epoch = 384400000.0, 2 * np.pi / (27.321661 * 86400), 1.4e9

    class CircularMoon:
        def position(self, t):
            angle = rate * (np.asarray(t) - epoch)
            return distance * np.array([np.cos(angle), np.sin(angle), np.zeros_like(angle)])

    monkeypatch.setattr(module, "get_body", lambda name: CircularMoon())
    r, v = np.array([7e6, 2e6, -1e6]), np.array([-1000., 7000., 2000.])
    for elapsed in (0., 60000., 120000.):
        c, s = np.cos(rate * elapsed), np.sin(rate * elapsed)
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        rotation_rate = rate * np.array([[-s, c, 0], [-c, -s, 0], [0, 0, 0]])
        relative_r = r - distance * np.array([c, s, 0]) if moon_centred else r
        relative_v = v - distance * rate * np.array([-s, c, 0]) if moon_centred else v
        position, velocity = function(r, epoch + elapsed, v=v)
        np.testing.assert_allclose(position, rotation @ relative_r, atol=1e-6, rtol=0)
        np.testing.assert_allclose(velocity, rotation @ relative_v + rotation_rate @ relative_r,
                                   atol=1e-6, rtol=0)


def test_leapfrog_still_rejects_a_physically_nonuniform_grid():
    """R6: a 0.1 s step change is physical, not Julian-date conversion noise."""
    with pytest.raises(ValueError, match="Non-uniform"):
        leapfrog([7e6, 0, 0], [0, 7500, 0], [0, 1, 2.1])
