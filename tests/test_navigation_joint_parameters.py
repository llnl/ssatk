"""Independent orbit/force/measurement references for joint batch estimation."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ssatk.accelerations_6dof import (
    SpacecraftAccelDrag, SpacecraftAccelSolRad, make_finite_burn_acceleration,
)
from ssatk.constants import AU, EARTH_MU, SOLAR_FLUX_1_AU, c
from ssatk.navigation import GroundStation, RadioTrackingConfig, StationObservation, solve_batch_orbit


@pytest.fixture
def fixed_sites(monkeypatch):
    sites = np.array([[6e6, 0, 0], [0, 6e6, 0], [0, 0, 6e6]])
    monkeypatch.setattr(GroundStation, "_station_rv", lambda self, time: (sites[int(self.lon_deg)], np.zeros(3)))
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: tuple(np.eye(3)))
    return sites, [GroundStation(i, 0, min_elevation_rad=-np.pi / 2) for i in range(3)]


def test_joint_drag_srp_burn_and_station_biases_against_independent_ode(fixed_sites):
    """R2: independent Newton/drag/SRP/burn ODE, 2 mm, 5 um/s, 0.2% parameters.

    Density and Sun position are prescribed synthetic inputs. The reference
    integrates each burn interval separately at 1e-13 relative tolerance.
    Fisher covariance uses independent finite-difference observation partials,
    checked to 0.5% relative to allow differentiation of numerical trajectories.
    """
    sites, stations = fixed_sites
    state = np.array([7e6, 1e5, 2e5, -100, 7400, 700.0])
    density, area, mass = 2e-11, 1000.0, 100.0
    sun = np.array([AU, 0, 0])
    times = np.array([0., 60, 100, 160, 220, 300, 400, 500, 650, 800])
    parameters = {"cd": 2.1, "cr": 1.4, "thrust_n": 0.2, "range_bias_m": 15., "rate_bias_m_s": 0.05}
    parameter_truth = np.array(list(parameters.values()))

    def reference(values, rtol=1e-13):
        initial, cd, cr, thrust, range_bias, rate_bias = values[:6], *values[6:]
        current = initial.copy()
        states = np.empty((len(times), 6))
        states[0] = current
        for left, right, firing in ((0., 100., False), (100., 160., True), (160., 800., False)):
            def rhs(time, y):
                r, v = y[:3], y[3:]
                delta = r - sun
                acceleration = -EARTH_MU * r / np.linalg.norm(r)**3
                acceleration -= 0.5 * density * cd * area / mass * np.linalg.norm(v) * v
                acceleration += SOLAR_FLUX_1_AU / c * cr * area / mass * AU**2 * delta / np.linalg.norm(delta)**3
                if firing:
                    acceleration += np.array([0, 0, thrust / mass])
                return np.r_[v, acceleration]
            solution = solve_ivp(rhs, (left, right), current, method="DOP853", rtol=rtol, atol=1e-10, dense_output=True)
            assert solution.success
            select = (times > left) & (times <= right)
            states[select] = solution.sol(times[select]).T
            current = solution.y[:, -1]
        predictions = []
        for station_index, site in enumerate(sites):
            for r, v in zip(states[:, :3], states[:, 3:]):
                delta = r - site
                distance = np.linalg.norm(delta)
                predictions.extend([distance + (range_bias if station_index == 0 else 0),
                                    np.dot(delta, v) / distance + (rate_bias if station_index == 1 else 0)])
        return np.array(predictions)

    truth = np.r_[state, parameter_truth]
    observed = reference(truth)
    convergence_error = observed - reference(truth, rtol=1e-12)
    np.testing.assert_allclose(convergence_error[::2], 0, atol=1e-5, rtol=0)  # metres
    np.testing.assert_allclose(convergence_error[1::2], 0, atol=1e-8, rtol=0)  # metres/second
    observations = []
    row = 0
    for station in stations:
        for time in times:
            for kind, sigma in (("range", 1.), ("range_rate", 0.001)):
                observations.append((station, StationObservation(time, kind, observed[row], np.array([[sigma**2]]),
                                                                0., None, None)))
                row += 1

    def propagation(p):
        return {"acceleration": (
            SpacecraftAccelDrag(density=density, area=area, mass=mass, cd=p["cd"],
                                atmosphere_velocity=np.zeros(3)),
            SpacecraftAccelSolRad(sun, area=area, mass=mass, cr=p["cr"]),
            make_finite_burn_acceleration(p["thrust_n"], mass=mass, direction=[0, 0, 1],
                                         frame="inertial", start=100., stop=160.),
        ), "rtol": 1e-12, "atol": 1e-9, "fd_step": 1e-4}

    def bias(station, observation, p):
        if station.lon_deg == 0 and observation.measurement == "range":
            return p["range_bias_m"]
        if station.lon_deg == 1 and observation.measurement == "range_rate":
            return p["rate_bias_m_s"]
        return 0.

    scales = np.r_[[1e7]*3, [1e3]*3, [2., 1., 0.2, 20., 0.1]]
    fit = solve_batch_orbit(
        observations, state + [50, -40, 30, 0.1, -0.05, 0.02], t0=0,
        parameters0=dict(zip(parameters, parameter_truth * 0.8)), parameter_scale=scales[6:],
        parameter_step=[0.002, 0.002, 0.0002, 0.1, 0.001],
        parameter_bounds=([0.1, 0.1, 0., -100., -1.], [5., 3., 1., 100., 1.]),
        propagation_kwargs=propagation, measurement_bias=bias, max_nfev=30,
    )
    assert fit.success and fit.numerical_rank == len(truth), fit.message
    np.testing.assert_allclose(fit.state0[:3], state[:3], atol=2e-3, rtol=0)
    np.testing.assert_allclose(fit.state0[3:], state[3:], atol=5e-6, rtol=0)
    np.testing.assert_allclose(list(fit.parameters.values()), parameter_truth, atol=0, rtol=2e-3)
    assert fit.estimate_names == ("x", "y", "z", "vx", "vy", "vz", *parameters)

    steps = np.r_[[1.]*3, [0.001]*3, [0.002, 0.002, 0.0002, 0.1, 0.001]]
    jacobian = np.empty((len(observed), len(truth)))
    sigma = np.tile([1., 0.001], len(observed)//2)
    for index, step in enumerate(steps):
        positive, negative = truth.copy(), truth.copy()
        positive[index] += step
        negative[index] -= step
        jacobian[:, index] = (reference(positive) - reference(negative)) / (2 * step * sigma)
    _, singular, vh = np.linalg.svd(jacobian * scales)
    covariance = scales[:, None] * ((vh.T / singular**2) @ vh) * scales[None, :]
    # Normalize the error by the independent standard deviations, including small cross terms.
    reference_sigma = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose((fit.joint_covariance - covariance) / np.outer(reference_sigma, reference_sigma),
                               0, atol=5e-3, rtol=0)
    expected_correlation = covariance / np.outer(reference_sigma, reference_sigma)
    np.testing.assert_allclose(fit.correlation, expected_correlation, atol=5e-3, rtol=0)
    np.testing.assert_array_equal(fit.covariance, fit.joint_covariance[:6, :6])


def test_unobservable_parameter_keeps_fisher_covariance_unavailable(fixed_sites):
    """R1: free flight is independent of an unused coefficient, so its information is zero."""
    sites, stations = fixed_sites
    state = np.array([20e6, 5e6, 3e6, -1000., 2000., 500.])
    observations = []
    for site, station in zip(sites, stations):
        for time in (0., 100., 300.):
            delta = state[:3] + state[3:] * time - site
            distance = np.linalg.norm(delta)
            for kind, value in (("range", distance), ("range_rate", np.dot(delta, state[3:]) / distance)):
                observations.append((station, StationObservation(time, kind, value, np.eye(1), 0., None, None)))
    fit = solve_batch_orbit(observations, state, propagation_kwargs={"mu": 0}, parameters0={"unused": 1.})
    assert not fit.success and fit.numerical_rank == 6
    assert fit.covariance is None and fit.joint_covariance is None and fit.correlation is None


def test_joint_radio_bias_against_analytic_light_time_and_fisher(fixed_sites, monkeypatch, gps_epoch):
    """R1/R5: free-flight radio bias/state within 1 mm/2 um/s; Fisher within 0.5%.

    Known 3-unit corrections are distinct from the fitted 12.5 m range bias.
    The reference solves light time algebraically and differences range over
    the full Doppler count; its Fisher matrix uses independent finite differences.
    """
    from ssatk.navigation import measurements

    monkeypatch.setattr(measurements, "_earth_state", lambda time, offset, kernel: (np.zeros(3), np.zeros(3)))
    sites, stations = fixed_sites
    state = np.array([20e6, 5e6, 3e6, -1000., 2000., 500.])
    radio = RadioTrackingConfig(7.2e9, 10., 880/749, 2e-6)
    truth = np.r_[state, 12.5]
    epochs = (100., 300., 600.)

    def reference(values):
        r0, velocity, bias = values[:3], values[3:6], values[6]
        rows = []
        for site in sites:
            def range_at(elapsed):
                delta = r0 + velocity * elapsed - site
                distance = np.linalg.norm(delta)
                projection = np.dot(delta, velocity)
                delay = distance**2 / (np.sqrt(projection**2 + (c*c - np.dot(velocity, velocity))*distance**2) + projection)
                receive = r0 + velocity*(elapsed-delay-radio.transponder_delay_s)
                return (c*(delay+radio.transponder_delay_s) + np.linalg.norm(receive-site))/2
            for elapsed in epochs:
                value = range_at(elapsed)
                rows.extend([value+bias, -2*radio.turnaround_ratio*radio.uplink_frequency_hz /
                             (c*radio.count_interval_s) * (value-range_at(elapsed-radio.count_interval_s))])
        return np.array(rows)

    observations = []
    values = iter(reference(truth))
    for station in stations:
        for elapsed in epochs:
            for kind in ("two_way_range", "two_way_doppler"):
                observations.append((station, StationObservation(gps_epoch+elapsed, kind, next(values)+3,
                                                                np.eye(1), 3., None, None, radio=radio)))
    fit = solve_batch_orbit(
        observations, state+[100, -200, 300, 1, -2, 3], t0=gps_epoch,
        parameters0={"range_bias_m": 8.}, parameter_scale=[10.], parameter_step=[0.1],
        propagation_kwargs={"mu": 0, "rtol": 1e-12, "atol": 1e-9},
        measurement_bias=lambda station, observation, p: p["range_bias_m"] if observation.measurement == "two_way_range" else 0.,
        max_nfev=20,
    )
    assert fit.success and fit.numerical_rank == 7, fit.message
    np.testing.assert_allclose(fit.state0[:3], state[:3], atol=1e-3, rtol=0)
    np.testing.assert_allclose(fit.state0[3:], state[3:], atol=2e-6, rtol=0)
    assert fit.parameters["range_bias_m"] == pytest.approx(12.5, abs=1e-3)
    scales = np.r_[[1e7]*3, [1e3]*3, 10.]
    steps = np.r_[[1.]*3, [0.01]*3, 0.1]
    jacobian = np.empty((len(observations), 7))
    for column, step in enumerate(steps):
        positive, negative = truth.copy(), truth.copy()
        positive[column] += step
        negative[column] -= step
        jacobian[:, column] = (reference(positive)-reference(negative))/(2*step)
    _, singular, vh = np.linalg.svd(jacobian*scales, full_matrices=False)
    covariance = scales[:, None] * ((vh.T / singular**2) @ vh) * scales[None, :]
    sigma = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose((fit.joint_covariance-covariance)/np.outer(sigma, sigma), 0, atol=5e-3, rtol=0)
    np.testing.assert_allclose(fit.correlation, covariance/np.outer(sigma, sigma), atol=5e-3, rtol=0)


def test_joint_burn_duration_against_piecewise_motion(fixed_sites):
    """R1: finite-burn thrust and duration, 0.01% and 1 ms; state within 1 mm/2 um/s."""
    sites, stations = fixed_sites
    state = np.array([20e6, 5e6, 3e6, -1000., 2000., 500.])
    thrust, mass, duration, start = 0.4, 100., 60., 100.
    observations = []
    for site, station in zip(sites, stations):
        for time in (0., 60., 120., 140., 180., 220., 300.):
            active = np.clip(time-start, 0, duration)
            r = state[:3] + state[3:]*time + np.array([0, 0, thrust/mass *
                (0.5*active**2 + duration*max(time-start-duration, 0))])
            v = state[3:] + np.array([0, 0, thrust/mass*active])
            delta = r-site
            distance = np.linalg.norm(delta)
            for kind, value in (("range", distance), ("range_rate", np.dot(delta, v)/distance)):
                observations.append((station, StationObservation(time, kind, value, np.eye(1), 0., None, None)))
    fit = solve_batch_orbit(
        observations, state+[20, -15, 10, 0.01, -0.02, 0.03], t0=0,
        parameters0={"thrust_n": 0.3, "duration_s": 50.}, parameter_scale=[0.4, 60.],
        parameter_step=[0.0002, 0.01], parameter_bounds=([0., 40.], [1., 80.]),
        propagation_kwargs=lambda p: {"mu": 0, "acceleration": make_finite_burn_acceleration(
            p["thrust_n"], mass=mass, direction=[0, 0, 1], frame="inertial", start=start, stop=start+p["duration_s"]),
            "rtol": 1e-12, "atol": 1e-9}, max_nfev=30,
    )
    assert fit.success and fit.numerical_rank == 8, fit.message
    np.testing.assert_allclose(fit.state0[:3], state[:3], atol=1e-3, rtol=0)
    np.testing.assert_allclose(fit.state0[3:], state[3:], atol=2e-6, rtol=0)
    assert fit.parameters["thrust_n"] == pytest.approx(thrust, rel=1e-4, abs=0)
    assert fit.parameters["duration_s"] == pytest.approx(duration, abs=1e-3)
