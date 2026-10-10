"""Spectral, chaos-indicator, statistics, sampling and geometry helpers against references."""

import numpy as np
import pytest
from scipy import stats

from ssatk.compute.calculate_errors import calculate_errors
from ssatk.compute.fft import FFT, FFTP
from ssatk.compute.find_bounding_cube import find_smallest_bounding_cube
from ssatk.compute.generate_sphere_of_vectors import generate_sphere_vectors
from ssatk.compute.lyapunov_exponent import lyapunov_exponent_from_statevectors
from ssatk.compute.mengos import megno
from ssatk.compute.sampling import perturb_state_3d
from ssatk.compute.segment_intersection import segment_intersects_sphere


@pytest.mark.parametrize("n", [64, 100, 1001])
def test_fft_frequency_axis_matches_numpy_and_finds_a_tone(n):
    # R2: numpy.fft.fftfreq gives bin k at k / (N dt). R1: a pure tone on bin 7
    # peaks there. Exact to 1e-12 relative.
    dt = 0.25
    freqs, amplitude = FFT(np.sin(2 * np.pi * 7 / (n * dt) * np.arange(n) * dt), dt)
    np.testing.assert_allclose(freqs, np.fft.fftfreq(n, dt)[: n // 2], rtol=1e-12, atol=0)
    assert freqs[np.argmax(amplitude)] == pytest.approx(7 / (n * dt), rel=1e-12)
    periods, _ = FFTP(np.ones(n), dt)
    np.testing.assert_allclose(periods[1:], n * dt / np.arange(1, n // 2), rtol=1e-12)
    assert periods[0] == pytest.approx(n * dt)


def test_megno_matches_closed_forms():
    # R1 (Cincotta & Simo 2000): delta = e^(lambda t) gives <Y>(T) = lambda T / 2
    # exactly (1e-12); delta = 1 + c t gives Y = 2 [1 - ln(1 + c t)/(c t)], -> 2,
    # matched to 2e-4 with 20,001 samples.
    t = np.linspace(0.0, 50.0, 2001)
    assert megno(t, np.exp(0.7 * t)) == pytest.approx(0.7 * 50.0 / 2.0, rel=1e-12)

    t = np.linspace(0.0, 200.0, 20001)
    _mean, y, _y_mean = megno(t, 1.0 + 3.0 * t, return_series=True)
    exact = 2.0 * (1.0 - np.log(1.0 + 3.0 * t[1:]) / (3.0 * t[1:]))
    np.testing.assert_allclose(y[1:], exact, rtol=0, atol=2e-4)


def _variational_particles(sim):
    # REBOUND 5 exposes them as sim.particles_var; REBOUND 4 appends them to
    # sim.particles after the N_real real particles.
    if hasattr(sim, "particles_var"):
        return list(sim.particles_var)
    return [sim.particles[i] for i in range(sim.N_real, sim.N)]


def test_megno_matches_rebound_for_a_regular_two_planet_system():
    # R2: REBOUND's built-in MEGNO (init_megno) integrated with WHFast; our value
    # from the sampled variational-vector norms agrees to 1e-3 at t = 500.
    rebound = pytest.importorskip("rebound")
    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=1e-3, a=1.0, e=0.1)
    sim.add(m=1e-3, a=1.6, e=0.05)
    sim.integrator = "whfast"
    sim.dt = 0.01
    sim.init_megno(seed=1)
    times = np.linspace(0.0, 500.0, 5001)
    norms = []
    for time in times:
        sim.integrate(time, exact_finish_time=1)
        norms.append(np.sqrt(sum(p.x**2 + p.y**2 + p.z**2 + p.vx**2 + p.vy**2 + p.vz**2 for p in _variational_particles(sim))))
    assert megno(times, np.array(norms)) == pytest.approx(sim.megno(), abs=1e-3)


def test_lyapunov_estimator_recovers_the_cat_map_exponent():
    # R1: Arnold's cat map (x, y) -> (2x + y, x + y) mod 1 has Lyapunov exponent
    # ln((3 + sqrt 5) / 2) = 0.9624 per iterate. Rosenstein's estimate from 8000
    # iterates, fitted over lags 1-3 before saturation, is within 3 % (measured
    # 0.8 %). A window starting at 0 is a fraction of the lag range, as documented.
    n = 8000
    xy = np.empty((n, 2))
    xy[0] = [0.1234567, 0.7654321]
    for i in range(1, n):
        x, y = xy[i - 1]
        xy[i] = [(2 * x + y) % 1.0, (x + y) % 1.0]
    r = np.column_stack([xy, np.zeros(n)])
    exponent, *_ = lyapunov_exponent_from_statevectors(
        r, np.zeros_like(r), dt=1.0, theiler_window=5, max_horizon=4, fit_window=(1 / 3, 1.0)
    )
    assert exponent == pytest.approx(np.log((3 + np.sqrt(5)) / 2), rel=0.03)

    *_, diagnostics = lyapunov_exponent_from_statevectors(
        r, np.zeros_like(r), dt=1.0, theiler_window=5, max_horizon=4, fit_window=(0.0, 0.5)
    )
    assert (diagnostics["fit_tmin"], diagnostics["fit_tmax"]) == (0.0, 1.5)


def test_calculate_errors_on_a_uniform_grid_ignores_nans():
    # R1: for 0..1000 the 5 % and 95 % order statistics are 50 and 950 around a
    # median of 500; trailing NaNs must not move them. Exact.
    errors, median = calculate_errors(np.r_[np.arange(1001.0), [np.nan] * 200], CI=0.05)
    assert errors == [450.0, 450.0]
    assert median == [500.0]


def test_bounding_cube_is_centred_and_sized_by_the_longest_extent():
    # R1: points spanning 10 x 4 x 2 about (1, 2, 3) give a cube of side 10 + 2 pad.
    points = np.array([[-4.0, 0.0, 2.0], [6.0, 4.0, 4.0], [1.0, 2.0, 3.0]])
    lower, upper = find_smallest_bounding_cube(points, pad=1.0)
    np.testing.assert_allclose(lower, [1.0 - 6.0, 2.0 - 6.0, 3.0 - 6.0])
    np.testing.assert_allclose(upper, [1.0 + 6.0, 2.0 + 6.0, 3.0 + 6.0])


@pytest.mark.parametrize("distribution", ["uniform", "random"])
def test_sphere_vectors_are_area_uniform(distribution):
    # R1 (Archimedes): a direction uniform on the sphere has z uniform on
    # [-1, 1] and azimuth uniform on [0, 2 pi). Norms equal the magnitude to
    # 1e-12; KS p-values exceed 0.01 for 20,000 seeded draws.
    vectors = generate_sphere_vectors(20000, 13.0, seed=7, distribution=distribution)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 13.0, rtol=1e-12)
    z = vectors[:, 2] / 13.0
    azimuth = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi)
    assert stats.kstest(z, stats.uniform(loc=-1, scale=2).cdf).pvalue > 0.01
    assert stats.kstest(azimuth, stats.uniform(loc=0, scale=2 * np.pi).cdf).pvalue > 0.01


def test_perturbation_radii_follow_their_distributions():
    # R1: a point uniform in a ball of radius R has P(rho < s) = (s/R)^3; a
    # shell sample has rho = R; isotropic normal components have
    # rho^2 / sigma^2 ~ chi^2(3). KS p-values exceed 0.01 for 5,000 seeded draws.
    rng = np.random.default_rng(11)
    r0, v0 = np.array([7000e3, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0])
    samples = [perturb_state_3d(r0, v0, pos_scale=100.0, vel_scale=0.1, pos_distribution="uniform",
                                vel_distribution="normal", rng=rng) for _ in range(5000)]
    rho = np.array([np.linalg.norm(r - r0) for r, _ in samples])
    speed = np.array([np.linalg.norm(v - v0) for _, v in samples])
    assert stats.kstest(rho, lambda s: np.clip(s / 100.0, 0, 1) ** 3).pvalue > 0.01
    assert stats.kstest((speed / 0.1) ** 2, stats.chi2(3).cdf).pvalue > 0.01
    shell_r, _ = perturb_state_3d(r0, v0, pos_scale=50.0, pos_distribution="shell", rng=rng)
    assert np.linalg.norm(shell_r - r0) == pytest.approx(50.0, abs=1e-8)  # 7000 km + 50 m in float64


def test_segment_sphere_intersection_cases():
    # R1: against a unit sphere, a chord through the centre, a tangent line, a
    # segment that stops short, a miss at distance 1.5, and a segment wholly
    # inside all have known answers.
    p0 = np.array([[-2.0, 0.0, 0.0], [-2.0, 1.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, 1.5, 0.0], [-0.2, 0.0, 0.0]])
    p1 = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.5, 0.0, 0.0], [2.0, 1.5, 0.0], [0.2, 0.0, 0.0]])
    hits = segment_intersects_sphere(p0, p1, radius=1.0)
    np.testing.assert_array_equal(hits, [True, True, False, False, True])
