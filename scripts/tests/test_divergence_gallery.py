"""Independent two-body references for the full-period divergence example."""

import numpy as np
import pytest
from ssapy import Orbit, compute

from demos.analysis_dashboards.demo_divergence_plot import main


@pytest.fixture(scope="module")
def ensemble():
    return main(make_figures=False, fast=True)


def test_gallery_spans_a_resolved_period_against_analytic_circle(ensemble):
    """R1: r=R(cos nt,sin nt,0), n=sqrt(mu/R^3); 1 cm and 1 um/s.

    The position budget includes millimetre-scale integration error and GPS epoch
    rounding (one float64 ULP is about 0.24 us, or 0.74 mm at GEO).
    The time-span budget is 0.5 us; sampling must be at most 120 s.
    These are synthetic point-mass dynamics, not measured flight accuracy.
    """
    orbit = ensemble["orbit_nominal"]
    radius = 42164000.0
    mean_motion = np.sqrt(orbit.mu / radius**3)
    period = 2 * np.pi / mean_motion
    epochs = ensemble["times_gps"]
    elapsed = epochs - epochs[0]
    assert elapsed[-1] == pytest.approx(period, abs=5e-7)
    assert np.all(np.diff(epochs) > 0)
    assert np.max(np.diff(epochs)) <= 120.0
    assert ensemble["r_histories"].shape[0] >= 100
    phase = mean_motion * elapsed
    zero = np.zeros_like(phase)
    r_true = radius * np.column_stack((np.cos(phase), np.sin(phase), zero))
    v_true = radius * mean_motion * np.column_stack((-np.sin(phase), np.cos(phase), zero))
    np.testing.assert_allclose(ensemble["r_nominal_hist"], r_true, rtol=0, atol=0.01)
    np.testing.assert_allclose(ensemble["v_nominal_hist"], v_true, rtol=0, atol=1e-6)


def test_propagated_cloud_against_ssapy_kepler_solution(ensemble):
    """R2: DOP853 versus the analytic Kepler solver; 1 cm and 1 um/s.

    Compare every perturbed sample over the whole period. This validates the
    separate integration method for point-mass gravity, not perturbing forces.
    The bounds resolve the initial 100 m cloud by four orders of magnitude.
    """
    nominal = ensemble["orbit_nominal"]
    for r_hist, v_hist in zip(ensemble["r_histories"], ensemble["v_histories"]):
        orbit = Orbit(r_hist[0], v_hist[0], t=nominal.t, mu=nominal.mu)
        r_true, v_true = compute.rv(orbit, ensemble["times_gps"])
        np.testing.assert_allclose(r_hist, r_true, rtol=0, atol=0.01)
        np.testing.assert_allclose(v_hist, v_true, rtol=0, atol=1e-6)
