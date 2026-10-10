"""
Demo script for divergence_plot - visualizing position error distributions.

In pytest mode:
- uses fewer samples / snapshots
- skips GIF creation and frame writing
- still exercises core propagation and divergence_plot/divergence_gif interfaces
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from ssapy import Orbit, compute
from ssapy.accel import AccelKepler
from ssapy.body import get_body
from ssapy.propagator import SciPyPropagator

from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.divergence_gif import divergence_gif
from ssatk.plots.figpath import ssatk_path

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "analysis_dashboards"
FIGDIR = "figures"
GIF_FRAME_DURATION_S = 1.0


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    n_samples = 12 if fast else 100
    duration_hours = 2.0 if fast else 24.0
    snapshot_interval_minutes = 60 if fast else 30
    n_snapshots = int((duration_hours * 60) / snapshot_interval_minutes) + 1

    r_nominal = np.array([42164000.0, 0.0, 0.0])
    v_nominal = np.array([0.0, 3074.66, 0.0])

    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t0_gps = t0.gps
    times_gps = np.linspace(t0_gps, t0_gps + duration_hours * 3600.0, n_snapshots)

    pos_radius = 100.0
    vel_radius = 1.0

    r_initial = []
    v_initial = []

    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        pos_direction = rng.standard_normal(3)
        pos_direction /= np.linalg.norm(pos_direction)
        pos_magnitude = pos_radius * np.cbrt(rng.uniform(0, 1))
        pos_error = pos_direction * pos_magnitude

        vel_direction = rng.standard_normal(3)
        vel_direction /= np.linalg.norm(vel_direction)
        vel_magnitude = vel_radius * np.cbrt(rng.uniform(0, 1))
        vel_error = vel_direction * vel_magnitude

        r_initial.append(r_nominal + pos_error)
        v_initial.append(v_nominal + vel_error)

    earth = get_body("earth")
    accel = AccelKepler(earth.mu)
    propagator = SciPyPropagator(accel)

    orbit_nominal = Orbit(r=r_nominal, v=v_nominal, t=t0_gps)
    r_nominal_hist, v_nominal_hist = compute.rv(orbit_nominal, times_gps, propagator)

    print(f"Propagating {n_samples} orbits for {duration_hours} hours with {n_snapshots} snapshots...")
    r_histories = []
    v_histories = []
    for i in range(n_samples):
        orbit = Orbit(r=r_initial[i], v=v_initial[i], t=t0_gps)
        r_hist, v_hist = compute.rv(orbit, times_gps, propagator)
        r_histories.append(r_hist)
        v_histories.append(v_hist)

    r_histories = np.array(r_histories)
    v_histories = np.array(v_histories)

    r_final = r_histories[:, -1, :]
    fig1 = orbit_plot(
        r_final,
        view="divergence",
        v_center=v_nominal_hist[-1],
        title=f"Position Errors at T+{duration_hours:.0f} hours (Median Center)",
        show=False,
        save=f"{FIGDIR}/divergence_plot_final_median" if make_figures else False,
    )
    fig2 = orbit_plot(
        r_final,
        view="divergence",
        r_center=r_nominal_hist[-1],
        v_center=v_nominal_hist[-1],
        title=f"Position Errors at T+{duration_hours:.0f} hours (Nominal Center)",
        show=False,
        save=f"{FIGDIR}/divergence_plot_final_explicit" if make_figures else False,
    )

    if make_figures:
        orbit_plot(
            np.transpose(r_histories, (1, 2, 0)),
            t=times_gps,
            view="orbit_divergence",
            r_moon=np.zeros_like(r_nominal_hist),
            title="Orbit divergence across propagated samples",
            show=False,
            save=f"{FIGDIR}/orbit_divergence_samples",
        )

    gif_results = {}
    if make_figures:
        gif_results["explicit"] = divergence_gif(
            r_histories=r_histories,
            times_gps=times_gps,
            output_path=ssatk_path(f"{FIGDIR}/divergence_test_explicit.gif"),
            r_nominal_hist=r_nominal_hist,
            v_nominal_hist=v_nominal_hist,
            duration=GIF_FRAME_DURATION_S,
            vmin=-50,
            vmax=50,
        )
        gif_results["median"] = divergence_gif(
            r_histories=r_histories,
            times_gps=times_gps,
            output_path=ssatk_path(f"{FIGDIR}/divergence_test_median.gif"),
            duration=GIF_FRAME_DURATION_S,
        )
    else:
        # Pytest-safe: just exercise function with minimal work, no file output
        pass

    plt.close("all")
    return {
        "r_histories": r_histories,
        "v_histories": v_histories,
        "fig_median": fig1,
        "fig_nominal": fig2,
        "gif_results": gif_results,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
