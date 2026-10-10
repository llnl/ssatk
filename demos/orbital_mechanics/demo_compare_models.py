"""
Demo: run compare_models (SSAPy accel ladder) and optionally save dashboard figures.
"""

import os
import sys

import numpy as np

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    from astropy.time import Time
    import ssapy

    from ssatk.orbital_mechanics.orbital_accel_model_comparisons import compare_models
    from ssatk.plots.plotutils import figsave

    t0 = Time("2026-01-21T00:00:00", scale="utc")
    r0_m = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0_mps = np.array([0.0, 7.5e3, 1.0e3], dtype=float)
    orbit = ssapy.Orbit(r=r0_m, v=v0_mps, t=t0)

    dt_s = 300.0 if fast else 60.0
    duration_s = 600.0 if fast else 3600.0
    times = np.arange(0.0, duration_s + dt_s, dt_s, dtype=float)

    out = compare_models(
        orbit=orbit,
        times=times,
        assume_times="offset",
        ode_kwargs={"rtol": 1e-8, "atol": 1e-10} if fast else {"rtol": 1e-10, "atol": 1e-12},
        reference=None,
        plot_title="Demo: SSAPy accel ladder divergences",
        show_legend=True,
        epsilon_m=1e-3,
        max_rungs=2 if fast else None,
    )

    dash = out["dashboard"]
    fig_time = None
    fig_rung = None
    if isinstance(dash, dict):
        fig_time = dash.get("figure_time_domain", None)
        fig_rung = dash.get("figure_rung_summary", None)
        if (fig_time is None or fig_rung is None) and isinstance(dash.get("figures", None), list):
            figs = dash["figures"]
            if len(figs) >= 2:
                fig_time = figs[0]
                fig_rung = figs[1]

    if fig_time is None or fig_rung is None:
        raise RuntimeError("Expected dashboard figures not found in output.")

    if make_figures:
        out_dir = "figures"
        figsave(fig_time, out_dir + "/accel_ladder_time_domain.jpg", dpi=200)
        figsave(fig_rung, out_dir + "/accel_ladder_rung_summary.jpg", dpi=200)
        print("Saved via ssatk figure output root:")
        print("  " + out_dir + "/accel_ladder_time_domain.jpg")
        print("  " + out_dir + "/accel_ladder_rung_summary.jpg")

    return {"out": out, "figure_time_domain": fig_time, "figure_rung_summary": fig_rung}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
