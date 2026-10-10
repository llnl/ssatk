"""
Demo script showing how to call synthetic_orbit_population and orbit_stats_dashboard.
"""

GALLERY_CATEGORY = "analysis_dashboards"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from ssatk.orbital_mechanics.synthetic_orbit_population import synthetic_orbit_population
from ssatk.orbital_mechanics.orbital_comparison_stats import orbit_stats_dashboard
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def print_population_summary(out, *, top_k=8):
    pop = out["population"]
    meta = out["meta"]
    per = pop["per_orbit"]
    M = int(meta["M"])
    mode = str(meta.get("mode", "population"))
    baseline = str(meta.get("baseline", "nominal"))
    ref = int(meta.get("reference", 0))
    print(f"M={M}, mode={mode}, baseline={baseline}, reference={ref}")
    print("Available keys:", sorted(pop.keys())[:top_k])


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    M = 8 if fast else 40
    N = 600 if fast else 7200
    dt = 5.0 if fast else 1.0

    orbits, r_list, v_list, t_list, mu_si = synthetic_orbit_population(
        M=M,
        N=N,
        dt=dt,
        seed=1,
    )

    labels = [f"orb_{i}" for i in range(len(r_list))]

    out_pop = orbit_stats_dashboard(
        r_list,
        v_list=v_list,
        t_list=t_list,
        mu=mu_si,
        reference=0,
        baseline="nominal",
        mode="population",
        resample="intersection",
        n_resample=300 if fast else 3000,
        percentiles=(5, 25, 50, 75, 95),
        make_plots=make_figures,
        plot_title="Orbit dashboard (population mode, SI, SSAPy population)",
        time_unit="s",
        r_unit="m",
        v_unit="m/s",
        labels=labels,
        show_legend=True,
    )

    print_population_summary(out_pop, top_k=8)

    if make_figures:
        fig_pop = out_pop["figure"]
        figsave(fig_pop, "orbital_stats_dashboard_population.jpg")
        if fig_pop is not None:
            plt.close(fig_pop)

    return out_pop


if __name__ == "__main__":
    main(make_figures=True, fast=False)
