import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from ssatk.plots.build_dashboard import build_dashboard
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "analysis_dashboards"


def panel_altitude(ax, fig, t_min, alt_km):
    ax.plot(t_min, alt_km, lw=2.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True)
    ax.set_title("Altitude vs Time")


def panel_xy(ax, fig, x, y):
    ax.scatter(x, y, s=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("XY")


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    n_t = 200 if fast else 500
    n_xy = 250 if fast else 1000

    t = np.linspace(0, 90, n_t)
    alt = 400 + 20 * np.sin(2 * np.pi * t / 90)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(n_xy)
    y = rng.standard_normal(n_xy)

    fig, axes, meta = build_dashboard(
        panels=[
            {"loc": (0, 0), "render": panel_altitude, "kwargs": {"t_min": t, "alt_km": alt}},
            {"loc": (0, 1), "render": panel_xy, "kwargs": {"x": x, "y": y}},
        ],
        nrows=1,
        ncols=2,
        figsize=(12, 4),
        show=make_figures,
    )

    if make_figures:
        figsave(fig, "build_dashboard_overview.jpg")
    else:
        plt.close(fig)

    return {"fig": fig, "axes": axes, "meta": meta}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
