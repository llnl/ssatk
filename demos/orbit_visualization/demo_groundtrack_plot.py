"""
Demo script for ssatk groundtrack_plot.

Pytest-safe mode:
- builds a smaller set of synthetic tracks
- does not save figures by default
"""

GALLERY_CATEGORY = "orbit_visualization"

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from ssatk.plots.groundtrack_plot import groundtrack_plot
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def make_circular_orbit_track(alt_km=500.0, inc_deg=51.6, npts=1200, n_orbits=3.0, t0=0.0):
    mu = 3.986004418e14
    re = 6378.137e3
    a = re + alt_km * 1e3
    inc = np.radians(inc_deg)
    n = np.sqrt(mu / a ** 3)

    t = np.linspace(0.0, n_orbits * 2.0 * np.pi / n, int(npts))
    nu = n * t
    x_pf = a * np.cos(nu)
    y_pf = a * np.sin(nu)

    x = x_pf
    y = y_pf * np.cos(inc)
    z = y_pf * np.sin(inc)
    r = np.column_stack((x, y, z))
    return r, t


def save_demo(fig, name):
    figsave(fig, f"{name}")


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    npts = 300 if fast else 1200
    n_orbits = 1.0 if fast else 3.0

    r1, t1 = make_circular_orbit_track(alt_km=500.0, inc_deg=28.5, npts=npts, n_orbits=n_orbits)
    r2, t2 = make_circular_orbit_track(alt_km=500.0, inc_deg=63.4, npts=npts, n_orbits=n_orbits)
    r3, t3 = make_circular_orbit_track(alt_km=700.0, inc_deg=98.0, npts=npts, n_orbits=n_orbits)

    ground_stations = None

    fig = groundtrack_plot(
        [r1, r2, r3],
        [t1, t2, t3],
        title="Demo: Multiple Orbits with Custom Styles",
        ground_stations=ground_stations,
        labels=["LEO 28.5 deg", "LEO 63.4 deg", "SSO 98 deg"],
        orbit_colors=["tab:blue", "tab:orange", "tab:green"],
        linestyles=["-", "--", ":"],
        central_longitude=0,
        relabel_xticks=True,
    )

    if make_figures:
        save_demo(fig, "demo_groundtrack_04_multi_custom_styles")
    else:
        plt.close(fig)

    return {"fig": fig, "tracks": [(r1, t1), (r2, t2), (r3, t3)]}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
