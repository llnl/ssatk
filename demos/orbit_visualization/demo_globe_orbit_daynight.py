#!/usr/bin/env python3
"""
Demo: globe orbit plot with day/night terminator.

This uses one low-Earth orbit and saves an interactive HTML file showing the
trajectory, sunlit hemisphere, night side, and satellite marker.
"""

GALLERY_CATEGORY = "orbit_visualization"

import os
import sys
from pathlib import Path

from ssatk.plots.figpath import figpath
from ssatk.plots.globe_orbit_daynight_plotly import plot_globe_orbit_daynight_plotly

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    save_path = None
    if make_figures:
        save_path = figpath("demo_globe_orbit_daynight.html")

    fig = plot_globe_orbit_daynight_plotly(
        a_km=7000.0,
        e=0.001,
        inc_deg=51.6,
        raan_deg=20.0,
        argp_deg=0.0,
        nu0_deg=10.0,
        sat_name="Demo LEO",
        n_orbits=0.25 if fast else 1.0,
        n_steps=120 if fast else 800,
        save_path=save_path,
        show_sun_body=False,
    )

    if save_path:
        print(f"Saved: {save_path}")

    return {"figure": fig, "n_traces": len(fig.data), "html": save_path}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
