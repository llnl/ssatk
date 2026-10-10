#!/usr/bin/env python3
"""
Demo: Van Allen radiation belts as transparent torus surfaces.

Outputs
-------
demo_van_allen.html / .png        — oblique view
demo_van_allen_equatorial.html/.png — equatorial cross-section view
demo_van_allen_polar.html/.png      — polar view

Run
---
    python -m demos.space_environment.demo_van_allen
"""

GALLERY_CATEGORY = "space_environment"

import os
import sys
from ssatk.plots.figpath import figpath
from ssatk.plots.van_allen_plot_3d import plot_van_allen_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=True):
    if UNDER_PYTEST:
        make_figures = False
    if not make_figures:
        return {"figures": [], "skipped": True, "reason": "figures_disabled"}

    figures = []

    # 1. Classic oblique
    figures.append(plot_van_allen_3d(
        title        = "Van Allen Radiation Belts — IGRF 2025",
        texture_path = "auto",
        elev=25, azim=-55,
        save_path    = figpath("demo_van_allen"),
    ))
    print("Saved: demo_van_allen.html/.png")

    # 2. Equatorial — best view of torus cross-section
    figures.append(plot_van_allen_3d(
        title        = "Van Allen Belts — equatorial view",
        texture_path = "auto",
        elev=5, azim=0,
        save_path    = figpath("demo_van_allen_equatorial"),
    ))
    print("Saved: demo_van_allen_equatorial.html/.png")

    # 3. Polar
    figures.append(plot_van_allen_3d(
        title        = "Van Allen Belts — polar view",
        texture_path = "auto",
        elev=80, azim=0,
        save_path    = figpath("demo_van_allen_polar"),
    ))
    print("Saved: demo_van_allen_polar.html/.png")
    return {"figures": figures, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
