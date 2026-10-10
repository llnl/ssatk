#!/usr/bin/env python3
"""
Demo: Earth magnetic field lines + Van Allen radiation belts (IGRF 2025).

Output
------
demo_magfield_plot.html  — interactive 3D (rotate/zoom in browser)

Run
---
    python -m demos.space_environment.demo_magfield_plot
"""

GALLERY_CATEGORY = "space_environment"

import sys
import os

from ssatk.plots.figpath import figpath
from ssatk.plots.magfield_plot_3d import plot_magfield_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

def main(make_figures=True):
    if UNDER_PYTEST:
        make_figures = False
    if not make_figures:
        return {"figure": None, "skipped": True, "reason": "figures_disabled"}

    try:
        fig, stats = plot_magfield_3d(
            epoch           = 2025.0,
            title           = "Earth Magnetic Field Lines & Van Allen Belts — IGRF 2025",
            texture_path    = "auto",
            elev            = 8,
            azim            = -15,
            max_r_re        = 15.0,
            show_van_allen  = True,
            save_path       = figpath("demo_magfield_plot"),
        )
    except ImportError as exc:
        print(f"Skipped: {exc}")
        return {"figure": None, "skipped": True, "reason": str(exc)}
    print("Saved: demo_magfield_plot.html")
    return {"figure": fig, "stats": stats, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
