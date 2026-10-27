#!/usr/bin/env python3
"""
Demo: Official IRENE AE9/AP9 mean radiation-belt flux in three views.
Set IRENE_HOME for official flux; otherwise show illustrative dipole L-shells.

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
from ssatk.io.irene import find_irene_runtime
from ssatk.plots.figpath import figpath
from ssatk.plots.van_allen_plot_3d import plot_van_allen_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=True):
    if UNDER_PYTEST:
        make_figures = False
    if not make_figures:
        return {"figures": [], "skipped": True, "reason": "figures_disabled"}

    irene_available = find_irene_runtime() is not None
    if not irene_available:
        print("IRENE unavailable: showing illustrative dipole L-shells without particle flux; set IRENE_HOME")
    model_label = "IRENE AE9/AP9 mean flux" if irene_available else "illustrative L-shells, no particle flux"

    figures = []

    # 1. Classic oblique
    # gallery-call: demo_van_allen.html
    figures.append(plot_van_allen_3d(
        title        = f"Van Allen belts - {model_label}",
        texture_path = "auto",
        epoch        = "2025-07-02T12:34:56Z",
        radiation_model = "irene",
        belt_style   = "flux" if irene_available else "lshell",
        elev=25, azim=-55,
        save_path    = figpath("demo_van_allen"),
    ))
    # gallery-end
    print("Saved: demo_van_allen.html/.png")

    # 2. Equatorial — best view of torus cross-section
    # gallery-call: demo_van_allen_equatorial.html
    figures.append(plot_van_allen_3d(
        title        = f"Van Allen belts - equatorial view; {model_label}",
        texture_path = "auto",
        epoch        = "2025-07-02T12:34:56Z",
        radiation_model = "irene",
        belt_style   = "flux" if irene_available else "lshell",
        elev=5, azim=0,
        save_path    = figpath("demo_van_allen_equatorial"),
    ))
    # gallery-end
    print("Saved: demo_van_allen_equatorial.html/.png")

    # 3. Polar
    # gallery-call: demo_van_allen_polar.html
    figures.append(plot_van_allen_3d(
        title        = f"Van Allen belts - polar view; {model_label}",
        texture_path = "auto",
        epoch        = "2025-07-02T12:34:56Z",
        radiation_model = "irene",
        belt_style   = "flux" if irene_available else "lshell",
        elev=80, azim=0,
        save_path    = figpath("demo_van_allen_polar"),
    ))
    # gallery-end
    print("Saved: demo_van_allen_polar.html/.png")
    return {"figures": figures, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
