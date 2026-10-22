#!/usr/bin/env python3
"""
Demo: Earth magnetic field lines + Van Allen radiation belts (IGRF 2025).

Output
------
demo_magfield_plot.html  — interactive 3D (rotate/zoom in browser)
demo_magfield_plot.png   — static preview
demo_magfield_lines.mp4  — camera tour of the internal field
demo_magfield_belts.mp4  — camera tour with illustrative L-shell belts

The videos move the camera around a fixed 2025 IGRF model; they do not
simulate field evolution or particle motion. Export needs ssatk[browser,video]
and Chrome. Set SSATK_CHROME_BINARY to select the browser.

Run
---
    python -m demos.space_environment.demo_magfield_plot
"""

GALLERY_CATEGORY = "space_environment"

import sys
import os

from ssatk.plots.figpath import figpath
from ssatk.plots.magfield_plot_3d import plot_magfield_3d
from demos.space_environment._magfield_video import save_field_videos

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if not make_figures:
        return {"figure": None, "skipped": True, "reason": "figures_disabled"}

    try:
        # gallery-call: *.html | *.png
        fig, stats = plot_magfield_3d(
            epoch           = 2025.0,
            fidelity        = "draft" if fast else "high",
            external_field  = None,
            use_omni        = False,
            belt_style      = "igrf",
            show_magnetopause = False,
            show_provenance = False,
            split_topology  = False,
            title           = "Earth internal field — IGRF 2025; illustrative L-shell belts",
            texture_path    = "auto",
            elev            = 8,
            azim            = -15,
            max_r_re        = 8.0,
            show_van_allen  = True,
            save_path       = figpath("demo_magfield_plot"),
        )
        # gallery-end
    except ImportError as exc:
        print(f"Skipped: {exc}")
        return {"figure": None, "skipped": True, "reason": str(exc)}
    # A PNG preview belongs on the gallery card; the HTML remains interactive.
    # Fail visibly if the configured publishing environment cannot export it.
    fig.write_image(figpath("demo_magfield_plot.png"), width=1200, height=900)
    # gallery-call: demo_magfield_lines.mp4 | demo_magfield_belts.mp4
    videos = save_field_videos(figpath("demo_magfield_plot.html"), fast=fast)
    # gallery-end
    print("Saved: demo_magfield_plot.html, demo_magfield_plot.png and field videos")
    return {"figure": fig, "stats": stats, "videos": videos, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
