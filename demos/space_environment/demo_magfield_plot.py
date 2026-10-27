#!/usr/bin/env python3
"""
Demo: Earth magnetic field lines + official IRENE AE9/AP9 radiation belts.
Set IRENE_HOME for official flux; otherwise show illustrative IGRF L-shells.

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

from ssatk.io.irene import find_irene_runtime
from ssatk.plots.figpath import figpath
from ssatk.plots.magfield_plot_3d import plot_magfield_3d
from demos.space_environment._magfield_video import save_field_videos

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def _browser_unavailable_errors():
    """Selenium exceptions that mean no browser or driver could be started.

    Selenium raises NoSuchDriverException when no driver matches the
    configured binary, and SessionNotCreatedException when the launched
    process is not a working Chrome. The import is lazy so the module loads
    without selenium; an ``except`` clause's expression is only evaluated
    when reached, after the ImportError clause has already been tried.
    """
    from selenium.common.exceptions import NoSuchDriverException, SessionNotCreatedException
    return (NoSuchDriverException, SessionNotCreatedException)


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if not make_figures:
        return {"figure": None, "skipped": True, "reason": "figures_disabled"}

    irene_available = find_irene_runtime() is not None
    if not irene_available:
        print("IRENE unavailable: showing illustrative IGRF L-shells without particle flux; set IRENE_HOME")

    try:
        # gallery-call: *.html | *.png
        fig, stats = plot_magfield_3d(
            epoch           = "2025-07-02T12:34:56Z",
            fidelity        = "draft" if fast else "high",
            external_field  = None,
            use_omni        = False,
            belt_style      = "flux" if irene_available else "igrf",
            radiation_model = "irene",
            show_magnetopause = False,
            show_provenance = False,
            split_topology  = False,
            title           = ("Earth internal field - IGRF 2025; " +
                               ("IRENE AE9/AP9 mean flux" if irene_available else
                                "illustrative L-shells, no particle flux")),
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
    try:
        # gallery-call: demo_magfield_lines.mp4 | demo_magfield_belts.mp4
        videos = save_field_videos(figpath("demo_magfield_plot.html"), fast=fast)
        # gallery-end
    except ImportError:
        return {"figure": fig, "stats": stats, "videos": None, "skipped": True,
                "reason": "Selenium is not installed; install ssatk[browser,video] "
                          "to capture the field-line/belt camera tours."}
    except _browser_unavailable_errors():
        return {"figure": fig, "stats": stats, "videos": None, "skipped": True,
                "reason": "No usable Chrome/driver could be started; install "
                          "Chrome/Chromium or set SSATK_CHROME_BINARY."}
    print("Saved: demo_magfield_plot.html, demo_magfield_plot.png and field videos")
    return {"figure": fig, "stats": stats, "videos": videos, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
