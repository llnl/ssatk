#!/usr/bin/env python3
"""
Demo: animated Earth-Sun-Moon Plotly scene.

This demo shows the compact API for ``earth_sun_plot``: copy the default
configuration, change only the user-facing values, build the figure, and save
an interactive HTML file under the standard SSATK figure directory.
"""

GALLERY_CATEGORY = "space_environment"

import os
import sys
from pathlib import Path

from ssatk.plots.earth_sun_plot import DEFAULT_CFG, build_figure, build_static_figure, _shrink_floats
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    cfg = DEFAULT_CFG.copy()
    cfg["start_year"] = 2025
    cfg["start_month"] = 6
    cfg["start_day"] = 1
    # Time resolution matters more than mesh resolution for this demo: the
    # point is to see Earth and the Moon move smoothly through the year.  Fast
    # gallery mode uses roughly two-day cadence; full mode uses daily cadence.
    cfg["n_frames"] = 180 if fast else 365
    cfg["frame_duration_ms"] = 45 if fast else 35
    cfg["sphere_resolution"] = 20 if fast else 36
    cfg["hero_resolution"] = 64 if fast else 180
    cfg["show_stars"] = False
    cfg["show_labels"] = True
    cfg["show_moon_trail"] = True

    animated_fig = build_figure(cfg)
    static_fig = build_static_figure(cfg)

    outputs = {
        "animated_fig": animated_fig,
        "static_fig": static_fig,
        "n_frames": len(animated_fig.frames),
        "n_traces": len(animated_fig.data),
        "html": None,
    }

    if make_figures:
        html_path = Path(figpath("demo_earth_sun_plot.html"))
        _shrink_floats(animated_fig).write_html(str(html_path))
        outputs["html"] = str(html_path)
        print(f"Saved: {html_path}")

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
