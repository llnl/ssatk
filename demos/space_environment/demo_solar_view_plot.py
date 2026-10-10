#!/usr/bin/env python3
"""
Demo: heliocentric solar-system Plotly scene.

The call sequence is intentionally top-to-bottom: configure the date and visible
bodies, build the Plotly figure, then save the HTML artifact.
"""

GALLERY_CATEGORY = "space_environment"

import os
import sys
from pathlib import Path

from ssatk.plots.figpath import figpath
from ssatk.plots.solar_view_plot import DEFAULT_CFG, build_figure

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    cfg = DEFAULT_CFG.copy()
    cfg["sol_year"] = 2025
    cfg["sol_month"] = 6
    cfg["sol_show_mercury"] = True
    cfg["sol_show_venus"] = True
    cfg["sol_show_earth"] = True
    cfg["sol_show_mars"] = True
    cfg["sol_show_jupiter"] = True
    cfg["sol_show_saturn"] = True
    cfg["sol_show_uranus"] = True
    cfg["sol_show_neptune"] = True
    cfg["sol_show_moon"] = True
    cfg["sol_show_trails"] = True
    cfg["sol_show_stars"] = True
    cfg["star_mag_limit"] = 5.5 if fast else 6.5
    cfg["sphere_resolution"] = 16 if fast else 50

    fig = build_figure(cfg)
    outputs = {"figure": fig, "n_traces": len(fig.data), "html": None}

    if make_figures:
        html_path = Path(figpath("demo_solar_view_plot.html"))
        fig.write_html(str(html_path))
        alias_path = Path(figpath("demo_solar_view.html"))
        fig.write_html(str(alias_path))
        outputs["html"] = str(html_path)
        outputs["html_alias"] = str(alias_path)
        print(f"Saved: {html_path}")
        print(f"Saved: {alias_path}")

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
