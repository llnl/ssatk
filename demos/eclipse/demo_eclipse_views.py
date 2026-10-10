#!/usr/bin/env python3
"""
Demo: eclipse appearance and space-view plotting.

This demo includes two complementary real events: the April 8, 2024
solar-eclipse animation over central Texas, and the April 15, 2014 total lunar
eclipse visible from Wisconsin near maximum eclipse.
"""

GALLERY_CATEGORY = "eclipse"

import os
import sys

from ssatk.plots.eclipse_appearance_strip import make_strip
from ssatk.plots.eclipse_space_view_plotly import (
    find_and_plot_eclipse,
    plot_2024_solar_eclipse_animated,
    plot_space_view_animated,
    plot_space_view_plotly,
)
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    outputs = {
        "strip_fig": None,
        "solar_strip_fig": None,
        "lunar_strip_fig": None,
        "search_fig": None,
        "lunar_search_fig": None,
        "space_fig": None,
        "lunar_space_fig": None,
        "space_anim_fig": None,
        "lunar_space_anim_fig": None,
        "solar_2024_fig": None,
        "solar_2024_stats": None,
        "stats": None,
        "lunar_stats": None,
        "files": [],
    }
    search_days = 730.0 if make_figures else 120.0
    lunar_event = "2014-04-15-total-lunar-wisconsin"

    solar_strip_path = figpath("eclipse/demo_eclipse_2024_solar_strip.png") if make_figures else None
    solar_strip_fig = make_strip(kind="solar", n_panels=5 if fast else 11, save_path=solar_strip_path)
    outputs["strip_fig"] = solar_strip_fig
    outputs["solar_strip_fig"] = solar_strip_fig
    if solar_strip_path:
        outputs["files"].append(solar_strip_path)
        print(f"Saved: {solar_strip_path}")

    lunar_search_path = figpath("eclipse/demo_eclipse_2014_lunar_wisconsin.png") if make_figures else None
    lunar_search_fig, lunar_stats = find_and_plot_eclipse(
        mode="lunar",
        search_days=search_days,
        event=lunar_event,
        save_path=lunar_search_path,
        verbose=not fast,
    )
    outputs["search_fig"] = lunar_search_fig
    outputs["lunar_search_fig"] = lunar_search_fig
    outputs["lunar_strip_fig"] = lunar_search_fig
    outputs["stats"] = lunar_stats
    outputs["lunar_stats"] = lunar_stats
    if lunar_search_path:
        outputs["files"].append(lunar_search_path)
        print(f"Saved: {lunar_search_path}")

    if make_figures:
        lunar_space_path = figpath("eclipse/demo_eclipse_space_2014_lunar_wisconsin.html")
        lunar_space_fig = plot_space_view_plotly(
            mode="lunar",
            search_days=search_days,
            event=lunar_event,
            save_path=lunar_space_path,
            verbose=not fast,
        )
        outputs["space_fig"] = lunar_space_fig
        outputs["lunar_space_fig"] = lunar_space_fig
        outputs["files"].append(lunar_space_path)

        lunar_anim_path = figpath("eclipse/demo_eclipse_space_2014_lunar_wisconsin_animated.html")
        lunar_space_anim_fig = plot_space_view_animated(
            mode="lunar",
            search_days=search_days,
            event=lunar_event,
            n_frames=8 if fast else 18,
            n_lat=48 if fast else 72,
            n_lon=96 if fast else 144,
            save_path=lunar_anim_path,
            verbose=not fast,
        )
        outputs["space_anim_fig"] = lunar_space_anim_fig
        outputs["lunar_space_anim_fig"] = lunar_space_anim_fig
        outputs["files"].append(lunar_anim_path)

        solar_2024_path = figpath("eclipse/demo_eclipse_space_2024_solar_animated.html")
        solar_2024_fig, solar_2024_stats = plot_2024_solar_eclipse_animated(
            save_path=solar_2024_path,
            n_frames=8 if fast else 26,
            n_lat=48 if fast else 144,
            n_lon=96 if fast else 288,
            show_stars=not fast,
            verbose=not fast,
        )
        outputs["solar_2024_fig"] = solar_2024_fig
        outputs["solar_2024_stats"] = solar_2024_stats
        outputs["files"].append(solar_2024_path)
    else:
        outputs["space_fig"] = None
        outputs["lunar_space_fig"] = None
        outputs["space_anim_fig"] = None
        outputs["lunar_space_anim_fig"] = None
        outputs["solar_2024_fig"] = None
        outputs["solar_2024_stats"] = None

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
