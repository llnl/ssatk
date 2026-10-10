#!/usr/bin/env python3
"""Demo: 3D Moon surface plot with star background."""

GALLERY_CATEGORY = "orbit_visualization"

import os
import sys

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv
from ssatk.plots.figpath import figpath
from ssatk.plots.moon_plot_3d import moon_plot_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

def main(make_figures=True, fast=False):
    if UNDER_PYTEST:
        make_figures = False

    t0  = Time("2025-01-01T00:00:00", scale="utc")
    t0_gps = t0.gps

    # Lunar orbit — low lunar orbit at 1000km altitude
    orbit = Orbit.fromKeplerianElements(
        a           = (1738e3 + 1000e3),
        e           = 0.01,
        i           = np.radians(90.0),
        pa          = np.radians(0.0),
        raan        = np.radians(0.0),
        trueAnomaly = np.radians(0.0),
        t           = t0_gps,
    )

    n_samples = 150 if fast else 500
    period = 2 * np.pi * np.sqrt(orbit.a**3 / 4.9048695e12)
    t_arr  = t0_gps + np.linspace(0, period * 3, n_samples)
    from astropy.time import Time as ATime
    r_arr, _ = rv(orbit, ATime(t_arr, format='gps'))
    r_arr    = np.array(r_arr)

    if make_figures:
        # Static view
        fig, ax = moon_plot_3d(
            r         = r_arr,
            t         = t_arr,
            title     = "Low Lunar Orbit — 1000km altitude",
            save_path = figpath("demo_moon_plot"),
            elev      = 30,
            azim      = 45,
        )
        print(f"Saved: {figpath('demo_moon_plot')}")

        # Polar view
        fig, ax = moon_plot_3d(
            r         = r_arr,
            t         = t_arr,
            title     = "Low Lunar Orbit — polar view",
            save_path = figpath("demo_moon_plot_polar"),
            elev      = 60,
            azim      = 0,
        )
        print(f"Saved: {figpath('demo_moon_plot_polar')}")

        # Moon only — no orbit
        fig, ax = moon_plot_3d(
            title     = "Moon surface",
            save_path = figpath("demo_moon_surface"),
            elev      = 20,
            azim      = 30,
        )
        print(f"Saved: {figpath('demo_moon_surface')}")

    return {
        "orbit": orbit,
        "r": r_arr,
        "t": t_arr,
        "figures_created": bool(make_figures),
    }

if __name__ == "__main__":
    main(make_figures=True, fast=False)
