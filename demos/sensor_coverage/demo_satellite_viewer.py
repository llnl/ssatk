#!/usr/bin/env python3
"""
Demo: self-contained Three.js satellite viewer export.

This builds the browser-based satellite viewer HTML artifact and a static PNG
companion. The resulting page can load satellite catalogs from JSON, CSV, or
HDF5 files. Both outputs go under the standard SSATK figure directory.
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import sys
from pathlib import Path

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv
from ssapy.propagator import KeplerianPropagator

from ssatk.constants import WGS84_A_KM
from ssatk.plots.build_satellite_viewer import build
from ssatk.plots.figpath import figpath
from ssatk.plots.globe_plot import globe_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    if not make_figures:
        return {
            "html": None,
            "png": None,
            "skipped": True,
            "reason": "figures_disabled",
        }

    html_path = Path(figpath("demo_satellite_viewer.html"))
    written = build(out_path=str(html_path), verbose=True)
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    radius_m = WGS84_A_KM * 1000.0 + 400e3
    inclination = np.radians(51.6)
    orbit = Orbit.fromKeplerianElements(
        radius_m, 0.0, inclination, 0.0, 0.0, 0.0, t=epoch.gps,
    )
    duration = orbit.period
    times = Time(
        epoch.gps + np.linspace(0.0, duration, 120 if fast else 480),
        format="gps",
    )
    r, _v = rv(orbit, times, propagator=KeplerianPropagator())
    png_path = Path(figpath("demo_satellite_viewer.png"))
    globe_plot(
        np.asarray(r),
        title="Satellite viewer: ISS-like LEO",
        labels=["ISS-like LEO"],
        orbit_colors=["#66d9ef"],
        save_path=str(png_path),
        scale=8.0,
    )
    return {"html": str(written), "png": str(png_path), "skipped": False}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
