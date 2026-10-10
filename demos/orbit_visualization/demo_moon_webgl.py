#!/usr/bin/env python3
"""Demo: animated high-detail Moon orbit viewer in WebGL."""

GALLERY_CATEGORY = "orbit_visualization"
TITLE = "Animated Moon Orbit — WebGL"
DESCRIPTION = (
    "A portable, high-detail WebGL Moon with a physically sampled low lunar "
    "orbit, animated spacecraft markers, runtime JSON loading, and orbit/day "
    "timeline controls."
)

import os
import sys

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.constants import MOON_MU, MOON_RADIUS
from ssatk.plots.figpath import figpath
from ssatk.plots.moon_webgl import moon_webgl


UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    epoch = Time("2025-01-01T00:00:00", scale="utc")
    orbit = Orbit.fromKeplerianElements(
        a=MOON_RADIUS + 1_000_000.0,
        e=0.01,
        i=np.radians(90.0),
        pa=0.0,
        raan=np.radians(25.0),
        trueAnomaly=0.0,
        t=epoch.gps,
        mu=MOON_MU,
    )

    n_samples = 180 if fast else 540
    period = 2.0 * np.pi * np.sqrt(orbit.a**3 / MOON_MU)
    times = epoch.gps + np.linspace(0.0, period, n_samples)
    positions, _ = rv(orbit, Time(times, format="gps"))
    positions = np.asarray(positions, dtype=float)

    html_path = None
    skipped = not make_figures
    reason = "figures_disabled" if skipped else None
    if make_figures:
        html_path = figpath(
            "orbit_visualization/demo_moon_webgl_animation.html"
        )
        try:
            moon_webgl(
                r=positions,
                t=times,
                r_frame="moon_centered",
                title="Animated Low Lunar Orbit",
                animation_seconds=18.0,
                embed_assets=True,
                save_path=html_path,
            )
            print(f"Saved: {html_path}")
        except FileNotFoundError as exc:
            # The high-resolution baked maps are optional local data. Keep the
            # gallery run alive and report the exact preparation instructions.
            html_path = None
            skipped = True
            reason = str(exc)
            print(f"Skipped portable Moon WebGL export: {exc}")

    return {
        "orbit": orbit,
        "r": positions,
        "t": times,
        "html": html_path,
        "skipped": skipped,
        "reason": reason,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
