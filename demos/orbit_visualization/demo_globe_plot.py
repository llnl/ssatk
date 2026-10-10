#!/usr/bin/env python3
import os
import sys

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.constants import EARTH_RADIUS
from ssatk.plots.orbit_plot import orbit_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "orbit_visualization"
FIGDIR = "figures"


def main(make_figures=None, make_video=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_video is None:
        make_video = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")

    # ISS-like LEO orbit
    a_iss = EARTH_RADIUS + 420e3
    orbit1 = Orbit.fromKeplerianElements(
        a=a_iss,
        e=0.001,
        i=np.radians(51.6),
        pa=np.radians(40.0),
        raan=np.radians(20.0),
        trueAnomaly=0.0,
        t=t0,
    )

    # GPS-like MEO orbit
    a_gps = EARTH_RADIUS + 20200e3
    orbit2 = Orbit.fromKeplerianElements(
        a=a_gps,
        e=0.01,
        i=np.radians(55.0),
        pa=np.radians(120.0),
        raan=np.radians(240.0),
        trueAnomaly=np.radians(45.0),
        t=t0,
    )

    if fast:
        duration_hr = 4.0
        dt_s = 300.0
        video_frames = 6
        fps = 8
        globe_scale = 16.0
    else:
        duration_hr = 12.0
        dt_s = 180.0
        video_frames = 8
        fps = 8
        globe_scale = 16.0

    times_gps = t0.gps + np.arange(0.0, duration_hr * 3600.0 + dt_s, dt_s)

    r1, _ = rv(orbit1, Time(times_gps, format="gps"))
    r2, _ = rv(orbit2, Time(times_gps, format="gps"))
    r1 = np.asarray(r1, dtype=float).reshape((-1, 3))
    r2 = np.asarray(r2, dtype=float).reshape((-1, 3))

    outputs = {
        "r1": r1,
        "r2": r2,
        "times_gps": times_gps,
    }

    if make_figures:
        static_path = f"{FIGDIR}/demo_globe_plot_two_orbits.png"
        orbit_plot(
            [r1, r2],
            t=[times_gps, times_gps],
            view="globe",
            title="Globe Plot Demo: ISS-like LEO and GPS-like MEO",
            c="black",
            labels=["ISS-like LEO", "GPS-like MEO"],
            orbit_colors=["cyan", "magenta"],
            globe_time=Time(times_gps[-1], format="gps"),
            scale=globe_scale,
            save=static_path,
        )
        outputs["static_plot"] = static_path

    if make_video:
        video_path = f"{FIGDIR}/demo_globe_plot_animation.mp4"
        orbit_plot(
            [r1, r2],
            t=[times_gps, times_gps],
            view="globe",
            title="Globe Plot Demo Animation",
            c="black",
            save=video_path,
            fps=fps,
            max_frames=video_frames,
            tail=max(6, video_frames // 4),
        )
        outputs["video"] = video_path

    return outputs


if __name__ == "__main__":
    main(make_figures=True, make_video=True, fast=False)
