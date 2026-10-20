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
TITLE = "Rotating Earth and a 1,584-satellite Starlink-style constellation"
DESCRIPTION = ("A synthetic Walker-delta shell: 72 planes × 22 satellites, 550 km altitude, "
               "53° inclination. Two 30-second animations show the globe and global ground tracks "
               "over two orbital periods. These are design orbits, not operational Starlink positions.")


def main(make_figures=None, make_video=None, fast=None, start_time="2025-01-01T00:00:00"):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_video is None:
        make_video = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # Preserve an explicit Time object's scale; ISO strings are UTC.
    t0 = start_time if isinstance(start_time, Time) else Time(start_time, scale="utc")
    if not t0.isscalar or not np.isfinite(t0.gps):
        raise ValueError("start_time must be one finite absolute date")

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
        dt_s = 10.0
        earth_resolution = 240
    else:
        duration_hr = 12.0
        dt_s = 10.0
        earth_resolution = 240

    times_gps = t0.gps + np.arange(0.0, duration_hr * 3600.0 + dt_s, dt_s)

    r1, _ = rv(orbit1, Time(times_gps, format="gps"))
    r2, _ = rv(orbit2, Time(times_gps, format="gps"))
    r1 = np.asarray(r1, dtype=float).reshape((-1, 3))
    r2 = np.asarray(r2, dtype=float).reshape((-1, 3))
    close_view = dict(view="globe", c="black", figsize=(9, 9),
                      earth_resolution=earth_resolution, el=25.0, az=-20.0)

    outputs = {
        "r1": r1,
        "r2": r2,
        "times_gps": times_gps,
    }

    if make_figures:
        snapshot_path = "demo_globe_plot_textured_earth.png"
        # gallery-call: demo_globe_plot_textured_earth.png
        orbit_plot(r1, t=times_gps, title="Rotating Earth and a synthetic LEO orbit",
                   reference_index=0, save=snapshot_path,
                   **dict(close_view, earth_resolution=720))
        # gallery-end
        outputs["snapshot"] = snapshot_path
        static_path = "demo_globe_plot_two_orbits.png"
        # gallery-call: demo_globe_plot_two_orbits.png
        orbit_plot(
            [r1, r2],
            t=[times_gps, times_gps],
            view="globe",
            title="Globe Plot Demo: ISS-like LEO and GPS-like MEO",
            c="black",
            labels=["ISS-like LEO", "GPS-like MEO"],
            orbit_colors=["cyan", "magenta"],
            globe_time=Time(times_gps[-1], format="gps"),
            earth_resolution=earth_resolution,
            save=static_path,
        )
        # gallery-end
        outputs["static_plot"] = static_path

    if make_video:
        from ssatk.orbital_mechanics.constellations import walker_delta
        orbits = walker_delta(epoch=t0)
        constellation_times = t0.gps + np.linspace(0., 2*orbits[0].period, 721)
        constellation_r, _ = rv(orbits, Time(constellation_times, format="gps"))
        tracks = list(np.asarray(constellation_r))
        caption = "1,584 synthetic Starlink-style satellites · 550 km · 53°"
        # gallery-call: demo_globe_plot_animation.mp4
        orbit_plot(tracks, t=constellation_times, title=caption,
                   view="globe", c="black", figsize=(10, 9),
                   earth_resolution=240, el=25., az=-20., fps=10,
                   max_frames=300, tail=8, save="demo_globe_plot_animation.mp4")
        # gallery-end
        # gallery-call: demo_constellation_groundtracks.mp4
        orbit_plot(tracks, t=constellation_times, title=caption,
                   view="groundtrack", c="black", figsize=(14, 7), fps=10,
                   max_frames=300, tail=8, save="demo_constellation_groundtracks.mp4")
        # gallery-end
        outputs.update(video="demo_globe_plot_animation.mp4",
                       groundtrack_video="demo_constellation_groundtracks.mp4",
                       constellation_count=len(orbits), constellation_times=constellation_times)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, make_video=True, fast=False)
