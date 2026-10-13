#!/usr/bin/env python3
"""A synthetic lunar two-body orbit, with dated Moon and Earth context."""
GALLERY_CATEGORY = "orbit_visualization"
TITLE = "Lunar orbit and textured Moon"
DESCRIPTION = (
    "One 3.57-hour synthetic lunar orbit at 1,000 km mean altitude; oblique and "
    "orbit-plane views, a dated surface close-up and an Earth–Moon overview. "
    "Close-ups use physical scale; the overview labels its enlarged body radii."
)

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv, get_body
from ssatk.constants import MOON_MU, MOON_RADIUS
from ssatk.plots.figpath import figpath
from ssatk.plots.moon_plot_3d import moon_plot_3d


def main(make_figures=True, fast=False):
    t0 = Time("2025-01-01T00:00:00", scale="utc")
    # Moon-centred GCRF-aligned two-body state; this is not an Earth orbit.
    orbit = Orbit.fromKeplerianElements(
        a=MOON_RADIUS + 1000e3, e=.01, i=np.radians(90.), pa=0.,
        raan=0., trueAnomaly=0., t=t0.gps, mu=MOON_MU)
    period = 2*np.pi*np.sqrt(orbit.a**3/MOON_MU)
    # At most one degree of mean anomaly per segment, even in gallery mode.
    t_arr = t0.gps + np.linspace(0., period, 361 if fast else 721)
    r_arr, v_arr = rv(orbit, Time(t_arr, format='gps'))
    if make_figures:
        common = dict(t=t_arr, r_frame="moon_centered", reference_time=t0,
                      physical_body_sizes=True, show_lagrange=False,
                      show_stars=True, show_sun_graphic=False,
                      scene_pad=.12, view_zoom=1.25)
        for name, title, elevation, azimuth in [
            ("demo_moon_plot", "Lunar orbit · 1,000 km mean altitude", 25, -55),
            ("demo_moon_plot_polar", "Lunar orbit · view normal to the orbit plane", 0, -90),
            ("demo_moon_surface", "Moon surface · dated texture and illumination", 12, -80),
        ]:
            moon_plot_3d(r=None if name.endswith('surface') else r_arr,
                         title=title, elev=elevation, azim=azimuth,
                         save_path=figpath(name), **common)
            print(f"Saved: {figpath(name)}")
        moon = get_body('moon')
        earth = -np.asarray(moon.position(t0.gps)).reshape(3)
        context_azimuth = np.degrees(np.arctan2(earth[1], earth[0])) - 90.
        moon_plot_3d(t=t_arr, r_frame='moon_centered', reference_time=t0,
                     show_earth=True, physical_body_sizes=False,
                     show_lagrange=False, show_stars=False,
                     title="Earth–Moon context · body radii enlarged as labelled",
                     elev=90, azim=context_azimuth, scene_pad=.08, view_zoom=1.25,
                     shade_ambient=.5, shade_diffuse=.5,
                     save_path=figpath('demo_moon_system'))
    return dict(orbit=orbit, r=np.asarray(r_arr), v=np.asarray(v_arr), t=t_arr,
                period_seconds=period, figures_created=bool(make_figures))


if __name__ == "__main__":
    main()
