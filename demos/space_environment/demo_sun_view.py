#!/usr/bin/env python3
"""
demos/space_environment/demo_sun_view.py
-----------------------
Demo for ssatk.plots.sun_view: builds a Plotly figure with a
Sun model, Earth day/night shading, and Moon day/night shading.
"""

GALLERY_CATEGORY = "space_environment"
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from astropy.time import Time

from ssatk.constants import LD
from ssatk.plots.figpath import figpath
from ssatk.plots.sun_view import (
    sun_position_eci,
    SunLayer,
    EarthShadingLayer,
    MoonShadingLayer,
    starfield_trace,
    VISUAL_DIST_KM_LEO,
)

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")

    sun_hat, dist_au = sun_position_eci(t0)
    sun_pos = sun_hat * VISUAL_DIST_KM_LEO

    outputs = {
        "sun_hat": sun_hat,
        "dist_au": dist_au,
    }

    fig = go.Figure()

    # Frame the camera around the near-field objects (Sun/Moon), not the
    # much larger starfield backdrop -- see below.
    # Real mean Earth-Moon distance (LD, ~384,400 km) -- unlike the Sun,
    # this is small enough to use at true scale rather than needing an
    # artistic "visual, not-to-scale" placement.
    moon_dist_km = LD / 1000.0
    frame_km = max(VISUAL_DIST_KM_LEO, moon_dist_km) * 1.3

    # Stars need to sit within (or just beyond) this frame to actually be
    # visible -- placing them at VISUAL_DIST_KM_LEO * 20 (as an earlier
    # version did) put them 4x beyond the camera's effective view once the
    # frame was tightened to the near-field scale, making them invisible
    # (confirmed by an actual screenshot showing an empty black background).
    sky_radius = frame_km * 1.5
    fig.add_trace(starfield_trace(sky_radius))

    fig.add_traces(SunLayer(sun_pos).build_traces())
    fig.add_traces(EarthShadingLayer(sun_pos).build_traces())

    # Place the Moon at a realistic quarter-phase angle relative to the
    # ACTUAL sun direction (perpendicular to it), rather than a fixed
    # placeholder position that only happens to look right if sun_hat
    # coincidentally points a certain way. Confirmed by testing: with a
    # real (non-arbitrary) sun direction, a fixed +X placeholder left the
    # Moon's camera-facing hemisphere ~92% near-black (invisible against
    # the black background); this sun-relative placement instead gives a
    # realistic half-lit quarter-Moon appearance regardless of the actual
    # date/sun direction.
    ref_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(sun_hat, ref_axis)) > 0.9:
        ref_axis = np.array([1.0, 0.0, 0.0])
    moon_dir = np.cross(sun_hat, ref_axis)
    moon_dir = moon_dir / np.linalg.norm(moon_dir)
    moon_center = moon_dir * moon_dist_km
    fig.add_traces(MoonShadingLayer(sun_pos, moon_center).build_traces())

    # aspectmode="data" is required here -- without it, Plotly scales each
    # of the x/y/z axes independently to fill the plot area, which turns
    # every sphere in the scene into a squashed ellipsoid (confirmed by an
    # actual rendered screenshot showing exactly this). "data" keeps the
    # axes proportional to their real data ranges instead.
    #
    # A dark background is also needed here, not just cosmetic -- Plotly's
    # default background is white, which would make white/pale star
    # markers essentially invisible.
    #
    # Axis range vs. camera zoom -- these need to be handled separately:
    #
    # An earlier version set the axis range tight around the near-field
    # objects (Sun/Moon), on the assumption that Plotly's 3D traces aren't
    # clipped to the axis range and only use it for camera framing. An
    # actual rendered screenshot proved that assumption WRONG: stars sitting
    # beyond that range were visibly cut off at a hard boundary. So the
    # axis range here is instead set generously large -- safely containing
    # the entire starfield -- so nothing gets clipped.
    #
    # Separately, camera.eye is set explicitly to start the view zoomed in
    # tight on the Sun/Earth scale, regardless of how large the full
    # (clipping-safe) axis range is. Plotly's default eye magnitude
    # (~1.25 per axis) frames the *entire* axis range; scaling that down
    # by (desired near-field view radius / actual axis range radius) gives
    # a tight initial zoom while the full starfield remains reachable by
    # zooming out manually (it's inside the range, just not what the
    # camera starts framed to).
    axis_radius_km = sky_radius * 1.1   # generous margin beyond the stars themselves
    # 2.2x (rather than a tighter 1.5x) gives both the Sun and Earth
    # comfortable room in the initial frame -- Earth sits at the scene's
    # coordinate origin, so a tighter zoom calibrated only off the Sun's
    # distance ends up unintentionally very close to Earth too (they'd
    # share the same "near zoom" distance), making Earth look oversized
    # relative to the Sun (confirmed by an actual screenshot showing
    # exactly this).
    near_view_radius_km = VISUAL_DIST_KM_LEO * 2.2
    eye_scale = 1.25 * (near_view_radius_km / axis_radius_km)

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False, range=[-axis_radius_km, axis_radius_km]),
            yaxis=dict(visible=False, range=[-axis_radius_km, axis_radius_km]),
            zaxis=dict(visible=False, range=[-axis_radius_km, axis_radius_km]),
            bgcolor="black",
            camera=dict(eye=dict(x=eye_scale, y=eye_scale, z=eye_scale)),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
    )

    outputs["n_traces"] = len(fig.data)

    if make_figures:
        out_html = Path(figpath("demo_sun_view"))
        if out_html.suffix == "":
            out_html = out_html.with_suffix(".html")
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html))
        print("Saved:", out_html)
        outputs["html"] = str(out_html)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
