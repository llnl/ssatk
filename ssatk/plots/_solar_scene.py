"""Shared J2000 ecliptic geometry and an infinitely distant Plotly sky.

The sky is a projected background image, never a sphere in the AU scene.
HTML exports update it on camera rotation/resize; static exports retain the
initial projection. Star marker sizes are illustrative, not angular diameters.
"""

import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from astropy.coordinates import (
    BarycentricMeanEcliptic, CartesianRepresentation, ICRS,
    get_body_barycentric, solar_system_ephemeris,
)
from astropy.time import Time

from .starfield import _load_stars


def ecliptic_vectors(xyz):
    """Rotate ICRS Cartesian vectors into the mean J2000 ecliptic axes."""
    xyz = np.asarray(xyz, dtype=float)
    coord = ICRS(CartesianRepresentation(np.moveaxis(xyz, -1, 0) * u.au))
    out = coord.transform_to(BarycentricMeanEcliptic(equinox=Time("J2000")))
    return np.moveaxis(out.cartesian.xyz.to_value(u.au), 0, -1)


def body_position_au(name, time, ephemeris="builtin", origin="sun"):
    """Geometric simultaneous positions, in AU and mean J2000 ecliptic axes.

    Numeric input is Julian Date UTC. Ephemeris errors propagate; a scene must
    not silently mix a different ephemeris or reference frame for one body.
    """
    time = time if isinstance(time, Time) else Time(time, format="jd", scale="utc")
    with solar_system_ephemeris.set(ephemeris):
        delta = get_body_barycentric(name.lower(), time) - get_body_barycentric(origin, time)
    return ecliptic_vectors(np.moveaxis(delta.xyz.to_value(u.au), 0, -1))


def project_sky(directions, camera, width, height):
    """Pinhole projection of directions at infinity; 45-degree vertical FOV.

    Matches gl-plot3d's perspective camera. Translations/zoom cannot create
    stellar parallax. Orthographic scenes retain this angular sky backdrop.
    Returns pixel coordinates and a visible mask, including the full viewport.
    """
    eye = np.array([camera.get("eye", {}).get(k, 1.25) for k in "xyz"], dtype=float)
    center = np.array([camera.get("center", {}).get(k, 0.0) for k in "xyz"], dtype=float)
    up = np.array([camera.get("up", {}).get(k, float(k == "z")) for k in "xyz"], dtype=float)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-12:
        right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    dirs = np.asarray(directions)
    depth = dirs @ forward
    focal = height / (2.0 * np.tan(np.pi / 8))
    safe_depth = np.where(depth > 0, depth, 1.0)
    xy = np.column_stack((width / 2 + focal * (dirs @ right) / safe_depth,
                          height / 2 - focal * (dirs @ up) / safe_depth))
    visible = ((depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= width)
               & (xy[:, 1] >= 0) & (xy[:, 1] <= height))
    return xy, visible


def add_sky(fig, cfg, time):
    """Attach all-sky catalogue directions and an initial SVG projection."""
    stars = _load_stars(mag_limit=float(cfg.get("star_mag_limit", 6.5)),
                        when=time.to_datetime(), frame="gcrf", catalog_path=cfg.get("star_catalog"))
    if stars is None:
        # Absence of a catalogue must not fabricate an astronomical sky.
        fig.add_annotation(text="Star catalogue unavailable", x=0, y=0,
                           xref="paper", yref="paper", showarrow=False)
        return
    directions = ecliptic_vectors(stars["v"])
    sizes = np.clip(0.5 + 0.22 * (6.5 - stars["mag"]), 0.5, 2.1)
    colors = ["rgb(%d,%d,%d)" % tuple(row) for row in (255 * stars["rgb"]).astype(int)]
    sky = dict(directions=directions.tolist(), sizes=sizes.tolist(), colors=colors)
    fig.update_layout(meta=dict(solar_sky=sky), scene_bgcolor="rgba(0,0,0,0)")
    fig.add_layout_image(source=sky_source(sky, fig.layout.scene.camera.to_plotly_json(), 1000, 700),
                         xref="paper", yref="paper", x=0, y=1,
                         sizex=1, sizey=1, sizing="stretch", layer="below")


def sky_source(sky, camera, width, height):
    """SVG projection for static export at the actual image aspect ratio."""
    xy, visible = project_sky(sky["directions"], camera, width, height)
    circles = [f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{s:.3f}" fill="{c}"/>'
               for (x, y), s, c in zip(xy[visible], np.array(sky["sizes"])[visible],
                                       np.array(sky["colors"])[visible])]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           + "".join(circles) + "</svg>")
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


class SolarFigure(go.Figure):
    """A Plotly figure whose HTML exports keep the sky aligned with the camera."""

    def write_image(self, *args, **kwargs):
        if not self.layout.meta or "solar_sky" not in self.layout.meta:
            return super().write_image(*args, **kwargs)
        width = kwargs.get("width") or self.layout.width or 700
        height = kwargs.get("height") or self.layout.height or 450
        margin = self.layout.margin
        source = self.layout.images[0].source
        try:
            self.layout.images[0].source = sky_source(
                self.layout.meta["solar_sky"], self.layout.scene.camera.to_plotly_json(),
                width - (margin.l or 0) - (margin.r or 0),
                height - (margin.t or 0) - (margin.b or 0))
            return super().write_image(*args, **kwargs)
        finally:
            self.layout.images[0].source = source

    def _sky_scripts(self, post_script):
        scripts = [] if post_script is None else ([post_script] if isinstance(post_script, str) else list(post_script))
        if self.layout.meta and "solar_sky" in self.layout.meta:
            scripts.append(Path(__file__).with_name("solar_sky.js").read_text(encoding="utf-8"))
        return scripts or None

    def write_html(self, *args, **kwargs):
        kwargs["post_script"] = self._sky_scripts(kwargs.get("post_script"))
        return super().write_html(*args, **kwargs)

    def to_html(self, *args, **kwargs):
        kwargs["post_script"] = self._sky_scripts(kwargs.get("post_script"))
        return super().to_html(*args, **kwargs)


def scale_caption(cfg):
    """Make every departure from physical body/distance scale visible."""
    planet = float(cfg.get("planet_scale", 1.0))
    sun = float(cfg.get("sun_scale", 1.0))
    moon = float(cfg.get("moon_orbit_scale", 1.0))
    if min(planet, sun, moon) <= 0 or not np.isfinite([planet, sun, moon]).all():
        raise ValueError("Body and Moon orbit scale factors must be positive and finite")
    if (planet, sun, moon) == (1.0, 1.0, 1.0):
        return "Physical radii and distances · crosses locate bodies; they are not body diameters"
    return f"DISPLAY SCALES: planets/Moon ×{planet:g}, Sun ×{sun:g}, Moon distance ×{moon:g}"
