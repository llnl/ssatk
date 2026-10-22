"""Shared dated Earth surface geometry for globe exports.

The texture is north-up, equirectangular, -180..180 degrees longitude.
All rotations act on column vectors.
"""
from __future__ import annotations

from .rendering import gallery_max_textures, is_low_spec
from ._body_surfaces import BODY_COLORS

import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..coordinates.frames import eci_to_ecf_matrix
from ._textures import earth_texture_path, viewer_earth_day_path


def earth_surface(resolution=180):
    """Return ITRF unit-sphere quads and RGB texel colours (one per quad).

    ``resolution`` is the number of longitude cells, with half as many latitude
    cells. Texels sample cell centres; vertices include both sides of the seam.
    No texture downloads occur here.
    """
    if int(resolution) != resolution or not 12 <= resolution <= 1440:
        raise ValueError("earth_resolution must be an integer between 12 and 1440")
    if is_low_spec():
        resolution = min(resolution, 36)
    width, height = int(resolution), int(resolution) // 2
    if is_low_spec():
        colors = np.broadcast_to(BODY_COLORS["earth"], (height, width, 3))
    else:
        path = earth_texture_path() if gallery_max_textures() else viewer_earth_day_path()
        if not path.is_file():
            path = earth_texture_path()
        with Image.open(path) as image:
            colors = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BOX)) / 255.0
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, width + 1),
                          np.linspace(np.pi / 2, -np.pi / 2, height + 1))
    vertices = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                         np.sin(lat)), axis=-1)
    quads = np.stack((vertices[:-1, :-1], vertices[:-1, 1:],
                      vertices[1:, 1:], vertices[1:, :-1]), axis=2)
    return quads.reshape(-1, 4, 3), colors.reshape(-1, 3)


def earth_to_scene(gps, frame="gcrf"):
    """ITRF -> scene, including pole tilt as well as sidereal rotation."""
    if frame == "itrf":
        return np.eye(3)
    if frame != "gcrf":
        raise ValueError("Textured globe views support GCRF or ITRF coordinates")
    return eci_to_ecf_matrix(float(gps)).T


class TexturedEarth:
    def __init__(self, ax, radius, resolution=180, *, cull_backfaces=False):
        self.ax = ax
        self.cull_backfaces = cull_backfaces
        self.fixed_quads, colors = earth_surface(resolution)
        self.fixed_quads *= radius
        self.colors = colors
        self.artist = Poly3DCollection(self.fixed_quads, facecolors=colors,
                                       edgecolors="none", linewidths=0, antialiased=False, zorder=1)
        ax.add_collection3d(self.artist)
        from .occultation import _register_body
        _register_body(ax, self.artist, np.zeros(3), radius)

    def orient(self, rotation):
        vertices = self.fixed_quads @ np.asarray(rotation).T
        if self.cull_backfaces:
            # A fixed-camera animation need not sort/draw the hidden hemisphere.
            # Retain cells crossing the limb. Stills keep the complete geometry
            # so users can rotate the returned Matplotlib figure afterwards.
            el, az = np.radians([self.ax.elev, self.ax.azim])
            view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
            front = np.max(vertices @ view, axis=1) >= 0
            self.artist.set_verts(vertices[front])
            self.artist.set_facecolor(self.colors[front])
        else:
            self.artist.set_verts(vertices)
        return self.artist


def visible_points(points, radius, elevation, azimuth):
    """Orthographic visibility outside an opaque spherical Earth.

    Interior points are also hidden. Trails drop segments with an occulted
    endpoint; they are not interpolated across a hidden section.
    """
    points = np.asarray(points, dtype=float)
    el, az = np.radians([elevation, azimuth])
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    depth = points @ view
    perpendicular = points - depth[..., None] * view
    inside = np.linalg.norm(points, axis=-1) < radius
    behind = (depth < 0) & (np.linalg.norm(perpendicular, axis=-1) < radius)
    return ~(inside | behind)
