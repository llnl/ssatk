"""Real north-up equirectangular maps on physical spheres in scene coordinates.

Longitude increases eastward from -180 to 180 degrees. Matrices act on column
vectors. Lunar maps use the DE440 mean-Earth/polar-axis frame. No synthetic craters or radial displacement are added.
"""
from functools import lru_cache
import warnings

import numpy as np
from PIL import Image

from .rendering import gallery_max_textures, is_low_spec

BODY_COLORS = {"earth": (0.10, 0.31, 0.54), "moon": (1., 1., 1.)}


def texture_path(body):
    if is_low_spec():
        return None
    from ._textures import earth_texture_path, moon_texture_path, viewer_earth_day_path
    try:
        if gallery_max_textures():
            return earth_texture_path() if body == "earth" else moon_texture_path()
        if body == "earth" and viewer_earth_day_path().is_file():
            return viewer_earth_day_path()
        return earth_texture_path() if body == "earth" else moon_texture_path()
    except FileNotFoundError as exc:
        if gallery_max_textures():
            raise
        warnings.warn(f"{exc} Rendering a plain {body.title()} sphere.", UserWarning, stacklevel=2)
        return None


@lru_cache(maxsize=16)
def _read_map(path, width, height, modified=None, raw=False):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BOX))
        return rgb if raw else rgb / 255.


def surface_map(body, width, height, path=None):
    """RGB texels, or a constant colour without any texture I/O in low mode."""
    if is_low_spec():
        return np.broadcast_to(BODY_COLORS[body], (height, width, 3)).copy()
    path = texture_path(body) if path is None or gallery_max_textures() else path
    if path is None:
        return np.broadcast_to(BODY_COLORS[body], (height, width, 3)).copy()
    from pathlib import Path
    path = Path(path)
    modified = path.stat().st_mtime_ns
    return _read_map(str(path), width, height, modified).copy()


def body_rotation(body, time=None, frame="gcrf"):
    """Body-fixed -> scene orientation. Undated bodies use body-fixed axes.

    ``lunar`` and ``lunar fixed`` are Earth–Moon rotating axes, not lunar
    selenographic axes. The origin translation belongs to the caller.
    """
    if time is None:
        return np.eye(3)
    from astropy.time import Time
    from ..coordinates.frames import eci_to_ecf_matrix
    from ..coordinates.lunar import lunar_map_rotation, gcrf_to_lunar
    time = time if isinstance(time, Time) else Time(time, format="gps") if np.issubdtype(np.asarray(time).dtype, np.number) else Time(time)
    gps = float(np.atleast_1d(time.gps)[0])
    rotation = eci_to_ecf_matrix(gps).T if body == "earth" else lunar_map_rotation(gps)[0].T
    key = frame.lower().replace("_", " ")
    if key in {"gcrf", "eci", "moon centered", "moon centered gcrf"}:
        return rotation
    if key in {"itrf", "ecef"}:
        return eci_to_ecf_matrix(gps) @ rotation
    if key in {"lunar", "lunar fixed", "lunar axis"}:
        return gcrf_to_lunar(np.eye(3), np.full(3, gps)).T @ rotation
    if key == "ecliptic":
        # Fixed mean ecliptic/equinox J2000, as used by the solar-system plots.
        from ._solar_scene import ecliptic_vectors
        return ecliptic_vectors(rotation.T).T
    raise ValueError(f"Unsupported body-surface frame: {frame}")


def surface_quads(body, n=90, path=None):
    n = min(n, 18) if is_low_spec() else n
    width, height = 2*int(n), int(n)
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, width+1), np.linspace(np.pi/2, -np.pi/2, height+1))
    vertices = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
    quads = np.stack((vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1]), axis=2)
    return quads.reshape(-1, 4, 3), surface_map(body, width, height, path).reshape(-1, 3)


class BodySurface:
    """Reusable Matplotlib surface; texture/mode captured once for animation."""
    def __init__(self, ax, body, radius, center=(0, 0, 0), rotation=None, *, n=90, path=None, zorder=10):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.low_spec = is_low_spec()
        self.quads, self.colors = surface_quads(body, n, path)
        self.radius = radius
        self.artist = Poly3DCollection([], edgecolors="none", linewidths=0, antialiased=False, zorder=zorder)
        from .occultation import _register_body
        self.occluder = _register_body(ax, self.artist, center, radius)
        self.update(center, rotation)
        ax.add_collection3d(self.artist)

    def update(self, center=(0, 0, 0), rotation=None, sun_hat=None, ambient=.22, diffuse=.78):
        self.occluder.center = np.asarray(center, dtype=float)
        rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        quads = self.quads @ rotation.T
        colors = self.colors
        if sun_hat is None and self.low_spec:
            # Neutral display light for an untextured ball, not an ephemeris Sun.
            sun_hat = np.array([1., -1., 2.])/np.sqrt(6.)
            ambient, diffuse = .35, .65
        if sun_hat is not None:
            normals = quads.mean(axis=1)
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            colors = np.clip(colors*(ambient+diffuse*np.maximum(0, normals @ sun_hat))[:, None], 0, 1)
        self.artist.set_verts(np.asarray(center) + self.radius*quads)
        self.artist.set_facecolor(colors)
        return self.artist


def body_mesh_plotly(body, center, radius, rotation=None, *, n=90, path=None, sun_hat=None):
    """Float64 vertices and per-face texels (no palette quantization)."""
    import plotly.graph_objects as go
    quads, colors = surface_quads(body, n, path)
    quads = quads @ (np.eye(3) if rotation is None else np.asarray(rotation).T)
    if sun_hat is not None:
        normals = quads.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        colors *= (.22+.78*np.maximum(0, normals @ sun_hat))[:, None]
    vertices = np.asarray(center) + radius*quads.reshape(-1, 3)
    indices = np.arange(len(vertices)).reshape(-1, 4)
    triangles = indices[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inverse[triangles]
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                     facecolor=np.repeat(np.round(colors*255).astype(np.uint8), 2, axis=0),
                     lighting=dict(ambient=.35 if is_low_spec() and sun_hat is None else 1,
                                   diffuse=.65 if is_low_spec() and sun_hat is None else 0, specular=0),
                     name=body.title(), hoverinfo="skip", showlegend=False)


def body_disc(ax, body, radius, center=(0, 0, 0), rotation=None, view="xy", *, resolution=1024):
    """Orthographic mapped front hemisphere for coordinate-plane plots."""
    from matplotlib.patches import Circle
    a, b = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
    center = np.asarray(center)
    circle = Circle(center[[a, b]], radius, facecolor=BODY_COLORS[body], edgecolor="none", zorder=2)
    ax.add_patch(circle)
    from .occultation import _register_body
    _register_body(ax, circle, center, radius, view)
    if not is_low_spec():
        size = int(resolution)
        axis = (np.arange(size)+.5)*2/size-1
        x, y = np.meshgrid(axis, axis)
        normals = np.zeros((size, size, 3))
        normals[..., a], normals[..., b] = x, y
        normals[..., 3-a-b] = np.sqrt(np.maximum(0, 1-x*x-y*y))
        normals = normals @ (np.eye(3) if rotation is None else rotation)
        lon = np.arctan2(normals[..., 1], normals[..., 0])
        lat = np.arcsin(np.clip(normals[..., 2], -1, 1))
        # Bilinear texel lookup avoids magnifying nearest-neighbour blocks.
        from scipy.ndimage import map_coordinates
        full = gallery_max_textures()
        width, height = (8192, 4096) if full else (4096, 2048)
        if full:
            path = texture_path(body)
            rgb = _read_map(str(path), width, height, path.stat().st_mtime_ns, raw=True)
        else:
            rgb = surface_map(body, width, height)
        col = (lon+np.pi)/(2*np.pi)*width-.5
        row = np.clip((np.pi/2-lat)/np.pi*height-.5, 0, height-1)
        mapped = np.stack([map_coordinates(rgb[..., channel], [row, col], order=1,
                                           output=np.float64, mode="grid-wrap")
                           for channel in range(3)], axis=-1)
        if full:
            mapped /= 255.
        image = ax.imshow(mapped, origin="lower", interpolation="bilinear",
                          extent=(center[a]-radius, center[a]+radius,
                                  center[b]-radius, center[b]+radius), zorder=2)
        image.set_clip_path(circle)
        circle._ssatk_body_images = [image]
    return circle
