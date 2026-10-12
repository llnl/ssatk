"""Spacecraft-centered mesh, attitude, Sun-vector, and celestial-background view.

All scene geometry is GCRF in metres. Mesh coordinates are body-frame metres;
quaternions are SSATK body-to-GCRF [w, x, y, z]. No orbit is propagated here.
See ``docs/spacecraft_view.rst`` for camera, interpolation, and rendering limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
from astropy.time import Time

from ssatk.constants import EARTH_RADIUS, MOON_RADIUS
from ssatk.coordinates.attitude import rotate_vector

__all__ = ["SpacecraftMesh", "SpacecraftView", "spacecraft_view"]


def _unit(vector, name="vector"):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    if not np.all(np.isfinite(vector)) or np.any(length == 0):
        raise ValueError(f"{name} must be finite and nonzero")
    return vector / length


def _track(values, count, columns, name):
    values = np.asarray(values, dtype=float)
    if values.shape == (columns,):
        values = np.broadcast_to(values, (count, columns))
    if values.shape != (count, columns) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape ({columns},) or ({count}, {columns})")
    return values.copy()


def _body_positions(name, gps):
    from ssapy import get_body

    body = get_body(name)  # Keep the ephemeris kernels alive during evaluation.
    return np.asarray(body.position(gps), dtype=float).reshape(3, -1).T


@dataclass
class SpacecraftMesh:
    """Body-frame vertices in metres and zero-based triangular/convex faces.

    Coordinates retain their physical size and origin: no centering or automatic
    normalization is applied. ``colors`` is one RGB triple per face, or one
    shared triple. Convex polygons are triangulated as a fan for rendering.
    """

    vertices: np.ndarray
    faces: object
    colors: object = (0.64, 0.72, 0.79)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).copy()
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or len(self.vertices) < 3 or not np.all(np.isfinite(self.vertices))):
            raise ValueError("vertices must be a finite (V, 3) array in metres")
        faces = []
        for face in self.faces:
            raw = np.asarray(face)
            if (raw.ndim != 1 or len(raw) < 3 or raw.dtype.kind not in "iu"
                    or np.any(raw < 0) or np.any(raw >= len(self.vertices))):
                raise ValueError("faces must contain at least three valid integer vertex indices")
            faces.append(raw.astype(int))
        if not faces:
            raise ValueError("mesh must have faces")
        self.faces = tuple(faces)
        self.colors = _track(self.colors, len(faces), 3, "colors")
        if np.any((self.colors < 0) | (self.colors > 1)):
            raise ValueError("RGB colors must lie in [0, 1]")
        triangles, colors = [], []
        for face, color in zip(self.faces, self.colors):
            for j in range(1, len(face) - 1):
                triangle = [face[0], face[j], face[j + 1]]
                a, b, c = self.vertices[triangle]
                if np.linalg.norm(np.cross(b - a, c - a)) == 0:
                    raise ValueError("mesh contains a degenerate triangle")
                triangles.append(triangle)
                colors.append(color)
        self.triangles = np.asarray(triangles, dtype=int)
        self.triangle_colors = np.asarray(colors)

    @classmethod
    def from_body(cls, body):
        """Read the actual ``SpacecraftBody.facets`` geometry without synthesis.

        Subtract the body's current center of mass so a propagated center-of-mass
        position can be used directly. The body and its mass center are a snapshot.
        Facets lacking ``vertices_body`` are rejected; an area and normal do
        not uniquely specify the shape of a physical surface.
        """
        vertices, faces, colors = [], [], []
        origin = np.asarray(getattr(body, "current_center_of_mass", (0, 0, 0)), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("body center of mass must be a finite three-vector")
        for facet in body.facets:
            if facet.vertices_body is None:
                raise ValueError(f"facet {facet.name!r} has no vertices_body")
            points = np.asarray(facet.vertices_body, dtype=float) - origin
            faces.append(np.arange(len(vertices), len(vertices) + len(points)))
            vertices.extend(points)
            panel = any(word in facet.name.lower() for word in ("solar", "array", "panel", "wing"))
            colors.append((0.16, 0.37, 0.66) if panel else (0.74, 0.65, 0.40))
        return cls(vertices, faces, colors)

    @classmethod
    def from_obj(cls, path, *, scale=1.0):
        """Read a Wavefront OBJ using SSATK's facet loader; ``scale`` is m/unit.

        Supports face indices with texture/normal suffixes and negative indices.
        OBJ material/texture files are not interpreted. Polygons must be convex.
        """
        from ssatk.satellites import load_obj_facets
        from types import SimpleNamespace

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive metres per OBJ unit")
        facets = load_obj_facets(path, scale=scale)
        return cls.from_body(SimpleNamespace(facets=facets))


def _camera_basis(forward, up):
    forward = _unit(forward, "camera forward")
    up = _unit(up, "camera up")
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-10:
        seed = np.eye(3)[np.argmin(np.abs(forward))]
        right = np.cross(forward, seed)
    right = _unit(right)
    return np.stack((right, np.cross(right, forward), forward))


def _project(points, basis, tan_x, tan_y):
    """Perspective image coordinates [-1, 1], plus forward depth in metres."""
    camera = np.asarray(points) @ basis.T
    depth = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = camera[..., :2] / depth[..., None] / [tan_x, tan_y]
    return uv, depth


def _ray_sphere(directions, center, radius):
    """Distance to the near sphere surface for unit rays from the origin.

    Infinity denotes a miss. The perpendicular-distance form avoids subtracting
    two squared planet distances when the body is far from the camera.
    """
    along = np.einsum("...i,i->...", directions, center)
    perpendicular = np.cross(directions, center)
    discriminant = radius**2 - np.sum(perpendicular**2, axis=-1)
    distance = along - np.sqrt(np.maximum(discriminant, 0))
    return np.where((discriminant >= 0) & (distance > 0), distance, np.inf)


@lru_cache(maxsize=2)
def _texture(name):
    from PIL import Image
    from .starfield import find_data_file

    filename = "earth_day_2048.jpg" if name == "earth" else "moon.png"
    path = find_data_file(filename)
    if path is None:
        warnings.warn(f"Missing {filename}; rendering an untextured {name} sphere", stacklevel=2)
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 512))
        return np.asarray(image, dtype=float) / 255


class SpacecraftView:
    """A supplied trajectory and attitude history with a spacecraft-following camera.

    ``t`` is Astropy Time or absolute GPS seconds, strictly increasing. ``r`` is
    (N, 3); ``q`` is (N, 4) or one constant body-to-GCRF [w, x, y, z] quaternion.
    ``frame='moon_centered_gcrf'`` accepts Moon-relative positions in GCRF axes,
    not lunar rotating or body-fixed coordinates. ``units`` applies to r and
    optional Sun/Moon geocentric GCRF position tracks; mesh and camera use metres.
    Missing ephemerides come from LLNL SSAPy at the supplied epochs.

    ``environment`` selects earth, moon, both, or auto (nearest surface at each
    frame). Stars are catalog GCRF directions at the first epoch. ``camera_offset``
    is a GCRF metre vector from the spacecraft to the camera, constant or (N, 3).
    The default camera puts the nearest body's limb behind the spacecraft.
    """

    def __init__(self, r, t, q, *, mesh=None, body=None, units="m", frame="gcrf",
                 environment="auto", sun_position=None, moon_position=None,
                 camera_offset=None, camera_up=(0, 0, 1), fov_deg=50.0,
                 stars=True, textures=True, title="Spacecraft attitude"):
        self.time = t if isinstance(t, Time) else Time(t, format="gps")
        self.gps = np.atleast_1d(self.time.gps).astype(float)
        if (self.gps.ndim != 1 or len(self.gps) == 0
                or not np.all(np.isfinite(self.gps)) or np.any(np.diff(self.gps) <= 0)):
            raise ValueError("t must contain finite, strictly increasing absolute epochs")
        self.time = Time(self.gps, format="gps")
        count = len(self.gps)
        if units not in {"m", "km"}:
            raise ValueError("units must be 'm' or 'km'")
        factor = 1.0 if units == "m" else 1000.0
        if frame not in {"gcrf", "moon_centered_gcrf"}:
            raise ValueError("frame must be 'gcrf' or 'moon_centered_gcrf'")
        if environment not in {"auto", "earth", "moon", "both"}:
            raise ValueError("environment must be 'auto', 'earth', 'moon', or 'both'")
        self.environment = environment
        self._supplied_sun = sun_position is not None
        self._supplied_moon = moon_position is not None
        self.moon = (_body_positions("moon", self.gps) if moon_position is None
                     else factor * _track(moon_position, count, 3, "moon_position"))
        self.sun = (_body_positions("sun", self.gps) if sun_position is None
                    else factor * _track(sun_position, count, 3, "sun_position"))
        self.r = factor * _track(r, count, 3, "r")
        if frame == "moon_centered_gcrf":
            self.r += self.moon
        self.q = _unit(_track(q, count, 4, "q"), "quaternions")
        _unit(self.sun - self.r, "spacecraft-to-Sun direction")
        if (np.any(np.linalg.norm(self.r, axis=1) <= EARTH_RADIUS)
                or np.any(np.linalg.norm(self.r - self.moon, axis=1) <= MOON_RADIUS)):
            raise ValueError("spacecraft positions must be outside Earth and Moon")
        if mesh is not None and body is not None:
            raise ValueError("supply mesh or body, not both")
        if mesh is None:
            if body is None:
                from ssatk.satellites import SpacecraftBody
                body = SpacecraftBody.box_wing(name="illustrative bus", mass=100.0)
            mesh = SpacecraftMesh.from_body(body)
        if not isinstance(mesh, SpacecraftMesh):
            raise TypeError("mesh must be a SpacecraftMesh")
        self.mesh = mesh
        self.mesh_radius = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
        self.camera_offset = (None if camera_offset is None else
                              _track(camera_offset, count, 3, "camera_offset"))
        if self.camera_offset is not None and np.any(
                np.linalg.norm(self.camera_offset, axis=1) <= 2 * self.mesh_radius):
            raise ValueError("camera_offset must be more than two mesh radii from the origin")
        self.camera_up = _unit(np.asarray(camera_up, dtype=float), "camera_up")
        if self.camera_up.shape != (3,):
            raise ValueError("camera_up must be a three-vector")
        self.fov_deg = float(fov_deg)
        if not np.isfinite(self.fov_deg) or not 5 <= self.fov_deg <= 120:
            raise ValueError("fov_deg must be between 5 and 120 degrees")
        self.stars, self.textures, self.title = bool(stars), bool(textures), str(title)
        self._star_data = None
        if self.stars:
            from .starfield import star_directions
            self._star_data = star_directions(when=self.time[0], frame="gcrf")
            if self._star_data is None:
                raise FileNotFoundError("Star catalog missing; install ssa-data-core or set stars=False")

    def _index(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.gps):
            raise IndexError("index must identify a supplied time sample")
        return int(index)

    def geometry(self, index=0):
        """Return testable physical scene vectors, without rendering or scaling."""
        i = self._index(index)
        axes = np.column_stack([rotate_vector(self.q[i], axis) for axis in np.eye(3)])
        sun_hat = _unit(self.sun[i] - self.r[i])
        blocked = False
        for center, radius in ((np.zeros(3), EARTH_RADIUS), (self.moon[i], MOON_RADIUS)):
            hit = _ray_sphere(sun_hat, center - self.r[i], radius)
            blocked |= bool(hit < np.linalg.norm(self.sun[i] - self.r[i]))
        return {
            "vertices_gcrf_m": self.mesh.vertices @ axes.T + self.r[i],
            "body_axes_gcrf": axes,
            "sun_direction_gcrf": sun_hat,
            "earth_direction_gcrf": _unit(-self.r[i]),
            "moon_direction_gcrf": _unit(self.moon[i] - self.r[i]),
            "sun_center_visible": not blocked,
        }

    def _bodies(self, index):
        bodies = [("earth", np.zeros(3), EARTH_RADIUS),
                  ("moon", self.moon[index], MOON_RADIUS)]
        nearest = min(bodies, key=lambda item: np.linalg.norm(item[1] - self.r[index]) - item[2])
        if self.environment == "auto":
            return [nearest]
        return [item for item in bodies if self.environment == "both" or item[0] == self.environment]

    def _camera(self, index):
        if self.camera_offset is None:
            _, center, radius = min(self._bodies(index),
                                    key=lambda item: np.linalg.norm(item[1] - self.r[index]) - item[2])
            toward = _unit(center - self.r[index])
            tangent = np.cross(toward, self.camera_up)
            if np.linalg.norm(tangent) < 1e-10:
                tangent = np.cross(toward, np.eye(3)[np.argmin(np.abs(toward))])
            tangent = _unit(tangent)
            limb = np.arcsin(radius / np.linalg.norm(center - self.r[index]))
            forward = np.cos(limb + 0.20) * toward + np.sin(limb + 0.20) * tangent
            offset = -forward * (5.0 * self.mesh_radius)
        else:
            offset = self.camera_offset[index]
        eye = self.r[index] + offset
        if (np.linalg.norm(eye) <= EARTH_RADIUS
                or np.linalg.norm(eye - self.moon[index]) <= MOON_RADIUS):
            raise ValueError("camera lies inside Earth or Moon; change camera_offset")
        return eye, offset, _camera_basis(-offset, self.camera_up)

    def _background(self, index, eye, basis, tan_x, tan_y, width, height):
        """Ray-sampled sphere image, at true apparent angular size and position."""
        h = min(height, 720)
        w = max(2, round(width * h / height))
        x, y = np.meshgrid(np.linspace(-tan_x, tan_x, w), np.linspace(tan_y, -tan_y, h))
        rays = _unit(np.stack((x, y, np.ones_like(x)), axis=-1) @ basis)
        rgba = np.zeros((h, w, 4))
        depth = np.full((h, w), np.inf)
        for name, center, radius in self._bodies(index):
            relative = center - eye
            hits = _ray_sphere(rays, relative, radius)
            mask = hits < depth
            if not mask.any():
                continue
            normal = (hits[mask, None] * rays[mask] - relative) / radius
            # Surface-to-Sun geometry; shading is illustrative Lambertian light.
            sunlight = _unit(self.sun[index] - center - radius * normal)
            light = np.maximum(np.sum(normal * sunlight, axis=1), 0)
            color = np.tile((0.16, 0.38, 0.65) if name == "earth" else (0.56, 0.55, 0.52), (mask.sum(), 1))
            texture = _texture(name) if self.textures else None
            if texture is not None:
                if name == "earth":
                    from ssatk.coordinates.frames import eci_to_ecf_matrix
                    rotation = eci_to_ecf_matrix(float(self.gps[index]))
                else:
                    from ssatk.coordinates.lunar import lunar_body_rotation
                    rotation = lunar_body_rotation(float(self.gps[index]))[0]
                fixed = normal @ rotation.T
                longitude = np.arctan2(fixed[:, 1], fixed[:, 0])
                latitude = np.arcsin(np.clip(fixed[:, 2], -1, 1))
                rows = np.clip(((0.5 - latitude / np.pi) * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
                cols = (((longitude / (2 * np.pi) + 0.5) % 1) * texture.shape[1]).astype(int)
                color = texture[rows, cols]
            color *= (0.055 + 0.945 * light[:, None] ** 0.65)
            if name == "earth":
                limb = (1 - np.clip(np.sum(normal * -rays[mask], axis=1), 0, 1)) ** 8
                color += limb[:, None] * np.array([0.06, 0.16, 0.24])
            rgba[mask, :3] = np.clip(color, 0, 1)
            rgba[mask, 3] = 1
            depth[mask] = hits[mask]
        return rgba

    def figure(self, index=0, *, width=1200, height=800, dpi=100):
        """Return a Matplotlib Figure for an exact input sample (no GUI needed)."""
        from matplotlib.figure import Figure
        from matplotlib.collections import PolyCollection

        i = self._index(index)
        if (not isinstance(width, int) or not isinstance(height, int)
                or width < 160 or height < 160 or not np.isfinite(dpi) or dpi <= 0):
            raise ValueError("width and height must be integers >= 160; dpi must be positive")
        figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#060c17")
        ax = figure.add_axes((0, 0, 1, 1), facecolor="#060c17")
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        ax.set_axis_off()
        tan_y = np.tan(np.deg2rad(self.fov_deg / 2))
        tan_x = tan_y * width / height
        eye, offset, basis = self._camera(i)
        if self._star_data is not None:
            directions, magnitudes, colors = self._star_data
            uv, depth = _project(directions, basis, tan_x, tan_y)
            visible = (depth > 0) & np.all(np.abs(uv) < 1, axis=1)
            ax.scatter(uv[visible, 0], uv[visible, 1],
                       s=np.clip(8 - np.asarray(magnitudes)[visible], 0.5, 14),
                       c=np.clip(np.asarray(colors)[visible], 0, 1), alpha=0.85,
                       linewidths=0, zorder=0)
        ax.imshow(self._background(i, eye, basis, tan_x, tan_y, width, height),
                  extent=(-1, 1, -1, 1), aspect="auto", interpolation="bilinear", zorder=1)
        geom = self.geometry(i)
        # Work relative to the spacecraft to retain metre-scale mesh precision.
        local = self.mesh.vertices @ geom["body_axes_gcrf"].T
        uv, depth = _project(local - offset, basis, tan_x, tan_y)
        tri = self.mesh.triangles
        normals = _unit(np.cross(local[tri[:, 1]] - local[tri[:, 0]],
                                 local[tri[:, 2]] - local[tri[:, 0]]))
        light = np.maximum(normals @ geom["sun_direction_gcrf"], 0)
        if not geom["sun_center_visible"]:
            light *= 0
        colors = np.clip(self.mesh.triangle_colors * (0.28 + 0.72 * light[:, None]), 0, 1)
        order = np.argsort(-np.mean(depth[tri], axis=1))
        order = order[np.all(depth[tri[order]] > 0, axis=1)]
        ax.add_collection(PolyCollection(uv[tri[order]], facecolors=colors[order],
                                        edgecolors="#728fa4", linewidths=0.35, zorder=3))
        origin = _project(-offset, basis, tan_x, tan_y)[0]
        arrows = [(geom["body_axes_gcrf"][:, j], label, color)
                  for j, (label, color) in enumerate(zip(("+X body", "+Y body", "+Z body"),
                                                       ("#ff737a", "#62e6ad", "#78b8ff")))]
        arrows.append((geom["sun_direction_gcrf"], "To Sun", "#ffdb70"))
        for direction, label, color in arrows:
            end = direction * self.mesh_radius * 1.75
            tip, z = _project(end - offset, basis, tan_x, tan_y)
            if z <= 0:
                continue
            # Vectors are annotation overlays: they stay legible through the mesh.
            ax.annotate("", xy=tip, xytext=origin, zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=13))
            ax.annotate(label, xy=tip, xytext=(5, 5), textcoords="offset points",
                        color=color, fontsize=9, zorder=6,
                        bbox=dict(facecolor="#060c17", alpha=0.65, edgecolor="none", pad=2))
        names = " + ".join(body[0].title() for body in self._bodies(i))
        ax.text(0.035, 0.952, self.title, transform=ax.transAxes, color="#f2f6fc",
                fontsize=18, weight="bold", va="top", zorder=8)
        ax.text(0.035, 0.900, f"{names} environment  /  spacecraft-following camera",
                transform=ax.transAxes, color="#acbed1", fontsize=10, va="top", zorder=8)
        elapsed = self.gps[i] - self.gps[0]
        caption = (f"{self.time[i].utc.isot} UTC   |   t + {elapsed:,.2f} s\n"
                   f"Body → GCRF   |   mesh in metres   |   Sun center "
                   f"{'visible' if geom['sun_center_visible'] else 'occulted'}")
        ax.text(0.035, 0.04, caption, transform=ax.transAxes, color="#d0dce9",
                fontsize=9, linespacing=1.8, zorder=8,
                bbox=dict(facecolor="#060c17", alpha=0.8, edgecolor="none", pad=7))
        return figure

    def render(self, index=0, **kwargs):
        """Return a uint8 RGB image of an exact supplied sample."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = self.figure(index, **kwargs)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def resample(self, t):
        """Linearly interpolate positions and SLERP attitude at absolute epochs.

        No extrapolation or dynamics integration. Default ephemerides are freshly
        evaluated; explicitly supplied ephemerides and camera offsets interpolate.
        """
        from scipy.spatial.transform import Rotation, Slerp

        gps = np.atleast_1d(t.gps if isinstance(t, Time) else np.asarray(t, dtype=float))
        if (len(self.gps) < 2 or not np.all(np.isfinite(gps))
                or np.any(gps < self.gps[0]) or np.any(gps > self.gps[-1])):
            raise ValueError("resampling requires at least two input samples and epochs within their range")
        # Subtract the epoch before interpolation to avoid large absolute times.
        old, new = self.gps - self.gps[0], gps - self.gps[0]
        def interpolate(values):
            return np.column_stack([np.interp(new, old, values[:, j]) for j in range(3)])
        rotations = Rotation.from_quat(self.q[:, [1, 2, 3, 0]])
        q = Slerp(old, rotations)(new).as_quat()[:, [3, 0, 1, 2]]
        return type(self)(interpolate(self.r), gps, q, mesh=self.mesh,
                          environment=self.environment,
                          sun_position=interpolate(self.sun) if self._supplied_sun else None,
                          moon_position=interpolate(self.moon) if self._supplied_moon else None,
                          camera_offset=interpolate(self.camera_offset) if self.camera_offset is not None else None,
                          camera_up=self.camera_up, fov_deg=self.fov_deg,
                          stars=self.stars, textures=self.textures, title=self.title)

    def save(self, path, *, index=0, fps=15, duration_s=8.0, **render_kwargs):
        """Export PNG/JPG or a GIF/MP4 spanning the entire supplied time interval.

        Still images use ``index``. Animations resample uniformly in physical time
        to ``round(fps * duration_s)`` frames, including both endpoints. Duration is
        playback seconds, not simulated seconds. GIF timing has 10 ms quantization.
        MP4 requires the optional imageio-ffmpeg package and even image dimensions.
        """
        from PIL import Image

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".mp4"}:
            raise ValueError("save path must end in .png, .jpg, .jpeg, .gif, or .mp4")
        if suffix in {".png", ".jpg", ".jpeg"}:
            image = Image.fromarray(self.render(index, **render_kwargs))
            image.save(path, **({"quality": 95} if suffix != ".png" else {}))
            return path
        if (not np.isfinite(fps) or not np.isfinite(duration_s)
                or fps <= 0 or duration_s <= 0 or len(self.gps) < 2):
            raise ValueError("animations need at least two samples and positive fps/duration_s")
        count = max(2, int(round(fps * duration_s)))
        sequence = self.resample(np.linspace(self.gps[0], self.gps[-1], count))
        if suffix == ".gif":
            frames = [Image.fromarray(sequence.render(j, **render_kwargs)).convert(
                "P", palette=Image.Palette.ADAPTIVE) for j in range(count)]
            frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                           duration=round(1000 / fps), disposal=2, optimize=False)
        else:
            import imageio.v2 as imageio
            if any(render_kwargs.get(key, default) % 2 for key, default in (("width", 1200), ("height", 800))):
                raise ValueError("MP4 width and height must be even for yuv420p")
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:
                raise ImportError("MP4 export requires imageio-ffmpeg; install ssatk[video]") from exc
            with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1,
                                    pixelformat="yuv420p") as writer:
                for j in range(count):
                    writer.append_data(sequence.render(j, **render_kwargs))
        return path


def spacecraft_view(r, t, q, *, save_path=None, index=0, **kwargs):
    """Build a :class:`SpacecraftView`; optionally export a default-sized image.

    For custom export size or playback timing, call ``view.save(...)`` on the
    returned object. Examples are in ``demos/six_dof/demo_spacecraft_view.py``.
    """
    view = SpacecraftView(r, t, q, **kwargs)
    if save_path is not None:
        view.save(save_path, index=index)
    return view
