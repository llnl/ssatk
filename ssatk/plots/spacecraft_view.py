"""Spacecraft-centered mesh, attitude, Sun-vector, and celestial-background view.

All scene geometry is GCRF in metres. Mesh coordinates are body-frame metres;
quaternions are SSATK body-to-GCRF [w, x, y, z]. No orbit is propagated here.
See ``docs/spacecraft_view.rst`` for camera, interpolation, and rendering limits.
"""

from __future__ import annotations

from .rendering import is_low_spec
from ._body_surfaces import BODY_COLORS

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
    shared triple. ``face_parts`` optionally supplies a name for each face.
    Convex polygons are triangulated as a fan for rendering.
    """

    vertices: np.ndarray
    faces: object
    colors: object = (0.64, 0.72, 0.79)
    face_parts: object = None

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
        if self.face_parts is None:
            self.face_parts = ("surface",) * len(faces)
        else:
            self.face_parts = tuple(self.face_parts)
            if len(self.face_parts) != len(faces) or not all(isinstance(p, str) for p in self.face_parts):
                raise ValueError("face_parts must contain one string per face")
        self.colors = _track(self.colors, len(faces), 3, "colors")
        if np.any((self.colors < 0) | (self.colors > 1)):
            raise ValueError("RGB colors must lie in [0, 1]")
        triangles, colors, boundaries, triangle_faces = [], [], [], []
        for face_index, (face, color) in enumerate(zip(self.faces, self.colors)):
            for j in range(1, len(face) - 1):
                triangle = [face[0], face[j], face[j + 1]]
                a, b, c = self.vertices[triangle]
                if np.linalg.norm(np.cross(b - a, c - a)) == 0:
                    raise ValueError("mesh contains a degenerate triangle")
                triangles.append(triangle)
                colors.append(color)
                # Edges opposite each triangle vertex. Hide fan diagonals.
                boundaries.append((True, j == len(face) - 2, j == 1))
                triangle_faces.append(face_index)
        self.triangles = np.asarray(triangles, dtype=int)
        self.triangle_colors = np.asarray(colors)
        self.triangle_boundaries = np.asarray(boundaries, dtype=bool)
        self.triangle_faces = np.asarray(triangle_faces, dtype=int)

    @staticmethod
    def presets(group=None):
        """Return the built-in visual mesh names; no files are required."""
        from .spacecraft_presets import preset_names
        return preset_names(group)

    @staticmethod
    def preset_info(name):
        """Return source links, fidelity, visual axes and approximation notes."""
        from .spacecraft_presets import preset_info
        return preset_info(name)

    @classmethod
    def from_preset(cls, name="starlink", *, scale=1.0):
        """Build an illustrative mesh in metres, optionally scaled uniformly.

        These rigid visual templates are not engineering, optical or force
        models. ``preset_info(name)`` documents sources, approximation limits
        and visual body axes for each mission. No assets are downloaded.
        """
        from ssatk.satellites import SpacecraftBody
        from .spacecraft_presets import build_preset

        name = cls.preset_info(name)["name"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a positive dimensionless factor")
        if name not in ("starlink", "boxwing"):
            model = build_preset(name)
            return cls(np.asarray(model.vertices) * scale, model.faces, model.colors, model.parts)

        size = (2.8, 1.4, 0.18) if name == "starlink" else (1.2, 1.0, 0.8)
        bus = cls.from_body(SpacecraftBody.box(mass=1.0, size=size))
        vertices, faces, colors = list(bus.vertices), list(bus.faces), list(bus.colors)
        if name == "starlink":
            colors = [(0.76, 0.80, 0.84)] * len(faces)
        # Each wing extends outwards from the bus edge in its local XY plane.
        # Starlink-like wings extend along +/-Y; box wings extend along +/-X.
        axis = 1 if name == "starlink" else 0
        length, width = (4.0, 2.8) if name == "starlink" else (2.0, 1.4)
        for sign in (-1, 1):
            for row in range(8):
                for col in range(4):
                    lo, hi = sorted(sign * (size[axis] / 2 + np.array([row, row + 1]) * length / 8))
                    left, right = -width / 2 + np.array([col, col + 1]) * width / 4
                    points = np.array([[lo, left, 0], [hi, left, 0],
                                       [hi, right, 0], [lo, right, 0]])
                    if axis == 1:
                        points = points[::-1, [1, 0, 2]]  # Preserve +Z winding.
                    faces.append(np.arange(len(vertices), len(vertices) + 4))
                    vertices.extend(points)
                    colors.append((0.12, 0.27, 0.48))
        return cls(np.asarray(vertices) * scale, faces, colors)

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


def _rasterize_mesh(uv, depth, triangles, colors, width, height, boundaries=None):
    """CPU depth-buffer rasterization at pixel centers; camera depth is in metres.

    Interpolate reciprocal depth in screen space to obtain perspective-correct
    visibility. Mean triangle distance cannot resolve even a long convex box.
    Optional boundary edges exclude artificial polygon triangulation diagonals.
    The view's camera restrictions keep all mesh vertices in front of the eye.
    """
    pixels = (np.asarray(uv) * [1, -1] + 1) * [width / 2, height / 2]
    image = np.zeros((height, width, 4), dtype=np.float32)
    zbuffer = np.full((height, width), np.inf)
    for i, triangle in enumerate(triangles):
        z = depth[triangle]
        if np.any(z <= 0):
            continue
        a, b, c = pixels[triangle]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue  # Edge-on or sub-pixel area.
        lo = np.maximum(np.floor(np.min([a, b, c], axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(np.max([a, b, c], axis=0)).astype(int), [width, height])
        if np.any(hi <= lo):
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]) + 0.5, np.arange(lo[1], hi[1]) + 0.5)
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        w = 1 - u - v
        reciprocal = u / z[0] + v / z[1] + w / z[2]
        distance = np.divide(1.0, reciprocal, out=np.full_like(u, np.inf), where=reciprocal > 0)
        section = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        visible = (u >= -1e-12) & (v >= -1e-12) & (w >= -1e-12) & (distance < zbuffer[section])
        if not np.any(visible):
            continue
        tile = image[section]
        tile[visible, :3] = colors[i]
        tile[visible, 3] = 1
        zbuffer[section][visible] = distance[visible]
        if boundaries is not None:
            lengths = np.linalg.norm([b - c, c - a, a - b], axis=1)
            edge = np.zeros_like(visible)
            for weight, length, draw in zip((u, v, w), lengths, boundaries[i]):
                if draw and length > 0:
                    edge |= weight * abs(denominator) / length < 0.55
            tile[visible & edge, :3] *= 0.65
    return image, zbuffer


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
    """A supplied trajectory and attitude history with a selectable camera.

    ``t`` is Astropy Time or absolute GPS seconds, strictly increasing. ``r`` is
    (N, 3); ``q`` is (N, 4) or one constant body-to-GCRF [w, x, y, z] quaternion.
    ``frame='moon_centered_gcrf'`` accepts Moon-relative positions in GCRF axes,
    not lunar rotating or body-fixed coordinates. ``units`` applies to r and
    optional Sun/Moon geocentric GCRF position tracks; mesh and camera use metres.
    Missing ephemerides come from LLNL SSAPy at the supplied epochs.

    ``environment`` selects earth, moon, both, or auto (nearest surface at each
    frame). Stars are catalog GCRF directions at the first epoch. ``camera_offset``
    is a GCRF metre vector from the spacecraft to the camera, constant or (N, 3).
    The default is a static overview; camera="limb" puts the nearest body's
    limb behind the spacecraft. See camera_modes.rst for all shots.
    ``mesh`` accepts a preset name or SpacecraftMesh. With neither mesh nor
    body supplied, a schematic Starlink-like mesh is used automatically.
    """

    def __init__(self, r, t, q, *, mesh=None, body=None, units="m", frame="gcrf",
                 environment="auto", sun_position=None, moon_position=None,
                 camera=None, camera_offset=None, camera_up=(0, 0, 1), fov_deg=50.0,
                 stars=True, textures=True, title="Spacecraft attitude", low_spec=None):
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
            mesh = (SpacecraftMesh.from_preset() if body is None
                    else SpacecraftMesh.from_body(body))
        elif isinstance(mesh, str):
            mesh = SpacecraftMesh.from_preset(mesh)
        if not isinstance(mesh, SpacecraftMesh):
            raise TypeError("mesh must be a preset name or SpacecraftMesh")
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
        from .camera import _camera_spec, _CameraPath
        from dataclasses import replace
        if camera_offset is not None:
            if camera is not None and _camera_spec(camera).mode != "follow":
                raise ValueError("camera_offset is a follow-camera option")
            camera = "follow"
        self.camera = "limb" if camera == "limb" else _camera_spec(camera)
        if self.camera != "limb":
            if self.camera.target != 0:
                raise ValueError("SpacecraftView has one camera target (index 0)")
            if isinstance(camera, (str, type(None))):
                self.camera = replace(self.camera, up=tuple(self.camera_up))
            points = [self.r-self.mesh_radius, self.r+self.mesh_radius]
            for i in range(len(self.r)):
                for _, center, radius in self._bodies(i):
                    points.extend([center[None, :]-radius, center[None, :]+radius])
            points = np.concatenate(points)
            self._camera_bounds = np.array([points.min(axis=0), points.max(axis=0)])
            # Fit a bounding sphere in the vertical field of view. Portrait
            # exports increase the distance using their narrower horizontal FOV.
            spec = self.camera
            if spec.mode in {"overview", "orbit"} and spec.distance is None and spec.offset is None:
                radius = np.linalg.norm(self._camera_bounds[1]-self._camera_bounds[0])/2
                spec = replace(spec, distance=1.15*radius/np.sin(np.deg2rad(self.fov_deg/2)))
            self._camera_phase = (self.gps[0], self.gps[-1])
            self._shot = _CameraPath(spec, self.r, self._camera_bounds, times=self.gps,
                                     default_distance=5*self.mesh_radius)
        self.low_spec = is_low_spec(low_spec)
        self.stars, self.textures, self.title = bool(stars), bool(textures) and not self.low_spec, str(title)
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

    def _camera(self, index, aspect=1.0):
        if self.camera_offset is None and self.camera != "limb":
            eye, target = self._shot.eyes[index].copy(), self._shot.targets[index]
            if (self.camera.mode in {"overview", "orbit"} and self.camera.distance is None
                    and self.camera.offset is None and aspect < 1):
                eye = target + (eye-target)/aspect
            offset = eye-self.r[index]
            basis = _camera_basis(target-eye, self._shot.up)
        elif self.camera_offset is None:
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
            eye = self.r[index] + offset
            basis = _camera_basis(-offset, self.camera_up)
        else:
            offset = self.camera_offset[index]
            eye = self.r[index] + offset
            basis = _camera_basis(-offset, self.camera_up)
        if (np.linalg.norm(eye) <= EARTH_RADIUS
                or np.linalg.norm(eye - self.moon[index]) <= MOON_RADIUS):
            raise ValueError("camera lies inside Earth or Moon; change camera_offset")
        return eye, offset, basis

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
            color = np.tile(BODY_COLORS[name], (mask.sum(), 1))
            texture = _texture(name) if self.textures else None
            if texture is not None:
                if name == "earth":
                    from ssatk.coordinates.frames import eci_to_ecf_matrix
                    rotation = eci_to_ecf_matrix(float(self.gps[index]))
                else:
                    from ssatk.coordinates.lunar import lunar_map_rotation
                    rotation = lunar_map_rotation(float(self.gps[index]))[0]
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
        eye, offset, basis = self._camera(i, width/height)
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
        wide = self.camera != "limb" and self.camera.mode in {"overview", "orbit"}
        if wide:
            from .occultation import _clip_segments, _visible
            from matplotlib.collections import LineCollection
            bodies = [(center, radius) for _, center, radius in self._bodies(i)]
            segments, _ = _clip_segments(np.stack((self.r[:-1], self.r[1:]), axis=1), bodies, eye=eye)
            path_uv, path_depth = _project((segments-eye).reshape(-1, 3), basis, tan_x, tan_y)
            path_uv = path_uv.reshape(-1, 2, 2)
            # Camera-plane clipping is separate from body occultation.
            path_uv = path_uv[np.all(path_depth.reshape(-1, 2) > 0, axis=1)]
            ax.add_collection(LineCollection(path_uv, colors="#edaa55", linewidths=1, alpha=.7, zorder=2))
            head_uv, head_depth = _project(self.r[i]-eye, basis, tan_x, tan_y)
            if head_depth > 0 and _visible(self.r[i], bodies, eye=eye):
                ax.plot(*head_uv, marker="+", color="#ffe2b0", markersize=7, zorder=4)
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
        pixels, mesh_depth = _rasterize_mesh(uv, depth, tri, colors, width, height,
                                    self.mesh.triangle_boundaries)
        # Compare distance along each mesh pixel's ray with foreground bodies.
        # Raster depth is camera Z; sphere intersections are distance along a unit ray.
        rows, cols = np.nonzero(pixels[..., 3])
        rays = np.column_stack((((cols + .5) * 2 / width - 1) * tan_x,
                                (1 - (rows + .5) * 2 / height) * tan_y,
                                np.ones(len(rows)))) @ basis
        lengths = np.linalg.norm(rays, axis=1)
        distances = mesh_depth[rows, cols] * lengths
        for _, center, radius in self._bodies(i):
            hidden = _ray_sphere(rays / lengths[:, None], center - eye, radius) < distances
            pixels[rows[hidden], cols[hidden], 3] = 0
        ax.imshow(pixels, extent=(-1, 1, -1, 1), aspect="auto",
                  interpolation="nearest", zorder=3)
        origin = _project(-offset, basis, tan_x, tan_y)[0]
        arrows = [(geom["body_axes_gcrf"][:, j], label, color)
                  for j, (label, color) in enumerate(zip(("+X body", "+Y body", "+Z body"),
                                                       ("#ff737a", "#62e6ad", "#78b8ff")))]
        arrows.append((geom["sun_direction_gcrf"], "To Sun", "#ffdb70"))
        for direction, label, color in ([] if wide else arrows):
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
        ax.text(0.035, 0.900, f"{names} / {self.camera if self.camera == 'limb' else self.camera.mode} camera",
                transform=ax.transAxes, color="#acbed1", fontsize=10, va="top", zorder=8)
        elapsed = self.gps[i] - self.gps[0]
        caption = (f"{self.time[i].utc.isot} UTC   |   t + {elapsed:,.2f} s\n"
                   f"{'+ marks spacecraft centre | ' if wide else ''}Body → GCRF   |   mesh in metres   |   Sun center "
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
        from dataclasses import replace
        from copy import copy
        chase = self.camera != "limb" and self.camera.mode == "chase"
        constructor_camera = replace(self.camera, mode="follow") if chase else self.camera
        result = type(self)(interpolate(self.r), gps, q, mesh=self.mesh,
                          environment=self.environment,
                          sun_position=interpolate(self.sun) if self._supplied_sun else None,
                          moon_position=interpolate(self.moon) if self._supplied_moon else None,
                          camera=constructor_camera,
                          camera_offset=interpolate(self.camera_offset) if self.camera_offset is not None else None,
                          camera_up=self.camera_up, fov_deg=self.fov_deg,
                          stars=self.stars, textures=self.textures, title=self.title, low_spec=self.low_spec)
        if self.camera != "limb":
            from .camera import _CameraPath
            result._camera_bounds = self._camera_bounds.copy()
            result._camera_phase = self._camera_phase
            if chase:
                result.camera = self.camera
                result._shot = copy(self._shot)
                result._shot.targets = result.r.copy()
                offset = interpolate(self._shot.eyes-self._shot.targets)
                degenerate = np.linalg.norm(offset, axis=1) <= 1e-12*self._shot.distance
                previous = np.clip(np.searchsorted(old, new, side="right")-1, 0, len(old)-1)
                offset[degenerate] = (self._shot.eyes-self._shot.targets)[previous[degenerate]]
                result._shot.eyes = result.r + _unit(offset)*self._shot.distance
            else:
                result._shot = _CameraPath(self._shot.spec, result.r, result._camera_bounds,
                    times=result.gps, default_distance=5*self.mesh_radius, phase_range=self._camera_phase)
        return result


    def save(self, path, *, index=0, fps=15, duration_s=8.0, **render_kwargs):
        """Export PNG/JPG or a GIF/MP4 spanning the entire supplied time interval.

        Still images use ``index``. Animations resample uniformly in physical time
        to ``round(fps * duration_s)`` frames, including both endpoints. Duration is
        playback seconds, not simulated seconds. GIF delays distribute 10 ms ticks
        without cumulative rounding drift; GIF fps must be <= 100.
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
        if suffix == ".gif" and fps > 100:
            raise ValueError("GIF supports at most 100 fps (10 ms minimum frame delay)")
        count = max(2, int(round(fps * duration_s)))
        sequence = self.resample(np.linspace(self.gps[0], self.gps[-1], count))
        if suffix == ".gif":
            frames = [Image.fromarray(sequence.render(j, **render_kwargs)).convert(
                "P", palette=Image.Palette.ADAPTIVE) for j in range(count)]
            ticks = np.rint(np.arange(count + 1) * 100 / fps).astype(int)
            delays = (np.diff(ticks) * 10).tolist()
            frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                           duration=delays, disposal=2, optimize=False)
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
