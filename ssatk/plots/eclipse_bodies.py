"""Earth and Moon meshes with physical attitude and finite-Sun eclipse shading.

These meshes place the WGS-84 Earth and the spherical Moon at their GCRF
positions and shade every vertex with the limb-darkened visible solar flux.
The general-purpose globes in :mod:`globe_orbit_daynight_plotly` and
:mod:`moon_render` are display-oriented and are left unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ssatk.compute.eclipse_brightness import irradiance_fraction
from ssatk.compute.eclipse_reference_events import AU_KM, R_MOON_MEAN_KM, R_SUN_KM, RE_KM, RP_KM
from ssatk.compute.eclipse_runtime import _itrf_to_gcrf_matrix
from ssatk.plots._textures import earth_texture_path, moon_texture_path


def _normalize(vector, *, name="vector"):
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must be finite and non-zero")
    return value / norm


def _srgb_to_linear(rgb):
    value = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    value = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1.0 / 2.4) - 0.055)


@lru_cache(maxsize=8)
def _texture(body: str, n_lat: int, n_lon: int) -> np.ndarray:
    from PIL import Image

    path = earth_texture_path() if body == "earth" else moon_texture_path()
    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        return np.asarray(image.convert("RGB").resize((n_lon, n_lat), resampling))


def _body_rows(texture: np.ndarray) -> np.ndarray:
    """Ring rows of a north-up texture plus one averaged colour per pole."""
    return np.vstack([
        texture[1:-1].reshape(-1, 3),
        texture[0].mean(axis=0, keepdims=True),
        texture[-1].mean(axis=0, keepdims=True),
    ]).astype(float) / 255.0


def _earth_rows_rotation(time_jd=None, rotation_deg=0.0) -> np.ndarray:
    """Right-multiplying matrix taking ITRF row vectors to GCRF."""
    if time_jd is not None:
        return _itrf_to_gcrf_matrix(float(time_jd)).T
    angle = np.radians(float(rotation_deg) % 360.0)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


@lru_cache(maxsize=32)
def _unit_sphere_mesh(n_lat: int, n_lon: int):
    """Latitude rings plus single pole vertices, with outward-facing faces."""
    n_lat, n_lon = int(n_lat), int(n_lon)
    if n_lat < 5 or n_lon < 8:
        raise ValueError("mesh resolution must be at least 5 x 8")
    if n_lat % 2 == 0:
        n_lat += 1
    lat_rows = np.linspace(90.0, -90.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, lat_rows[1:-1])
    columns = np.arange(n_lon, dtype=np.int32)
    next_columns = (columns + 1) % n_lon
    rows = n_lat - 2
    faces = []
    for row in range(rows - 1):
        top, bottom = row * n_lon, (row + 1) * n_lon
        faces.extend((
            np.column_stack([top + columns, bottom + columns, bottom + next_columns]),
            np.column_stack([top + columns, bottom + next_columns, top + next_columns]),
        ))
    north, south = rows * n_lon, rows * n_lon + 1
    faces.extend((
        np.column_stack([np.full(n_lon, north), next_columns, columns]),
        np.column_stack([np.full(n_lon, south), (rows - 1) * n_lon + columns, (rows - 1) * n_lon + next_columns]),
    ))
    lat_v = np.concatenate([Lat.ravel(), [90.0, -90.0]])
    lon_v = np.concatenate([Lon.ravel(), [0.0, 0.0]])
    phi, lam = np.radians(lat_v), np.radians(lon_v)
    directions = np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    faces = np.vstack(faces).astype(np.int32)
    p0, p1, p2 = directions[faces[:, 0]], directions[faces[:, 1]], directions[faces[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), (p0 + p1 + p2) / 3.0) < 0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()
    return directions, faces, lat_rows, lat_v, lon_v


def _moon_unit_mesh(n_lat, n_lon):
    return _unit_sphere_mesh(int(n_lat), int(n_lon))


@lru_cache(maxsize=32)
def _wgs84_vertices_cached(n_lat: int, n_lon: int, altitude_km: float):
    directions, faces, lat_rows, lat_v, lon_v = _unit_sphere_mesh(n_lat, n_lon)
    phi, lam = np.radians(lat_v), np.radians(lon_v)
    e2 = 1.0 - (RP_KM / RE_KM) ** 2
    prime = RE_KM / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    h = float(altitude_km)
    vertices = np.column_stack([
        (prime + h) * np.cos(phi) * np.cos(lam),
        (prime + h) * np.cos(phi) * np.sin(lam),
        (prime * (1.0 - e2) + h) * np.sin(phi),
    ])
    # Geodetic normals are the unit directions at geodetic latitude.
    return vertices, directions, faces, lat_rows, lat_v, lon_v


def _wgs84_vertices(n_lat, n_lon, *, altitude_km=0.0):
    return _wgs84_vertices_cached(int(n_lat), int(n_lon), float(altitude_km))


def _shadow_visibility(points_km, occluder_center_km, occluder_radius_km, sun_position_km):
    center = np.asarray(occluder_center_km, dtype=float).reshape(3)
    visibility = irradiance_fraction(
        points_km - center, R_body_km=float(occluder_radius_km), R_sun_km=R_SUN_KM,
        sun_position_km=np.broadcast_to(np.asarray(sun_position_km, dtype=float) - center, points_km.shape),
    )
    return np.clip(np.asarray(visibility, dtype=float).reshape(-1), 0.0, 1.0)


@dataclass
class EarthSurfaceData:
    display_vertices: np.ndarray
    physical_vertices_gcrf_km: np.ndarray
    normals_gcrf: np.ndarray
    faces: np.ndarray
    base_rgb_srgb: np.ndarray
    shaded_rgb_srgb: np.ndarray
    eclipse_visibility: np.ndarray
    rotation_rows: np.ndarray


def _earth_surface_data(
    sun_hat, n_lat=180, n_lon=360, radius_scale=1.0, center=(0.0, 0.0, 0.0),
    shadow_body_center_km=None, shadow_body_radius_km=None, rotation_deg=0.0,
    time_jd=None, sun_position_km=None, physical_center_km=(0.0, 0.0, 0.0),
    night_floor=0.05, exposure=1.0,
):
    physical_center = np.asarray(physical_center_km, dtype=float).reshape(3)
    if sun_position_km is None:
        sun_hat = _normalize(sun_hat, name="sun_hat")
        sun_absolute = physical_center + sun_hat * AU_KM
    else:
        sun_absolute = np.asarray(sun_position_km, dtype=float).reshape(3)
        sun_hat = _normalize(sun_absolute - physical_center, name="Sun relative position")
    vertices, body_normals, faces, lat_rows, _, _ = _wgs84_vertices(n_lat, n_lon)
    rotation = _earth_rows_rotation(time_jd, rotation_deg)
    local = vertices @ rotation
    normals = body_normals @ rotation
    physical = physical_center + local
    display = np.asarray(center, dtype=float).reshape(3) + local * float(radius_scale)
    base = _body_rows(_texture("earth", len(lat_rows), int(n_lon)))
    visibility = np.ones(len(display))
    if shadow_body_center_km is not None and shadow_body_radius_km is not None:
        visibility = _shadow_visibility(physical, shadow_body_center_km, shadow_body_radius_km, sun_absolute)
    illumination = float(np.clip(night_floor, 0.0, 0.08)) + np.clip(normals @ sun_hat, 0.0, 1.0) * visibility
    shaded = _linear_to_srgb(_srgb_to_linear(base) * illumination[:, None] * float(exposure))
    return EarthSurfaceData(display, physical, normals, faces, base, shaded, visibility, rotation)


def _display_lunar_albedo(rgb):
    """Stretch the low-exposure lunar photomosaic to a readable grey albedo.

    Used directly, the packaged texture renders an almost-full Moon nearly
    black. The luminance stretch keeps maria and crater contrast.
    """
    luminance = np.asarray(rgb, dtype=float) @ np.array([0.2126, 0.7152, 0.0722])
    low, high = np.nanpercentile(luminance, [1.0, 99.5])
    stretched = np.clip((luminance - low) / (high - low), 0.0, 1.0) if high > low else np.clip(luminance, 0.0, 1.0)
    return (0.34 + 0.58 * stretched)[:, None] * np.array([1.0, 0.985, 0.955])


@dataclass
class MoonSurfaceData:
    display_vertices: np.ndarray
    physical_vertices_gcrf_km: np.ndarray
    normals_gcrf: np.ndarray
    faces: np.ndarray
    base_rgb_srgb: np.ndarray
    shaded_rgb_srgb: np.ndarray
    eclipse_visibility: np.ndarray
    body_axes: np.ndarray


def _moon_surface_data(
    center, radius, sun_hat=None, real_center_km=None, mode="lunar",
    eclipse_tint=True, n_lat=181, n_lon=360, real_sun_position_km=None,
    ambient_floor=0.008, view_hat=None, exposure=1.08,
    eclipse_occluder_radius_km=RE_KM, body_to_gcrf=None,
):
    directions, faces, lat_rows, _, _ = _unit_sphere_mesh(int(n_lat), int(n_lon))
    center = np.asarray(center, dtype=float).reshape(3)
    physical_center = np.zeros(3) if real_center_km is None else np.asarray(real_center_km, dtype=float).reshape(3)
    radius = float(radius)
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    orientation = np.eye(3) if body_to_gcrf is None else np.asarray(body_to_gcrf, dtype=float).reshape(3, 3)
    if (not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-8)):
        raise ValueError("body_to_gcrf must be a proper rotation matrix")
    base = _display_lunar_albedo(_body_rows(_texture("moon", len(lat_rows), int(n_lon))))
    normals = directions @ orientation.T
    display_vertices = center + radius * normals
    physical_vertices = physical_center + R_MOON_MEAN_KM * normals
    if real_sun_position_km is not None:
        sun_position = np.asarray(real_sun_position_km, dtype=float).reshape(3)
        light_hat = _normalize(sun_position - physical_center)
    elif sun_hat is not None:
        light_hat = _normalize(sun_hat)
        sun_position = physical_center + light_hat * AU_KM
    else:
        light_hat = sun_position = None
    if view_hat is None:
        view_hat = -physical_center if np.linalg.norm(physical_center) > 0.0 else light_hat
    view_hat = _normalize(np.array([1.0, 0.0, 0.0]) if view_hat is None else view_hat)
    direct = np.zeros(len(normals))
    if light_hat is not None:
        # Lommel-Seeliger plus a Lambert term, with a mild opposition surge.
        mu0 = np.clip(normals @ light_hat, 0.0, 1.0)
        mu = np.clip(normals @ view_hat, 0.0, 1.0)
        lommel_seeliger = np.divide(2.0 * mu0, mu0 + mu, out=np.zeros_like(mu0), where=(mu0 + mu) > 1e-12)
        direct = np.clip(0.78 * lommel_seeliger + 0.22 * mu0, 0.0, 1.18)
        phase = np.arccos(np.clip(np.dot(light_hat, view_hat), -1.0, 1.0))
        direct *= 1.0 + 0.14 * np.exp(-(phase / np.radians(5.0)) ** 2)
    visibility = np.ones(len(normals))
    if mode == "lunar" and real_center_km is not None and sun_position is not None:
        visibility = _shadow_visibility(physical_vertices, np.zeros(3), eclipse_occluder_radius_km, sun_position)
    floor = float(np.clip(ambient_floor, 0.0, 0.08))
    linear = _srgb_to_linear(base) * float(np.clip(exposure, 0.1, 4.0))
    linear *= floor + (1.0 - floor) * direct[:, None] * visibility[:, None]
    if eclipse_tint and mode == "lunar":
        # Refracted red light in the umbra.
        mix = np.clip((0.35 - visibility) / 0.35, 0.0, 1.0)[:, None]
        linear += _srgb_to_linear(base) * mix * np.array([0.12, 0.018, 0.004])
    shaded = _linear_to_srgb(linear)
    return MoonSurfaceData(display_vertices, physical_vertices, normals, faces, base, shaded, visibility, orientation)
