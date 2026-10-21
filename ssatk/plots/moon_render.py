"""Mapped physical Moon sphere with illustrative solar/eclipse shading.

Use time (GPS seconds or Astropy Time) for DE440 map orientation; undated
geometry uses lunar body-fixed axes. The public mesh retains its spherical
geometry and display shading. Private eclipse surfaces additionally accept
finite-Sun states and explicit body attitude, keeping physical and display
geometry separate.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from ssatk.compute.eclipse_brightness import irradiance_fraction
from ssatk.constants import AU_KM, EARTH_RADIUS_KM, MOON_MEAN_RADIUS_KM, SUN_NOMINAL_RADIUS_KM
from ._body_surfaces import body_rotation, surface_map
from .rendering import is_low_spec, _with_low_spec
from .scene_primitives import stabilize_sphere_poles
from .eclipse_brightness_plot import illumination_fraction

R_MOON_KM = MOON_MEAN_RADIUS_KM
R_SUN_KM = SUN_NOMINAL_RADIUS_KM

_moon_texture_cache = None


def _display_lunar_albedo(rgb):
    """Map a dark lunar photomosaic into a readable display albedo.

    Some packaged Moon textures are stored as low-exposure imagery.  Using those
    RGB values directly makes even an almost-full Moon render nearly black in
    Plotly.  Percentile-stretch the luminance, then remap it to a lunar-grey
    diagnostic albedo while preserving maria/crater contrast.
    """
    rgb = np.asarray(rgb, dtype=float) / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    low, high = np.nanpercentile(luminance, [1.0, 99.5])
    if not np.isfinite(high - low) or high <= low:
        stretched = np.clip(luminance, 0.0, 1.0)
    else:
        stretched = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    albedo = 0.34 + 0.58 * stretched
    return np.repeat(albedo[..., None], 3, axis=-1) * np.array([1.0, 0.985, 0.955])


def _load_real_moon_texture(n_lat, n_lon, texture_path=None):
    """
    Real lunar photomosaic, same discovery pattern as the Earth texture in
    globe_orbit_daynight_plotly.py: ssapy.utils.find_file + PIL. This is
    what the checkpoint's Section 6 flagged as unconfirmed on the real
    machine — this is that check, applied to the Moon instead of Earth.
    """
    global _moon_texture_cache
    if texture_path is None and _moon_texture_cache is not None and _moon_texture_cache.shape[:2] == (n_lat, n_lon):
        return _moon_texture_cache
    try:
        from PIL import Image
        from ._textures import moon_texture_path

        path = Path(texture_path).expanduser() if texture_path else moon_texture_path()
        img = Image.open(path).convert("RGB")
        img = img.resize((n_lon, n_lat), Image.LANCZOS)
        _moon_texture_cache = np.array(img)
        print("[moon_render] Moon base: real ssapy texture found")
        return _moon_texture_cache
    except Exception as ex:
        print(f"[moon_render] Real Moon texture not found ({ex}) — "
              f"using procedural maria+crater model instead.")
        return None


def _smooth(field, sigma=1.4):
    """Light Gaussian smoothing (wrapping in longitude) so procedural
    albedo/relief never shows hard tile/facet edges once displaced and
    lit — this is what turns a blocky, grid-like sphere (visible facet
    boundaries wherever two adjacent random blotches meet head-on) into
    a continuous, photographic-looking surface. Falls back to an
    unsmoothed field if scipy isn't available."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma, mode="wrap")
    except Exception:
        return field


def _procedural_moon_albedo(Lat, Lon, seed=3):
    """Two-scale albedo: large dark maria + small bright/dark crater fields,
    matching how the real Moon's near side actually looks patchy rather
    than uniformly grey."""
    rng = np.random.default_rng(seed)

    # --- Maria: a handful of large, soft, irregular dark blobs, mostly
    # confined to one hemisphere (real maria are lopsided — concentrated
    # on the historically Earth-facing side) ---
    albedo = np.full_like(Lat, 0.78)
    n_maria = 8
    for _ in range(n_maria):
        clat = rng.uniform(-45, 45)
        clon = rng.uniform(-60, 60)          # biased to one hemisphere
        spread_lat = rng.uniform(10, 26)
        spread_lon = rng.uniform(14, 32)
        dlat = (Lat - clat)
        dlon = (Lon - clon + 180) % 360 - 180
        d2 = (dlat / spread_lat) ** 2 + (dlon / spread_lon) ** 2
        albedo -= 0.22 * np.exp(-d2 / 2)

    # --- Crater fields on top: large + medium, then many small ---
    for _ in range(60):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(4, 11)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        albedo -= 0.12 * bowl ** 3
    for _ in range(500):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(0.4, 2.2)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        albedo -= 0.10 * bowl ** 3

    albedo += rng.normal(0, 0.01, Lat.shape)  # fine regolith speckle
    albedo = _smooth(albedo, sigma=0.45)       # just enough to kill hard blotch edges
    return np.clip(albedo, 0.30, 1.0)


def _procedural_moon_relief(Lat, Lon, seed=3):
    """Radial bump field: bright rim + dark bowl per crater, at two size
    scales, plus fine roughness noise — this is what gets displaced into
    actual 3D geometry (not just painted) so the lighting step below has
    real bumps to shade."""
    rng = np.random.default_rng(seed)
    relief = np.zeros_like(Lat)
    for _ in range(60):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(4, 11)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        relief -= 0.65 * bowl ** 3
        rim = np.clip(1 - np.abs(d - rad * 0.92) / (rad * 0.1), 0, 1)
        relief += 0.45 * rim
    for _ in range(500):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(0.4, 2.2)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        relief -= 0.5 * bowl ** 3
        rim = np.clip(1 - np.abs(d - rad * 0.9) / (rad * 0.2), 0, 1)
        relief += 0.35 * rim
    fine_noise = np.zeros_like(Lat)
    for _ in range(25):
        clat, clon = rng.uniform(-90, 90), rng.uniform(-180, 180)
        spread = rng.uniform(2, 6)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        fine_noise += rng.uniform(-0.15, 0.15) * np.exp(-(d ** 2) / (2 * spread ** 2))
    relief += fine_noise
    # Smooth before displacement — this is what removes any visible
    # facet/grid pattern from the mesh: unsmoothed relief has sharp
    # per-blotch edges that, once turned into real geometry and lit,
    # show up as a faint quad grid across the sphere (exactly the
    # artifact in the reference screenshot). A touch of blur keeps the
    # crater shapes but rounds their edges into the surrounding terrain.
    return _smooth(relief, sigma=0.4)


def _vertex_normals(X, Y, Z, center=None):
    """Return outward normals of a displaced grid.

    Finite differences are translation-invariant, but the final outward-facing
    orientation test must use body-local coordinates.  A Moon plotted hundreds
    of thousands of kilometres from Earth otherwise orients normals using the
    Earth-centered position vector and can render an almost-full Moon as dark.
    """
    Xu = np.gradient(X, axis=1); Yu = np.gradient(Y, axis=1); Zu = np.gradient(Z, axis=1)
    Xv = np.gradient(X, axis=0); Yv = np.gradient(Y, axis=0); Zv = np.gradient(Z, axis=0)
    nx = Yu * Zv - Zu * Yv
    ny = Zu * Xv - Xu * Zv
    nz = Xu * Yv - Yu * Xv
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-12
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    # Orient outward (dot with the sphere's own radial direction should be
    # positive; flip any that came out pointing inward from the cross
    # product's arbitrary handedness)
    if center is None:
        Xr, Yr, Zr = X, Y, Z
    else:
        cx, cy, cz = np.asarray(center, dtype=float)
        Xr, Yr, Zr = X - cx, Y - cy, Z - cz
    R = np.sqrt(Xr ** 2 + Yr ** 2 + Zr ** 2) + 1e-12
    rad_dot = (nx * Xr + ny * Yr + nz * Zr) / R
    flip = rad_dot < 0
    nx, ny, nz = np.where(flip, -nx, nx), np.where(flip, -ny, ny), np.where(flip, -nz, nz)
    return nx, ny, nz


@lru_cache(maxsize=32)
def _moon_unit_mesh_cached(n_lat, n_lon):
    n_lat, n_lon = int(n_lat), int(n_lon)
    if n_lat < 5 or n_lon < 8:
        raise ValueError("Moon mesh resolution must be at least 5 x 8")
    if n_lat % 2 == 0:
        n_lat += 1
    lat_rows = np.linspace(90.0, -90.0, n_lat)
    ring_lat = lat_rows[1:-1]
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, ring_lat)
    phi, lam = np.radians(Lat), np.radians(Lon)
    directions = np.vstack([
        np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1).reshape(-1, 3),
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ])
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
        np.column_stack([np.full(n_lon, south), (rows - 1) * n_lon + columns,
                         (rows - 1) * n_lon + next_columns]),
    ))
    faces = np.vstack(faces).astype(np.int32)
    p0, p1, p2 = directions[faces[:, 0]], directions[faces[:, 1]], directions[faces[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), (p0 + p1 + p2) / 3.0) < 0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()
    lat_v = np.concatenate([Lat.ravel(), [90.0, -90.0]])
    lon_v = np.concatenate([Lon.ravel(), [0.0, 0.0]])
    return directions, faces, lat_rows, lat_v, lon_v


def _moon_unit_mesh(n_lat, n_lon):
    return _moon_unit_mesh_cached(int(n_lat), int(n_lon))


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
    used_relief: bool


def _srgb_to_linear(rgb):
    value = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    value = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1.0 / 2.4) - 0.055)


def _mesh_vertex_normals(vertices, faces):
    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int32)]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=float)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    normals[np.einsum("ij,ij->i", normals, vertices) < 0.0] *= -1.0
    return normals


def _moon_surface_data(
    center, radius, sun_hat=None, seed=3, real_center_km=None, mode="lunar",
    eclipse_tint=True, n_lat=181, n_lon=360, real_sun_position_km=None,
    relief_exaggeration=0.0, ambient_floor=0.008, texture_path=None,
    view_hat=None, exposure=1.08, eclipse_occluder_radius_km=EARTH_RADIUS_KM,
    body_to_gcrf=None, time=None, rotation=None,
):
    if is_low_spec():
        n_lat, n_lon = min(n_lat, 19), min(n_lon, 36)
        relief_exaggeration = 0.0
    directions, faces, lat_rows, _, _ = _moon_unit_mesh(n_lat, n_lon)
    n_lat_effective, n_lon = len(lat_rows), int(n_lon)
    center = np.asarray(center, dtype=float).reshape(3)
    physical_center = np.zeros(3) if real_center_km is None else np.asarray(real_center_km, dtype=float).reshape(3)
    radius = float(radius)
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    orientation_value = body_to_gcrf if body_to_gcrf is not None else rotation
    orientation = body_rotation("moon", time) if orientation_value is None else np.asarray(orientation_value, dtype=float).reshape(3, 3)
    if (not np.all(np.isfinite(orientation))
            or not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-8)):
        raise ValueError("body_to_gcrf must be a proper rotation matrix")
    texture = (surface_map("moon", n_lon, n_lat_effective)*255.0 if is_low_spec()
               else _load_real_moon_texture(n_lat_effective, n_lon, texture_path))
    if texture is None:
        Lon, Lat = np.meshgrid(np.linspace(-180.0, 180.0, n_lon, endpoint=False), lat_rows)
        albedo = _procedural_moon_albedo(Lat, Lon, seed=seed)
        base = np.vstack([
            np.repeat(albedo[1:-1, :, None], 3, axis=2).reshape(-1, 3),
            np.repeat(albedo[0].mean()[None, None], 3, axis=2).reshape(1, 3),
            np.repeat(albedo[-1].mean()[None, None], 3, axis=2).reshape(1, 3),
        ]) * np.array([1.0, 0.98, 0.94])
    else:
        display_texture = _display_lunar_albedo(np.asarray(texture))
        base = display_texture[1:-1].reshape(-1, 3)
        base = np.vstack([
            base,
            display_texture[0].mean(axis=0, keepdims=True),
            display_texture[-1].mean(axis=0, keepdims=True),
        ])
    relief = float(relief_exaggeration)
    display_dirs = directions.copy()
    used_relief = bool(relief)
    if used_relief:
        Lon, Lat = np.meshgrid(np.linspace(-180.0, 180.0, n_lon, endpoint=False), lat_rows)
        field = _procedural_moon_relief(Lat, Lon, seed=seed)
        relief_rows = np.concatenate([field[1:-1].reshape(-1), [field[0].mean(), field[-1].mean()]])
        display_dirs *= (1.0 + relief * relief_rows[:, None])
    local_normals = _mesh_vertex_normals(display_dirs, faces) if used_relief else directions
    world_directions = directions @ orientation.T
    world_display_directions = display_dirs @ orientation.T
    normals = local_normals @ orientation.T
    display_vertices = center + radius * world_display_directions
    physical_vertices = physical_center + R_MOON_KM * world_directions
    if real_sun_position_km is not None:
        sun_position = np.asarray(real_sun_position_km, dtype=float).reshape(3)
        light_hat = (sun_position - physical_center) / np.linalg.norm(sun_position - physical_center)
    elif sun_hat is not None:
        light_hat = np.asarray(sun_hat, dtype=float).reshape(3)
        light_hat /= np.linalg.norm(light_hat)
        sun_position = physical_center + light_hat * AU_KM
    else:
        light_hat = None
        sun_position = None
    if view_hat is None:
        view_hat = -physical_center if np.linalg.norm(physical_center) > 0.0 else light_hat
    if view_hat is None:
        view_hat = np.array([1.0, 0.0, 0.0])
    view_hat = np.asarray(view_hat, dtype=float).reshape(3)
    view_hat /= np.linalg.norm(view_hat)
    direct = np.zeros(len(normals), dtype=float)
    if light_hat is not None:
        mu0 = np.clip(normals @ light_hat, 0.0, 1.0)
        mu = np.clip(normals @ view_hat, 0.0, 1.0)
        lommel_seeliger = np.divide(2.0 * mu0, mu0 + mu, out=np.zeros_like(mu0), where=(mu0 + mu) > 1e-12)
        direct = np.clip(0.78 * lommel_seeliger + 0.22 * mu0, 0.0, 1.18)
        phase = np.arccos(np.clip(np.dot(light_hat, view_hat), -1.0, 1.0))
        direct *= 1.0 + 0.14 * np.exp(-(phase / np.radians(5.0)) ** 2)
    visibility = np.ones(len(normals), dtype=float)
    if mode == "lunar" and real_center_km is not None and sun_position is not None:
        visibility = np.asarray(irradiance_fraction(
            physical_vertices, R_body_km=float(eclipse_occluder_radius_km),
            R_sun_km=R_SUN_KM, sun_position_km=np.broadcast_to(sun_position, physical_vertices.shape),
        ), dtype=float).reshape(-1)
        visibility = np.clip(visibility, 0.0, 1.0)
    floor = float(np.clip(ambient_floor, 0.0, 0.08))
    linear = _srgb_to_linear(base) * float(np.clip(exposure, 0.1, 4.0))
    linear *= floor + (1.0 - floor) * direct[:, None] * visibility[:, None]
    if eclipse_tint and mode == "lunar":
        mix = np.clip((0.35 - visibility) / 0.35, 0.0, 1.0)[:, None]
        linear += _srgb_to_linear(base) * mix * np.array([0.12, 0.018, 0.004])
    shaded = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return MoonSurfaceData(
        display_vertices=display_vertices, physical_vertices_gcrf_km=physical_vertices,
        normals_gcrf=normals, faces=faces, base_rgb_srgb=base,
        shaded_rgb_srgb=shaded, eclipse_visibility=visibility,
        body_axes=orientation, used_relief=used_relief,
    )


@_with_low_spec
def moon_mesh_plotly(center, radius, sun_hat=None, seed=3,
                     real_center_km=None, mode="lunar",
                     eclipse_tint=True, n_lat=180, n_lon=360, time=None, rotation=None):
    """
    Real-texture-first, properly-lit Moon mesh.

    center, radius : display-space sphere placement (km, display units —
                      same size_boost convention as the rest of the toolkit)
    sun_hat         : unit vector toward the Sun, for real diffuse shading
                      and (mode="lunar") the Earth-shadow eclipse check
    real_center_km  : TRUE physical Moon-center position relative to Earth
                      (only needed for mode="lunar" eclipse shading)
    mode            : "lunar" (Moon may be in Earth's shadow) or "solar"
                      (Moon is just normally sunlit; it's Earth being
                      shadowed elsewhere in the scene)
    eclipse_tint    : apply the warm/red multiplicative bias during deep
                      shadow (mode="lunar" only) — set False if the caller
                      wants to apply its own tint instead
    """
    if is_low_spec():
        n_lat, n_lon = min(n_lat, 19), min(n_lon, 36)
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, lat)
    latr, lonr = np.radians(Lat), np.radians(Lon)
    normals = np.stack((np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)), axis=-1)
    rotation = body_rotation("moon", time) if rotation is None else np.asarray(rotation)
    normals = normals @ rotation.T
    nx0, ny0, nz0 = np.moveaxis(normals, -1, 0)
    X, Y, Z = np.moveaxis(np.asarray(center) + radius*normals, -1, 0)
    nxr, nyr, nzr = nx0, ny0, nz0
    base_rgb = surface_map("moon", n_lon, n_lat)
    if not is_low_spec():
        base_rgb = _display_lunar_albedo(base_rgb*255.)

    # Real per-vertex diffuse lighting from the spherical normals — this is
    # what makes craters look sculpted instead of painted. Ambient floor
    # so the unlit portion of a gibbous/crescent Moon isn't pure black.
    if sun_hat is not None:
        diffuse = np.clip(nxr * sun_hat[0] + nyr * sun_hat[1] + nzr * sun_hat[2], 0, 1)
        shading = np.clip(0.14 + 0.86 * diffuse ** 0.80, 0.14, 1.0)
        # Opposition effect: real regolith backscatters extra strongly
        # right where the viewer, Sun, and surface point are nearly
        # aligned (why a full Moon looks noticeably brighter overall
        # than the diffuse-only model predicts, not just "more lit
        # area"). Approximated with the same fixed view-ish reference
        # used for the Sun's limb darkening elsewhere in the toolkit —
        # imperfect (doesn't track the actual camera), but keeps the lit
        # face from looking flat and grey the way a pure Lambertian
        # sphere does.
        opp = np.clip(diffuse - 0.92, 0, 1) / 0.08
        shading = np.clip(shading + 0.18 * opp, 0.14, 1.15)
    else:
        shading = np.ones_like(Lat)

    rgb = np.clip(base_rgb * shading[..., None], 0, 1)

    if mode == "lunar" and real_center_km is not None and sun_hat is not None \
            and illumination_fraction is not None:
        real_surface = np.stack([nx0, ny0, nz0], axis=-1) * R_MOON_KM
        real_positions = real_center_km[None, None, :] + real_surface
        flat_pos = real_positions.reshape(-1, 3)
        flat_sun = np.tile(sun_hat, (flat_pos.shape[0], 1))
        illum = illumination_fraction(flat_pos, flat_sun, R_body_km=EARTH_RADIUS_KM,
                                      R_sun_km=R_SUN_KM, D_km=AU_KM).reshape(Lat.shape)
        floor = 0.12
        eclipse_brightness = np.clip(floor + (1 - floor) * illum, floor, 1.0)
        rgb = rgb * eclipse_brightness[..., None]
        if eclipse_tint:
            red_mix = np.clip((0.35 - illum) / 0.35, 0, 1) ** 1.5
            warm = np.array([1.15, 0.55, 0.42])
            tint = (1 - red_mix[..., None]) + warm[None, None, :] * red_mix[..., None]
            rgb = np.clip(rgb * tint, 0, 1)

    rgb = stabilize_sphere_poles(rgb)

    vertexcolor = [f"rgb({int(rgb[r, c, 0]*255)},{int(rgb[r, c, 1]*255)},{int(rgb[r, c, 2]*255)})"
                  for r in range(n_lat) for c in range(n_lon)]

    ii, jj, kk = [], [], []
    for r in range(n_lat - 1):
        for c in range(n_lon):
            cn = (c + 1) % n_lon
            v0 = r * n_lon + c; v1 = r * n_lon + cn; v2 = (r + 1) * n_lon + c; v3 = (r + 1) * n_lon + cn
            ii += [v0, v1]; jj += [v1, v3]; kk += [v2, v2]

    return go.Mesh3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), i=ii, j=jj, k=kk,
        vertexcolor=vertexcolor, flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name="Moon", hoverinfo="skip", showlegend=False,
    )


@_with_low_spec
def _eclipse_moon_mesh_plotly(
    center, radius, sun_hat=None, seed=3, real_center_km=None, mode="lunar",
    eclipse_tint=True, n_lat=181, n_lon=360, real_sun_position_km=None,
    relief_exaggeration=0.0, ambient_floor=0.008, texture_path=None,
    view_hat=None, exposure=1.08, eclipse_occluder_radius_km=EARTH_RADIUS_KM,
    body_to_gcrf=None, time=None, rotation=None,
):
    """Return a high-resolution Moon mesh with physical attitude and eclipse shading.

    ``body_to_gcrf`` rotates the body-fixed texture and geometry.  Optional
    relief affects display geometry only; eclipse calculations retain the
    spherical physical Moon at ``R_MOON_KM``.
    """
    data = _moon_surface_data(
        center, radius, sun_hat=sun_hat, seed=seed, real_center_km=real_center_km,
        mode=mode, eclipse_tint=eclipse_tint, n_lat=n_lat, n_lon=n_lon,
        real_sun_position_km=real_sun_position_km, relief_exaggeration=relief_exaggeration,
        ambient_floor=ambient_floor, texture_path=texture_path, view_hat=view_hat,
        exposure=exposure, eclipse_occluder_radius_km=eclipse_occluder_radius_km,
        body_to_gcrf=body_to_gcrf, time=time, rotation=rotation,
    )
    return go.Mesh3d(
        x=data.display_vertices[:, 0], y=data.display_vertices[:, 1], z=data.display_vertices[:, 2],
        i=data.faces[:, 0], j=data.faces[:, 1], k=data.faces[:, 2],
        vertexcolor=[
            f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            for r, g, b in data.shaded_rgb_srgb
        ], flatshading=False, lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        name="Moon", hoverinfo="skip", showlegend=False,
    )
