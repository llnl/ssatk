"""Mapped physical Moon sphere with illustrative solar/eclipse shading.

No synthetic terrain is added. Use time (GPS seconds or Astropy Time) for
DE440 body orientation; undated geometry uses lunar body-fixed axes.
"""
from __future__ import annotations

from .rendering import is_low_spec
from ._body_surfaces import surface_map, body_rotation

from .rendering import _with_low_spec
import numpy as np
import plotly.graph_objects as go

from ssatk.constants import AU_KM, EARTH_RADIUS_KM, MOON_MEAN_RADIUS_KM, SUN_NOMINAL_RADIUS_KM

try:
    from .scene_primitives import stabilize_sphere_poles
except ImportError:
    from scene_primitives import stabilize_sphere_poles

try:
    from .eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
except ImportError:
    try:
        from eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
    except ImportError:
        illumination_fraction = None
        R_SUN_KM = SUN_NOMINAL_RADIUS_KM

R_MOON_KM = MOON_MEAN_RADIUS_KM

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
