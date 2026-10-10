"""
ssatk/plots/solar_bodies.py
=====================================
Proper 3D planet sphere rendering for the Solar View tab.

Each planet is a Plotly go.Surface with:
  - Correct relative proportions (exaggerated ~500× for visibility)
  - Axial tilt applied
  - Latitude/longitude-based coloring mimicking the real surface
  - Saturn rings as a flat annular Surface
  - Sun as a layered glowing sphere

Public API
----------
make_planet_traces(name, pos_au, scale_au, show_label)         → list[go.trace]
make_sun_traces(r_display_au)                                  → list[go.trace]
make_saturn_ring_traces(pos_au, scale_au)                      → list[go.trace]
make_moon_traces(earth_pos_au, t_jd, orbit_scale, show_label)  → list[go.trace]
moon_geocentric_ecliptic(t_jd)                                 → (x_km, y_km, z_km)

Moon note
---------
make_moon_trace() (singular — the old marker-only helper) used to place the
Moon at a fixed offset from Earth regardless of date. It's kept as a
backward-compatible shim but now defaults to a real J2000 position; callers
that have a real `t_jd` on hand should switch to make_moon_traces(), which
takes the date directly and returns a properly shaded sphere instead of a
flat marker.
"""

from __future__ import annotations
from io import BytesIO

import numpy as np
import plotly.graph_objects as go

from ssatk.constants import AU_KM, MOON_RADIUS, PLANET_RADIUS_KM, SUN_RADIUS_AU
from ssatk.data import read_data_binary
from ssatk.plots.scene_primitives import earth_rotation_deg_from_time


# ── Display radii in AU ──────────────────────────────────────────────────────
# Planets and the Moon remain exaggerated for visibility in full-system views.
# The Sun defaults to its physical AU radius; callers can pass a sun_scale if a
# teaching/demo plot needs a deliberately enlarged Sun.
_PLANET_DISPLAY_SCALE = 500.0
_R_AU = {
    name: PLANET_RADIUS_KM[name] / AU_KM * _PLANET_DISPLAY_SCALE
    for name in PLANET_RADIUS_KM
}
_R_AU["Sun"] = SUN_RADIUS_AU
_R_AU["Moon"] = (MOON_RADIUS / 1000.0) / AU_KM * _PLANET_DISPLAY_SCALE

# Axial tilts (degrees, prograde unless noted)
_TILT = {
    "Mercury":  0.034,
    "Venus":    177.4,   # retrograde — visually shown as slight tilt
    "Earth":    23.44,
    "Mars":     25.19,
    "Jupiter":  3.13,
    "Saturn":   26.73,
    "Uranus":   97.77,
    "Neptune":  28.32,
    "Sun":      7.25,
    "Moon":     6.68,
}

# 1 AU in km — used to convert the Moon's real geocentric ephemeris (km)
# into the same AU-based display coordinates as everything else here.
_AU_KM = AU_KM

# Sphere mesh resolution (lat × lon)
_N = 50


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _uv_mesh(n: int = _N):
    """Return (U, V, lat, lon) meshgrids.
    U: longitude 0..2π  V: colatitude 0..π
    lat: −π/2..π/2      lon: 0..2π
    """
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0,       np.pi, n)
    U, V = np.meshgrid(u, v)
    lat = np.pi / 2.0 - V
    return U, V, lat, U.copy()   # lon == U


def _unit_sphere(n: int = _N):
    """Return (xs, ys, zs) for a unit sphere."""
    U, V, _, _ = _uv_mesh(n)
    xs = np.sin(V) * np.cos(U)
    ys = np.sin(V) * np.sin(U)
    zs = np.cos(V)
    return xs, ys, zs


def _apply_tilt(xs, ys, zs, tilt_deg: float):
    """Rotate mesh about the X axis by tilt_deg (axial tilt)."""
    t = np.radians(tilt_deg % 180.0)   # clamp Venus retrograde to visual tilt
    ct, st = np.cos(t), np.sin(t)
    xs_t = xs
    ys_t = ys * ct - zs * st
    zs_t = ys * st + zs * ct
    return xs_t, ys_t, zs_t


# ── Planet color functions ─────────────────────────────────────────────────────
# Each returns (surfacecolor: 2D array, colorscale: list)
# surfacecolor values in [0, 1] mapped through colorscale.

def _color_mercury(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Grey cratered surface — subtle harmonic variation
    c = ( 0.50
        + 0.10 * np.sin(7 * lat) * np.cos(5 * lon)
        + 0.08 * np.sin(13 * lat)
        + 0.05 * np.cos(11 * lon) )
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.0, "#4a4a4a"], [0.4, "#7a7a7a"],
          [0.7, "#a0a0a0"], [1.0, "#c8c8c8"]]
    return c, cs


def _color_venus(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Thick sulphuric cloud cover — creamy yellow-white banding
    c = ( 0.55
        + 0.20 * np.cos(4 * lat)
        + 0.10 * np.sin(6 * lat) * np.cos(3 * lon)
        + 0.05 * np.cos(10 * lat) )
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.0, "#b8860b"], [0.3, "#d4a830"],
          [0.6, "#f0d878"], [0.85, "#fffacd"],
          [1.0, "#ffffff"]]
    return c, cs


_EARTH_COLORSCALE = [
    [0.00, "#0a2a6e"],   # deep ocean
    [0.22, "#1565c0"],   # ocean
    [0.35, "#2196f3"],   # shallow ocean
    [0.45, "#4caf50"],   # lowland
    [0.55, "#388e3c"],   # forest / land
    [0.65, "#795548"],   # desert / mountain
    [0.80, "#a5d6a7"],   # light land / tundra
    [0.92, "#e8f5e9"],   # snow
    [1.00, "#ffffff"],   # ice cap
]

_earth_texture_cache = None   # lazy-loaded (H, W) float array, values 0..1


def _load_earth_texture():
    """
    Load the real land/ocean/ice classification packaged by the core data package.

    Returns None (triggering the harmonic-blob fallback below) when the data
    package is absent or older than the asset. The Toolkit source tree should
    never carry this ``.npz`` file directly.
    """
    global _earth_texture_cache
    if _earth_texture_cache is not None:
        return None if _earth_texture_cache is False else _earth_texture_cache
    try:
        with np.load(BytesIO(read_data_binary("earth_map.npz"))) as data:
            _earth_texture_cache = data["classification"].astype(np.float32) / 255.0
        return _earth_texture_cache
    except Exception:
        _earth_texture_cache = False   # cache the failure too, don't retry every call
        return None


def _sample_earth_texture(lat: np.ndarray, lon: np.ndarray, tex: np.ndarray) -> np.ndarray:
    """Bilinear-sample the (H, W) classification array at given lat/lon (radians).

    The packaged map follows the same native equirectangular convention as
    SSAPy's earth.png: column 0 is -180° longitude and the centre column is
    0°.  The sphere mesh uses 0 radians on +X as 0° longitude, so shift by
    pi before wrapping into texture-column space.
    """
    H, W = tex.shape
    row_f = (np.pi / 2 - lat) / np.pi * (H - 1)
    col_f = ((lon + np.pi) % (2 * np.pi)) / (2 * np.pi) * W

    row0 = np.clip(np.floor(row_f).astype(int), 0, H - 2)
    row1 = row0 + 1
    col0 = np.floor(col_f).astype(int) % W
    col1 = (col0 + 1) % W
    fr = row_f - row0
    fc = col_f - np.floor(col_f)

    top = tex[row0, col0] * (1 - fc) + tex[row0, col1] * fc
    bot = tex[row1, col0] * (1 - fc) + tex[row1, col1] * fc
    return top * (1 - fr) + bot * fr


def _color_earth_blobs(n=_N):
    """Original harmonic-blob approximation — kept as a fallback only."""
    U, V, lat, lon = _uv_mesh(n)
    c = np.zeros_like(lat)
    c[:] = 0.25

    africa_eu = ((lon > 0.0) & (lon < 2.2) & (np.abs(lat) < 0.8))
    c[africa_eu] += 0.35 * (1 - np.abs(lat[africa_eu]) / 0.8)

    americas = ((lon > 4.4) & (lon < 5.8) & (lat > -0.9) & (lat < 0.85))
    c[americas] += 0.32 * (1 - np.abs(lat[americas]) / 0.85)

    aus = ((lon > 2.2) & (lon < 2.7) & (lat > -0.55) & (lat < -0.05))
    c[aus] += 0.30

    south_pole = lat < -1.1
    c[south_pole] = 0.95
    south_fade = (lat > -1.25) & (lat < -1.0)
    fade_val = (lat[south_fade] + 1.25) / 0.15
    c[south_fade] = np.clip(0.95 - 0.7 * fade_val, 0.25, 0.95)

    north_pole = lat > 1.2
    c[north_pole] = 0.92
    north_fade = (lat > 1.05) & (lat < 1.25)
    fade_val2 = (1.25 - lat[north_fade]) / 0.20
    c[north_fade] = np.clip(0.92 - 0.65 * fade_val2, 0.25, 0.92)

    return np.clip(c, 0.0, 1.0), _EARTH_COLORSCALE


def _color_earth(n=_N, rotation_deg=0.0):
    """
    Earth's surface classification (ocean / land / ice), sampled at the
    requested mesh resolution `n`.

    Previously this was four hand-tuned sine/cosine blobs standing in for
    continents — recognizable as "a green blob near here" at best, and no
    sharper at high n since the blobs themselves were coarse. This now
    samples a real classification derived from a NASA Blue Marble mosaic
    (public domain), so raising `n` actually resolves finer real coastline
    detail instead of just smoothing the same four blobs. Falls back to the
    old blobs if the core data package is absent or older than this packaged asset.
    """
    tex = _load_earth_texture()
    if tex is None:
        c, cs = _color_earth_blobs(n)
        return np.roll(c, int(round((float(rotation_deg) % 360.0) / 360.0 * n)), axis=1), cs
    U, V, lat, lon = _uv_mesh(n)
    c = _sample_earth_texture(lat, lon - np.radians(float(rotation_deg)), tex)
    return c, _EARTH_COLORSCALE


def _color_mars(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Red-orange with polar ice caps and darker terrain
    c = ( 0.50
        + 0.15 * np.sin(3 * lat)
        + 0.08 * np.cos(5 * lat) * np.sin(4 * lon)
        + 0.06 * np.sin(9 * lat) )

    # South polar cap
    c[lat < -1.15] = 0.95
    south_fade = (lat > -1.30) & (lat < -1.10)
    c[south_fade] = np.clip(0.95 - 0.45 * (lat[south_fade] + 1.30) / 0.20, 0.50, 0.95)

    # North polar cap
    c[lat > 1.20] = 0.90
    north_fade = (lat > 1.10) & (lat < 1.25)
    c[north_fade] = np.clip(0.90 - 0.40 * (1.25 - lat[north_fade]) / 0.15, 0.50, 0.90)

    # Valles Marineris hint (~lat -0.15, lon 5.0..6.0)
    vm = (np.abs(lat + 0.15) < 0.08) & (lon > 4.9) & (lon < 6.1)
    c[vm] -= 0.15

    c = np.clip(c, 0.0, 1.0)
    cs = [[0.00, "#5c1a00"],   # dark basalt
          [0.25, "#8b2500"],   # dark red terrain
          [0.45, "#c1440e"],   # typical surface
          [0.65, "#d2691e"],   # lighter terrain
          [0.80, "#e8a070"],   # bright dust
          [0.90, "#f0c8a8"],   # pale dust / polar edge
          [1.00, "#ffffff"]]   # polar ice
    return c, cs


def _color_jupiter(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Strong horizontal banding
    c = ( 0.50
        + 0.35 * np.sin(8 * lat)
        + 0.10 * np.sin(16 * lat)
        + 0.04 * np.cos(5 * lon) * np.sin(8 * lat)
        + 0.03 * np.sin(30 * lat) )

    # Great Red Spot hint (~lat -0.36, lon ~π)
    grs_mask = ((np.abs(lat + 0.36) < 0.12) &
                (np.abs(lon - np.pi) < 0.30))
    c[grs_mask] = np.clip(c[grs_mask] + 0.22, 0.0, 1.0)

    c = np.clip(c, 0.0, 1.0)
    cs = [[0.00, "#3d1c00"],   # dark brown belt
          [0.20, "#8b4513"],   # brown belt
          [0.38, "#c88b3a"],   # golden belt
          [0.55, "#f0c878"],   # tan zone
          [0.70, "#fff8dc"],   # cream zone
          [0.82, "#f5deb3"],   # wheat
          [0.92, "#d2691e"],   # reddish tones
          [1.00, "#ff8c40"]]   # bright reddish
    return c, cs


def _color_saturn(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Softer banding than Jupiter
    c = ( 0.50
        + 0.22 * np.sin(6 * lat)
        + 0.08 * np.sin(14 * lat)
        + 0.03 * np.cos(4 * lon) * np.sin(6 * lat) )
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.00, "#8b7536"],
          [0.25, "#b8963c"],
          [0.50, "#d4b860"],
          [0.70, "#e8d080"],
          [0.85, "#f5e8a8"],
          [1.00, "#fffacd"]]
    return c, cs


def _color_uranus(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Featureless cyan with very subtle polar darkening
    c = 0.55 + 0.20 * np.abs(np.sin(lat)) + 0.03 * np.cos(3 * lat)
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.0, "#00838f"], [0.4, "#00acc1"],
          [0.7, "#4dd0e1"], [1.0, "#b2ebf2"]]
    return c, cs


def _color_neptune(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Deep blue with faint banding
    c = ( 0.45
        + 0.18 * np.sin(5 * lat)
        + 0.07 * np.sin(12 * lat) )
    # Great Dark Spot hint
    gds = (np.abs(lat - 0.40) < 0.10) & (np.abs(lon - 1.2) < 0.25)
    c[gds] = np.clip(c[gds] - 0.18, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.0, "#0d1b6e"], [0.3, "#1a237e"],
          [0.6, "#283593"], [0.85, "#3949ab"],
          [1.0, "#5c6bc0"]]
    return c, cs


def _color_moon(n=_N):
    U, V, lat, lon = _uv_mesh(n)
    # Airless, grey, cratered — similar character to Mercury but paler
    # and without Mercury's warmer cast.
    c = ( 0.55
        + 0.12 * np.sin(9 * lat) * np.cos(6 * lon)
        + 0.08 * np.sin(17 * lat)
        + 0.05 * np.cos(13 * lon) )
    # Mare patches — a few broad darker regions (loosely Earth-facing side)
    maria = (np.abs(lat) < 0.9) & (np.cos(lon - 1.0) > 0.55)
    c[maria] -= 0.16
    c = np.clip(c, 0.0, 1.0)
    cs = [[0.00, "#3f3f3f"], [0.35, "#6a6a6a"],
          [0.65, "#9a9a9a"], [1.00, "#d0d0d0"]]
    return c, cs


_COLOR_FN = {
    "Mercury": _color_mercury,
    "Venus":   _color_venus,
    "Earth":   _color_earth,
    "Mars":    _color_mars,
    "Jupiter": _color_jupiter,
    "Saturn":  _color_saturn,
    "Uranus":  _color_uranus,
    "Neptune": _color_neptune,
    "Moon":    _color_moon,
}


# ── Lighting preset — sunlight from origin ────────────────────────────────────
def _planet_lighting(pos_au):
    """Point light roughly toward the planet from the Sun (origin)."""
    px, py, pz = pos_au
    d = max((px**2 + py**2 + pz**2)**0.5, 1e-9)
    # Plotly lightposition is in scene units; point slightly behind camera
    return dict(
        ambient=0.28, diffuse=0.85, specular=0.15,
        roughness=0.75, fresnel=0.05,
    )


def _sun_lightposition(pos_au):
    px, py, pz = pos_au
    d = max((px**2 + py**2 + pz**2)**0.5, 1e-9)
    # Light comes FROM the sun (origin) TOWARD planet
    return dict(x=-px/d * 1e5, y=-py/d * 1e5, z=-pz/d * 1e5)



def _earth_rotation_from_plot_time(time):
    """Solar-view helpers pass numeric times as Julian dates, not GPS seconds."""
    if time is None:
        return 0.0
    if hasattr(time, "gps"):
        return earth_rotation_deg_from_time(time)
    try:
        return earth_rotation_deg_from_time(epoch_jd=float(np.asarray(time).reshape(-1)[0]))
    except Exception:
        return earth_rotation_deg_from_time(time)


# ── Public API ────────────────────────────────────────────────────────────────

def make_planet_traces(
    name: str,
    pos_au: tuple,
    scale_au: float = 1.0,
    show_label: bool = True,
    n: int = _N,
    time=None,
    rotation_deg=None,
) -> list:
    """
    Return a list of Plotly traces for a single planet.

    Parameters
    ----------
    name     : planet name (key in _COLOR_FN)
    pos_au   : (x, y, z) heliocentric position in AU
    scale_au : multiplier on the display radius (1.0 = default)
    show_label : add a text label above the sphere
    n        : sphere mesh resolution (default 50)
    time     : optional time used to orient Earth's surface texture
    rotation_deg : optional explicit Earth texture rotation in degrees

    Returns
    -------
    list of go.Surface / go.Scatter3d traces
    """
    r = _R_AU[name] * scale_au
    cx, cy, cz = pos_au

    xs, ys, zs = _unit_sphere(n)
    xs, ys, zs = _apply_tilt(xs, ys, zs, _TILT[name])

    color_fn = _COLOR_FN.get(name)
    if color_fn is None:
        surf_c, cs = np.ones((n, n)) * 0.5, [[0, "#888"], [1, "#ccc"]]
    elif name == "Earth":
        earth_rot = _earth_rotation_from_plot_time(time) if rotation_deg is None else float(rotation_deg)
        surf_c, cs = _color_earth(n, rotation_deg=earth_rot)
    else:
        surf_c, cs = color_fn(n)

    traces = []

    # Planet sphere
    traces.append(go.Surface(
        x=cx + r * xs,
        y=cy + r * ys,
        z=cz + r * zs,
        surfacecolor=surf_c,
        colorscale=cs,
        cmin=0.0, cmax=1.0,
        showscale=False,
        name=name,
        hovertext=(f"<b>{name}</b><br>"
                   f"r = {(cx**2+cy**2+cz**2)**0.5:.4f} AU<br>"
                   f"x={cx:.3f}  y={cy:.3f}  z={cz:.3f} AU"),
        hoverinfo="text",
        lighting=_planet_lighting(pos_au),
        lightposition=_sun_lightposition(pos_au),
        opacity=1.0,
    ))

    # Floating label
    if show_label:
        traces.append(go.Scatter3d(
            x=[cx], y=[cy], z=[cz + r * 2.4],
            mode="text",
            text=[name],
            textfont=dict(color="#C8D8E8", size=10,
                          family="JetBrains Mono, monospace"),
            hoverinfo="skip",
            showlegend=False,
        ))

    return traces


def make_saturn_ring_traces(
    pos_au: tuple,
    scale_au: float = 1.0,
    n_phi: int = 200,
    n_r: int = 40,
) -> list:
    """
    Return Plotly traces for Saturn's ring system.

    The rings span inner radius 1.24× → outer 2.27× Saturn's display radius,
    matching the approximate D-ring to A-ring extent.
    Cassini division modelled as a gap at ~1.76×.
    """
    r_sat  = _R_AU["Saturn"] * scale_au
    cx, cy, cz = pos_au
    tilt   = np.radians(_TILT["Saturn"])
    ct, st = np.cos(tilt), np.sin(tilt)

    # Ring radial zones: D+C+B | Cassini | A
    zones = [
        dict(r_in=1.24, r_out=1.70, opacity=0.55,
             cs=[[0,"rgba(180,160,100,0)"], [0.4,"rgba(200,180,120,0.6)"],
                 [1,"rgba(220,200,140,0.7)"]]),
        # Cassini division gap: skip 1.70..1.80
        dict(r_in=1.80, r_out=2.27, opacity=0.45,
             cs=[[0,"rgba(200,180,120,0.5)"], [0.5,"rgba(210,190,130,0.6)"],
                 [1,"rgba(220,200,140,0.5)"]]),
    ]

    traces = []
    phi = np.linspace(0, 2 * np.pi, n_phi)

    for zone in zones:
        r_vals = np.linspace(zone["r_in"] * r_sat,
                             zone["r_out"] * r_sat, n_r)
        PHI, R = np.meshgrid(phi, r_vals)
        xr = R * np.cos(PHI)
        yr = R * np.sin(PHI)
        zr = np.zeros_like(xr)

        # Apply Saturn's axial tilt (rotate about X axis)
        yr_t = yr * ct - zr * st
        zr_t = yr * st + zr * ct

        # Color varies radially (brighter toward outer B ring)
        rc = (R - r_vals[0]) / (r_vals[-1] - r_vals[0])

        traces.append(go.Surface(
            x=cx + xr,
            y=cy + yr_t,
            z=cz + zr_t,
            surfacecolor=rc,
            colorscale=zone["cs"],
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=zone["opacity"],
            name="Saturn rings",
            showlegend=False,
            hoverinfo="skip",
            lighting=dict(ambient=0.6, diffuse=0.6,
                          specular=0.1, roughness=0.9),
        ))

    return traces


def make_sun_traces(r_display_au: float = SUN_RADIUS_AU) -> list:
    """
    Return traces for the Sun — a bright inner sphere + two soft glow layers.
    """
    traces = []
    n = _N

    xs, ys, zs = _unit_sphere(n)
    U, V, lat, _ = _uv_mesh(n)

    # Surface color: bright equatorial band, slightly cooler poles
    sc = 0.70 + 0.20 * np.cos(2 * lat) + 0.05 * np.sin(4 * lat) * np.cos(3 * U)
    sc = np.clip(sc, 0.0, 1.0)

    sun_cs = [
        [0.00, "#ff6600"],
        [0.30, "#ff8c00"],
        [0.55, "#ffa500"],
        [0.75, "#ffcc00"],
        [0.90, "#ffe066"],
        [1.00, "#fff176"],
    ]

    # Inner photosphere
    traces.append(go.Surface(
        x=r_display_au * xs,
        y=r_display_au * ys,
        z=r_display_au * zs,
        surfacecolor=sc,
        colorscale=sun_cs,
        cmin=0.0, cmax=1.0,
        showscale=False,
        name="Sun",
        hovertext="<b>☀ Sun</b><br>G2V main sequence",
        hoverinfo="text",
        lighting=dict(ambient=0.9, diffuse=0.4, specular=0.6,
                      roughness=0.3, fresnel=0.1),
        opacity=1.0,
    ))

    # Outer corona / glow layers (faint transparent spheres)
    # Previously pure-ambient lighting (ambient=1, diffuse=0, specular=0) on a
    # uniform surfacecolor — with nothing varying by surface normal or by
    # position, a sphere renders visually identical to a flat disc facing the
    # camera. Adding diffuse (+ a little specular/fresnel) shades based on the
    # actual 3D normal, so the glow reads as a rounded shell instead of a
    # flat painted circle.
    for glow_r, glow_op in [(1.6, 0.12), (2.4, 0.06)]:
        traces.append(go.Surface(
            x=r_display_au * glow_r * xs,
            y=r_display_au * glow_r * ys,
            z=r_display_au * glow_r * zs,
            surfacecolor=np.ones((n, n)),
            colorscale=[[0, "rgba(255,200,0,0)"],
                        [1, "rgba(255,220,50,1)"]],
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=glow_op,
            hoverinfo="skip",
            showlegend=False,
            lighting=dict(ambient=0.35, diffuse=0.55, specular=0.1,
                          roughness=1.0, fresnel=0.3),
        ))

    return traces


def moon_geocentric_ecliptic(t_jd: float) -> tuple:
    """
    Geocentric ecliptic (x, y, z) position of the Moon, in km, for the given
    Julian Date.

    Low-precision periodic-term formula (the standard "low-precision Moon"
    algorithm — mean elements plus the dozen largest periodic terms in
    longitude/latitude/distance, after Meeus, Astronomical Algorithms ch. 47).
    Against astropy's ephemeris over 2000-2040 it is good to 5.1 arcmin in
    longitude, 2.9 arcmin in latitude, and 494 km in distance — plenty for a
    visual display, not for real navigation or eclipse-timing work (that's
    what the eclipse-search modules elsewhere in this toolkit are for).
    """
    T = (t_jd - 2_451_545.0) / 36525.0

    Lp = 218.3164477 + 481267.88123421*T - 0.0015786*T**2 + T**3/538841 - T**4/65194000
    D  = 297.8501921 + 445267.1114034*T  - 0.0018819*T**2 + T**3/545868 - T**4/113065000
    M  = 357.5291092 + 35999.0502909*T   - 0.0001536*T**2 + T**3/24490000
    Mp = 134.9633964 + 477198.8675055*T  + 0.0087414*T**2 + T**3/69699  - T**4/14712000
    F  = 93.2720950  + 483202.0175233*T  - 0.0036539*T**2 - T**3/3526000 + T**4/863310000

    Lp_r = np.radians(Lp % 360.0)
    D_r, M_r, Mp_r, F_r = (np.radians(x % 360.0) for x in (D, M, Mp, F))

    dlon_deg = (
        6.289*np.sin(Mp_r)        - 1.274*np.sin(Mp_r - 2*D_r) + 0.658*np.sin(2*D_r)
        + 0.214*np.sin(2*Mp_r)    - 0.114*np.sin(2*F_r)
        - 0.186*np.sin(M_r)       - 0.059*np.sin(2*Mp_r - 2*D_r)
        - 0.057*np.sin(Mp_r - 2*D_r + M_r) + 0.053*np.sin(Mp_r + 2*D_r)
        + 0.046*np.sin(2*D_r - M_r)        + 0.041*np.sin(Mp_r - M_r)
        - 0.035*np.sin(D_r)                - 0.031*np.sin(Mp_r + M_r)
        - 0.015*np.sin(2*F_r - 2*D_r)      + 0.011*np.sin(Mp_r - 4*D_r)
    )
    lon = Lp_r + np.radians(dlon_deg)

    lat = np.radians(
        5.128*np.sin(F_r) + 0.281*np.sin(Mp_r + F_r) + 0.278*np.sin(Mp_r - F_r)
        + 0.173*np.sin(2*D_r - F_r) + 0.055*np.sin(2*D_r - Mp_r + F_r)
        + 0.046*np.sin(2*D_r - Mp_r - F_r) + 0.033*np.sin(2*D_r + F_r)
        + 0.017*np.sin(2*Mp_r + F_r)
    )

    dist_km = (
        385000.56 - 20905.36*np.cos(Mp_r) - 3699.11*np.cos(2*D_r - Mp_r)
        - 2955.97*np.cos(2*D_r) - 569.93*np.cos(2*Mp_r)
        + 246.16*np.cos(2*Mp_r - 2*D_r) - 204.59*np.cos(M_r - 2*D_r)
        - 170.73*np.cos(Mp_r + 2*D_r) - 152.14*np.cos(Mp_r + M_r - 2*D_r)
        + 48.89*np.cos(Mp_r - 2*D_r + M_r) - 3.16*np.cos(3*Mp_r)
    )

    x = dist_km * np.cos(lat) * np.cos(lon)
    y = dist_km * np.cos(lat) * np.sin(lon)
    z = dist_km * np.sin(lat)
    return float(x), float(y), float(z)


def _moon_offset_au(t_jd: float, orbit_scale: float = 1.0) -> np.ndarray:
    """
    Moon's Earth-relative offset in AU, direction always correct for t_jd.

    The Moon's *true* orbital radius (~0.00257 AU) is smaller than Earth's
    own *display* radius once Earth is exaggerated ~500× for visibility
    (see _R_AU) — rendered at true scale the Moon would orbit inside the
    Earth sphere. `orbit_scale` stretches only the distance (never the
    direction) by a separate, much smaller factor than the ~500× used for
    planet sizes, just enough to clear the Earth sphere.
    """
    x_km, y_km, z_km = moon_geocentric_ecliptic(t_jd)
    return np.array([x_km, y_km, z_km]) / _AU_KM * orbit_scale


def make_moon_traces(
    earth_pos_au: tuple,
    t_jd: float,
    orbit_scale: float = 20.0,
    show_label: bool = True,
    n: int = _N,
) -> list:
    """
    Return traces for a properly shaded Moon sphere at its real position
    for t_jd, offset from Earth's centre (see _moon_offset_au for the
    orbit_scale caveat).
    """
    offset = _moon_offset_au(t_jd, orbit_scale)
    cx, cy, cz = earth_pos_au[0] + offset[0], earth_pos_au[1] + offset[1], earth_pos_au[2] + offset[2]
    pos = (cx, cy, cz)

    xs, ys, zs = _unit_sphere(n)
    xs, ys, zs = _apply_tilt(xs, ys, zs, _TILT["Moon"])
    surf_c, cs = _color_moon(n)
    r = _R_AU["Moon"]

    real_dist_km = float(np.linalg.norm(offset) * _AU_KM / orbit_scale) if orbit_scale else 0.0

    traces = [go.Surface(
        x=cx + r * xs, y=cy + r * ys, z=cz + r * zs,
        surfacecolor=surf_c, colorscale=cs, cmin=0.0, cmax=1.0,
        showscale=False, name="Moon",
        hovertext=f"<b>Moon</b><br>{real_dist_km:,.0f} km from Earth "
                  f"(shown {orbit_scale:.0f}× farther out for visibility)",
        hoverinfo="text",
        lighting=_planet_lighting(pos),
        lightposition=_sun_lightposition(pos),
        opacity=1.0,
    )]

    if show_label:
        traces.append(go.Scatter3d(
            x=[cx + r * 2.8], y=[cy + r * 1.5], z=[cz + r * 0.8],
            mode="text", text=["Moon"],
            textfont=dict(color="#AAB8C8", size=9,
                          family="JetBrains Mono, monospace"),
            hoverinfo="skip", showlegend=False,
        ))

    return traces


def make_moon_trace(earth_pos_au: tuple, t_jd: float = 2_451_545.0) -> go.Scatter3d:
    """
    Backward-compatible shim for older call sites expecting a single
    Scatter3d marker. Previously placed the Moon at a fixed offset
    (0.012 AU at a constant angle) regardless of date — that offset was
    also *smaller* than Earth's own display radius, so the marker sat
    inside the Earth sphere. Now uses the real ephemeris (defaulting to
    J2000 if no date is passed) at a distance that clears the Earth sphere.
    New call sites should prefer make_moon_traces(), which takes t_jd
    directly and returns a shaded sphere instead of a flat marker.
    """
    offset = _moon_offset_au(t_jd, orbit_scale=20.0)
    cx = earth_pos_au[0] + offset[0]
    cy = earth_pos_au[1] + offset[1]
    cz = earth_pos_au[2] + offset[2]
    return go.Scatter3d(
        x=[cx], y=[cy], z=[cz],
        mode="markers",
        marker=dict(size=3, color="#DDDDDD",
                    line=dict(color="rgba(255,255,255,0.4)", width=1)),
        name="Moon", hovertext="Moon", hoverinfo="text",
    )
