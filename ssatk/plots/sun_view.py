"""
ssatk/plots/sun_view.py
--------------------------------
Sun model and shadow-terminator utilities for SSATK's 3D Plotly plots.

Public API
----------
  sun_position_eci(t)                    -> (unit_vec: ndarray, dist_au: float)
  SunLayer(sun_pos_eci, ...)             -> .build_traces() -> list[go.Surface | go.Scatter3d]
  EarthShadingLayer(sun_pos_eci, ...)    -> .build_traces() -> list[go.Mesh3d | go.Surface]
  MoonShadingLayer(sun_pos_eci, moon_center, ...) -> .build_traces() -> list[go.Mesh3d | go.Surface]
  starfield_trace(sky_radius, mag_limit=6.5) -> go.Scatter3d

Usage
-----
::

    from ssatk.plots.sun_view import (
        sun_position_eci, SunLayer, EarthShadingLayer, starfield_trace,
        VISUAL_DIST_KM_LEO,
    )

    sun_hat, dist_au = sun_position_eci(t)
    sun_pos = sun_hat * VISUAL_DIST_KM_LEO

    fig.add_trace(starfield_trace(sky_radius=VISUAL_DIST_KM_LEO * 20))
    fig.add_traces(SunLayer(sun_pos).build_traces())
    fig.add_traces(EarthShadingLayer(sun_pos).build_traces())

(There is no Scene/add_layer registry in this toolkit -- traces are built
directly and added to a plotly Figure with fig.add_traces(), as above.)

Note: EarthShadingLayer / MoonShadingLayer render the full, opaque Earth /
Moon sphere themselves, with day/night baked in as solid color -- they are
not a transparent overlay to stack on top of a separately-drawn sphere.
(An earlier version tried a semi-transparent overlay approach; Plotly's
Surface/Mesh3d traces only support one opacity value for an entire trace,
not per-vertex alpha, so that didn't actually render correctly -- see the
comment above _EARTH_DAYNIGHT_COLORSCALE for details.)
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS

from ssatk.constants import AU_KM, EARTH_RADIUS, MOON_RADIUS, SUN_RADIUS
from ssatk.plots.scene_primitives import earth_rotation_deg_from_time, stabilize_sphere_poles

# ---------------------------------------------------------------------------
# Physical / display constants
# ---------------------------------------------------------------------------
# EARTH_RADIUS / MOON_RADIUS / SUN_RADIUS come from ssatk.constants
# (WGS84 equatorial radius for Earth, in *meters*) rather than being
# redefined here -- this module works in km throughout (matching the rest
# of the toolkit's Plotly scene code), so they're converted once, here,
# explicitly.
R_EARTH_KM = EARTH_RADIUS / 1000.0
R_MOON_KM  = MOON_RADIUS / 1000.0
R_SUN_KM   = SUN_RADIUS / 1000.0

# AU_KM comes from ssatk.constants so every plot uses the same
# physical distance scale.

# Visual sun placement: distance in the scene is compressed for readable
# Earth-orbit plots, but the display radius preserves the Sun's physical
# angular radius at that compressed distance.
VISUAL_DIST_KM_LEO      = 80_000.0    # good for LEO / GEO
VISUAL_DIST_KM_CISLUNAR = 600_000.0   # good for cislunar / Moon plots
VISUAL_SUN_RADIUS_KM    = VISUAL_DIST_KM_LEO * (R_SUN_KM / AU_KM)

# Corona glow rings: (scale_factor_vs_core, opacity, colour)
_CORONA_RINGS = [
    (1.35, 0.22, "rgba(255, 200,  60, {a})"),
    (1.80, 0.12, "rgba(255, 150,  20, {a})"),
    (2.60, 0.05, "rgba(255, 100,   0, {a})"),
]

# Earth/Moon day-night shading: solid, fully-opaque RGB blend from a lit
# color to a dark night color.
#
# An earlier version of this tried to encode a soft, spatially-varying
# transparency (day side clear, night side dark) via alpha embedded in
# "rgba(...)" colorscale strings, layered as a separate overlay sphere on
# top of an existing Earth/Moon trace. That does not work: Plotly's
# go.Surface (and go.Mesh3d) only support a single opacity value for the
# *entire* trace -- there is no per-vertex transparency in either 3D
# surface trace type, in any current Plotly version. The alpha channel in
# the colorscale strings was silently ignored, which is why it rendered as
# a single solid opaque color with no visible gradient at all (confirmed
# by an actual screenshot, not just a suspicion).
#
# The fix: don't rely on transparency. EarthShadingLayer / MoonShadingLayer
# now render the full, fully-opaque body sphere themselves, with the
# day/night terminator baked directly into solid RGB color -- the same
# proven approach already used successfully elsewhere in this toolkit for
# the matplotlib case (see sun_mpl.py's shade_texture/apply_shading, which
# multiplies texture brightness by a lit factor and outputs solid colors,
# never relying on alpha compositing). This means these classes now
# *replace* needing a separate plain Earth/Moon trace, rather than
# stacking on top of one -- see the updated module docstring.
_EARTH_DAYNIGHT_COLORSCALE = [
    [0.00, "rgb( 70, 130, 220)"],   # full sun  - lit blue
    [0.40, "rgb( 40,  80, 150)"],   # penumbra
    [0.72, "rgb( 15,  30,  70)"],   # deep penumbra
    [1.00, "rgb(  5,  10,  30)"],   # umbra core - near black
]

_MOON_DAYNIGHT_COLORSCALE = [
    [0.00, "rgb(200, 200, 195)"],   # full sun  - lit grey
    [0.40, "rgb(120, 115, 115)"],
    [0.72, "rgb( 55,  50,  55)"],
    [1.00, "rgb( 15,  12,  15)"],   # umbra core - near black
]


# ---------------------------------------------------------------------------
# Solar ephemeris
# ---------------------------------------------------------------------------

def sun_position_eci(t) -> tuple[np.ndarray, float]:
    """Return the geocentric ECI (GCRS) unit vector toward the Sun, plus
    the Earth-Sun distance in AU.

    This uses Astropy's configured solar ephemeris, which defaults to its
    offline built-in model. Use :class:`astropy.coordinates.solar_system_ephemeris`
    when a JPL kernel is required.

    Parameters
    ----------
    t : astropy.time.Time or float
        Epoch. A bare float is interpreted as a Julian Date (UTC), matching
        this function's original signature; pass an astropy.time.Time
        directly for full control over the time scale.

    Returns
    -------
    sun_hat : ndarray, shape (3,)
        Unit vector from Earth to Sun in GCRS/ECI.
    dist_au : float
        Earth-Sun distance in AU.
    """
    if not isinstance(t, Time):
        t = Time(float(t), format="jd", scale="utc")

    sun_gcrs = get_body("sun", t).transform_to(GCRS(obstime=t))

    r_sun_m = sun_gcrs.cartesian.xyz.to(u.m).value
    dist_m = float(np.linalg.norm(r_sun_m))
    sun_hat = r_sun_m / dist_m
    dist_au = (dist_m * u.m).to(u.au).value

    return sun_hat, dist_au


def jd_from_datetime(dt) -> float:
    """Convert a Python datetime (UTC) to Julian Date, via astropy."""
    import datetime
    if not isinstance(dt, datetime.datetime):
        raise TypeError("dt must be a datetime.datetime")
    return Time(dt, scale="utc").jd


# ---------------------------------------------------------------------------
# Helper: unit sphere mesh
# ---------------------------------------------------------------------------

def _unit_sphere_mesh(nu: int = 50, nv: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) arrays for a unit sphere surface."""
    uu_axis = np.linspace(0, 2 * np.pi, nu)
    vv_axis = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(uu_axis, vv_axis)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    return x, y, z


# ---------------------------------------------------------------------------
# SunLayer
# ---------------------------------------------------------------------------

class SunLayer:
    """Plotly traces forming a glowing 3D sun model.

    The sun is rendered as:
      - A bright inner sphere (yellow -> white core)
      - 3 nested corona rings with decreasing opacity and warm orange tones
      - A directional label annotation (optional)

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Position of the sun *centre* in the scene coordinate frame (km).
        Typically ``sun_hat * VISUAL_DIST_KM_LEO``. The direction is what
        matters visually; the magnitude sets how far the sphere is placed.
    radius_km : float, optional
        Display radius of the core sun sphere (km). Default 5,500 km.
    show_corona : bool
        Add concentric glow rings. Default True.
    show_label : bool
        Add a text annotation "Sun". Default True.
    name_prefix : str
        Prefix for trace names / legendgroups.
    """

    def __init__(
        self,
        sun_pos_eci,
        radius_km: float = VISUAL_SUN_RADIUS_KM,
        show_corona: bool = True,
        show_label: bool = True,
        name_prefix: str = "Sun",
    ):
        self.pos = np.asarray(sun_pos_eci, dtype=float)
        self.radius = radius_km
        self.show_corona = show_corona
        self.show_label = show_label
        self.name_prefix = name_prefix

    def _core_sphere(self) -> go.Surface:
        """Bright inner sphere with a yellow-white radial gradient."""
        x, y, z = _unit_sphere_mesh(nu=60, nv=30)
        brightness = np.clip(0.6 + 0.4 * z, 0, 1)  # [0.6, 1.0]

        colorscale = [
            [0.0, "rgb(255, 180,  20)"],   # warm gold (limb)
            [0.5, "rgb(255, 220,  80)"],   # yellow
            [0.8, "rgb(255, 245, 180)"],   # pale yellow
            [1.0, "rgb(255, 255, 240)"],   # near-white core
        ]

        return go.Surface(
            x=self.pos[0] + self.radius * x,
            y=self.pos[1] + self.radius * y,
            z=self.pos[2] + self.radius * z,
            surfacecolor=brightness,
            colorscale=colorscale,
            cmin=0, cmax=1,
            showscale=False,
            opacity=1.0,
            name=self.name_prefix,
            legendgroup=self.name_prefix,
            showlegend=True,
            hoverinfo="name",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0),
        )

    def _corona_sphere(self, scale: float, opacity: float, colour_template: str) -> go.Surface:
        """One translucent corona halo shell."""
        x, y, z = _unit_sphere_mesh(nu=40, nv=20)
        r = self.radius * scale
        colour = colour_template.format(a=opacity)

        return go.Surface(
            x=self.pos[0] + r * x,
            y=self.pos[1] + r * y,
            z=self.pos[2] + r * z,
            surfacecolor=np.ones_like(x),
            colorscale=[[0, colour], [1, colour]],
            cmin=0, cmax=1,
            showscale=False,
            opacity=opacity,
            name=f"{self.name_prefix} corona",
            legendgroup=self.name_prefix,
            showlegend=False,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )

    def _label_scatter(self) -> go.Scatter3d:
        """Small text annotation at the sun position."""
        return go.Scatter3d(
            x=[self.pos[0]],
            y=[self.pos[1]],
            z=[self.pos[2] + self.radius * 2.8],
            mode="text",
            text=["Sun"],
            textfont=dict(color="rgba(255,230,100,0.85)", size=12),
            name=f"{self.name_prefix} label",
            legendgroup=self.name_prefix,
            showlegend=False,
            hoverinfo="skip",
        )

    def build_traces(self) -> list:
        """Return a list of Plotly traces ready to add to a Figure."""
        traces = [self._core_sphere()]
        if self.show_corona:
            for scale, opacity, colour in _CORONA_RINGS:
                traces.append(self._corona_sphere(scale, opacity, colour))
        if self.show_label:
            traces.append(self._label_scatter())
        return traces


# ---------------------------------------------------------------------------
# Shadow terminator helpers
# ---------------------------------------------------------------------------

def _shadow_surface_color(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sun_hat: np.ndarray,
    R: float,
) -> np.ndarray:
    """Compute per-vertex shadow intensity in [0, 1].

    0  -> full sunlight (transparent overlay)
    1  -> deep shadow / umbra (opaque dark overlay)

    The transition spans ~10 degrees around the terminator for a soft
    penumbra.
    """
    dot = (x * sun_hat[0] + y * sun_hat[1] + z * sun_hat[2]) / R

    width = 0.17
    shadow = np.clip((-dot) / (1.0 + width) + width / (1.0 + width), 0.0, 1.0)

    return shadow ** 0.7


# ---------------------------------------------------------------------------
# EarthShadingLayer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Real texture loading (reusing the toolkit's existing pattern)
# ---------------------------------------------------------------------------

def _find_texture(name):
    """Locate a real planetary texture (e.g. "earth", "moon") as a .png.

    Same pattern already established elsewhere in this toolkit: try the
    installed ssapy package first via ssapy.utils.find_file, which is robust regardless of
    where SSAPy and SSATK are checked out relative to each other.
    Returns None (not a fabricated path) if unavailable, so callers can
    fall back to a plain color gradient instead of crashing.
    """
    try:
        from ssapy.utils import find_file
        return find_file(name, ext=".png")
    except Exception:
        return None


def _uv_sphere_with_faces(nu: int = 60, nv: int = 30):
    """Build a UV-sphere as flat vertex arrays plus triangle face indices,
    for use with go.Mesh3d (which needs explicit connectivity, unlike
    go.Surface's implicit grid).

    Returns
    -------
    x, y, z : ndarray, shape (nu*nv,)
        Flattened unit-sphere vertex coordinates.
    phi, theta : ndarray, shape (nu*nv,)
        The corresponding spherical angles for each vertex (needed for
        both shading and texture-coordinate lookup).
    i, j, k : ndarray, shape (num_faces,)
        Triangle vertex indices, in the format go.Mesh3d expects.
    """
    phi_axis = np.linspace(0, np.pi, nv)
    theta_axis = np.linspace(0, 2 * np.pi, nu)
    phi_grid, theta_grid = np.meshgrid(phi_axis, theta_axis, indexing="ij")

    x = (np.sin(phi_grid) * np.cos(theta_grid)).ravel()
    y = (np.sin(phi_grid) * np.sin(theta_grid)).ravel()
    z = np.cos(phi_grid).ravel()
    phi = phi_grid.ravel()
    theta = theta_grid.ravel()

    i_list, j_list, k_list = [], [], []
    for r in range(nv - 1):
        for c in range(nu - 1):
            v00 = r * nu + c
            v01 = r * nu + (c + 1)
            v10 = (r + 1) * nu + c
            v11 = (r + 1) * nu + (c + 1)
            # Two triangles per quad
            i_list += [v00, v01]
            j_list += [v01, v11]
            k_list += [v10, v10]
    return x, y, z, phi, theta, np.array(i_list), np.array(j_list), np.array(k_list)


def _sample_texture_rgb(texture_path, phi, theta):
    """Sample a real texture image at each (phi, theta) vertex.

    Row mapping matches sun_mpl.py's shade_texture / apply_shading for the
    matplotlib case: row = phi / pi * (H - 1).

    Column mapping includes a longitude-alignment shift: standard
    equirectangular textures start at -180 degrees at the left edge, but
    this sphere's theta=0 maps to the texture's centre column, not its
    left edge -- so the raw column index needs a +W/2 shift (wrapped) to
    line up correctly. This is the same fix already established and
    proven necessary for this exact kind of texture-sphere mapping
    elsewhere in this toolkit (see van_allen_plot_3d.py's
    _build_earth_mesh); without it, continents render offset/seamed
    rather than in their correct real-world positions.

    Returns
    -------
    ndarray, shape (N, 3), dtype float in [0, 1], or None if the texture
    can't be loaded (caller should fall back to a solid color gradient).
    """
    if texture_path is None:
        return None
    try:
        from PIL import Image
        img = Image.open(texture_path).convert("RGB")
        # Use the texture at its native resolution -- matching sun_mpl.py's
        # real, already-established texture-shading pattern (used
        # elsewhere in this toolkit), which doesn't discard resolution
        # unnecessarily. An earlier version of this function forced every
        # texture down to a tiny 256x128, matching orbit_plot's
        # _textured_sphere helper without reconsidering whether that
        # choice (made for a different rendering technology, matplotlib's
        # plot_surface) was appropriate here -- it wasn't, and produced
        # visibly low-fidelity results. Only cap unreasonably large source
        # images, to avoid excessive memory use.
        max_dim = 1024
        if img.width > max_dim or img.height > max_dim:
            scale = max_dim / max(img.width, img.height)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
        img_arr = np.asarray(img, dtype=float) / 255.0  # (H, W, 3)

        H, W = img_arr.shape[0], img_arr.shape[1]
        row = np.clip((phi / np.pi * (H - 1)).astype(int), 0, H - 1)
        col = ((theta / (2 * np.pi) * W).astype(int) + W // 2) % W
        return img_arr[row, col]
    except Exception:
        return None


def _lit_multiplier(x, y, z, sun_hat, ambient: float = 0.12):
    """Per-vertex brightness multiplier: `ambient` on the night side, up to
    1.0 on the sun-facing side. Same diffuse-lighting convention as
    sun_mpl.py's shade_texture (bright where facing the Sun, dim -- not
    pure black -- on the far side)."""
    sun_hat = np.asarray(sun_hat, dtype=float)
    sun_hat = sun_hat / np.linalg.norm(sun_hat)
    dot = x * sun_hat[0] + y * sun_hat[1] + z * sun_hat[2]
    return ambient + (1.0 - ambient) * np.clip(dot, 0.0, 1.0)


def _rgb_strings(rgb_float):
    """Convert an (N, 3) float-in-[0,1] array to a list of 'rgb(r,g,b)'
    strings, as required by go.Mesh3d's vertexcolor parameter."""
    rgb_255 = np.clip(rgb_float * 255.0, 0, 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb_255]


class EarthShadingLayer:
    """Fully-opaque Earth sphere with day/night terminator baked in as
    solid color.

    Renders the Earth body itself (radius = R_EARTH_KM) with per-vertex
    color computed from the angle to the Sun -- bright blue on the
    sun-facing side, fading to near-black on the night side. This IS the
    Earth trace; it does not stack on top of a separately-drawn sphere
    (see the module docstring for why a translucent-overlay approach
    doesn't work reliably in Plotly).

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Sun position in the scene frame (km). Only the direction matters.
    earth_center : array-like, shape (3,), optional
        Earth centre in scene coordinates. Default [0, 0, 0].
    radius_km : float, optional
        Sphere radius. Default R_EARTH_KM.
    nu, nv : int
        Sphere mesh resolution. Default 140 x 70 (higher than a plain
        color-gradient sphere needs, since per-vertex texture sampling
        means visual fidelity is bounded by vertex density, not just
        texture resolution -- a low-density mesh would still look blocky
        even with a high-resolution source texture).
    """

    def __init__(
        self,
        sun_pos_eci,
        earth_center=None,
        radius_km: float = R_EARTH_KM,
        nu: int = 140,
        nv: int = 70,
        time=None,
        rotation_deg=None,
    ):
        self.sun_hat = np.asarray(sun_pos_eci, dtype=float)
        if np.linalg.norm(self.sun_hat) > 0:
            self.sun_hat = self.sun_hat / np.linalg.norm(self.sun_hat)
        self.center = np.zeros(3) if earth_center is None else np.asarray(earth_center, dtype=float)
        self.radius = radius_km
        self.nu = nu
        self.nv = nv
        self.rotation_deg = (earth_rotation_deg_from_time(time)
                             if rotation_deg is None else float(rotation_deg))

    def build_traces(self) -> list:
        """Return a single, fully-opaque trace for the Earth: a real
        texture-mapped go.Mesh3d if earth.png is found, or a plain
        color-gradient go.Surface as a fallback."""
        texture_path = _find_texture("earth")

        if texture_path is not None:
            x, y, z, phi, theta, i, j, k = _uv_sphere_with_faces(self.nu, self.nv)
            rgb = _sample_texture_rgb(texture_path, phi, theta - np.radians(self.rotation_deg))
            if rgb is not None:
                lit = _lit_multiplier(x, y, z, self.sun_hat)
                shaded_rgb = (rgb * lit[:, None]).reshape(self.nv, self.nu, 3)
                shaded_rgb = stabilize_sphere_poles(shaded_rgb).reshape(-1, 3)
                return [go.Mesh3d(
                    x=self.center[0] + self.radius * x,
                    y=self.center[1] + self.radius * y,
                    z=self.center[2] + self.radius * z,
                    i=i, j=j, k=k,
                    vertexcolor=_rgb_strings(shaded_rgb),
                    opacity=1.0,
                    name="Earth",
                    legendgroup="Earth",
                    showlegend=True,
                    hoverinfo="skip",
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                                  roughness=1.0, fresnel=0.0),
                )]

        # Fallback: no real texture available -- plain color gradient
        # (still fully opaque, still a correct day/night terminator, just
        # not photographically textured).
        x, y, z = _unit_sphere_mesh(self.nu, self.nv)
        shadow = _shadow_surface_color(x, y, z, self.sun_hat, 1.0)
        return [go.Surface(
            x=self.center[0] + self.radius * x,
            y=self.center[1] + self.radius * y,
            z=self.center[2] + self.radius * z,
            surfacecolor=shadow,
            colorscale=_EARTH_DAYNIGHT_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=1.0,
            name="Earth",
            legendgroup="Earth",
            showlegend=True,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )]


# ---------------------------------------------------------------------------
# MoonShadingLayer
# ---------------------------------------------------------------------------

class MoonShadingLayer:
    """Fully-opaque Moon sphere with day/night terminator baked in as
    solid color (see EarthShadingLayer / module docstring for why this
    renders the whole body rather than a transparent overlay).

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Sun position in the scene frame (km).
    moon_center : array-like, shape (3,)
        Moon centre in scene frame (km). Required -- Moon is not at origin.
    radius_km : float, optional
        Sphere radius. Default R_MOON_KM.
    nu, nv : int
        Mesh resolution. Default 100 x 50 (see EarthShadingLayer's
        docstring for why -- per-vertex texture sampling needs a denser
        mesh to actually show the texture's real resolution).
    """

    def __init__(
        self,
        sun_pos_eci,
        moon_center,
        radius_km: float = R_MOON_KM,
        nu: int = 100,
        nv: int = 50,
    ):
        self.moon_center = np.asarray(moon_center, dtype=float)
        sun_from_moon = np.asarray(sun_pos_eci, dtype=float) - self.moon_center
        norm = np.linalg.norm(sun_from_moon)
        self.sun_hat = sun_from_moon / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self.radius = radius_km
        self.nu = nu
        self.nv = nv

    def build_traces(self) -> list:
        """Return a single, fully-opaque trace for the Moon: a real
        texture-mapped go.Mesh3d if moon.png is found, or a plain
        color-gradient go.Surface as a fallback."""
        texture_path = _find_texture("moon")

        if texture_path is not None:
            x, y, z, phi, theta, i, j, k = _uv_sphere_with_faces(self.nu, self.nv)
            rgb = _sample_texture_rgb(texture_path, phi, theta)
            if rgb is not None:
                lit = _lit_multiplier(x, y, z, self.sun_hat)
                shaded_rgb = (rgb * lit[:, None]).reshape(self.nv, self.nu, 3)
                shaded_rgb = stabilize_sphere_poles(shaded_rgb).reshape(-1, 3)
                return [go.Mesh3d(
                    x=self.moon_center[0] + self.radius * x,
                    y=self.moon_center[1] + self.radius * y,
                    z=self.moon_center[2] + self.radius * z,
                    i=i, j=j, k=k,
                    vertexcolor=_rgb_strings(shaded_rgb),
                    opacity=1.0,
                    name="Moon",
                    legendgroup="Moon",
                    showlegend=True,
                    hoverinfo="skip",
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                                  roughness=1.0, fresnel=0.0),
                )]

        # Fallback: no real texture available -- plain color gradient.
        x, y, z = _unit_sphere_mesh(self.nu, self.nv)
        shadow = _shadow_surface_color(x, y, z, self.sun_hat, 1.0)
        return [go.Surface(
            x=self.moon_center[0] + self.radius * x,
            y=self.moon_center[1] + self.radius * y,
            z=self.moon_center[2] + self.radius * z,
            surfacecolor=shadow,
            colorscale=_MOON_DAYNIGHT_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=1.0,
            name="Moon",
            legendgroup="Moon",
            showlegend=True,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )]


# ---------------------------------------------------------------------------
# Starfield
# ---------------------------------------------------------------------------
# Real HYG star catalog (bright_stars.csv) rendering -- this is the same
# catalog-loading and Scatter3d-building logic already proven working
# elsewhere in this toolkit's Plotly plots (van_allen_plot_3d.py), reused
# here rather than reimplemented. The matplotlib-only starfield.py can't be
# used directly since this module is Plotly-based.

_STAR_CACHE = None
_HYG_PATHS = [
    os.path.expanduser("~/bright_stars.csv"),
    os.path.expanduser("~/SSAPy/ssapy/data/bright_stars.csv"),
    os.path.join(os.path.dirname(__file__), "bright_stars.csv"),
]

_SPECT_COLORS = {
    "O": np.array([0.61, 0.69, 1.00]),
    "B": np.array([0.67, 0.75, 1.00]),
    "A": np.array([0.79, 0.85, 1.00]),
    "F": np.array([0.97, 0.97, 1.00]),
    "G": np.array([1.00, 0.96, 0.92]),
    "K": np.array([1.00, 0.82, 0.63]),
    "M": np.array([1.00, 0.80, 0.44]),
}


def _load_star_catalog(mag_limit: float = 6.5):
    """Load the real HYG star catalog (bright_stars.csv) if available.

    Returns a dict of unit-vector components plus magnitude/size/color
    arrays, or None if no catalog file is found -- callers fall back to a
    clearly-synthetic starfield rather than fabricating fake star positions.
    """
    global _STAR_CACHE
    if _STAR_CACHE is not None and _STAR_CACHE.get("mag_limit") == mag_limit:
        return _STAR_CACHE

    csv_path = None
    for p in _HYG_PATHS:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[(df["mag"] < mag_limit) & (df["mag"] > -10)].copy()
        df = df.dropna(subset=["ra", "dec", "mag"])
        # RA in the catalog is in hours, not degrees -- must multiply by 15
        # before converting to radians (a real bug this exact step has
        # caught before elsewhere in this toolkit's starfield code).
        ra_rad = np.radians(df["ra"].values * 15.0)
        dec_rad = np.radians(df["dec"].values)
        mag = df["mag"].values

        cx = np.cos(dec_rad) * np.cos(ra_rad)
        cy = np.cos(dec_rad) * np.sin(ra_rad)
        cz = np.sin(dec_rad)

        sizes = np.clip(0.8 * (mag_limit - mag) ** 1.2, 0.3, 5.0)
        spect = df["spect"].fillna("G").str[:1].values
        colors = np.array([_SPECT_COLORS.get(s, _SPECT_COLORS["G"]) for s in spect])

        _STAR_CACHE = dict(cx=cx, cy=cy, cz=cz, mag=mag, sizes=sizes,
                           colors=colors, mag_limit=mag_limit)
        return _STAR_CACHE
    except Exception:
        return None


def starfield_trace(sky_radius: float, mag_limit: float = 6.5) -> go.Scatter3d:
    """Return a single go.Scatter3d trace of background stars.

    Uses the real HYG catalog (bright_stars.csv) if found on disk (see
    _HYG_PATHS for search locations) -- real RA/dec positions, magnitude-
    based sizing, spectral-type coloring. Falls back to a synthetic,
    randomly-distributed starfield (clearly decorative, not real star
    positions) if no catalog file is available, rather than fabricating
    star data or crashing.

    Parameters
    ----------
    sky_radius : float
        Distance to place the star sphere at, in the same units (km) as
        the rest of the scene -- should be well beyond the Sun/planet
        placements so stars read as a distant backdrop.
    mag_limit : float
        Faintest apparent magnitude to include (lower = fewer, brighter
        stars only). Default 6.5, the traditional naked-eye limit.
    """
    stars = _load_star_catalog(mag_limit=mag_limit)

    if stars is None:
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(rng.uniform(-1, 1, n))
        return go.Scatter3d(
            x=sky_radius * np.sin(phi) * np.cos(theta),
            y=sky_radius * np.sin(phi) * np.sin(theta),
            z=sky_radius * np.cos(phi),
            mode="markers",
            marker=dict(size=1.5, color="white", opacity=0.6),
            hoverinfo="none", showlegend=False, name="Stars (synthetic)",
        )

    mag = stars["mag"]
    mag_min, mag_max = mag.min(), mag.max()
    depth = 0.85 + 0.15 * (mag - mag_min) / (mag_max - mag_min + 1e-6)
    x = stars["cx"] * sky_radius * depth
    y = stars["cy"] * sky_radius * depth
    z = stars["cz"] * sky_radius * depth
    color_strs = _rgb_strings(stars["colors"])

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=stars["sizes"] * 0.5, color=color_strs, opacity=0.85),
        hoverinfo="none", showlegend=False, name="Stars",
    )


# ---------------------------------------------------------------------------
# Convenience: auto-detect visual distance from scene scale
# ---------------------------------------------------------------------------

def auto_sun_position(sun_hat: np.ndarray, scene_radius_km: float) -> np.ndarray:
    """Place the visual sun at 12x the scene radius in the sun direction.

    Parameters
    ----------
    sun_hat : ndarray
        Unit vector toward the Sun.
    scene_radius_km : float
        Approximate radius of the plotted region (e.g. 8,000 for LEO,
        400,000 for cislunar).
    """
    dist = max(scene_radius_km * 12.0, VISUAL_DIST_KM_LEO)
    return sun_hat * dist


def auto_sun_radius(scene_radius_km: float) -> float:
    """Return an angular-correct display Sun radius for auto_sun_position."""
    dist = max(scene_radius_km * 12.0, VISUAL_DIST_KM_LEO)
    return max(dist * (R_SUN_KM / AU_KM), 1.0)
