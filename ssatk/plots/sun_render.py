"""
core/sun_render.py
-------------------
A dedicated, physically-styled 3D Sun renderer for SSATK's
matplotlib plots, plus the light-projection step that ties the Sun's
real position to shading on other bodies (Moon, Earth).

Why a separate file from core/sun_mpl.py
-----------------------------------------
core/sun_mpl.py owns the Sun's *ephemeris* (where is it, real position via
get_body("sun")) and the *texture shading* applied to reflective bodies
like the Moon/Earth (diffuse lighting: bright where facing the Sun, dark
where not).

This module owns the *Sun's own appearance*. The Sun is self-luminous, so
it should NOT be shaded the way a reflective body is — there's no external
light source illuminating one side of it. What real photos of the Sun show
is limb darkening: it looks brighter at the center of its disk and dims
toward the edge, an effect that depends on the VIEWING angle (camera), not
on a light-source direction. Mixing that logic into sun_mpl.py would be
confusing, so it lives here instead.

Public API
----------
  render_sun(ax, position, radius, elev_deg, azim_deg, ...) -> draws the Sun
  light_direction_from_positions(sun_pos, target_pos)        -> unit vector
  project_light_onto_body(...)                               -> facecolors

Typical usage inside moon_plot_3d.py
-------------------------------------
    from ssatk.plots.sun_render import render_sun, light_direction_from_positions

    sun_pos_local = sun_hat * sun_dist      # Sun's position in this scene
    render_sun(ax, sun_pos_local, radius=sun_radius, elev_deg=elev, azim_deg=azim)

    # Light direction for shading the Moon (Moon sits at origin here):
    light_dir = light_direction_from_positions(sun_pos_local, np.zeros(3))
    facecolors = apply_shading(img_arr, row, col, PHI, THETA, light_dir, ...)
    # (apply_shading itself still lives in core/sun_mpl.py — this module
    #  only computes the geometry of "where is the light coming from".)
"""

from __future__ import annotations

import numpy as np

from ssatk.constants import SUN_EARTH_AVERAGE_DISTANCE_KM, SUN_RADIUS_KM


# ---------------------------------------------------------------------------
# Light projection — ties the Sun's real position to shading on other bodies
# ---------------------------------------------------------------------------

def light_direction_from_positions(sun_pos, target_pos):
    """Return the unit vector from a target body toward the Sun.

    This is the actual "project the Sun's light onto the body" step: given
    where the Sun is and where the body (e.g. Moon) is in the same scene
    coordinates, compute the direction light travels from Sun to body, then
    flip it — what callers want is "which way is the Sun, as seen from the
    surface of this body" so it can be dotted against surface normals.

    Parameters
    ----------
    sun_pos : array-like, shape (3,)
        Sun's position in the scene's coordinate frame.
    target_pos : array-like, shape (3,)
        The illuminated body's center position, same frame/units.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector pointing from target_pos toward sun_pos.
    """
    sun_pos = np.asarray(sun_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    vec = sun_pos - target_pos
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def _view_direction(elev_deg, azim_deg):
    """Unit vector pointing from the scene origin toward the camera,
    matching matplotlib's (elev, azim) convention used elsewhere in the
    toolkit (see globe_plot.py's _view_unit_vector)."""
    el = np.radians(float(elev_deg))
    az = np.radians(float(azim_deg))
    v = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# The Sun's own rendering — limb darkening, not external-light shading
# ---------------------------------------------------------------------------

def render_sun(
    ax,
    position,
    radius: float,
    elev_deg: float = 30.0,
    azim_deg: float = 45.0,
    n: int = 64,
    limb_darkening: float = 0.7,
    core_color=(1.00, 0.95, 0.82),
    limb_color=(1.00, 0.50, 0.10),
    corona_max_scale: float = 1.8,
    corona_layers_n: int = 16,
    corona_peak_alpha: float = 0.22,
    corona_color=(1.0, 0.55, 0.15),
    label: str = "Sun",
):
    """Render a physically-styled, limb-darkened 3D Sun on a matplotlib
    3D axis, with a smooth (not ringed) corona glow.

    Limb darkening is computed from the angle between each point's surface
    normal and the direction toward the CAMERA (elev_deg, azim_deg) — this
    is what makes a self-luminous sphere look spherical: brighter where
    you're looking straight at the surface (disk center), dimmer near the
    edge, the same from whichever side you view it from. This is distinct
    from diffuse shading on a reflective body (Moon/Earth), which depends
    on the light SOURCE direction instead — see project_light_onto_body
    and core/sun_mpl.py's apply_shading for that.

    The limb-darkening direction is deliberately biased ~25-30° off the
    exact camera axis (see the bias_dir blend below). Pure camera-aligned
    limb darkening is physically correct but renders as a perfectly
    centered, symmetric "bullseye" gradient when viewed face-on — flat to
    the eye, with no depth cue. An off-axis highlight (like a glossy
    sphere's specular highlight sitting away from dead-center) is what
    actually reads as a sphere rather than a flat gradient circle.

    Parameters
    ----------
    ax : Axes3D
    position : array-like, shape (3,)
        Sun's center position in the scene's coordinate frame.
    radius : float
        Core sphere radius (same units as position).
    elev_deg, azim_deg : float
        Current camera angle — pass the same values used for
        ax.view_init(...) elsewhere in the plot.
    n : int
        Mesh resolution.
    limb_darkening : float, 0-1
        Strength of the center-to-edge falloff on the core disk itself.
    core_color, limb_color : RGB tuples (0-1 each channel)
        Colors at disk center vs. the limb (edge).
    corona_max_scale : float
        Outermost corona radius as a multiple of the core radius. Kept
        modest (~1.8x) so the CORE — not the glow — is the dominant
        visual feature; a corona that dwarfs the core is what made the
        Sun look like flat concentric rings rather than a glowing ball.
    corona_layers_n : int
        Number of corona shells. Many thin layers (vs. a few thick ones)
        blend into a smooth gradient instead of visible discrete rings.
    corona_peak_alpha : float
        Opacity of the innermost corona layer (alpha decays smoothly
        outward from here).
    corona_color : RGB tuple
        Single glow color — using one consistent color across all
        layers (rather than several different hues) also helps the
        glow blend smoothly instead of reading as banded rings.
    label : str
        Text label drawn above the Sun. Pass "" to omit.

    Returns
    -------
    None
    """
    position = np.asarray(position, dtype=float)
    cx, cy, cz = position

    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    bx = np.cos(uu) * np.sin(vv)
    by = np.sin(uu) * np.sin(vv)
    bz = np.cos(vv)

    # ---- Smooth corona: many thin layers with exponentially decaying
    #      alpha, so it blends into a soft glow instead of visible rings.
    #      Drawn first so the core renders on top. ----
    for i in range(corona_layers_n, 0, -1):
        frac = i / corona_layers_n               # 1.0 (outermost) -> ~0 (innermost)
        scale = 1.0 + (corona_max_scale - 1.0) * frac
        # Exponential decay reads as smoother than linear — outer layers
        # fade much faster than inner ones, avoiding a visible hard edge.
        alpha = corona_peak_alpha * (1.0 - frac) ** 2
        if alpha < 0.003:
            continue
        r = radius * scale
        ax.plot_surface(
            cx + r * bx, cy + r * by, cz + r * bz,
            color=corona_color, alpha=alpha, shade=False, edgecolor="none",
            linewidth=0, antialiased=True,
        )

    # ---- Core: limb-darkened disk, biased off-axis for a 3D look ----
    # Pure camera-aligned limb darkening (mu = dot(normal, view_dir)) is
    # physically correct but renders as a perfectly centered, symmetric
    # "bullseye" gradient when viewed face-on — it reads as flat rather
    # than spherical, since there's no asymmetric highlight for the eye
    # to use as a depth cue. Blending in a fixed off-axis bias direction
    # restores that asymmetric highlight (similar to how a glossy sphere's
    # specular highlight sits offset from dead-center) while still mostly
    # following the real limb-darkening falloff.
    view_dir = _view_direction(elev_deg, azim_deg)
    bias_dir = np.array([0.55, -0.45, 0.70])
    bias_dir = bias_dir / np.linalg.norm(bias_dir)
    effective_dir = 0.5 * view_dir + 0.5 * bias_dir
    effective_dir = effective_dir / np.linalg.norm(effective_dir)

    mu = np.clip(bx * effective_dir[0] + by * effective_dir[1] + bz * effective_dir[2],
                 0.0, 1.0)

    # Linear limb-darkening law: I(mu) = 1 - limb_darkening * (1 - mu)
    intensity = 1.0 - limb_darkening * (1.0 - mu)
    intensity = np.clip(intensity, 0.0, 1.0)

    core_rgb = np.array(core_color)
    limb_rgb = np.array(limb_color)
    facecolors = (limb_rgb[None, None, :]
                  + intensity[..., None] * (core_rgb - limb_rgb)[None, None, :])
    facecolors = np.clip(facecolors, 0.0, 1.0)

    ax.plot_surface(
        cx + radius * bx, cy + radius * by, cz + radius * bz,
        facecolors=facecolors, alpha=1.0, shade=False,
        edgecolor="none", linewidth=0, antialiased=True,
        rstride=1, cstride=1,
    )

    if label:
        ax.text(cx, cy, cz + radius * (corona_max_scale * 1.1),
                label, color="#ffe9a8", fontsize=9, ha="center")


# ---------------------------------------------------------------------------
# Placement helper — keeps the Sun a true background object
# ---------------------------------------------------------------------------

def background_sun_position(sun_hat, plot_range: float, distance_factor: float = 2.5):
    """Place the Sun well behind the main scene (a true background object)
    rather than near the frame edge, while staying closer than the
    starfield. Combine with a small `radius` (see background_sun_radius)
    so it reads as distant rather than nearby.

    Because callers typically disable axis clipping (ax.set_clip_on(False)),
    placing the Sun beyond the axis limits is fine — it still renders in
    the scene, just appropriately small and far away.
    """
    sun_hat = np.asarray(sun_hat, dtype=float)
    sun_hat = sun_hat / np.linalg.norm(sun_hat)
    return sun_hat * (plot_range * distance_factor)


def background_sun_radius(
    plot_range: float,
    size_factor: float | None = None,
    distance_factor: float = 2.5,
) -> float:
    """Display radius for a background Sun.

    By default the radius preserves the Sun's real angular radius at the
    background display distance. Pass ``size_factor`` for deliberately
    stylized scene-fraction sizing.
    """
    if size_factor is not None:
        return max(plot_range * size_factor, 1.0)
    angular_ratio = SUN_RADIUS_KM / SUN_EARTH_AVERAGE_DISTANCE_KM
    return max(plot_range * distance_factor * angular_ratio, 1.0)
