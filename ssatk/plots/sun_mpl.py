"""
core/sun_mpl.py
----------------
Sun model for SSATK's matplotlib-based 3D plots
(globe_plot.py, moon_plot_3d.py, and orbit_plot.py).

Unlike the earlier Plotly-based core/sun.py, this module works directly
with the textured-sphere pattern already used throughout the toolkit
(`plot_surface(..., facecolors=img_arr[row, col])`), so it integrates
without replacing any existing rendering code.

Public API
----------
  get_sun_position(t)                       -> ndarray (N,3) meters, GCRF
  sun_direction_in_frame(t, transform_func)  -> unit vector in target frame
  shade_texture(img_arr, PHI, THETA, sun_hat, ambient, diffuse) -> (H,W) lighting array
  draw_sun(ax, sun_pos_local, radius, ...)   -> draws Sun sphere + corona

Typical usage inside moon_plot_3d.py / globe_plot.py
-----------------------------------------------------
    from ssatk.plots.sun_mpl import (get_sun_position, sun_direction_in_frame,
                                  shade_texture, draw_sun, auto_sun_distance)

    sun_hat = sun_direction_in_frame(t_current, gcrf_to_lunar_fixed)
    lit = shade_texture(img_arr, PHI, THETA, sun_hat)
    facecolors = np.clip(img_arr[row, col] * lit[row, col, None], 0, 1)
    ax.plot_surface(x, y, z, facecolors=facecolors, ...)

    sun_dist = auto_sun_distance(plot_range)
    draw_sun(ax, sun_hat * sun_dist, radius=plot_range * 0.06)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sun ephemeris
# ---------------------------------------------------------------------------

def get_sun_position(t):
    """Return Sun position in GCRF (meters) at time(s) t.

    Tries SSAPy's own body ephemeris first (consistent with how Moon/Earth
    positions are obtained elsewhere in the toolkit via ``get_body``).
    Falls back to a low-precision analytic solar position (Meeus,
    Astronomical Algorithms, accurate to ~0.01 deg) if SSAPy has no "sun"
    body registered.

    Parameters
    ----------
    t : astropy.time.Time, float (GPS seconds), or array of either.

    Returns
    -------
    ndarray, shape (..., 3)
        Sun position in GCRF, meters.
    """
    try:
        from ssapy import get_body
        # Keep the Body referenced: SSAPy closes its ephemeris when a temporary
        # Body is collected, so get_body("sun").position(t) raises TypeError and
        # this function silently fell back to the low-precision series.
        sun = get_body("sun")
        return sun.position(t).T
    except Exception:
        return _sun_position_fallback(t)


def _sun_position_fallback(t):
    """Low-precision solar ephemeris (Meeus), used only if get_body('sun')
    is unavailable. Returns GCRF position in meters."""
    from astropy.time import Time

    if isinstance(t, Time):
        jd = np.atleast_1d(t.tt.jd)
    else:
        # interpret as GPS seconds -> TT JD (consistent with globe_plot.py's
        # gst94 mapping: GPS seconds -> TT MJD)
        t_gps = np.atleast_1d(np.asarray(t, dtype=float))
        mjd_tt = 44244.0 + (t_gps + 51.184) / 86400.0
        jd = mjd_tt + 2400000.5

    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T) % 360.0
    M_deg = (357.52911 + 35999.05029 * T - 0.0001537 * T**2) % 360.0
    M = np.radians(M_deg)
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M)
         + (0.019993 - 0.000101 * T) * np.sin(2 * M)
         + 0.000289 * np.sin(3 * M))
    sun_lon = L0 + C
    e = 0.016708634 - 0.000042037 * T
    nu_deg = M_deg + C
    dist_au = 1.000001018 * (1 - e**2) / (1 + e * np.cos(np.radians(nu_deg)))

    omega = 125.04 - 1934.136 * T
    apparent_lon = sun_lon - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = (23.0 + 26.0/60 + 21.448/3600 - (46.8150/3600) * T)
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))

    lon_r = np.radians(apparent_lon)
    eps_r = np.radians(eps)

    AU_M = 1.495978707e11
    x = dist_au * AU_M * np.cos(lon_r)
    y = dist_au * AU_M * np.cos(eps_r) * np.sin(lon_r)
    z = dist_au * AU_M * np.sin(eps_r) * np.sin(lon_r)

    out = np.stack([x, y, z], axis=-1)
    return out if out.shape[0] > 1 else out[0]


def sun_direction_in_frame(t, transform_func=None):
    """Return the Sun's unit direction vector in the requested frame.

    Parameters
    ----------
    t : time argument forwarded to get_sun_position / transform_func.
        If t is an array/Time array (e.g. a full orbit time series), only
        the first epoch is used — the Sun's apparent direction changes by
        a negligible amount over a typical propagation window, and using
        a single epoch avoids shape mismatches in downstream frame
        transforms that expect position/time arrays of matching length.
    transform_func : callable or None
        A frame transform matching the toolkit's convention, e.g.
        ``gcrf_to_lunar_fixed`` or ``gcrf_to_itrf``. Called as
        ``transform_func(pos_gcrf, t)``. If None, returns the GCRF direction.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector toward the Sun in the target frame.
    """
    t_single = _first_epoch(t)

    sun_pos = get_sun_position(t_single)
    sun_pos = np.atleast_2d(sun_pos)

    if transform_func is not None:
        sun_pos = transform_func(sun_pos, t_single)

    sun_pos = np.atleast_2d(sun_pos)[0]
    norm = np.linalg.norm(sun_pos)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    return sun_pos / norm


def _first_epoch(t):
    """Reduce a scalar-or-array time argument to a single epoch, keeping
    it as a length-1 array/Time rather than a bare scalar.

    Frame transforms like gcrf_to_lunar_fixed build one rotation matrix
    per timestep and expect position/time arrays of matching length N —
    even N=1 must stay array-shaped, or the internal einsum over the
    rotation-matrix stack fails with a subscript-count mismatch.
    """
    from astropy.time import Time

    if isinstance(t, Time):
        if t.isscalar:
            return t.reshape(1)
        return t[0:1]

    arr = np.atleast_1d(np.asarray(t, dtype=float))
    return arr[0:1]


# ---------------------------------------------------------------------------
# Texture shading
# ---------------------------------------------------------------------------

def shade_texture(PHI, THETA, sun_hat, ambient: float = 0.30, diffuse: float = 0.70):
    """Compute a per-vertex lighting factor for a textured sphere.

    Designed to match the ``(PHI, THETA)`` meshgrid convention used by
    ``_textured_moon`` / ``_textured_earth`` in ``moon_plot_3d.py``::

        x = r * sin(PHI) * cos(THETA)
        y = r * sin(PHI) * sin(THETA)
        z = r * cos(PHI)

    Parameters
    ----------
    PHI, THETA : ndarray
        Meshgrid angles, same shape as the sphere mesh.
    sun_hat : array-like, shape (3,)
        Unit vector toward the Sun, in the SAME frame as the sphere
        (i.e. body-centred, after any frame transform has been applied).
    ambient : float
        Minimum brightness on the night side (0-1). Default 0.30 so the
        unlit hemisphere is dim but the texture is still slightly visible.
    diffuse : float
        Additional brightness added on the sun-facing side (0-1).
        ambient + diffuse should be <= 1.0 for a normal exposure.

    Returns
    -------
    ndarray, same shape as PHI
        Lighting multiplier in [ambient, ambient+diffuse].
    """
    sun_hat = np.asarray(sun_hat, dtype=float)
    sun_hat = sun_hat / np.linalg.norm(sun_hat)

    nx = np.sin(PHI) * np.cos(THETA)
    ny = np.sin(PHI) * np.sin(THETA)
    nz = np.cos(PHI)

    dot = nx * sun_hat[0] + ny * sun_hat[1] + nz * sun_hat[2]
    lit = ambient + diffuse * np.clip(dot, 0.0, 1.0)
    return lit


def apply_shading(img_arr, row, col, PHI, THETA, sun_hat,
                   ambient: float = 0.30, diffuse: float = 0.70):
    """Convenience wrapper: returns ready-to-use facecolors for plot_surface.

    Equivalent to:
        lit = shade_texture(PHI, THETA, sun_hat, ambient, diffuse)
        facecolors = np.clip(img_arr[row, col] * lit[..., None], 0, 1)
    """
    lit = shade_texture(PHI, THETA, sun_hat, ambient, diffuse)
    base = img_arr[row, col]
    return np.clip(base * lit[..., None], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Drawing the Sun itself
# ---------------------------------------------------------------------------

def auto_sun_distance(plot_range: float) -> float:
    """Distance to place the visual Sun: well inside the axis limits but
    clearly closer than the starfield (which sits far outside plot_range).

    Using ~0.85x the half-extent of the typical view keeps the Sun sphere
    visible near the edge of frame without being clipped, while still
    reading as much closer than the background stars.
    """
    return plot_range * 0.42


def auto_sun_radius(plot_range: float) -> float:
    """Display radius for the Sun sphere, proportional to scene scale."""
    return max(plot_range * 0.045, 1.0)


def draw_sun(ax, sun_pos_local, radius: float, n: int = 48,
             core_color: str = "#ffe9a0",
             corona_colors=(("#ffcc33", 0.35), ("#ff9d2e", 0.22), ("#ff7a1a", 0.12), ("#ff5500", 0.06))):
    """Draw a glowing Sun sphere (core + corona rings) on a matplotlib 3D axis.

    Parameters
    ----------
    ax : Axes3D
    sun_pos_local : array-like, shape (3,)
        Sun centre position in the same coordinate frame/units as the rest
        of the scene (e.g. km, in lunar-fixed or GCRF-equivalent frame).
    radius : float
        Core sphere radius (same units as sun_pos_local).
    n : int
        Mesh resolution.
    core_color : str
        Matplotlib color for the bright core. Default near-white for a
        blazing/overexposed look rather than a flat yellow disk.
    corona_colors : tuple of (color, alpha)
        Concentric glow shells, outer to inner. Spread wider and with
        higher inner-ring opacity than a subtle halo so the Sun reads
        as radiant rather than as a flat colored coin.
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    bx = np.cos(uu) * np.sin(vv)
    by = np.sin(uu) * np.sin(vv)
    bz = np.cos(vv)

    cx, cy, cz = sun_pos_local

    # Corona shells first (so the bright core draws on top), widest/faintest
    # to innermost/brightest — gives a glow gradient rather than flat rings.
    scales = [4.2, 2.8, 1.9, 1.35]
    for (color, alpha), scale in zip(corona_colors, scales):
        r = radius * scale
        ax.plot_surface(
            cx + r * bx, cy + r * by, cz + r * bz,
            color=color, alpha=alpha, shade=False, edgecolor="none",
            linewidth=0, antialiased=True,
        )

    # Bright core — manual per-vertex shading (rather than relying on
    # matplotlib's built-in shade=True, which blends quite subtly) so the
    # sphere's curvature is unmistakable: a clearly bright "near" side and
    # a dimmer "far" side, not just a faint highlight.
    light_dir = np.array([0.55, -0.45, 0.70])
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.clip(bx * light_dir[0] + by * light_dir[1] + bz * light_dir[2], 0.0, 1.0)
    intensity = intensity ** 0.6   # gamma: brightens mid-tones, keeps contrast

    from matplotlib.colors import to_rgb
    base_rgb = np.array(to_rgb(core_color))
    dim_rgb  = base_rgb * 0.35      # far side: same hue, much darker

    facecolors = (dim_rgb[None, None, :]
                  + intensity[..., None] * (base_rgb - dim_rgb)[None, None, :])
    facecolors = np.clip(facecolors, 0.0, 1.0)

    ax.plot_surface(
        cx + radius * bx, cy + radius * by, cz + radius * bz,
        facecolors=facecolors, alpha=1.0, shade=False,
        edgecolor="none", linewidth=0, antialiased=True,
        rstride=1, cstride=1,
    )

    ax.text(cx, cy, cz + radius * 4.6, "Sun", color="#ffe9a8",
            fontsize=9, ha="center")
