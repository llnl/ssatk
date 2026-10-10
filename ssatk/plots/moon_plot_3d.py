"""
moon_plot_3d.py — Standalone 3D Moon surface plot with star background
======================================================================
Renders the Moon as a textured 3D sphere with accurate star field,
Lagrange points, optional satellite orbit tracks, AND a real Sun model
that shades the Moon/Earth textures based on true solar ephemeris.

Drop into:
  ~/ssatk/ssatk/plots/moon_plot_3d.py

Usage:
    from ssatk.plots.moon_plot_3d import moon_plot_3d
    fig, ax = moon_plot_3d(r=r, t=t, save_path="moon_orbit.jpg")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from .starfield import add_starfield
from .plotutils import valid_orbits, figsave
from ..constants import MOON_RADIUS, EARTH_RADIUS
from ..coordinates import gcrf_to_lunar_fixed

# Real Sun ephemeris + texture shading (diffuse lighting on the Moon/Earth)
from .sun_mpl import (
    sun_direction_in_frame,
    apply_shading,
)
# Dedicated Sun renderer (limb-darkened, camera-angle-based) + light
# projection helper that ties the Sun's position to the shading above
from .sun_render import (
    render_sun,
    background_sun_radius,
)


def _split_orbit_segments_for_bodies(xyz, bodies, elev_deg, azim_deg):
    """Split line segments into camera-occluded and foreground groups."""
    segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
    midpoints = segments.mean(axis=1)
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    camera = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    hidden = np.zeros(len(midpoints), dtype=bool)
    for center, radius in bodies:
        relative = midpoints - center
        depth = relative @ camera
        perpendicular = relative - depth[:, None] * camera
        hidden |= (depth < 0.0) & (np.einsum("ij,ij->i", perpendicular, perpendicular) < radius**2)
    return segments[hidden], segments[~hidden]


def _ssapy_texture(name):
    """Return a base SSAPy texture path, or None if unavailable."""
    try:
        from ssapy.utils import find_file
        return find_file(name, ext=".png")
    except Exception:
        return None



def _textured_moon(ax, cx, cy, cz, radius, moon_img_path, n=64, sun_hat=None,
                    ambient=0.22, diffuse=0.78, zorder=10):
    """Render Moon as a textured 3D sphere, optionally shaded by sun_hat.

    sun_hat : array-like (3,), unit vector toward the Sun in the same
        local frame as (cx, cy, cz) — i.e. lunar-fixed frame, centred on
        the Moon. If None, texture is rendered unshaded (original behavior).
    """
    try:
        img     = Image.open(moon_img_path).convert("RGB")
        img     = img.resize((512, 256), Image.LANCZOS)
        img_arr = np.array(img) / 255.0

        phi   = np.linspace(0,    np.pi,  n)
        theta = np.linspace(0, 2*np.pi, 2*n)
        PHI, THETA = np.meshgrid(phi, theta)

        x = cx + radius * np.sin(PHI) * np.cos(THETA)
        y = cy + radius * np.sin(PHI) * np.sin(THETA)
        z = cz + radius * np.cos(PHI)

        row = np.clip((PHI   / np.pi    * (img_arr.shape[0]-1)).astype(int),
                      0, img_arr.shape[0]-1)
        col = np.clip((THETA / (2*np.pi) * (img_arr.shape[1]-1)).astype(int),
                      0, img_arr.shape[1]-1)

        if sun_hat is not None:
            facecolors = apply_shading(img_arr, row, col, PHI, THETA, sun_hat,
                                        ambient=ambient, diffuse=diffuse)
        else:
            facecolors = img_arr[row, col]

        ax.plot_surface(x, y, z, facecolors=facecolors,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, zorder=zorder,
                        shade=False)
        return True
    except Exception as e:
        print(f"[moon_plot_3d] Moon texture failed: {e}")
        return False


def _textured_earth(ax, cx, cy, cz, radius, earth_img_path, n=128, sun_hat=None,
                     ambient=0.22, diffuse=0.78, zorder=10):
    """Render Earth as a small textured sphere, optionally shaded by sun_hat.

    sun_hat : array-like (3,), unit vector toward the Sun in the same
        local frame as (cx, cy, cz) — i.e. lunar-fixed frame, centred on
        Earth's plotted position. If None, texture is unshaded.
    """
    try:
        img     = Image.open(earth_img_path).convert("RGB")
        img     = img.resize((1024, 512), Image.LANCZOS)
        img_arr = np.array(img) / 255.0

        phi   = np.linspace(0,    np.pi,  n)
        theta = np.linspace(0, 2*np.pi, 2*n)
        PHI, THETA = np.meshgrid(phi, theta)

        x = cx + radius * np.sin(PHI) * np.cos(THETA)
        y = cy + radius * np.sin(PHI) * np.sin(THETA)
        z = cz + radius * np.cos(PHI)

        row = np.clip((PHI   / np.pi    * (img_arr.shape[0]-1)).astype(int),
                      0, img_arr.shape[0]-1)
        col = np.clip((THETA / (2*np.pi) * (img_arr.shape[1]-1)).astype(int),
                      0, img_arr.shape[1]-1)

        if sun_hat is not None:
            facecolors = apply_shading(img_arr, row, col, PHI, THETA, sun_hat,
                                        ambient=ambient, diffuse=diffuse)
        else:
            facecolors = img_arr[row, col]

        ax.plot_surface(x, y, z, facecolors=facecolors,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, zorder=zorder,
                        shade=False)
        return True
    except Exception as e:
        print(f"[moon_plot_3d] Earth texture failed: {e}")
        return False


def moon_plot_3d(r=None, t=None, title='', figsize=(10, 10),
                 save_path=False, show=False,
                 elev=30, azim=45,
                 show_lagrange=True,
                 show_stars=True,
                 show_sun=True,
                 show_sun_graphic=None,
                 show_earth=None,
                 mag_limit=5.5,
                 sun_azimuth_deg=None,
                 sun_elevation_deg=None,
                 shade_ambient=0.22,
                 shade_diffuse=0.78,
                 scene_pad=0.30,
                 view_zoom=1.0,
                 physical_body_sizes=False,
                 r_frame='gcrf'):
    """
    3D Moon surface plot with star background, real Sun shading, and
    optional orbit tracks.

    Parameters
    ----------
    r : ssapy.Orbit, numpy array, or list, optional
        SSAPy ``Orbit``/``Orbit.at`` object or satellite position vectors in
        GCRF [m].  Raw ``ssapy.rv`` position output can be passed directly.
    t : numpy array, astropy Time, or list, optional
        Times corresponding to r [GPS seconds] or Astropy Time samples.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str or False
        Path to save figure.
    show : bool
        Display figure interactively.
    elev : float
        Camera elevation [degrees].
    azim : float
        Camera azimuth [degrees].
    show_lagrange : bool
        Show Earth-Moon Lagrange points.
    show_stars : bool
        Show star background.
    show_earth : bool or None, default None
        Whether to draw Earth. None (default) keeps the original
        auto-by-zoom-level behavior: Earth is only drawn when the scene's
        plot_range exceeds 100,000 km, since at tighter zoom levels
        (e.g. a close-up on a small portion of cislunar space) Earth
        would normally fall outside any reasonable view anyway. Pass
        True to force Earth to render regardless of zoom level (e.g. for
        a close-up that should still show Earth for context), or False
        to force it off.
    show_sun : bool
        Compute real solar ephemeris, shade Moon/Earth textures by true
        illumination.
    show_sun_graphic : bool or None, default None
        Draw the Sun sphere. ``None`` preserves the historical behavior and
        follows ``show_sun``; set ``False`` to retain solar illumination
        without a visible Sun graphic.
    mag_limit : float
        Star magnitude limit.
    sun_azimuth_deg : float, optional
        Manually override the Sun's azimuth (degrees, measured from the
        +x axis toward +y in the lunar-fixed xy-plane), bypassing real
        ephemeris. Use together with sun_elevation_deg to pose the Sun
        at any angle — e.g. to illustrate a quarter-phase terminator
        for a demo, independent of the true current solar position.
        If only one of sun_azimuth_deg / sun_elevation_deg is given, the
        other defaults to 0.
    sun_elevation_deg : float, optional
        Manually override the Sun's elevation (degrees above the
        lunar-fixed xy-plane). See sun_azimuth_deg.
    shade_ambient : float, default 0.22
        Minimum brightness on the Moon/Earth night side (0-1). Raise this
        to make the unlit hemisphere less dark; lower it for a starker
        terminator.
    shade_diffuse : float, default 0.78
        Additional brightness added on the sun-facing side (0-1).
        shade_ambient + shade_diffuse should not exceed 1.0. Together
        these two act as the "shading percentage" control — e.g.
        ambient=0.5, diffuse=0.5 gives a soft, low-contrast terminator;
        ambient=0.05, diffuse=0.95 gives a harsh, high-contrast one.
    scene_pad : float, default 0.30
        Fractional padding around the trajectory bounds. Smaller values give
        a tighter orbit framing.
    view_zoom : float, default 1.0
        Camera zoom applied after framing; values above one fill more of the
        figure canvas.
    physical_body_sizes : bool, default False
        Draw Earth and Moon at physical radii rather than applying the
        visibility enlargement used for wide cislunar scenes.
    r_frame : str, default 'gcrf'
        Frame that `r` is already expressed in.
          'gcrf'          : r is Earth-centered GCRF/equatorial (the
                            normal case for orbits propagated by
                            OrbitalState). Transformed into the Moon-fixed
                            frame internally via gcrf_to_lunar_fixed.
          'moon_centered' : r is ALREADY Moon-centered (e.g. real JPL
                            Horizons data queried with CENTER='coord@301'),
                            in an inertial equatorial-aligned frame. Used
                            as-is, with NO additional frame transform —
                            passing Moon-centered data through with
                            r_frame='gcrf' would incorrectly re-center it
                            a second time.

    Returns
    -------
    fig, ax
    """
    from ..orbital_mechanics import lagrange_points_lunar_fixed_frame

    if show_sun_graphic is None:
        show_sun_graphic = show_sun

    moon_img = _ssapy_texture("moon")
    earth_img = _ssapy_texture("earth")

    plotcolor = '#000008'
    textcolor = 'white'

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(dpi=120, figsize=figsize, facecolor=plotcolor)
    ax  = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor(plotcolor)
    fig.patch.set_facecolor(plotcolor)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.xaxis.pane.set_facecolor(plotcolor)
    ax.yaxis.pane.set_facecolor(plotcolor)
    ax.zaxis.pane.set_facecolor(plotcolor)
    ax.set_axis_off()
    ax.set_clip_on(False)
    ax.computed_zorder = False

    # ── Unit conversion ───────────────────────────────────────────────────────
    unit_conversion = 1e3   # km
    unit_label      = 'km'
    moon_r          = MOON_RADIUS / unit_conversion

    # ── Orbit tracks in lunar fixed frame ────────────────────────────────────
    plot_range = moon_r * 10   # default range around Moon
    xyz_list   = []

    # Reference epoch for Sun direction (use first orbit time if available,
    # else "now" via the orbit time array; falls back to t=0 if nothing given)
    t_ref = None

    if r is not None:
        r_list, t_list = valid_orbits(r, t)

        raw_lower = np.array([ np.inf,  np.inf,  np.inf])
        raw_upper = np.array([-np.inf, -np.inf, -np.inf])
        _all_t_for_midpoint = []

        for orbit_index in range(len(r_list)):
            xyz       = r_list[orbit_index] / unit_conversion
            t_current = t_list[orbit_index]
            if t_ref is None:
                t_ref = t_current
            _all_t_for_midpoint.append(np.atleast_1d(t_current))

            # Transform to lunar fixed frame (only if r is GCRF/Earth-
            # centered to begin with — already-Moon-centered data, e.g.
            # real Horizons ephemeris queried with CENTER='coord@301',
            # is used directly with no further transform).
            if r_frame == 'moon_centered':
                xyz_lunar = r_list[orbit_index] / unit_conversion
            else:
                xyz_lunar = gcrf_to_lunar_fixed(
                    r_list[orbit_index], t_current) / unit_conversion

            raw_lower = np.minimum(raw_lower, xyz_lunar.min(axis=0))
            raw_upper = np.maximum(raw_upper, xyz_lunar.max(axis=0))
            xyz_list.append((orbit_index, xyz_lunar.copy()))

        # Use the MIDPOINT of the full displayed time range for Sun
        # direction, not just the first instant. A single static Sun
        # snapshot is necessarily an approximation for any multi-day
        # trajectory plotted in the rotating lunar-fixed frame (the
        # apparent Sun direction in that frame sweeps through a real arc
        # as the Moon orbits Earth — about 13 deg/day). Freezing on the
        # very first timestamp made that approximation badly wrong for
        # anything spanning more than ~1 day; the midpoint at least
        # centers the error across the displayed range instead of
        # concentrating it entirely at one end.
        #
        # t_current may be either an astropy.time.Time array or a plain
        # float array of GPS seconds, depending on caller convention.
        # Time + Time isn't a valid operation (only Time + duration is),
        # so convert to plain GPS-second floats first, do the midpoint
        # arithmetic in that numeric space, then hand back a plain float
        # array — downstream code (sun_direction_in_frame's epoch
        # reduction) already accepts plain floats just as well as Time.
        if _all_t_for_midpoint:
            try:
                from astropy.time import Time as _Time
            except ImportError:
                _Time = None
            _numeric_arrays = []
            for _t_arr in _all_t_for_midpoint:
                if _Time is not None and isinstance(_t_arr, _Time):
                    _numeric_arrays.append(np.atleast_1d(_t_arr.gps))
                else:
                    _numeric_arrays.append(np.atleast_1d(np.asarray(_t_arr, dtype=float)))
            _t_concat = np.concatenate(_numeric_arrays)
            t_ref = np.array([(_t_concat.min() + _t_concat.max()) / 2.0])

        plot_range = max(
            raw_upper[0] - raw_lower[0],
            raw_upper[1] - raw_lower[1],
            raw_upper[2] - raw_lower[2],
            moon_r * 6,
        )

        # Set axis limits
        cx_ = (raw_lower[0] + raw_upper[0]) / 2
        cy_ = (raw_lower[1] + raw_upper[1]) / 2
        cz_ = (raw_lower[2] + raw_upper[2]) / 2
        pad = plot_range * scene_pad
        ax.set_xlim(cx_ - plot_range/2 - pad, cx_ + plot_range/2 + pad)
        ax.set_ylim(cy_ - plot_range/2 - pad, cy_ + plot_range/2 + pad)
        ax.set_zlim(cz_ - plot_range/2 - pad, cz_ + plot_range/2 + pad)

        # IMPORTANT: plot_range must reflect the FULL rendered span the
        # axes actually show (including pad), not just the orbit's raw
        # bounding-box diagonal. Everything placed below (Sun distance,
        # Moon's minimum visible size, starfield) is scaled from
        # plot_range — if it under-counts the pad, those placements end
        # up smaller/closer than what the axes actually display, and
        # matplotlib's mplot3d has no true depth perspective to fall back
        # on: anything placed outside the axis limits doesn't render
        # "farther away," it just risks rendering outside the canvas
        # entirely (as seen when the Sun's label got clipped at the
        # figure edge).
        plot_range = plot_range + 2 * pad

    else:
        # No orbit — show Moon with some surrounding space.
        # IMPORTANT: plot_range must match these axis limits (full span,
        # not half-span) or the Sun/starfield placement below — which is
        # scaled from plot_range — will be inconsistent with what the
        # camera actually sees, making the Sun appear far closer than
        # intended relative to the visible frame.
        half_span = moon_r * 3
        ax.set_xlim(-half_span, half_span)
        ax.set_ylim(-half_span, half_span)
        ax.set_zlim(-half_span, half_span)
        plot_range = half_span * 2

    # Equal aspect ratio on all three axes — without this, matplotlib
    # scales each axis independently and every sphere (Moon, Earth, Sun)
    # renders as a squashed ellipsoid instead of a circle.
    ax.set_box_aspect([1, 1, 1], zoom=view_zoom)

    # ── Sun direction ──────────────────────────────────────────────────────
    # By default this comes from real ephemeris (get_body("sun") transformed
    # into the lunar-fixed frame). If sun_azimuth_deg / sun_elevation_deg are
    # given, those override it entirely — useful for posing a specific
    # phase/terminator for a demo regardless of the true current Sun
    # position. Azimuth is measured from +x toward +y in the lunar-fixed
    # xy-plane; elevation is measured up from that plane toward +z.
    sun_hat = None
    if sun_azimuth_deg is not None or sun_elevation_deg is not None:
        az = np.radians(sun_azimuth_deg if sun_azimuth_deg is not None else 0.0)
        el = np.radians(sun_elevation_deg if sun_elevation_deg is not None else 0.0)
        sun_hat = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
    elif show_sun or show_sun_graphic:
        try:
            if t_ref is None:
                # No orbit time given — default to "now"
                from astropy.time import Time
                t_ref = Time.now()
            if r_frame == 'moon_centered':
                # r is already Moon-centered, INERTIAL, equatorial-aligned
                # (e.g. real Horizons data) — gcrf_to_lunar_fixed outputs a
                # frame that ROTATES with the Moon's orbital motion, which
                # would put the Sun direction in a different, inconsistent
                # frame from the trajectory. Use the raw equatorial
                # direction instead (origin shift Earth->Moon is
                # negligible at solar distance, same approximation already
                # used for Earth's shading elsewhere in this file).
                sun_hat = sun_direction_in_frame(t_ref, transform_func=None)
            else:
                sun_hat = sun_direction_in_frame(t_ref, gcrf_to_lunar_fixed)
        except Exception as e:
            print(f"[moon_plot_3d] Sun ephemeris failed, rendering unshaded: {e}")
            sun_hat = None

    # NOTE: the camera (elev, azim) is NOT auto-rotated to face the Sun.
    # If it were, the entire visible hemisphere would always be the day
    # side and no terminator/shading would ever be visible. Leaving the
    # camera angle independent of the Sun angle is what lets shading show
    # up at all — adjust `azim`/`elev` (or sun_azimuth_deg/sun_elevation_deg)
    # to dial in whatever phase you want to see.

    # ── Sun's actual position in this scene (a true background object) ───
    # Computed once here so both the Sun's own rendering AND the light
    # projected onto the Moon/Earth come from the SAME position — this is
    # the explicit "project the Sun's light onto the body" step, rather
    # than just reusing the raw ephemeris unit vector everywhere.
    #
    # IMPORTANT: mplot3d has no true depth perspective — an object placed
    # outside the axes' set xlim/ylim/zlim does NOT render smaller/farther
    # the way a real camera would; it just risks rendering partially or
    # fully outside the figure canvas (this is exactly what was happening
    # before — the Sun's "Sun" label was getting clipped at the image
    # edge). So distance is placed at a fraction strictly INSIDE the axis
    # half-width, and the "distant background object" feel comes from a
    # small display radius (see background_sun_radius below) instead.
    sun_pos_local = None
    if sun_hat is not None:
        axis_half_width = plot_range / 2.0
        sun_pos_local = sun_hat * (axis_half_width * 0.78)

    # ── Stars ─────────────────────────────────────────────────────────────────
    if show_stars:
        add_starfield(ax, plot_range, elev=elev, azim=azim,
                      mag_limit=mag_limit, epoch=t_ref)

    # ── Earth — only show if plot range is large enough to see it ────────────
    # The Earth-Moon line only sits fixed along the x-axis in the ROTATING
    # lunar-fixed frame (that's the gcrf_to_lunar_fixed convention this
    # hardcoded position assumed). For r_frame='moon_centered' inertial
    # data, Earth's actual position relative to the Moon depends on the
    # real calendar date and is NOT pinned to any particular axis — it
    # has to be computed from the Moon's real ephemeris instead.
    if r_frame == 'moon_centered':
        try:
            from ssapy import get_body
            from astropy.time import Time
            t_for_earth = t_ref if t_ref is not None else Time.now()
            # Reduce to a single epoch the same way sun direction does —
            # get_body().position() and downstream code expect array-
            # shaped time, even for N=1, not a bare scalar.
            if isinstance(t_for_earth, Time):
                t_single = t_for_earth if not t_for_earth.isscalar else t_for_earth.reshape(1)
                t_single = t_single[0:1]
            else:
                t_single = np.atleast_1d(np.asarray(t_for_earth, dtype=float))[0:1]
            moon_body = get_body("moon")
            moon_pos_gcrf_m = moon_body.position(t_single).T
            moon_pos_gcrf_m = np.atleast_2d(moon_pos_gcrf_m)[0]
            # Earth relative to Moon = -(Moon relative to Earth).
            # get_body().position() returns meters; convert to km to match this
            # function's internal
            # units (unit_conversion=1e3, unit_label='km').
            earth_pos_km = -moon_pos_gcrf_m / 1e3
        except Exception as e:
            print(f"[moon_plot_3d] Could not compute real Earth position "
                  f"for moon_centered frame: {e} — Earth will not be drawn.")
            earth_pos_km = None
    else:
        # Rotating lunar-fixed frame: Earth-Moon line fixed along x-axis
        earth_pos_km = np.array([-384400.0, 0.0, 0.0])

    _earth_visible = (plot_range > 100000) if show_earth is None else show_earth
    earth_draw = None
    if earth_pos_km is not None and _earth_visible:
        earth_r = EARTH_RADIUS / unit_conversion
        visible_earth_r = earth_r if physical_body_sizes else max(earth_r, plot_range * 0.03)
        # Use the real sun_hat directly — NOT a projection from
        # sun_pos_local. sun_pos_local is a STYLIZED rendering position
        # (scaled to plot_range so the Sun fits inside the visible box,
        # typically tens-to-hundreds of thousands of km), not the Sun's
        # true ~150,000,000 km distance. That stand-in works fine for the
        # Moon, which sits at the origin (the projection trivially cancels
        # out to sun_hat regardless of distance — verified earlier), but
        # for Earth — whose real distance from the Moon (~384,400 km) is
        # comparable in magnitude to the stylized sun_pos_local — using
        # that projection introduces direction errors of 60-80+ degrees,
        # nowhere close to the true solar direction. The real Sun is far
        # enough away (~150M km) that the same sun_hat direction is
        # correct (to well under 1 degree of parallax) for every body in
        # the Earth-Moon system, so just reuse it directly.
        earth_light_dir = sun_hat
        earth_draw = (earth_pos_km, visible_earth_r, earth_light_dir)
   # ── Orbit tracks ──────────────────────────────────────────────────────────
    visible_moon_r = moon_r if physical_body_sizes else max(moon_r, plot_range * 0.03)
    bodies = [(np.zeros(3), visible_moon_r)]
    if earth_draw is not None:
        bodies.append((earth_draw[0], earth_draw[1]))
    foreground_tracks = []
    for orbit_index, xyz in xyz_list:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        background, foreground = _split_orbit_segments_for_bodies(xyz, bodies, elev, azim)
        color = "#ff6600" if len(xyz_list) == 1 else cm.rainbow(np.linspace(0, 1, len(xyz_list)))[orbit_index]
        linewidth = 2.0 if len(xyz_list) == 1 else 1.5
        if len(background):
            ax.add_collection(Line3DCollection(background, colors=[color], linewidth=linewidth, alpha=0.9, zorder=5))
        foreground_tracks.append((foreground, color, linewidth))
 # ── Moon ─────────────────────────────────────────────────────────────────
    # Visibility floor matching Earth's existing pattern (visible_earth_r =
    # max(earth_r, plot_range*0.03)) — without this, the Moon renders at
    # its true physical radius (~1737 km) regardless of scene scale, so
    # any plot spanning hundreds of thousands of km (e.g. a real cislunar
    # trajectory) shrinks it to an invisible sub-pixel dot.
    if earth_draw is not None:
        earth_pos, earth_radius, earth_light_dir = earth_draw
        _textured_earth(ax, earth_pos[0], earth_pos[1], earth_pos[2],
                        earth_radius, earth_img, n=128, sun_hat=earth_light_dir,
                        ambient=shade_ambient, diffuse=shade_diffuse, zorder=10)
    # Use sun_hat directly (same reasoning as Earth's shading above) — the
    # Moon sits at the origin, so light_direction_from_positions(
    # sun_pos_local, zeros(3)) would reduce to exactly sun_hat anyway,
    # just through an unnecessary detour. Using sun_hat directly here
    # keeps both bodies' shading logic consistent and avoids relying on
    # that projection pattern at all, given how easily it goes wrong for
    # any body NOT at the origin (see the Earth shading fix above).
    moon_light_dir = sun_hat
    drawn = _textured_moon(ax, 0, 0, 0, visible_moon_r, moon_img, n=96,
                           sun_hat=moon_light_dir,
                           ambient=shade_ambient, diffuse=shade_diffuse, zorder=10)
    if not drawn:
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi,   30)
        ax.plot_surface(
            visible_moon_r * np.outer(np.cos(u), np.sin(v)),
            visible_moon_r * np.outer(np.sin(u), np.sin(v)),
            visible_moon_r * np.outer(np.ones(30), np.cos(v)),
            color='#aaaaaa', alpha=0.9, zorder=10)
    for foreground, color, linewidth in foreground_tracks:
        if len(foreground):
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            ax.add_collection(Line3DCollection(foreground, colors=[color], linewidth=linewidth, alpha=0.9, zorder=15))
    # Moon position indicator
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(visible_moon_r*np.cos(theta), visible_moon_r*np.sin(theta),
            np.zeros(200), color='white', linewidth=0.8, alpha=0.4, linestyle='--')
    ax.text(0, 0, visible_moon_r*1.35, 'Moon', color='white', fontsize=13,
            alpha=0.9, zorder=30)

    # ── Sun sphere — rendered as a true background object: pushed well
    #    behind the Moon (2.5x plot_range, vs. the Moon's ~3x moon_r extent)
    #    with a correspondingly small display radius, so it reads as
    #    distant rather than as a nearby second body. Limb darkening is
    #    computed from the actual camera angle (elev/azim) so the Sun
    #    looks genuinely spherical from whatever viewpoint is chosen. ──────
    if show_sun_graphic and sun_pos_local is not None:
        sun_radius = background_sun_radius(plot_range, size_factor=0.018)
        render_sun(ax, sun_pos_local, radius=sun_radius,
                   elev_deg=elev, azim_deg=azim)

    # ── Lagrange points ───────────────────────────────────────────────────────
    # Lagrange points are fixed positions only in the ROTATING lunar-fixed
    # frame, at a single instant. Real moon-centered inertial data (e.g.
    # Horizons ephemeris) typically spans hours-to-days, during which the
    # Moon moves tens of degrees around Earth — overlaying static L-point
    # markers on that data would misrepresent the geometry, so they're
    # skipped automatically for r_frame='moon_centered'.
    if show_lagrange and r_frame != 'moon_centered':
        label_offset = plot_range * 0.02
        for point, pos in lagrange_points_lunar_fixed_frame().items():
            pos_geo = pos / unit_conversion
            ax.scatter([pos_geo[0]], [pos_geo[1]], [pos_geo[2]],
                       color='white', s=15, zorder=6, clip_on=False)
            ax.text(pos_geo[0], pos_geo[1] + label_offset,
                    pos_geo[2] + label_offset, point,
                    color='white', fontsize=12, zorder=30)
    elif show_lagrange and r_frame == 'moon_centered':
        print("[moon_plot_3d] Skipping Lagrange points: they're only valid "
              "in the rotating lunar-fixed frame at a single instant, not "
              "over a multi-day inertial trajectory.")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(f'Moon — Lunar Fixed Frame\n{title}',
                 color=textcolor, fontsize=11, pad=15)

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig, ax


if __name__ == "__main__":
    # ── GUI entry point ──────────────────────────────────────────────────────
    # toolkit_gui.py's run_script() writes the Export Plots tab's settings
    # (orbital elements + the shade_ambient / shade_diffuse sliders) to a
    # small Python file and points the GUI_CONFIG env var at it, then runs
    # this script via `conda run -n <env> python moon_plot_3d.py`. Read
    # that config here, build a real propagated trajectory via the
    # toolkit's own OrbitalState, and pass everything into moon_plot_3d.
    import os

    cfg = {}
    cfg_path = os.environ.get("GUI_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            exec(f.read(), cfg)
    else:
        print("[moon_plot_3d] No GUI_CONFIG found — running with defaults.")

    output_dir = cfg.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "moon_plot_3d.jpg")

    # Build and propagate the orbit via the toolkit's real OrbitalState —
    # now a genuine same-folder sibling (core/ was moved into
    # ssatk/plots/), so this is a normal relative import, same as
    # .sun_mpl/.sun_render above.
    r_m, t_gps = None, None
    try:
        from .orbit_state import OrbitalState

        state = OrbitalState(
            a_km=cfg.get("a_km", 6928.0),
            e=cfg.get("e", 0.001),
            inc_deg=cfg.get("inc_deg", 51.6),
            raan_deg=cfg.get("raan_deg", 0.0),
            argp_deg=cfg.get("argp_deg", 0.0),
            nu_deg=cfg.get("nu_deg", 0.0),
            epoch=cfg.get("epoch", None),
        )
        traj = state.propagate(
            n_orbits=cfg.get("n_orbits", 3.0),
            dt_s=cfg.get("dt_s", 60.0),
        )
        if traj.ok:
            r_m = traj.r * 1e3   # Trajectory.r is km; moon_plot_3d expects meters
            t_gps = traj.t
        else:
            print(f"[moon_plot_3d] Propagation failed: {traj.msg} — "
                  f"rendering Moon with no orbit track.")
    except Exception as ex:
        print(f"[moon_plot_3d] Could not build orbit from config: {ex} — "
              f"rendering Moon with no orbit track.")

    fig, ax = moon_plot_3d(
        r=r_m, t=t_gps,
        title=cfg.get("title", ""),
        show_lagrange=cfg.get("show_lagrange", True),
        show_stars=cfg.get("show_stars", True),
        show_sun=cfg.get("show_sun", True),
        shade_ambient=cfg.get("shade_ambient", 0.22),
        shade_diffuse=cfg.get("shade_diffuse", 0.78),
        sun_azimuth_deg=cfg.get("sun_azimuth_deg", None),
        sun_elevation_deg=cfg.get("sun_elevation_deg", None),
        save_path=out_path,
        show=False,
    )
    print(f"[moon_plot_3d] Saved -> {out_path}")
