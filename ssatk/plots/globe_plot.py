
from .rendering import _with_low_spec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


from ._globe_scene import TexturedEarth, earth_to_scene, visible_points
from .plotutils import (
    make_black,
    make_white,
    figsave,
    valid_orbits,
    _pop_save_path_aliases,
    _raise_unrecognized_kwargs,
)
from ..constants import RGEO, EARTH_RADIUS
from .scene_primitives import earth_rotation_deg_from_time, gps_seconds_at


def _earth_lon0_from_time(t):
    """Compute the globe texture rotation angle in degrees for ``t``."""
    return earth_rotation_deg_from_time(t)


def _earth_occultation_mask(r_scaled, earth_radius_scaled, el_deg, az_deg):
    """Compatibility wrapper for spherical Earth occultation."""
    return visible_points(r_scaled, earth_radius_scaled, el_deg, az_deg)


@_with_low_spec
def globe_plot(
    r,
    t=None,
    limits=None,
    title="",
    c="black",
    figsize=(7, 8),
    save_path=None,
    el=5.0,
    az=5.0,
    scale=5.0,
    fontsize=18,
    labels=None,          # legend labels, one per orbit
    orbit_colors=None,    # per-orbit color(s)
    show_legend=True,     # toggle legend
    legend_kwargs=None,   # kwargs passed to ax.legend()
    lon0=0.0,             # manual rotation of globe about z-axis in degrees
    globe_time=None,      # optional time to orient globe based on Earth rotation
    reference_index=-1,   # sample whose time/orbit point matches the Earth texture
    highlight_reference=True,
    highlight_kwargs=None,
    ax=None,
    frame="gcrf",
    earth_resolution=None,
    **save_kwargs,
):
    """
    Plot a textured Earth and scatter satellite positions in 3D.

    Parameters
    ----------
    r, t : passed to valid_orbits to normalize [1].
    limits : None, scalar, or [[x1,x2],[y1,y2],[z1,z2]]
        None -> auto; scalar -> cube [-L,L] for all axes;
        3x2-like -> explicit per-axis limits.
    labels : list of str, optional
        Per-orbit labels for legend; must match number of tracks if provided.
    orbit_colors : list or array-like, optional
        Per-orbit colors. If provided, length must match number of tracks.
        Each entry can be any Matplotlib color (name, hex, RGB tuple, etc.).
    show_legend : bool
        If True and labels are provided, draw a legend.
    legend_kwargs : dict
        Extra kwargs forwarded to ax.legend().
    lon0 : float
        Rotation of the globe texture about the z-axis in degrees. Positive
        values rotate longitudes eastward.
    globe_time : astropy.time.Time or float, optional
        If provided, overrides lon0 and orients the globe with the full
        GCRF<-ITRF rotation at this time (the chain SSAPy's groundTrack uses).  If omitted and
        a time series is supplied, the final sample time is used so the Earth
        texture is accurate at the highlighted reference point.
    reference_index : int
        Orbit sample to highlight as the instant represented by the Earth
        texture orientation.  Defaults to the last sample.
    highlight_reference : bool
        Draw a larger marker at the sample matching the Earth epoch. A track
        without a sample within 1 microsecond of that epoch has no reference
        marker. Hidden points behind Earth are not highlighted.
    frame : {"gcrf", "itrf"}
        Display frame; input positions always use GCRF metres. Earth rotates
        in GCRF and is fixed in ITRF, for both still and animated exports.
    earth_resolution : int, optional
        Longitude cells, with half as many latitude cells. Overrides ``scale``.

    Returns
    -------
    fig, ax : Matplotlib Figure and 3D Axes.
    """

    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "globe_plot")

    # ---- Normalize/validate inputs ----
    r_list, t_list = valid_orbits(r, t)  # t_list kept for consistency
    n_tracks = len(r_list)
    from ._frames import normalize_orbit_frame
    frame = normalize_orbit_frame(frame)
    if frame not in {"gcrf", "itrf"}:
        raise ValueError("Textured globe views support GCRF or ITRF coordinates")
    if frame == "itrf":
        if t is None:
            raise ValueError("ITRF globe requires absolute times for GCRF input positions")
        from ..coordinates.earth_fixed import gcrf_to_itrf
        r_list = [gcrf_to_itrf(ri, ti) for ri, ti in zip(r_list, t_list)]

    # --- Labels sanity ---
    if labels is not None and len(labels) != n_tracks:
        raise ValueError("labels must have same length as number of orbits (tracks)")

    # --- Orbit colors sanity ---
    if orbit_colors is not None and len(orbit_colors) != n_tracks:
        raise ValueError("orbit_colors must have same length as number of orbits (tracks)")

    # --- Orient globe at an explicit time, or the final supplied sample time ---
    if globe_time is None and t is not None and t_list and len(t_list[0]):
        ref_idx = int(reference_index) % len(t_list[0])
        globe_time = t_list[0][ref_idx]
    # With a time, orient the textured Earth with the full GCRF<-ITRF rotation
    # (precession-nutation, GAST at UT1, polar motion; the chain SSAPy's
    # groundTrack uses), not a z-rotation: a z-rotation by GAST alone put the
    # globe 1242 arcsec (38.4 km at the equator) off SSAPy's ground tracks on
    # 2026-10-09, and no z-rotation carries the 0.15 deg tilt of the true pole.
    rotation = None
    reference_indices = []
    if globe_time is not None:
        epoch = gps_seconds_at(globe_time, default=0.0)
        if not np.isfinite(epoch):
            raise ValueError("globe_time must be finite")
        rotation = earth_to_scene(epoch, frame)
        for times in t_list:
            matches = np.flatnonzero(np.abs(times.gps - epoch) <= 1e-6) if t is not None else []
            reference_indices.append(int(matches[0]) if len(matches) else None)
    else:
        angle = np.radians(float(lon0))
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        reference_indices = [int(reference_index) % len(track) for track in r_list]

    # ---------- Theme ----------
    if c in ("black", "b"):
        plotcolor, textcolor = "black", "white"
    elif c in ("white", "w"):
        plotcolor, textcolor = "white", "black"
    else:
        plotcolor, textcolor = "white", "black"

    # Retain the old scale knob (formerly a 5400/scale image, stride four),
    # but sample colours at cell centres and share geometry with the animation.
    if earth_resolution is None:
        scale = 1.0 if (not np.isfinite(scale) or scale <= 0) else float(scale)
        earth_resolution = max(12, min(1440, int(round(5400 / (4 * scale)))))
    scale_fac = EARTH_RADIUS / RGEO

    # ---------- Figure and axes ----------
    if ax is None:
        fig = plt.figure(dpi=100, figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        if not hasattr(ax, "zaxis"):
            raise ValueError("globe_plot ax must be a 3D Matplotlib axis")
        fig = ax.figure
    fig.patch.set_facecolor(plotcolor)
    ax.set_facecolor(plotcolor)
    ax.view_init(elev=float(el), azim=float(az))
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    ax.tick_params(axis="both", colors=textcolor)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

    earth = TexturedEarth(ax, scale_fac, earth_resolution)
    earth.orient(rotation)
    if globe_time is not None:
        from astropy.time import Time
        ax.text2D(0.02, 0.96, f"{Time(epoch, format='gps').utc.isot} UTC · {frame.upper()}",
                  transform=ax.transAxes, color=textcolor, fontsize=10)

    # ---------- Colors for satellites ----------
    if orbit_colors is not None:
        colors = list(orbit_colors)
    else:
        # 1 orbit -> colormap over points; >1 -> one color per orbit
        if n_tracks == 1:
            colors = None  # handled in loop
        else:
            colors = [plt.cm.rainbow(v) for v in np.linspace(0.0, 1.0, n_tracks)]

    # ---------- Scatter satellites ----------
    max_extent = 0.0

    for i, ri in enumerate(r_list):
        ri_scaled = ri / RGEO
        if ri_scaled.size:
            max_extent = max(max_extent, float(np.nanmax(np.abs(ri_scaled))))

        visible = _earth_occultation_mask(
            ri_scaled,
            earth_radius_scaled=scale_fac,
            el_deg=el,
            az_deg=az,
        )
        ri_vis = ri_scaled[visible]

        if n_tracks == 1:
            # Single orbit:
            if orbit_colors is not None:
                color = orbit_colors[0]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=color, s=1, zorder=2,
                )
            else:
                # original: color by point
                point_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, ri_scaled.shape[0]))
                point_colors = point_colors[visible]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=point_colors, s=1, zorder=2,
                )
        else:
            # Multiple orbits: one color per orbit
            color = colors[i] if colors is not None else None
            ax.scatter(
                ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                color=color, s=1, zorder=2,
            )

        ref_idx = reference_indices[i]
        if (highlight_reference and ref_idx is not None and visible[ref_idx]):
            ref_point = ri_scaled[ref_idx]
            marker_kwargs = dict(
                color="#FFD700",
                edgecolors="black" if textcolor == "black" else "white",
                linewidths=0.8,
                s=45,
                marker="*",
                label=None,
                zorder=10,
            )
            if highlight_kwargs:
                marker_kwargs.update(highlight_kwargs)
            ax.scatter([ref_point[0]], [ref_point[1]], [ref_point[2]], **marker_kwargs)

    # ---------- Axis limits: scalar or explicit ----------
    # limits can be:
    #   None -> auto
    #   scalar -> cube [-L,L]
    #   [[x1,x2],[y1,x2],[z1,z2]] -> explicit
    if limits is None:
        L = max(max_extent, scale_fac) * 1.2
        xlim = ylim = zlim = (-L, L)
        limits_is_scalar_or_none = True
    elif np.isscalar(limits):
        L = float(limits)
        xlim = ylim = zlim = (-L, L)
        limits_is_scalar_or_none = True
    else:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (3, 2):
            raise ValueError("limits must be scalar or array-like of shape (3,2): [[x1,x2],[y1,y2],[z1,z2]]")
        xlim = tuple(limits[0])
        ylim = tuple(limits[1])
        zlim = tuple(limits[2])
        limits_is_scalar_or_none = False

    # Treat very large scalar limits as meters and convert to GEO units.
    if limits_is_scalar_or_none and max(abs(xlim[0]), abs(xlim[1])) > 1e5:
        xlim = (xlim[0] / RGEO, xlim[1] / RGEO)
        ylim = (ylim[0] / RGEO, ylim[1] / RGEO)
        zlim = (zlim[0] / RGEO, zlim[1] / RGEO)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # Matplotlib's default 3-D box aspect is 4:4:3, which drew the globe 25%
    # flattened along z. Scale the box to the data spans instead.
    ax.set_box_aspect((abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])))

    # ---------- Ticks/labels ----------
    # For symmetric cubes use integer ticks around center; for general limits, keep it simple
    if xlim[0] == -xlim[1] and ylim[0] == -ylim[1] and zlim[0] == -zlim[1]:
        L_tick = int(xlim[1])
        ax.set_xticks([-L_tick, 0, L_tick])
        ax.set_yticks([-L_tick, 0, L_tick])
        ax.set_zticks([-L_tick, 0, L_tick])

    # hide tick labels as in your original
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("x [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_ylabel("y [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_zlabel("z [GEO]", color=textcolor, fontsize=fontsize)
    if title:
        ax.set_title(title, color=textcolor, fontsize=fontsize)

    ax.tick_params(axis="x", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="y", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="z", colors=textcolor, labelsize=fontsize)

    # ---------- Legend (lines instead of dots) ----------
    if show_legend and labels is not None:
        handles = []
        for i, label in enumerate(labels):
            if orbit_colors is not None:
                color = orbit_colors[i]
            else:
                if n_tracks == 1:
                    color = plt.cm.rainbow(0.5)
                else:
                    color = colors[i]
            h = Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=label,
            )
            handles.append(h)

        kw = dict(loc="best")
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=handles, **kw)

    # Apply theme helpers
    if c in ("black", "b"):
        fig, axes = make_black(fig, ax)
        ax = axes[0]
    elif c in ("white", "w"):
        fig, axes = make_white(fig, ax)
        ax = axes[0]

    # A globe should not be enclosed in opaque grey Matplotlib panes.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.5, 0.5, 0.5, 0.2))
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.2)
    if save_path:
        figsave(fig, save_path)

    return fig, ax
