import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image as PILImage
import numpy as np


from ._textures import earth_texture_path
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


def _view_unit_vector(el_deg, az_deg):
    el = np.radians(float(el_deg))
    az = np.radians(float(az_deg))
    v = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _earth_occultation_mask(r_scaled, earth_radius_scaled, el_deg, az_deg):
    """
    Return boolean mask where True means point is visible (not hidden behind Earth)
    for the current camera direction defined by (el, az).
    """
    r_scaled = np.asarray(r_scaled, dtype=float)
    if r_scaled.ndim != 2 or r_scaled.shape[1] != 3:
        raise ValueError("r_scaled must be shape (N, 3)")

    # Unit vector pointing from origin toward the viewer
    v_hat = _view_unit_vector(el_deg, az_deg)

    # Signed depth along viewing axis
    depth = r_scaled @ v_hat

    # Perpendicular distance from viewing axis
    perp = r_scaled - np.outer(depth, v_hat)
    rho = np.linalg.norm(perp, axis=1)

    # Points are hidden if they are behind the planet center and project inside the Earth disk
    hidden = (depth < 0.0) & (rho < float(earth_radius_scaled))
    return ~hidden


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
        Draw a larger marker at ``reference_index`` for each orbit track.

    Returns
    -------
    fig, ax : Matplotlib Figure and 3D Axes.
    """

    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "globe_plot")

    # ---- Normalize/validate inputs ----
    r_list, t_list = valid_orbits(r, t)  # t_list kept for consistency
    n_tracks = len(r_list)

    # --- Labels sanity ---
    if labels is not None and len(labels) != n_tracks:
        raise ValueError("labels must have same length as number of orbits (tracks)")

    # --- Orbit colors sanity ---
    if orbit_colors is not None and len(orbit_colors) != n_tracks:
        raise ValueError("orbit_colors must have same length as number of orbits (tracks)")

    # --- Orient globe at an explicit time, or the final supplied sample time ---
    if globe_time is None and t_list and len(t_list[0]):
        ref_idx = int(reference_index) % len(t_list[0])
        globe_time = t_list[0][ref_idx]
    # With a time, orient the textured Earth with the full GCRF<-ITRF rotation
    # (precession-nutation, GAST at UT1, polar motion; the chain SSAPy's
    # groundTrack uses), not a z-rotation: a z-rotation by GAST alone put the
    # globe 1242 arcsec (38.4 km at the equator) off SSAPy's ground tracks on
    # 2026-10-09, and no z-rotation carries the 0.15 deg tilt of the true pole.
    earth_to_scene = None
    if globe_time is not None:
        from ..coordinates.frames import eci_to_ecf_matrix

        earth_to_scene = eci_to_ecf_matrix(gps_seconds_at(globe_time, default=0.0)).T
        lon0 = 0.0

    # ---------- Theme ----------
    if c in ("black", "b"):
        plotcolor, textcolor = "black", "white"
    elif c in ("white", "w"):
        plotcolor, textcolor = "white", "black"
    else:
        plotcolor, textcolor = "white", "black"

    # ---------- Earth texture ----------
    scale = 1.0 if (not np.isfinite(scale) or scale <= 0) else float(scale)
    tex_w = int(round(5400 / scale))
    tex_h = int(round(2700 / scale))
    earth_png = PILImage.open(earth_texture_path()).resize(
        (tex_w, tex_h), resample=PILImage.BILINEAR
    )
    bm = np.asarray(earth_png, dtype=float) / 255.0  # (H, W, C)

    # ---------- Globe mesh (match texture resolution) ----------
    lons = np.radians(np.linspace(-180.0, 180.0, bm.shape[1]))
    lats = np.radians(np.linspace(-90.0, 90.0, bm.shape[0])[::-1])

    # Apply longitude offset (rotate globe about z-axis)
    lon0_rad = np.radians(float(lon0))
    lons = lons + lon0_rad

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    scale_fac = EARTH_RADIUS / RGEO
    mesh_x = np.cos(lat_grid) * np.cos(lon_grid) * scale_fac
    mesh_y = np.cos(lat_grid) * np.sin(lon_grid) * scale_fac
    mesh_z = np.sin(lat_grid) * scale_fac
    if earth_to_scene is not None:
        scene_xyz = np.stack([mesh_x, mesh_y, mesh_z], axis=-1) @ earth_to_scene.T
        mesh_x, mesh_y, mesh_z = scene_xyz[..., 0], scene_xyz[..., 1], scene_xyz[..., 2]

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
    ax.tick_params(axis="both", colors=textcolor)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

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

        if ri_vis.size == 0:
            continue

        if n_tracks == 1:
            # Single orbit:
            if orbit_colors is not None:
                color = orbit_colors[0]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=color, s=1,
                )
            else:
                # original: color by point
                point_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, ri_scaled.shape[0]))
                point_colors = point_colors[visible]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=point_colors, s=1,
                )
        else:
            # Multiple orbits: one color per orbit
            color = colors[i] if colors is not None else None
            ax.scatter(
                ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                color=color, s=1,
            )

        if highlight_reference and len(ri_scaled) > 1:
            ref_idx = int(reference_index) % len(ri_scaled)
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
        L = (max_extent if max_extent > 0 else scale_fac) * 1.2
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

    if save_path:
        figsave(fig, save_path)

    return fig, ax
