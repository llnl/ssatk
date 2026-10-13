from __future__ import annotations

from ._body_surfaces import BodySurface as _BodySurface, body_rotation as _body_rotation, body_disc as _body_disc

from ssapy import get_body as _get_body
import matplotlib.cm as _cm
import matplotlib.font_manager as _font_manager
import matplotlib.pyplot as _plt
from matplotlib.lines import Line2D as _Line2D
from matplotlib.ticker import MaxNLocator as _MaxNLocator
import numpy as _np

from ..constants import EARTH_RADIUS as _EARTH_RADIUS
from ..constants import MOON_RADIUS as _MOON_RADIUS
from ..constants import RGEO as _RGEO
from ..coordinates import gcrf_to_lunar_fixed as _gcrf_to_lunar_fixed
from ._legend_handlers import GradientLineHandler as _GradientLineHandler
from ._legend_handlers import BodyMapPatch, BodyMapHandler, body_legend_handle
from .occultation import _projected_orbit
from .plotutils import figsave as _figsave
from .plotutils import valid_orbits as _valid_orbits


def _lagrange_points_lunar_fixed_frame():
    from ..orbital_mechanics import lagrange_points_lunar_fixed_frame

    return lagrange_points_lunar_fixed_frame()


def _cislunar_plot_core(
    r,
    t=None,
    figsize=(8, 8),
    fontsize=12,
    save_path=False,
    show=False,
    title=None,
    c="white",
    *,
    mode="combined",
    legend=True,
    scene_pad=0.08,
):
    mode = mode.lower()
    if mode not in {"combined", "3d", "xy"}:
        raise ValueError("mode must be one of: combined, 3d, xy")

    if not _np.isfinite(scene_pad) or scene_pad < 0:
        raise ValueError("scene_pad must be a finite nonnegative fraction")
    r, t = _valid_orbits(r, t)
    if not r or any(not _np.all(_np.isfinite(xyz)) for xyz in r):
        raise ValueError("r must contain nonempty finite tracks")
    textcolor, plotcolor = _plot_colors(c)

    # Compute every track before choosing limits or units. In particular, the
    # Earth-centred distance of a lunar orbiter must not determine the scale of
    # the Moon-centred panel, and input order must not change a panel's units.
    moon_body = _get_body("moon")
    lunar_tracks, moon_tracks, earth_tracks = [], [], []
    for xyz, times in zip(r, t):
        lunar_tracks.append(_gcrf_to_lunar_fixed(xyz, times))
        moon_tracks.append(_np.asarray(moon_body.position(times)).reshape(3, -1).T)
        earth_tracks.append(_gcrf_to_lunar_fixed(_np.zeros_like(xyz), times))
    lunar_unit, lunar_label = _scene_units(lunar_tracks)
    gcrf_unit, gcrf_label = _scene_units(r)
    dimensions = 2 if mode == "xy" else 3
    bounds_lunar = _scene_bounds(lunar_tracks, _MOON_RADIUS, lunar_unit, scene_pad, dimensions)
    bounds_gcrf = _scene_bounds(r, _EARTH_RADIUS, gcrf_unit, scene_pad, dimensions)
    fig, axes = _create_cislunar_axes(mode, figsize, plotcolor)
    earth_visible = mode != "3d"  # The GCRF panel always contains Earth.
    for orbit_index, xyz_raw in enumerate(r):
        xyz, xyz_lunar = xyz_raw/gcrf_unit, lunar_tracks[orbit_index]/lunar_unit
        r_moon, r_earth_lunar = moon_tracks[orbit_index]/gcrf_unit, earth_tracks[orbit_index]/lunar_unit
        mask_gcrf = _bounds_mask(r_moon, bounds_gcrf, dimensions)
        mask_lunar = _bounds_mask(r_earth_lunar, bounds_lunar, dimensions)
        earth_visible |= bool(_np.any(mask_lunar))
        grey_colors, blue_colors = _body_colors(r_moon, mask_gcrf, mask_lunar, mode)
        scatter_dot_colors = _scatter_colors(r, xyz, orbit_index)
        if mode == "combined":
            _plot_cislunar_gcrf_3d(axes["gcrf"], xyz, r_moon, mask_gcrf, grey_colors, gcrf_unit, gcrf_label, scatter_dot_colors, textcolor, fontsize)
            _plot_cislunar_lunar_3d(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, lunar_unit, lunar_label, scatter_dot_colors, textcolor, fontsize)
        elif mode == "xy":
            _plot_cislunar_gcrf_xy(axes["gcrf"], xyz, r_moon, mask_gcrf, grey_colors, gcrf_unit, gcrf_label, scatter_dot_colors, textcolor, fontsize)
            _plot_cislunar_lunar_xy(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, lunar_unit, lunar_label, scatter_dot_colors, textcolor, fontsize)
        else:
            _plot_cislunar_lunar_3d(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, lunar_unit, lunar_label, scatter_dot_colors, textcolor, fontsize)

    epoch = t[0][0]
    for panel, ax in axes.items():
        unit = lunar_unit if panel == "lunar" else gcrf_unit
        frame = "lunar" if panel == "lunar" else "gcrf"
        centers = {"moon": _np.zeros(3) if panel == "lunar" else moon_tracks[0][0]/unit,
                   "earth": earth_tracks[0][0]/unit if panel == "lunar" else _np.zeros(3)}
        bounds = bounds_lunar if panel == "lunar" else bounds_gcrf
        for body, radius in [("earth", _EARTH_RADIUS/unit), ("moon", _MOON_RADIUS/unit)]:
            center = centers[body]
            if _bounds_mask(center.reshape(1, 3), bounds, dimensions)[0]:
                rotation = _body_rotation(body, epoch, frame)
                if mode == "xy":
                    _body_disc(ax, body, radius, center, rotation)
                else:
                    _BodySurface(ax, body, radius, center, rotation)

    _finish_cislunar_axes(axes, bounds_gcrf, bounds_lunar, mode, title, textcolor, plotcolor, fontsize)
    lagrange_visible = any(_bounds_mask(_np.asarray(pos/lunar_unit).reshape(1, 3), bounds_lunar, dimensions)[0]
                           for pos in _lagrange_points_lunar_fixed_frame().values())
    _add_cislunar_legend(axes["lunar"], len(r), mode, legend, textcolor, plotcolor, fontsize,
                         earth_visible=earth_visible, lagrange_visible=lagrange_visible)

    if save_path:
        _figsave(fig, save_path)
    if show:
        _plt.show()
    _plt.close(fig)

    if mode == "3d":
        return fig, axes["lunar"]
    return fig, [axes["gcrf"], axes["lunar"]]


def _scene_units(tracks):
    """Choose one distance unit for all tracks in the panel's own frame."""
    if any(_np.max(_np.linalg.norm(xyz, axis=1)) >= .95*_RGEO for xyz in tracks):
        return _RGEO, "GEO"
    return 1e3, "km"


def _scene_bounds(tracks, body_radius, unit, padding, dimensions):
    """Tight equal-axis bounds for tracks plus the central body's true sphere.

    Padding is a fraction of the largest displayed extent, on each side.
    Bounds are computed in metres first; changing axis units cannot change
    the physical field of view. XY panels ignore out-of-plane excursions.
    """
    lower, upper = _np.full(3, -body_radius), _np.full(3, body_radius)
    for xyz in tracks:
        lower = _np.minimum(lower, xyz.min(axis=0))
        upper = _np.maximum(upper, xyz.max(axis=0))
    center = (lower+upper)/2
    half_span = max((upper-lower)[:dimensions])*(.5+padding)
    return {"lower": (center-half_span)/unit, "upper": (center+half_span)/unit}


def _plot_colors(c):
    if "w" in c:
        return "black", "white"
    return "white", "black"


def _create_cislunar_axes(mode, figsize, plotcolor):
    if mode == "combined":
        fig, (ax_gcrf, ax_lunar) = _plt.subplots(
            1,
            2,
            figsize=figsize,
            dpi=100,
            subplot_kw={"projection": "3d"},
            facecolor=plotcolor,
        )
        return fig, {"gcrf": ax_gcrf, "lunar": ax_lunar}
    if mode == "xy":
        fig, (ax_gcrf, ax_lunar) = _plt.subplots(1, 2, figsize=figsize, dpi=100, facecolor=plotcolor)
        return fig, {"gcrf": ax_gcrf, "lunar": ax_lunar}

    fig = _plt.figure(figsize=figsize, dpi=100, facecolor=plotcolor)
    return fig, {"lunar": fig.add_subplot(111, projection="3d")}


def _bounds_mask(points, bounds, dimensions):
    mask = _np.ones(points.shape[0], dtype=bool)
    for dimension in range(dimensions):
        mask &= (points[:, dimension] >= bounds["lower"][dimension]) & (points[:, dimension] <= bounds["upper"][dimension])
    return mask


def _body_colors(r_moon, mask_gcrf, mask_lunar, mode):
    if _np.size(r_moon[:, 0]) > 1:
        grey_colors = _cm.Greys(_np.linspace(0, 0.8, len(r_moon[:, 0])))[::-1][mask_gcrf]
        blue_colors = _cm.Blues(_np.linspace(0.2, 0.8, len(r_moon[:, 0])))[::-1][mask_lunar]
    else:
        grey_colors = "grey"
        blue_colors = "lightblue" if mode == "xy" else "blue"
    return grey_colors, blue_colors


def _scatter_colors(r, xyz, orbit_index):
    if len(r) == 1:
        return _cm.rainbow(_np.linspace(0, 1, len(xyz[:, 0])))
    return _cm.rainbow(_np.linspace(0, 1, len(r)))[orbit_index]


def _plot_cislunar_gcrf_3d(ax, xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1, label="Orbit Path")
    ax.scatter3D(r_moon[mask_gcrf, 0], r_moon[mask_gcrf, 1], r_moon[mask_gcrf, 2], color=grey_colors, s=(_MOON_RADIUS / unit_conversion))
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=True)


def _plot_cislunar_lunar_3d(ax, xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    ax.scatter3D(*xyz_lunar.T, color=scatter_dot_colors, s=1, label="Orbit Path", zorder=15)
    ax.scatter3D(r_earth_lunar[mask_lunar, 0], r_earth_lunar[mask_lunar, 1], r_earth_lunar[mask_lunar, 2], color=blue_colors, s=55)
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=True)
    for point, pos in _lagrange_points_lunar_fixed_frame().items():
        pos = pos / unit_conversion
        if all(bounds_lunar["lower"][idx] <= pos[idx] <= bounds_lunar["upper"][idx] for idx in range(3)):
            ax.scatter(pos[0], pos[1], pos[2], color=textcolor, label=point, s=10)
            ax.text(pos[0], pos[1], pos[2], point, color=textcolor)


def _plot_cislunar_gcrf_xy(ax, xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    _projected_orbit(ax.scatter(xyz[:, 0], xyz[:, 1], color=scatter_dot_colors, s=1, label="Orbit Path"), xyz)
    ax.scatter(r_moon[mask_gcrf, 0], r_moon[mask_gcrf, 1], color=grey_colors, s=(_MOON_RADIUS / unit_conversion) ** 2 * 100)
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=False)


def _plot_cislunar_lunar_xy(ax, xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    _projected_orbit(ax.scatter(xyz_lunar[:, 0], xyz_lunar[:, 1], color=scatter_dot_colors, s=1, label="Orbit Path"), xyz_lunar)
    ax.scatter(r_earth_lunar[mask_lunar, 0], r_earth_lunar[mask_lunar, 1], color=blue_colors, s=(_EARTH_RADIUS / unit_conversion) ** 2 * 100)
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=False)
    for point, pos in _lagrange_points_lunar_fixed_frame().items():
        pos = pos / unit_conversion
        if bounds_lunar["lower"][0] <= pos[0] <= bounds_lunar["upper"][0] and bounds_lunar["lower"][1] <= pos[1] <= bounds_lunar["upper"][1]:
            ax.scatter(pos[0], pos[1], color=textcolor, label=point, s=20)
            ax.text(pos[0], pos[1], point, color=textcolor, fontsize=fontsize)


def _set_axis_labels(ax, unit_label, textcolor, fontsize, *, three_d):
    ax.set_xlabel(f"x [{unit_label}]", color=textcolor, fontsize=fontsize)
    ax.set_ylabel(f"y [{unit_label}]", color=textcolor, fontsize=fontsize)
    if three_d:
        ax.set_zlabel(f"z [{unit_label}]", color=textcolor, fontsize=fontsize)


def _finish_cislunar_axes(axes, bounds_gcrf, bounds_lunar, mode, title, textcolor, plotcolor, fontsize):
    if mode != "3d":
        _set_cislunar_limits(axes["gcrf"], bounds_gcrf, three_d=(mode == "combined"))
        gcrf_title = "Frame: GCRF" if title is None else f"{title}\nFrame: GCRF"
        axes["gcrf"].set_title(gcrf_title, color=textcolor, fontsize=fontsize + 2)
        axes["gcrf"].set_zorder(2)

    _set_cislunar_limits(axes["lunar"], bounds_lunar, three_d=(mode != "xy"))
    if mode == "3d":
        lunar_title = "Frame: Moon-centred Earth–Moon rotating"
        if title:
            lunar_title = f"{title}\n{lunar_title}"
        axes["lunar"].set_title(lunar_title, color=textcolor, fontsize=fontsize + 2)
    else:
        axes["lunar"].set_title("Frame: Moon-centred rotating", color=textcolor, fontsize=fontsize + 2)
        axes["lunar"].set_zorder(1)

    if mode == "combined":
        fig = axes["lunar"].figure
        fig.subplots_adjust(left=0.06, right=0.91, bottom=0.12, top=0.90, wspace=0.32)
    elif mode == "xy":
        fig = axes["lunar"].figure
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.90, wspace=0.35)

    for ax in axes.values():
        _style_cislunar_axis(ax, plotcolor, textcolor, fontsize, three_d=(mode != "xy"))


def _set_cislunar_limits(ax, bounds, *, three_d):
    ax.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax.set_ylim(bounds["lower"][1], bounds["upper"][1])
    if three_d:
        ax.set_zlim(bounds["lower"][2], bounds["upper"][2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type("ortho")
    else:
        ax.set_aspect("equal")


def _style_cislunar_axis(ax, plotcolor, textcolor, fontsize, *, three_d):
    ax.set_facecolor(plotcolor)
    ax.tick_params(axis="both", colors=textcolor, labelsize=fontsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(textcolor)
        label.set_fontsize(fontsize)
    for spine in ax.spines.values():
        spine.set_edgecolor(textcolor)

    # Fractional GEO coordinates are useful on close views; integer-only
    # locators or a minimum one-GEO tick step leave these axes nearly empty.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis) if three_d else (ax.xaxis, ax.yaxis):
        axis.set_major_locator(_MaxNLocator(nbins=4))
        if three_d:
            axis.set_pane_color((*_rgb(plotcolor), .12))
            axis._axinfo["grid"]["color"] = (*_rgb(textcolor), .18)


def _rgb(color):
    from matplotlib.colors import to_rgb
    return to_rgb(color)


def _add_cislunar_legend(ax, orbit_count, mode, legend, textcolor, plotcolor, fontsize, *, earth_visible, lagrange_visible):
    if not legend:
        return

    rainbow_line = _Line2D([0], [0], color="w", linestyle="-", linewidth=2, label="Orbit Path")
    markerfacecolor = textcolor
    markersize = 10 if mode == "xy" else 6
    legend_elements = [body_legend_handle("moon", edgecolor=textcolor)]
    if earth_visible:
        legend_elements.insert(0, body_legend_handle("earth", edgecolor=textcolor))
    if lagrange_visible:
        legend_elements.append(_Line2D([0], [0], marker="o", color="none", markerfacecolor=markerfacecolor,
                                       markersize=markersize, label="Lagrange Points"))
    if orbit_count == 1:
        legend_elements.append(rainbow_line)

    legend_size = fontsize - 4 if mode == "xy" and fontsize > 16 else fontsize if mode == "xy" else 12
    font_properties = _font_manager.FontProperties(size=legend_size)
    ax.legend(
        handles=legend_elements,
        handler_map={BodyMapPatch: BodyMapHandler(), rainbow_line: _GradientLineHandler()},
        loc="upper left",
        facecolor=plotcolor,
        edgecolor=textcolor,
        prop=font_properties,
        labelcolor=textcolor,
    )
