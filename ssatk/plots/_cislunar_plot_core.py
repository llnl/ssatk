from __future__ import annotations

from ssapy import get_body as _get_body
import matplotlib.cm as _cm
import matplotlib.font_manager as _font_manager
import matplotlib.pyplot as _plt
from matplotlib.lines import Line2D as _Line2D
from matplotlib.patches import Patch as _Patch
from matplotlib.ticker import MaxNLocator as _MaxNLocator
from matplotlib.ticker import MultipleLocator as _MultipleLocator
import numpy as _np

from ..compute import find_smallest_bounding_cube as _find_smallest_bounding_cube
from ..constants import EARTH_RADIUS as _EARTH_RADIUS
from ..constants import MOON_RADIUS as _MOON_RADIUS
from ..constants import RGEO as _RGEO
from ..coordinates import gcrf_to_lunar_fixed as _gcrf_to_lunar_fixed
from ._legend_handlers import GradientLineHandler as _GradientLineHandler
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
):
    mode = mode.lower()
    if mode not in {"combined", "3d", "xy"}:
        raise ValueError("mode must be one of: combined, 3d, xy")

    r, t = _valid_orbits(r, t)
    textcolor, plotcolor = _plot_colors(c)
    fig, axes = _create_cislunar_axes(mode, figsize, plotcolor)

    bounds_gcrf = {
        "lower": _np.array([_np.inf, _np.inf, _np.inf]),
        "upper": _np.array([-_np.inf, -_np.inf, -_np.inf]),
    }
    bounds_lunar = {
        "lower": _np.array([_np.inf, _np.inf, _np.inf]),
        "upper": _np.array([-_np.inf, -_np.inf, -_np.inf]),
    }

    unit_label = "km"
    unit_conversion = 1e3
    for orbit_index, xyz_raw in enumerate(r):
        xyz = xyz_raw
        t_current = t[orbit_index]
        moon_body = _get_body("moon")
        r_moon = moon_body.position(t_current).T
        r_earth = _np.zeros(_np.shape(r_moon))

        if max(_np.linalg.norm(xyz, axis=-1) >= 0.95 * _RGEO):
            unit_conversion = _RGEO
            unit_label = "GEO"
        else:
            unit_conversion = 1e3
            unit_label = "km"

        xyz_lunar = _gcrf_to_lunar_fixed(xyz, t_current) / unit_conversion
        r_earth_lunar = _gcrf_to_lunar_fixed(r_earth, t_current) / unit_conversion
        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion

        if mode != "3d":
            lower_bound_temp, upper_bound_temp = _find_smallest_bounding_cube(xyz, pad=1)
            bounds_gcrf["lower"] = _np.minimum(bounds_gcrf["lower"], lower_bound_temp)
            bounds_gcrf["upper"] = _np.maximum(bounds_gcrf["upper"], upper_bound_temp)

        lower_bound_lunar_temp, upper_bound_lunar_temp = _find_smallest_bounding_cube(xyz_lunar, pad=1)
        bounds_lunar["lower"] = _np.minimum(bounds_lunar["lower"], lower_bound_lunar_temp)
        bounds_lunar["upper"] = _np.maximum(bounds_lunar["upper"], upper_bound_lunar_temp)

        mask_gcrf = _bounds_mask(r_moon, bounds_gcrf, dimensions=3 if mode == "combined" else 2)
        mask_lunar = _bounds_mask(r_earth_lunar, bounds_lunar, dimensions=3 if mode != "xy" else 2)
        grey_colors, blue_colors = _body_colors(r_moon, mask_gcrf, mask_lunar, mode)
        scatter_dot_colors = _scatter_colors(r, xyz, orbit_index)

        if mode == "combined":
            _plot_cislunar_gcrf_3d(axes["gcrf"], xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize)
            _plot_cislunar_lunar_3d(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize)
        elif mode == "xy":
            _plot_cislunar_gcrf_xy(axes["gcrf"], xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize)
            _plot_cislunar_lunar_xy(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize)
        else:
            _plot_cislunar_lunar_3d(axes["lunar"], xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize)

    _finish_cislunar_axes(axes, bounds_gcrf, bounds_lunar, mode, title, textcolor, plotcolor, fontsize)
    _add_cislunar_legend(axes["lunar"], len(r), mode, legend, textcolor, plotcolor, fontsize)

    if save_path:
        _figsave(fig, save_path)
    if show:
        _plt.show()
    _plt.close()

    if mode == "3d":
        return fig, axes["lunar"]
    return fig, [axes["gcrf"], axes["lunar"]]


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


def _sphere_mesh(radius):
    u = _np.linspace(0, 2 * _np.pi, 180)
    v = _np.linspace(-_np.pi / 2, _np.pi / 2, 180)
    mesh_x = _np.outer(_np.cos(u), _np.cos(v)).T * radius
    mesh_y = _np.outer(_np.sin(u), _np.cos(v)).T * radius
    mesh_z = _np.outer(_np.ones(_np.size(u)), _np.sin(v)).T * radius
    return mesh_x, mesh_y, mesh_z


def _plot_cislunar_gcrf_3d(ax, xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1)
    ax.plot_surface(*_sphere_mesh(_EARTH_RADIUS / unit_conversion), color="blue", alpha=0.6, edgecolor="none")
    ax.scatter3D(r_moon[mask_gcrf, 0], r_moon[mask_gcrf, 1], r_moon[mask_gcrf, 2], color=grey_colors, s=(_MOON_RADIUS / unit_conversion))
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=True)


def _plot_cislunar_lunar_3d(ax, xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    ax.scatter3D(xyz_lunar[:, 0], xyz_lunar[:, 1], xyz_lunar[:, 2], color=scatter_dot_colors, s=1)
    ax.plot_surface(*_sphere_mesh(_MOON_RADIUS / unit_conversion), color="grey", alpha=0.6, edgecolor="none")
    ax.scatter3D(r_earth_lunar[mask_lunar, 0], r_earth_lunar[mask_lunar, 1], r_earth_lunar[mask_lunar, 2], color=blue_colors, s=55)
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=True)
    for point, pos in _lagrange_points_lunar_fixed_frame().items():
        pos = pos / unit_conversion
        if all(bounds_lunar["lower"][idx] <= pos[idx] <= bounds_lunar["upper"][idx] for idx in range(3)):
            ax.scatter(pos[0], pos[1], pos[2], color=textcolor, label=point, s=10)
            ax.text(pos[0], pos[1], pos[2], point, color=textcolor)


def _plot_cislunar_gcrf_xy(ax, xyz, r_moon, mask_gcrf, grey_colors, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    earth_circle = _plt.Circle((0, 0), _EARTH_RADIUS / unit_conversion, color="lightblue", alpha=0.8, linestyle="dashed", fill=True)
    ax.scatter(xyz[:, 0], xyz[:, 1], color=scatter_dot_colors, s=1)
    ax.add_patch(earth_circle)
    ax.scatter(r_moon[mask_gcrf, 0], r_moon[mask_gcrf, 1], color=grey_colors, s=(_MOON_RADIUS / unit_conversion) ** 2 * 100)
    _set_axis_labels(ax, unit_label, textcolor, fontsize, three_d=False)


def _plot_cislunar_lunar_xy(ax, xyz_lunar, r_earth_lunar, mask_lunar, blue_colors, bounds_lunar, unit_conversion, unit_label, scatter_dot_colors, textcolor, fontsize):
    moon_circle = _plt.Circle((0, 0), _MOON_RADIUS / unit_conversion, color="lightgrey", alpha=0.8, linestyle="dashed", fill=True)
    ax.scatter(xyz_lunar[:, 0], xyz_lunar[:, 1], color=scatter_dot_colors, s=1)
    ax.add_patch(moon_circle)
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
        axes["lunar"].set_title(f"Frame: Lunar Fixed\n{title}", color=textcolor, fontsize=fontsize + 2)
    else:
        axes["lunar"].set_title("Frame: Lunar Fixed", color=textcolor, fontsize=fontsize + 2)
        axes["lunar"].set_zorder(1)

    if mode == "combined":
        _plt.subplots_adjust(left=0.0, right=1.5, bottom=0.05, top=0.95, wspace=-0.0)
    elif mode == "xy":
        _plt.subplots_adjust(left=0.0, right=1.5, bottom=0.05, top=0.95, wspace=0.2)

    for ax in axes.values():
        _style_cislunar_axis(ax, plotcolor, textcolor, fontsize, three_d=(mode != "xy"))


def _set_cislunar_limits(ax, bounds, *, three_d):
    ax.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax.set_ylim(bounds["lower"][1], bounds["upper"][1])
    if three_d:
        ax.set_zlim(bounds["lower"][2], bounds["upper"][2])
        ax.set_box_aspect([1, 1, 1])
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

    if three_d:
        ax.xaxis.set_major_locator(_MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(_MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(_MaxNLocator(integer=True))
    else:
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.xaxis.set_major_locator(_MultipleLocator(base=max(_np.ceil(x_range / 6), 1)))
        ax.yaxis.set_major_locator(_MultipleLocator(base=max(_np.ceil(y_range / 6), 1)))


def _add_cislunar_legend(ax, orbit_count, mode, legend, textcolor, plotcolor, fontsize):
    if mode == "3d" and not legend:
        return

    rainbow_line = _Line2D([0], [0], color="w", linestyle="-", linewidth=2, label="Orbit Path")
    markerfacecolor = "grey" if mode == "xy" else "black"
    markersize = 10 if mode == "xy" else 6
    legend_elements = [
        _Patch(facecolor="lightblue", edgecolor=textcolor, label="Earth"),
        _Patch(facecolor="lightgrey", edgecolor=textcolor, label="Moon"),
        _Line2D([0], [0], marker="o", color="none", markerfacecolor=markerfacecolor, markersize=markersize, label="Lagrange Points"),
    ]
    if orbit_count == 1:
        legend_elements.append(rainbow_line)

    legend_size = fontsize - 4 if mode == "xy" and fontsize > 16 else fontsize if mode == "xy" else 12
    font_properties = _font_manager.FontProperties(size=legend_size)
    ax.legend(
        handles=legend_elements,
        handler_map={rainbow_line: _GradientLineHandler()} if orbit_count == 1 else {},
        loc="upper left",
        facecolor=plotcolor,
        edgecolor=textcolor,
        prop=font_properties,
        labelcolor=textcolor,
    )
