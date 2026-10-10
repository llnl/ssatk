from __future__ import annotations

from collections.abc import Iterable as _Iterable
from pathlib import Path

from ._cislunar_plot_core import _cislunar_plot_core
from ._orbit_plot_core import _orbit_plot_core
from .orbit_animation import orbit_animation as _orbit_animation_core
from .plotutils import _pop_save_path_aliases


_DEFAULT_FIGSIZE = (7, 7)
_ORBIT_DASHBOARD_FIGSIZE = (16, 12)
_CISLUNAR_DASHBOARD_FIGSIZE = (12, 7)
_FULL_ORBIT_VIEWS = ("xy", "xz", "yz", "3d")
_DASHBOARD_ORBIT_VIEWS = ("groundtrack", "globe", "xy", "xz", "yz", "3d")
_ORBIT_DASHBOARD_ALIAS_KEYS = {
    "dashboard",
    "dashboard_plot",
    "plot_dashboard",
    "orbit_dashboard",
    "orbit_plot_dashboard",
    "orbitplot_dashboard",
    "orbit_dashboard_plot",
    "trajectory_dashboard",
    "space_dashboard",
    "in_space_dashboard",
}
_CISLUNAR_DASHBOARD_ALIAS_KEYS = {
    "cislunar_dashboard",
    "cislunar_dashboard_plot",
    "cislunar_plot_dashboard",
    "cislunardashboard",
    "cislunar_dash",
}
_ORBIT_VIEW_ALIASES = {
    "default": _FULL_ORBIT_VIEWS,
    "full": _FULL_ORBIT_VIEWS,
    "all": _FULL_ORBIT_VIEWS,
    "orbit": _FULL_ORBIT_VIEWS,
    "orbit_full": _FULL_ORBIT_VIEWS,
    "cartesian": _FULL_ORBIT_VIEWS,
    "slices_3d": _FULL_ORBIT_VIEWS,
    "xy": ("xy",),
    "orbit_xy": ("xy",),
    "xz": ("xz",),
    "orbit_xz": ("xz",),
    "yz": ("yz",),
    "orbit_yz": ("yz",),
    "3d": ("3d",),
    "orbit_3d": ("3d",),
    "xyz": ("3d",),
    "xyxz": ("xy", "xz"),
    "xy_xz": ("xy", "xz"),
    "orbit_xyxz": ("xy", "xz"),
    "xy_yz": ("xy", "yz"),
    "xyyz": ("xy", "yz"),
    "xz_yz": ("xz", "yz"),
    "xzyz": ("xz", "yz"),
    "2d": ("xy", "xz", "yz"),
    "slices": ("xy", "xz", "yz"),
    "projections": ("xy", "xz", "yz"),
    **{key: _DASHBOARD_ORBIT_VIEWS for key in _ORBIT_DASHBOARD_ALIAS_KEYS},
}
_LUNAR_VIEW_ALIASES = {
    "lunar": _FULL_ORBIT_VIEWS,
    "lunar_full": _FULL_ORBIT_VIEWS,
    "lunar_all": _FULL_ORBIT_VIEWS,
    "lunar_orbit": _FULL_ORBIT_VIEWS,
    "lunar_xy": ("xy",),
    "lunar_xz": ("xz",),
    "lunar_yz": ("yz",),
    "lunar_3d": ("3d",),
    "lunar_xyz": ("3d",),
    "lunar_xyxz": ("xy", "xz"),
    "lunar_xy_xz": ("xy", "xz"),
    "lunar_xy_yz": ("xy", "yz"),
    "lunar_xz_yz": ("xz", "yz"),
    "lunar_2d": ("xy", "xz", "yz"),
    "lunar_slices": ("xy", "xz", "yz"),
    "lunar_projections": ("xy", "xz", "yz"),
}
_MAP_VIEW_ALIASES = {
    "ground_track": ("groundtrack",),
    "groundtrack": ("groundtrack",),
    "groundtrack_plot": ("groundtrack",),
    "groundtrackplot": ("groundtrack",),
    "ground_track_plot": ("groundtrack",),
    "globe": ("globe",),
    "globe_plot": ("globe",),
    "globeplot": ("globe",),
}
_CISLUNAR_VIEW_ALIASES = {
    "cislunar": "combined",
    "cislunar_combined": "combined",
    "cislunar_full": "combined",
    "cislunar_3d": "3d",
    "cislunar3d": "3d",
    "cislunar_xyz": "3d",
    "cislunar_xy": "xy",
    "cislunar_2d": "xy",
    **{key: "combined" for key in _CISLUNAR_DASHBOARD_ALIAS_KEYS},
}
_TRANSFER_VIEW_ALIASES = {
    "transfer": "auto",
    "transfer_plot": "auto",
    "transferplot": "auto",
    "transfer_trajectory": "trajectory",
    "transfertrajectory": "trajectory",
    "trajectory_transfer": "trajectory",
    "transfer_arc": "trajectory",
    "transfer_trajectory_2d": "trajectory",
    "transfer_arc_2d": "trajectory",
    "transfer_trajectory_3d": "trajectory_3d",
    "transfer_3d_trajectory": "trajectory_3d",
    "transfer_arc_3d": "trajectory_3d",
    "transfer_burn": "burn_profile",
    "transfer_burns": "burn_profile",
    "transfer_burn_profile": "burn_profile",
    "transfer_burn_profile_plot": "burn_profile",
    "burn_profile": "burn_profile",
    "burn_profile_plot": "burn_profile",
    "burn_timeline": "burn_profile",
    "transfer_timeline": "burn_profile",
    "transfer_designer": "designer",
    "transfer_designer_plot": "designer",
    "transfer_designer_curves": "designer",
    "transfer_designer_curves_plot": "designer",
    "designer": "designer",
    "designer_curves": "designer",
    "porkchop": "designer",
    "porkchop_plot": "designer",
}
_DIVERGENCE_VIEW_ALIASES = {
    "divergence": "divergence",
    "divergence_plot": "divergence",
    "position_divergence": "divergence",
    "velocity_plane_divergence": "divergence",
    "ntw_divergence": "divergence",
    "orbit_divergence": "orbit_divergence",
    "orbit_divergence_plot": "orbit_divergence",
    "cislunar_divergence": "orbit_divergence",
}


def orbit_plot(
    r,
    t=None,
    title="",
    figsize=(7, 7),
    save_path=False,
    frame=None,
    show=False,
    c="black",
    pad=1,
    *,
    view=None,
    fontsize=12,
    legend=True,
    coordinate=None,
    coordinates=None,
    lunar_transform=None,
    **plot_kwargs,
):
    """Plot an orbit or cislunar trajectory through one stable entry point.

    Parameters
    ----------
    r : ssapy.Orbit or array-like
        SSAPy ``Orbit``/``Orbit.at`` object, raw SSAPy position output in
        metres, or a list/batch of position tracks.  If an ``Orbit`` is passed,
        ``t`` may optionally specify the GPS seconds/Astropy Time samples to
        plot; otherwise one orbital period is sampled when the object exposes a
        finite ``period``.
    view : str or iterable of str, optional
        Plot view selector. A single string may be a compact selector such as
        ``"xy"``, ``"xz"``, ``"yz"``, ``"3d"``, ``"xyxz"``,
        ``"lunar_xy"``, ``"lunar_yz"``, ``"full"``, ``"cislunar"``,
        ``"cislunar_3d"``, ``"cislunar_xy"``, ``"dashboard"``,
        ``"cislunar_dashboard"``, ``"groundtrack"``, ``"globe"``,
        ``"transfer_trajectory"``, ``"transfer_burn_profile"``,
        ``"transfer_designer"``, or ``"divergence"``. Multiple standard orbit
        views can be provided as an iterable, for example
        ``("xy", "xz", "3d")``. Multiple orbit views are placed in order on
        an automatic grid: 2x2, 2x3, 3x3, etc.
        When ``save_path`` or a save alias ends in ``.mp4`` or ``.gif``,
        coordinate views are saved as an animated orbit with a short fading
        tail. Static image extensions such as ``.png`` and ``.jpg`` save the
        full time-series figure.
    frame, coordinate, coordinates : str, optional
        Coordinate-frame aliases for non-cislunar orbit views. Use one of these
        synonyms per call. Supported aliases include GCRF/GCRS, ITRF/ITRS,
        lunar, lunar fixed, and lunar axis. ``lunar_*`` view selectors default
        to ``coordinate="lunar_fixed"`` unless this is explicitly overwritten.
    lunar_transform : {"standard", "fixed"}, optional
        Override how lunar-frame orbit views are transformed. If omitted,
        ``frame="lunar_fixed"`` and similar fixed-frame aliases use the fixed
        transform; other lunar aliases preserve the historical transform.
    """

    save_path, plot_kwargs = _pop_save_path_aliases(plot_kwargs, save_path=save_path)

    plot_family, target, default_coordinate = _resolve_view_selector(view)
    if figsize == _DEFAULT_FIGSIZE:
        if plot_family == "orbit" and _is_single_alias(view, _ORBIT_DASHBOARD_ALIAS_KEYS):
            figsize = _ORBIT_DASHBOARD_FIGSIZE
        elif plot_family == "cislunar" and _is_single_alias(view, _CISLUNAR_DASHBOARD_ALIAS_KEYS):
            figsize = _CISLUNAR_DASHBOARD_FIGSIZE

    if plot_family == "cislunar":
        if plot_kwargs:
            unknown = ", ".join(sorted(plot_kwargs))
            raise TypeError(f"Unsupported cislunar orbit_plot keyword(s): {unknown}")
        return _cislunar_plot_core(
            r,
            t=t,
            figsize=figsize,
            fontsize=fontsize,
            save_path=save_path,
            show=show,
            title=title,
            c=c,
            mode=target,
            legend=legend,
        )

    if plot_family == "transfer":
        return _dispatch_transfer_view(
            target,
            r,
            title=title,
            save_path=save_path,
            show=show,
            plot_kwargs=plot_kwargs,
        )

    if plot_family == "divergence":
        return _dispatch_divergence_view(
            target,
            r,
            t=t,
            title=title,
            save_path=save_path,
            show=show,
            plot_kwargs=plot_kwargs,
        )

    coordinate_frame = _resolve_coordinate_frame(
        frame=frame,
        coordinate=coordinate,
        coordinates=coordinates,
        default_coordinate=default_coordinate,
    )
    if _is_animation_save_path(save_path):
        return _orbit_animation_core(
            r,
            t=t,
            title=title,
            figsize=figsize,
            save_path=save_path,
            frame=coordinate_frame,
            show=show,
            c=c,
            pad=pad,
            views=target,
            lunar_transform=_resolve_lunar_transform(coordinate_frame, lunar_transform),
            **plot_kwargs,
        )

    return _orbit_plot_core(
        r,
        t=t,
        title=title,
        figsize=figsize,
        save_path=save_path,
        frame=coordinate_frame,
        show=show,
        c=c,
        pad=pad,
        views=target,
        lunar_transform=_resolve_lunar_transform(coordinate_frame, lunar_transform),
        special_plot_kwargs={"fontsize": fontsize, "show_legend": legend, **plot_kwargs},
    )


def _resolve_view_selector(view):
    if view is None:
        return "orbit", _FULL_ORBIT_VIEWS, None

    if isinstance(view, str):
        key = _normalize_selector_key(view)
        if key in _CISLUNAR_VIEW_ALIASES:
            return "cislunar", _CISLUNAR_VIEW_ALIASES[key], None
        if key in _TRANSFER_VIEW_ALIASES:
            return "transfer", _TRANSFER_VIEW_ALIASES[key], None
        if key in _DIVERGENCE_VIEW_ALIASES:
            return "divergence", _DIVERGENCE_VIEW_ALIASES[key], None
        if key in _LUNAR_VIEW_ALIASES:
            return "orbit", _LUNAR_VIEW_ALIASES[key], "lunar_fixed"
        if key in _MAP_VIEW_ALIASES:
            return "orbit", _MAP_VIEW_ALIASES[key], None
        if key in _ORBIT_VIEW_ALIASES:
            return "orbit", _ORBIT_VIEW_ALIASES[key], None
        if any(separator in view for separator in (",", "+", "/")):
            views, default_coordinate = _normalize_views(view)
            return "orbit", views, default_coordinate
        valid = sorted(
            set(_ORBIT_VIEW_ALIASES)
            | set(_LUNAR_VIEW_ALIASES)
            | set(_MAP_VIEW_ALIASES)
            | set(_CISLUNAR_VIEW_ALIASES)
            | set(_TRANSFER_VIEW_ALIASES)
            | set(_DIVERGENCE_VIEW_ALIASES)
        )
        raise ValueError(f"Unknown orbit_plot view {view!r}. Supported views include: {', '.join(valid)}")

    if isinstance(view, _Iterable):
        view_items = list(view)
        if len(view_items) == 1:
            key = _normalize_selector_key(view_items[0])
            if key in _CISLUNAR_VIEW_ALIASES:
                return "cislunar", _CISLUNAR_VIEW_ALIASES[key], None
            if key in _TRANSFER_VIEW_ALIASES:
                return "transfer", _TRANSFER_VIEW_ALIASES[key], None
            if key in _DIVERGENCE_VIEW_ALIASES:
                return "divergence", _DIVERGENCE_VIEW_ALIASES[key], None
        views, default_coordinate = _normalize_views(view_items)
        return "orbit", views, default_coordinate

    raise TypeError("view must be a string or iterable of strings")


def _is_animation_save_path(save_path):
    if save_path in (None, False, True):
        return False
    return Path(str(save_path)).suffix.lower() in {".gif", ".mp4"}


def _dispatch_transfer_view(target, r, *, title, save_path, show, plot_kwargs):
    if target == "auto":
        if not _looks_like_transfer_result(r):
            raise TypeError("view='transfer' requires a transfer result with burns or trajectory data.")
        target = "trajectory"

    if target in {"trajectory", "trajectory_3d"}:
        from .transfer_trajectory_plot import transfer_trajectory_plot

        plot_kwargs.setdefault("three_d", target == "trajectory_3d")
        ax = transfer_trajectory_plot(
            r,
            title=title or None,
            save_path=save_path,
            **plot_kwargs,
        )
        _show_if_requested(show, ax)
        return ax

    if target == "burn_profile":
        from .transfer_burn_profile_plot import transfer_burn_profile_plot

        fig = transfer_burn_profile_plot(
            r,
            title=title or None,
            save_path=save_path,
            **plot_kwargs,
        )
        _show_if_requested(show, fig)
        return fig

    if target == "designer":
        from .transfer_designer_curves_plot import transfer_designer_curves_plot

        fig = transfer_designer_curves_plot(
            r,
            title=title or None,
            save_path=save_path,
            **plot_kwargs,
        )
        _show_if_requested(show, fig)
        return fig

    raise ValueError(f"Unknown transfer orbit_plot target {target!r}")


def _dispatch_divergence_view(target, r, *, t, title, save_path, show, plot_kwargs):
    if target == "divergence":
        from .divergence_plot import divergence_plot

        return divergence_plot(
            r,
            title=title or "Position Errors Projected onto Velocity Plane",
            save_path=save_path,
            show=show,
            **plot_kwargs,
        )

    if target == "orbit_divergence":
        from .misc_plotting import orbit_divergence_plot

        return orbit_divergence_plot(
            r,
            t=t,
            title=title,
            save_path=save_path,
            show=show,
            **plot_kwargs,
        )

    raise ValueError(f"Unknown divergence orbit_plot target {target!r}")


def _looks_like_transfer_result(value):
    transfer = getattr(value, "transfer", value)
    return hasattr(transfer, "burns") or hasattr(transfer, "trajectory") or hasattr(transfer, "transfer_orbit")


def _show_if_requested(show, artist):
    if show and artist is not None:
        import matplotlib.pyplot as plt

        plt.show()


def _normalize_views(views):
    if isinstance(views, str):
        pieces = views.replace("+", ",").replace("/", ",").split(",")
    elif isinstance(views, _Iterable):
        pieces = list(views)
    else:
        raise TypeError("view must be a string or iterable of strings")

    normalized = []
    default_coordinate = None
    for piece in pieces:
        if not str(piece).strip():
            continue
        piece_views, piece_default_coordinate = _normalize_view_piece(piece)
        normalized.extend(piece_views)
        if piece_default_coordinate is not None:
            default_coordinate = piece_default_coordinate
    if not normalized:
        raise ValueError("view must contain at least one view")
    return tuple(normalized), default_coordinate


def _normalize_view_piece(view):
    key = _normalize_selector_key(view)
    if key in _LUNAR_VIEW_ALIASES:
        return _LUNAR_VIEW_ALIASES[key], "lunar_fixed"
    if key in _MAP_VIEW_ALIASES:
        return _MAP_VIEW_ALIASES[key], None
    if key in _ORBIT_VIEW_ALIASES:
        return _ORBIT_VIEW_ALIASES[key], None
    if key in {"xy", "xz", "yz", "3d"}:
        return (key,), None
    if key == "xyz":
        return ("3d",), None
    raise ValueError("multiple view entries must be one of: xy, xz, yz, 3d, or lunar_* aliases")


def _normalize_selector_key(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_single_alias(view, aliases):
    if isinstance(view, str):
        return _normalize_selector_key(view) in aliases
    if isinstance(view, _Iterable):
        view_items = list(view)
        return len(view_items) == 1 and _normalize_selector_key(view_items[0]) in aliases
    return False


def _resolve_coordinate_frame(*, frame, coordinate, coordinates, default_coordinate):
    provided = [value for value in (frame, coordinate, coordinates) if value not in (None, "")]
    if not provided:
        return default_coordinate or "gcrf"

    normalized = {_normalize_selector_key(value) for value in provided}
    if len(normalized) > 1:
        raise ValueError("Use only one of frame=, coordinate=, or coordinates= per orbit_plot call.")
    return str(provided[0])


def _resolve_lunar_transform(frame, lunar_transform):
    if lunar_transform is not None:
        key = _normalize_selector_key(lunar_transform)
        if key not in {"standard", "fixed"}:
            raise ValueError("lunar_transform must be 'standard' or 'fixed'")
        return key

    frame_key = _normalize_selector_key(frame)
    if "fixed" in frame_key:
        return "fixed"
    return "standard"
