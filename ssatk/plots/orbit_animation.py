from __future__ import annotations

from pathlib import Path

import matplotlib.cm as _cm
import matplotlib.pyplot as _plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as _np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ssapy import get_body as _get_body
from ssapy import groundTrack as _ground_track

from ..compute import find_smallest_bounding_cube as _find_smallest_bounding_cube
from ..constants import RGEO as _RGEO
from ._frames import normalize_orbit_frame as _normalize_orbit_frame
from ._orbit_plot_core import _COORDINATE_VIEWS, _VIEW_AXES
from ._orbit_plot_core import _create_orbit_axes, _orbit_frame_transformations
from ._orbit_plot_core import _plot_settings, _set_orbit_limits, _style_orbit_axes
from .groundtrack_video import _ensure_ffmpeg_path
from .plotutils import _figure_save_path, _raise_unrecognized_kwargs, valid_orbits as _valid_orbits

_ANIMATED_VIEWS = _COORDINATE_VIEWS + ("groundtrack", "globe")


def orbit_animation(
    r,
    t=None,
    title="",
    figsize=(7, 7),
    save_path=None,
    frame="gcrf",
    show=False,
    c="black",
    pad=1,
    *,
    views=("xy", "xz", "yz", "3d"),
    lunar_transform="standard",
    tail_points=30,
    tail=None,
    fps=20,
    max_frames=240,
    dpi=100,
    bitrate=1800,
    interval=None,
    **animation_kwargs,
):
    """Save an animated orbit plot as GIF or MP4.

    This is the animation backend for :func:`ssatk.plots.orbit_plot`.
    Static image extensions still use the normal full-track plot; ``.gif`` and
    ``.mp4`` save an animation with moving heads and short fading tails.
    """
    _raise_unrecognized_kwargs(animation_kwargs, "orbit_animation")
    save_path = _animation_save_path(save_path)
    suffix = Path(save_path).suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("orbit_animation save_path must end with '.gif' or '.mp4'.")

    views = tuple(views)
    unsupported = [view for view in views if view not in _ANIMATED_VIEWS]
    if unsupported:
        raise ValueError(
            "Animated orbit_plot outputs currently support coordinate, "
            f"groundtrack, and globe views; unsupported view(s): {unsupported}"
        )

    if tail is not None:
        tail_points = tail
    tail_points = max(2, int(tail_points))
    fps = max(1, int(fps))
    max_frames = max(2, int(max_frames))
    interval = 1000.0 / fps if interval is None else interval

    r_list, t_list = _valid_orbits(r, t)
    if not r_list:
        raise ValueError("Empty trajectory; nothing to animate.")
    r_list, t_list = _downsample_tracks(r_list, t_list, max_frames=max_frames)

    if "w" in c:
        textcolor = "black"
        plotcolor = "white"
    else:
        textcolor = "white"
        plotcolor = "black"

    prepared = _prepare_animation_tracks(
        r_list,
        t_list,
        frame=frame,
        lunar_transform=lunar_transform,
        pad=pad,
    )

    fig = _plt.figure(dpi=dpi, figsize=figsize, facecolor=plotcolor)
    axes = _create_orbit_axes(fig, views)
    _draw_animation_backgrounds(axes, views, prepared, title, textcolor)
    _set_orbit_limits(axes, prepared["bounds"])
    _style_orbit_axes([axes[view] for view in views], plotcolor, textcolor)

    colors = _cm.rainbow(_np.linspace(0, 1, max(1, len(prepared["tracks"]))))
    artists = _init_animation_artists(axes, views, prepared["tracks"], colors)
    frame_count = max(len(track["xyz"]) for track in prepared["tracks"])

    def update(frame_index):
        frame_artists = []
        for track_index, track in enumerate(prepared["tracks"]):
            color = colors[track_index % len(colors)]
            idx = min(frame_index, len(track["xyz"]) - 1)
            xyz = track["xyz"]
            start = max(0, idx - tail_points + 1)
            tail_xyz = xyz[start: idx + 1]
            for view in views:
                artist_group = artists[(track_index, view)]
                _remove_collection(artist_group["tail"])
                if view in {"3d", "globe"}:
                    tail_artist = _add_3d_tail(axes[view], tail_xyz, color)
                    artist_group["head"].set_data_3d([xyz[idx, 0]], [xyz[idx, 1]], [xyz[idx, 2]])
                elif view == "groundtrack":
                    groundtrack_tail = track["groundtrack"][start: idx + 1]
                    tail_artist = _add_groundtrack_tail(axes[view], groundtrack_tail, color)
                    artist_group["head"].set_data([track["groundtrack"][idx, 0]], [track["groundtrack"][idx, 1]])
                else:
                    x_idx, y_idx, _, _ = _VIEW_AXES[view]
                    tail_artist = _add_2d_tail(axes[view], tail_xyz[:, [x_idx, y_idx]], color)
                    artist_group["head"].set_data([xyz[idx, x_idx]], [xyz[idx, y_idx]])
                artist_group["tail"] = tail_artist
                frame_artists.extend([artist_group["head"], tail_artist])

        _update_secondary_markers(axes, views, artists, prepared, frame_index, frame_artists)
        return frame_artists

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=interval,
        blit=False,
        repeat=False,
    )
    writer = _animation_writer(suffix, fps=fps, bitrate=bitrate)
    animation.save(save_path, writer=writer, dpi=dpi)
    if show:
        _plt.show()
    _plt.close(fig)
    print(f"Animation saved at: {save_path}")
    return save_path


def _animation_save_path(save_path):
    path = _figure_save_path(save_path, default_name="orbit_animation")
    if path is None:
        raise ValueError("Animated orbit_plot output requires a save path ending in .gif or .mp4.")
    return path


def _animation_writer(suffix, *, fps, bitrate):
    if suffix == ".gif":
        return PillowWriter(fps=fps)

    ffmpeg_path = _ensure_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg executable not found. Install imageio-ffmpeg, conda ffmpeg, "
            "or add ffmpeg to PATH to write MP4 orbit animations."
        )
    from matplotlib import rcParams

    rcParams["animation.ffmpeg_path"] = ffmpeg_path
    return FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )


def _downsample_tracks(r_list, t_list, *, max_frames):
    max_len = max(len(track) for track in r_list)
    if max_len <= max_frames:
        return r_list, t_list

    r_downsampled = []
    t_downsampled = []
    for track, time in zip(r_list, t_list):
        indices = _np.linspace(0, len(track) - 1, max_frames).astype(int)
        r_downsampled.append(track[indices])
        t_downsampled.append(time[indices])
    return r_downsampled, t_downsampled


def _prepare_animation_tracks(r_list, t_list, *, frame, lunar_transform, pad):
    if any(_np.max(_np.linalg.norm(xyz, axis=-1)) >= 0.95 * _RGEO for xyz in r_list):
        unit_conversion = _RGEO
        unit_label = "GEO"
    else:
        unit_conversion = 1e3
        unit_label = "km"

    frame_key = _normalize_orbit_frame(frame)
    frame_transformations = _orbit_frame_transformations(lunar_transform)
    if frame_key not in frame_transformations:
        raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

    title2, transform_func = frame_transformations[frame_key]
    bounds = {
        "lower": _np.array([_np.inf, _np.inf, _np.inf]),
        "upper": _np.array([-_np.inf, -_np.inf, -_np.inf]),
    }
    tracks = []
    for xyz_raw, t_current in zip(r_list, t_list):
        xyz = _np.asarray(xyz_raw, dtype=float)
        groundtrack = _groundtrack_degrees(xyz, t_current)
        moon_body = _get_body("moon")
        r_moon = moon_body.position(t_current).T
        r_earth = _np.zeros(_np.shape(r_moon))
        if transform_func:
            xyz = transform_func(xyz, t_current)
            r_moon = transform_func(r_moon, t_current)
            r_earth = transform_func(r_earth, t_current)

        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion
        r_earth = r_earth / unit_conversion
        lower_bound, upper_bound = _find_smallest_bounding_cube(xyz, pad=pad)
        bounds["lower"] = _np.minimum(bounds["lower"], lower_bound)
        bounds["upper"] = _np.maximum(bounds["upper"], upper_bound)
        tracks.append({"xyz": xyz, "r_moon": r_moon, "r_earth": r_earth, "groundtrack": groundtrack})

    stn = _plot_settings(frame_key, tracks[0]["r_moon"], tracks[0]["r_earth"], unit_conversion)
    return {
        "bounds": bounds,
        "tracks": tracks,
        "stn": stn,
        "unit_label": unit_label,
        "unit_conversion": unit_conversion,
        "frame_key": frame_key,
        "frame_title": title2,
    }


def _draw_animation_backgrounds(axes, views, prepared, title, textcolor):
    stn = prepared["stn"]
    unit_label = prepared["unit_label"]
    frame_key = prepared["frame_key"]
    secondary = _np.column_stack((stn["secondary_x"], stn["secondary_y"], stn["secondary_z"]))
    for view in views:
        ax = axes[view]
        if view == "groundtrack":
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude [deg]", color=textcolor)
            ax.set_ylabel("Latitude [deg]", color=textcolor)
            ax.set_title(title or "Ground Track", color=textcolor)
            ax.grid(True, alpha=0.3)
            continue

        if view in {"3d", "globe"}:
            _draw_3d_body_background(ax, stn)
            ax.scatter3D(secondary[:, 0], secondary[:, 1], secondary[:, 2], color=stn["secondary_color"], s=2, alpha=0.35)
            ax.set_xlabel(f"x [{unit_label}]", color=textcolor)
            ax.set_ylabel(f"y [{unit_label}]", color=textcolor)
            ax.set_zlabel(f"z [{unit_label}]", color=textcolor)
            ax.set_title(title or ("Globe" if view == "globe" else f"{prepared['frame_title']} orbit"), color=textcolor)
            if view == "globe":
                _set_3d_limits(ax, prepared["bounds"])
            continue

        x_idx, y_idx, x_name, y_name = _VIEW_AXES[view]
        ax.add_patch(_plt.Circle(xy=(0, 0), radius=stn["primary_size"], color=stn["primary_color"], alpha=0.3))
        ax.scatter(secondary[:, x_idx], secondary[:, y_idx], color=stn["secondary_color"], s=2, alpha=0.35)
        ax.set_xlabel(f"{x_name} [{unit_label}]", color=textcolor)
        ax.set_ylabel(f"{y_name} [{unit_label}]", color=textcolor)
        ax.set_aspect("equal")
        ax.set_title(_view_title(view, title, prepared["frame_title"], frame_key), color=textcolor)


def _draw_3d_body_background(ax, stn):
    u = _np.linspace(0, 2 * _np.pi, 36)
    v = _np.linspace(-_np.pi / 2, _np.pi / 2, 18)
    mesh_x = _np.outer(_np.cos(u), _np.cos(v)).T * stn["primary_size"]
    mesh_y = _np.outer(_np.sin(u), _np.cos(v)).T * stn["primary_size"]
    mesh_z = _np.outer(_np.ones(_np.size(u)), _np.sin(v)).T * stn["primary_size"]
    ax.plot_surface(mesh_x, mesh_y, mesh_z, color=stn["primary_color"], alpha=0.3, edgecolor="none")


def _view_title(view, title, frame_title, frame_key):
    if view == "xy" and title:
        return f"{title}\n{frame_title}"
    if view == "xy":
        return frame_title
    if view == "xz":
        return "X-Z"
    if view == "yz":
        return "Y-Z" if "lunar" not in frame_key else "Y-Z lunar plane"
    return frame_title


def _init_animation_artists(axes, views, tracks, colors):
    artists = {}
    for track_index, track in enumerate(tracks):
        color = colors[track_index % len(colors)]
        xyz = track["xyz"]
        for view in views:
            ax = axes[view]
            if view in {"3d", "globe"}:
                head = ax.plot([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], marker="o", linestyle="None", color=color, markersize=5)[0]
            elif view == "groundtrack":
                lonlat = track["groundtrack"]
                head = ax.plot([lonlat[0, 0]], [lonlat[0, 1]], marker="o", linestyle="None", color=color, markersize=5)[0]
            else:
                x_idx, y_idx, _, _ = _VIEW_AXES[view]
                head = ax.plot([xyz[0, x_idx]], [xyz[0, y_idx]], marker="o", linestyle="None", color=color, markersize=5)[0]
            artists[(track_index, view)] = {"head": head, "tail": None}
    return artists


def _update_secondary_markers(axes, views, artists, prepared, frame_index, frame_artists):
    if "secondary" not in artists:
        artists["secondary"] = {}
        secondary = _np.column_stack((prepared["stn"]["secondary_x"], prepared["stn"]["secondary_y"], prepared["stn"]["secondary_z"]))
        for view in views:
            if view == "groundtrack":
                continue
            ax = axes[view]
            if view in {"3d", "globe"}:
                marker = ax.plot([secondary[0, 0]], [secondary[0, 1]], [secondary[0, 2]], marker="o", linestyle="None", color="grey", markersize=4)[0]
            else:
                x_idx, y_idx, _, _ = _VIEW_AXES[view]
                marker = ax.plot([secondary[0, x_idx]], [secondary[0, y_idx]], marker="o", linestyle="None", color="grey", markersize=4)[0]
            artists["secondary"][view] = marker

    secondary = _np.column_stack((prepared["stn"]["secondary_x"], prepared["stn"]["secondary_y"], prepared["stn"]["secondary_z"]))
    idx = min(frame_index, len(secondary) - 1)
    for view in views:
        if view == "groundtrack":
            continue
        marker = artists["secondary"][view]
        if view in {"3d", "globe"}:
            marker.set_data_3d([secondary[idx, 0]], [secondary[idx, 1]], [secondary[idx, 2]])
        else:
            x_idx, y_idx, _, _ = _VIEW_AXES[view]
            marker.set_data([secondary[idx, x_idx]], [secondary[idx, y_idx]])
        frame_artists.append(marker)


def _add_2d_tail(ax, xy, color):
    if len(xy) < 2:
        collection = LineCollection([], colors=[])
    else:
        segments = _np.stack([xy[:-1], xy[1:]], axis=1)
        colors = _tail_colors(color, len(segments))
        collection = LineCollection(segments, colors=colors, linewidths=2.0)
    ax.add_collection(collection)
    return collection


def _add_3d_tail(ax, xyz, color):
    if len(xyz) < 2:
        point = xyz[0] if len(xyz) else _np.zeros(3)
        collection = Line3DCollection([[point, point]], colors=[_tail_colors(color, 1)[0]])
    else:
        segments = _np.stack([xyz[:-1], xyz[1:]], axis=1)
        colors = _tail_colors(color, len(segments))
        collection = Line3DCollection(segments, colors=colors, linewidths=2.0)
    ax.add_collection3d(collection)
    return collection


def _add_groundtrack_tail(ax, lonlat, color):
    if len(lonlat) < 2:
        collection = LineCollection([], colors=[])
    else:
        start = lonlat[:-1]
        stop = lonlat[1:]
        keep = _np.abs(stop[:, 0] - start[:, 0]) <= 180.0
        segments = _np.stack([start[keep], stop[keep]], axis=1) if _np.any(keep) else _np.empty((0, 2, 2))
        collection = LineCollection(segments, colors=_tail_colors(color, len(segments)), linewidths=2.0)
    ax.add_collection(collection)
    return collection


def _tail_colors(color, count):
    rgba = _np.asarray(color, dtype=float)
    if rgba.size == 3:
        rgba = _np.append(rgba, 1.0)
    colors = _np.tile(rgba, (count, 1))
    colors[:, 3] = _np.linspace(0.08, 0.95, count)
    return colors


def _remove_collection(collection):
    if collection is not None:
        collection.remove()


def _groundtrack_degrees(r, t):
    lon, lat, _height = _ground_track(_np.asarray(r), t, format="geodetic")
    lon_deg = ((_np.degrees(lon) + 180.0) % 360.0) - 180.0
    lat_deg = _np.degrees(lat)
    return _np.column_stack((lon_deg, lat_deg))


def _set_3d_limits(ax, bounds):
    ax.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax.set_ylim(bounds["lower"][1], bounds["upper"][1])
    ax.set_zlim(bounds["lower"][2], bounds["upper"][2])
    ax.set_box_aspect([1, 1, 1])
