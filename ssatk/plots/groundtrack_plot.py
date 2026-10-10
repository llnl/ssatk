"""
Ground-track plotting with configurable map center longitude.

Features
--------
- Single or multiple orbit ground tracks
- Arbitrary map center longitude (e.g. Pacific-centered at 180 deg)
- Earth background image shifting
- Ground station plotting
- Wrap-safe orbit lines
- Longitude tick labels formatted as 120W, 60E, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

from ssapy import groundTrack
from ._groundtrack_helpers import as_list as _as_list
from ._groundtrack_helpers import broadcast_time_list as _broadcast_time_list
from ._groundtrack_helpers import clean_lonlat_wrap as _clean_lonlat_wrap
from .plotutils import load_earth_file, figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs


def _wrap_longitudes(lon_deg, central_longitude=0.0):
    """
    Wrap longitudes into the display interval:
        [central_longitude - 180, central_longitude + 180)

    Examples
    --------
    central_longitude = 0   -> [-180, 180)
    central_longitude = 180 -> [0, 360)
    """
    lon_deg = np.asarray(lon_deg, dtype=float)
    return ((lon_deg - central_longitude + 180.0) % 360.0) - 180.0 + central_longitude


def _display_to_standard_lon(lon_disp):
    """
    Convert displayed longitudes back to conventional [-180, 180) values.
    """
    lon_disp = np.asarray(lon_disp, dtype=float)
    return ((lon_disp + 180.0) % 360.0) - 180.0


def _normalize_central_longitude(central_longitude):
    """Normalize longitude into [0, 360)."""
    return float(central_longitude) % 360.0


def _to_numpy_image(img):
    """
    Convert a PIL image or array-like image into a NumPy array.
    """
    return np.asarray(img)


def _shift_earth_image(img, central_longitude):
    """
    Shift an equirectangular Earth image by splitting and re-stitching.

    Assumes the source image spans longitudes [-180, 180), with longitude
    increasing left-to-right.
    """
    img = _to_numpy_image(img)
    cl = _normalize_central_longitude(central_longitude)

    if np.isclose(cl, 0.0):
        return img

    ncols = img.shape[1]

    # Desired displayed longitude span is [cl - 180, cl + 180)
    new_left_lon = cl - 180.0

    # Convert left edge into equivalent standard longitude in [-180, 180)
    new_left_lon_std = ((new_left_lon + 180.0) % 360.0) - 180.0

    # Convert longitude to image column
    split_col = int(round(((new_left_lon_std + 180.0) / 360.0) * ncols)) % ncols

    # Re-stitch image so the new left edge starts at column 0
    shifted_img = np.concatenate((img[:, split_col:], img[:, :split_col]), axis=1)
    return shifted_img


def _format_lon_label(lon_deg):
    """
    Format longitude in degrees as cardinal text.

    Examples
    --------
    -180 -> 180
    -120 -> 120W
     -60 -> 60W
       0 -> 0
      60 -> 60E
     120 -> 120E
     180 -> 180
    """
    lon = int(round(lon_deg))

    if lon == 0:
        return "0"
    if abs(lon) == 180:
        return "180"
    if lon < 0:
        return f"{abs(lon)}W"
    return f"{lon}E"


def _set_longitude_ticks(ax, xmin, xmax, relabel_xticks=True):
    """
    Set x-axis longitude ticks and cardinal-direction labels.
    """
    xticks = np.arange(xmin, xmax + 1e-9, 60.0)
    ax.set_xticks(xticks)

    if relabel_xticks:
        tickvals = _display_to_standard_lon(xticks)
    else:
        tickvals = xticks

    ax.set_xticklabels([_format_lon_label(v) for v in tickvals])


def groundtrack_plot(
    r,
    t,
    ground_stations=None,
    save_path=None,
    title="Ground Track",
    show_legend=True,
    fontsize=18,
    start_end_markers=True,
    labels=None,
    orbit_colors=None,
    linestyles=None,
    legend_kwargs=None,
    central_longitude=0.0,
    relabel_xticks=True,
    figsize=(14, 8),
    ax=None,
    **save_kwargs,
):
    """
    Plot one or more orbit ground tracks on an Earth map.

    Parameters
    ----------
    r : (n,3) array_like or list of (n,3)
        GCRF positions [m]. Either a single orbit or a list of orbits.
    t : (n,) array_like or list of (n,)
        Absolute times matching r.
    ground_stations : (k,2) array_like, optional
        Rows of (lat_deg, lon_deg).
    save_path : str, optional
        Save figure path.
    title : str, optional
        Plot title.
    show_legend : bool, default True
        Show legend.
    fontsize : int, default 18
        Base font size.
    start_end_markers : bool, default True
        Mark orbit start/end points.
    labels : list of str, optional
        Per-orbit labels.
    orbit_colors : list, optional
        Per-orbit colors.
    linestyles : list, optional
        Per-orbit line styles.
    legend_kwargs : dict, optional
        Extra legend kwargs.
    central_longitude : float, default 0.0
        Center longitude of the displayed map.
        Use 0 for Greenwich-centered or 180 for Pacific-centered maps.
    relabel_xticks : bool, default True
        If True, relabel x ticks into conventional [-180, 180) longitude values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "groundtrack_plot")

    # Normalize inputs
    r_list = _as_list(r)
    t_list = _broadcast_time_list(r_list, t)
    n_tracks = len(r_list)
    central_longitude = float(central_longitude)

    # Sanity checks
    if labels is not None and len(labels) != n_tracks:
        raise ValueError("labels must have same length as number of orbits (tracks)")
    if orbit_colors is not None and len(orbit_colors) != n_tracks:
        raise ValueError("orbit_colors must have same length as number of orbits (tracks)")
    if linestyles is not None and len(linestyles) != n_tracks:
        raise ValueError("linestyles must have same length as number of orbits (tracks)")

    # Figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    xmin = central_longitude - 180.0
    xmax = central_longitude + 180.0

    # Background Earth image
    try:
        earth_img = load_earth_file()
        earth_img = _shift_earth_image(earth_img, central_longitude)

        ax.imshow(
            earth_img,
            extent=[xmin, xmax, -90, 90],
            origin="upper",
            aspect="auto",
            zorder=-1,
        )
    except Exception as e:
        print(f"Warning: Earth background not displayed: {e}")

    # Axes styling
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude", fontsize=fontsize)
    ax.set_ylabel("Latitude (deg)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 4)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    _set_longitude_ticks(ax, xmin, xmax, relabel_xticks=relabel_xticks)

    # Colors
    if orbit_colors is not None:
        colors = list(orbit_colors)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_tracks)))

    # Styles
    if linestyles is not None:
        styles = list(linestyles)
    else:
        styles = ["-"] * n_tracks

    line_handles = []

    # Plot each orbit
    for i, (ri, ti) in enumerate(zip(r_list, t_list)):
        lon, lat, _ = groundTrack(np.asarray(ri), ti, format="geodetic")

        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)

        # Wrap into displayed longitude interval
        lon_disp = _wrap_longitudes(lon_deg, central_longitude=central_longitude)

        # Break line at wrap crossings
        lon_plot, lat_plot = _clean_lonlat_wrap(lon_disp, lat_deg, threshold=179.0)

        color = colors[i % len(colors)]
        linestyle = styles[i]
        label = labels[i] if labels is not None else f"Orbit {i + 1}"

        line, = ax.plot(
            lon_plot,
            lat_plot,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            label=label,
        )
        line_handles.append(line)

        if start_end_markers and len(lon_disp) > 0:
            ax.plot(
                lon_disp[0],
                lat_deg[0],
                marker="*",
                color=color,
                markersize=12,
                linestyle="None",
            )
            ax.plot(
                lon_disp[-1],
                lat_deg[-1],
                marker="x",
                color=color,
                markersize=9,
                linestyle="None",
            )

    # Ground stations: rows are (lat_deg, lon_deg)
    gs_handle = None
    if ground_stations is not None:
        gs = np.asarray(ground_stations, dtype=float)
        if gs.ndim == 2 and gs.shape[1] == 2:
            gs_lon_disp = _wrap_longitudes(gs[:, 1], central_longitude=central_longitude)
            gs_handle = ax.scatter(
                gs_lon_disp,
                gs[:, 0],
                s=50,
                color="red",
                label="Ground Station",
            )

    # Legend
    if show_legend:
        handles = list(line_handles)
        if gs_handle is not None:
            handles.append(gs_handle)

        kw = dict(loc="lower left", fontsize=fontsize - 2)
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=handles, **kw)

    # Save if requested
    if save_path:
        figsave(fig, save_path)

    return fig
