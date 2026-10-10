import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from ssapy import groundTrack
# from ..yastropy import astropy_gcrf_to_llh
from ..compute import find_smallest_bounding_cube
from ..constants import EARTH_RADIUS
from ..time_functions import to_gps
from .plotutils import figsave, valid_orbits, _pop_save_path_aliases, _raise_unrecognized_kwargs
from ._groundtrack_helpers import clean_lonlat, force_title


def groundtrack_dashboard(
    r,
    t,
    save_path=None,
    pad=500,
    show=False,
    show_legend=True,
    t0=None,
    limit=None,
    fontsize=18,
    labels=None,
    **save_kwargs,
):
    """
    Visualizes multiple satellite ground tracks, altitude/velocity over time, and 3D trajectories.

    Parameters
    ----------
    r : array_like or list of array_like
        Position vectors in meters. Either a single (n,3) array or a list of (n,3) arrays for multiple orbits.
    t : array_like or list of array_like
        Time arrays in seconds or any format convertible by `to_gps`. Single array or list matching `r`.
    save_path : str or None, optional
        If provided, saves the figure to the specified path.
    pad : float, optional
        Padding (in meters) for the 3D plot's bounding cube. Defaults to 500 m.
    show : bool, optional
        If True, displays the figure.
    show_legend : bool, optional
        If True, shows legends.
    t0 : float or None, optional
        Reference start time in GPS seconds for relative-time plots. If None, uses the min of first timestamps.
    limit : float or None, optional
        Fixed half-extent (in km) for 3D axes. If None, computed from data with padding.
    fontsize : int, optional
        Base fontsize. Other text scales relative to this.
    labels : list of str or None, optional
        Optional per-orbit labels. If None, defaults to ["Orbit 1", "Orbit 2", ...].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete dashboard figure object.

    Author: Travis Yeager
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "groundtrack_dashboard")

    r, t = valid_orbits(r, t)
    n_orbits = len(r)

    # Normalize labels
    if labels is None:
        labels_norm = [f"Orbit {i + 1}" for i in range(n_orbits)]
    else:
        labels_arr = list(labels)
        if len(labels_arr) < n_orbits:
            labels_arr = labels_arr + [f"Orbit {i + 1}" for i in range(len(labels_arr), n_orbits)]
        labels_norm = labels_arr[:n_orbits]

    # Ensure times are converted to GPS and normalized
    t_gps = [to_gps(ti) for ti in t]
    if t0 is None:
        try:
            t0 = min(float(ti[0]) for ti in t_gps if len(ti) > 0)
        except Exception:
            t0 = 0.0
    t_rel = [ti - t0 for ti in t_gps]

    # Process each orbit
    lons, lats, altitudes, velocities, x_gts, y_gts, z_gts = [], [], [], [], [], [], []
    for r_i, t_i in zip(r, t):
        xyz = np.array(r_i)
        x_gt, y_gt, z_gt = groundTrack(xyz, t_i, format='cartesian')
        lon, lat, height = groundTrack(xyz, t_i, format='geodetic')
        # lon, lat, height = astropy_gcrf_to_llh(xyz, t_i)
        try:
            velocity = np.linalg.norm(np.gradient(xyz, axis=0), axis=1)
        except Exception:
            velocity = 0

        lons.append(np.degrees(lon))
        lats.append(np.degrees(lat))
        altitudes.append(height)
        velocities.append(velocity)
        x_gts.append(x_gt)
        y_gts.append(y_gt)
        z_gts.append(z_gt)

    # Earth surface for 3D plots
    phi_earth = np.linspace(0, np.pi, 50)
    theta_earth = np.linspace(0, 2 * np.pi, 50)
    phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
    earth_x = EARTH_RADIUS * np.sin(phi_earth) * np.cos(theta_earth)
    earth_y = EARTH_RADIUS * np.sin(phi_earth) * np.sin(theta_earth)
    earth_z = EARTH_RADIUS * np.cos(phi_earth)

    # Create figure and grid
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Ground track plot (lon/lat axes)
    ax_ground = fig.add_subplot(gs[0, 0:2])
    ax_ground.set_xlim(-180, 180)
    ax_ground.set_ylim(-90, 90)
    ax_ground.set_xlabel('Longitude (deg)', fontsize=fontsize)
    ax_ground.set_ylabel('Latitude (deg)', fontsize=fontsize)
    ax_ground.set_title('', fontsize=fontsize + 6)
    ax_ground.grid(True, alpha=0.3)
    ax_ground.tick_params(axis='both', labelsize=18)

    # Optional background image
    try:
        from .plotutils import load_earth_file
        ax_ground.imshow(load_earth_file(), extent=[-180, 180, -90, 90], aspect='auto', zorder=-1)
    except Exception:
        pass

    colors = plt.cm.tab10(np.linspace(0, 1, n_orbits))
    ground_handles = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        lon_c, lat_c = clean_lonlat(lon, lat)
        h_line = ax_ground.plot(lon_c, lat_c, color=colors[i], linewidth=2.5, label=labels_norm[i])[0]
        ax_ground.plot(lon[0], lat[0], '*', color=colors[i], markersize=20)
        ax_ground.plot(lon[-1], lat[-1], 'x', color=colors[i], markersize=14)
        ground_handles.append(h_line)

    # Legends on ground track
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, label='Orbit Track'),
            Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=12, label='Orbit Start'),
            Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Orbit End')
        ]
        leg1 = ax_ground.legend(handles=legend_elements, loc='lower left', fontsize=fontsize)
        ax_ground.add_artist(leg1)
        ax_ground.legend(handles=ground_handles, loc='lower right', fontsize=fontsize - 2, title="Orbits")

    # Altitude plot
    ax_alt = fig.add_subplot(gs[1, 0])
    altmax = 0
    for i, (ti, alt) in enumerate(zip(t_rel, altitudes)):
        ax_alt.plot(ti / 60, alt / 1e3, color=colors[i], linewidth=2.5, label=labels_norm[i])
        altmax = max(altmax, np.max(alt))
    ax_alt.set_ylim(0, altmax / 1e3 * 1.1 if altmax > 0 else 1)
    ax_alt.set_xlabel('Time (minutes)', fontsize=fontsize)
    ax_alt.set_ylabel('Altitude (km)', fontsize=fontsize)
    force_title(ax_alt, "Altitude vs Time", fontsize)
    ax_alt.tick_params(axis='both', labelsize=fontsize)
    ax_alt.grid(True)
    if show_legend:
        ax_alt.legend(fontsize=fontsize - 4, title="Orbits")

    # Velocity plot
    ax_velocity = fig.add_subplot(gs[1, 1])
    vmax = 0
    for i, (ti, vel) in enumerate(zip(t_rel, velocities)):
        if np.ndim(vel) > 0 and len(vel) > 3:
            ti = ti[1:-1]
            vel = vel[1:-1]
        ax_velocity.plot(ti / 60, vel / 1e3, color=colors[i], linewidth=2.5, label=labels_norm[i])
        vmax = max(vmax, np.max(vel) if np.size(vel) > 0 else 0)
    ax_velocity.set_ylim(0, vmax / 1e3 * 1.1 if vmax > 0 else 1)
    ax_velocity.set_xlabel('Time (minutes)', fontsize=fontsize)
    ax_velocity.set_ylabel('Velocity (km/s)', fontsize=fontsize)
    force_title(ax_velocity, "Velocity vs Time", fontsize)
    ax_velocity.tick_params(axis='both', labelsize=fontsize)
    ax_velocity.grid(True)
    if show_legend:
        ax_velocity.legend(fontsize=fontsize - 4, title="Orbits")

    # ITRF 3D plot
    ax_itrf = fig.add_subplot(gs[0, 2], projection='3d')
    ax_itrf.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color='blue', alpha=0.5, linewidth=0)
    for i, (x_gt, y_gt, z_gt) in enumerate(zip(x_gts, y_gts, z_gts)):
        ax_itrf.plot(x_gt / 1e3, y_gt / 1e3, z_gt / 1e3, color=colors[i], linewidth=2.5)
        ax_itrf.scatter(x_gt[0] / 1e3, y_gt[0] / 1e3, z_gt[0] / 1e3, color=colors[i], marker='*', s=120)
        ax_itrf.scatter(x_gt[-1] / 1e3, y_gt[-1] / 1e3, z_gt[-1] / 1e3, color=colors[i], marker='x', s=100)
    force_title(ax_itrf, "ITRF", fontsize)
    ax_itrf.set_xlabel('X (km)', fontsize=fontsize)
    ax_itrf.set_ylabel('Y (km)', fontsize=fontsize)
    ax_itrf.set_zlabel('Z (km)', fontsize=fontsize)
    ax_itrf.tick_params(axis='both', labelsize=fontsize - 2)

    # Set 3D plot limits
    if limit is None:
        all_xyz = np.concatenate([r_i for r_i in r], axis=0)
        lower_bound, upper_bound = find_smallest_bounding_cube(all_xyz, pad=pad)
        max_bound = np.max(np.abs([lower_bound, upper_bound])) / 1e3
        limit = max(10.0, max_bound)
    ax_itrf.set_xlim([-limit, limit])
    ax_itrf.set_ylim([-limit, limit])
    ax_itrf.set_zlim([-limit, limit])
    ax_itrf.set_xticks([-limit, 0, limit])

    # GCRF 3D plot
    ax_gcrf = fig.add_subplot(gs[1, 2], projection='3d')
    ax_gcrf.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color='blue', alpha=0.5, linewidth=0)
    for i, r_i in enumerate(r):
        x, y, z = r_i[:, 0], r_i[:, 1], r_i[:, 2]
        ax_gcrf.plot(x / 1e3, y / 1e3, z / 1e3, color=colors[i], linewidth=2.5)
        ax_gcrf.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3, color=colors[i], marker='*', s=120)
        ax_gcrf.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3, color=colors[i], marker='x', s=100)
    force_title(ax_gcrf, "GCRF", fontsize)
    ax_gcrf.set_xlabel('X (km)', fontsize=fontsize)
    ax_gcrf.set_ylabel('Y (km)', fontsize=fontsize)
    ax_gcrf.set_zlabel('Z (km)', fontsize=fontsize)
    ax_gcrf.tick_params(axis='both', labelsize=fontsize - 2)
    ax_gcrf.set_xlim([-limit, limit])
    ax_gcrf.set_ylim([-limit, limit])
    ax_gcrf.set_zlim([-limit, limit])
    ax_gcrf.set_xticks([-limit, 0, limit])

    ticks = [-limit, 0, limit]
    for ax in (ax_itrf, ax_gcrf):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.zaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(["", "0", f"{limit:.0f}"]))
        ax.yaxis.set_major_formatter(FixedFormatter(["", "0", ""]))
        ax.zaxis.set_major_formatter(FixedFormatter(["", "0", f"{limit:.0f}"]))
        ax.tick_params(pad=2)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    return fig
