def groundtrack_dashboard_gamma_heading(r, t, save_path=None, pad=500, show=False, show_legend=True, t0=None, limit=None, fontsize=18, **save_kwargs):
    """
    Visualizes multiple satellite ground tracks, altitude/velocity over time,
    gamma over time, heading over time, and 3D trajectories (ITRF and GCRF).

    Layout (2 rows x 4 cols):
      Row 0: [ Ground map (1x2 span) | ITRF 3D | GCRF 3D ]
      Row 1: [ Gamma vs Time | Heading vs Time | Altitude vs Time | Velocity vs Time ]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    from ssapy import groundTrack
    from ..compute import find_smallest_bounding_cube
    from ..constants import EARTH_RADIUS
    from ..time_functions import to_gps
    from ..orbital_mechanics.gamma_and_heading import calc_gamma_and_heading
    from .plotutils import figsave, valid_orbits, _pop_save_path_aliases, _raise_unrecognized_kwargs
    from ._groundtrack_helpers import clean_lonlat, force_title

    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "groundtrack_dashboard_gamma_heading")

    r, t = valid_orbits(r, t)

    # Times -> GPS seconds and relative timeline
    t_gps = [to_gps(ti) for ti in t]
    if t0 is None:
        try:
            t0 = min(float(ti[0]) for ti in t_gps if len(ti) > 0)
        except Exception:
            t0 = 0.0
    t_rel = [ti - t0 for ti in t_gps]

    # Per-orbit derived series
    lons, lats, altitudes, velocities = [], [], [], []
    x_gts, y_gts, z_gts = [], [], []
    gammas, headings = [], []

    for r_i, t_i in zip(r, t):
        xyz = np.array(r_i)  # (n,3)

        # Ground track + geodetic
        x_gt, y_gt, z_gt = groundTrack(xyz, t_i, format="cartesian")
        lon, lat, height = groundTrack(xyz, t_i, format="geodetic")

        # Simple speed magnitude from finite differences (display only)
        try:
            vel = np.linalg.norm(np.gradient(xyz, axis=0), axis=1)
        except Exception:
            vel = 0

        # Gamma/Heading
        try:
            g_deg, h_deg = calc_gamma_and_heading(xyz, t_i)
        except Exception:
            n = xyz.shape[0]
            g_deg = np.full(n, np.nan)
            h_deg = np.full(n, np.nan)

        lons.append(np.degrees(lon))
        lats.append(np.degrees(lat))
        altitudes.append(height)
        velocities.append(vel)
        x_gts.append(x_gt)
        y_gts.append(y_gt)
        z_gts.append(z_gt)
        gammas.append(g_deg)
        headings.append(h_deg)

    # Earth surface for 3D plots
    phi_earth = np.linspace(0, np.pi, 50)
    theta_earth = np.linspace(0, 2 * np.pi, 50)
    phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
    earth_x = EARTH_RADIUS * np.sin(phi_earth) * np.cos(theta_earth)
    earth_y = EARTH_RADIUS * np.sin(phi_earth) * np.sin(theta_earth)
    earth_z = EARTH_RADIUS * np.cos(phi_earth)

    # Figure and grid: 2 rows x 4 columns
    fig = plt.figure(figsize=(28, 16))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # TOP ROW: Ground map (spans 0:2), ITRF 3D (col 2), GCRF 3D (col 3)
    ax_ground = fig.add_subplot(gs[0, 0:2])
    ax_itrf = fig.add_subplot(gs[0, 2], projection="3d")
    ax_gcrf = fig.add_subplot(gs[0, 3], projection="3d")

    # Ground map setup and plotting
    ax_ground.set_xlim(-180, 180)
    ax_ground.set_ylim(-90, 90)
    ax_ground.set_xlabel("Longitude (deg)", fontsize=fontsize)
    ax_ground.set_ylabel("Latitude (deg)", fontsize=fontsize)
    ax_ground.set_title("", fontsize=fontsize + 6)
    ax_ground.grid(True, alpha=0.3)
    ax_ground.tick_params(axis="both", labelsize=18)
    try:
        from .plotutils import load_earth_file
        ax_ground.imshow(load_earth_file(), extent=[-180, 180, -90, 90], aspect="auto", zorder=-1)
    except Exception:
        pass

    colors = plt.cm.tab10(np.linspace(0, 1, len(r)))
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        lon_c, lat_c = clean_lonlat(lon, lat)
        ax_ground.plot(lon_c, lat_c, color=colors[i], linewidth=2.5)
        ax_ground.plot(lon[0], lat[0], "*", color=colors[i], markersize=20)
        ax_ground.plot(lon[-1], lat[-1], "x", color=colors[i], markersize=14)

    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2.5, label="Orbit Track"),
        Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=12, label="Orbit Start"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10, label="Orbit End"),
    ]
    ax_ground.legend(handles=legend_elements, loc="lower left", fontsize=fontsize)

    # ITRF 3D
    ax_itrf.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color="blue", alpha=0.5, linewidth=0)
    for i, (x_gt, y_gt, z_gt) in enumerate(zip(x_gts, y_gts, z_gts)):
        ax_itrf.plot(x_gt / 1e3, y_gt / 1e3, z_gt / 1e3, color=colors[i], linewidth=2.5)
        ax_itrf.scatter(x_gt[0] / 1e3, y_gt[0] / 1e3, z_gt[0] / 1e3, color=colors[i], marker="*", s=120)
        ax_itrf.scatter(x_gt[-1] / 1e3, y_gt[-1] / 1e3, z_gt[-1] / 1e3, color=colors[i], marker="x", s=100)
    force_title(ax_itrf, "ITRF", fontsize)
    ax_itrf.set_xlabel("X (km)", fontsize=fontsize)
    ax_itrf.set_ylabel("Y (km)", fontsize=fontsize)
    ax_itrf.set_zlabel("Z (km)", fontsize=fontsize)
    ax_itrf.tick_params(axis="both", labelsize=fontsize - 2)

    # GCRF 3D
    ax_gcrf.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color="blue", alpha=0.5, linewidth=0)
    for i, r_i in enumerate(r):
        x, y, z = r_i[:, 0], r_i[:, 1], r_i[:, 2]
        ax_gcrf.plot(x / 1e3, y / 1e3, z / 1e3, color=colors[i], linewidth=2.5)
        ax_gcrf.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3, color=colors[i], marker="*", s=120)
        ax_gcrf.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3, color=colors[i], marker="x", s=100)
    force_title(ax_gcrf, "GCRF", fontsize)
    ax_gcrf.set_xlabel("X (km)", fontsize=fontsize)
    ax_gcrf.set_ylabel("Y (km)", fontsize=fontsize)
    ax_gcrf.set_zlabel("Z (km)", fontsize=fontsize)
    ax_gcrf.tick_params(axis="both", labelsize=fontsize - 2)

    # 3D plot limits
    if limit is None:
        all_xyz = np.concatenate([r_i for r_i in r], axis=0)
        lower_bound, upper_bound = find_smallest_bounding_cube(all_xyz, pad=pad)
        max_bound = np.max(np.abs([lower_bound, upper_bound])) / 1e3
        limit = max(10.0, float(max_bound))
    for ax in (ax_itrf, ax_gcrf):
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_xticks([-limit, 0, limit])
        ax.xaxis.set_major_locator(FixedLocator([-limit, 0, limit]))
        ax.yaxis.set_major_locator(FixedLocator([-limit, 0, limit]))
        ax.zaxis.set_major_locator(FixedLocator([-limit, 0, limit]))
        ax.xaxis.set_major_formatter(FixedFormatter(["", "0", f"{limit:.0f}"]))
        ax.yaxis.set_major_formatter(FixedFormatter(["", "0", ""]))
        ax.zaxis.set_major_formatter(FixedFormatter(["", "0", f"{limit:.0f}"]))
        ax.tick_params(pad=2)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

    # BOTTOM ROW: all time series (Gamma, Heading, Altitude, Velocity)
    ax_gamma = fig.add_subplot(gs[1, 0])
    ax_heading = fig.add_subplot(gs[1, 1])
    ax_alt = fig.add_subplot(gs[1, 2])
    ax_velocity = fig.add_subplot(gs[1, 3])

    # Gamma vs Time
    for i, (ti, g) in enumerate(zip(t_rel, gammas)):
        n = min(len(ti), len(g))
        if n == 0:
            continue
        ax_gamma.plot(ti[:n] / 60.0, g[:n], color=colors[i], linewidth=2.2, label=f"orbit {i + 1}")
    ax_gamma.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax_gamma.set_ylabel("Gamma (deg)", fontsize=fontsize)
    ax_gamma.set_ylim(-90, 90)
    ax_gamma.grid(True)
    force_title(ax_gamma, "Gamma vs Time", fontsize)
    ax_gamma.tick_params(axis="both", labelsize=fontsize)
    if show_legend and len(gammas) > 1:
        ax_gamma.legend(fontsize=fontsize - 4, loc="best")

    # Heading vs Time
    for i, (ti, h) in enumerate(zip(t_rel, headings)):
        n = min(len(ti), len(h))
        if n == 0:
            continue
        ax_heading.plot(ti[:n] / 60.0, h[:n], color=colors[i], linewidth=2.2, label=f"orbit {i + 1}")
    ax_heading.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax_heading.set_ylabel("Heading (deg)", fontsize=fontsize)
    ax_heading.set_ylim(0, 360)
    ax_heading.grid(True)
    force_title(ax_heading, "Heading vs Time", fontsize)
    ax_heading.tick_params(axis="both", labelsize=fontsize)
    if show_legend and len(headings) > 1:
        ax_heading.legend(fontsize=fontsize - 4, loc="best")

    # Altitude vs Time
    altmax = 0.0
    for i, (ti, alt) in enumerate(zip(t_rel, altitudes)):
        ax_alt.plot(ti / 60.0, alt / 1e3, color=colors[i], linewidth=2.5, label=f"orbit {i + 1}")
        if np.size(alt) > 0:
            altmax = max(altmax, float(np.nanmax(alt)))
    ax_alt.set_ylim(0, altmax / 1e3 * 1.1 if altmax > 0 else 1)
    ax_alt.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax_alt.set_ylabel("Altitude (km)", fontsize=fontsize)
    force_title(ax_alt, "Altitude vs Time", fontsize)
    ax_alt.tick_params(axis="both", labelsize=fontsize)
    ax_alt.grid(True)
    if show_legend and len(altitudes) > 1:
        ax_alt.legend(fontsize=fontsize - 4, loc="best")

    # Velocity vs Time
    vmax = 0.0
    for i, (ti, vel) in enumerate(zip(t_rel, velocities)):
        if np.ndim(vel) > 0 and len(vel) > 3:
            ti_plot = ti[1:-1]
            vel_plot = vel[1:-1]
        else:
            ti_plot = ti
            vel_plot = vel
        ax_velocity.plot(ti_plot / 60.0, np.asarray(vel_plot) / 1e3, color=colors[i], linewidth=2.5)
        if np.size(vel_plot) > 0:
            vmax = max(vmax, float(np.nanmax(vel_plot)))
    ax_velocity.set_ylim(0, vmax / 1e3 * 1.1 if vmax > 0 else 1)
    ax_velocity.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax_velocity.set_ylabel("Velocity (km/s)", fontsize=fontsize)
    force_title(ax_velocity, "Velocity vs Time", fontsize)
    ax_velocity.tick_params(axis="both", labelsize=fontsize)
    ax_velocity.grid(True)

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    return fig
