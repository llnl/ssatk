import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; required for 3D projection
from matplotlib.lines import Line2D
from .plotutils import figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs
from ..constants import EARTH_MU, EARTH_RADIUS
from ..propagators_orbit import leapfrog
from ssapy import Orbit


def find_intersection_time(r_start, v_start, r_ref, t_max):
    """Find time when orbit from (r_start, v_start) nearly intersects r_ref."""
    r_orbit, _ = leapfrog(r_start, v_start, t=np.arange(0, t_max, 1))
    distances = np.linalg.norm(r_orbit - r_ref, axis=1)
    idx = np.argmin(distances)
    return idx  # Time index of closest approach


def transfer_plot(r0, v0, rtransfer, vtransfer, rf, vf, show=False, c='black',
                  figsize=(7, 7), save_path=False, title='', **save_kwargs):
    """Plots a 3D orbital transfer with transfer orbit from departure to arrival.

    Args:
        r0: Initial position vector (m, SI units)
        v0: Initial velocity vector (m/s, SI units)
        rtransfer: Transfer orbit position vector (m, SI units)
        vtransfer: Transfer orbit velocity vector (m/s, SI units)
        rf: Final position vector (m, SI units)
        vf: Final velocity vector (m/s, SI units)
        show: If True, display the plot
        c: Color theme ('black', 'b', 'white', 'w')
        figsize: Figure size (width, height)
        save_path: Path to save plot, or False to not save
        title: Plot title

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_plot")

    from ..orbital_mechanics import period, a_from_periap

    r0, v0 = np.array(r0), np.array(v0)
    rtransfer, vtransfer = np.array(rtransfer), np.array(vtransfer)
    rf, vf = np.array(rf), np.array(vf)

    if c in ('black', 'b'):
        plotcolor, textcolor = 'black', 'white'
    elif c in ('white', 'w'):
        plotcolor, textcolor = 'white', 'black'
    else:
        plotcolor, textcolor = 'black', 'white'

    orbit_colors = ['cyan', 'yellow', 'magenta']  # Initial, Transfer, Final

    earth_radius_km = EARTH_RADIUS / 1000.0
    fig = plt.figure(dpi=100, figsize=figsize, facecolor=plotcolor)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(plotcolor)

    # Plot Earth sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.5)

    handles = []
    labels = []
    orbits = [(r0, v0, "Initial"), (rtransfer, vtransfer, "Transfer"), (rf, vf, "Final")]

    for idx, (r, v, orbit_type) in enumerate(orbits):
        if r.ndim > 1 and r.shape[0] > 1:
            r_km = r[0] / 1000
            r_orbit = r
            orbit = Orbit(r=r[0], v=v[0], t=0, mu=EARTH_MU)
        else:
            r_km = r / 1000
            orbit = Orbit(r=r, v=v, t=0, mu=EARTH_MU)
            max_time = orbit.period
            if orbit.period >= 1e7:
                max_time = period(a_from_periap(np.linalg.norm(r), np.linalg.norm(r)))

            if orbit_type == "Transfer":  # Transfer orbit: integrate from departure to arrival
                t_intersect = find_intersection_time(r, v, rf, max_time)
                t_array = np.arange(0, t_intersect + 1, 1)  # From 0 to intersection
                r_orbit, _ = leapfrog(r, v, t=t_array)
            else:
                r_orbit, _ = leapfrog(r, v, t=np.arange(0, max_time, 1))

        x = r_orbit[:, 0] / 1000
        y = r_orbit[:, 1] / 1000
        z = r_orbit[:, 2] / 1000

        orbit_line = ax.plot(x, y, z, color=orbit_colors[idx], linewidth=4)[0]

        if idx == 0:  # Initial orbit
            ax.plot([r_km[0]], [r_km[1]], [r_km[2]], 'o',
                    color=orbit_colors[idx], markersize=10)
            dep_proxy = Line2D([0], [0], linestyle='none', marker='o',
                               markersize=10, markerfacecolor=orbit_colors[idx])
        elif idx == 2:  # Final orbit
            ax.plot([r_km[0]], [r_km[1]], [r_km[2]], 'o',
                    color=orbit_colors[idx], markersize=10)
            arr_proxy = Line2D([0], [0], linestyle='none', marker='o',
                               markersize=10, markerfacecolor=orbit_colors[idx])

        orbit_type = ['Initial', 'Transfer', 'Final'][idx]
        label = (f'{orbit_type} Orbit\n(a={orbit.a / 1e3:.0f}km '
                 f'e={orbit.e:.2f} i={np.degrees(orbit.i):.0f}°)')
        handles.append(orbit_line)
        labels.append(label)

    ax.set_xlabel("X (km)", color=textcolor)
    ax.set_ylabel("Y (km)", color=textcolor)
    ax.set_zlabel("Z (km)", color=textcolor)
    ax.set_title(title, color=textcolor)
    ax.tick_params(axis='x', colors=textcolor)
    ax.tick_params(axis='y', colors=textcolor)
    ax.tick_params(axis='z', colors=textcolor)

    earth_proxy = Line2D([0], [0], linestyle='none', marker='o', markersize=10,
                         markerfacecolor='blue', alpha=0.5)

    handles.extend([earth_proxy, dep_proxy, arr_proxy])
    labels.extend(['Earth', 'Departure Point', 'Arrival Point'])
    handles, labels = [handles[0], dep_proxy, handles[1], arr_proxy, handles[2], earth_proxy], [labels[0], 'Departure Point', labels[1], 'Arrival Point', labels[2], 'Earth']
    ax.legend(handles, labels, facecolor=plotcolor, edgecolor=textcolor,
              labelcolor=textcolor, loc='upper left', bbox_to_anchor=(0, 1))

    plt.axis('equal')

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    from ..orbital_mechanics import hkoe, kepler_to_state
    from ..constants import RGEO

    # Example: LEO to GEO transfer
    r1, v1 = kepler_to_state(*hkoe(1.1 * RGEO, 0.1, 0, 0, 0, 0))
    r2, v2 = kepler_to_state(*hkoe(2 * RGEO, 0.7, 0, 0, 0, 45))
    r3, v3 = kepler_to_state(*hkoe(42164137, 0, 0, 0, 0, 90))

    transfer_plot(r1, v1, r2, v2, r3, v3, show=True, c='black')
