import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from ..constants import EARTH_RADIUS
import numpy as np
from ssapy import Orbit, rv


def rendezvous_plot(r1, v1,
                    rtransfer, vtransfer,
                    r2=None, v2=None,
                    title=''):
    """
    3D plot of chaser and target rendezvous.

    Parameters
    ----------
    r1 : array_like
        Initial position of chaser (m).
    v1 : array_like
        Initial velocity of chaser (m/s).
    rtransfer : ndarray
        Chaser positions during transfer (m).
    vtransfer : ndarray
        Chaser velocities during transfer (m/s).
    r2 : ndarray or None, optional
        Initial position of target (m). If None, target orbit is not plotted.
    v2 : ndarray or None, optional
        Initial velocity of target (m/s). If None, target orbit is not plotted.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The 3D rendezvous plot.
    """
    def get_full_orbit_positions(r, v):
        orbit = Orbit(r=r, v=v, t=0)
        times = np.arange(0, orbit.period)
        r_arr, v_arr = rv(orbit, time=times)
        return r_arr / 1e3

    fig = plt.figure(dpi=100, figsize=(7, 7), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = EARTH_RADIUS / 1e3 * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS / 1e3 * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS / 1e3 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)

    # Chaser initial point
    r1_km = r1 / 1e3
    ax.plot(*r1_km, 'o', color='r', label='Chaser Start')

    # Chaser initial orbit
    r_chaser_full = get_full_orbit_positions(r1, v1)
    ax.plot(r_chaser_full[:, 0],
            r_chaser_full[:, 1],
            r_chaser_full[:, 2],
            'r--', alpha=0.5, label='Initial Orbit')

    # Plot target orbit only if both r2 and v2 are provided
    if r2 is not None and v2 is not None:
        r_target_full = get_full_orbit_positions(r2, v2)
        ax.plot(r_target_full[:, 0],
                r_target_full[:, 1],
                r_target_full[:, 2],
                'b--', alpha=0.5, label='Target Orbit')
        # Highlight transfer segment overlay
        ax.plot(r_target_full[:len(rtransfer), 0],
                r_target_full[:len(rtransfer), 1],
                r_target_full[:len(rtransfer), 2],
                'b', alpha=1)
        # Target start point
        r2_km = r2 / 1e3
        ax.plot(*r2_km, 'o', color='b', markersize=8, label='Target Start')

    # Transfer arc and rendezvous point
    rt_km = rtransfer / 1e3
    ax.plot(*rt_km.T, color='cyan', lw=2, label='Transfer Arc')
    ax.plot(*rt_km[-1, :], 'o', color='cyan', markersize=10, label='Rendezvous Point')

    # Axes and legend
    ax.set_xlabel("X (km)", color='white')
    ax.set_ylabel("Y (km)", color='white')
    ax.set_zlabel("Z (km)", color='white')
    ax.set_title(title, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper left')
    plt.axis('equal')

    return fig
