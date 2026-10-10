import numpy as np
import matplotlib.pyplot as plt
from ..constants import EARTH_MU, EARTH_RADIUS
from .plotutils import figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs
from ..propagators_orbit import leapfrog
from ssapy import Orbit


def orbit_plot_rv(state_vectors, colors=False, mu=EARTH_MU, show=True, c='black', figsize=(7, 7), save_path=False, title='', **save_kwargs):
    """
    Plots the 3D orbital ellipse(s) given one or more sets of state vectors.

    Parameters
    ----------
    state_vectors : tuple or list of tuple
        A single ``(r, v)`` tuple or a list of tuples, with position ``r`` in
        metres and velocity ``v`` in m/s.
    mu : float
        Gravitational parameter in m³/s².
    show : bool
        Display the plot when true.
    c : str
        Color theme: ``'black'``, ``'b'``, ``'white'``, or ``'w'``.
    figsize : tuple
        Figure size as ``(width, height)``.
    save_path : str or bool
        Output path, or false to skip saving.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "orbit_plot_rv")

    from ..orbital_mechanics import period, a_from_periap

    if c in ('black', 'b'):
        plotcolor = 'black'
        textcolor = 'white'
        orbitcolor = 'cyan'
    elif c in ('white', 'w'):
        plotcolor = 'white'
        textcolor = 'black'
        orbitcolor = 'blue'
    else:
        plotcolor = 'white'
        textcolor = 'black'
        orbitcolor = 'blue'

    # Normalize input to a list of (r, v) tuples
    if isinstance(state_vectors, tuple) and len(state_vectors) == 2 and isinstance(state_vectors[0], (list, np.ndarray)):
        state_vectors = [state_vectors]  # Single pair case
    elif not isinstance(state_vectors, list) or not all(isinstance(sv, tuple) and len(sv) == 2 for sv in state_vectors):
        raise ValueError("state_vectors must be a tuple (r, v) or list of tuples [(r1, v1), ...]")

    # Create figure
    earth_radius_km = EARTH_RADIUS / 1000.0
    fig = plt.figure(dpi=300, figsize=figsize, facecolor=plotcolor)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(plotcolor)

    # Plot Earth sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.5)

    # Plot each orbit
    handles = []
    labels = []
    for idx, (r, v) in enumerate(state_vectors):
        r = np.array(r)
        v = np.array(v)
        orbit = Orbit(r=r, v=v, t=0, mu=EARTH_MU)
        max_time = orbit.period
        if orbit.period >= 1e7:
            max_time = period(a_from_periap(np.linalg.norm(r), np.linalg.norm(r)))
        r_orbit, vs = leapfrog(r, v, t=np.arange(0, max_time, 0.1))

        x = r_orbit[:, 0]
        y = r_orbit[:, 1]
        z = r_orbit[:, 2]

        x_km = x / 1000
        y_km = y / 1000
        z_km = z / 1000
        r_km = r / 1000

        orbit_line = ax.plot(x_km, y_km, z_km, color=orbitcolor, linewidth=6)[0]
        initial_point = ax.plot([r_km[0]], [r_km[1]], [r_km[2]], 'ro')[0]

        handles.append(orbit_line)
        labels.append(f'Orbit {idx+1}\n(a={orbit.a / 1e3:.0f}km e={orbit.e:.2f} i={np.degrees(orbit.i):.0f})')

    # Set labels and styling
    ax.set_xlabel("X (km)", color=textcolor)
    ax.set_ylabel("Y (km)", color=textcolor)
    ax.set_zlabel("Z (km)", color=textcolor)
    ax.set_title(title, color=textcolor)
    ax.tick_params(axis='x', colors=textcolor)
    ax.tick_params(axis='y', colors=textcolor)
    ax.tick_params(axis='z', colors=textcolor)

    # Add Earth to legend
    from matplotlib.lines import Line2D
    earth_proxy = Line2D([0], [0], linestyle='none', marker='o', markersize=10, markerfacecolor='blue', alpha=0.5)
    handles.append(earth_proxy)
    labels.append('Earth')
    handles.append(initial_point)
    labels.append('Initial Position')
    ax.legend(handles, labels, facecolor=plotcolor, edgecolor=textcolor, labelcolor=textcolor, loc='upper left', bbox_to_anchor=(0, 1))

    plt.axis('equal')

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    from ..orbital_mechanics import hkoe, kepler_to_state
    from ..constants import RGEO

    # Single orbit
    r1, v1 = kepler_to_state(*hkoe(1 * RGEO, 0.9, 90, 0, 0, 120))
    orbit_plot_rv((r1, v1), show=True, c='w')

    # Multiple orbits
    r2, v2 = kepler_to_state(*hkoe(2 * RGEO, 0.5, 45, 0, 0, 0))
    r3, v3 = kepler_to_state(*hkoe(42164137, 0, 0, 0, 0, 0))  # GEO
    orbit_plot_rv([(r1, v1), (r2, v2), (r3, v3)], show=True, c='black')
