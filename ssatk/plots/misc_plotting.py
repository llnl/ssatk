######################################################################
# COLLECTION OF ALL PLOTTING AND MEDIA
######################################################################
# flake8: noqa: E501
import numpy as np
from ssapy.body import get_body
from astropy.time import Time

from ..constants import RGEO, EARTH_MU, MOON_MU
from .plotutils import make_black, make_white, figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize

lunar_semi_major = 384399000  # m


def koe_plot(r: np.ndarray, v: np.ndarray, t=None,
             elements=None, save_path=None, body: str = 'Earth', **save_kwargs):
    """
    Generates a plot of orbital elements (eccentricity, inclination, and semi-major axis)
    for a given position and velocity vectors.

    Parameters
    ----------
    r : np.ndarray
        Position vectors of the satellite in 3D space (shape: [n, 3]).
    v : np.ndarray
        Velocity vectors of the satellite in 3D space (shape: [n, 3]).
    t : Optional[Time], optional
        Time instance(s) associated with the orbital elements. If None, the x-axis is an index.
    elements : list, optional
        Subset of elements to plot among ['a', 'e', 'i']. Default: all three.
    save_path : str or None, optional
        Path to save the plot. If None, the plot is not saved.
    body : str, optional
        'Earth' or 'Moon'.

    Returns
    -------
    (plt.Figure, plt.Axes)
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "koe_plot")

    if elements is None:
        elements = ['a', 'e', 'i']

    # Calculate orbital elements for Earth or Moon
    from ..orbital_mechanics import calculate_orbital_elements

    if 'earth' in body.lower():
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    else:
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=MOON_MU)

    # Choose an x source based on what's available
    def _len_of_first_present():
        for key in ('e', 'i', 'a'):
            if key in orbital_elements:
                return len(orbital_elements[key])
        return 0

    fig, ax1 = plt.subplots(dpi=100)
    ax2 = ax1.twinx()
    make_white(fig, *[ax1, ax2])

    x_values = Time(t).decimalyear if t is not None else np.arange(_len_of_first_present())

    # Plot eccentricity and inclination (left axis)
    if 'e' in elements and 'e' in orbital_elements:
        ax1.plot(x_values, [x for x in orbital_elements['e']], label='eccentricity', c='C1')
    if 'i' in elements and 'i' in orbital_elements:
        ax1.plot(x_values, [x for x in orbital_elements['i']], label='inclination [rad]', c='C2')

    ax1.set_xlabel('Year' if t is not None else 'Index')
    ax1.set_ylim((0, np.pi / 2))
    ylabel = ax1.set_ylabel('', color='black')

    # Annotate only what we actually plotted
    xlab = ylabel.get_position()[0] + 0.05
    ylab = ylabel.get_position()[1]
    if 'e' in elements and 'e' in orbital_elements:
        fig.text(xlab - 0.001, ylab - 0.225, 'Eccentricity', color='C1', rotation=90)
    if ('e' in elements and 'e' in orbital_elements) and ('i' in elements and 'i' in orbital_elements):
        fig.text(xlab, ylab - 0.05, '/', color='k', rotation=90)
    if 'i' in elements and 'i' in orbital_elements:
        fig.text(xlab, ylab - 0.025, 'Inclination [Radians]', color='C2', rotation=90)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='upper left')

    # Semi-major axis on right axis
    if 'a' in elements and 'a' in orbital_elements:
        a = [x / RGEO for x in orbital_elements['a']]
        ax2.plot(x_values, a, label='semi-major axis [GEO]', c='C0', linestyle='-')
        ax2.set_ylabel('semi-major axis [GEO]', color='C0')
        ax2.yaxis.label.set_color('C0')
        ax2.tick_params(axis='y', colors='C0')
        ax2.spines['right'].set_color('C0')
        if np.abs(np.max(a) - np.min(a)) < 2:
            ax2.set_ylim((np.min(a) - 0.5, np.max(a) + 0.5))

    # Optionally save the plot
    if save_path:
        from .plotutils import figsave
        figsave(fig, save_path)

    return fig, ax1


def koe_2dhist(stable_data, title: str = "Initial orbital elements of\n1 year stable cislunar orbits",
               limits: list = [1, 50], bins: int = 200, logscale: bool = False, cmap: str = 'coolwarm',
               save_path: str = None, **save_kwargs) -> plt.Figure:
    """
    Generates a 2D histogram plot of orbital elements for a set of stable orbital data.
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "koe_2dhist")

    # Validate angle data ranges
    if not (np.all((0 <= stable_data.i) & (stable_data.i <= 2 * np.pi))):
        raise ValueError("Inclination (`i`) must be in the range [0, 2π] radians.")
    if not (np.all((0 <= stable_data.ta) & (stable_data.ta <= 2 * np.pi))):
        raise ValueError("True Anomaly (`ta`) must be in the range [0, 2π] radians.")

    if logscale or logscale == 'log':
        norm = LogNorm(limits[0], limits[1])
    else:
        norm = Normalize(limits[0], limits[1])

    fig, axes = plt.subplots(dpi=100, figsize=(10, 8), nrows=3, ncols=3)
    st = fig.suptitle(title, fontsize=12)
    st.set_x(0.46)
    st.set_y(0.9)

    mappable = None  # will hold a QuadMesh from hist2d

    # Semi-major axis vs Eccentricity
    ax = axes.flat[0]
    *_, mappable = ax.hist2d([x / RGEO for x in stable_data.a],
                             [x for x in stable_data.e],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("eccentricity")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_xlim((1, 18))

    # Empty plots
    axes.flat[1].set_axis_off()
    axes.flat[2].set_axis_off()

    # Semi-major axis vs Inclination
    ax = axes.flat[3]
    *_, mappable = ax.hist2d([x / RGEO for x in stable_data.a],
                             [np.degrees(x) for x in stable_data.i],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("inclination [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_xlim((1, 18))

    # Eccentricity vs Inclination
    ax = axes.flat[4]
    *_, mappable = ax.hist2d([x for x in stable_data.e],
                             [np.degrees(x) for x in stable_data.i],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 91, 15))

    # Empty plot
    axes.flat[5].set_axis_off()

    # Semi-major axis vs True Anomaly
    ax = axes.flat[6]
    *_, mappable = ax.hist2d([x / RGEO for x in stable_data.a],
                             [np.degrees(x) for x in stable_data.ta],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("semi-major axis [GEO]")
    ax.set_ylabel("True Anomaly [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax.set_xlim((1, 18))

    # Eccentricity vs True Anomaly
    ax = axes.flat[7]
    *_, mappable = ax.hist2d([x for x in stable_data.e],
                             [np.degrees(x) for x in stable_data.ta],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 361, 60))

    # Inclination vs True Anomaly
    ax = axes.flat[8]
    *_, mappable = ax.hist2d([np.degrees(x) for x in stable_data.i],
                             [np.degrees(x) for x in stable_data.ta],
                             bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("inclination [deg]")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_yticks(np.arange(0, 361, 60))

    # Colorbar with a real mappable
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    if mappable is not None:
        fig.colorbar(mappable, cax=cbar_ax)
    fig, ax = make_white(fig, ax)

    if save_path:
        figsave(fig, save_path)
    return fig


def scatter2d(x: list, y: list, cs: list, xlabel: str = 'x', ylabel: str = 'y', title: str = '',
              cbar_label: str = '', dotsize: int = 1, colorsMap: str = 'jet', colorscale: str = 'linear',
              colormin: float = None, colormax: float = None, save_path: str = None, **save_kwargs) -> None:
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "scatter2d")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cs_arr = np.asarray(cs)
    if colormax is None:
        colormax = np.nanmax(cs_arr)
    if colormin is None:
        colormin = np.nanmin(cs_arr)

    cmap = plt.get_cmap(colorsMap)
    if colorscale == 'linear':
        cNorm = Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = LogNorm(vmin=colormin, vmax=colormax)
    else:
        cNorm = Normalize(vmin=colormin, vmax=colormax)

    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    ax.scatter(x, y, c=scalarMap.to_rgba(cs_arr), s=dotsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    scalarMap.set_array(cs_arr)
    fig.colorbar(scalarMap, ax=ax, shrink=.5, label=f'{cbar_label}', pad=0.04)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        figsave(fig, save_path)
    return


def scatter3d(x: list, y: list = None, z: list = None, cs: list = None,
              xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'z', cbar_label: str = '', dotsize: int = 1,
              colorsMap: str = 'jet', title: str = '', save_path: str = None, **save_kwargs):
    """
    Returns
    -------
    (plt.Figure, matplotlib.axes._subplots.Axes3DSubplot)
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "scatter3d")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if hasattr(x, "ndim") and getattr(x, "ndim", 1) > 1:
        r = np.asarray(x)
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]

    if cs is None:
        ax.scatter(x, y, z, s=dotsize)
    else:
        cs_arr = np.asarray(cs)
        cmap = plt.get_cmap(colorsMap)
        cNorm = Normalize(vmin=np.nanmin(cs_arr), vmax=np.nanmax(cs_arr))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs_arr), s=dotsize)
        scalarMap.set_array(cs_arr)
        fig.colorbar(scalarMap, ax=ax, shrink=.5, label=f'{cbar_label}', pad=0.075)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        figsave(fig, save_path)
    return fig, ax


def dotcolors_scaled(num_colors: int) -> list:
    return cm.rainbow(np.linspace(0, 1, num_colors))


# Make a plot of multiple cislunar orbit in GCRF frame.
def orbit_divergence_plot(rs: np.ndarray, r_moon: np.ndarray = None, t=None,
                          limits: float = None, title: str = '', save_path: str = None,
                          show: bool = True, **save_kwargs):
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "orbit_divergence_plot")

    if limits is None:
        limits = np.nanmax(np.linalg.norm(rs, axis=1) / RGEO) * 1.2
        print(f'limits: {limits}')

    # Acquire or validate moon positions
    if r_moon is None or np.size(r_moon) < 1:
        moon = get_body("moon")
        r_moon = moon.position(t)  # expect shape (N, 3) or (3, N)

    r_moon = np.asarray(r_moon)
    if r_moon.ndim != 2:
        raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")

    # Ensure shape (N, 3)
    if r_moon.shape[1] == 3:
        pass
    elif r_moon.shape[0] == 3:
        r_moon = r_moon.T
    else:
        raise IndexError(f"input moon data shape: {np.shape(r_moon)}, expected (N,3) or (3,N).")

    # Normalize once
    r_moon = r_moon / RGEO

    fig = plt.figure(dpi=100, figsize=(15, 4))
    for i in range(rs.shape[-1]):
        r = rs[:, :, i]
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # XY
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(r_moon[:, 0], r_moon[:, 1], color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('y [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], y[0], '$\\leftarrow$ start')
        plt.text(x[-1], y[-1], '$\\leftarrow$ end')

        # XZ
        plt.subplot(1, 3, 2)
        plt.scatter(x, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(r_moon[:, 0], r_moon[:, 2], color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], z[0], '$\\leftarrow$ start')
        plt.text(x[-1], z[-1], '$\\leftarrow$ end')
        plt.title(f'{title}')

        # YZ
        plt.subplot(1, 3, 3)
        plt.scatter(y, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(r_moon[:, 1], r_moon[:, 2], color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('y [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(y[0], z[0], '$\\leftarrow$ start')
        plt.text(y[-1], z[-1], '$\\leftarrow$ end')

    plt.tight_layout()
    if show:
        plt.show(block=False)
    if save_path:
        figsave(fig, save_path)
        return None
    return fig
