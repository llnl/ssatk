import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILImage

from ssapy import groundTrack
from ssapy.utils import norm
from ._textures import earth_texture_path
from ..constants import RGEO, EARTH_RADIUS
from .plotutils import figsave, make_black, valid_orbits, _pop_save_path_aliases, _raise_unrecognized_kwargs


def tracking_plot(
    r: np.ndarray,
    t: np.ndarray,
    ground_stations=None,
    limits=None,
    title: str = '',
    figsize=(12, 8),
    save_path=None,
    scale: float = 1,
    **save_kwargs,
) -> None:
    """
    Create a 3D tracking plot of satellite positions over time on Earth's surface.

    Parameters
    ----------
    r : numpy.ndarray or list of numpy.ndarray
        Satellite positions in GCRF coordinates. If a single numpy array, it represents
        the satellite's position vector over time. If a list of numpy arrays, it
        represents multiple satellite position vectors.
    t : numpy.ndarray
        Timestamps corresponding to the satellite positions.
    ground_stations : list of (lat, lon) or None
        Optional list of ground stations as (latitude, longitude) pairs [deg].
    limits : float or None
        Plot limits for x, y, z axes. If float, use that for all axes. If None,
        limits are auto-determined from the data.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches (width, height).
    save_path : str or None
        If provided, path where the plot image/PDF will be saved.
    scale : float
        Scaling factor for the Earth image.

    Returns
    -------
    None
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "tracking_plot")

    r, t = valid_orbits(r, t)

    def _make_plot(r_one, t_one, ground_stations, limits_val, title, figsize, save_path, scale):
        lon, lat, height = groundTrack(r_one, t_one)
        lon[np.where(np.abs(np.diff(lon)) >= np.pi)] = np.nan
        lat[np.where(np.abs(np.diff(lat)) >= np.pi)] = np.nan

        x = r_one[:, 0] / RGEO
        y = r_one[:, 1] / RGEO
        z = r_one[:, 2] / RGEO

        # Resolve plot limits
        if isinstance(limits_val, (int, float)):
            limits_plot = limits_val
        elif limits_val is None:
            limits_plot = np.nanmax(np.abs([x, y, z])) * 1.1
        else:
            limits_plot = limits_val

        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Figure & Earth texture
        fig = plt.figure(dpi=100, figsize=figsize)
        fig.patch.set_facecolor('black')
        earth_png = PILImage.open(earth_texture_path())
        earth_png = earth_png.resize((5400 // max(1, int(scale)), 2700 // max(1, int(scale))))

        # 1) Long/Lat world map
        ax = fig.add_subplot(2, 3, (1, 2))
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.set_title(title, color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        # 2) Long/Lat zoomed
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-150, -60)
        ax.set_ylim(0, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        # 3) XY
        ax = fig.add_subplot(2, 3, 4)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO) ** 2)
        ax.scatter(x, y, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('y [GEO]', color='white')
        ax.set_title('XY', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        # 4) XZ
        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO) ** 2)
        ax.scatter(x, z, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('XZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        # 5) YZ
        ax = fig.add_subplot(2, 3, 6)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO) ** 2)
        ax.scatter(y, z, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')
        ax.set_xlabel('y [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('YZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        fig, ax = make_black(fig, ax)
        plt.tight_layout()

        if save_path:
            figsave(fig, save_path)
        return fig

    # Compute a consistent limit across all orbits if not provided
    if isinstance(limits, (int, float)):
        limits_for_all = limits
    else:
        limits_for_all = None
        try:
            limits_for_all = np.nanmax([np.nanmax(norm(rr) / RGEO) for rr in r]) * 1.2
        except Exception:
            pass

    fig = None
    for i, r_one in enumerate(r):
        t_one = t[i]
        fig = _make_plot(
            r_one,
            t_one,
            ground_stations=ground_stations,
            limits_val=limits_for_all if limits_for_all is not None else limits,
            title=title,
            figsize=figsize,
            save_path=save_path,
            scale=scale
        )
    return fig
