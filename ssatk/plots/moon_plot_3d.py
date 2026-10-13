"""Dated, textured lunar scenes in Moon-centred inertial or rotating axes."""

from .rendering import is_low_spec
from ._body_surfaces import BODY_COLORS, texture_path

from .rendering import _with_low_spec
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from PIL import Image
from ssapy import get_body
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .plotutils import valid_orbits, figsave
from ._globe_scene import earth_to_scene
from .sun_mpl import get_sun_position
from .sun_render import render_sun, background_sun_radius
from ..constants import MOON_RADIUS, EARTH_RADIUS
from ..coordinates import gcrf_to_lunar, gcrf_to_lunar_fixed
from ..coordinates.lunar import lunar_map_rotation


def _ssapy_texture(name):
    """Resolve installed textures without downloading anything."""
    return texture_path(name)


def _textured_body(ax, center, radius, path, rotation, sun_hat, *,
                   n=180, ambient=0.22, diffuse=0.78, color=0.6):
    """North-up -180..180 degree map on a rotated spherical mesh.

    Lunar map coordinates use the DE440 mean-Earth/polar-axis frame. Colours are illustrative
    ambient + Lambert shading, not calibrated photometry.
    """
    if is_low_spec():
        n, path = min(n, 18), None
    width, height = n * 2, n
    if path is None:
        rgb = np.full((height, width, 3), color)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize(
                (width, height), Image.Resampling.BOX), dtype=float) / 255.
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, width + 1),
                          np.linspace(np.pi/2, -np.pi/2, height + 1))
    vertices = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon),
                         np.sin(lat)), axis=-1) @ rotation.T
    quads = np.stack((vertices[:-1, :-1], vertices[:-1, 1:],
                      vertices[1:, 1:], vertices[1:, :-1]), axis=2).reshape(-1, 4, 3)
    colors = rgb.reshape(-1, 3)
    if sun_hat is not None:
        normals = quads.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        colors = np.clip(colors * (ambient + diffuse * np.maximum(0, normals @ sun_hat))[:, None], 0, 1)
    artist = Poly3DCollection(center + radius * quads, facecolors=colors,
                              edgecolors="none", linewidths=0, antialiased=False, zorder=10)
    ax.add_collection3d(artist)
    from .occultation import _register_body
    _register_body(ax, artist, center, radius)
    return artist


def _scene_stars(ax, rotation, epoch, elev, azim, mag_limit):
    """Catalogue sky behind the scene, in a 120-degree gnomonic window.

    The orthographic orbital diagram has no physical field of view. This
    separate sky projection supplies directional context, not an astrometric
    camera model. No random fallback or finite-distance star sphere is used.
    """
    from .starfield import _load_stars, _to_datetime
    stars = _load_stars(mag_limit=mag_limit, when=_to_datetime(epoch), frame="gcrf")
    if stars is None:
        return
    el, az = np.radians([elev, azim])
    forward = -np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    up = np.cross(right, forward)
    vectors = stars['v'] @ rotation.T
    depth = vectors @ forward
    visible = depth > 0
    vectors, depth = vectors[visible], depth[visible]
    scale = np.tan(np.radians(60))
    xy = np.column_stack((vectors @ right, vectors @ up)) / (depth[:, None]*scale)
    mask = np.all(np.abs(xy) < 1, axis=1)
    bg = ax.figure.add_axes(ax.get_position(), zorder=-1)
    bg.scatter(xy[mask, 0], xy[mask, 1], s=stars['sizes'][visible][mask],
               c=stars['rgb'][visible][mask], alpha=.65, linewidths=0)
    bg.set(xlim=(-1, 1), ylim=(-1, 1))
    bg.set_axis_off()
    ax.patch.set_alpha(0)


@_with_low_spec
def moon_plot_3d(r=None, t=None, title='', figsize=(10, 10),
                 save_path=False, show=False, elev=30, azim=45,
                 show_lagrange=True, show_stars=True, show_sun=True,
                 show_sun_graphic=None, show_earth=None, mag_limit=5.5,
                 sun_azimuth_deg=None, sun_elevation_deg=None,
                 shade_ambient=0.22, shade_diffuse=0.78,
                 scene_pad=0.30, view_zoom=1.0,
                 physical_body_sizes=False, r_frame='gcrf', reference_time=None):
    """Plot lunar trajectories and a textured Moon at a dated reference epoch.

    ``r`` contains positions in metres; ``t`` contains absolute GPS seconds or
    Astropy Time. GCRF input is Earth-centred and transformed at every sample
    into Moon-centred Earth-Moon rotating axes (+X away from Earth). Input
    ``r_frame='moon_centered'`` is already Moon-centred in GCRF-aligned inertial
    axes and is used unchanged. These are distinct from the lunar body axes.

    ``reference_time`` (scalar Time or GPS seconds) sets the texture, Earth and
    lighting snapshot. Otherwise use the midpoint of the trajectory's time
    extent, standalone ``t``, or the current time for an undated surface view.
    Trajectories require explicit times. Multi-epoch tracks are shown with
    body textures and illumination frozen at this stated reference epoch.

    Camera angles are degrees. Equal-axis orthographic limits always contain
    the Moon and tracks. ``show_earth=True`` expands the limits to include
    Earth; None includes it only if already in the view; False omits it.
    ``scene_pad`` is a nonnegative fractional margin; ``view_zoom`` is positive.
    ``physical_body_sizes=True`` uses physical radii; False permits enlarged
    bodies in wide views, with their display scale printed on the figure.

    ``show_sun`` enables ephemeris shading. ``show_sun_graphic`` defaults to
    False: if requested, the labelled Sun symbol indicates direction only,
    without physical distance or size. Optional solar azimuth/elevation override
    ephemeris lighting in scene axes. Ambient/diffuse terms are display gains.
    Stars are a separate catalogue sky projection, not a camera measurement.
    Lagrange points are approximate markers in rotating axes, shown only when
    inside the view. Return the Matplotlib ``(fig, ax)``; optionally save/show.
    """
    if r_frame not in {'gcrf', 'moon_centered'}:
        raise ValueError("r_frame must be 'gcrf' or 'moon_centered'")
    if not np.isfinite(scene_pad) or scene_pad < 0 or not np.isfinite(view_zoom) or view_zoom <= 0:
        raise ValueError("scene_pad must be nonnegative and view_zoom positive")
    tracks = []
    epochs = []
    if r is not None:
        if t is None:
            raise ValueError("Lunar trajectories require absolute times in t")
        r_list, t_list = valid_orbits(r, t)
        for positions, times in zip(r_list, t_list):
            if not len(positions) or not np.all(np.isfinite(positions)):
                raise ValueError("r must contain nonempty finite positions")
            gps = np.atleast_1d(times.gps if isinstance(times, Time) else times).astype(float)
            epochs.extend(gps)
            tracks.append(np.asarray(positions if r_frame == 'moon_centered' else
                                     gcrf_to_lunar_fixed(positions, times)) / 1000.)
    elif t is not None:
        epochs.extend(np.atleast_1d(t.gps if isinstance(t, Time) else t).astype(float))
    if reference_time is not None:
        epoch = reference_time if isinstance(reference_time, Time) else Time(reference_time, format='gps')
        if not epoch.isscalar:
            raise ValueError("reference_time must be scalar")
    elif epochs:
        epoch = Time((min(epochs) + max(epochs)) / 2., format='gps')
    else:
        epoch = Time.now()
    if (epochs and not np.all(np.isfinite(epochs))) or not np.isfinite(epoch.gps):
        raise ValueError("Times must be finite")

    moon = get_body('moon')
    moon_gcrf = np.asarray(moon.position(epoch.gps)).reshape(3)
    rotation = np.eye(3) if r_frame == 'moon_centered' else gcrf_to_lunar(np.eye(3), epoch).T
    earth_pos = rotation @ -moon_gcrf / 1000.
    moon_rotation = rotation @ lunar_map_rotation(epoch)[0].T
    earth_rotation = rotation @ earth_to_scene(epoch.gps)
    moon_r, earth_r = MOON_RADIUS/1000., EARTH_RADIUS/1000.
    lower, upper = np.full(3, -moon_r), np.full(3, moon_r)
    for xyz in tracks:
        lower, upper = np.minimum(lower, xyz.min(axis=0)), np.maximum(upper, xyz.max(axis=0))
    if show_earth is True:
        lower, upper = np.minimum(lower, earth_pos-earth_r), np.maximum(upper, earth_pos+earth_r)
    center = (lower+upper)/2
    span = float(max(upper-lower)) * (1+2*scene_pad)
    lower, upper = center-span/2, center+span/2
    draw_earth = bool(show_earth) if show_earth is not None else bool(
        np.all(earth_pos-earth_r >= lower) and np.all(earth_pos+earth_r <= upper))
    moon_display = moon_r if physical_body_sizes else max(moon_r, span*.03)
    earth_display = earth_r if physical_body_sizes else max(earth_r, span*.03)
    # Include enlarged bodies as well as their physical bounds.
    lower, upper = np.minimum(lower, -moon_display), np.maximum(upper, moon_display)
    if draw_earth:
        lower, upper = np.minimum(lower, earth_pos-earth_display), np.maximum(upper, earth_pos+earth_display)
    center, half = (lower+upper)/2, max(upper-lower)/2
    lower, upper = center-half, center+half
    span = 2*half
    sun_hat = None
    earth_sun_hat = None
    if sun_azimuth_deg is not None or sun_elevation_deg is not None:
        az, el = np.radians([sun_azimuth_deg or 0., sun_elevation_deg or 0.])
        sun_hat = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        earth_sun_hat = sun_hat
    elif show_sun or show_sun_graphic:
        sun_gcrf = np.asarray(get_sun_position(epoch)).reshape(3)
        sun_hat = rotation @ (sun_gcrf-moon_gcrf)
        sun_hat /= np.linalg.norm(sun_hat)
        earth_sun_hat = rotation @ sun_gcrf
        earth_sun_hat /= np.linalg.norm(earth_sun_hat)

    fig = plt.figure(dpi=120, figsize=figsize, facecolor='#000008')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('#000008')
    ax.set_position([.04, .12, .92, .76])
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    ax.set_box_aspect([1, 1, 1], zoom=view_zoom)
    if show_stars:
        _scene_stars(ax, rotation, epoch, elev, azim, mag_limit)
    for index, xyz in enumerate(tracks):
        color = '#ff9234' if len(tracks) == 1 else plt.cm.rainbow(index/max(1, len(tracks)-1))
        segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
        ax.add_collection3d(Line3DCollection(segments, colors=[color], linewidth=1.5, zorder=15))
    _textured_body(ax, np.zeros(3), moon_display, _ssapy_texture('moon'), moon_rotation,
                   sun_hat, ambient=shade_ambient, diffuse=shade_diffuse, color=BODY_COLORS["moon"])
    if draw_earth:
        earth_artist = _textured_body(ax, earth_pos, earth_display, _ssapy_texture('earth'), earth_rotation,
                                      earth_sun_hat, n=90, ambient=shade_ambient, diffuse=shade_diffuse, color=BODY_COLORS["earth"])
        el, az = np.radians([elev, azim])
        camera = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        earth_artist.set_zorder(11 if earth_pos @ camera > 0 else 9)
        ax.text(*earth_pos, '  Earth', color='white', zorder=30)
        ax.text(0, 0, 0, '  Moon', color='white', zorder=30)
        ax.plot(*np.stack((np.zeros(3), earth_pos)).T, color='#60738a',
                linestyle=':', linewidth=.8, zorder=4)
    if show_sun_graphic and sun_hat is not None:
        symbol_pos = center + sun_hat*span*.35
        render_sun(ax, symbol_pos, radius=background_sun_radius(span, size_factor=.018),
                   elev_deg=elev, azim_deg=azim)
        ax.text(*symbol_pos, '  Sun direction only', color='white', fontsize=8, zorder=30)
    if show_lagrange and r_frame == 'gcrf':
        from ..orbital_mechanics import lagrange_points_lunar_fixed_frame
        for name, pos in lagrange_points_lunar_fixed_frame().items():
            pos = pos/1000.
            if np.all(pos >= lower) and np.all(pos <= upper):
                ax.scatter(*pos, color='white', s=15, zorder=20)
                ax.text(*pos, ' '+name+' (approx.)', color='white', fontsize=8, zorder=30)
    axes_label = 'Moon-centred inertial axes' if r_frame == 'moon_centered' else 'Earth–Moon rotating axes'
    fig.suptitle(title or 'Textured Moon', color='white', fontsize=17, y=.96)
    fig.text(.5, .90, f'{axes_label}  |  {epoch.utc.isot[:19]} UTC', ha='center', color='#ccd6e0', fontsize=10, zorder=50)
    scale_note = f'Moon radius {moon_r:,.1f} km'
    if moon_display != moon_r or (draw_earth and earth_display != earth_r):
        scale_note += f'  |  display radii: Moon ×{moon_display/moon_r:.1f}'
        if draw_earth:
            scale_note += f', Earth ×{earth_display/earth_r:.1f}'
    else:
        scale_note += '  |  physical body and orbit scale'
    if draw_earth:
        scale_note += f'\nCentre-to-centre distance {np.linalg.norm(earth_pos):,.0f} km at the stated epoch'
    else:
        scale_note += f'\nEarth is {np.linalg.norm(earth_pos):,.0f} km from the Moon, outside this close-up'
    fig.text(.5, .06, scale_note, ha='center', color='#ccd6e0', fontsize=10, zorder=50)
    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


if __name__ == "__main__":
    # ── GUI entry point ──────────────────────────────────────────────────────
    # toolkit_gui.py's run_script() writes the Export Plots tab's settings
    # (orbital elements + the shade_ambient / shade_diffuse sliders) to a
    # small Python file and points the GUI_CONFIG env var at it, then runs
    # this script via `conda run -n <env> python moon_plot_3d.py`. Read
    # that config here, build a real propagated trajectory via the
    # toolkit's own OrbitalState, and pass everything into moon_plot_3d.
    import os

    cfg = {}
    cfg_path = os.environ.get("GUI_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            exec(f.read(), cfg)
    else:
        print("[moon_plot_3d] No GUI_CONFIG found — running with defaults.")

    output_dir = cfg.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "moon_plot_3d.jpg")

    # Build and propagate the orbit via the toolkit's real OrbitalState —
    # now a genuine same-folder sibling (core/ was moved into
    # ssatk/plots/), so this is a normal relative import, same as
    # .sun_mpl/.sun_render above.
    r_m, t_gps = None, None
    try:
        from .orbit_state import OrbitalState

        state = OrbitalState(
            a_km=cfg.get("a_km", 6928.0),
            e=cfg.get("e", 0.001),
            inc_deg=cfg.get("inc_deg", 51.6),
            raan_deg=cfg.get("raan_deg", 0.0),
            argp_deg=cfg.get("argp_deg", 0.0),
            nu_deg=cfg.get("nu_deg", 0.0),
            epoch=cfg.get("epoch", None),
        )
        traj = state.propagate(
            n_orbits=cfg.get("n_orbits", 3.0),
            dt_s=cfg.get("dt_s", 60.0),
        )
        if traj.ok:
            r_m = traj.r * 1e3   # Trajectory.r is km; moon_plot_3d expects meters
            t_gps = traj.t
        else:
            print(f"[moon_plot_3d] Propagation failed: {traj.msg} — "
                  f"rendering Moon with no orbit track.")
    except Exception as ex:
        print(f"[moon_plot_3d] Could not build orbit from config: {ex} — "
              f"rendering Moon with no orbit track.")

    fig, ax = moon_plot_3d(
        r=r_m, t=t_gps,
        title=cfg.get("title", ""),
        show_lagrange=cfg.get("show_lagrange", True),
        show_stars=cfg.get("show_stars", True),
        show_sun=cfg.get("show_sun", True),
        shade_ambient=cfg.get("shade_ambient", 0.22),
        shade_diffuse=cfg.get("shade_diffuse", 0.78),
        sun_azimuth_deg=cfg.get("sun_azimuth_deg", None),
        sun_elevation_deg=cfg.get("sun_elevation_deg", None),
        save_path=out_path,
        show=False,
    )
    print(f"[moon_plot_3d] Saved -> {out_path}")
