"""R1/R2 geometry and C export checks; not measured spacecraft telemetry."""
import importlib

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from PIL import Image

from ssatk.constants import EARTH_RADIUS, RGEO
from ssatk.plots import _globe_scene as scene


@pytest.fixture
def texture(tmp_path, monkeypatch):
    # Deliberately asymmetric north/south and east/west, so flips or longitude
    # shifts cannot accidentally pass. Same projection as the installed map.
    rgb = np.zeros((6, 12, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(12) * 20
    rgb[..., 1] = np.arange(6)[:, None] * 40
    rgb[..., 2] = 90
    path = tmp_path / 'earth.png'
    Image.fromarray(rgb).save(path)
    monkeypatch.setattr(scene, 'viewer_earth_day_path', lambda: path)
    return rgb


def _transform(points, epoch, source, target):
    representation = CartesianRepresentation(np.asarray(points).T * u.m)
    result = source(representation, obstime=epoch).transform_to(target(obstime=epoch))
    return result.cartesian.xyz.to_value(u.m).T


def _literal_sphere():
    # Independent spherical-coordinate definition, north to south, west to east.
    out = []
    for north in np.arange(90., -90., -30.):
        for west in np.arange(-180., 180., 30.):
            for lon, lat in [(west, north), (west+30, north),
                             (west+30, north-30), (west, north-30)]:
                lon, lat = np.radians([lon, lat])
                out.append(EARTH_RADIUS * np.array([np.cos(lat)*np.cos(lon),
                                                    np.cos(lat)*np.sin(lon), np.sin(lat)]))
    return np.array(out)


def _front_faces(vertices, elevation=90., azimuth=0.):
    # A sphere normal is radial. Preserve all cells touching the visible side
    # of the orthographic silhouette, including cells straddling its edge.
    el, az = np.radians([elevation, azimuth])
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    faces = vertices.reshape(-1, 4, 3)
    return faces[(faces @ view).max(axis=1) >= 0].reshape(-1, 3)


def test_texture_cells_follow_north_up_east_positive_projection(texture):
    """R1/C: exact RGB cells, unit sphere and closed seam; 0.1 micrometre float64 budget."""
    quads, colors = scene.earth_surface(12)
    np.testing.assert_array_equal(colors * 255, texture.reshape(-1, 3))
    # Trigonometric argument construction and metre scaling each round; 0.1 um
    # is < 128 ulps at Earth radius and does not hide a texture-cell displacement.
    np.testing.assert_allclose(quads.reshape(-1, 3) * EARTH_RADIUS,
                               _literal_sphere(), rtol=0, atol=1e-7)


@pytest.mark.parametrize('frame', ['gcrf', 'itrf'])
@pytest.mark.parametrize('distance', [7e6, 45e6], ids=['km-axis', 'GEO-axis'])
def test_rendered_earth_and_markers_at_each_epoch_match_astropy(texture, monkeypatch, tmp_path, frame, distance, submitted_vertices):
    """R2/R1: actual frame vertices and positions vs Astropy GCRS/ITRS.

    The IAU 1976/1980 SSAPy chain differs from Astropy IAU 2006/2000A. Budget:
    0.5 arcsec times radius; 20 nm for exact GCRF positions. Epoch labels
    must be exact UTC even when the input Time array is TT. Tests both axis units.
    """
    module = importlib.import_module('ssatk.plots.orbit_animation')
    epochs = Time(['2025-01-01T00:00:00', '2025-01-01T03:00:00', '2025-01-01T06:00:00'])
    r = distance * np.array([[1., 0, 0], [0, 1., 0], [-1., 0, 0]])
    unit = RGEO if distance == 45e6 else 1000.
    captured = []

    class Capture:
        def __init__(self, fig, update, frames, **kwargs):
            self.fig, self.update, self.frames = fig, update, frames
        def save(self, *args, **kwargs):
            for index in range(self.frames):
                self.update(index)
                ax = self.fig.axes[0]
                earth = next(c for c in ax.collections if c.get_zorder() == 1)
                head = ax.lines[0]
                captured.append((submitted_vertices(earth) * unit, np.asarray(head.get_data_3d()).T * unit,
                                 [text.get_text() for text in ax.texts]))
    monkeypatch.setattr(module, 'FuncAnimation', Capture)
    # Looking down the pole keeps the equatorial test spacecraft unocculted
    # even after its ITRF coordinate rotation through six hours.
    module.orbit_animation(r, epochs.tt, views=('globe',), frame=frame, el=90, az=0,
                           earth_resolution=12,
                           max_frames=2, save_path=tmp_path/'capture.gif')
    for (earth, position, labels), index in zip(captured, [0, 2]):
        epoch = epochs[index]
        expected_position = r[index:index+1]
        fixed = _literal_sphere()
        expected_earth = _transform(fixed, epoch, ITRS, GCRS) if frame == 'gcrf' else fixed
        expected_position = _transform(expected_position, epoch, GCRS, ITRS) if frame == 'itrf' else expected_position
        earth_budget = EARTH_RADIUS * np.radians(.5/3600) if frame == 'gcrf' else 1e-7
        position_budget = distance * np.radians(.5/3600) if frame == 'itrf' else 2e-8
        np.testing.assert_allclose(earth, _front_faces(expected_earth), rtol=0, atol=earth_budget)
        np.testing.assert_allclose(position, expected_position, rtol=0, atol=position_budget)
        assert f'{epoch.utc.isot} UTC · {frame.upper()}' in labels


@pytest.mark.parametrize('frame', ['gcrf', 'itrf'])
def test_still_uses_selected_epoch_and_marker_position(texture, frame, submitted_vertices):
    """R2/C: static public dispatch also honours frame and reference date (0.5")."""
    from ssatk.plots import orbit_plot
    from matplotlib import pyplot as plt
    epochs = Time(['2025-01-01', '2025-01-02'])
    r = np.array([[7e6, 0, 0], [0, 7e6, 0]])
    fig, axes = orbit_plot(r, epochs, view='globe', frame=frame, el=90, az=0,
                          earth_resolution=12, reference_index=0)
    try:
        earth = next(c for c in axes[0].collections if c.get_zorder() == 1)
        marker = next(c for c in axes[0].collections if c.get_zorder() == 10)
        fixed = _literal_sphere()
        expected_earth = _transform(fixed, epochs[0], ITRS, GCRS) if frame == 'gcrf' else fixed
        expected_position = r[:1]
        if frame == 'itrf':
            expected_position = _transform(expected_position, epochs[0], GCRS, ITRS)
        np.testing.assert_allclose(submitted_vertices(earth) * RGEO, expected_earth, rtol=0, atol=16)
        np.testing.assert_allclose(np.array(marker._offsets3d).T * RGEO, expected_position, rtol=0, atol=18)
    finally:
        plt.close(fig)


def test_rendered_pixels_hide_marker_and_trail_behind_earth(texture, monkeypatch, tmp_path):
    """R1/C: +x view of opaque radius R; a point at -1.1R must vanish.

    Inspect actual raster pixels, including the animation update/drawing order.
    The green head marker must show at least ten solid pixels in front,
    while the hidden frame must have none, including its occulted trail.
    """
    module = importlib.import_module('ssatk.plots.orbit_animation')
    monkeypatch.setattr(module._cm, 'rainbow', lambda values: np.tile([0, 1., 0, 1.], (len(values), 1)))
    counts = []
    class Capture:
        def __init__(self, fig, update, frames, **kwargs):
            self.fig, self.update, self.frames = fig, update, frames
        def save(self, *args, **kwargs):
            for index in range(self.frames):
                self.update(index)
                self.fig.canvas.draw()
                pixels = np.asarray(self.fig.canvas.buffer_rgba())
                green = (pixels[..., 1] > 230) & (pixels[..., 0] < 30) & (pixels[..., 2] < 30)
                counts.append(int(green.sum()))
                if index == 1:
                    from mpl_toolkits.mplot3d.art3d import Line3DCollection
                    trails = [c for c in self.fig.axes[0].collections if isinstance(c, Line3DCollection)]
                    # Source paths are retained between draws; geometric
                    # clipping, rather than zeroing their alpha, hides them.
                    assert trails
                    np.testing.assert_array_equal(trails[0]._segments3d,
                                                  [[[7000., 0, 0], [-7000., 0, 0]]])
    monkeypatch.setattr(module, 'FuncAnimation', Capture)
    module.orbit_animation(np.array([[7e6, 0, 0], [-7e6, 0, 0]]), 1.4e9+np.arange(2.),
                           views=('globe',), el=0, az=0, earth_resolution=12,
                           save_path=tmp_path/'hidden.gif')
    assert counts[0] >= 10 and counts[1] == 0


def test_constellation_renders_empty_first_trails_and_occulted_second_frame(monkeypatch, tmp_path):
    """R1/C: 32 synthetic tracks enter the batched globe export path.

    Viewed along +X, their first positions at +7000 km are in front of Earth;
    their second positions at -7000 km, |y| <= 100 km are behind its disc.
    Draw real frames to catch empty-collection autoscaling failures, and check
    rendered head pixels against that literal ray/sphere visibility oracle.
    """
    module = importlib.import_module('ssatk.plots.orbit_animation')
    monkeypatch.setattr(module._cm, 'rainbow', lambda values: np.tile([0, 1., 0, 1.], (len(values), 1)))
    tracks = [np.array([[7e6, y, 0.], [-7e6, y, 0.]])
              for y in np.linspace(-1e5, 1e5, 32)]
    counts = []

    class Capture:
        def __init__(self, fig, update, frames, **kwargs):
            self.fig, self.update, self.frames = fig, update, frames

        def save(self, *args, **kwargs):
            ax = self.fig.axes[0]
            limits = None
            for index in range(self.frames):
                self.update(index)
                self.fig.canvas.draw()
                if limits is None:
                    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
                np.testing.assert_array_equal(
                    [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], limits)
                pixels = np.asarray(self.fig.canvas.buffer_rgba())
                green = (pixels[..., 1] > 230) & (pixels[..., 0] < 30) & (pixels[..., 2] < 30)
                counts.append(int(green.sum()))

    monkeypatch.setattr(module, 'FuncAnimation', Capture)
    module.orbit_animation(tracks, 1.4e9+np.arange(2.), views=('globe',),
                           el=0, az=0, earth_resolution=12, low_spec=True,
                           save_path=tmp_path/'constellation.gif')
    assert len(counts) == 2 and counts[0] >= 10 and counts[1] == 0


@pytest.mark.parametrize('bad_time', ['missing', 'repeated', 'reversed', 'different-tracks'])
def test_dated_animation_rejects_ambiguous_physical_epochs(tmp_path, bad_time):
    """C: never label an Earth date while displaying another track's epoch."""
    from ssatk.plots import orbit_plot
    r = np.array([[7e6, 0, 0], [0, 7e6, 0]])
    t = {'missing': None, 'repeated': np.array([1.4e9, 1.4e9]),
         'reversed': np.array([1.4e9+1, 1.4e9]),
         'different-tracks': [np.array([1.4e9, 1.4e9+1]), np.array([1.4e9+1, 1.4e9+2])]}[bad_time]
    if bad_time == 'different-tracks':
        r = [r, r]
    with pytest.raises(ValueError, match='times|time grid'):
        orbit_plot(r, t, view='globe', save=tmp_path/'invalid.gif')


def test_still_does_not_highlight_a_position_from_a_different_epoch(texture):
    """C: explicit Earth date cannot borrow a satellite marker from another time."""
    from ssatk.plots.globe_plot import globe_plot
    from matplotlib import pyplot as plt
    fig, ax = globe_plot(np.array([[7e6, 0, 0], [0, 7e6, 0]]),
                        t=np.array([1.4e9, 1.4e9+10]), globe_time=1.4e9+5,
                        earth_resolution=12)
    try:
        assert not any(c.get_zorder() == 10 for c in ax.collections)
    finally:
        plt.close(fig)
