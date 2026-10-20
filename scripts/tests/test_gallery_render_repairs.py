"""R1/R2 numerical references and C rendering contracts for gallery repairs."""
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import pytest
from astropy.time import Time

from ssatk.plots import low_spec_mode
from ssatk.plots._legend_handlers import BodyMapPatch, BodyMapHandler, body_legend_handle


def test_body_legend_map_orientation_and_low_mode(tmp_path, monkeypatch):
    """C: the actual legend image preserves north/east orientation; low mode does no I/O."""
    textures = importlib.import_module('ssatk.plots._textures')
    rgb = np.zeros((64, 128, 3), dtype=np.uint8)
    rgb[:32, :64] = [255, 0, 0]
    rgb[:32, 64:] = [0, 255, 0]
    rgb[32:, :64] = [0, 0, 255]
    rgb[32:, 64:] = [255, 255, 0]
    path = tmp_path/'map.png'
    Image.fromarray(rgb).save(path)
    monkeypatch.setattr(textures, 'earth_texture_path', lambda: path)
    monkeypatch.setattr(textures, 'moon_texture_path', lambda: path)
    monkeypatch.setattr(textures, 'viewer_earth_day_path', lambda: tmp_path/'absent.jpg')
    for body in ('earth', 'moon'):
        with low_spec_mode(False):
            handle = body_legend_handle(body)
        fig, ax = plt.subplots()
        legend = ax.legend(handles=[handle], handler_map={BodyMapPatch: BodyMapHandler()})
        fig.canvas.draw()
        artists = legend.findobj(BboxImage)
        assert len(artists) == 1 and artists[0].origin == 'upper'
        np.testing.assert_array_equal(artists[0].get_array()*255, rgb)
        assert artists[0].get_window_extent().width > artists[0].get_window_extent().height
        plt.close(fig)
    monkeypatch.setattr(Image, 'open', lambda *a, **k: pytest.fail('low mode read a map'))
    with low_spec_mode():
        for body, color in [('earth', (.10, .31, .54, 1)), ('moon', (1, 1, 1, 1))]:
            handle = body_legend_handle(body)
            assert type(handle) is Patch
            np.testing.assert_array_equal(handle.get_facecolor(), color)


def test_animation_map_pixels_and_panel_bounds_at_first_middle_last_frame(monkeypatch, tmp_path):
    """C: an asymmetric world map is rendered behind the track, without panel overlap."""
    animation = importlib.import_module('ssatk.plots.orbit_animation')
    plotutils = importlib.import_module('ssatk.plots.plotutils')
    rgb = np.zeros((30, 60, 3), dtype=np.uint8)
    rgb[:15, :30] = [255, 0, 0]
    rgb[15:, 30:] = [0, 255, 0]
    monkeypatch.setattr(plotutils, 'load_earth_file', lambda: rgb)
    captures = []
    class Capture:
        def __init__(self, fig, update, frames, **kw):
            self.fig, self.update, self.frames = fig, update, frames
        def save(self, *args, **kw):
            for i in (0, self.frames//2, self.frames-1):
                self.update(i)
                self.fig.canvas.draw()
                renderer = self.fig.canvas.get_renderer()
                axes = self.fig.axes
                np.testing.assert_array_equal(axes[0].images[0].get_array(), rgb)
                assert axes[0].images[0].origin == 'upper'
                np.testing.assert_array_equal(axes[0].images[0].get_extent(), [-180, 180, -90, 90])
                bounds = [ax.get_tightbbox(renderer) for ax in axes]
                for j, a in enumerate(bounds):
                    for b in bounds[j+1:]:
                        assert not a.overlaps(b), (a.bounds, b.bounds)
                pixels = np.asarray(self.fig.canvas.buffer_rgba())
                assert np.sum(np.all(pixels[..., :3] == [255, 0, 0], axis=-1)) > 100
                captures.append(i)
    monkeypatch.setattr(animation, 'FuncAnimation', Capture)
    angle = np.linspace(0, 2*np.pi, 9)
    r = 4.2e7*np.column_stack([np.cos(angle), np.sin(angle), .1*np.sin(angle)])
    animation.orbit_animation(r, 1.4e9+np.linspace(0, 86400, 9),
        views=('groundtrack', 'globe', 'xy'), figsize=(12, 9), earth_resolution=12,
        save_path=tmp_path/'dashboard.gif', low_spec=True)
    assert captures == [0, 4, 8]


def test_groundtrack_dashboard_speed_uses_nonuniform_seconds():
    """R1: constant (3,4,0) km/s on nonuniform samples must plot 5 km/s."""
    from ssatk.plots.groundtrack_dashboard import groundtrack_dashboard
    elapsed = np.array([0., 10., 35., 90., 200.])
    r = np.array([7e6, 0, 0]) + elapsed[:, None]*np.array([3000., 4000., 0])
    for step in (1, -1):
        fig = groundtrack_dashboard(r[::step], Time((1.4e9+elapsed)[::step], format='gps'))
        try:
            speed = next(ax for ax in fig.axes if ax.get_ylabel() == 'Velocity (km/s)')
            np.testing.assert_allclose(speed.lines[0].get_ydata(), 5., rtol=0, atol=1e-12)
        finally:
            plt.close(fig)
    fig = groundtrack_dashboard(r[:1], Time([1.4e9], format='gps'))
    try:
        speed = next(ax for ax in fig.axes if ax.get_ylabel() == 'Velocity (km/s)')
        assert np.isnan(speed.lines[0].get_ydata()).all()
        assert any('two epochs' in text.get_text() for text in speed.texts)
    finally:
        plt.close(fig)


def test_static_3d_orbit_panel_contains_supplied_positions():
    """C: body rendering must not drop the trajectory from the 3-D panel."""
    from ssatk.plots import orbit_plot
    r = np.array([[7e6, 0, 0], [0, 7e6, 0], [-7e6, 0, 1e6]])
    fig, axes = orbit_plot(r, t=1.4e9+np.arange(3.), view='3d', low_spec=True)
    try:
        track = next(c for c in axes[0].collections if c.get_label() == 'Orbit Path')
        np.testing.assert_allclose(np.array(track._offsets3d).T, r/1000, rtol=0, atol=1e-12)
    finally:
        plt.close(fig)


def test_lunar_map_rotation_against_cspice_fixtures():
    """R2: 50 independent CSPICE epochs, 2000–2040 + weekly 2025 samples.

    2 cm at lunar radius covers SSAPy's TT evaluation of a TDB kernel (<9 mm)
    and floating-point time conversion. No live downloads or SPICE dependency.
    """
    from ssatk.coordinates.lunar import lunar_map_rotation
    from ssatk.plots._body_surfaces import body_rotation
    rows = json.loads((Path(__file__).parent/'data/lunar_map_spice.json').read_text())['samples']
    epoch = Time([row['utc'] for row in rows])
    expected = np.array([row['gcrf_to_me'] for row in rows])
    for t in (epoch, epoch.tt, epoch.gps):
        np.testing.assert_allclose(lunar_map_rotation(t), expected, rtol=0, atol=.02/1738100.)
    for i in (0, 1, 10, 25, 49):
        np.testing.assert_allclose(body_rotation('moon', epoch[i]), expected[i].T,
                                   rtol=0, atol=.02/1738100.)


def test_walker_states_match_literal_circular_orbits():
    """R1: independently rotated circular states, 1 mm / 1 um/s propagation budget."""
    from ssatk.orbital_mechanics.constellations import walker_delta
    from ssapy import rv
    orbits = walker_delta(planes=3, satellites_per_plane=4, inclination_deg=53., phasing=1)
    mu, radius = 3.986004418e14, 6378137.+550e3
    elapsed = np.array([0., 30., 300., 1000.])
    r, v = rv(orbits, Time(orbits[0].t+elapsed, format='gps'))
    expected_r, expected_v = [], []
    for plane in range(3):
        angle = 2*np.pi*plane/3
        ci, si, co, so = np.cos(np.deg2rad(53)), np.sin(np.deg2rad(53)), np.cos(angle), np.sin(angle)
        basis = np.array([[co, -so*ci], [so, co*ci], [0, si]])
        for slot in range(4):
            phase = 2*np.pi*(slot/4+plane/12)+np.sqrt(mu/radius**3)*elapsed
            expected_r.append(radius*np.column_stack([np.cos(phase), np.sin(phase)])@basis.T)
            expected_v.append(np.sqrt(mu/radius)*np.column_stack([-np.sin(phase), np.cos(phase)])@basis.T)
    np.testing.assert_allclose(r, expected_r, rtol=0, atol=.001)
    np.testing.assert_allclose(v, expected_v, rtol=0, atol=1e-6)


def test_batched_constellation_keeps_visible_heads_and_breaks_dateline():
    """R1/C: opaque sphere +/-X geometry and explicit +/-179-degree map crossing."""
    from ssatk.plots.orbit_animation import _ConstellationArtists
    fig = plt.figure()
    globe = fig.add_subplot(121, projection='3d')
    globe.view_init(elev=0, azim=0)
    globe.set_proj_type('ortho')
    globe.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3))
    from ssatk.plots._body_surfaces import BodySurface
    from ssatk.plots.rendering import low_spec_mode
    with low_spec_mode():
        BodySurface(globe, 'earth', 1., n=12)
    ground = fig.add_subplot(122)
    tracks = [dict(xyz=np.array([[2., 0, 0], [2., 0, 0]]), groundtrack=np.array([[179., 0], [-179., 0]])),
              dict(xyz=np.array([[-2., 0, 0], [-2., 0, 0]]), groundtrack=np.array([[10., 0], [11., 0]]))]
    batch = _ConstellationArtists(dict(globe=globe, groundtrack=ground), tracks,
                                  np.array([[1., 0, 0, 1], [0, 1., 0, 1]]), 1.)
    batch.update(0, 1)
    drawn = []
    head = batch.heads['globe']
    original_draw = head.draw
    def capture(renderer):
        drawn.append(np.array(head._offsets3d).T.copy())
        return original_draw(renderer)
    head.draw = capture
    fig.canvas.draw()
    np.testing.assert_array_equal(drawn[0][0], [2., 0, 0])
    assert np.isnan(drawn[0][1]).all()
    np.testing.assert_array_equal(np.array(head._offsets3d).T, [[2., 0, 0], [-2., 0, 0]])
    assert len(batch.tails['groundtrack'].get_segments()) == 1
    np.testing.assert_array_equal(batch.tails['groundtrack'].get_segments()[0], [[10., 0], [11., 0]])
    plt.close(fig)


def test_ladder_summary_preserves_zero_and_uses_data_color_range():
    """C: hand-specified metre residuals stay exact, including a zero reference."""
    from ssatk.orbital_mechanics.orbital_accel_model_comparisons import make_accel_ladder_dashboard_figures
    from matplotlib.collections import QuadMesh
    calc = dict(labels=['Kepler', 'Kepler + Moon', 'Reference'], reference=2, worst_idx=0,
                t_rel_s=np.array([0., 10., 40.]),
                drn_vs_ref=np.array([[0., 1., 4.], [0., .1, .4], [0., 0., 0.]]),
                drn_inc=np.array([[0., 0., 0.], [0., .9, 3.6], [0., .1, .4]]),
                final_drn_vs_ref=np.array([4., .4, 0.]),
                final_drn_inc=np.array([0., 3.6, .4]), ntw_worst=np.zeros((3, 3)))
    figures = make_accel_ladder_dashboard_figures(calc)
    try:
        fig = figures['figure_rung_summary']
        final, increment, heat = fig.axes[1:4]
        np.testing.assert_array_equal(final.lines[0].get_ydata(), [4., .4, 0.])
        np.testing.assert_array_equal(increment.lines[0].get_xdata(), [1, 2])
        im = next(a for a in heat.collections if isinstance(a, QuadMesh))
        assert im.norm.vmax == 4 and im.norm.vmin == .001
        assert np.ma.getmaskarray(im.get_array())[2].all()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in (final, increment, heat)]
        for i, a in enumerate(boxes):
            for b in boxes[i+1:]:
                assert not a.overlaps(b)
    finally:
        for fig in figures['figures']:
            plt.close(fig)
