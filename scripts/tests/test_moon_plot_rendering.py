"""R1/R2 lunar geometry and C image contracts; synthetic states, not flight data."""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import get_body_barycentric, get_sun
from astropy.time import Time
from PIL import Image
from scipy.optimize import brentq

from demos.orbit_visualization.demo_moon_plot import main

scene = importlib.import_module('ssatk.plots.moon_plot_3d')


@pytest.mark.parametrize('fast', [True, False])
def test_demo_matches_independent_lunar_kepler_ellipse(fast):
    """R1: scalar Kepler equation, 1 mm / 1 um/s including GPS epoch rounding.

    The lunar GM below is the numerical DE200 value used by SSAPy. The oracle
    does not read orbit.mu, so an accidental Earth-GM default cannot pass.
    """
    result = main(make_figures=False, fast=fast)
    a, e, mu = 2738100., .01, 4.902799059741106e12
    elapsed = result['t'] - Time('2025-01-01T00:00:00').gps
    mean = elapsed * np.sqrt(mu/a**3)
    anomaly = np.array([brentq(lambda E: E-e*np.sin(E)-M, M-e-1e-9, M+e+1e-9,
                               xtol=1e-14) for M in mean])
    expected_r = np.column_stack((a*(np.cos(anomaly)-e), np.zeros(len(mean)),
                                  a*np.sqrt(1-e*e)*np.sin(anomaly)))
    rate = np.sqrt(mu/a**3)/(1-e*np.cos(anomaly))
    expected_v = np.column_stack((-a*np.sin(anomaly)*rate, np.zeros(len(mean)),
                                  a*np.sqrt(1-e*e)*np.cos(anomaly)*rate))
    np.testing.assert_allclose(result['r'], expected_r, rtol=0, atol=.001)
    np.testing.assert_allclose(result['v'], expected_v, rtol=0, atol=1e-6)
    assert abs(mean[-1]-2*np.pi) < 1e-10
    assert np.max(np.diff(mean)) <= np.radians(1)+1e-10


def test_texture_vertices_and_texels_follow_literal_longitude_latitude(tmp_path):
    """R1/C: quarter-turned north-up map; 1e-12 km vertex budget, exact texels."""
    rgb = np.arange(3*6*3, dtype=np.uint8).reshape(3, 6, 3)*4
    path = tmp_path/'map.png'
    Image.fromarray(rgb).save(path)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    center = np.array([10., 20., 30.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    body = scene._textured_body(ax, center, 5., path, rotation, None, n=3)
    expected = []
    for north in [90, 30, -30]:
        for west in [-180, -120, -60, 0, 60, 120]:
            for lon, lat in [(west, north), (west+60, north), (west+60, north-60), (west, north-60)]:
                lon, lat = np.radians([lon+90, lat])
                expected.append(center+5*np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]))
    np.testing.assert_allclose(body._vec[:3].T, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(body._facecolor3d[:, :3]*255, rgb.reshape(-1, 3))
    plt.close(fig)


@pytest.mark.parametrize('frame', ['gcrf', 'moon_centered'])
@pytest.mark.parametrize('date', ['2025-01-01', '2025-06-01'])
def test_bodies_lighting_and_bounds_match_independent_ephemeris(monkeypatch, frame, date):
    """R2: Astropy ERFA Moon (20 km) and apparent Sun (0.02 deg) vs DE440.

    These budgets cover independent ephemeris model differences and solar
    aberration, not arbitrary display offsets. Tidal locking constrains the
    rendered prime meridian to 9 deg of Earth (optical libration envelope).
    """
    epoch = Time(date)
    capture = []
    original = scene._textured_body
    def draw(ax, center, radius, path, rotation, sun_hat, **kwargs):
        artist = original(ax, center, radius, None, rotation, sun_hat, n=6)
        capture.append((np.array(center), radius, rotation, sun_hat, artist._vec[:3].T))
        return artist
    monkeypatch.setattr(scene, '_textured_body', draw)
    fig, ax = scene.moon_plot_3d(t=epoch.tt, r_frame=frame, show_earth=True,
                                 physical_body_sizes=True, show_stars=False, show_lagrange=False)
    moon, earth = capture
    expected = (get_body_barycentric('earth', epoch)-get_body_barycentric('moon', epoch)).xyz.to_value(u.km)
    expected_distance = np.linalg.norm(expected)
    if frame == 'gcrf':
        np.testing.assert_allclose(earth[0], [-expected_distance, 0., 0.], rtol=0, atol=20.)
    else:
        np.testing.assert_allclose(earth[0], expected, rtol=0, atol=20.)
        sun = get_sun(epoch).cartesian.xyz.to_value(u.km) + expected
        sun /= np.linalg.norm(sun)
        assert np.degrees(np.arccos(np.clip(sun @ moon[3], -1, 1))) < .02
    # Inspect the orientation used on the mesh: map's longitude zero is +X.
    toward_earth = earth[0]/np.linalg.norm(earth[0])
    assert np.degrees(np.arccos(np.clip(toward_earth @ moon[2][:, 0], -1, 1))) < 9.
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    for center, radius, rotation, sunlight, vertices in capture:
        assert np.all(vertices >= limits[:, 0]-1e-8)
        assert np.all(vertices <= limits[:, 1]+1e-8)
        # The SSAPy-compatible polar-motion matrix is first-order in xp/yp.
        # For |xp|, |yp| < 1 arcsec, norm error <= R*(1 arcsec)^2;
        # retain 1e-8 km for subtracting a lunar-distance centre in float64.
        radius_budget = radius*np.radians(1/3600)**2 + 1e-8
        np.testing.assert_allclose(np.linalg.norm(vertices-center, axis=1), radius, rtol=0, atol=radius_budget)
    # C: a surface-only Time input must set the date (including TT -> UTC).
    assert any(epoch.utc.isot[:19]+' UTC' in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_short_arc_keeps_moon_in_view_and_hides_far_side_pixels(monkeypatch):
    """R1/C: orthographic opaque sphere; visible orange front line, hidden back line."""
    original = scene._textured_body
    monkeypatch.setattr(scene, '_textured_body', lambda ax, center, radius, path, rotation, sun_hat, **kw:
                        original(ax, center, radius, None, rotation, None, n=30))
    # Camera on +X. Both tracks project to the disc, one in front, one behind.
    front = np.array([[2500000., -200000., 0.], [2500000., 200000., 0.]])
    counts = []
    for points in [front, front*np.array([-1, 1, 1])]:
        fig, ax = scene.moon_plot_3d(r=points, t=np.array([1.4e9, 1.4e9+10]),
            r_frame='moon_centered', elev=0, azim=0, show_sun=False, show_stars=False,
            show_lagrange=False, physical_body_sizes=True, scene_pad=.05, figsize=(4, 4))
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        assert np.all(limits[:, 0] <= -1738.1) and np.all(limits[:, 1] >= 1738.1)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
        orange = (pixels[..., 0] > 160) & (pixels[..., 1] > 65) & (pixels[..., 1] < 190) & (pixels[..., 2] < 100)
        counts.append(int(orange.sum()))
        assert ((pixels.max(axis=2)-pixels.min(axis=2) < 2) & (pixels[..., 0] > 100)).sum() > 1000
        plt.close(fig)
    assert counts[0] > 10 and counts[1] == 0


def test_sky_uses_rotated_catalogue_directions_and_excludes_camera_back(monkeypatch):
    """R1/C: literal 45-deg and central sightlines in a 120-deg gnomonic sky."""
    stars = importlib.import_module('ssatk.plots.starfield')
    directions = np.array([[0, 1, 0], [1, 1, 0], [0, -1, 0]], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    monkeypatch.setattr(stars, '_load_stars', lambda **kw: dict(v=directions, sizes=np.ones(3), rgb=np.ones((3, 3))))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Rotate +Y into -X, the camera's central forward direction.
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1]])
    scene._scene_stars(ax, rotation, Time('2025-01-01'), 0., 0., 5.)
    points = np.asarray(fig.axes[1].collections[0].get_offsets())
    np.testing.assert_allclose(points, [[0, 0], [1/np.sqrt(3), 0]], rtol=0, atol=1e-15)
    plt.close(fig)


@pytest.mark.parametrize('kwargs', [dict(r=np.ones((2, 3))), dict(r_frame='typo'), dict(scene_pad=-1)])
def test_invalid_scene_inputs_are_rejected(kwargs):
    """C only: do not silently invent an epoch/frame or negative extent."""
    with pytest.raises(ValueError):
        scene.moon_plot_3d(**kwargs)


def test_demo_rendered_orbit_surrounds_moon_without_second_origin_shift(monkeypatch):
    """R1/C: end-to-end demo wiring retains 972.619..1027.381 km altitudes."""
    demo = importlib.import_module('demos.orbit_visualization.demo_moon_plot')
    original = scene._textured_body
    monkeypatch.setattr(scene, '_textured_body', lambda ax, center, radius, path, rotation, sun_hat, **kw:
                        original(ax, center, radius, None, rotation, None, n=6))
    records = []
    def render(**kwargs):
        kwargs.update(save_path=False, show_stars=False)
        fig, ax = scene.moon_plot_3d(**kwargs)
        segments = [np.asarray(c._segments3d) for c in ax.collections
                    if isinstance(c, scene.Line3DCollection)]
        records.append(segments)
        return fig, ax
    monkeypatch.setattr(demo, 'moon_plot_3d', render)
    main(make_figures=True, fast=True)
    assert len(records) == 4
    for segments in records[:2]:
        vertices = np.concatenate(segments).reshape(-1, 3)
        radius = np.linalg.norm(vertices, axis=1)
        np.testing.assert_allclose([radius.min(), radius.max()], [2710.719, 2765.481], rtol=0, atol=1e-6)
    assert records[2:] == [[], []]
