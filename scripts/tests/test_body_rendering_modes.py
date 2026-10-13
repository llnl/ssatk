"""R1/R2 surface geometry references and C presentation-mode contracts."""
import importlib

from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from ssatk.plots import low_spec_mode, is_low_spec, set_low_spec_mode
from ssatk.plots._body_surfaces import BodySurface, body_mesh_plotly, body_rotation
from ssatk.plots.moon_render import moon_mesh_plotly


def test_mode_precedence_restoration_and_validation(monkeypatch):
    """C: environment, explicit overrides and exceptions cannot leak a mode."""
    monkeypatch.setenv('SSATK_LOW_SPEC', '1')
    set_low_spec_mode(None)
    assert is_low_spec()
    try:
        with low_spec_mode(False):
            assert not is_low_spec()
            with low_spec_mode():
                assert is_low_spec()
            raise RuntimeError('leave context')
    except RuntimeError:
        pass
    assert is_low_spec()
    with pytest.raises(TypeError):
        set_low_spec_mode('false')
    set_low_spec_mode(False)
    assert not is_low_spec()
    set_low_spec_mode(None)


def test_mapped_vertices_match_literal_longitudes_and_texels(tmp_path):
    """R1/C: exact north-up map and 90-degree rotation; 1e-12 length budget.

    A deliberately asymmetric six-column map exposes half-turn/latitude flips.
    Compare both backend face colours and literal spherical coordinates.
    """
    rgb = np.arange(3*6*3, dtype=np.uint8).reshape(3, 6, 3)*4
    path = tmp_path/'map.png'
    Image.fromarray(rgb).save(path)
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1]])
    center = np.array([10., 20., 30.])
    expected = []
    for north in [90, 30, -30]:
        for west in [-180, -120, -60, 0, 60, 120]:
            for lon, lat in [(west, north), (west+60, north), (west+60, north-60), (west, north-60)]:
                lon, lat = np.radians([lon+90, lat])
                expected.append(center+5*np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]))
    with low_spec_mode(False):
        fig = plt.figure()
        surface = BodySurface(fig.add_subplot(projection='3d'), 'moon', 5., center, rotation, n=3, path=path)
        np.testing.assert_allclose(surface.artist._vec[:3].T, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(surface.artist._facecolor3d[:, :3]*255, rgb.reshape(-1, 3))
        mesh = body_mesh_plotly('moon', center, 5., rotation, n=3, path=path)
        vertices = np.column_stack((mesh.x, mesh.y, mesh.z))
        faces = vertices[np.column_stack((mesh.i, mesh.j, mesh.k))]
        reference = np.asarray(expected).reshape(-1, 4, 3)[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)
        np.testing.assert_allclose(faces, reference, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(mesh.facecolor, np.repeat(rgb.reshape(-1, 3), 2, axis=0))
        plt.close(fig)


@pytest.mark.parametrize('date', ['2025-01-01', '2025-06-01'])
def test_lunar_position_and_map_orientation_against_independent_ephemeris(date):
    """R2: ERFA Moon within 20 km of DE; longitude zero within 9 deg of Earth.

    The orientation budget is the Moon's optical libration envelope, not a
    fitted tolerance. The frame origin must not drift with the render mode.
    """
    from ssatk.plots.layers import MoonLayer
    epoch = Time(date)
    expected = (get_body_barycentric('moon', epoch)-get_body_barycentric('earth', epoch)).xyz.to_value(u.km)
    for low in [False, True]:
        layer = MoonLayer(low_spec=low)
        position = layer._moon_position_km(epoch.gps)
        np.testing.assert_allclose(position, expected, rtol=0, atol=20.)
        rotation = body_rotation('moon', epoch)
        cosine = -expected @ rotation[:, 0] / np.linalg.norm(expected)
        assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 9.
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)


def test_low_mode_skips_all_texture_and_relief_loaders(monkeypatch):
    """C/R1: no texture I/O; reduced geometry remains on the specified sphere."""
    def forbidden(*args, **kwargs):
        raise AssertionError('low mode must not load maps or generate decorative relief')
    monkeypatch.setattr(Image, 'open', forbidden)
    globe = importlib.import_module('ssatk.plots.globe_orbit_daynight_plotly')
    for name in ['_load_real_earth_texture', '_land_mask', '_terrain_relief', '_city_lights']:
        monkeypatch.setattr(globe, name, forbidden)
    from ssatk.plots._globe_scene import earth_surface
    from ssatk.plots.magnetosphere_core import _build_earth_mesh
    from ssatk.plots.sun_view import MoonShadingLayer, EarthShadingLayer
    center = np.array([384400., 0., 0.])
    with low_spec_mode():
        quads, colors = earth_surface(180)
        assert len(quads) == 648
        np.testing.assert_allclose(np.linalg.norm(quads, axis=-1), 1., rtol=0, atol=1e-15)
        assert np.unique(colors, axis=0).shape == (1, 3)
        earth = globe._earth_mesh(np.array([1., 0, 0]))
        np.testing.assert_allclose(np.linalg.norm(np.column_stack((earth.x, earth.y, earth.z)), axis=1), 6378.137, rtol=0, atol=1e-8)
        _build_earth_mesh(None, allow_download=True)
        MoonShadingLayer([1e8, 0, 0], center).build_traces()
        EarthShadingLayer([1e8, 0, 0]).build_traces()
        fig = plt.figure()
        surface = BodySurface(fig.add_subplot(projection='3d'), 'moon', 1738.1, center)
    # Lazy updates outside the context retain their plain map.
    surface.update(center, np.eye(3))
    np.testing.assert_array_equal(surface.colors, np.ones((648, 3)))
    plt.close(fig)
    mesh = moon_mesh_plotly(center, 1738.1, low_spec=True)
    assert len(mesh.x) == 19*36
    np.testing.assert_allclose(np.linalg.norm(np.column_stack((mesh.x, mesh.y, mesh.z))-center, axis=1), 1738.1, rtol=0, atol=1e-10)


def test_spacecraft_resampling_retains_low_mode(monkeypatch):
    """C: stored viewers remain lightweight when exported after context exit."""
    module = importlib.import_module('ssatk.plots.spacecraft_view')
    monkeypatch.setattr(module, '_texture', lambda *args: pytest.fail('unexpected texture load'))
    with low_spec_mode():
        view = module.SpacecraftView([[42164000, 0, 0], [42164000, 1000, 0]], [1.4e9, 1.4e9+1],
            [1, 0, 0, 0], sun_position=[1.5e11, 0, 0], moon_position=[3.8e8, 0, 0], stars=False)
    sampled = view.resample([1.4e9, 1.4e9+.5, 1.4e9+1])
    assert sampled.low_spec and not sampled.textures
    sampled.render(index=1, width=300, height=200)


def test_web_exports_skip_image_assets_and_moon_cache(tmp_path, monkeypatch):
    """C: low-spec portable exports need no Earth images or baked Moon cache."""
    moon = importlib.import_module('ssatk.plots.moon_webgl')
    sat = importlib.import_module('ssatk.plots.build_satellite_viewer')
    monkeypatch.setattr(moon, 'find_moon_cache', lambda *a: pytest.fail('baked Moon cache requested'))
    monkeypatch.setattr(sat, 'load_textures', lambda: pytest.fail('Earth textures requested'))
    monkeypatch.setattr(moon, '_starfield_payload', lambda epoch: None)
    monkeypatch.setattr(sat, 'load_star_catalog', lambda **kw: [])
    lunar_path = tmp_path/'moon.html'
    moon.moon_webgl(cache=tmp_path, save_path=lunar_path, embed_assets=True, low_spec=True)
    earth_path = tmp_path/'earth.html'
    sat.build(out_path=earth_path, database_path=None, state_vectors=[], low_spec=True)
    lunar = lunar_path.read_text()
    earth = earth_path.read_text()
    assert 'const LOW_SPEC = true;' in lunar and 'const LOW_SPEC = true;' in earth
    assert 'moon_albedo.jpg": "data:' not in lunar
    assert 'const ASSET_URLS = {};' in lunar


@pytest.mark.parametrize('low', [False, True])
def test_plotly_maps_never_displace_physical_body_radii(low):
    """R1: every vertex is on the given sphere; 0.1 mm subtraction budget."""
    globe = importlib.import_module('ssatk.plots.globe_orbit_daynight_plotly')
    center = np.array([384400., -1400., 2300.])
    with low_spec_mode(low):
        earth = globe._earth_mesh(np.array([1., 0, 0]), n_lat=25, n_lon=48)
        moon = moon_mesh_plotly(center, 1738.1, n_lat=25, n_lon=48)
    for mesh, origin, radius in [(earth, np.zeros(3), 6378.137), (moon, center, 1738.1)]:
        vertices = np.column_stack((mesh.x, mesh.y, mesh.z))
        np.testing.assert_allclose(np.linalg.norm(vertices-origin, axis=1), radius, rtol=0, atol=1e-7)


@pytest.mark.parametrize('body', ['Earth', 'Moon'])
def test_notebook_mesh_rotates_geometry_in_three_dimensions(monkeypatch, body):
    """R1/C: a literal 90-degree rotation tilts the body pole; UV-only spin cannot."""
    from types import SimpleNamespace
    surfaces = importlib.import_module('ssatk.plots._body_surfaces')
    utils = importlib.import_module('ssatk.plots.plotutils')
    quarter_turn = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    monkeypatch.setattr(surfaces, 'body_rotation', lambda body, epoch: quarter_turn)
    monkeypatch.setattr(utils, '_require_ipyvolume', lambda: SimpleNamespace(plot_mesh=lambda x, y, z, **kw: (x, y, z, kw)))
    x, y, z, style = getattr(utils, 'draw'+body)([1.4e9, 1.4e9+10], ngrid=5, R=2., low_spec=True)
    assert x.shape == (2, 5, 5) and style['texture'] is None
    # North pole after +90 deg rotation about X is -Y, for both epochs.
    np.testing.assert_allclose(np.stack((x[:, 0, -1], y[:, 0, -1], z[:, 0, -1]), axis=-1),
                               [[0., -2., 0.], [0., -2., 0.]], rtol=0, atol=1e-15)


def test_solar_system_moon_orientation_uses_utc_julian_date(monkeypatch):
    """C: positions and map orientation share the documented JD UTC timescale."""
    solar = importlib.import_module('ssatk.plots.solar_bodies')
    received = []
    def capture(body, time, frame):
        received.append(time.utc.isot)
        return np.eye(3)
    monkeypatch.setattr(solar, 'body_rotation', capture)
    solar.make_moon_traces((1., 0, 0), 2460676.5, n=3, low_spec=True)
    assert received == ['2025-01-01T00:00:00.000']


def test_shared_scene_body_wrappers_accept_epoch_without_texture_io(monkeypatch):
    """C/R1: wrapper-to-renderer arguments retain dates and physical radii."""
    from ssatk.plots.scene_primitives import earth_trace, moon_trace
    monkeypatch.setattr(Image, 'open', lambda *a: pytest.fail('low-spec texture load'))
    epoch = Time('2025-01-01')
    center = np.array([384400., 0., 0.])
    earth = earth_trace(time=epoch, n_lat=13, n_lon=24, low_spec=True)
    moon = moon_trace(time=epoch, center_km=center, mode='solar', n_lat=13, n_lon=24, low_spec=True)
    for mesh, origin, radius in [(earth, np.zeros(3), 6378.137), (moon, center, 1738.1)]:
        vertices = np.column_stack((mesh.x, mesh.y, mesh.z))
        np.testing.assert_allclose(np.linalg.norm(vertices-origin, axis=1), radius, rtol=0, atol=1e-7)


def test_eclipse_moon_map_has_no_half_turn_shift(tmp_path, monkeypatch):
    """R1/C: asymmetric actual map file, literal equator vertices and texels.

    Permit one 8-bit colour count for float-to-integer conversion. Disable
    only the display contrast stretch so the source colours are the oracle.
    """
    surfaces = importlib.import_module('ssatk.plots._body_surfaces')
    renderer = importlib.import_module('ssatk.plots.moon_render')
    rgb = np.zeros((5, 8, 3), dtype=np.uint8)
    rgb[:, :4] = [240, 32, 16]
    rgb[:, 4:] = [16, 64, 224]
    path = tmp_path/'asymmetric.png'
    Image.fromarray(rgb).save(path)
    monkeypatch.setattr(surfaces, 'texture_path', lambda body: path)
    monkeypatch.setattr(renderer, '_display_lunar_albedo', lambda array: array/255.)
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1]])
    mesh = moon_mesh_plotly(np.zeros(3), 5., rotation=rotation, n_lat=5, n_lon=8, low_spec=False)
    points = np.column_stack((mesh.x, mesh.y, mesh.z)).reshape(5, 8, 3)
    np.testing.assert_allclose(points[2, [0, 4]], [[0, -5., 0], [0, 5., 0]], rtol=0, atol=1e-14)
    values = np.array([[int(value) for value in color[4:-1].split(',')] for color in mesh.vertexcolor]).reshape(5, 8, 3)
    np.testing.assert_allclose(values[1:-1], rgb[1:-1], rtol=0, atol=1)


@pytest.mark.parametrize('body,radius', [('earth', 12756.274), ('moon', 3476.2)])
def test_layer_radius_scale_is_applied_in_both_backends(body, radius):
    """R1: explicit 2x display scale times SSAPy's adopted radius, 0.1 mm budget.

    This catches the former Matplotlib path ignoring radius_scale while the
    Plotly path applied it. The scale affects radius, never the body centre.
    """
    from types import SimpleNamespace
    import plotly.graph_objects as go
    from ssatk.plots.layers import EarthLayer, MoonLayer
    epoch = Time('2025-01-01')
    state = SimpleNamespace(_epoch_gps=lambda: epoch.gps)
    layer = EarthLayer(radius_scale=2, low_spec=True) if body == 'earth' else MoonLayer(radius_scale=2, low_spec=True)
    center = np.zeros(3) if body == 'earth' else layer._moon_position_km(epoch.gps)
    fig = plt.figure()
    artist = layer.add_to_mpl(fig.add_subplot(projection='3d'), state)[0]
    np.testing.assert_allclose(np.linalg.norm(artist._vec[:3].T-center, axis=1), radius, rtol=0, atol=1e-7)
    plotly = go.Figure()
    layer.add_to_plotly(plotly, state)
    mesh = plotly.data[0]
    vertices = np.stack((mesh.x, mesh.y, mesh.z), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices-center, axis=1), radius, rtol=0, atol=1e-7)
    plt.close(fig)
