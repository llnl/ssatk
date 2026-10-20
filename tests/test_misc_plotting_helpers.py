from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.constants import RGEO
from ssapy_toolkit.plots import misc_plotting


def test_koe_plot_with_monkeypatched_elements(monkeypatch, tmp_path):
    import ssapy_toolkit.orbital_mechanics as orbital_mechanics

    def fake_elements(r, v, mu_barycenter):
        n = len(r)
        return {
            "a": np.linspace(RGEO, RGEO * 1.1, n),
            "e": np.linspace(0.01, 0.02, n),
            "i": np.linspace(0.1, 0.2, n),
        }

    monkeypatch.setattr(orbital_mechanics, "calculate_orbital_elements", fake_elements, raising=False)
    r = np.zeros((3, 3))
    v = np.ones((3, 3))
    fig, ax = misc_plotting.koe_plot(r, v, elements=["e", "i", "a"], body="Earth")
    assert ax.get_xlabel() == "Index"
    assert len(fig.axes) == 2
    plt.close(fig)

    save_path = tmp_path / "koe.png"
    fig, ax = misc_plotting.koe_plot(r, v, t=Time([0.0, 1.0, 2.0], format="gps"), elements=["a"], body="Moon", save_path=save_path)
    assert save_path.exists()
    plt.close(fig)
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.koe_plot(r, v, bad=True)


def test_koe_2dhist_validates_and_plots(tmp_path):
    data = SimpleNamespace(
        a=np.linspace(RGEO, 2 * RGEO, 10),
        e=np.linspace(0.1, 0.8, 10),
        i=np.linspace(0.0, np.pi / 2, 10),
        ta=np.linspace(0.0, 2 * np.pi, 10),
    )
    fig = misc_plotting.koe_2dhist(data, bins=4, logscale=False)
    assert len(fig.axes) >= 9
    plt.close(fig)
    fig = misc_plotting.koe_2dhist(data, bins=4, logscale="log", save_path=tmp_path / "hist.png")
    assert (tmp_path / "hist.png").exists()
    plt.close(fig)

    bad_i = SimpleNamespace(a=data.a, e=data.e, i=np.array([-1.0]), ta=np.array([0.0]))
    with pytest.raises(ValueError, match="Inclination"):
        misc_plotting.koe_2dhist(bad_i)
    bad_ta = SimpleNamespace(a=data.a, e=data.e, i=np.array([0.0]), ta=np.array([3 * np.pi]))
    with pytest.raises(ValueError, match="True Anomaly"):
        misc_plotting.koe_2dhist(bad_ta)


def test_scatter_helpers_and_orbit_divergence(monkeypatch, tmp_path, capsys):
    misc_plotting.scatter2d([1, 2], [3, 4], [0.1, 0.2], colorscale="linear", save_path=tmp_path / "scatter2d.png")
    assert (tmp_path / "scatter2d.png").exists()
    misc_plotting.scatter2d([1, 2], [3, 4], [1.0, 2.0], colorscale="log")
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.scatter2d([1], [2], [3], bad=True)

    fig, ax = misc_plotting.scatter3d(np.array([[1, 2, 3], [4, 5, 6]]), cs=[0.0, 1.0], save_path=tmp_path / "scatter3d.png")
    assert (tmp_path / "scatter3d.png").exists()
    ax = ax[0] if isinstance(ax, tuple) else ax
    assert hasattr(ax, "get_zlim")
    plt.close(fig)
    fig, ax = misc_plotting.scatter3d([1, 2], [3, 4], [5, 6], cs=None)
    plt.close(fig)
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.scatter3d([1], [2], [3], bad=True)

    assert len(misc_plotting.dotcolors_scaled(3)) == 3

    rs = np.zeros((3, 3, 2))
    rs[:, 0, 0] = [RGEO, 1.1 * RGEO, 1.2 * RGEO]
    rs[:, 1, 0] = [0, 0.1 * RGEO, 0.2 * RGEO]
    rs[:, 2, 1] = [RGEO, 1.1 * RGEO, 1.2 * RGEO]
    r_moon = np.tile([2 * RGEO, 0.0, 0.0], (3, 1))
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon, limits=None, show=False)
    assert "limits" in capsys.readouterr().out
    plt.close(fig)
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon.T, limits=3, show=False)
    plt.close(fig)
    assert misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon, limits=3, show=False, save_path=tmp_path / "div.png") is None
    assert (tmp_path / "div.png").exists()

    with pytest.raises(IndexError, match="2 dimensions"):
        misc_plotting.orbit_divergence_plot(rs, r_moon=np.zeros(3), show=False)
    with pytest.raises(IndexError, match="expected"):
        misc_plotting.orbit_divergence_plot(rs, r_moon=np.zeros((2, 2)), show=False)

    class FakeMoon:
        def position(self, t):
            return r_moon

    monkeypatch.setattr(misc_plotting, "get_body", lambda name: FakeMoon())
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=None, t=np.arange(3), limits=3, show=False)
    plt.close(fig)


def test_sensor_fov_sun_moon_geometry_uses_epoch_and_far_sun():
    import numpy as np
    from astropy.time import Time

    from ssapy_toolkit.plots import sensor_fov_plot as sfp

    cfg = sfp.DEFAULT_CFG.copy()
    cfg.update(
        epoch="2025-04-25 00:00:00",
        axis_range_km=50_000.0,
        show_sensor=False,
        show_stars=False,
        earth_n_lat=8,
        earth_n_lon=16,
        moon_n_lat=8,
        moon_n_lon=16,
    )
    expected_sun = sfp.sun_direction_eci(
        np.array([0.0]),
        epoch_jd=float(Time(cfg["epoch"], format="iso", scale="utc").jd),
    )[0]
    assert np.allclose(sfp._sun_direction_from_cfg(cfg), expected_sun)

    r_km = np.array([[42_164.0, 0.0, 0.0], [0.0, 42_164.0, 0.0]])
    v_kms = np.array([[0.0, 3.07, 0.0], [-3.07, 0.0, 0.0]])
    fig = sfp.build_figure(cfg, r_km, v_kms)
    traces = {getattr(trace, "name", ""): trace for trace in fig.data}

    def surface_center(trace):
        coords = []
        for axis in (trace.x, trace.y, trace.z):
            values = np.asarray(axis, dtype=float)
            coords.append((np.nanmin(values) + np.nanmax(values)) / 2.0)
        return np.asarray(coords)

    moon_center = surface_center(traces["Moon"])
    sun_center = surface_center(traces["Sun"])
    assert np.linalg.norm(sun_center) > 7.5 * np.linalg.norm(moon_center)




def test_scene_earth_rotation_helper_matches_globe_plot():
    from astropy.time import Time

    from ssapy_toolkit.plots.globe_plot import _earth_lon0_from_time
    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    sample_time = Time(12_345.0, format="gps")
    np.testing.assert_allclose(
        earth_rotation_deg_from_time(sample_time),
        _earth_lon0_from_time(sample_time),
    )
    later = Time(sample_time.gps + 3600.0, format="gps")
    assert earth_rotation_deg_from_time(later) != earth_rotation_deg_from_time(sample_time)


def test_plotly_globe_orients_earth_at_final_sample(monkeypatch):
    import plotly.graph_objects as go
    from astropy.time import Time

    from ssapy_toolkit.plots import globe_orbit_daynight_plotly as globe
    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    captured = {}

    def fake_earth_mesh(sun_hat, **kwargs):
        captured["rotation_deg"] = kwargs.get("rotation_deg")
        return go.Mesh3d(x=[0.0], y=[0.0], z=[0.0], i=[], j=[], k=[], name="Earth")

    monkeypatch.setattr(globe, "_earth_mesh", fake_earth_mesh)
    r_km = np.array([[7000.0, 0.0, 0.0], [0.0, 7100.0, 0.0], [0.0, 0.0, 7200.0]])
    times = Time([10.0, 20.0, 30.0], format="gps")

    fig = globe.plot_globe_orbit_daynight_plotly(r=r_km, t=times, n_steps=3)

    np.testing.assert_allclose(
        captured["rotation_deg"],
        earth_rotation_deg_from_time(times[-1]),
    )
    marker = next(trace for trace in fig.data if getattr(trace, "name", "") == "Earth orientation point")
    assert marker.mode == "markers"
    assert not getattr(marker, "showlegend", True)
    assert "Earth oriented at sample" not in str(fig.layout.title.text)


def test_sensor_fov_animation_rotates_earth_frames(monkeypatch):
    import plotly.graph_objects as go
    from astropy.time import Time

    from ssapy_toolkit.plots import sensor_fov_plot as sfp

    frame_rotations = []

    def fake_earth_trace(**kwargs):
        frame_rotations.append(float(kwargs.get("rotation_deg", 0.0)))
        return go.Mesh3d(
            x=[0.0], y=[0.0], z=[0.0], i=[], j=[], k=[],
            name="Earth", meta={"rotation_deg": float(kwargs.get("rotation_deg", 0.0))},
        )

    monkeypatch.setattr(sfp, "earth_trace", fake_earth_trace)
    cfg = sfp.DEFAULT_CFG.copy()
    cfg.update(
        show_stars=False,
        show_moon=False,
        show_sun=False,
        fov_animate=True,
        fov_anim_step=1,
        earth_n_lat=6,
        earth_n_lon=12,
        axis_range_km=12_000.0,
    )
    r_km = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0], [-7000.0, 0.0, 0.0]])
    v_kms = np.array([[0.0, 7.5, 0.0], [-7.5, 0.0, 0.0], [0.0, -7.5, 0.0]])
    times = Time([0.0, 3600.0, 7200.0], format="gps")

    fig = sfp.build_figure(cfg, r_km, v_kms, t=times)

    assert fig.frames
    assert all(frame.traces[0] == 0 for frame in fig.frames)
    assert len(set(round(rot, 6) for rot in frame_rotations)) == len(frame_rotations)


def test_solar_view_earth_texture_accepts_rotation():
    from ssapy_toolkit.plots.solar_bodies import make_planet_traces

    earth0 = make_planet_traces("Earth", (1.0, 0.0, 0.0), n=18, show_label=False, rotation_deg=0.0)[0]
    earth90 = make_planet_traces("Earth", (1.0, 0.0, 0.0), n=18, show_label=False, rotation_deg=90.0)[0]

    assert not np.allclose(np.asarray(earth0.surfacecolor), np.asarray(earth90.surfacecolor))

def test_offset_moon_uses_local_normals_and_stays_readable():
    import re

    from ssapy_toolkit.plots.moon_render import moon_mesh_plotly

    moon = moon_mesh_plotly(
        [-384_400.0, 0.0, 0.0],
        1_737.4,
        sun_hat=[1.0, 0.0, 0.0],
        mode="solar",
        n_lat=16,
        n_lon=32,
    )
    values = []
    for color in moon.vertexcolor:
        match = re.match(r"rgb\((\d+),(\d+),(\d+)\)", color)
        if match:
            values.append(tuple(map(int, match.groups())))

    assert values
    assert max(sum(rgb) / 3.0 for rgb in values) > 150.0
    assert sum(sum(rgb) / 3.0 > 80.0 for rgb in values) / len(values) > 0.20




def test_earth_city_lights_are_not_latitude_bands():
    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import _city_lights

    n_lat, n_lon = 60, 120
    lat = np.linspace(90.0, -90.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    land = np.ones_like(lat_grid)

    lights = _city_lights(n_lat, n_lon, land, lat_grid, lon_grid)
    residual = lights - lights.mean(axis=1, keepdims=True)

    assert lights.shape == (n_lat, n_lon)
    assert np.isfinite(lights).all()
    assert residual.std() > 0.035
    assert np.mean(np.std(lights, axis=1) > 0.004) > 0.50


def test_earth_mesh_keeps_native_earth_texture_longitudes(monkeypatch):
    import re

    from ssapy_toolkit.plots import globe_orbit_daynight_plotly as daynight

    n_lat = 5
    n_lon = 8
    tex = np.zeros((n_lat, n_lon, 3), dtype=np.uint8)
    tex[..., 2] = 240
    # The mesh uses endpoint=False longitudes [-180, -135, ..., 0, 45, ...];
    # column 4 is 0° longitude in SSAPy's native earth.png convention.
    tex[:, 4, :] = [240, 0, 0]
    monkeypatch.setattr(daynight, "_load_real_earth_texture", lambda lat, lon: tex.copy())

    trace = daynight._earth_mesh([1.0, 0.0, 0.0], n_lat=n_lat, n_lon=n_lon, rotation_deg=0.0)
    color = trace.vertexcolor[2 * n_lon + 4]
    red, green, blue = map(int, re.match(r"rgb\((\d+),(\d+),(\d+)\)", color).groups())
    assert red > blue
    assert green < red


def test_earth_layer_plotly_keeps_native_longitudes_and_time(tmp_path):
    from PIL import Image
    from astropy.time import Time
    import plotly.graph_objects as go

    from ssapy_toolkit.plots.layers import EarthLayer
    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    n_lat = 5
    n_lon = 8
    tex = np.zeros((n_lat, n_lon, 3), dtype=np.uint8)
    tex[..., 2] = 240
    tex[:, 4, :] = [240, 0, 0]
    texture_path = tmp_path / "earth.png"
    Image.fromarray(tex, mode="RGB").save(texture_path)

    fig = go.Figure()
    EarthLayer(texture_path=texture_path, n_lat=n_lat, n_lon=n_lon, rotation_deg=0.0).add_to_plotly(fig, None)
    color = fig.data[0].facecolor[(2 * (n_lon - 1) + 4) * 2]
    assert color == "rgb(240,0,0)"

    sample_time = Time("2024-04-08T18:40:00", scale="utc")
    timed_layer = EarthLayer(texture_path=texture_path, time=sample_time)
    np.testing.assert_allclose(timed_layer.rotation_deg, earth_rotation_deg_from_time(sample_time))

    _, _, z = timed_layer._sphere_xyz()
    assert np.nanmean(z[0]) > 0.0
    assert np.nanmean(z[-1]) < 0.0


def test_solar_bodies_earth_texture_and_time_conventions(monkeypatch):
    from astropy.time import Time

    from ssapy_toolkit.plots import solar_bodies
    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    tex = np.zeros((3, 8), dtype=float)
    tex[:, 0] = 0.25       # -180° / +180° seam
    tex[:, 4] = 1.0        # 0° longitude, Greenwich meridian
    np.testing.assert_allclose(solar_bodies._sample_earth_texture(np.array([0.0]), np.array([0.0]), tex), [1.0])
    np.testing.assert_allclose(solar_bodies._sample_earth_texture(np.array([0.0]), np.array([np.pi]), tex), [0.25])

    captured = {}

    def fake_color_earth(n, rotation_deg=0.0):
        captured["rotation_deg"] = rotation_deg
        return np.zeros((n, n)), [[0, "#000"], [1, "#fff"]]

    monkeypatch.setattr(solar_bodies, "_color_earth", fake_color_earth)
    sample_time = Time("2024-04-08T18:40:00", scale="utc")
    solar_bodies.make_planet_traces("Earth", (1.0, 0.0, 0.0), n=4, time=sample_time, show_label=False)
    np.testing.assert_allclose(captured["rotation_deg"], earth_rotation_deg_from_time(sample_time))


def test_sun_view_earth_texture_and_time_conventions(tmp_path):
    from PIL import Image
    from astropy.time import Time

    from ssapy_toolkit.plots.sun_view import EarthShadingLayer, _sample_texture_rgb
    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    tex = np.zeros((4, 8, 3), dtype=np.uint8)
    tex[..., 2] = 240
    tex[:, 4, :] = [240, 0, 0]
    texture_path = tmp_path / "earth.png"
    Image.fromarray(tex, mode="RGB").save(texture_path)

    rgb = _sample_texture_rgb(texture_path, np.array([np.pi / 2.0]), np.array([0.0]))
    assert rgb[0, 0] > rgb[0, 2]

    sample_time = Time("2024-04-08T18:40:00", scale="utc")
    body = EarthShadingLayer([1.0, 0.0, 0.0], time=sample_time)
    np.testing.assert_allclose(body.rotation_deg, earth_rotation_deg_from_time(sample_time))


def test_magnetosphere_shared_earth_mesh_keeps_native_longitudes(tmp_path):
    from PIL import Image

    from ssapy_toolkit.plots.magnetosphere_core import _build_earth_mesh

    tex = np.zeros((5, 8, 3), dtype=np.uint8)
    tex[..., 2] = 240
    tex[:, 4, :] = [240, 0, 0]
    texture_path = tmp_path / "earth.png"
    Image.fromarray(tex, mode="RGB").save(texture_path)

    trace = _build_earth_mesh(texture_path, n_lon=9, n_lat=5, allow_download=False)
    colors = np.asarray(trace.vertexcolor, dtype=int).reshape(5, 9, 3)
    red, green, blue = colors[2, 4]
    assert red > blue
    assert green < red


def test_earth_mesh_dark_side_has_no_latitude_band_artifact():
    import re

    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import _earth_mesh

    n_lat, n_lon = 24, 48
    earth = _earth_mesh([1.0, 0.0, 0.0], n_lat=n_lat, n_lon=n_lon)
    colors = []
    for color in earth.vertexcolor:
        match = re.match(r"rgb\((\d+),(\d+),(\d+)\)", color)
        assert match is not None
        colors.append(tuple(map(int, match.groups())))
    ring_rows = (n_lat + 1 if n_lat % 2 == 0 else n_lat) - 2
    colors = np.asarray(colors[:-2], dtype=float).reshape(ring_rows, n_lon, 3)
    brightness = colors.mean(axis=2)

    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    night_cols = np.cos(np.radians(lon)) < -0.25
    night = brightness[:, night_cols]
    residual = night - night.mean(axis=1, keepdims=True)

    assert residual.std() > 1.0
    assert np.nanmax(np.abs(np.diff(night.mean(axis=1)))) < 90.0


def test_earth_and_moon_pole_vertices_are_stable():
    import re

    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import _earth_mesh
    from ssapy_toolkit.plots.moon_render import moon_mesh_plotly

    n_lat, n_lon = 18, 36
    earth = _earth_mesh([1.0, 0.0, 0.0], n_lat=n_lat, n_lon=n_lon)
    moon = moon_mesh_plotly([0.0, 0.0, 0.0], 1_000.0, sun_hat=[1.0, 0.0, 0.0], n_lat=n_lat, n_lon=n_lon)

    effective_n_lat = n_lat + 1
    expected_vertices = (effective_n_lat - 2) * n_lon + 2
    assert len(earth.x) == expected_vertices
    np.testing.assert_allclose(
        np.column_stack([earth.x[-2:], earth.y[-2:], earth.z[-2:]]),
        [[0.0, 0.0, 6_356.752314245], [0.0, 0.0, -6_356.752314245]], atol=1e-9,
    )
    assert np.count_nonzero(np.asarray(earth.i) == expected_vertices - 2) == n_lon
    assert np.count_nonzero(np.asarray(earth.i) == expected_vertices - 1) == n_lon

    # The Moon mesh has one vertex per pole, avoiding duplicate degenerate
    # vertices while retaining a closed n_lon-triangle cap at each end.
    assert len(moon.x) == expected_vertices
    np.testing.assert_allclose(
        np.column_stack([moon.x[-2:], moon.y[-2:], moon.z[-2:]]),
        [[0.0, 0.0, 1_000.0], [0.0, 0.0, -1_000.0]], atol=1e-9,
    )
    assert np.count_nonzero(np.asarray(moon.i) == expected_vertices - 2) == n_lon
    assert np.count_nonzero(np.asarray(moon.i) == expected_vertices - 1) == n_lon


def test_earth_and_moon_meshes_close_longitude_seams():
    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import _earth_mesh
    from ssapy_toolkit.plots.moon_render import moon_mesh_plotly

    n_lat = 6
    n_lon = 12
    expected_triangles = 2 * (n_lat - 1) * n_lon

    earth = _earth_mesh([1.0, 0.0, 0.0], n_lat=n_lat, n_lon=n_lon)
    moon = moon_mesh_plotly(
        [0.0, 0.0, 0.0],
        1_000.0,
        sun_hat=[1.0, 0.0, 0.0],
        mode="solar",
        n_lat=n_lat,
        n_lon=n_lon,
    )

    assert len(earth.i) == expected_triangles
    assert len(moon.i) == expected_triangles


def test_sun_position_and_radius_uses_angular_sizing():
    import numpy as np

    from ssapy_toolkit.constants import SUN_EARTH_AVERAGE_DISTANCE_KM, SUN_RADIUS_KM
    from ssapy_toolkit.plots.scene_primitives import (
        light_direction_from_sun,
        sun_light_position_km,
        sun_position_and_radius,
    )

    pos, radius = sun_position_and_radius(scene_radius_km=10_000.0, sun_hat=[1, 0, 0])
    assert np.allclose(pos, [25_000.0, 0.0, 0.0])
    assert np.isclose(radius / np.linalg.norm(pos), SUN_RADIUS_KM / SUN_EARTH_AVERAGE_DISTANCE_KM)

    pos, radius = sun_position_and_radius(
        scene_radius_km=10_000.0,
        sun_hat=[0, 1, 0],
        distance_mode="real",
        radius_mode="real",
    )
    assert np.allclose(pos, [0.0, SUN_EARTH_AVERAGE_DISTANCE_KM, 0.0])
    assert radius == SUN_RADIUS_KM

    pos, radius = sun_position_and_radius(
        scene_radius_km=10_000.0,
        sun_hat=[1, 0, 0],
        distance_factor=4.0,
        radius_mode="match_moon",
        match_radius_km=1_738.1 * 2.5,
        match_distance_km=384_399.0,
    )
    assert np.isclose(radius / np.linalg.norm(pos), (1_738.1 * 2.5) / 384_399.0)

    physical_sun = sun_light_position_km([1, 0, 0])
    moon_light = light_direction_from_sun(target_km=[384_399.0, 1_000.0, 0.0], sun_position_km=physical_sun)
    assert moon_light[0] > 0.999
    assert moon_light[1] < 0.0


def test_2024_solar_eclipse_track_reaches_central_texas():
    """NASA/GSFC's 2024-04-08 totality path crosses central Texas near 18:39 UTC."""
    from ssapy_toolkit.plots.eclipse_space_view_plotly import (
        RE_KM,
        _eci_surface_to_latlon,
        _great_circle_distance_km,
        _real_solar_eclipse_geometry,
        _shadow_ground_point,
        _time_grid_utc,
    )

    travis_county_lat = 30.3630
    travis_county_lon = -97.9790
    times = _time_grid_utc("2024-04-08T18:00:00", "2024-04-08T19:20:00", 81)
    moon_pos, sun_hat = _real_solar_eclipse_geometry(times)

    hits = []
    for time, moon_i, sun_i in zip(times, moon_pos, sun_hat):
        hit = _shadow_ground_point(moon_i, sun_i, np.zeros(3), earth_r_real=RE_KM)
        assert hit is not None
        lat, lon = _eci_surface_to_latlon(hit, time)
        distance_km = _great_circle_distance_km(lat, lon, travis_county_lat, travis_county_lon)
        hits.append((distance_km, time, lat, lon))

    latlon_path = [(lat, lon) for _, _, lat, lon in hits]
    assert any(21.0 < lat < 26.0 and -109.0 < lon < -104.0 for lat, lon in latlon_path)
    assert any(35.0 < lat < 44.0 and -93.0 < lon < -78.0 for lat, lon in latlon_path)

    distance_km, closest_time, closest_lat, closest_lon = min(hits, key=lambda row: row[0])
    assert abs((closest_time - Time("2024-04-08T18:39:00", scale="utc")).sec) <= 90.0
    assert distance_km < 150.0
    assert 30.0 < closest_lat < 32.0
    assert -100.0 < closest_lon < -97.0


def test_eclipse_surface_latlon_round_trips_with_earth_rotation():
    from ssapy_toolkit.plots.eclipse_space_view_plotly import (
        _eci_surface_to_latlon,
        _latlon_to_eci_surface,
    )

    sample_time = Time("2024-04-08T18:39:00", scale="utc")
    for lat, lon in [(-45.0, -170.0), (0.0, 0.0), (30.363, -97.979), (71.0, 145.0)]:
        point = _latlon_to_eci_surface(lat, lon, sample_time)
        got_lat, got_lon = _eci_surface_to_latlon(point, sample_time)
        np.testing.assert_allclose(got_lat, lat, atol=1e-9)
        wrapped_diff = ((got_lon - lon + 180.0) % 360.0) - 180.0
        np.testing.assert_allclose(wrapped_diff, 0.0, atol=1e-9)


def test_2014_lunar_eclipse_event_is_visible_from_wisconsin():
    """NASA lists the 2014-04-15 total lunar eclipse as visible from North America."""
    from ssapy_toolkit.plots.eclipse_space_view_plotly import _real_lunar_event_window

    window = _real_lunar_event_window("2014-04-15", n_steps=241)

    assert window["event_key"] == "2014-04-15-total-lunar-wisconsin"
    assert window["peak_utc"] == "2014-04-15T07:46:48"
    assert window["illum"][window["peak_idx"]] < 0.02
    assert window["observer_name"] == "Madison, Wisconsin"
    assert window["observer_moon_alt_deg"] > 25.0
    assert window["observer_sun_alt_deg"] < -20.0


def test_generic_lunar_catalog_event_does_not_require_observer_metadata():
    from ssapy_toolkit.plots.eclipse_space_view_plotly import find_and_plot_eclipse

    fig, stats = find_and_plot_eclipse(mode="lunar", event="2001-01-09", verbose=False)
    plt.close(fig)

    assert stats["event_key"] == "lunar-2001-01-09"
    assert stats["peak_utc"] == "2001-01-09T20:21:40"
    assert "observer_name" not in stats
    assert stats["min_illum"] < 0.1


def test_lunar_animation_keeps_starfield_static_to_avoid_final_blank_frame():
    from ssapy_toolkit.plots.eclipse_space_view_plotly import plot_space_view_animated

    fig = plot_space_view_animated(
        mode="lunar",
        event="2001-01-09",
        n_frames=3,
        n_lat=8,
        n_lon=12,
        verbose=False,
    )

    assert fig.data[0].name == "Stars"
    dynamic_trace_count = len(fig.data) - 1
    assert dynamic_trace_count > 0
    assert len(fig.frames) == 3
    for frame in fig.frames:
        assert len(frame.data) == dynamic_trace_count
        assert tuple(frame.traces) == tuple(range(1, len(fig.data)))


def test_eclipse_catalog_dropdown_contains_21st_century_defaults(tmp_path):
    from ssapy_toolkit.plots.eclipse_space_view_plotly import (
        _inject_eclipse_catalog_dropdown,
        eclipse_catalog_21st_century,
    )

    lunar = eclipse_catalog_21st_century("lunar")
    solar = eclipse_catalog_21st_century("solar")

    assert len(lunar) == 226
    assert len(solar) == 222
    assert any(entry["key"] == "lunar-2014-04-15" for entry in lunar)
    assert any(entry["key"] == "solar-2024-04-08" for entry in solar)

    html_path = tmp_path / "eclipse.html"
    html_path.write_text("<html><body><div>plotly figure</div></body></html>", encoding="utf-8")
    _inject_eclipse_catalog_dropdown(html_path, default_mode="lunar", default_event="2014-04-15")
    _inject_eclipse_catalog_dropdown(html_path, default_mode="lunar", default_event="2014-04-15")
    text = html_path.read_text(encoding="utf-8")

    assert text.count("SSATK_ECLIPSE_CATALOG_SELECTOR_START") == 1
    assert "21st-Century Eclipse Selector" in text
    assert "lunar-2014-04-15" in text
    assert "solar-2024-04-08" in text
    assert "Default lunar scene selected by Travis" in text
