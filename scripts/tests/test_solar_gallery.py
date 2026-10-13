"""Solar rendering/export regressions; these are not flight-data validation."""

import importlib
import numpy as np
import pytest
from astropy.time import Time
from ssatk.plots.earth_sun_plot import DEFAULT_CFG as EARTH_CFG, build_static_figure, _resolve_start_jd
from ssatk.plots.solar_view_plot import DEFAULT_CFG as SOLAR_CFG, build_figure
from ssatk.plots.solar_bodies import make_planet_traces, make_sun_traces, make_moon_traces


@pytest.mark.parametrize('name,radius_km', [('Sun',696340.),('Earth',6378.137),('Moon',1738.1)])
def test_photosphere_mesh_vertices_have_ssapy_physical_radius(name,radius_km):
    """R2/C: SSAPy's adopted spherical radii in AU; all vertices within 0.02 mm.

    Sun=696340 km is SSAPy's adopted value, not IAU's 695700 km nominal unit.
    Earth uses the WGS84 equatorial radius; these meshes are spheres.
    """
    if name == 'Sun':
        trace = make_sun_traces()[0]
    elif name == 'Earth':
        trace = make_planet_traces(name,(0,0,0),n=16)[0]
    else:
        trace = make_moon_traces((0,0,0),2451545.,n=16)[0]
    points = np.stack([trace.x,trace.y,trace.z],axis=-1)
    # The sphere poles are symmetric around the center even for an even UV grid.
    center = (points[0,0] + points[-1,0]) / 2
    radii = np.linalg.norm(points-center,axis=-1) * 149597870.7
    np.testing.assert_allclose(radii,radius_km,rtol=0,atol=2e-8)


@pytest.mark.parametrize('builder,cfg',[(build_figure,SOLAR_CFG),(build_static_figure,EARTH_CFG)])
def test_solar_default_exports_are_physical_with_background_sky(builder,cfg):
    """C: no nearby Stars geometry; physical scales disclosed in both figures."""
    fig=builder(dict(cfg,sphere_resolution=8,hero_resolution=8))
    assert 'Physical radii and distances' in fig.layout.title.text
    assert all(t.name != 'Stars' for t in fig.data)
    if fig.layout.meta:  # Catalogue is optional outside the gallery environment.
        directions=np.array(fig.layout.meta['solar_sky']['directions'])
        assert (directions[:,2] < 0).any() and (directions[:,2] > 0).any()
        html=fig.to_html(include_plotlyjs=False)
        assert "gd.on('plotly_relayout'" in html
        assert 'ResizeObserver' in html


def test_earth_sun_scale_factors_and_epoch_are_explicit():
    """R2/C: UTC midnight equals Astropy; custom Sun radius is independent of Earth."""
    cfg=dict(EARTH_CFG,start_year=2025,start_month=6,start_day=1,
             planet_scale=7.,sun_scale=2.,moon_orbit_scale=9.,show_stars=False,hero_resolution=8)
    assert _resolve_start_jd(cfg) == Time('2025-06-01',scale='utc').jd
    fig=build_static_figure(cfg)
    assert 'planets/Moon ×7, Sun ×2, Moon distance ×9' in fig.layout.title.text
    sun=next(t for t in fig.data if t.name == 'Sun')
    radius=np.linalg.norm(np.stack([sun.x,sun.y,sun.z],axis=-1),axis=-1)
    np.testing.assert_allclose(radius*149597870.7,2*696340.,rtol=0,atol=1e-8)


def test_animation_frame_limit_preserves_circular_path_chords(monkeypatch,tmp_path):
    """R1/C: 7000 km circle sampled at 1 degree has r_min=R*cos(0.5 degree).

    Rendering only three frames must still draw all 360 short line segments,
    within 1 micrometre. Former frame decimation drew a diameter through Earth.
    """
    module=importlib.import_module('ssatk.plots.orbit_animation')
    captured=[]
    class CaptureAnimation:
        def __init__(self,fig,update,frames,**kwargs):
            self.fig,self.update,self.frames=fig,update,frames
        def save(self,*args,**kwargs):
            for i in range(self.frames):
                self.update(i)
            captured.extend(self.fig.axes[0].collections[-1].get_segments())
    monkeypatch.setattr(module,'FuncAnimation',CaptureAnimation)
    theta=np.linspace(0,2*np.pi,361)
    r=7e6*np.column_stack((np.cos(theta),np.sin(theta),np.zeros_like(theta)))
    module.orbit_animation(r,Time('2025-01-01').gps+np.arange(361.),views=('xy',),
                           max_frames=3,tail=3,save_path=tmp_path/'capture.gif')
    segments=np.asarray(captured)*1000  # scene uses km
    assert len(segments)==360
    mid=segments.mean(axis=1)
    np.testing.assert_allclose(np.linalg.norm(mid,axis=1),7e6*np.cos(np.pi/360),rtol=0,atol=1e-6)


@pytest.mark.parametrize("duration_days", [7.0, None], ids=["custom-span", "lunar-month"])
def test_fancy_movie_encoded_duration_is_independent_of_physical_span(tmp_path, monkeypatch, duration_days):
    """C/R3: decode 1-second MP4s with custom and mean-lunar-month spans.

    This checks the encoder output, not just len(time)/fps arithmetic. The
    full 60-second default is also generated and inspected in the handoff.
    Default span: NASA Moon Fact Sheet's 27.3217-day revolution period,
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html . Allow 1 us
    for float64 GPS endpoint rounding; this is not a trajectory accuracy test.
    """
    monkeypatch.setenv("SSATK_OUTPUT_DIR", str(tmp_path))
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from demos.video_animation.demo_fancy_video import orbit_moon_video_demo
    kwargs = {} if duration_days is None else {"duration_days": duration_days}
    result = orbit_moon_video_demo(
        **kwargs, playback_seconds=1, fps=12, fast=True,
        out_name=str(tmp_path / "fancy.mp4"), make_figures=True)
    frames, seconds = ffmpeg.count_frames_and_secs(result["out_mp4"])
    assert frames == 12
    assert seconds == pytest.approx(1.0, abs=0.001)
    epochs = result["t_frames_ok"]
    expected_days = 27.3217 if duration_days is None else duration_days
    assert epochs[-1] - epochs[0] == pytest.approx(expected_days * 86400.0, abs=1e-6)
    assert np.all(np.diff(epochs) > 0)
    assert np.max(np.diff(epochs)) <= 3600.0 + 1e-6


def test_solar_export_retains_double_precision_body_coordinates():
    """C: serialization compaction must not discard km-scale AU coordinates."""
    from ssatk.plots.earth_sun_plot import _shrink_floats
    cfg=dict(EARTH_CFG,start_year=2025,start_month=6,start_day=1,show_stars=False,hero_resolution=8)
    fig=build_static_figure(cfg)
    earth=next(t for t in fig.data if t.name == 'Earth')
    xyz=[np.asarray(getattr(earth,k)).copy() for k in 'xyz']
    _shrink_floats(fig)
    for axis, reference in zip('xyz',xyz):
        np.testing.assert_array_equal(getattr(earth,axis),reference)
