"""R1 camera geometry and C renderer integration; no flight dynamics claims."""

import importlib

import numpy as np
import pytest
from astropy.time import Time
from matplotlib import pyplot as plt
from plotly import graph_objects as go

from ssatk.plots.camera import AnimationCamera, _CameraPath, _plotly_camera
from ssatk.plots.spacecraft_view import SpacecraftView

BOUNDS = np.array([[-10.0, -10, -10], [10.0, 10, 10]])


def test_fixed_overview_and_follow_against_literal_translation():
    """R1: fixed scene eye vs eye = target + (3,4,0); 1e-13 scene-unit budget."""
    p = np.array([[1.0, 2, 3], [4, 6, 3], [7, 10, 3]])
    original = p.copy()
    overview = _CameraPath("overview", p, BOUNDS)
    np.testing.assert_array_equal(overview.eyes, np.tile(overview.eyes[0], (3, 1)))
    np.testing.assert_array_equal(overview.targets, np.zeros((3, 3)))
    follow = _CameraPath(AnimationCamera("follow", offset=(3, 4, 0)), p, BOUNDS)
    np.testing.assert_allclose(
        follow.eyes, [[4, 6, 3], [7, 10, 3], [10, 14, 3]], rtol=0, atol=1e-13
    )
    np.testing.assert_array_equal(p, original)


def test_orbit_quarter_turn_uses_elapsed_time_not_frame_number():
    """R1: Rodrigues +Z rotation at 0,90,360 degrees on nonuniform epochs."""
    path = _CameraPath(
        AnimationCamera("orbit", offset=(4, 0, 3)),
        np.zeros((3, 3)),
        BOUNDS,
        times=[100, 101, 104],
    )
    np.testing.assert_allclose(
        path.eyes, [[4, 0, 3], [0, 4, 3], [4, 0, 3]], rtol=0, atol=2e-14
    )


def test_chase_constant_velocity_and_pause_reference():
    """R1: +X motion yields (-10,0,3) direction; 1e-13 scene-unit budget."""
    t = np.array([0.0, 1, 4])
    p = np.column_stack((2 * t, np.zeros((3, 2))))
    path = _CameraPath(
        AnimationCamera("chase", distance=np.sqrt(109)), p, BOUNDS, times=t
    )
    np.testing.assert_allclose(
        path.eyes - p, np.tile([-10, 0, 3], (3, 1)), rtol=0, atol=1e-13
    )
    pause = _CameraPath(
        "chase", np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]), BOUNDS
    )
    assert np.isfinite(pause.eyes).all()
    with pytest.raises(ValueError, match="moving target"):
        _CameraPath("chase", np.zeros((3, 3)), BOUNDS)


def test_matplotlib_follow_projection_centres_the_target():
    """R1: a look-at target projects to the optical centre, 1e-12 NDC budget."""
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    p = np.array([[1.0, 2, 3], [7, 8, 9]])
    path = _CameraPath(AnimationCamera("follow", offset=(3, 4, 2)), p, BOUNDS)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(2):
        path.mpl(ax, i)
        x, y, _ = proj_transform(*p[i], ax.get_proj())
        np.testing.assert_allclose([x, y], [0, 0], rtol=0, atol=1e-12)
    plt.close(fig)


def test_plotly_camera_frames_preserve_data_and_literal_scaled_target():
    """C/R1: cube-normalized targets and fixed offset; physical traces unchanged."""
    p = np.array([[0.0, 0, 0], [4, 2, 0]])
    fig = go.Figure(
        data=[go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2])],
        frames=[go.Frame(name=str(i)) for i in range(2)],
    )
    fig.update_layout(
        scene=dict(xaxis_range=[-10, 10], yaxis_range=[-10, 10], zaxis_range=[-10, 10])
    )
    before = fig.data[0].to_json()
    _plotly_camera(fig, AnimationCamera("follow", offset=(0, 0, 5)), p)
    assert fig.data[0].to_json() == before
    c = fig.frames[1].layout.scene.camera
    np.testing.assert_allclose(
        [c.center.x, c.center.y, c.center.z], [0.2, 0.1, 0], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        [c.eye.x, c.eye.y, c.eye.z], [0.2, 0.1, 0.25], rtol=0, atol=1e-14
    )


def test_spacecraft_overview_and_orbit_resampling_keep_original_scene():
    """C/R1: resampling a subset must not reframe the scene or restart the sweep."""
    epoch = Time("2025-01-01").gps
    p = np.array([[7e6, 0, 0], [0, 7e6, 0], [-7e6, 0, 0]])
    for mode in ("overview", "orbit", "follow", "chase", "limb"):
        view = SpacecraftView(
            p,
            epoch + np.array([0, 1, 4]),
            [1, 0, 0, 0],
            camera=mode,
            stars=False,
            textures=False,
        )
        sampled = view.resample([epoch + 1])
        np.testing.assert_allclose(
            sampled._camera(0)[0], view._camera(1)[0], rtol=0, atol=1e-8
        )
        assert np.isfinite(view.render(1, width=160, height=160)).all()


def test_sensor_and_solar_camera_frames_use_actual_target_positions():
    """C: public Plotly adapters generate camera layouts for every supplied frame."""
    earth = importlib.import_module("ssatk.plots.earth_sun_plot")
    fig = earth.build_figure(
        dict(
            earth.DEFAULT_CFG,
            n_frames=3,
            sphere_resolution=6,
            show_stars=False,
            camera="follow",
        ),
        low_spec=True,
    )
    centres = np.array(
        [[f.layout.scene.camera.center[k] for k in "xyz"] for f in fig.frames]
    )
    assert len(centres) == 3 and np.ptp(centres[:, 0]) > 0.1
    sensor = importlib.import_module("ssatk.plots.sensor_fov_plot")
    t = Time("2025-01-01").gps + np.arange(3) * 100
    r = np.array([[7e6, 0, 0], [6.9e6, 1e6, 0], [6.8e6, 2e6, 0]])
    fig = sensor.plot_sensor_fov(
        r=r,
        v=np.tile([0.0, 7000, 0], (3, 1)),
        t=t,
        camera="follow",
        fov_animate=True,
        fov_anim_step=1,
        show_stars=False,
        low_spec=True,
    )
    assert len(fig.frames) == 3
    assert all(f.layout.scene.camera.center.x is not None for f in fig.frames)


def test_overview_bounding_cube_corners_fit_through_full_sweep():
    """R1: eight literal corners stay inside the viewport with >=2% margin.

    This constrains full-scene framing while avoiding excessive empty space.
    Projection uses Matplotlib's independent matrix and display transforms.
    """
    from itertools import product
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    points = np.array(list(product([-1.0, 1.0], repeat=3)))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for elevation in [-75.0, 0.0, 25.0, 75.0]:
        path = _CameraPath(
            AnimationCamera("orbit", elevation=elevation),
            np.zeros((17, 3)),
            [[-1] * 3, [1] * 3],
        )
        for i in range(17):
            path.mpl(ax, i)
            fig.canvas.draw()
            xy = np.array(proj_transform(*points.T, ax.get_proj())[:2]).T
            xy = ax.transData.transform(xy)
            fractions = (xy - [ax.bbox.x0, ax.bbox.y0]) / [
                ax.bbox.width,
                ax.bbox.height,
            ]
            assert np.all(fractions > 0.02) and np.all(fractions < 0.98)
    plt.close(fig)
