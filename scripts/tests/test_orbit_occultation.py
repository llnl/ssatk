"""Analytic ray/sphere oracles (R1) and Matplotlib rendering contracts (C)."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ssatk.plots._body_surfaces import BodySurface, body_disc
from ssatk.plots.occultation import (
    _clip_segments, _projected_orbit, _visible, set_orbit_occlusion,
)
from ssatk.plots.rendering import low_spec_mode


def test_points_front_back_inside_tangent_and_translated_bodies():
    """R1: unit sphere viewed along +Z; exact Pythagorean surface points."""
    points = np.array([[0, 0, -2], [0, 0, 2], [.8, 0, .6],
                       [.8, 0, -.6], [.8, 0, .5], [1, 0, -2], [1.1, 0, -2]])
    expected = [False, True, True, False, False, True, True]
    for center in [np.zeros(3), np.array([12., -7, 3])]:
        np.testing.assert_array_equal(_visible(points+center, [(center, 1)], direction=[0, 0, 1]), expected)


@pytest.mark.parametrize('perspective', [False, True])
def test_segment_with_visible_endpoints_clips_at_analytic_limb(perspective):
    """R1: orthographic limb x=±1; eye Z=5 gives x=±7/sqrt(24) at Z=-2.

    sin(half cone angle)=R/d=1/5. Absolute tolerance 1e-12 scene units
    accommodates double-precision quadratic roots; no sampled oracle.
    """
    camera = dict(eye=np.array([0., 0, 5])) if perspective else dict(direction=np.array([0., 0, 1]))
    limb = 7/np.sqrt(24) if perspective else 1.
    result, indices = _clip_segments([[[-3, 0, -2], [3, 0, -2]]], [(np.zeros(3), 1)], **camera)
    np.testing.assert_allclose(result, [[[-3, 0, -2], [-limb, 0, -2]],
                                        [[limb, 0, -2], [3, 0, -2]]], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(indices, [0, 0])


def test_two_bodies_and_sphere_crossing_preserve_colors_and_nan_gaps():
    """R1: disjoint radius-1 shadows centred at x=-2,+2 leave three intervals."""
    segments = np.array([[[-5, 0, -2], [5, 0, -2]],
                         [[np.nan, 0, 0], [6, 0, 0]], [[-2, 0, 0], [2, 0, 0]]])
    result, indices = _clip_segments(segments[:2], [(np.array([-2., 0, 0]), 1),
                                                   (np.array([2., 0, 0]), 1)], direction=[0, 0, 1])
    np.testing.assert_allclose(result[..., 0], [[-5, -3], [-1, 1], [3, 5]], atol=1e-12)
    np.testing.assert_array_equal(indices, [0, 0, 0])
    result, _ = _clip_segments(segments[2:], [(np.zeros(3), 1)], direction=[0, 0, 1])
    np.testing.assert_allclose(result[..., 0], [[-2, -1], [1, 2]], atol=1e-12)


def _pixels(fig):
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    red = (rgb[..., 0] > 180) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    green = (rgb[..., 1] > 100) & (rgb[..., 0] < 80) & (rgb[..., 2] < 80)
    return red.sum(), green.sum()


@pytest.mark.parametrize('projection', ['ortho', 'persp'])
@pytest.mark.parametrize('kind', ['line', 'scatter', 'collection'])
def test_render_camera_rotation_restores_states_and_changes_visible_side(projection, kind):
    """C/R1: camera ±Z sees only its corresponding red/green exterior track."""
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(projection='3d')
    with low_spec_mode():
        body = BodySurface(ax, 'moon', 1, n=24)
    points = np.array([[-.3, .35, 2], [.3, .35, 2]], dtype=float)
    originals = []
    for sign, color in [(1, 'red'), (-1, 'lime')]:
        xyz = points*np.array([1, sign, sign])
        if kind == 'line':
            artist, = ax.plot(*xyz.T, color=color, lw=4, marker='o')
            original = lambda a=artist: np.column_stack(a.get_data_3d())
        elif kind == 'scatter':
            artist = ax.scatter(*xyz.T, c=color, s=60, depthshade=False)
            original = lambda a=artist: np.column_stack(a._offsets3d)
        else:
            artist = Line3DCollection([xyz], colors=color, linewidths=4)
            ax.add_collection3d(artist)
            original = lambda a=artist: np.asarray(a._segments3d)[0]
        originals.append((original, xyz.copy()))
    ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3))
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type(projection)
    ax.set_axis_off()
    for angle, index in [(90, 0), (-90, 1), (90, 0)]:
        ax.view_init(elev=angle, azim=0, roll=30)
        counts = _pixels(fig)
        assert counts[index] > 10 and counts[1-index] == 0
        for read, expected in originals:
            np.testing.assert_array_equal(read(), expected)
    # Updating/removing a body cannot leave a phantom occluder behind.
    body.update(center=(2, 2, 0))
    assert all(n > 10 for n in _pixels(fig))
    body.update()
    set_orbit_occlusion(ax, False)
    assert all(n > 10 for n in _pixels(fig))
    set_orbit_occlusion(ax)
    body.artist.remove()
    assert all(n > 10 for n in _pixels(fig))
    plt.close(fig)


@pytest.mark.parametrize('view,axes', [('xy', (0, 1)), ('xz', (0, 2)), ('yz', (1, 2))])
def test_projected_coordinate_panels_use_depth_and_keep_front_line(view, axes):
    """C/R1: orthographic panels look along the positive omitted coordinate."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    with low_spec_mode():
        body_disc(ax, 'moon', 1, view=view)
    missing = 3-sum(axes)
    for depth, color in [(2., 'red'), (-2., 'lime')]:
        xyz = np.zeros((2, 3))
        xyz[:, missing] = depth
        xyz[:, axes[0]] = [-.5, .5]
        artist = LineCollection([xyz[:, axes]], colors=color, linewidths=5)
        ax.add_collection(artist)
        _projected_orbit(artist, xyz[None], view)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect='equal')
    ax.set_axis_off()
    red, green = _pixels(fig)
    assert red > 10 and green == 0
    plt.close(fig)
