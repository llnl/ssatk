"""R1 literal scene geometry and C plotting contracts for cislunar views."""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ssatk.constants import MOON_RADIUS, EARTH_RADIUS, RGEO
from ssatk.plots.orbit_plot import orbit_plot


@pytest.fixture
def fixed_frame(monkeypatch):
    """Synthetic Moon ephemeris: at the test epoch the rotating axes are XYZ.

    The Moon is on +X with +Y velocity. The +/-5-second derivative used by
    the real coordinate transform therefore gives an exact identity basis.
    """
    class Moon:
        def orientation(self, t):
            return np.eye(3)  # Explicit synthetic body axes; framing oracle is unchanged.
        def position(self, t):
            gps = np.atleast_1d(t.gps if hasattr(t, 'gps') else t)
            return np.array([np.full_like(gps, 384400000.), 1000.*(gps-1.4e9), np.zeros_like(gps)])
    lunar = importlib.import_module('ssatk.coordinates.lunar')
    core = importlib.import_module('ssatk.plots._cislunar_plot_core')
    moon = Moon()
    monkeypatch.setattr(lunar, 'get_body', lambda name: moon)
    monkeypatch.setattr(core, '_get_body', lambda name: moon)
    return np.array([384400000., 0., 0.])


def tracks(ax):
    return [artist for artist in ax.collections if artist.get_label() == 'Orbit Path']


def points(artist):
    if hasattr(artist, '_offsets3d'):
        return np.array(artist._offsets3d).T
    return np.asarray(artist.get_offsets())


@pytest.mark.parametrize('view', ['cislunar_3d', 'cislunar', 'cislunar_xy'])
def test_lunar_circle_fits_literal_limits_and_physical_radius(fixed_frame, view):
    """R1: 3000-km circle + 8% padding => +/-3480 km; 1e-8 km roundoff budget."""
    theta = np.linspace(0, 2*np.pi, 33)
    circle = 3e6*np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    fig, axes = orbit_plot(circle+fixed_frame, t=1.4e9, view=view)
    ax = axes[-1] if isinstance(axes, list) else axes
    expected = [(-3480., 3480.)]*2
    limits = [ax.get_xlim(), ax.get_ylim()]
    if view != 'cislunar_xy':
        expected.append((-3480., 3480.))
        limits.append(ax.get_zlim())
    np.testing.assert_allclose(limits, expected, rtol=0, atol=1e-8)
    assert ax.get_xlabel() == 'x [km]'
    dims = 2 if view == 'cislunar_xy' else 3
    visible = np.ones(len(circle), dtype=bool)
    if dims == 3:
        # Default orthographic camera at elev=30, azim=-60. A far-side point
        # is hidden exactly when its projected radius is inside the Moon.
        toward_camera = np.array([np.sqrt(3)/4, -.75, .5])
        depth = circle @ toward_camera
        visible = (depth >= 0) | (np.sum(circle**2, axis=1)-depth**2 >= 1738100.**2)
    np.testing.assert_allclose(points(tracks(ax)[0]), circle[visible, :dims]/1000., rtol=0, atol=1e-8)
    if dims == 3:
        surface = next(c for c in ax.collections if hasattr(c, '_vec'))
        np.testing.assert_allclose(np.linalg.norm(surface._vec[:3].T, axis=1), 1738.1, rtol=0, atol=1e-9)
    else:
        assert ax.patches[0].radius == pytest.approx(1738.1, abs=1e-10)
    # C: no legend entries for off-screen Earth/Lagrange points; every panel
    # stays inside the canvas (the former right=1.5 cropped the second panel).
    if view == 'cislunar_3d':
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Moon', 'Orbit Path']
    for panel in fig.axes:
        rect = panel.get_position()
        assert 0 <= rect.x0 < rect.x1 <= 1 and 0 <= rect.y0 < rect.y1 <= 1
    plt.close(fig)


@pytest.mark.parametrize('view', ['cislunar_3d', 'cislunar', 'cislunar_xy'])
def test_mixed_near_earth_and_moon_tracks_keep_one_unit_per_panel(fixed_frame, view):
    """R1/C: literal metre positions, 0.1-mm budget, either input order/length."""
    near_moon = fixed_frame + np.array([[3e6, 0, 0], [0, 3e6, 0], [-3e6, 0, 0]])
    near_earth = np.array([[7e6, 0, 0], [0, 7e6, 0], [-7e6, 0, 0], [0, -7e6, 0]])
    limits = []
    for source in [[near_moon, near_earth], [near_earth, near_moon]]:
        fig, axes = orbit_plot(source, t=1.4e9, view=view, legend=False)
        panels = axes if isinstance(axes, list) else [axes]
        for index, ax in enumerate(panels):
            lunar = len(panels) == 1 or index == 1
            assert ax.get_xlabel() == 'x [GEO]'
            dims = 2 if view == 'cislunar_xy' else 3
            expected = [(x-fixed_frame if lunar else x)[:, :dims]/RGEO for x in source]
            for artist, positions in zip(tracks(ax), expected, strict=True):
                np.testing.assert_allclose(points(artist)*RGEO, positions*RGEO, rtol=0, atol=1e-4)
            assert ax.get_legend() is None
            radius = (MOON_RADIUS if lunar else EARTH_RADIUS)/RGEO
            bounds = np.array([ax.get_xlim(), ax.get_ylim()] + ([] if dims == 2 else [ax.get_zlim()]))
            assert np.all(bounds[:, 0] <= -radius) and np.all(bounds[:, 1] >= radius)
        limits.append([(ax.get_xlim(), ax.get_ylim()) for ax in panels])
        plt.close(fig)
    np.testing.assert_allclose(limits[0], limits[1], rtol=0, atol=1e-12)


def test_xy_framing_ignores_height_but_contains_moon_and_accepts_fractional_padding(fixed_frame):
    """R1: a vertical 100,000-km excursion must not zoom out the XY view."""
    r = fixed_frame + np.array([[0, 0, 1e8], [0, 0, 2e8]])
    fig, axes = orbit_plot(r, t=1.4e9, view='cislunar_xy', scene_pad=.1)
    # GEO axis unit is based on all three coordinates, but XY physical bounds
    # are the Moon's diameter with 10% padding on each side.
    np.testing.assert_allclose(np.array([axes[1].get_xlim(), axes[1].get_ylim()])*RGEO,
                               [[-1.2*MOON_RADIUS, 1.2*MOON_RADIUS]]*2, rtol=0, atol=1e-7)
    ticks = axes[1].get_xticks()
    inside = ticks[(ticks>=axes[1].get_xlim()[0]) & (ticks<=axes[1].get_xlim()[1])]
    assert len(inside) >= 3
    plt.close(fig)
