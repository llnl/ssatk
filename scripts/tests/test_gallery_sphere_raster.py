"""Analytic ray/sphere intersections and independent mapped-color landmarks."""
import numpy as np
import pytest
from PIL import Image
import matplotlib.pyplot as plt

from ssatk.plots._sphere_raster import sphere_intersections


def test_orthographic_intersections_match_unit_circle():
    """z = sqrt(1-x²-y²) on the near unit hemisphere, within 1e-14 units."""
    points = np.array([[0., 0., -4.], [.6, 0., -4.], [0., -.8, -4.], [1.2, 0., -4.]])
    hits, clearance = sphere_intersections(points, np.zeros(3), 1., direction=[0., 0., 1.])
    np.testing.assert_allclose(hits[:3], [[0, 0, 1], [.6, 0, .8], [0, -.8, .6]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(clearance, [1., .4, .2, -.2], rtol=0, atol=1e-14)


def test_perspective_intersections_match_analytic_tangent():
    """Eye at z=3: tangent at (sqrt(8)/3, 0, 1/3), within 3e-8 units.

    Square-root conditioning at a double root sets the tangent tolerance.
    """
    points = np.array([[0., 0., 0.], [3/np.sqrt(8), 0., 0.], [2., 0., 0.]])
    hits, clearance = sphere_intersections(points, np.zeros(3), 1., eye=np.array([0., 0., 3.]))
    np.testing.assert_allclose(hits[:2], [[0., 0., 1.], [np.sqrt(8)/3, 0., 1/3]], rtol=0, atol=3e-8)
    assert clearance[2] < 0


@pytest.mark.parametrize('projection,roll,inverted', [('ortho', 0, False), ('persp', 32, False), ('persp', -17, True)])
def test_projected_landmarks_retain_known_rgb(projection, roll, inverted, tmp_path, monkeypatch):
    """Four literal hemisphere RGB references survive projection, within 1/255.

    Landmark positions use closed-form latitude ±30°, longitude ±30° on a
    radius-two sphere. The plotting projection locates those independently
    defined points in the output, including roll and unequal/inverted limits.
    """
    from ssatk.plots import _body_surfaces

    path = tmp_path/'map.png'
    image = Image.new('RGB', (8192, 4096))
    colors = [(240, 30, 20), (20, 220, 40), (30, 40, 230), (220, 210, 30)]
    for color, box in zip(colors, [(0, 0, 4096, 2048), (4096, 0, 8192, 2048),
                                   (0, 2048, 4096, 4096), (4096, 2048, 8192, 4096)]):
        image.paste(color, box)
    image.save(path)
    monkeypatch.setenv('SSATK_GALLERY_MAX_TEXTURES', '1')
    monkeypatch.setattr(_body_surfaces, 'texture_path', lambda body: path)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set(xlim=(3, -3) if inverted else (-3, 3), ylim=(-3.5, 3.5), zlim=(-2.5, 2.5))
    ax.set_proj_type(projection); ax.view_init(elev=0, azim=0, roll=roll)
    ax.set_axis_off()
    try:
        _body_surfaces.BodySurface(ax, 'earth', 2., n=30)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())
        # x=2*cos(30)*cos(30), y=±2*cos(30)*sin(30), z=±2*sin(30).
        xyz = np.array([[1.5, -np.sqrt(3)/2, 1.], [1.5, np.sqrt(3)/2, 1.],
                        [1.5, -np.sqrt(3)/2, -1.], [1.5, np.sqrt(3)/2, -1.]])
        homogeneous = np.column_stack((xyz, np.ones(4))) @ ax.get_proj().T
        xy = ax.transData.transform(homogeneous[:, :2]/homogeneous[:, 3, None])
        ij = np.floor(xy).astype(int)
        actual = pixels[pixels.shape[0]-1-ij[:, 1], ij[:, 0], :3]
        np.testing.assert_allclose(actual, colors, rtol=0, atol=1)
    finally:
        plt.close(fig)
