"""Gallery raster detail against literal, asymmetric 8K color landmarks."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from ssatk.demo_gallery import _gallery_textures
from ssatk.plots import build_satellite_viewer
from ssatk.plots import plotutils
from ssatk.plots.rendering import low_spec_mode


@pytest.fixture
def body_maps(tmp_path, monkeypatch):
    """Known colors at four distant texels, with no geographic interpretation."""
    monkeypatch.setenv("SSATK_CACHE", str(tmp_path))
    expected = [(240, 30, 20), (20, 220, 40), (30, 40, 230), (220, 210, 30)]
    image = Image.new("RGB", (8192, 4096))
    for color, box in zip(expected, [(0, 0, 4096, 2048), (4096, 0, 8192, 2048),
                                    (0, 2048, 4096, 4096), (4096, 2048, 8192, 4096)]):
        image.paste(color, box)
    paths = [tmp_path / "earth/earth_day_8192.jpg", tmp_path / "moon_albedo.jpg"]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        image.save(path, quality=100, subsampling=0)
    return paths, expected


@pytest.mark.parametrize("loader", [plotutils.load_earth_file, plotutils.load_moon_file])
def test_gallery_map_retains_native_grid_and_color_landmarks(body_maps, loader):
    """8K dimensions are exact; JPEG landmark RGB differs by at most 2/255."""
    _, expected = body_maps
    with low_spec_mode(True), _gallery_textures(), loader() as image:
        assert image.size == (8192, 4096)
        samples = [image.getpixel(point) for point in
                   [(512, 512), (7680, 512), (512, 3584), (7680, 3584)]]
        np.testing.assert_allclose(samples, expected, rtol=0, atol=2)


def test_gallery_browser_embeds_full_grid_and_color_landmarks(body_maps):
    """The browser payload preserves the same independent 8K raster reference."""
    paths, expected = body_maps
    with _gallery_textures():
        payload = base64.b64decode(build_satellite_viewer.load_textures()["day"])
    assert payload == paths[0].read_bytes()
    with Image.open(BytesIO(payload)) as image:
        assert image.size == (8192, 4096)
        samples = [image.getpixel(point) for point in
                   [(512, 512), (7680, 512), (512, 3584), (7680, 3584)]]
        np.testing.assert_allclose(samples, expected, rtol=0, atol=2)
