"""Moon source/cache contracts; these do not validate terrain measurements."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from ssatk.data import data_path
from ssatk.io import moon_maps


def test_default_moon_bake_works_offline_without_modifying_packaged_data(tmp_path, monkeypatch):
    """C: real installed inputs produce decodable viewer textures with no fetch.

    Check source hashes before/after and decoded dimensions, not terrain truth.
    The small horizon grid keeps this preparation regression inexpensive.
    """
    def reject_download(*args, **kwargs):
        pytest.fail("The default Moon bake must not download its inputs")

    monkeypatch.setattr(moon_maps.urllib.request, "urlretrieve", reject_download)
    with data_path("moon_dem.npz", package="ssa_data_lunar") as dem_path, \
            data_path("moon.png", package="ssa_data_lunar") as image_path:
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dem_path, image_path)]
        with np.load(dem_path) as source:
            height, width = source["elev_km"].shape
        # Console launchers pass the return value to sys.exit; a returned path
        # falsely signals failure even when all textures were written.
        assert moon_maps.cli(["--outdir", str(tmp_path), "--horizon-width", "16",
                              "--n-step", "4", "--albedo-max", "64"]) is None
        after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dem_path, image_path)]
        assert after == before

    with Image.open(tmp_path / "moon_normal.png") as image:
        image.load()
        assert image.size == (width, height)
    with Image.open(tmp_path / "moon_albedo.jpg") as image:
        image.load()
        assert image.size == (64, 32)
    meta = json.loads((tmp_path / "moon_horizon_meta.json").read_text())
    for i in range(meta["n_az"] // 4):
        with Image.open(tmp_path / f"moon_horizon_{i}.png") as image:
            image.load()
            assert image.size == (16, 8) and image.mode == "RGBA"


@pytest.mark.parametrize("source", ["packaged", "nasa"])
def test_explicit_dem_overrides_source_while_albedo_uses_selected_source(tmp_path, monkeypatch, source):
    """C: a custom elevation grid wins; only the opted-in NASA albedo fetches.

    Loaded values must exactly equal the specified array and source pixels.
    The NASA branch uses a literal local download fixture, not NASA truth.
    """
    dem_path = tmp_path / "custom.npz"
    prescribed = np.arange(32, dtype=float).reshape(4, 8) / 1000
    np.savez(dem_path, elev_km=prescribed)
    fixture = tmp_path / "download.png"
    Image.new("RGB", (8, 4), (13, 29, 47)).save(fixture)
    requests = []

    def download(url, directory):
        assert source == "nasa", "Packaged sources must not fetch"
        requests.append(url)
        return str(fixture)

    monkeypatch.setattr(moon_maps, "download_source", download)
    dem, albedo = moon_maps._load_sources(str(dem_path), None, source, str(tmp_path))
    np.testing.assert_array_equal(dem, prescribed)
    if source == "nasa":
        assert requests == [moon_maps.ALBEDO_URL]
        expected = np.full((4, 8, 3), (13, 29, 47), dtype=np.uint8)
    else:
        assert requests == []
        with data_path("moon.png", package="ssa_data_lunar") as path, Image.open(path) as image:
            expected = np.array(image.convert("RGB"))
    np.testing.assert_array_equal(np.asarray(albedo), expected)
