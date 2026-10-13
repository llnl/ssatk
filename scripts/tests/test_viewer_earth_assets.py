"""Earth texture publishing contracts; these do not validate orbit physics."""

import base64
import hashlib
from io import BytesIO

from PIL import Image
import pytest

from scripts.prepare_gallery_data import _download
from ssatk.plots import build_satellite_viewer as viewer
from ssatk.plots._textures import viewer_earth_day_path


def test_cached_day_texture_embeds_exact_bytes_and_preserves_offline_fallback(tmp_path, monkeypatch):
    """C: asymmetric north/south colors survive byte-for-byte embedding.

    A tiny literal fixture checks selection and serialization only; it is
    not claimed as geographic truth. Other channels keep their original bytes.
    """
    monkeypatch.setenv("SSATK_CACHE", str(tmp_path))
    packaged = {key: base64.b64decode(data) for key, data in viewer.load_textures().items()}
    image = Image.new("RGB", (16, 8), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 8, 4))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    fixture = buffer.getvalue()
    target = viewer_earth_day_path()
    target.parent.mkdir(parents=True)
    target.write_bytes(fixture)
    textures = {key: base64.b64decode(data) for key, data in viewer.load_textures().items()}
    assert textures == dict(packaged, day=fixture)
    target.unlink()
    assert {key: base64.b64decode(data) for key, data in viewer.load_textures().items()} == packaged


def test_reference_download_checks_hash_before_replacing_cached_asset(tmp_path):
    """C: corrupted source bytes must not replace the last verified asset."""
    source = tmp_path / "source.jpg"
    source.write_bytes(b"NASA asset stand-in for the download integrity contract")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    target = _download(source.as_uri(), cache, expected)
    verified = target.read_bytes()
    source.write_bytes(b"corrupted replacement")
    # A correct cached source needs no download, even if the upstream changes.
    assert _download(source.as_uri(), cache, expected).read_bytes() == verified
    # A different required hash forces a fetch; a mismatch preserves the cache.
    with pytest.raises(ValueError, match="checksum mismatch"):
        _download(source.as_uri(), cache, "0" * 64)
    assert target.read_bytes() == verified
    assert not target.with_suffix(".jpg.part").exists()
