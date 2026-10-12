"""Publishing contracts, checked against files actually served over HTTP.

These are deployment regressions, not astrodynamics or image-fidelity tests.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.request import urlopen
import zipfile

import pytest

from ssatk.demo_gallery import run_all_demos, run_demo_script
from ssatk.gallery_validation import validate_gallery


SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="red"/></svg>'


def _demo(folder, source):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "demo_portable.py"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize("directory", ["site", "custom output/documents"])
def test_published_zip_serves_real_assets_under_project_url(tmp_path, monkeypatch, directory):
    """C: relocate the ZIP alone; HTTP bytes must equal literal fixture assets."""
    old_output = tmp_path / "unrelated-output"
    old_output.mkdir()
    (old_output / "keep.txt").write_text("untouched")
    monkeypatch.setenv("SSATK_OUTPUT_DIR", str(old_output))
    monkeypatch.setenv("SSAPY_DEMO_OUTPUT_DIR", "previous-value")
    demos = tmp_path / "demos"
    _demo(demos, f'''
from pathlib import Path
from ssatk.plots.figpath import figpath
def main(make_figures=True, fast=False):
    Path(figpath("linked/image space#1.svg")).write_text({SVG!r})
    Path(figpath("linked/view.html")).write_text('<link rel="stylesheet" href="style.css"><img src="image%20space%231.svg">')
    Path(figpath("linked/style.css")).write_text('body {{ background: url("image%20space%231.svg"); }}')
''')
    output = tmp_path / directory
    results = run_all_demos(demos, output)
    assert [r.status for r in results] == ["success"]
    assert "figures/linked/image%20space%231.svg" in results[0].files
    assert __import__("os").environ["SSATK_OUTPUT_DIR"] == str(old_output)
    assert __import__("os").environ["SSAPY_DEMO_OUTPUT_DIR"] == "previous-value"
    assert list(old_output.iterdir()) == [old_output / "keep.txt"]

    # Serve only the archive at /ssatk/, with no access to the build tree.
    published = tmp_path / "published"
    relocated = published / "ssatk"
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        archive.extractall(relocated)
    validate_gallery(relocated)
    handler = partial(SimpleHTTPRequestHandler, directory=str(published))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/ssatk/"
        with urlopen(base) as response:
            page = response.read().decode()
        assert "<title>SSATK Demo Gallery</title>" in page
        assert "<h1>SSATK Demo Gallery</h1>" in page
        assert "SSAPy Demo Gallery" not in page
        with urlopen(base + "figures/linked/image%20space%231.svg") as response:
            assert response.read().decode() == SVG
        with urlopen(base + "figures/linked/view.html") as response:
            assert 'src="image%20space%231.svg"' in response.read().decode()
        with urlopen(base + "figures/linked/style.css") as response:
            assert "background:" in response.read().decode()
        for url in results[0].files:
            with urlopen(base + url) as response:
                assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_failed_demo_is_not_retried_and_environment_is_restored(tmp_path, monkeypatch):
    """C: an internal TypeError must execute once and remain a reported failure."""
    monkeypatch.delenv("SSATK_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SSAPY_DEMO_OUTPUT_DIR", raising=False)
    demo = _demo(tmp_path / "demos", '''
from pathlib import Path
import os
def main(make_figures=True, fast=True, verbose=True):
    with (Path(os.environ["SSATK_OUTPUT_DIR"]) / "calls.txt").open("a") as stream:
        stream.write("one call\\n")
    raise TypeError("internal demo defect")
''')
    root = tmp_path / "site"
    result = run_demo_script(demo, root)
    assert result.status == "failed"
    assert "internal demo defect" in result.error
    assert (root / "calls.txt").read_text() == "one call\n"
    assert "SSATK_OUTPUT_DIR" not in __import__("os").environ
    assert "SSAPY_DEMO_OUTPUT_DIR" not in __import__("os").environ


def test_skipped_demo_is_not_claimed_successful(tmp_path):
    """C: a missing optional dependency is visibly skipped, not successful."""
    demos = tmp_path / "demos"
    _demo(demos, 'def main():\n    return {"skipped": True, "reason": "Optional Moon maps unavailable"}\n')
    site = tmp_path / "site"
    results = run_all_demos(demos, site)
    assert results[0].status == "skipped"
    assert "Optional Moon maps unavailable" in results[0].description
    page = (site / "index.html").read_text()
    assert '<strong>Skipped</strong><br>1' in page
    assert '<strong>Successful</strong><br>0' in page


@pytest.mark.parametrize("url", [
    "missing.png", "../../ssatk_output/figures/plot.jpg",
    "/ssatk_output/figures/plot.jpg", "file:///tmp/plot.jpg",
])
def test_validator_rejects_missing_and_build_machine_urls(tmp_path, url):
    """C: literal counterexamples include the URL pattern on the broken live site."""
    (tmp_path / "index.html").write_text(f'<img src="{url}">')
    with pytest.raises(ValueError, match="Gallery cannot be published"):
        validate_gallery(tmp_path)


@pytest.mark.parametrize("filename,content", [
    ("viewer.html", '<img src="lost-texture.png">'),
    ("style.css", 'body {background: url("lost-texture.png")}'),
    ("manifest.json", json.dumps([{"files": ["missing-video.mp4"]}])),
])
def test_validator_checks_nested_viewer_css_and_manifest(tmp_path, filename, content):
    """C: checking only the landing page cannot establish asset completeness."""
    (tmp_path / "index.html").write_text("<h1>SSATK</h1>")
    (tmp_path / filename).write_text(content)
    with pytest.raises(ValueError, match="Missing asset"):
        validate_gallery(tmp_path)


def test_build_rejects_broken_nested_viewer_before_packaging(tmp_path):
    """C: a successful demo with a broken viewer must block the deployable ZIP."""
    demos = tmp_path / "demos"
    _demo(demos, '''
def run(output_root):
    (output_root / "viewer.html").write_text('<img src="missing-texture.png">')
''')
    with pytest.raises(ValueError, match="missing-texture.png"):
        run_all_demos(demos, tmp_path / "site")
    assert not (tmp_path / "site.zip").exists()
