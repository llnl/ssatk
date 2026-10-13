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

from ssatk.demo_gallery import discover_demo_files, run_all_demos, run_demo_script
from ssatk.gallery_validation import validate_gallery


SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="red"/></svg>'


def _demo(folder, source):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "demo_portable.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_published_zip_serves_real_assets_under_project_url(tmp_path, monkeypatch):
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
    output = tmp_path / "custom output/documents.v1"
    results = run_all_demos(demos, output)
    assert [r.status for r in results] == ["success"]
    assert "figures/linked/image%20space%231.svg" in results[0].files
    assert __import__("os").environ["SSATK_OUTPUT_DIR"] == str(old_output)
    assert __import__("os").environ["SSAPY_DEMO_OUTPUT_DIR"] == "previous-value"
    assert list(old_output.iterdir()) == [old_output / "keep.txt"]

    # Serve only the archive at /ssatk/, with no access to the build tree.
    published = tmp_path / "published"
    relocated = published / "ssatk"
    with zipfile.ZipFile(Path(str(output) + ".zip")) as archive:
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


@pytest.mark.parametrize("strict", [False, True])
def test_cli_skip_policy_blocks_publication_but_preserves_diagnostics(tmp_path, strict):
    """C: execute the CLI with a real skipped fixture and a stale archive."""
    from ssatk.run_all_demos import main

    demos, site = tmp_path / "demos", tmp_path / "site"
    _demo(demos, 'def main(): return {"skipped": True, "reason": "Missing reference data"}\n')
    archive = tmp_path / "site.zip"
    archive.write_bytes(b"stale previous build")
    args = ["--demos-dir", str(demos), "--output", str(site)]
    status = main(args + (["--strict"] if strict else []))
    assert status == (1 if strict else 0)
    assert archive.exists() is not strict
    manifest = json.loads((site / "manifest.json").read_text())
    assert manifest[0]["status"] == "skipped"
    assert "Missing reference data" in (site / "index.html").read_text()
    if not strict:
        assert zipfile.is_zipfile(archive)


def test_gallery_excludes_standalone_gmat_but_keeps_other_comparisons():
    """C: discover real demo metadata without importing optional runtimes."""
    demos = Path(__file__).resolve().parents[2] / "demos"
    included = {p.relative_to(demos).as_posix() for p in discover_demo_files(demos)}
    assert "benchmarks/demo_gmat_benchmark.py" not in included
    assert {
        "benchmarks/demo_orekit_benchmark.py",
        "benchmarks/demo_long_term_propagation_benchmark.py",
        "benchmarks/demo_nbody_propagation_benchmark.py",
        "orbit_visualization/demo_moon_webgl.py",
        "space_environment/demo_magfield_plot.py",
    } <= included


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


@pytest.mark.parametrize("entry", ["top_level", "run"])
def test_demo_executes_once_with_named_fast_argument(tmp_path, entry):
    """C: one append per execution; a path must never be passed as the fast flag."""
    body = 'with (Path(os.environ["SSATK_OUTPUT_DIR"]) / "calls.txt").open("a") as f:\n    f.write("one call\\n")\n'
    if entry == "run":
        body = 'def run(fast=False):\n    assert fast is False\n' + ''.join('    '+line+'\n' for line in body.splitlines())
    demo = _demo(tmp_path / "demos", 'from pathlib import Path\nimport os\n'+body)
    result = run_demo_script(demo, tmp_path / "site", fast=False)
    assert result.status == "success", result.error
    assert (tmp_path / "site/calls.txt").read_text() == "one call\n"


@pytest.mark.parametrize("stage", ["import", "entry"])
def test_failure_preserves_output_for_diagnosis(tmp_path, stage):
    """C: the message emitted before a failure must survive in the report/log."""
    source = 'print("before failure")\nraise RuntimeError("fixture defect")\n'
    if stage == "entry":
        source = 'def main():\n' + ''.join('    '+line+'\n' for line in source.splitlines())
    else:
        source += 'def main(): pass\n'
    result = run_demo_script(_demo(tmp_path / "demos", source), tmp_path / "site")
    assert result.status == "failed" and "fixture defect" in result.error
    assert result.stdout == "before failure"
    assert (tmp_path / "site/logs/demo_portable_stdout.txt").read_text() == "before failure"


@pytest.mark.parametrize("entry", [False, True])
def test_timeout_includes_module_execution(tmp_path, entry):
    """C: a slow top-level statement must be bounded before the entry is called."""
    import signal
    if entry and not hasattr(signal, "SIGALRM"):
        pytest.skip("In-process entrypoint deadlines require Unix SIGALRM")
    source = 'import time\ntime.sleep(10)\n' + ('def main(): pass\n' if entry else '')
    result = run_demo_script(_demo(tmp_path / "demos", source), tmp_path / "site", timeout_seconds=.1)
    assert result.status == "failed" and "Timeout" in result.error
    assert result.runtime_seconds < 5  # generous scheduler margin; fixture sleeps 10 s


@pytest.mark.parametrize("level", ["same", "parent"])
def test_clean_build_cannot_delete_demo_sources(tmp_path, level):
    """C: output/source overlap must be rejected before deleting user files."""
    demos = tmp_path / "source/demos"
    demo = _demo(demos, 'pass\n')
    output = demos if level == "same" else demos.parent
    with pytest.raises(ValueError, match="overlap|contains"):
        run_all_demos(demos, output)
    assert demo.read_text() == 'pass\n'


def test_failed_rebuild_removes_stale_publishable_archive(tmp_path):
    """C: a previous successful ZIP must not masquerade as the new failed build."""
    demos, site = tmp_path / "demos", tmp_path / "site"
    _demo(demos, 'def main(): pass\n')
    run_all_demos(demos, site)
    assert (tmp_path / "site.zip").is_file()
    _demo(demos, 'def main(): raise RuntimeError("new defect")\n')
    results = run_all_demos(demos, site)
    assert results[0].status == "failed"
    assert not (tmp_path / "site.zip").exists()


@pytest.mark.parametrize("css_url", ['"image space.png"', r"image\ space.png", r"image\20 space.png"])
def test_css_quoted_spaces_and_comments_are_handled(tmp_path, css_url):
    """C: CSS permits whitespace in quoted URLs and ignores comments."""
    (tmp_path / "index.html").write_text('<style>/* url("ignored.png") */ body {content: "url(also-ignored.png)"; background:url(' + css_url + ')}</style>')
    with pytest.raises(ValueError, match="Missing asset"):
        validate_gallery(tmp_path)
    (tmp_path / "image space.png").write_bytes(b'fixture')
    assert validate_gallery(tmp_path)["local_references_checked"] == 1


@pytest.mark.skipif(__import__("os").name == "nt", reason="Symlink privileges are not assumed on Windows")
def test_validator_rejects_directory_symlinks_omitted_by_zip(tmp_path):
    """C: a linked directory is accessible locally but is not traversed by make_archive."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "assets").mkdir()
    (site / "assets/image.png").write_bytes(b"fixture")
    (site / "alias").symlink_to(site / "assets", target_is_directory=True)
    (site / "index.html").write_text('<img src="alias/image.png">')
    with pytest.raises(ValueError, match="Symlink"):
        validate_gallery(site)
