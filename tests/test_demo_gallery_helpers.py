import importlib
import subprocess
from types import SimpleNamespace

import pytest


def test_demo_discovery_include_flags_and_console_filter(tmp_path):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    demos = tmp_path / "demos"
    demos.mkdir()
    (demos / "demo_keep.py").write_text("GALLERY_INCLUDE = True\n", encoding="utf-8")
    (demos / "demo_skip.py").write_text("GALLERY_INCLUDE = False\n", encoding="utf-8")
    (demos / "demo_skip_annotated.py").write_text("GALLERY_INCLUDE: bool = False\n", encoding="utf-8")
    (demos / "demo_bad_syntax.py").write_text("if = bad\n", encoding="utf-8")
    (demos / "run_all_demos.py").write_text("", encoding="utf-8")

    names = [path.name for path in module.discover_demo_files(demos)]
    assert names == ["demo_bad_syntax.py", "demo_keep.py"]
    assert module._filter_console_text("Figure saved at: x\nuseful\n\nSaved GIF: y\nlast") == "useful\n\nlast"
    long_text = "\n".join(f"line {idx}" for idx in range(10))
    filtered = module._filter_console_text(long_text, max_lines=4)
    assert "lines omitted" in filtered


def test_demo_invoke_variants_and_changed_files(tmp_path, monkeypatch):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    output_root = tmp_path / "gallery"
    output_root.mkdir()
    fig_root = tmp_path / "figs"
    fig_root.mkdir()
    monkeypatch.setattr(module, "FIGSAVE_ROOT", fig_root)

    before = module.snapshot_figsave_files()
    figure = fig_root / "figure.png"
    figure.write_text("png", encoding="utf-8")
    after = module.snapshot_figsave_files()
    assert module.changed_files(before, after) == [figure.resolve()]
    assert module.relpath_for_report(output_root / "logs" / "x.txt", output_root) == "logs/x.txt"

    run_module = SimpleNamespace(run=lambda root: {"title": "Run Title", "description": "Run Description"})
    assert module._invoke_demo(run_module, tmp_path / "demo.py", output_root)["title"] == "Run Title"

    noarg_run_module = SimpleNamespace(run=lambda: "ok")
    assert module._invoke_demo(noarg_run_module, tmp_path / "demo.py", output_root) == "ok"

    def main_without_kwargs():
        return "main-ok"

    assert module._invoke_demo(SimpleNamespace(main=main_without_kwargs), tmp_path / "demo.py", output_root) == "main-ok"

    completed = subprocess.CompletedProcess(["demo"], 0, stdout="subprocess out", stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    assert module._invoke_demo(SimpleNamespace(), tmp_path / "demo.py", output_root) is completed


def test_run_demo_script_success_failure_and_subprocess(tmp_path, monkeypatch):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    fig_root = tmp_path / "figs"
    fig_root.mkdir()
    monkeypatch.setattr(module, "FIGSAVE_ROOT", fig_root)
    output_root = tmp_path / "gallery"
    output_root.mkdir()

    success_demo = tmp_path / "demo_success.py"
    success_demo.write_text(
        "TITLE = 'Custom Success'\n"
        "DESCRIPTION = 'Custom description'\n"
        "from pathlib import Path\n"
        "def run(output_root):\n"
        "    print('Figure saved at: noisy')\n"
        "    print('important output')\n"
        f"    Path({str(fig_root / 'demo.png')!r}).write_text('png')\n"
        "    return {'title': 'Returned Title', 'description': 'Returned Description'}\n",
        encoding="utf-8",
    )
    result = module.run_demo_script(success_demo, output_root)
    assert result.status == "success"
    assert result.title == "Returned Title"
    assert result.stdout == "important output"
    assert any(name.endswith("demo_success_stdout.txt") for name in result.files)
    assert any("demo.png" in name for name in result.files)

    failure_demo = tmp_path / "demo_failure.py"
    failure_demo.write_text("def run(output_root):\n    print('before failure')\n    raise ValueError('bad demo')\n", encoding="utf-8")
    failed = module.run_demo_script(failure_demo, output_root)
    assert failed.status == "failed"
    assert "ValueError" in failed.error
    assert any(name.endswith("demo_failure_ERROR.txt") for name in failed.files)

    subprocess_demo = tmp_path / "demo_subprocess.py"
    subprocess_demo.write_text("print('imported')\n", encoding="utf-8")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="child out", stderr="child err"))
    child = module.run_demo_script(subprocess_demo, output_root)
    assert child.status == "success"
    assert child.stdout == "child out"
    assert child.stderr == "child err"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, stdout="bad out", stderr="bad err"))
    child_failed = module.run_demo_script(subprocess_demo, output_root)
    assert child_failed.status == "failed"
    assert "Subprocess exited" in child_failed.error


def test_run_demo_script_gallery_category_override_organizes_files(tmp_path, monkeypatch):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    output_root = tmp_path / "gallery"
    fig_root = output_root / "figures"
    fig_root.mkdir(parents=True)
    monkeypatch.setattr(module, "FIGSAVE_ROOT", fig_root)

    demos_dir = tmp_path / "demos"
    demo_dir = demos_dir / "legacy_bucket"
    demo_dir.mkdir(parents=True)
    demo = demo_dir / "demo_sensor.py"
    demo.write_text(
        "GALLERY_CATEGORY = 'sensor_coverage'\n"
        "from pathlib import Path\n"
        "def run(output_root):\n"
        "    Path(output_root, 'figures', 'demo_sensor.html').write_text('html')\n",
        encoding="utf-8",
    )

    result = module.run_demo_script(demo, output_root, demos_dir=demos_dir)

    assert result.status == "success"
    assert result.name == "legacy_bucket/demo_sensor"
    assert result.category == "sensor_coverage"
    assert result.files == ["figures/sensor_coverage/demo_sensor.html"]
    assert not (fig_root / "demo_sensor.html").exists()
    assert (fig_root / "sensor_coverage" / "demo_sensor.html").exists()


def test_organize_demo_files_moves_flat_figure_outputs(tmp_path):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    output_root = tmp_path / "gallery"
    figures = output_root / "figures"
    figures.mkdir(parents=True)
    flat = figures / "demo.png"
    flat.write_text("png", encoding="utf-8")
    nested = figures / "orbit_visualization" / "old" / "already.png"
    nested.parent.mkdir(parents=True)
    nested.write_text("png", encoding="utf-8")
    outside = tmp_path / "outside.png"
    outside.write_text("png", encoding="utf-8")

    moved = module._organize_demo_files(
        [flat, nested, outside],
        output_root,
        category="orbit_visualization",
        name="orbit_visualization/demo_globe_plot",
    )

    dest = figures / "orbit_visualization" / "demo.png"
    assert dest.exists()
    assert not flat.exists()
    assert nested.resolve() in moved
    assert outside.resolve() in moved
    assert dest.resolve() in moved


def test_render_previews_report_and_run_all(tmp_path, monkeypatch):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    assert "<img" in module.render_file_preview("figures/demo.png")
    assert "<img" in module.render_file_preview("figures/demo.gif")
    assert "video" in module.render_file_preview("figures/demo.mp4")
    assert "video" in module.render_file_preview("figures/demo.webm")
    assert "iframe" in module.render_file_preview("figures/demo.html")
    assert "File output" in module.render_file_preview("logs/demo.txt")
    assert "File output" in module.render_file_preview("data/demo.bin")

    output_root = tmp_path / "gallery"
    output_root.mkdir()
    result = module.DemoResult(
        name="demo_ok",
        title="Demo OK",
        description="desc",
        status="success",
        runtime_seconds=0.1,
        output_dir=str(output_root),
        files=["figures/demo.png", "logs/demo.txt"],
        stdout="hello",
        stderr="warning",
    )
    failed = module.DemoResult(
        name="demo_bad",
        title="Demo Bad",
        description="desc",
        status="failed",
        runtime_seconds=0.2,
        output_dir=str(output_root),
        files=[],
        error="traceback",
    )
    module.build_html_report([result, failed], output_root)
    assert (output_root / "index.html").exists()
    assert (output_root / "manifest.json").exists()

    demos = tmp_path / "demos"
    demos.mkdir()
    (demos / "demo_a.py").write_text("def run(output_root):\n    print('ok')\n", encoding="utf-8")
    archived = []
    monkeypatch.setattr(module.shutil, "make_archive", lambda base, fmt, root_dir: archived.append((base, fmt, root_dir)))
    results = module.run_all_demos(demos, output_root, clean=True)
    assert len(results) == 1
    assert results[0].status == "success"
    assert archived and archived[0][1] == "zip"
    assert (output_root / "index.html").exists()


def test_import_module_from_path_error(monkeypatch, tmp_path):
    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Could not load"):
        module.import_module_from_path(tmp_path / "demo.py")


def test_import_module_from_path_supports_dataclasses(tmp_path):
    """dataclasses resolve annotations via sys.modules[cls.__module__]."""

    import sys

    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    demo = tmp_path / "demo_dataclass_under_future_annotations.py"
    demo.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field\n"
        "GALLERY_INCLUDE = True\n"
        "@dataclass(frozen=True)\n"
        "class Thing:\n"
        "    values: dict = field(default_factory=dict)\n"
        "def main():\n"
        "    return Thing({'a': 1})\n",
        encoding="utf-8",
    )

    loaded = module.import_module_from_path(demo)
    assert loaded.main().values == {"a": 1}
    assert sys.modules.get(demo.stem) is loaded
    del sys.modules[demo.stem]


def test_import_module_from_path_does_not_leave_a_broken_module_registered(tmp_path):
    import sys

    module = importlib.import_module("ssapy_toolkit.demo_gallery")
    demo = tmp_path / "demo_raises_on_import.py"
    demo.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        module.import_module_from_path(demo)
    assert demo.stem not in sys.modules
