from __future__ import annotations
import html
import ast
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict
from pathlib import Path

from ssatk.plots.figpath import figpath

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".log", ".pdf", ".mp4", ".html"}

NOISE_PATTERNS = [
    r"^Figure saved at:.*$",
    r"^Saved plot to:.*$",
    r"^Saved dashboard:.*$",
    r"^Saved MP4:.*$",
    r"^Saved GIF:.*$",
    r"^Saved: .*?$",
    r"^Writing gif:.*$",
    r"^Wrote .*?$",
    r"^Writing video:.*$",
    r"^Frame cleanup complete\..*$",
    r"^Rendered frame .*?$",
    r"^Rendering MP4 .*?$",
    r"^GIF saved to:.*$",
    r"^Returning arrays shaped:.*$",
    r"^Returning arrays with varying shapes:.*$",
    r"^Plotting orbit\..*$",
    r"^Plotting two orbits.*$",
    r"^Calculating orbit\..*$",
    r"^Calculating 2 orbit\..*$",
    r"^Calculating 2 different orbit\..*$",
    r"^Saving frames to temp dir:.*$",
    r"^\[gifify\].*$",
    r"^\s*\d+%\|.*$",
    r"^Rendered \d+/\d+ frames.*$",
    r"^differential_evolution step \d+:.*$",
]
NOISE_REGEXES = [re.compile(p) for p in NOISE_PATTERNS]


def get_figsave_root() -> Path:
    probe = Path(figpath("probe.tmp")).expanduser().resolve()
    return probe.parent


FIGSAVE_ROOT = get_figsave_root()


@dataclass
class DemoResult:
    name: str
    title: str
    description: str
    status: str
    runtime_seconds: float
    output_dir: str
    files: list[str]
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    category: str = ""


class DemoTimeoutError(TimeoutError):
    """Raised when one demo exceeds the gallery time budget."""


class demo_timeout:
    """Unix SIGALRM timeout wrapper; no-op where alarms are unavailable."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if not self.seconds or not hasattr(signal, "SIGALRM"):
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False

    def _raise_timeout(self, signum, frame):
        raise DemoTimeoutError(f"Demo exceeded {self.seconds:g} seconds")


def discover_demo_files(demos_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(demos_dir.rglob("demo_*.py")):
        if "__pycache__" in path.parts:
            continue
        if not _gallery_include_enabled(path):
            continue
        candidates.append(path)
    return candidates


def _gallery_include_enabled(path: Path) -> bool:
    """Return False when a demo module opts out of gallery rendering."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return True

    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "GALLERY_INCLUDE" in names and isinstance(node.value, ast.Constant):
                return bool(node.value.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "GALLERY_INCLUDE" and isinstance(node.value, ast.Constant):
                return bool(node.value.value)
    return True


def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution. dataclasses resolves annotations through
    # sys.modules[cls.__module__], so a module defining a dataclass under
    # "from __future__ import annotations" raises AttributeError on a module
    # that is not registered first.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def snapshot_figsave_files() -> dict[Path, float]:
    if not FIGSAVE_ROOT.exists():
        return {}
    out = {}
    for p in FIGSAVE_ROOT.rglob("*"):
        if p.is_file():
            try:
                out[p.resolve()] = p.stat().st_mtime
            except OSError:
                pass
    return out


def changed_files(before: dict[Path, float], after: dict[Path, float], slack: float = 1e-9) -> list[Path]:
    changed = []
    for p, m_after in after.items():
        m_before = before.get(p)
        if m_before is None or (m_after - m_before) > slack:
            changed.append(p)
    return sorted(changed)


def relpath_for_report(target: Path, report_root: Path) -> str:
    return os.path.relpath(str(target), str(report_root))


def _filter_console_text(text: str, max_lines: int = 120) -> str:
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if any(rx.match(s) for rx in NOISE_REGEXES):
            continue
        kept.append(s)
    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()
    if len(kept) > max_lines:
        head_n = max_lines // 2
        tail_n = max_lines - head_n
        omitted = len(kept) - head_n - tail_n
        kept = kept[:head_n] + ["", f"... [{omitted} lines omitted] ...", ""] + kept[-tail_n:]
    return "\n".join(kept)


def _write_text_if_nonempty(path: Path, text: str) -> None:
    if text.strip():
        path.write_text(text, encoding="utf-8")


def _invoke_demo(module, path: Path, output_root: Path, cwd: Path | None = None, *, fast: bool = True):
    if hasattr(module, "run"):
        try:
            return module.run(output_root)
        except TypeError:
            return module.run()
    if hasattr(module, "main"):
        call_variants = [
            {"make_figures": True, "fast": fast, "verbose": True},
            {"make_figures": True, "fast": fast},
            {"make_figures": True},
            {"fast": fast, "verbose": True},
            {"fast": fast},
            {},
        ]
        last_exc = None
        for kwargs in call_variants:
            try:
                return module.main(**kwargs)
            except TypeError as ex:
                last_exc = ex
        if last_exc is not None:
            raise last_exc

    env = os.environ.copy()
    env["SSAPY_DEMO_OUTPUT_DIR"] = str(output_root)
    return subprocess.run(
        [sys.executable, str(path.resolve())],
        cwd=str((cwd or path.parent).resolve()),
        env=env,
        capture_output=True,
        text=True,
    )


def _demo_relative_path(path: Path, demos_dir: Path | None) -> Path:
    if demos_dir is not None:
        try:
            return path.resolve().relative_to(Path(demos_dir).resolve())
        except ValueError:
            pass
    return Path(path.name)


def _safe_log_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def _safe_path_part(value: str, default: str = "uncategorized") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return cleaned or default


def _organized_figure_dir(output_root: Path, category: str, name: str) -> Path:
    category_part = _safe_path_part(category or "misc", default="misc")
    return _gallery_figures_root(output_root) / category_part


def _gallery_figures_root(output_root: Path) -> Path:
    local_root = output_root / "figures"
    return local_root if local_root.exists() else output_root.parent / "figures"


def _organize_demo_files(paths: list[Path], output_root: Path, category: str, name: str) -> list[Path]:
    """Move flat gallery figure outputs into ``figures/<category>/``.

    Demos historically saved everything under ``figures``.  Category
    folders keep the gallery browsable without burying files under a second
    per-demo directory.  Files already in subfolders, logs, and outputs outside
    ``output_root`` are left untouched.
    """
    figures_root = _gallery_figures_root(output_root).resolve()
    target_dir = _organized_figure_dir(output_root, category, name)
    organized: list[Path] = []
    for path in paths:
        src = Path(path).resolve()
        try:
            rel = src.relative_to(figures_root)
        except ValueError:
            organized.append(src)
            continue
        if len(rel.parts) != 1:
            organized.append(src)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / src.name
        if dest.exists() and dest.resolve() != src:
            stem = dest.stem
            suffix = dest.suffix
            counter = 2
            while dest.exists() and dest.resolve() != src:
                dest = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        if dest.resolve() != src:
            shutil.move(str(src), str(dest))
        organized.append(dest.resolve())
    return sorted(organized)


def run_demo_script(
    path: Path,
    output_root: Path,
    demos_dir: Path | None = None,
    *,
    fast: bool = True,
    timeout_seconds: float | None = 180,
) -> DemoResult:
    relative_path = _demo_relative_path(path, demos_dir)
    name = relative_path.with_suffix("").as_posix()
    category = "" if relative_path.parent == Path(".") else relative_path.parent.as_posix()
    safe_name = _safe_log_name(name)
    title = path.stem.replace("_", " ").title()
    description = f"Demo from {relative_path.as_posix()}"
    start = time.time()
    before = snapshot_figsave_files()
    stdout_text = ""
    stderr_text = ""
    demo_cwd = Path(demos_dir).resolve().parent if demos_dir is not None else path.parent.resolve()

    try:
        module = import_module_from_path(path)
        if hasattr(module, "GALLERY_CATEGORY"):
            category = str(module.GALLERY_CATEGORY).strip() or category
        if hasattr(module, "TITLE"):
            title = str(module.TITLE)
        if hasattr(module, "DESCRIPTION"):
            description = str(module.DESCRIPTION)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with demo_timeout(timeout_seconds), redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = _invoke_demo(module, path, output_root, cwd=demo_cwd, fast=fast)

        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()

        if isinstance(result, subprocess.CompletedProcess):
            stdout_text = _filter_console_text((captured_stdout or "") + (result.stdout or ""))
            stderr_text = _filter_console_text((captured_stderr or "") + (result.stderr or ""))
            if result.returncode != 0:
                raise RuntimeError(
                    f"Subprocess exited with code {result.returncode}\n\n"
                    f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                )
        else:
            stdout_text = _filter_console_text(captured_stdout)
            stderr_text = _filter_console_text(captured_stderr)

        if isinstance(result, dict):
            title = str(result.get("title", title))
            description = str(result.get("description", description))

        after = snapshot_figsave_files()
        touched = _organize_demo_files(changed_files(before, after), output_root, category, name)
        files = sorted({relpath_for_report(p, output_root) for p in touched})

        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = log_dir / f"{safe_name}_stdout.txt"
        stderr_file = log_dir / f"{safe_name}_stderr.txt"
        _write_text_if_nonempty(stdout_file, stdout_text)
        _write_text_if_nonempty(stderr_file, stderr_text)
        if stdout_text:
            files.append(relpath_for_report(stdout_file, output_root))
        if stderr_text:
            files.append(relpath_for_report(stderr_file, output_root))
        files = sorted(set(files))

        return DemoResult(
            name=name,
            title=title,
            description=description,
            status="success",
            runtime_seconds=time.time() - start,
            output_dir=str(output_root),
            files=files,
            stdout=stdout_text,
            stderr=stderr_text,
            category=category,
        )
    except Exception:
        err = traceback.format_exc()
        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        err_file = log_dir / f"{safe_name}_ERROR.txt"
        err_file.write_text(err, encoding="utf-8")

        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = log_dir / f"{safe_name}_stdout.txt"
        stderr_file = log_dir / f"{safe_name}_stderr.txt"
        _write_text_if_nonempty(stdout_file, stdout_text)
        _write_text_if_nonempty(stderr_file, stderr_text)

        after = snapshot_figsave_files()
        touched = _organize_demo_files(changed_files(before, after), output_root, category, name)
        files = {relpath_for_report(p, output_root) for p in touched}
        files.add(relpath_for_report(err_file, output_root))
        if stdout_text:
            files.add(relpath_for_report(stdout_file, output_root))
        if stderr_text:
            files.add(relpath_for_report(stderr_file, output_root))

        return DemoResult(
            name=name,
            title=title,
            description=description,
            status="failed",
            runtime_seconds=time.time() - start,
            output_dir=str(output_root),
            files=sorted(files),
            stdout=stdout_text,
            stderr=stderr_text,
            error=err,
            category=category,
        )


def render_file_preview(file_rel: str) -> str:
    path = html.escape(file_rel)
    name = html.escape(Path(file_rel).name)
    ext = Path(file_rel).suffix.lower()

    if ext in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return (
            f'<div class="asset">'
            f'<a href="{path}" target="_blank">'
            f'<img src="{path}" alt="{name}"></a>'
            f'<div class="caption">{html.escape(file_rel)}</div>'
            f'</div>'
        )

    if ext == ".gif":
        return (
            f'<div class="asset">'
            f'<a href="{path}" target="_blank">'
            f'<img src="{path}" alt="{name}"></a>'
            f'<div class="caption">{html.escape(file_rel)}</div>'
            f'</div>'
        )

    if ext == ".mp4":
        return (
            f'<div class="asset">'
            f'<video controls autoplay muted loop playsinline preload="metadata" '
            f'style="width:100%; height:180px; object-fit:contain; background:#0a0f22; border-radius:8px;">'
            f'<source src="{path}" type="video/mp4">'
            f'</video>'
            f'<div class="caption">{html.escape(file_rel)}</div>'
            f'</div>'
        )

    if ext == ".webm":
        return (
            f'<div class="asset">'
            f'<video controls autoplay muted loop playsinline preload="metadata" '
            f'style="width:100%; height:180px; object-fit:contain; background:#0a0f22; border-radius:8px;">'
            f'<source src="{path}" type="video/webm">'
            f'</video>'
            f'<div class="caption">{html.escape(file_rel)}</div>'
            f'</div>'
        )

    if ext in TEXT_EXTS:
        return (
            f'<div class="asset">'
            f'<a class="file-link" href="{path}" target="_blank">{html.escape(file_rel)}</a>'
            f'<div class="caption">File output</div>'
            f'</div>'
        )

    return (
        f'<div class="asset">'
        f'<a class="file-link" href="{path}" target="_blank">{html.escape(file_rel)}</a>'
        f'<div class="caption">File output</div>'
        f'</div>'
    )


def build_html_report(results: list[DemoResult], out_root: Path) -> None:
    cards = []
    for r in results:
        badge_class = "ok" if r.status == "success" else "fail"
        previews = "\n".join(render_file_preview(f) for f in r.files) or "<p>No output files found.</p>"

        stdout_block = ""
        if r.stdout:
            stdout_block = (
                "<details open><summary>Console output</summary>"
                f"<pre>{html.escape(r.stdout)}</pre></details>"
            )

        stderr_block = ""
        if r.stderr:
            stderr_block = (
                "<details><summary>Warnings / stderr</summary>"
                f"<pre>{html.escape(r.stderr)}</pre></details>"
            )

        error_block = ""
        if r.error:
            error_block = (
                "<details open><summary>Error traceback</summary>"
                f"<pre>{html.escape(r.error)}</pre></details>"
            )

        category_line = ""
        if r.category:
            category_line = f'<p class="meta"><strong>Category:</strong> {html.escape(r.category)}</p>'

        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <h2>{html.escape(r.title)}</h2>
                <span class="badge {badge_class}">{html.escape(r.status)}</span>
              </div>
              <p class="meta"><strong>Demo:</strong> {html.escape(r.name)}</p>
              {category_line}
              <p>{html.escape(r.description)}</p>
              <p class="meta"><strong>Runtime:</strong> {r.runtime_seconds:.2f}s</p>
              <div class="gallery">
                {previews}
              </div>
              {stdout_block}
              {stderr_block}
              {error_block}
            </section>
            """
        )

    total = len(results)
    success = sum(r.status == "success" for r in results)
    failed = sum(r.status == "failed" for r in results)
    manifest = [asdict(r) for r in results]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SSAPy Demo Gallery</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      padding: 0;
      background: #0b1020;
      color: #e8ecf3;
    }}
    header {{
      padding: 32px;
      background: linear-gradient(90deg, #111936, #1c2d5a);
      border-bottom: 1px solid #2b3f79;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 2.2rem;
    }}
    .subtitle {{
      color: #c7d2e6;
      margin: 0;
    }}
    .summary {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin: 24px 0;
    }}
    .summary-box {{
      background: #121a31;
      border: 1px solid #24345f;
      border-radius: 12px;
      padding: 16px 20px;
      min-width: 180px;
    }}
    .card {{
      background: #121a31;
      border: 1px solid #24345f;
      border-radius: 14px;
      padding: 18px;
      margin: 18px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge.ok {{
      background: #16351f;
      color: #7CFC9A;
      border: 1px solid #2a6a3e;
    }}
    .badge.fail {{
      background: #3a1616;
      color: #ff9f9f;
      border: 1px solid #7c2b2b;
    }}
    .meta {{
      color: #b8c3da;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 16px;
      margin-bottom: 16px;
    }}
    .asset {{
      background: #0f1730;
      border: 1px solid #22325e;
      border-radius: 10px;
      padding: 10px;
    }}
    .asset img {{
      width: 100%;
      height: 180px;
      object-fit: contain;
      background: #0a0f22;
      border-radius: 8px;
      display: block;
    }}
    .caption {{
      margin-top: 8px;
      font-size: 0.9rem;
      color: #cbd5e1;
      word-break: break-word;
    }}
    .file-link {{
      color: #8ec5ff;
      text-decoration: none;
    }}
    details {{
      margin-top: 12px;
      background: #0f1730;
      border: 1px solid #22325e;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: bold;
      color: #d8e3f4;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      color: #dbe7ff;
      margin-top: 10px;
    }}
    footer {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 24px 40px 24px;
      color: #99a7c4;
    }}
  </style>
</head>
<body>
  <header>
    <h1>SSAPy Demo Gallery</h1>
    <p class="subtitle">Auto-generated demo report with artifacts and filtered console summaries.</p>
  </header>
  <main>
    <section class="summary">
      <div class="summary-box"><strong>Total demos</strong><br>{total}</div>
      <div class="summary-box"><strong>Successful</strong><br>{success}</div>
      <div class="summary-box"><strong>Failed</strong><br>{failed}</div>
      <div class="summary-box"><strong>Manifest</strong><br><a class="file-link" href="manifest.json">manifest.json</a></div>
    </section>
    {''.join(cards)}
  </main>
  <footer>
    Generated at {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}
  </footer>
</body>
</html>
"""
    (out_root / "index.html").write_text(html_text, encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_all_demos(
    demos_dir: Path,
    output_root: Path,
    clean: bool = True,
    *,
    fast: bool = True,
    timeout_seconds: float | None = 180,
) -> list[DemoResult]:
    demos_dir = Path(demos_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[DemoResult] = []
    for path in discover_demo_files(demos_dir):
        rel = path.relative_to(demos_dir)
        print(f"[demo] running {rel.as_posix()}")
        result = run_demo_script(
            path,
            output_root,
            demos_dir=demos_dir,
            fast=fast,
            timeout_seconds=timeout_seconds,
        )
        print(f"[demo] {rel.as_posix()}: {result.status} ({result.runtime_seconds:.2f}s)")
        results.append(result)

    build_html_report(results, output_root)

    import shutil as _shutil
    bundle_base = output_root.parent / output_root.name
    _shutil.make_archive(str(bundle_base), "zip", root_dir=str(output_root))

    return results
