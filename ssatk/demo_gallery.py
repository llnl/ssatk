from __future__ import annotations
import html
import ast
import importlib.util
import inspect
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from urllib.parse import quote, unquote

from ssatk._paths import SSATK_OUTPUT_ENV, output_root as default_output_root
from ssatk._gallery_layout import gallery_group, gallery_sort_key, gallery_title
from ssatk._gallery_calls import collect_figure_calls
from ssatk.gallery_validation import validate_gallery

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
    return default_output_root().expanduser().resolve() / "figures"


@contextmanager
def _gallery_output(root: Path):
    """Route plotting helpers and child processes to the portable site root."""
    names = (SSATK_OUTPUT_ENV, "SSAPY_DEMO_OUTPUT_DIR")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(root)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _gallery_textures():
    """Require the prepared body maps in demos and their child processes."""
    name = "SSATK_GALLERY_MAX_TEXTURES"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


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
    figure_calls: dict = field(default_factory=dict)


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


def snapshot_figsave_files(root: Path | None = None) -> dict[Path, tuple[int, int]]:
    root = get_figsave_root() if root is None else root
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            stat = p.stat()
            out[p.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return out


def changed_files(before: dict, after: dict) -> list[Path]:
    return sorted(p for p, stamp in after.items() if before.get(p) != stamp)


def relpath_for_report(target: Path, report_root: Path) -> str:
    # A published site cannot serve a build-machine path outside its root.
    relative = target.resolve().relative_to(report_root.resolve())
    return quote(relative.as_posix(), safe="/")


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


def _has_function_entrypoint(path: Path) -> bool:
    """Inspect without executing top-level code twice for standalone scripts."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(node, ast.FunctionDef) and node.name in {"run", "main"}
               for node in tree.body)


def _invoke_demo(module, output_root: Path, *, fast: bool = True):
    function = getattr(module, "run", None) or getattr(module, "main", None)
    signature = inspect.signature(function)
    values = {"output_dir": output_root, "output_root": output_root,
              "make_figures": True, "fast": fast, "verbose": True}
    args, kwargs = [], {}
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if name in values:
                args.append(values[name])
            elif parameter.default is not inspect.Parameter.empty:
                args.append(parameter.default)
        elif name in values and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            kwargs[name] = values[name]
    # Bind before executing: an internal TypeError must remain a single failure.
    signature.bind(*args, **kwargs)
    return function(*args, **kwargs)


def _demo_relative_path(path: Path, demos_dir: Path | None) -> Path:
    if demos_dir is not None:
        try:
            return path.resolve().relative_to(Path(demos_dir).resolve())
        except ValueError:
            pass
    return Path(path.name)


def _safe_log_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def run_demo_script(
    path: Path,
    output_root: Path,
    demos_dir: Path | None = None,
    *,
    fast: bool = True,
    timeout_seconds: float | None = 180,
) -> DemoResult:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with _gallery_output(output_root):
        result = _run_demo_script(path, output_root, demos_dir, fast, timeout_seconds)
        result.figure_calls = collect_figure_calls(path, output_root, result.name, result.files)
        return result


def _run_demo_script(path, output_root, demos_dir, fast, timeout_seconds):
    relative_path = _demo_relative_path(path, demos_dir)
    name = relative_path.with_suffix("").as_posix()
    category = "" if relative_path.parent == Path(".") else relative_path.parent.as_posix()
    safe_name = _safe_log_name(name)
    title = path.stem.replace("_", " ").title()
    description = f"Demo from {relative_path.as_posix()}"
    start = time.time()
    before = snapshot_figsave_files(output_root)
    stdout_text = ""
    stderr_text = ""
    demo_cwd = Path(demos_dir).resolve().parent if demos_dir is not None else path.parent.resolve()

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        if _has_function_entrypoint(path):
            # Import is part of demo execution: include its output and deadline.
            with demo_timeout(timeout_seconds), redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                module = import_module_from_path(path)
                category = str(getattr(module, "GALLERY_CATEGORY", category)).strip() or category
                title = str(getattr(module, "TITLE", title))
                description = str(getattr(module, "DESCRIPTION", description))
                result = _invoke_demo(module, output_root, fast=fast)
        else:
            # A script without a declared entry point executes only as __main__.
            # subprocess.run also enforces the deadline on platforms without SIGALRM.
            result = subprocess.run(
                [sys.executable, str(path.resolve())], cwd=str(demo_cwd),
                env=os.environ.copy(), capture_output=True, text=True,
                timeout=timeout_seconds or None,
            )
            stdout_buf.write(result.stdout or "")
            stderr_buf.write(result.stderr or "")
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess exited with code {result.returncode}")
        stdout_text = _filter_console_text(stdout_buf.getvalue())
        stderr_text = _filter_console_text(stderr_buf.getvalue())

        if isinstance(result, dict):
            title = str(result.get("title", title))
            description = str(result.get("description", description))
            if result.get("skipped"):
                description += " — " + str(result.get("reason", "Optional demo unavailable"))

        after = snapshot_figsave_files(output_root)
        touched = changed_files(before, after)
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
        files = sorted(set(files), key=_file_preview_sort_key)

        return DemoResult(
            name=name,
            title=title,
            description=description,
            status="skipped" if isinstance(result, dict) and result.get("skipped") else "success",
            runtime_seconds=time.time() - start,
            output_dir=str(output_root),
            files=files,
            stdout=stdout_text,
            stderr=stderr_text,
            category=category,
        )
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            # TimeoutExpired can hold bytes even when text=True.
            for buffer, value in ((stdout_buf, exc.stdout), (stderr_buf, exc.stderr)):
                if value:
                    buffer.write(value.decode(errors="replace") if isinstance(value, bytes) else value)
        stdout_text = _filter_console_text(stdout_buf.getvalue())
        stderr_text = _filter_console_text(stderr_buf.getvalue())
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

        after = snapshot_figsave_files(output_root)
        touched = changed_files(before, after)
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
            files=sorted(files, key=_file_preview_sort_key),
            stdout=stdout_text,
            stderr=stderr_text,
            error=err,
            category=category,
        )


def _file_preview_sort_key(file_rel: str) -> tuple[int, str, str, str]:
    """Animations/viewers, stills, downloads, then logs; stable names within each."""
    decoded = unquote(file_rel)
    path = Path(decoded)
    ext = path.suffix.lower()
    if ext in {".gif", ".mp4", ".webm", ".html", ".htm"}:
        group = 0
    elif ext in IMAGE_EXTS:
        group = 1
    elif ext in {".txt", ".log"} or "logs" in {part.casefold() for part in path.parts}:
        group = 3
    else:
        group = 2
    return group, decoded.casefold(), decoded, file_rel


def render_file_preview(file_rel: str, example: dict | None = None) -> str:
    preview = _render_file_preview(file_rel)
    if not example:
        return preview
    call = (
        '<div class="figure-call"><span>Figure call</span> '
        f'<code>{html.escape(example["function"])}(...)</code>'
        '<details><summary>Arguments and source</summary>'
        f'<pre><code>{html.escape(example["code"])}</code></pre>'
        '<p>Inputs and configuration are defined in the demo. '
        f'<a class="file-link" href="{html.escape(example["source"], quote=True)}" '
        f'target="_blank" rel="noopener">Demo source, lines '
        f'{int(example["first_line"])}–{int(example["last_line"])}</a>.</p>'
        '</details></div>'
    )
    return preview.removesuffix('</div>') + call + '</div>'


def _render_file_preview(file_rel: str) -> str:
    path = html.escape(file_rel)
    name = html.escape(unquote(Path(file_rel).name))
    ext = Path(unquote(file_rel)).suffix.lower()

    if ext in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return (
            f'<div class="asset">'
            f'<a href="{path}" target="_blank">'
            f'<img src="{path}" alt="{name}" loading="lazy"></a>'
            f'<div class="caption">{name}</div>'
            f'</div>'
        )

    if ext == ".gif":
        return (
            f'<div class="asset">'
            f'<a href="{path}" target="_blank">'
            f'<img src="{path}" alt="{name}" loading="lazy"></a>'
            f'<div class="caption">{name}</div>'
            f'</div>'
        )

    if ext == ".mp4":
        return (
            f'<div class="asset">'
            f'<video controls muted loop playsinline preload="metadata" '
            f'style="width:100%; height:180px; object-fit:contain; background:#0a0f22; border-radius:8px;">'
            f'<source src="{path}" type="video/mp4">'
            f'</video>'
            f'<div class="caption">{name}</div>'
            f'</div>'
        )

    if ext == ".webm":
        return (
            f'<div class="asset">'
            f'<video controls muted loop playsinline preload="metadata" '
            f'style="width:100%; height:180px; object-fit:contain; background:#0a0f22; border-radius:8px;">'
            f'<source src="{path}" type="video/webm">'
            f'</video>'
            f'<div class="caption">{name}</div>'
            f'</div>'
        )

    if ext in {".html", ".htm"}:
        return (
            f'<div class="asset"><a class="file-link" href="{path}" '
            f'target="_blank" rel="noopener">Open interactive viewer</a>'
            f'<div class="caption">{name}</div></div>'
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
    # Rebuilding an existing/externally supplied manifest must follow the same
    # order as a fresh run, without mutating the caller's DemoResult objects.
    results = sorted(
        (replace(r, title=gallery_title(r.name, r.title),
                 files=sorted(r.files, key=_file_preview_sort_key)) for r in results),
        key=gallery_sort_key,
    )
    groups = {}
    cards = {}
    for r in results:
        group = gallery_group(r.name, r.category)
        groups.setdefault(group.key, group)
        cards.setdefault(group.key, [])
        badge_class = {"success": "ok", "skipped": "skip"}.get(r.status, "fail")
        previews = "\n".join(render_file_preview(f, r.figure_calls.get(f)) for f in r.files) or "<p>No output files found.</p>"

        stdout_block = ""
        if r.stdout:
            stdout_block = (
                "<details><summary>Console output</summary>"
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

        # Search demo metadata, not captured logs, file paths or tracebacks.
        search_text = " ".join((r.title, r.name, r.description, group.title, r.category))
        description = "" if r.description == f"Demo from {r.name}.py" else f"<p>{html.escape(r.description)}</p>"
        # Every bundled demo is runnable as a module. This is a standalone
        # invocation, not a claim to reproduce the gallery's injected options.
        command = shlex.join(["python", "-m", "demos." + r.name.replace("/", ".")])
        cards[group.key].append(
            f"""
            <section class="card" data-category="{group.key}" data-demo="{html.escape(r.name, quote=True)}"
                     data-search="{html.escape(search_text, quote=True)}">
              <div class="card-header">
                <h3>{html.escape(r.title)}</h3>
                <span class="badge {badge_class}">{html.escape(r.status)}</span>
              </div>
              <p class="meta"><strong>Demo:</strong> {html.escape(r.name)}</p>
              {description}
              <div class="demo-call"><span>Run this demo</span><code>{html.escape(command)}</code></div>
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
    skipped = sum(r.status == "skipped" for r in results)
    manifest = [asdict(r) for r in results]
    category_options = ''.join(
        f'<option value="{group.key}">{html.escape(group.title)}</option>'
        for group in groups.values()
    )
    navigation = ''.join(
        f'<a class="topic-link" href="#group-{group.key}" data-group="{group.key}">'
        f'<span>{html.escape(group.title)}</span><span class="topic-count">{len(cards[group.key])}</span></a>'
        for group in groups.values()
    )
    sections = ''.join(
        f'<section class="demo-group" id="group-{group.key}" aria-labelledby="heading-{group.key}">'
        f'<div class="group-heading"><h2 id="heading-{group.key}">{html.escape(group.title)}</h2>'
        f'<span class="group-count">{len(cards[group.key])} {"demo" if len(cards[group.key]) == 1 else "demos"}</span></div>'
        f'<p class="group-description">{html.escape(group.description)}</p>'
        f'{"".join(cards[group.key])}</section>'
        for group in groups.values()
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SSATK Demo Gallery</title>
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
    .filters {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 24px 0; }}
    .filters input, .filters select {{
      background: #121a31; color: #e8ecf3; border: 1px solid #526692;
      border-radius: 6px; padding: 10px; margin-left: 8px; max-width: 100%;
    }}
    .browse-intro, .group-description {{ color: #c7d2e6; line-height: 1.5; }}
    .topic-nav {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .topic-link {{
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 14px; border: 1px solid #3b5284; border-radius: 8px;
      background: #121a31; color: #b9d9ff; text-decoration: none;
    }}
    .topic-link:hover, .topic-link:focus-visible {{ background: #1c2d5a; border-color: #8ec5ff; }}
    .topic-count, .group-count {{ color: #b8c3da; white-space: nowrap; }}
    .demo-group {{ margin-top: 42px; scroll-margin-top: 20px; }}
    .group-heading {{ display: flex; align-items: baseline; gap: 18px; border-bottom: 1px solid #3b5284; }}
    .group-heading h2 {{ margin: 0 0 12px; }}
    .card-header h3 {{ margin: 0; font-size: 1.25rem; }}
    [hidden] {{ display: none !important; }}
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
    .badge.skip {{
      background: #38311a;
      color: #f0d583;
      border: 1px solid #786a35;
    }}
    .badge.fail {{
      background: #3a1616;
      color: #ff9f9f;
      border: 1px solid #7c2b2b;
    }}
    .meta {{
      color: #b8c3da;
    }}
    .demo-call {{
      display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 12px;
      align-items: baseline; padding: 12px 14px; margin: 14px 0;
      background: #0f1730; border: 1px solid #33466f; border-radius: 8px;
      font-size: 0.9rem;
    }}
    .demo-call span {{ color: #b8c3da; }}
    .demo-call code {{ color: #dbe7ff; overflow-wrap: anywhere; user-select: all; }}
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
    .figure-call {{ margin-top: 12px; font-size: 0.85rem; overflow-wrap: anywhere; }}
    .figure-call > span {{ color: #b8c3da; margin-right: 6px; }}
    .figure-call code {{ color: #dbe7ff; }}
    .figure-call pre {{ max-height: 360px; overflow-y: auto; }}
    .file-link {{
      color: #8ec5ff;
      text-decoration: none;
      overflow-wrap: anywhere;
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
    @media (max-width: 600px) {{
      header, main {{ padding: 18px; }}
      .summary {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
      .summary-box {{ min-width: 0; padding: 12px; }}
      .filters {{ flex-direction: column; }}
      .filters label {{ display: flex; flex-direction: column; gap: 6px; min-width: 0; }}
      .filters input, .filters select {{ margin-left: 0; min-width: 0; width: 100%; box-sizing: border-box; }}
      .card {{ padding: 12px; }}
      .card-header {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ overflow-wrap: anywhere; }}
      .demo-call {{ grid-template-columns: minmax(0, 1fr); gap: 6px; }}
      .gallery, .topic-nav {{ grid-template-columns: minmax(0, 1fr); }}
      .group-heading {{ flex-wrap: wrap; gap: 8px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SSATK Demo Gallery</h1>
    <p class="subtitle">Space Situational Awareness Toolkit — plots, animations and interactive viewers.</p>
  </header>
  <main>
    <section class="summary">
      <div class="summary-box"><strong>Total demos</strong><br>{total}</div>
      <div class="summary-box"><strong>Successful</strong><br>{success}</div>
      <div class="summary-box"><strong>Skipped</strong><br>{skipped}</div>
      <div class="summary-box"><strong>Failed</strong><br>{failed}</div>
      <div class="summary-box"><strong>Manifest</strong><br><a class="file-link" href="manifest.json">manifest.json</a></div>
    </section>
    <p class="browse-intro">Start with the introductory workflows or jump to a topic below.
      Within each topic, examples progress from basic views to specialized workflows.
      Each demo shows animations and interactive views first, then images, downloads and logs.</p>
    <p class="browse-intro">The <strong>Run this demo</strong> commands work from an SSATK checkout
      or an installation that includes the demos. They use each script’s standalone defaults;
      the gallery may use faster settings.</p>
    <div class="filters">
      <label>Search <input id="search" type="search" placeholder="Orbit, Moon, attitude…"></label>
      <label>Topic <select id="category"><option value="">All topics</option>{category_options}</select></label>
      <button id="reset-filters" type="button" class="topic-link">Clear filters</button>
    </div>
    <p id="match-count" class="meta" role="status" aria-live="polite">Showing {total} of {total} demos</p>
    <nav class="topic-nav" aria-label="Demo topics">{navigation}</nav>
    {sections}
    <p id="no-matches" hidden>No demos match these filters.</p>
  </main>
  <footer>
    Generated at {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}
  </footer>
  <script>
    const search = document.getElementById('search');
    const category = document.getElementById('category');
    const cards = [...document.querySelectorAll('.card')];
    const groups = [...document.querySelectorAll('.demo-group')];
    function filterCards() {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const matches = (!category.value || card.dataset.category === category.value)
          && card.dataset.search.toLowerCase().includes(query);
        card.hidden = !matches;
        if (matches) visible++;
      }}
      for (const group of groups) {{
        const count = group.querySelectorAll('.card:not([hidden])').length;
        group.hidden = count === 0;
        group.querySelector('.group-count').textContent = `${{count}} demo${{count === 1 ? '' : 's'}}`;
        const link = document.querySelector(`.topic-link[data-group="${{group.id.slice(6)}}"]`);
        link.hidden = count === 0;
        link.querySelector('.topic-count').textContent = count;
      }}
      document.getElementById('match-count').textContent = `Showing ${{visible}} of ${{cards.length}} demos`;
      document.getElementById('no-matches').hidden = visible > 0;
    }}
    search.addEventListener('input', filterCards);
    category.addEventListener('change', filterCards);
    document.getElementById('reset-filters').addEventListener('click', () => {{
      search.value = '';
      category.value = '';
      filterCards();
    }});
    filterCards();
  </script>
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
    strict: bool = False,
) -> list[DemoResult]:
    demos_dir = Path(demos_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    # Cleaning the source or its parent irreversibly deletes user code.
    source_root = Path(__file__).resolve().parents[1]
    if (demos_dir.is_relative_to(output_root) or output_root.is_relative_to(demos_dir)
            or source_root.is_relative_to(output_root)):
        raise ValueError("Gallery output overlaps demo sources or contains the package source")
    paths = discover_demo_files(demos_dir)
    if not paths:
        raise ValueError(f"No enabled demos found in {demos_dir}")
    archive_path = output_root.parent / (output_root.name + ".zip")
    archive_path.unlink(missing_ok=True)

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[DemoResult] = []
    for path in paths:
        rel = path.relative_to(demos_dir)
        print(f"[demo] running {rel.as_posix()}")
        with _gallery_textures():
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
    validate_gallery(output_root)

    if all(result.status == "success" if strict else result.status != "failed"
           for result in results):
        shutil.make_archive(str(output_root), "zip", root_dir=str(output_root))

    return results
