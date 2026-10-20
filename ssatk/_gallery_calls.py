"""Extract figure examples from the demo source, without importing or running it.

Keep examples beside the actual drawing calls with ``# gallery-call: glob``
and ``# gallery-end`` comments. The published excerpt therefore changes with
the implementation instead of becoming a second, hand-maintained API example.
"""

from __future__ import annotations

import ast
from fnmatch import fnmatchcase
from pathlib import Path
import textwrap
from urllib.parse import quote, unquote


MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
                    ".mp4", ".webm", ".html", ".htm"}


def source_examples(source: str) -> list[dict]:
    """Read explicitly marked source regions and reject malformed markers."""
    examples = []
    start = None
    lines = source.splitlines()
    for index, line in enumerate(lines):
        marker = line.strip()
        if marker.startswith("# gallery-call:"):
            if start is not None:
                raise ValueError("Nested gallery-call markers")
            patterns = [s.strip() for s in marker.partition(":")[2].split("|") if s.strip()]
            if not patterns:
                raise ValueError("gallery-call needs an output filename or glob")
            start = index + 1
        elif marker == "# gallery-end":
            if start is None:
                raise ValueError("gallery-end without gallery-call")
            code = textwrap.dedent("\n".join(lines[start:index])).strip()
            tree = ast.parse(code)
            # Prefer calls used as statements/results, not a surrounding
            # compatibility predicate or loop iterator. A list append often
            # wraps the plotting call solely to retain its returned figure.
            statements = [n.value for n in ast.walk(tree)
                          if isinstance(n, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return))
                          and isinstance(n.value, ast.Call)]
            calls = sorted(statements or (n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                           key=lambda n: (n.lineno, n.col_offset))
            if not calls:
                raise ValueError("gallery-call region must contain a function call")
            main_call = calls[0]
            if (isinstance(main_call.func, ast.Attribute) and main_call.func.attr == "append"
                    and len(main_call.args) == 1 and isinstance(main_call.args[0], ast.Call)):
                main_call = main_call.args[0]
            examples.append(dict(patterns=patterns, code=code,
                                 function=ast.unparse(main_call.func),
                                 first_line=start + 1, last_line=index))
            start = None
    if start is not None:
        raise ValueError("Unclosed gallery-call marker")
    return examples


def collect_figure_calls(path: Path, output_root: Path, name: str, files: list[str]) -> dict:
    """Associate actual output names with source excerpts and portable source text."""
    source = path.read_text(encoding="utf-8")
    examples = source_examples(source)
    bundled = path.resolve().is_relative_to(Path(__file__).resolve().parents[1] / "demos")
    # Derive a portable source filename without trusting report metadata as a path.
    parts = Path(name).parts
    if not parts or any(p in {"..", "/", "\\"} for p in parts) or Path(name).is_absolute():
        raise ValueError("Gallery source name must be a relative demo name")
    source_path = Path("sources", *parts).with_suffix(".py.txt")
    calls = {}
    for filename in files:
        asset = Path(unquote(filename))
        if asset.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        matches = [e for e in examples if any(fnmatchcase(asset.name, p) for p in e["patterns"])]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous figure-call examples for {name}: {asset.name}")
        if not matches and bundled:
            raise ValueError(f"Missing gallery-call source example for {name}: {asset.name}")
        if matches:
            calls[filename] = dict(matches[0], source=quote(source_path.as_posix(), safe="/"))
    if calls:
        target = output_root / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return calls
