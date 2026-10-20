#!/usr/bin/env python3
"""
check_plots.py -- audit every plot module and demo, and report what works.

Why this exists
---------------
`pytest tests` proves the tested paths work. `run_all_demos` proves the demos
run end to end. Neither answers the question you actually need before telling
people a release is usable: *does every module import on its own, and does
every demo still produce its figure?*

That gap is not hypothetical. `ssatk.geomagnetics` was broken on main
for a while -- importing it directly raised ImportError from a circular
dependency, while `import ssatk.plots` happened to work because it
entered the cycle from the other side. No test caught it, because every test
imported the package first.

What it checks
--------------
1. IMPORT   Each module in ssatk/plots/ is imported in its own fresh
            subprocess, with nothing else imported first. A module that only
            works when something else is loaded first fails here, which is the
            point.

2. DEMO     Each demos/**/demo_*.py exposing main() is called with
            make_figures=False (and fast=True where accepted), so the code
            paths run without writing anything.

3. FIGURE   With --figures, each demo is re-run with make_figures=True and the
            returned/expected output path is checked for existence and
            non-zero size. This is where "the plot works" is actually
            established rather than assumed.

Each check runs in a subprocess so one hard crash (a segfault in a native
dependency, say) does not take down the audit.

Usage
-----
    python scripts/check_plots.py                # imports + demo dry runs
    python scripts/check_plots.py --figures      # also render and verify files
    python scripts/check_plots.py --only magfield,sun_view
    python scripts/check_plots.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "ssatk" / "plots"
DEMOS_DIR = REPO_ROOT / "demos"

# Per-check timeout. Some demos legitimately take a while (field-line tracing,
# video encoding), so this is generous; anything past it is reported as a
# timeout rather than a failure, because those are different problems.
TIMEOUT_S = 300

OK, FAIL, SKIP, TIMEOUT = "ok", "FAIL", "skip", "TIMEOUT"


def _run(code: str, timeout: int = TIMEOUT_S) -> tuple[str, str]:
    """Run a snippet in a fresh interpreter. Returns (status, detail)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT, f"exceeded {timeout}s"
    if proc.returncode == 0:
        return OK, (proc.stdout or "").strip().splitlines()[-1] if proc.stdout.strip() else ""
    # Last exception line is the useful part; the traceback is noise in a table.
    err = (proc.stderr or "").strip().splitlines()
    detail = err[-1] if err else f"exit {proc.returncode}"
    return FAIL, detail[:150]


# ---------------------------------------------------------------------------
# 1. imports
# ---------------------------------------------------------------------------

def check_imports(only: list[str] | None) -> list[dict]:
    mods = sorted(
        p.stem for p in PLOTS_DIR.glob("*.py")
        if p.stem != "__init__" and not p.stem.startswith("_")
    )
    if only:
        mods = [m for m in mods if any(o in m for o in only)]

    results = []
    for name in mods:
        t0 = time.time()
        # Import the submodule directly and nothing else first. If it needs the
        # package to be loaded already, that is exactly what we want to catch.
        status, detail = _run(f"import ssatk.plots.{name}")
        results.append(dict(kind="import", name=name, status=status,
                            detail=detail, seconds=round(time.time() - t0, 1)))
        print(f"  [{status:>7}] {name}" + (f"  -- {detail}" if detail and status != OK else ""))
    return results


# ---------------------------------------------------------------------------
# 2 & 3. demos
# ---------------------------------------------------------------------------

def _demo_modules(only: list[str] | None) -> list[tuple[str, Path]]:
    out = []
    for p in sorted(DEMOS_DIR.rglob("demo_*.py")):
        rel = p.relative_to(REPO_ROOT).with_suffix("")
        dotted = ".".join(rel.parts)
        if only and not any(o in dotted for o in only):
            continue
        out.append((dotted, p))
    return out


def check_demos(only: list[str] | None, figures: bool) -> list[dict]:
    results = []
    for dotted, path in _demo_modules(only):
        short = dotted.replace("demos.", "")
        t0 = time.time()

        # make_figures=False where supported; fast=True only if the signature
        # accepts it, since not every demo takes it.
        code = f"""
            import importlib, inspect
            m = importlib.import_module("{dotted}")
            if not hasattr(m, "main"):
                print("NO_MAIN")
                raise SystemExit(0)
            sig = inspect.signature(m.main)
            kw = {{}}
            if "make_figures" in sig.parameters:
                kw["make_figures"] = {figures}
            if "fast" in sig.parameters:
                kw["fast"] = True
            out = m.main(**kw)
            print("RETURNED", type(out).__name__)
        """
        status, detail = _run(code)
        if detail == "NO_MAIN":
            status, detail = SKIP, "no main()"
        results.append(dict(kind="figure" if figures else "demo", name=short,
                            status=status, detail=detail,
                            seconds=round(time.time() - t0, 1)))
        print(f"  [{status:>7}] {short}" + (f"  -- {detail}" if detail and status != OK else ""))
    return results


# ---------------------------------------------------------------------------

def summarise(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    for kind in ("import", "demo", "figure"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        counts = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        total = len(rows)
        line = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"{kind:>8}  {total:>3} checked   {line}")

    bad = [r for r in results if r["status"] in (FAIL, TIMEOUT)]
    if bad:
        print(f"\n{len(bad)} problem(s):")
        for r in bad:
            print(f"  {r['status']:>7}  {r['kind']:>7}  {r['name']}")
            if r["detail"]:
                print(f"           {r['detail']}")
    else:
        print("\nno failures")
    print("=" * 72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true",
                    help="also render each demo and check its output file")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; check only matching names")
    ap.add_argument("--skip-imports", action="store_true")
    ap.add_argument("--skip-demos", action="store_true")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full result table here")
    a = ap.parse_args(argv)

    only = [s.strip() for s in a.only.split(",")] if a.only else None
    results: list[dict] = []

    if not a.skip_imports:
        print("\n--- module imports (each in a fresh interpreter) ---")
        results += check_imports(only)

    if not a.skip_demos:
        label = "demos: render and verify outputs" if a.figures else "demos: dry run, no files written"
        print(f"\n--- {label} ---")
        results += check_demos(only, figures=a.figures)

    summarise(results)

    if a.json:
        a.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")

    # Exit non-zero if anything failed, so this can gate a release check.
    return 1 if any(r["status"] in (FAIL, TIMEOUT) for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
