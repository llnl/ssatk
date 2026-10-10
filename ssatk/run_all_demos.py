from __future__ import annotations
import argparse
from importlib.metadata import PackageNotFoundError, distribution
import webbrowser
from pathlib import Path
from ssatk._paths import output_root
from .demo_gallery import run_all_demos


def default_output_dir() -> Path:
    return output_root() / "documents"


def _looks_like_demos_dir(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("demo_*.py"))


def find_default_demos_dir() -> Path | None:
    """Find the demo scripts for source-tree and installed-package use."""
    candidates = [
        Path.cwd() / "demos",
        Path(__file__).resolve().parents[1] / "demos",
    ]
    for candidate in candidates:
        if _looks_like_demos_dir(candidate):
            return candidate.resolve()

    try:
        dist = distribution("ssatk")
    except PackageNotFoundError:
        return None

    for file in dist.files or []:
        if str(file).replace("\\", "/") == "demos/__init__.py":
            candidate = Path(dist.locate_file(file)).parent
            if _looks_like_demos_dir(candidate):
                return candidate.resolve()

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all SSAPy demos and build a local gallery report."
    )
    parser.add_argument(
        "--demos-dir",
        default=None,
        help="Directory containing demo scripts (default: auto-detect repo or installed demos)",
    )
    parser.add_argument(
        "--output",
        default=str(default_output_dir()),
        help="Output directory for generated demo artifacts and report "
             "(default: ~/ssatk_output/documents)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in a browser after it is written",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Deprecated no-op; reports are not opened unless --open is provided",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the existing output directory before running",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run demos in full-quality mode instead of CI-friendly fast mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-demo timeout in seconds; use 0 to disable (default: 180)",
    )
    args = parser.parse_args(argv)

    if args.demos_dir is None:
        demos_dir = find_default_demos_dir()
        if demos_dir is None:
            raise SystemExit(
                "Demo directory not found. Run from a source checkout, install a "
                "wheel that includes demos, or pass --demos-dir PATH."
            )
    else:
        demos_dir = Path(args.demos_dir).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not _looks_like_demos_dir(demos_dir):
        raise SystemExit(f"Demo directory not found: {demos_dir}")

    results = run_all_demos(
        demos_dir=demos_dir,
        output_root=output_root,
        clean=not args.no_clean,
        fast=not args.full,
        timeout_seconds=None if args.timeout <= 0 else args.timeout,
    )

    success = sum(r.status == "success" for r in results)
    failed = sum(r.status == "failed" for r in results)
    report = output_root / "index.html"

    print()
    print("Demo gallery complete")
    print(f"  demos  : {demos_dir}")
    print(f"  output : {output_root}")
    print(f"  report : {report}")
    print(f"  success: {success}")
    print(f"  failed : {failed}")

    # report is already written by run_all_demos before we get here
    if args.open and report.exists():
        try:
            webbrowser.open(report.as_uri())
        except Exception:
            pass

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
