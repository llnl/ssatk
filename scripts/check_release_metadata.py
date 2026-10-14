#!/usr/bin/env python3
"""Check that every file carrying the SSATK release version agrees.

The version lives in five places: ``pyproject.toml``, ``codemeta.json``,
``CITATION.cff``, and the ``ssapy-toolkit`` compatibility distribution's own
version and its ``ssatk==`` pin. Without arguments this checks that they all
agree (the Repository Policy workflow runs it on every push and pull request).
With ``--tag vX.Y.Z`` it also checks that they equal the tag (the Publish
workflow runs it before building anything).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        sys.exit("check_release_metadata.py needs Python >= 3.11 or the tomli package")

ROOT = Path(__file__).resolve().parents[1]
COMPAT_PYPROJECT = Path("packaging", "ssapy-toolkit-compat", "pyproject.toml")


def _toml_project(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)["project"]


def release_versions(root: Path = ROOT) -> dict[str, str]:
    """Return {location: version} for every place the release version is recorded."""
    found = {"pyproject.toml": _toml_project(root / "pyproject.toml")["version"]}

    found["codemeta.json"] = json.loads((root / "codemeta.json").read_text(encoding="utf-8"))["version"]

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", citation, flags=re.MULTILINE)
    found["CITATION.cff"] = match.group(1) if match else "<missing>"

    compat = _toml_project(root / COMPAT_PYPROJECT)
    found["ssapy-toolkit-compat version"] = compat["version"]
    pins = [dep for dep in compat.get("dependencies", []) if re.match(r"ssatk\s*==", dep)]
    found["ssapy-toolkit-compat ssatk pin"] = pins[0].split("==", 1)[1].strip() if len(pins) == 1 else "<missing>"
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="release tag, e.g. v1.1.4; versions must equal it without the leading v")
    args = parser.parse_args(argv)

    found = release_versions()
    expected = args.tag[1:] if args.tag and args.tag.startswith("v") else args.tag
    if expected is None:
        expected = found["pyproject.toml"]

    mismatched = {where: version for where, version in found.items() if version != expected}
    width = max(len(where) for where in found)
    for where, version in found.items():
        mark = "ok " if where not in mismatched else "BAD"
        print(f"{mark} {where:<{width}}  {version}")
    if mismatched:
        target = f"tag {args.tag}" if args.tag else "pyproject.toml"
        print(f"Release metadata disagrees with {target} ({expected}); set every location above to the same version.")
        return 1
    print(f"Release metadata agrees on {expected}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
