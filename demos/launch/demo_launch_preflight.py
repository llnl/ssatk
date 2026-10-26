"""Validate a launch contract and optionally retain its preflight evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssatk.launch.preflight import run_launch_preflight

GALLERY_INCLUDE = False


def main(argv=None):
    template = Path(__file__).with_name("leo_preflight_template.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=template, help="Mission JSON; default is the unconfigured LEO template.")
    parser.add_argument("--output", type=Path, help="New JSON evidence file in a data/output location; never overwritten.")
    parser.add_argument(
        "--repository", action="append", default=[], metavar="PACKAGE=PATH",
        help="Repository to verify against source_revisions; repeat for each required package.",
    )
    arguments = parser.parse_args(argv)
    if arguments.output is not None and arguments.output.suffix.lower() != ".json":
        parser.error("--output must name a JSON file")
    try:
        contract = json.loads(arguments.contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    repositories = {}
    for entry in arguments.repository:
        name, separator, path = entry.partition("=")
        if not separator or not name.strip() or not path.strip() or name in repositories:
            parser.error("--repository requires unique PACKAGE=PATH entries")
        repositories[name] = Path(path)
    report = run_launch_preflight(contract, repositories=repositories)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return report


if __name__ == "__main__":
    sys.exit(0 if main()["status"] == "PASS" else 2)
