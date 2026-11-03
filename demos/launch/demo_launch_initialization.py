"""Verify pad-fixed launch initialization, not ascent or mission qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astropy import units as u
from astropy.time import Time

from ssatk.launch.preflight import (
    EOPCoverageError,
    collect_software_baseline,
    launch_initial_state,
    verify_data_resources,
)

GALLERY_INCLUDE = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="Kennedy Space Center LC-39A")
    parser.add_argument("--epoch", default="2020-05-30T19:22:00")
    parser.add_argument("--heading", required=True,
                        help="Illustrative body +X heading clockwise from north, degrees.")
    parser.add_argument("--height", default="0.0",
                        help="WGS84 ellipsoidal height, m; not terrain height.")
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwritten.")
    arguments = parser.parse_args(argv)
    if arguments.output is not None and arguments.output.suffix.lower() != ".json":
        parser.error("--output must name a JSON file")
    report = {
        "scope": "STEP 3 INITIALIZATION ONLY",
        "qualification_status": "NOT RUN",
        "software": collect_software_baseline(),
        "status": "NOT RUN",
        "inputs": {"site": arguments.site, "epoch_utc": arguments.epoch,
                   "heading_deg": arguments.heading, "height_m": arguments.height},
    }
    try:
        report["resources"] = verify_data_resources(
            [{"path": "environment/eop/finals2000A.all"}], strict=False,
        )
        statuses = {resource["status"] for resource in report["resources"]}
        if statuses != {"PASS"}:
            report["status"] = "FAIL" if "FAIL" in statuses else "BLOCKED"
        else:
            epoch = Time(arguments.epoch, scale="utc")
            report["samples"] = [
                launch_initial_state(arguments.site, epoch + offset * u.s,
                                     heading_deg=float(arguments.heading), height_m=float(arguments.height))
                for offset in (-1.0, 0.0, 1.0)
            ]
            report["status"] = "PASS"
    except (EOPCoverageError, FileNotFoundError) as exc:
        report.update(status="BLOCKED", detail=str(exc))
    except (ValueError, KeyError, OSError) as exc:
        report.update(status="ERROR", detail=str(exc))
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return report


if __name__ == "__main__":
    sys.exit(0 if main()["status"] == "PASS" else 2)
