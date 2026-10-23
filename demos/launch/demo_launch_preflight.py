"""Report the unresolved inputs in the LEO launch preflight template."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssatk.launch.preflight import run_launch_preflight

GALLERY_INCLUDE = False


def main():
    template = Path(__file__).with_name("leo_preflight_template.json")
    contract = json.loads(template.read_text(encoding="utf-8"))
    report = run_launch_preflight(contract)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
