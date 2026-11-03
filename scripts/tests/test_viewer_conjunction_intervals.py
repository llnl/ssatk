"""Between-sample circular encounters against a closed-form node crossing."""

import json
from pathlib import Path
import subprocess

import numpy as np


def test_browser_finds_circular_orbits_crossing_between_samples():
    """R1: two 7000 km circles share a node at 30 s; miss <= 1 mm."""
    source = (Path(__file__).parents[2] / "ssatk/plots/satellite_viewer_scene.js").read_text(encoding="utf-8")
    helpers = []
    for name in ("stateVectorStateAtMs", "eciAtMs", "pairSeparationKm", "refineConjunction",
                 "conjunctionIntervalLowerBound", "screenConjunctions"):
        start = source.index(f"function {name}(")
        helpers.append(source[start:source.index("\n}", start) + 2])
    rate = np.sqrt(398600.4418 / 7000.**3)
    phase = rate * (np.array([0., 60.]) - 30.)
    entries = []
    for name, inclination in (("a", 0.), ("b", np.pi / 3)):
        r = 7000 * np.column_stack((np.cos(phase), np.sin(phase)*np.cos(inclination), np.sin(phase)*np.sin(inclination)))
        v = 7000 * rate * np.column_stack((-np.sin(phase), np.cos(phase)*np.cos(inclination), np.cos(phase)*np.sin(inclination)))
        entries.append([name, {"entry": {"mode": "state", "name": name},
                              "stateTrack": {"t": [0, 60000], "r": r.tolist(), "v": v.tolist()}}])
    program = "\n".join(helpers) + "\n" + f"""
const activeSatellites = new Map({json.dumps(entries)});
let conjunctionEvents = [];
const conjunctionWindowHours = 1/60, conjunctionThresholdKm = .001, CONJ_COARSE_STEP_MS = 60000;
function getCurrentSimMs() {{ return 0; }}
screenConjunctions();
if (conjunctionEvents.length !== 1 || conjunctionEvents[0].missKm > 1e-6 ||
    Math.abs(conjunctionEvents[0].tcaMs - 30000) > 1) throw Error(JSON.stringify(conjunctionEvents));
"""
    subprocess.run(["node", "-e", program], check=True, capture_output=True, text=True)
