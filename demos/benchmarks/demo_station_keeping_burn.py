"""Reproducible orbit station-keeping benchmark with finite perturbation and burns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssatk.accelerations_6dof import SpacecraftAccelConstNTW
from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.misc import orbital_elements_from_state
from ssatk.propagators_6dof import ImpulseManeuver, Spacecraft, propagate_spacecraft_segments

GALLERY_INCLUDE = False
OUTPUT_DIR = Path.home() / "ssatk_output" / "data" / "benchmarks"


def _propagate(spacecraft, times, drag, *, controlled):
    segments = []
    for index, (start, stop) in enumerate(zip(times[:-1], times[1:])):
        segment = {"times": [start, stop], "models": [drag], "mu": EARTH_MU}
        if controlled and index:
            segment["impulses"] = ImpulseManeuver(
                [0.0, 1.0e-3, 0.0], frame="ntw"
            )
        segments.append(segment)
    return propagate_spacecraft_segments(spacecraft, segments)


def run(*, output_dir: Path = OUTPUT_DIR) -> dict:
    """Compare uncontrolled drag decay with exact-epoch NTW station-keeping burns."""
    radius = 7_000_000.0
    initial_r = np.array([radius, 0.0, 0.0])
    initial_v = np.array([0.0, np.sqrt(EARTH_MU / radius), 0.0])
    spacecraft = Spacecraft(
        r=initial_r, v=initial_v, inertia=np.eye(3), mass=100.0
    )
    times = np.arange(0.0, 3600.0 + 100.0, 100.0)
    drag = SpacecraftAccelConstNTW([0.0, -1.0e-5, 0.0])
    uncontrolled = _propagate(spacecraft, times, drag, controlled=False)
    controlled = _propagate(spacecraft, times, drag, controlled=True)
    a0 = orbital_elements_from_state(initial_r, initial_v, EARTH_MU)[0]
    a_uncontrolled = orbital_elements_from_state(
        uncontrolled.r[-1], uncontrolled.v[-1], EARTH_MU
    )[0]
    a_controlled = orbital_elements_from_state(
        controlled.r[-1], controlled.v[-1], EARTH_MU
    )[0]
    result = {
        "benchmark": "constant NTW drag with periodic NTW station-keeping burns",
        "duration_s": float(times[-1]),
        "burn_count": int(len(times) - 2),
        "burn_dv_m_s": 1.0e-3,
        "uncontrolled_semimajor_axis_change_m": float(a_uncontrolled - a0),
        "controlled_semimajor_axis_change_m": float(a_controlled - a0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "station_keeping_burn_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
