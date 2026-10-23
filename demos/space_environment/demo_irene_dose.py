"""Illustrative AE9/AP9 absorbed dose for one GEO orbit, excluding SEP/GCR.

Run: python -m demos.space_environment.demo_irene_dose
Requires the separately installed official IRENE runtime.
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import ssatk
from ssatk._paths import output_root
from ssatk.io.irene import find_irene_runtime, prepare_irene_run, run_irene
from ssatk.plots import figsave
from ssatk.time_functions import get_times


def main(make_figures=True, fast=False):
    if find_irene_runtime() is None:
        reason = "IRENE unavailable; configure IRENE_HOME for official absorbed dose"
        print(reason)
        return {"skipped": True, "reason": reason}

    times = get_times((1, "day"), freq=(1, "hour"), t0="2025-07-02T00:00:00")
    orbit = ssatk.Orbit.fromKeplerianElements(
        a=ssatk.constants.RGEO, e=0, i=0, pa=0, raan=0, trueAnomaly=0, t=times[0].gps)
    positions, _ = ssatk.rv(orbit, times)
    base = output_root() / "data" / "irene_dose"
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="geo_", dir=base)) / "run"
    prepare_irene_run(positions, times, directory)
    result = run_irene(directory)
    print(f"IRENE input, logs and absorbed dose: {directory}")
    if make_figures:
        # gallery-call: demo_irene_dose.png
        fig, ax = plt.subplots()
        for depth, dose in zip(result["depths_mm"], np.asarray(result["total_mgy"]).T):
            ax.plot(result["mission_day"], dose, label=f"{depth} mm aluminum")
        ax.set(xlabel="Mission day", ylabel="Absorbed dose (mGy)",
               title="IRENE AE9/AP9 mean; 4-pi sphere, tissue; SEP/GCR excluded")
        ax.legend()
        figsave(fig, "demo_irene_dose.png")
        # gallery-end
        plt.close(fig)
    return {"skipped": False, "directory": str(directory), "dose": result}


if __name__ == "__main__":
    main()
