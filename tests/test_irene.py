"""IRENE dose arithmetic against exact synthetic values, not radiation predictions."""

import json

import numpy as np

from ssatk.io import irene


HEADER = """Synthetic {model} fixture, cumulative dose (rad)
Irene Software Version: 1.58.001
Model type: {model}
Particle species: H+
Flux type: 1PtDiff
Flux direction: Omnidirectional
Accumulation mode: Cumulative
Dose calculation method: ShieldDose2 Model
Shielding geometry: Spherical4pi
Shielding depths (mm): 1 3 10
Detector type: Tissue
Nuclear atten. mode: NuclearAndNeutrons
Time format: Modified Julian Date
Coordinate system: GEO (Geographic) Cartesian in km"""


def output_fixture(tmp_path):
    orbit = np.array([[62000, 7000, 0, 0], [62000.5, 8000, 0, 0], [62001, 9000, 0, 0]])
    np.savetxt(tmp_path / "orbit.csv", orbit, delimiter=",")
    (tmp_path / "run.json").write_text(json.dumps({"depths_mm": [1, 3, 10]}))
    for model, scale in (("AE9", 1), ("AP9", 2)):
        dose = np.array([[0, 0, 0], [1, 0.5, 0.1], [2, 1, 0.2]]) * scale
        np.savetxt(tmp_path / f"{model}_mean_doseaccum.txt", np.column_stack([orbit, dose]),
                   delimiter=",", header=HEADER.format(model=model).replace("H+", "e-" if model == "AE9" else "H+"))
    return orbit


def test_summed_absorbed_dose_and_time_units(tmp_path):
    """Closed-form sum: 1 rad = 10 mGy; tolerances 1e-12 mGy and 1e-12 days."""
    output_fixture(tmp_path)
    result = irene.read_irene_dose(tmp_path)
    np.testing.assert_allclose(result["mission_day"], [0, 0.5, 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["total_mgy"], [[0, 0, 0], [30, 15, 3], [60, 30, 6]],
                               rtol=0, atol=1e-12)
    assert "mSv" not in (tmp_path / "dose_vs_mission_day.csv").read_text()
