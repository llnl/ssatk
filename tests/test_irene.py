"""IRENE unit conversions against exact synthetic values, not radiation predictions."""

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


def test_flux_grid_si_positions_and_exact_flux(tmp_path):
    """Closed-form km-to-m conversion (1e-9 m), flux unchanged (1e-12 cm^-2 s^-1)."""
    positions = np.array([[7_000_000, 0, 0], [0, 8_000_000, 1000]])
    irene.prepare_irene_flux_grid(positions, "2025-01-01T00:00:00", tmp_path / "run")
    directory = tmp_path / "run"
    orbit = np.array([[60676, 7000, 0, 0], [60676, 0, 8000, 1]])
    np.testing.assert_allclose(np.loadtxt(directory / "orbit.csv", delimiter=","),
                               orbit, rtol=0, atol=1e-12)
    for model, species, energy, flux in (("AE9", "e-", 1, [12, 34]), ("AP9", "H+", 10, [56, 78])):
        header = (f"Irene Software Version: 1.58.001\nModel type: {model}\n"
                  f"Particle species: {species}\nEnergy levels (MeV): {energy}\n"
                  "Flux type: Integral\nFlux direction: Omnidirectional\n"
                  "Time format: Modified Julian Date\n"
                  "Coordinate system: GEO (Geographic) Cartesian in km\nflux(#/cm^2/s)")
        np.savetxt(directory / f"{model}_mean_flux.txt", np.column_stack([orbit, flux]),
                   delimiter=",", header=header)
    result = irene.read_irene_flux_grid(directory)
    np.testing.assert_allclose(result["geo_m"], positions, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["electron_flux"], [12, 34], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["proton_flux"], [56, 78], rtol=0, atol=1e-12)
