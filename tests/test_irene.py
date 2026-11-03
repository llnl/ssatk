"""IRENE unit conversions against exact synthetic values, not radiation predictions."""

import json

import numpy as np
import pytest

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


@pytest.mark.parametrize("normal", [(1, 0, 0), (1, 2, 3), (-2, 1, -3)])
def test_flux_surface_mesh_linear_contours(normal):
    """Linear field contour: plane residual within 1e-14 grid units."""
    from ssatk.plots.magfield_plot_3d import _flux_surface_mesh

    coordinates = np.stack(np.meshgrid(*([np.linspace(-1, 1, 4)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    values = coordinates @ normal
    vertices, faces = _flux_surface_mesh(coordinates, values, 4, 0.17)
    assert len(faces) > 0
    np.testing.assert_allclose(vertices @ normal, 0.17, rtol=0, atol=1e-14)
    assert np.all(np.abs(vertices) <= 1)
    assert faces.min() == 0 and faces.max() == len(vertices) - 1
    for level in (values.min() - 1, values.max() + 1):
        empty_vertices, empty_faces = _flux_surface_mesh(coordinates, values, 4, level)
        assert empty_vertices.shape == empty_faces.shape == (0, 3)



def test_flux_surface_mesh_closed_sphere():
    """Sphere contour: bounded radius within 1e-14 grid units, two faces per edge."""
    from ssatk.plots.magfield_plot_3d import _flux_surface_mesh

    coordinates = np.stack(np.meshgrid(*([np.linspace(-1.5, 1.5, 9)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    vertices, faces = _flux_surface_mesh(coordinates, np.sum(coordinates ** 2, axis=1), 9, 0.83)
    assert len(faces) > 0
    assert np.all(np.linalg.norm(vertices, axis=1) <= np.sqrt(0.83) + 1e-14)
    _, inverse = np.unique(np.round(vertices, 12), axis=0, return_inverse=True)
    indexed = inverse[faces]
    edges = np.sort(np.concatenate([indexed[:, [0, 1]], indexed[:, [1, 2]], indexed[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "Contour must be closed without cracks or duplicate faces"



def test_artemis_mission_endpoint():
    """Lambert arrival is 2,000 km Earthward of the Moon, within 1 m."""
    from demos.space_environment import demo_artemis_radiation as demo
    times, positions, moon = demo.mission()
    assert (times[-1] - times[0]).to_value("day") == pytest.approx(4)
    assert np.linalg.norm(positions[-1] - moon[-1]) == pytest.approx(2e6, abs=1)
    assert np.all(np.diff(times.gps) > 0)
    assert np.max(np.diff(times.gps)[:719]) <= 30
