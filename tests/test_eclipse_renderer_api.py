from __future__ import annotations

import numpy as np
import pytest


def test_shared_eclipse_renderer_apis_build_mesh_data(monkeypatch):
    from ssatk.plots.globe_orbit_daynight_plotly import (
        _earth_surface_data,
        _wgs84_vertices,
    )
    from ssatk.plots import moon_render
    from ssatk.plots.moon_render import _moon_surface_data, _moon_unit_mesh

    monkeypatch.setattr(
        moon_render,
        "_load_real_moon_texture",
        lambda n_lat, n_lon, texture_path=None: np.full((n_lat, n_lon, 3), 128, dtype=np.uint8),
    )
    monkeypatch.setattr(
        moon_render,
        "irradiance_fraction",
        lambda positions, **kwargs: np.full(len(positions), 0.75),
    )

    earth_vertices, _, earth_faces, *_ = _wgs84_vertices(5, 8)
    moon_directions, moon_faces, *_ = _moon_unit_mesh(5, 8)
    earth = _earth_surface_data([1.0, 0.0, 0.0], n_lat=5, n_lon=8)
    body_to_gcrf = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moon_center = np.array([0.0, 0.0, 384_400.0])
    moon = _moon_surface_data(
        center=[0.0, 0.0, 384_400.0],
        radius=1_737.4,
        real_center_km=moon_center,
        real_sun_position_km=[149_597_870.7, 0.0, 0.0],
        body_to_gcrf=body_to_gcrf,
        n_lat=5,
        n_lon=8,
    )

    assert earth_vertices.shape == (26, 3)
    assert earth.faces.shape == earth_faces.shape
    assert moon_directions.shape == (26, 3)
    assert moon.faces.shape == moon_faces.shape
    assert np.all(np.isfinite(earth.shaded_rgb_srgb))
    assert np.all(np.isfinite(moon.shaded_rgb_srgb))
    assert np.all(np.isfinite(moon.eclipse_visibility))
    np.testing.assert_allclose(moon.eclipse_visibility, 0.75)
    np.testing.assert_allclose(
        (moon.physical_vertices_gcrf_km - moon_center) / moon_render.R_MOON_KM,
        moon_directions @ body_to_gcrf.T,
    )

    with pytest.raises(ValueError, match="proper rotation"):
        _moon_surface_data(
            [0.0, 0.0, 0.0], 1.0,
            body_to_gcrf=np.diag([1.0, 1.0, -1.0]), n_lat=5, n_lon=8,
        )


def test_scientific_summary_uses_toolkit_figure_output(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ssatk.plots import eclipse_scientific_suite, plotutils

    event = SimpleNamespace(jd=np.array([1.0]), greatest_jd=1.0)
    renderer = SimpleNamespace(render=lambda index: np.zeros((6, 8, 3), dtype=np.uint8))
    requested = tmp_path / "eclipse" / "summary.png"
    seen = {}

    def fake_figsave(figure, output, dpi):
        seen["output"] = output
        return str(requested)

    monkeypatch.setattr(eclipse_scientific_suite, "renderer_for_event", lambda *args, **kwargs: renderer)
    monkeypatch.setattr(plotutils, "figsave", fake_figsave)

    saved = eclipse_scientific_suite.generate_event_scientific_summary(
        event, "eclipse/summary.png", width=8, height=6, dpi=100
    )

    assert saved == str(requested)
    assert seen["output"] == "eclipse/summary.png"


def test_browser_worker_reports_unavailable_chromium(monkeypatch, tmp_path):
    import json

    from ssatk.plots import eclipse_browser_matrix

    result_path = tmp_path / "result.json"
    monkeypatch.setattr(eclipse_browser_matrix.shutil, "which", lambda name: None)

    status = eclipse_browser_matrix._worker(
        tmp_path / "page.html", tmp_path, "smoke", "chromium-default", result_path
    )

    assert status == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["available"] is False
