"""C: real public products using installed data, without asset-lookup mocks."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ssatk.eclipse import EventRequest, ObserverConfig, RenderRequest, build_event, render_product
from ssatk.io.eclipse_asset_resolver import resolve_asset


def test_installed_split_data_kernel():
    from ssapy.utils import find_file

    resolved = resolve_asset("de430_kernel", policy="strict-data")
    assert Path(resolved.path).samefile(find_file("de430_1900_2150.bsp"))
    assert not resolved.git_lfs_pointer


@pytest.mark.parametrize("kind", ["solar", "lunar"])
def test_public_products_with_real_assets(tmp_path, kind):
    event = build_event(EventRequest(kind=kind, backend="reference", sample_count=17))
    for product in ("cinematic", "plotly", "scientific-suite"):
        output = tmp_path / (product if product == "scientific-suite" else product + ".html")
        render_product(RenderRequest(product, output, event=event, quality="motion"))
    for name in ("cinematic.html", "plotly.html"):
        document = (tmp_path / name).read_text(encoding="utf-8")
        assert "<html" in document.lower()
        assert len(document) > 1000
    images = list((tmp_path / "scientific-suite").glob("*.png"))
    assert images
    for path in images:
        with Image.open(path) as image:
            assert image.width > 0 and image.height > 0
            assert np.ptp(np.asarray(image)) > 0


def test_local_observer_suite_and_demo_with_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("SSATK_OUTPUT_DIR", str(tmp_path))
    observer = ObserverConfig(latitude_deg=30.41028, longitude_east_deg=-97.96312)
    event = build_event(EventRequest(
        backend="reference", sample_count=17, solar_scope="local", observer=observer
    ))
    render_product(RenderRequest("observer-suite", tmp_path / "observer", event=event, quality="motion"))
    assert list((tmp_path / "observer").glob("*.csv"))
    from demos.eclipse.demo_eclipse_scientific import main

    result = main(make_figures=True, fast=True)
    assert len(result["files"]) == 2
    assert all(Path(path).is_file() for path in result["files"])


def test_promotion_cli_reports_blocked_failure(tmp_path, monkeypatch, capsys):
    import json
    from ssatk import eclipse_cli
    from ssatk.compute import eclipse_promotion
    from ssatk.compute.eclipse_runtime import BackendUnavailableError

    def blocked(*args, **kwargs):
        raise BackendUnavailableError("strict ephemeris unavailable")

    monkeypatch.setattr(eclipse_promotion, "resolve_backend", blocked)
    output = tmp_path / "promotion.json"
    assert eclipse_cli.main(["promote", "--output", str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False and result["status"] == "blocked"
    assert json.loads(output.read_text(encoding="utf-8"))["promoted"] is False


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
