from __future__ import annotations

import numpy as np


def test_finite_source_geometry_and_ray_intersections():
    from ssatk.compute.eclipse_core import (
        circle_overlap_visible_fraction,
        finite_source_visibility,
        ray_ellipsoid_intersections,
        ray_sphere_intersections,
        shadow_cone,
        shadow_cross_section,
    )

    np.testing.assert_allclose(
        circle_overlap_visible_fraction([1.0, 0.5, 1.0], 1.0, [3.0, 0.0, 0.0]),
        [1.0, 0.75, 0.0],
    )
    np.testing.assert_allclose(finite_source_visibility([-10.0, 0.0, 0.0], [100.0, 0.0, 0.0], 1.0, 0.5), 0.0)
    np.testing.assert_allclose(ray_sphere_intersections([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0), [2.0, 4.0])
    np.testing.assert_allclose(
        ray_ellipsoid_intersections([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
        [1.0, 5.0],
    )

    cone = shadow_cone(10.0, 1.0, 100.0)
    section = shadow_cross_section(cone, [0.0, cone.umbra_apex_distance, 2.0 * cone.umbra_apex_distance])
    np.testing.assert_allclose(section.umbra_radius, [1.0, 0.0, 1.0], atol=1e-14)
    assert section.regime.tolist() == ["umbra", "umbra", "antumbra"]


def test_reference_eclipses_pass_physical_validation():
    from ssatk.eclipse import EventRequest, build_event, validate_event

    for kind, expected_key in (("solar", "solar_2024_04_08"), ("lunar", "lunar_2025_03_14")):
        event = build_event(EventRequest(kind=kind, backend="reference", sample_count=25))
        validation = validate_event(event)

        assert event.definition.key == expected_key
        assert validation.passed, validation
        assert np.all(np.diff(event.jd) > 0.0)
        assert np.all((event.center_visibility >= 0.0) & (event.center_visibility <= 1.0))
        assert validation.metrics["ray_penetrations"] == {"earth": 0, "moon": 0}


def test_ssapy_backend_is_strict_and_passes_physical_validation():
    from ssatk.eclipse import EventRequest, build_event, validate_event

    for kind in ("solar", "lunar"):
        event = build_event(EventRequest(kind=kind, backend="ssapy", sample_count=25))
        validation = validate_event(event)

        assert event.metadata["state_source"] == "ssapy"
        assert event.metadata["ephemeris_backend"] == "llnl-ssapy/DE430"
        assert event.metadata["frame_backend"] == "ssapy-toolkit GCRF<->ITRF"
        assert event.metadata["moon_orientation_backend"] == "llnl-ssapy DE440 binary lunar PCK"
        assert event.metadata["strict_backend"] is True
        assert event.metadata["backend_fallback_reason"] == ""
        assert validation.passed, validation
