from __future__ import annotations

import numpy as np
import pytest


def test_finite_source_geometry_and_ray_intersections():
    """R1: closed-form disc overlap, ray roots and similar-triangle shadow cones (abs 1e-14)."""
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
    # A point source is hidden strictly inside the occluder and visible outside it.
    np.testing.assert_allclose(circle_overlap_visible_fraction(1.0, 0.0, [0.5, 1.5]), [0.0, 1.0])
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


def _arcsec(a, b) -> float:
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0)


def test_ssapy_sun_is_light_time_corrected_against_astropy():
    """R2: SSAPy eclipse Sun direction against Astropy's apparent geocentric Sun.

    Tolerance 0.5 arcsec. The geometric Sun differs from the apparent Sun by
    about 20 arcsec of light-time aberration.
    """
    from astropy.coordinates import get_body
    from astropy.time import Time

    from ssatk.compute.eclipse_reference_events import SOLAR_2024
    from ssatk.compute.eclipse_runtime import ssapy_ephemeris_positions_km

    jd = SOLAR_2024.greatest_jd
    _, sun = ssapy_ephemeris_positions_km([jd])
    astropy_sun = get_body("sun", Time(jd, format="jd", scale="utc")).cartesian.xyz.to_value("km")
    assert _arcsec(sun[0], astropy_sun) < 0.5


@pytest.mark.parametrize("kind", ["solar", "lunar"])
def test_reference_backend_states_are_gcrf(kind):
    """R2: reference-backend Sun direction against Astropy GCRS at greatest eclipse.

    Tolerance 5 arcsec, set by the Besselian polynomial and the published
    apparent coordinates. True-of-date coordinates sit about 1,200 arcsec from
    GCRF in 2024-2025.
    """
    from astropy.coordinates import get_body
    from astropy.time import Time

    from ssatk.compute.eclipse_state import build_event, event_positions_gcrf

    event = build_event(kind, backend="reference", n_frames=25)
    sun, _ = event_positions_gcrf(event, event.definition.greatest_jd)
    astropy_sun = get_body("sun", Time(event.definition.greatest_jd, format="jd", scale="utc"))
    assert _arcsec(sun, astropy_sun.cartesian.xyz.to_value("km")) < 5.0


def test_ssapy_discovery_matches_nasa_lunar_2025_contacts():
    """R3: NASA/GSFC contacts and umbral magnitude of the 2025-03-14 total lunar eclipse.

    Tolerance 15 s per contact and 0.003 in magnitude: the discovery uses
    Danjon's spherical shadow enlargement rather than NASA's full shadow model.
    """
    from ssatk.compute.eclipse_event_search import discover_eclipses
    from ssatk.compute.eclipse_reference_events import LUNAR_2025

    (record,) = discover_eclipses("2025-03-13", "2025-03-16", kind="lunar")
    assert record.eclipse_type == "total"
    for name, published in LUNAR_2025.contacts_jd.items():
        assert abs(record.contacts_jd_utc[name] - published) * 86400.0 < 15.0, name
    assert record.magnitude == pytest.approx(1.1784, abs=0.003)


def test_ssapy_discovery_matches_nasa_solar_2024_contacts():
    """R3: NASA/GSFC global contacts of the 2024-04-08 total solar eclipse.

    Tolerance 25 s: the global contact equation treats Earth as a sphere of
    equatorial radius, so contacts on the polar limb are predicted late.
    """
    from ssatk.compute.eclipse_event_search import discover_eclipses
    from ssatk.compute.eclipse_reference_events import SOLAR_2024

    (record,) = discover_eclipses("2024-04-07", "2024-04-10", kind="solar")
    assert record.eclipse_type == "total"
    for name in ("P1", "U1", "MAX", "U4", "P4"):
        assert abs(record.contacts_jd_utc[name] - SOLAR_2024.contacts_jd[name]) * 86400.0 < 25.0, name
