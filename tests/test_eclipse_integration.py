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


def test_de430_sun_is_light_time_corrected_against_astropy():
    """R2: DE430 eclipse Sun direction against Astropy's apparent geocentric Sun.

    Tolerance 0.5 arcsec. The geometric Sun differs from the apparent Sun by
    about 20 arcsec of light-time aberration.
    """
    from astropy.coordinates import get_body
    from astropy.time import Time

    from ssatk.compute.eclipse_reference_events import SOLAR_2024
    from ssatk.compute.eclipse_runtime import ephemeris_positions_km

    jd = SOLAR_2024.greatest_jd
    _, sun = ephemeris_positions_km([jd])
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


def test_de430_discovery_matches_nasa_lunar_2025_contacts():
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


def test_de430_discovery_matches_nasa_solar_2024_contacts():
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


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2024-04-07", "2024-04-10", 1.0566),
        ("2023-10-12", "2023-10-16", 0.9520),
        ("2025-03-27", "2025-03-31", 0.9376),
        ("2025-09-19", "2025-09-23", 0.8550),
        ("2022-10-23", "2022-10-27", 0.8619),
    ],
)
def test_de430_solar_magnitudes_match_nasa_published_values(start, end, expected):
    """R3: topocentric central and WGS-84 projected partial magnitudes.

    NASA/GSFC values are checked to 0.005. Central events use the WGS-84
    greatest-eclipse surface point; partial events use the minimum-separation
    WGS-84 surface view and projected obscured diameter.
    """
    from ssatk.compute.eclipse_event_search import discover_eclipses

    (record,) = discover_eclipses(start, end, kind="solar")
    assert record.magnitude == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize(
    ("start", "end"),
    [("2023-04-18", "2023-04-22"), ("2013-11-01", "2013-11-05")],
)
def test_de430_known_hybrid_solar_eclipses_are_classified_hybrid(start, end):
    """R3: topocentric radius sign changes across the central contact span."""
    from ssatk.compute.eclipse_event_search import discover_eclipses

    (record,) = discover_eclipses(start, end, kind="solar")
    assert record.eclipse_type == "hybrid"


def test_equal_disc_overlap_matches_closed_form():
    """R1: equal unit discs one radius apart have the analytic lens area (abs 2e-15)."""
    from ssatk.compute.eclipse_core import circle_overlap_visible_fraction

    overlap = 2.0 * np.pi / 3.0 - np.sqrt(3.0) / 2.0
    np.testing.assert_allclose(
        circle_overlap_visible_fraction(1.0, 1.0, 1.0), 1.0 - overlap / np.pi, atol=2e-15, rtol=0,
    )


def test_quadratic_photometry_matches_independent_radial_integral():
    """R1/R2: SciPy adaptive radial integration split at the occultation edges (abs 2e-7 flux)."""
    from scipy.integrate import quad

    from ssatk.compute.eclipse_core import limb_darkened_visibility_fraction

    def intensity(r):
        mu = np.sqrt(1.0 - r * r)
        return 1.0 - 0.47 * (1.0 - mu) - 0.28 * (1.0 - mu) ** 2

    total = 1.0 / 2.0 - 0.47 / 6.0 - 0.28 / 12.0
    for radius, separation in [(0.4, 0.0), (0.8, 0.5), (1.2, 0.7)]:
        def blocked(r):
            if separation == 0.0:
                fraction = float(r <= radius)
            else:
                cosine = (r * r + separation**2 - radius**2) / (2 * r * separation)
                fraction = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
            return r * intensity(r) * fraction

        edges = sorted({0.0, 1.0, *[x for x in (abs(separation - radius), separation + radius) if 0.0 < x < 1.0]})
        integral = sum(quad(blocked, a, b, epsabs=1e-12, epsrel=1e-12)[0] for a, b in zip(edges, edges[1:]))
        actual = limb_darkened_visibility_fraction(
            radius, 1.0, separation, law="quadratic-visible", quadrature_order=256,
        )
        np.testing.assert_allclose(actual, 1.0 - integral / total, atol=2e-7, rtol=0)


def test_photometry_lut_matches_centered_quadratic_closed_form():
    """R1: centred circular occultation of a quadratic limb-darkened disc.

    Tolerance 0.001 in visible flux covers the 8-bit encoding of the
    limb-darkening correction.
    """
    from ssatk.compute.eclipse_photometry_lut import build_delta_lut

    encoded, meta = build_delta_lut()
    rows = np.linspace(0.0, meta["q_max"], meta["height"])
    index = np.searchsorted(rows, [0.2, 0.4, 0.6, 0.8])
    q = rows[index]
    delta = encoded[index, 0] / 255.0 * (meta["delta_max"] - meta["delta_min"]) + meta["delta_min"]
    mu = np.sqrt(1.0 - q * q)
    # Integral of r * ((1-u1-u2) + (u1+2u2)mu - u2 mu^2) over the blocked disc.
    blocked = (1.0 - 0.47 - 0.28) * q * q / 2 + (0.47 + 2 * 0.28) * (1 - mu**3) / 3 - 0.28 * (q * q / 2 - q**4 / 4)
    total = 0.5 - 0.47 / 6 - 0.28 / 12
    np.testing.assert_allclose((1.0 - q * q) + delta, 1.0 - blocked / total, atol=0.001, rtol=0)


@pytest.mark.parametrize("backend, tolerance_deg", [("reference", 0.001), ("de430", 0.03)])
def test_greatest_eclipse_ground_point_matches_nasa_gsfc(backend, tolerance_deg):
    """R3: NASA/GSFC places 2024-04-08 greatest eclipse at 25 deg 17.2 min N, 104 deg 08.3 min W.

    The reference backend replays NASA's Besselian elements (0.001 deg). The
    DE430 backend uses JPL DE430 and SSATK Earth orientation rather than NASA's
    VSOP87/ELP2000-85 with their Delta T, so it is held to 0.03 deg (about 3 km).
    """
    from ssatk.compute.eclipse_local_finite_geometry import shadow_axis_world
    from ssatk.compute.eclipse_raytrace import trace_event_rays
    from ssatk.compute.eclipse_reference_events import geodetic_from_itrf
    from ssatk.compute.eclipse_runtime import gcrf_to_itrf_km
    from ssatk.compute.eclipse_state import build_event

    event = build_event("solar", backend=backend, n_frames=25)
    jd = event.definition.greatest_jd
    bundle = trace_event_rays(event, jd, n_azimuth=8)
    end = shadow_axis_world(bundle, jd)[-1]
    end_itrf = end if bundle.frame.startswith("WGS84") else gcrf_to_itrf_km(end, jd)
    latitude, longitude, _ = geodetic_from_itrf(end_itrf)
    assert abs(latitude - (25.0 + 17.2 / 60.0)) < tolerance_deg
    assert abs(longitude + (104.0 + 8.3 / 60.0)) < tolerance_deg
