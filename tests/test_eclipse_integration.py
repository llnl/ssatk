from __future__ import annotations

import numpy as np


def _separation_arcsec(a, b) -> float:
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))) * 3600.0)


def test_equal_disc_overlap_matches_closed_form():
    """R1: equal unit discs one radius apart have the analytic lens area."""
    from ssatk.compute.eclipse_core import circle_overlap_visible_fraction

    overlap = 2.0 * np.pi / 3.0 - np.sqrt(3.0) / 2.0
    expected_visible = 1.0 - overlap / np.pi
    actual = circle_overlap_visible_fraction(1.0, 1.0, 1.0)
    np.testing.assert_allclose(actual, expected_visible, atol=2e-15, rtol=0)


def test_ray_intersections_match_quadratic_closed_forms():
    """R1: axial sphere and ellipsoid intersections are exact quadratic roots."""
    from ssatk.compute.eclipse_core import ray_ellipsoid_intersections, ray_sphere_intersections

    np.testing.assert_allclose(
        ray_sphere_intersections([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0),
        [2.0, 4.0], atol=1e-14, rtol=0,
    )
    np.testing.assert_allclose(
        ray_ellipsoid_intersections(
            [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]
        ),
        [1.0, 5.0], atol=1e-14, rtol=0,
    )


def test_shadow_cone_matches_similar_triangles():
    """R1: the umbra reaches zero at the similar-triangle apex distance."""
    from ssatk.compute.eclipse_core import shadow_cone, shadow_cross_section

    cone = shadow_cone(10.0, 1.0, 100.0)
    expected_apex = 100.0 / (10.0 - 1.0)
    np.testing.assert_allclose(cone.umbra_apex_distance, expected_apex, atol=2e-15, rtol=0)
    section = shadow_cross_section(
        cone, [0.0, cone.umbra_apex_distance, 2.0 * cone.umbra_apex_distance]
    )
    np.testing.assert_allclose(section.umbra_radius, [1.0, 0.0, 1.0], atol=1e-14, rtol=0)
    assert section.regime.tolist() == ["umbra", "umbra", "antumbra"]


def test_solar_reference_gcrf_matches_astropy_itrs_transform():
    """R2: reference Sun/Moon GCRF vectors agree with Astropy within 1 arcsec."""
    import astropy.units as u
    from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
    from astropy.time import Time

    from ssatk.compute.eclipse_reference_events import (
        SOLAR_2024,
        build_reference_event,
        solar_besselian_state,
    )

    jd = SOLAR_2024.greatest_jd
    time = Time(jd, format="jd", scale="utc")
    state = solar_besselian_state(jd)
    event = build_reference_event("solar", jd=[jd])
    for itrf, actual in (
        (state.sun_itrf_km, event.sun_km[0]),
        (state.moon_itrf_km, event.moon_km[0]),
    ):
        representation = CartesianRepresentation(*(itrf * u.km))
        expected = ITRS(representation, obstime=time).transform_to(
            GCRS(obstime=time)
        ).cartesian.xyz.to_value(u.km)
        assert _separation_arcsec(actual, expected) < 1.0
    assert event.frame.startswith("GCRF")


def test_lunar_reference_gcrf_matches_astropy_tete_transform():
    """R2/R3: NASA apparent RA/Dec transformed by Astropy agrees within 0.2 arcsec."""
    import astropy.units as u
    from astropy.coordinates import GCRS, SkyCoord, TETE
    from astropy.time import Time

    import ssatk.compute.eclipse_reference_events as reference

    jd = reference.LUNAR_2025.greatest_jd
    time = Time(jd, format="jd", scale="utc")
    state = reference.lunar_reference_state(jd)
    for ra, dec, actual in (
        (reference._LUNAR_SUN_RA_DEG, reference._LUNAR_SUN_DEC_DEG, state.sun_hat_gcrf),
        (reference._LUNAR_MOON_RA_DEG, reference._LUNAR_MOON_DEC_DEG, state.moon_hat_gcrf),
    ):
        apparent = SkyCoord(TETE(ra=ra * u.deg, dec=dec * u.deg, obstime=time))
        expected = apparent.transform_to(GCRS(obstime=time)).cartesian.xyz.value
        assert _separation_arcsec(actual, expected) < 0.2


def test_solar_greatest_point_matches_nasa_gsfc():
    """R3: NASA/GSFC gives 25 deg 17.2 min N, 104 deg 08.3 min W at greatest."""
    from ssatk.compute.eclipse_reference_events import (
        SOLAR_2024,
        solar_central_line_wgs84,
    )

    latitude, longitude, _ = solar_central_line_wgs84(SOLAR_2024.greatest_jd)
    np.testing.assert_allclose(latitude, 25.0 + 17.2 / 60.0, atol=0.001, rtol=0)
    np.testing.assert_allclose(longitude, -(104.0 + 8.3 / 60.0), atol=0.001, rtol=0)


def test_quadratic_photometry_matches_independent_radial_integral():
    """R1/R2: scipy adaptive radial integration; 2e-7 fractional-flux budget."""
    from scipy.integrate import quad
    from ssatk.compute.eclipse_photometry import limb_darkened_visibility_fraction

    def intensity(r):
        mu = np.sqrt(1.0 - r*r)
        return 1.0 - 0.47*(1.0-mu) - 0.28*(1.0-mu)**2

    total = 1.0/2.0 - 0.47/6.0 - 0.28/12.0
    for radius, separation in [(0.4, 0.0), (0.8, 0.5), (1.2, 0.7)]:
        def blocked(r):
            if separation == 0.0:
                fraction = float(r <= radius)
            else:
                cosine = (r*r + separation**2 - radius**2)/(2*r*separation)
                fraction = np.arccos(np.clip(cosine, -1.0, 1.0))/np.pi
            return r*intensity(r)*fraction

        edges = sorted({0.0, 1.0, *[x for x in
            (abs(separation-radius), separation+radius) if 0.0 < x < 1.0]})
        coarse = sum(quad(blocked, a, b, epsabs=1e-9)[0] for a, b in zip(edges, edges[1:]))
        fine = sum(quad(blocked, a, b, epsabs=1e-12, epsrel=1e-12)[0] for a, b in zip(edges, edges[1:]))
        np.testing.assert_allclose(coarse, fine, atol=1e-9, rtol=0)
        expected = 1.0 - fine/total
        actual = limb_darkened_visibility_fraction(
            radius, 1.0, separation, law="quadratic-visible", quadrature_order=256
        )
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


def test_generated_lut_matches_centered_quadratic_closed_form():
    """R1: centered circular occultation; 0.001 flux covers 8-bit interpolation."""
    from ssatk.compute.eclipse_photometry_lut import visibility_from_lut
    from ssatk.plots.eclipse_cinematic_light_3d import _generated_photometry_lut

    encoded, metadata = _generated_photometry_lut()
    table = np.frombuffer(encoded, dtype=np.uint8).reshape(metadata["height"], metadata["width"])
    q = np.array([0.2, 0.4, 0.6, 0.8])
    mu = np.sqrt(1.0-q*q)
    # Integral of r * ((1-u1-u2) + (u1+2u2)mu - u2*mu**2).
    blocked = (1.0-0.47-0.28)*q*q/2 + (0.47+2*0.28)*(1-mu**3)/3 - 0.28*(q*q/2-q**4/4)
    total = 0.5-0.47/6-0.28/12
    np.testing.assert_allclose(
        visibility_from_lut(table, metadata, q, np.zeros_like(q)),
        1-blocked/total, atol=0.001, rtol=0,
    )


def test_ssapy_ephemeris_adapter_matches_native_body_positions():
    """R2: installed SSAPy Body positions in SI; adapter agrees within 1e-8 km."""
    from astropy.time import Time
    from ssapy import get_body
    from ssatk.compute.eclipse_runtime import ssapy_ephemeris_positions_km

    jd = np.array([2460409.25, 2460748.75])
    gps = Time(jd, format="jd", scale="utc").gps
    moon, sun = ssapy_ephemeris_positions_km(jd)
    for name, actual in (("moon", moon), ("sun", sun)):
        body = get_body(name)
        expected = np.array([np.ravel(body.position(float(t))) for t in gps])/1000.0
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
