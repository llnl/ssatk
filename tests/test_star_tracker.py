"""R1 angular-navigation geometry, plus explicit input/source contracts."""

import numpy as np
import pytest
from astropy.time import Time

from ssatk.navigation.star_tracker import (
    StarCatalogue, catalogue_directions, fit_star_attitude,
    match_star_directions, position_from_landmark_bearings,
)


def test_shuffled_stars_recover_a_known_quarter_turn_and_reject_an_outlier():
    # R1: exact +90-degree camera-to-inertial rotation about z. No rotation
    # helper or solver generates this reference; the extra ray is unmatched.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    body = np.array([[1., 0., 3.], [0., 1., 2.], [-1., -1., 4.], [1., -2., 3.]])
    body /= np.linalg.norm(body, axis=1)[:, None]
    reference = np.column_stack([-body[:, 1], body[:, 0], body[:, 2]])
    observations = np.vstack([body[[2, 0, 3]], [0., 0., -1.]])
    measured, catalog = match_star_directions(observations, reference, rotation, .01)
    np.testing.assert_array_equal(measured, [0, 1, 2])
    np.testing.assert_array_equal(catalog, [2, 0, 3])
    actual = fit_star_attitude(observations[measured], reference[catalog])
    np.testing.assert_allclose(actual, rotation, rtol=0, atol=2e-15)
    assert np.linalg.det(actual) == pytest.approx(1., abs=2e-15)


def test_landmark_position_has_an_exact_orthogonal_bearing_reference():
    # R1: rays from (100,-200,300) to points 1, 2, and 3 km along its positive
    # coordinate axes intersect exactly at that position (metre inputs).
    landmarks = np.array([[1100., -200., 300.], [100., 1800., 300.], [100., -200., 3300.]])
    np.testing.assert_allclose(position_from_landmark_bearings(landmarks, np.eye(3)),
                               [100., -200., 300.], rtol=0, atol=1e-12)
    # R4: translating all finite landmarks translates the fix by the same amount.
    offset = np.array([7e6, -3e6, 1e6])
    np.testing.assert_allclose(position_from_landmark_bearings(landmarks+offset, np.eye(3)),
                               offset+[100., -200., 300.], rtol=0, atol=1e-8)


def test_parallel_bearings_cannot_claim_attitude_or_position():
    # R1: parallel stars leave rotation about that ray free; parallel finite
    # bearings leave translation along it free. Both solvers must reject them.
    parallel = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    with pytest.raises(ValueError, match='parallel'):
        fit_star_attitude(parallel, parallel)
    with pytest.raises(ValueError, match='constrain position'):
        position_from_landmark_bearings(np.eye(3)*1e6, parallel)
    with pytest.raises(ValueError, match='nonzero'):
        fit_star_attitude(np.zeros((3, 3)), parallel)


def test_proper_motion_matches_analytic_tangent_motion_at_the_pole():
    # R1: one arcsecond/year in the local RA tangent at RA=0 moves toward +y,
    # including at the north pole. With zero radial motion, a straight-line
    # displacement normalized to the sphere is the distant-star solution.
    cat = StarCatalogue(np.arange(3), np.array(['equator', 'pole', 'west']),
        np.array([0., 0., 12.]), np.array([0., 90., 0.]),
        np.ones(3)*1000, np.zeros(3), np.ones(3), 'analytic fixture', '')
    expected = np.array([[1., 0., 0.], [0., 0., 1.], [-1., 0., 0.]])
    tangent = np.array([[0., 1., 0.], [0., 1., 0.], [0., -1., 0.]])
    expected += 25*np.pi/(180*3600)*tangent
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    actual = catalogue_directions(cat, Time(2025., format='jyear', scale='tt'))
    # 2 milliarcsec allowance covers ERFA's safe-distance/light-time handling.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


def test_moon_sky_passes_its_epoch_to_proper_motion(monkeypatch):
    # C/R1: date must reach the star model before applying an independent
    # quarter-turn body rotation. Formerly it was always omitted (J2000 sky).
    from ssatk.plots import starfield
    import ssatk.coordinates.lunar as lunar
    calls = []
    def directions(**kwargs):
        calls.append(kwargs.get('when'))
        return np.array([[1., 0., 0.]]), np.array([1.]), np.ones((1, 3))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    monkeypatch.setattr(starfield, 'star_directions', directions)
    monkeypatch.setattr(lunar, 'lunar_map_rotation', lambda t: (rotation, np.zeros((3, 3))))
    epoch = Time('2025-06-01', scale='utc')
    result = starfield.moon_fixed_webgl_stars(epoch=epoch, radius_km=4e6)
    assert calls[0] is not None and calls[0].gps == pytest.approx(epoch.gps, abs=1e-6)
    np.testing.assert_array_equal(result['p'], [0., 4e6, 0.])
