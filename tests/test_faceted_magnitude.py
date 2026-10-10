from types import SimpleNamespace

import numpy as np
import pytest
from astropy.time import Time

from ssatk.compute.faceted_magnitude import (
    facet_scattering_area,
    faceted_reflection,
    line_of_sight_blocked,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
EPOCH = Time("2026-06-09T06:00:00")
GEO_RADIUS = 4.2164e7


def _facet(normal=(1.0, 0.0, 0.0), area=4.0, reflectivity=0.5):
    return SimpleNamespace(normal_body=normal, area=area, diffuse_reflectivity=reflectivity)


def test_head_on_facet_matches_closed_form():
    area = facet_scattering_area([_facet()], IDENTITY, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert area == pytest.approx(0.5 * 4.0 / np.pi)


def test_cosine_factors_apply_to_both_paths():
    source = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0])
    area = facet_scattering_area([_facet()], IDENTITY, source, (1.0, 0.0, 0.0))
    assert area == pytest.approx(0.5 * 4.0 * np.cos(np.pi / 3) / np.pi)


@pytest.mark.parametrize(
    "source,observer",
    [((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))],
)
def test_unlit_or_hidden_facets_contribute_nothing(source, observer):
    assert facet_scattering_area([_facet()], IDENTITY, source, observer) == 0.0


def test_attitude_rotates_the_facet_out_of_view():
    half = 0.5 * np.pi / 2.0
    quaternion = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # +90 deg about Z
    assert facet_scattering_area(
        [_facet()], quaternion, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    ) == pytest.approx(0.0, abs=1e-12)


def test_earth_occults_opposite_sides_but_not_the_same_side():
    near = np.array([GEO_RADIUS, 0.0, 0.0])
    far = np.array([-GEO_RADIUS, 0.0, 0.0])
    assert line_of_sight_blocked(near, far)
    assert not line_of_sight_blocked(near, np.array([GEO_RADIUS, 1.0e7, 0.0]))


def test_doubling_facet_area_brightens_by_the_expected_amount():
    facets_small = [_facet(normal=direction) for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    facets_large = [
        _facet(normal=direction, area=8.0) for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    position = np.array([GEO_RADIUS, 0.0, 0.0])
    observer = np.array([1.5 * GEO_RADIUS, 1.0e7, 1.0e7])
    common = {"observer": observer, "time": EPOCH, "band": "V"}
    small = faceted_reflection(position, IDENTITY, facets_small, **common)
    large = faceted_reflection(position, IDENTITY, facets_large, **common)
    difference = small["ab_mag_exoatmospheric"] - large["ab_mag_exoatmospheric"]
    assert difference == pytest.approx(2.5 * np.log10(2.0), rel=1e-9)
