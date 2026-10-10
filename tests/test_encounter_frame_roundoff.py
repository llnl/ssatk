"""Encounter frames stay orthonormal when the miss projection is small."""

import numpy as np
import pytest

from ssatk.ssa.conjunction import (
    encounter_frame,
)


@pytest.mark.parametrize("distance_scale", [1.0, 1.0e3, 1.0e6])
@pytest.mark.parametrize("speed_scale", [1.0, 7500.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_collinear_encounter_uses_orthonormal_fallback(distance_scale, speed_scale, sign):
    direction = np.array([1.0, 2.0, 3.0])
    velocity = speed_scale * direction
    normal = velocity / np.linalg.norm(velocity)
    basis = encounter_frame(sign * distance_scale * direction, velocity)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(basis.T @ normal, 0.0, atol=1e-14)
    expected = np.eye(3)[np.argmin(np.abs(normal))]
    expected -= normal * np.dot(normal, expected)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(basis[:, 0], expected, atol=1e-14)


def test_nearly_collinear_encounter_preserves_resolved_miss_direction():
    normal = np.array([1.0, 2.0, 2.0]) / 3.0
    transverse = np.array([2.0, -1.0, 0.0]) / np.sqrt(5.0)
    position = 1.0e6 * normal + 0.1 * transverse
    basis = encounter_frame(position, 7500.0 * normal)
    np.testing.assert_allclose(basis[:, 0], transverse, atol=5e-9)
    np.testing.assert_allclose(basis.T @ normal, 0.0, atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)


def test_well_separated_frame_agrees_with_direct_projection():
    rng = np.random.default_rng(20)
    for _ in range(100):
        position, velocity = rng.normal(size=(2, 3))
        normal = velocity / np.linalg.norm(velocity)
        expected = position - normal * np.dot(normal, position)
        expected /= np.linalg.norm(expected)
        basis = encounter_frame(position, velocity)
        np.testing.assert_allclose(basis[:, 0], expected, atol=1e-13)
        np.testing.assert_allclose(basis[:, 1], np.cross(normal, expected), atol=1e-13)
