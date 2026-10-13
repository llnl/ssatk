"""R1/R2 direction rotations in degrees; 2e-14 dimensionless component budget."""

import numpy as np
from scipy.spatial.transform import Rotation

from ssatk.vectors import rotate_vector


def test_rotate_vector_cardinal_right_hand_quarter_turns():
    """R1: literal right-hand rotations, including both formerly singular Y poles."""
    cases = [
        ([0, 1, 0], 90, 0, [1, 0, 0]),
        ([0, -1, 0], 90, 0, [1, 0, 0]),
        ([0, 1, 0], 90, 90, [0, 0, -1]),
        ([0, -1, 0], 90, 90, [0, 0, 1]),
        ([1, 0, 0], 90, 0, [0, 1, 0]),
        ([1, 0, 0], 90, 90, [0, 0, 1]),
        ([0, 0, -1], 90, 90, [1, 0, 0]),
        ([0, 1, 0], 180, 37, [0, -1, 0]),
    ]
    for vector, theta, phi, expected in cases:
        np.testing.assert_allclose(
            rotate_vector(vector, theta, phi), expected, rtol=0, atol=2e-14
        )


def test_rotate_vector_against_independent_scipy_quaternions():
    """R2: SciPy quaternion rotations about specified axes, not SSATK matrices.

    The ordinary oblique direction preserves the legacy +Y reference. The
    near-pole cases use the documented +X fallback and literal +/-Z axes.
    """
    cases = [([2, -3, 6], [-6, 0, 2])]
    cases += [([1e-14, sign, 0], [0, 0, -sign]) for sign in (-1, 1)]
    for vector, axis in cases:
        direction = np.asarray(vector) / np.linalg.norm(vector)
        axis = np.asarray(axis) / np.linalg.norm(axis)
        first = Rotation.from_rotvec(axis * np.deg2rad(37)).apply(direction)
        expected = Rotation.from_rotvec(direction * np.deg2rad(-61)).apply(first)
        np.testing.assert_allclose(
            rotate_vector(vector, 37, -61), expected, rtol=0, atol=2e-14
        )


def test_rotate_vector_direction_is_independent_of_input_magnitude():
    """R1: rotating any positive multiple of +X by 90 deg yields +Y.

    Large/small finite scales catch overflow/underflow during normalization.
    """
    for scale in (1e-200, 5.0, 1e200):
        np.testing.assert_allclose(
            rotate_vector([scale, 0, 0], 90, 0), [0, 1, 0], rtol=0, atol=2e-14
        )
