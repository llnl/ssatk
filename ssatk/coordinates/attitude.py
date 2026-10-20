"""Quaternion and attitude-frame helpers.

Quaternion convention is ``[w, x, y, z]`` and rotates body-frame vectors into
the inertial/GCRF frame.
"""

from __future__ import annotations

import numpy as np

from .satellite_frames import frame_to_gcrf_matrix

ArrayLike = np.ndarray | list[float] | tuple[float, ...]

__all__ = [
    "attitude_quaternion_from_frame",
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_from_matrix",
    "quaternion_multiply",
    "rotate_vector",
]


def normalize_quaternion(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector [w, x, y, z].")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("q must be non-zero.")
    return q / norm


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("quaternion operands must be 4-vectors.")
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_conjugate(q: ArrayLike) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def rotate_vector(q: ArrayLike, vector: ArrayLike) -> np.ndarray:
    q = normalize_quaternion(q)
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("vector must be a 3-vector.")
    rotated = quaternion_multiply(
        quaternion_multiply(q, [0.0, *vector]),
        quaternion_conjugate(q),
    )
    return rotated[1:]


def quaternion_from_matrix(matrix: ArrayLike) -> np.ndarray:
    """Return a body-to-inertial quaternion from a 3x3 direction-cosine matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3.")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8):
        raise ValueError("matrix must be orthonormal.")
    if np.linalg.det(matrix) <= 0.0:
        raise ValueError("matrix must be right-handed.")

    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quaternion(q)


def attitude_quaternion_from_frame(frame: str, **kwargs) -> np.ndarray:
    """Return a body-to-GCRF quaternion whose body axes match ``frame`` axes."""

    return quaternion_from_matrix(frame_to_gcrf_matrix(frame, **kwargs))
