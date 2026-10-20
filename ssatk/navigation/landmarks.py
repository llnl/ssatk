"""Geometric position fixes from identified finite-distance landmarks.

Landmarks and bearings share one Cartesian frame. Distances are metres;
these instantaneous geometric rays do not include optical propagation effects.
"""

import numpy as np


def _unit_rows(vectors, name):
    vectors = np.asarray(vectors, float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must be a finite (N,3) array")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} must contain nonzero directions")
    return vectors / norms[:, None]


def position_from_landmark_bearings(landmarks_m, inertial_bearings):
    """Triangulate position from identified finite-distance landmark bearings.

    Solve perpendicular distances to bearing lines in metres. Bearings point
    from observer to landmarks; this is an unweighted geometric fix, not an
    optimal orbit filter. Stars at infinity must not be passed as landmarks.
    """
    landmarks = np.asarray(landmarks_m, float)
    directions = _unit_rows(inertial_bearings, "inertial_bearings")
    if landmarks.shape != directions.shape or len(landmarks) < 2 or not np.isfinite(landmarks).all():
        raise ValueError("at least two finite landmark positions and bearings are required")
    projectors = np.eye(3) - directions[:, :, None]*directions[:, None, :]
    matrix = projectors.sum(axis=0)
    if np.linalg.cond(matrix) > 1e8:
        raise ValueError("landmark bearings do not constrain position")
    rhs = np.einsum("nij,nj->i", projectors, landmarks)
    position = np.linalg.solve(matrix, rhs)
    if np.any(np.einsum("ni,ni->n", landmarks-position, directions) <= 0):
        raise ValueError("landmark solution lies behind a measured bearing")
    return position


__all__ = ["position_from_landmark_bearings"]
