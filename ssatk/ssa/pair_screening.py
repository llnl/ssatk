"""One-epoch angular and velocity pair screening.

The screen is deliberately a state-only operation.  Positions and velocities
are SI GCRF vectors for one epoch; no propagation, frame conversion, or
catalogue handling is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PairScreenResult:
    """Result of a one-epoch pair screen."""

    pairs: np.ndarray
    candidate_count: int
    strategy: str
    valid_count: int


def _chord_radius(cosine_limit: float, angular_limit_rad=None) -> float:
    """Return a conservative unit-sphere chord radius.

    The allowance covers normalization and distance roundoff at exact
    coincidence (notably cosine one and angular limit zero).  Final acceptance
    always uses the direct predicates in :func:`screen_pairs`. Supplied angles
    use sin(angle/2) to avoid cancellation in 1-cos(angle) near zero.
    """
    if angular_limit_rad is None:
        # The direct scalar cosine can differ from a dot product of the
        # pre-normalized rows by a few ulps near one.  Broaden candidates only;
        # final acceptance remains strictly predicate-based.
        cosine_limit = max(
            -1.0, cosine_limit - _roundoff_envelope(cosine_limit)
        )
        radius = np.sqrt(max(0.0, 2.0 - 2.0 * cosine_limit))
    else:
        radius = 2.0 * np.sin(angular_limit_rad / 2.0)
    allowance = 32.0 * np.finfo(float).eps
    return float(np.nextafter(radius + allowance, np.inf))


def _roundoff_envelope(*values: float) -> float:
    return 32.0 * np.finfo(float).eps * max(1.0, *(abs(value) for value in values))


def _near_cutoff(values: np.ndarray, cutoff: float) -> np.ndarray:
    return np.abs(values - cutoff) <= _roundoff_envelope(cutoff)


def _scalar_predicate(
    first_los,
    second_los,
    first_velocity,
    second_velocity,
    angular_limit,
    velocity_threshold,
) -> bool:
    first_speed = np.linalg.norm(first_velocity)
    second_speed = np.linalg.norm(second_velocity)
    velocity_cosine = np.dot(first_velocity, second_velocity) / (
        first_speed * second_speed
    )
    angle = np.arctan2(
        np.linalg.norm(np.cross(first_los, second_los)),
        np.dot(first_los, second_los),
    )
    return (
        angle <= angular_limit
        and np.clip(velocity_cosine, -1.0, 1.0) > velocity_threshold
    )


def _tree_pairs(features: np.ndarray, cosine_limit: float, angular_limit_rad=None) -> np.ndarray:
    if len(features) < 2:
        return np.empty((0, 2), dtype=np.intp)
    pairs = np.asarray(
        cKDTree(features).query_pairs(_chord_radius(cosine_limit, angular_limit_rad), output_type="ndarray"),
        dtype=np.intp,
    )
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.intp)
    pairs = pairs.reshape(-1, 2)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _tree_pair_batches(
    features: np.ndarray, cosine_limit: float, batch_size: int, angular_limit_rad=None
):
    """Yield tree candidates in bounded batches, retaining no global pair list."""
    tree = cKDTree(features)
    radius = _chord_radius(cosine_limit, angular_limit_rad)
    for first in range(len(features) - 1):
        neighbors = np.asarray(
            tree.query_ball_point(features[first], radius, return_sorted=True),
            dtype=np.intp,
        )
        neighbors = neighbors[neighbors > first]
        for start in range(0, len(neighbors), batch_size):
            second = neighbors[start : start + batch_size]
            yield np.column_stack(
                (np.full(second.size, first, dtype=np.intp), second)
            )


def _brute_pair_batches(count: int, batch_size: int):
    """Yield each unordered pair once without materializing the triangular matrix."""
    for first in range(count - 1):
        for start in range(first + 1, count, batch_size):
            second = np.arange(start, min(start + batch_size, count), dtype=np.intp)
            yield np.column_stack(
                (np.full(second.size, first, dtype=np.intp), second)
            )


def _sample_indices(count: int, maximum: int = 256) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.intp)
    return np.linspace(0, count - 1, maximum, dtype=np.intp)


def _choose_adaptive(
    velocity_directions: np.ndarray,
    sightline_directions: np.ndarray,
    velocity_cosine: float,
    sightline_angle: float,
) -> str:
    sample = _sample_indices(len(velocity_directions))
    velocity_count = len(
        _tree_pairs(velocity_directions[sample], velocity_cosine)
    )
    sightline_count = len(
        _tree_pairs(sightline_directions[sample], np.cos(sightline_angle), sightline_angle)
    )
    return "velocity" if velocity_count <= sightline_count else "sightline"


def screen_pairs(
    positions,
    velocities,
    observer_position,
    *,
    angular_limit_rad=np.pi / 4,
    velocity_threshold=0.98,
    strategy="adaptive",
    valid=None,
    pair_batch_size=65536,
) -> PairScreenResult:
    """Screen one epoch of states for angularly and kinematically similar pairs.

    A pair passes when the angle between its observer sightlines is at most
    ``angular_limit_rad`` and the cosine between its velocity directions is
    strictly greater than ``velocity_threshold``.  KD-tree candidates use
    outward-rounded chord cutoffs; the final predicates are evaluated directly
    with ``atan2`` and a clipped dot product.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    observer_position = np.asarray(observer_position, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3).")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have shape (N, 3).")
    if observer_position.shape != (3,):
        raise ValueError("observer_position must have shape (3,).")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (len(positions),):
            raise ValueError("valid must have shape (N,).")
    angular_limit_rad = float(angular_limit_rad)
    velocity_threshold = float(velocity_threshold)
    if not np.isfinite(angular_limit_rad) or angular_limit_rad < 0.0:
        raise ValueError("angular_limit_rad must be in [0, pi].")
    if angular_limit_rad > np.pi:
        raise ValueError("angular_limit_rad must be in [0, pi].")
    if not np.isfinite(velocity_threshold) or not -1.0 <= velocity_threshold <= 1.0:
        raise ValueError("velocity_threshold must be in [-1, 1].")
    if isinstance(pair_batch_size, (bool, np.bool_)) or not isinstance(
        pair_batch_size, Integral
    ):
        raise TypeError("pair_batch_size must be an integer, not bool.")
    pair_batch_size = int(pair_batch_size)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive.")
    if strategy not in {"velocity", "sightline", "adaptive", "brute_force"}:
        raise ValueError(f"unknown screening strategy: {strategy!r}")

    position_finite = np.all(np.isfinite(positions), axis=1)
    velocity_finite = np.all(np.isfinite(velocities), axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    line_of_sight = positions - observer_position
    sightline_finite = np.all(np.isfinite(line_of_sight), axis=1)
    sightline_lengths = np.linalg.norm(line_of_sight, axis=1)
    row_valid = position_finite & velocity_finite & sightline_finite
    row_valid &= np.isfinite(speeds) & (speeds > 0.0)
    row_valid &= np.isfinite(sightline_lengths) & (sightline_lengths > 0.0)
    if valid is not None:
        row_valid &= valid

    indices = np.flatnonzero(row_valid)
    valid_count = int(indices.size)
    if valid_count < 2:
        return PairScreenResult(
            np.empty((0, 2), dtype=np.intp), 0, "brute_force" if strategy == "brute_force" else strategy, valid_count
        )
    if velocity_threshold == 1.0:
        return PairScreenResult(
            np.empty((0, 2), dtype=np.intp), 0, "velocity" if strategy == "adaptive" else strategy, valid_count
        )

    velocity_directions = velocities[indices] / speeds[indices, None]
    sightline_directions = line_of_sight[indices] / sightline_lengths[indices, None]
    angular_cosine = np.cos(min(angular_limit_rad, np.pi))
    if strategy == "adaptive":
        selected_strategy = _choose_adaptive(
            velocity_directions,
            sightline_directions,
            velocity_threshold,
            angular_limit_rad,
        )
    else:
        selected_strategy = strategy

    if selected_strategy == "velocity":
        candidate_batches = _tree_pair_batches(
            velocity_directions, velocity_threshold, pair_batch_size
        )
    elif selected_strategy == "sightline":
        candidate_batches = _tree_pair_batches(
            sightline_directions, angular_cosine, pair_batch_size, angular_limit_rad
        )
    else:
        candidate_batches = _brute_pair_batches(valid_count, pair_batch_size)

    accepted = []
    candidate_count = 0
    for batch in candidate_batches:
        candidate_count += len(batch)
        first, second = batch[:, 0], batch[:, 1]
        first_los = line_of_sight[indices[first]]
        second_los = line_of_sight[indices[second]]
        cross_norm = np.linalg.norm(np.cross(first_los, second_los), axis=1)
        dot = np.einsum("ij,ij->i", first_los, second_los)
        angles = np.arctan2(cross_norm, dot)
        velocity_cosine = np.einsum(
            "ij,ij->i", velocity_directions[first], velocity_directions[second]
        )
        keep = (angles <= angular_limit_rad) & (
            np.clip(velocity_cosine, -1.0, 1.0) > velocity_threshold
        )
        ambiguous = _near_cutoff(angles, angular_limit_rad) | _near_cutoff(
            velocity_cosine, velocity_threshold
        )
        for local_index in np.flatnonzero(ambiguous):
            keep[local_index] = _scalar_predicate(
                first_los[local_index],
                second_los[local_index],
                velocities[indices[first[local_index]]],
                velocities[indices[second[local_index]]],
                angular_limit_rad,
                velocity_threshold,
            )
        if np.any(keep):
            accepted.append(np.column_stack((indices[first[keep]], indices[second[keep]])))

    if accepted:
        pairs = np.concatenate(accepted).astype(np.intp, copy=False)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
    else:
        pairs = np.empty((0, 2), dtype=np.intp)
    return PairScreenResult(pairs, int(candidate_count), selected_strategy, valid_count)
