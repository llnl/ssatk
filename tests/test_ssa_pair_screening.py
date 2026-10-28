import numpy as np
import pytest

from ssatk.ssa.pair_screening import screen_pairs


def _scalar_reference(
    positions,
    velocities,
    observer,
    angular_limit,
    velocity_threshold,
    valid=None,
):
    """Independent scalar double-loop reference, using the specified predicates."""
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    observer = np.asarray(observer, dtype=float)
    valid_mask = np.ones(len(positions), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    rows = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not valid_mask[i] or not valid_mask[j]:
                continue
            first_los = positions[i] - observer
            second_los = positions[j] - observer
            first_speed = np.linalg.norm(velocities[i])
            second_speed = np.linalg.norm(velocities[j])
            first_length = np.linalg.norm(first_los)
            second_length = np.linalg.norm(second_los)
            if not all(
                np.isfinite(value)
                for value in (first_speed, second_speed, first_length, second_length)
            ):
                continue
            if first_speed == 0.0 or second_speed == 0.0:
                continue
            if first_length == 0.0 or second_length == 0.0:
                continue
            velocity_cosine = np.dot(velocities[i], velocities[j]) / (first_speed * second_speed)
            angle = np.arctan2(
                np.linalg.norm(np.cross(first_los, second_los)),
                np.dot(first_los, second_los),
            )
            if angle <= angular_limit and np.clip(velocity_cosine, -1.0, 1.0) > velocity_threshold:
                rows.append((i, j))
    return np.asarray(rows, dtype=int).reshape(-1, 2)


@pytest.mark.parametrize("strategy", ["velocity", "sightline", "adaptive", "brute_force"])
@pytest.mark.parametrize("batch_size", [1, 7, 65536])
def test_screen_matches_independent_scalar_reference(strategy, batch_size):
    rng = np.random.default_rng(20261027)
    positions = rng.normal(size=(48, 3)) * 1.0e7
    velocities = rng.normal(size=(48, 3)) * 7.0e3
    observer = np.array([1.2e6, -2.3e6, 0.7e6])
    valid = rng.random(48) > 0.12
    expected = _scalar_reference(positions, velocities, observer, 0.7, 0.93, valid)

    result = screen_pairs(
        positions,
        velocities,
        observer,
        angular_limit_rad=0.7,
        velocity_threshold=0.93,
        strategy=strategy,
        valid=valid,
        pair_batch_size=batch_size,
    )

    np.testing.assert_array_equal(result.pairs, expected)
    assert result.valid_count == int(valid.sum())
    assert result.candidate_count >= len(expected)


def test_boundary_predicates_are_inclusive_for_angle_and_strict_for_velocity():
    observer = np.zeros(3)
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cos(0.4), np.sin(0.4), 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [np.cos(0.3), np.sin(0.3), 0.0],
        ]
    )
    result = screen_pairs(
        positions,
        velocities,
        observer,
        angular_limit_rad=0.4,
        velocity_threshold=np.cos(0.3),
        strategy="brute_force",
    )
    expected = _scalar_reference(
        positions,
        velocities,
        observer,
        0.4,
        np.cos(0.3),
    )
    np.testing.assert_array_equal(result.pairs, expected)


def test_nonfinite_zero_and_explicitly_invalid_rows_are_excluded():
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    valid = np.array([True, True, True, True, False])
    result = screen_pairs(positions, velocities, np.zeros(3), valid=valid, strategy="brute_force")
    np.testing.assert_array_equal(result.pairs, np.array([[0, 1]], dtype=int))
    assert result.valid_count == 2


def test_identical_and_nearly_identical_los_matches_scalar_reference():
    angle = np.nextafter(0.2, np.inf)
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
            [np.cos(np.nextafter(0.2, 0.0)), np.sin(np.nextafter(0.2, 0.0)), 0.0],
        ]
    )
    velocities = np.tile([2.0, 0.0, 0.0], (4, 1))
    expected = _scalar_reference(positions, velocities, np.zeros(3), 0.2, 0.99)

    for strategy in ("velocity", "sightline", "adaptive", "brute_force"):
        result = screen_pairs(
            positions,
            velocities,
            np.zeros(3),
            angular_limit_rad=0.2,
            velocity_threshold=0.99,
            strategy=strategy,
            pair_batch_size=2,
        )
        np.testing.assert_array_equal(result.pairs, expected)


def test_random_boundary_rotations_match_reference_for_every_strategy():
    rng = np.random.default_rng(991)
    count = 30
    positions = rng.normal(size=(count, 3))
    positions /= np.linalg.norm(positions, axis=1)[:, None]
    velocities = rng.normal(size=(count, 3))
    velocities /= np.linalg.norm(velocities, axis=1)[:, None]
    angular_limit = 0.35
    velocity_threshold = np.cos(0.42)
    for index in range(0, count, 3):
        positions[index] = [1.0, 0.0, 0.0]
        positions[index + 1] = [np.cos(angular_limit), np.sin(angular_limit), 0.0]
        velocities[index] = [1.0, 0.0, 0.0]
        velocities[index + 1] = [np.cos(0.42), np.sin(0.42), 0.0]
    expected = _scalar_reference(
        positions,
        velocities,
        np.zeros(3),
        angular_limit,
        velocity_threshold,
    )

    results = [
        screen_pairs(
            positions,
            velocities,
            np.zeros(3),
            angular_limit_rad=angular_limit,
            velocity_threshold=velocity_threshold,
            strategy=strategy,
            pair_batch_size=5,
        )
        for strategy in ("velocity", "sightline", "adaptive", "brute_force")
    ]
    for result in results:
        np.testing.assert_array_equal(result.pairs, expected)
    for result in results[1:]:
        np.testing.assert_array_equal(result.pairs, results[0].pairs)


@pytest.mark.parametrize("strategy", ["velocity", "sightline", "adaptive", "brute_force"])
def test_sub_nanoradian_angular_gate_matches_scalar_reference(strategy):
    """Analytic 1e-10 rad separation, exact unordered-pair reference."""
    angle = 1e-10
    positions = np.array([[1., 0., 0.], [np.cos(angle), np.sin(angle), 0.]])
    velocities = np.array([[0., 1., 0.], [0., 1., 0.]])
    reference = _scalar_reference(positions, velocities, np.zeros(3), 2e-10, .98)
    np.testing.assert_array_equal(reference, [[0, 1]])
    result = screen_pairs(positions, velocities, np.zeros(3), strategy=strategy,
                          angular_limit_rad=2e-10, velocity_threshold=.98)
    np.testing.assert_array_equal(result.pairs, reference)
