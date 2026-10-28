import numpy as np

from ssatk.ssa.event_windows import EventWindow, EventWindowAccumulator, buffered_event_windows


def _assert_windows_exact(actual, expected):
    assert [(window.pair, window.trigger_count) for window in actual] == [
        (window.pair, window.trigger_count) for window in expected
    ]
    for actual_window, expected_window in zip(actual, expected):
        np.testing.assert_allclose(
            [
                actual_window.start,
                actual_window.end,
                actual_window.trigger_start,
                actual_window.trigger_end,
            ],
            [
                expected_window.start,
                expected_window.end,
                expected_window.trigger_start,
                expected_window.trigger_end,
            ],
            rtol=0.0,
            atol=0.0,
        )


def test_reference_union_handles_unordered_pairs_invalid_samples_and_gap():
    times = [0.0, 2.0, 5.0, 20.0, 21.0]
    pairs = [
        np.array([[3, 1], [4, 5]]),
        np.array([[1, 3], [5, 4]]),
        np.array([[1, 3], [7, 8]]),
        np.array([[3, 1]]),
        np.array([[1, 3]]),
    ]
    valid = [
        np.ones(9, dtype=bool),
        np.array([True, True, True, True, False, True, True, True, True]),
        np.ones(9, dtype=bool),
        np.ones(9, dtype=bool),
        np.ones(9, dtype=bool),
    ]
    expected = [
        EventWindow((1, 3), -3.0, 9.0, 0.0, 5.0, 3),
        EventWindow((1, 3), 17.0, 25.0, 20.0, 21.0, 2),
        EventWindow((4, 5), -3.0, 4.0, 0.0, 0.0, 1),
        EventWindow((7, 8), 2.0, 9.0, 5.0, 5.0, 1),
    ]
    actual = buffered_event_windows(times, pairs, 3.0, 4.0, 10.0, valid)
    _assert_windows_exact(actual, expected)
    assert actual[0].trigger_count == 3
    assert actual[0].trigger_start == 0.0
    assert actual[0].trigger_end == 5.0


def test_irregular_gap_and_finish_clipping_do_not_change_trigger_truth():
    accumulator = EventWindowAccumulator(pre_padding_s=2.0, post_padding_s=2.0, max_gap_s=4.0)
    accumulator.append(10.0, [[2, 0]])
    accumulator.append(10.5, [[0, 2]])
    accumulator.append(20.0, [[0, 2]])  # gap closes the first run
    snapshot = accumulator.finish(start_time=8.0, end_time=22.0)
    assert snapshot == [
        EventWindow((0, 2), 8.0, 12.5, 10.0, 10.5, 2),
        EventWindow((0, 2), 18.0, 22.0, 20.0, 20.0, 1),
    ]
    assert accumulator.finish(start_time=8.0, end_time=22.0) == snapshot


def test_chunking_is_exactly_equivalent_and_finish_is_non_mutating():
    times = np.array([0.0, 1.25, 4.0, 4.5, 9.0, 10.0])
    pairs = [
        [[2, 1]],
        [[1, 2], [9, 8]],
        [],
        [[8, 9]],
        [[1, 2]],
        [[2, 1]],
    ]
    valid = [
        None,
        np.array([True] * 10),
        None,
        np.array([True] * 10),
        None,
        None,
    ]
    one_shot = buffered_event_windows(times, pairs, 2.0, 3.0, 6.0, valid)
    accumulator = EventWindowAccumulator(2.0, 3.0, 6.0)
    for index in range(0, len(times), 2):
        for time, pair_rows, row_valid in zip(
            times[index : index + 2],
            pairs[index : index + 2],
            valid[index : index + 2],
        ):
            accumulator.append(time, pair_rows, row_valid)
    assert accumulator.finish() == one_shot
    assert accumulator.finish() == one_shot
