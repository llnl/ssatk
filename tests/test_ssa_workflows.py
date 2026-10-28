"""End-to-end sampled screening against explicit geometric references."""

import numpy as np
import pytest

from ssatk.ssa.trajectories import StateBatch, ArrayStateProvider
from ssatk.ssa.observer import ObserverStates
from ssatk.ssa.event_windows import EventWindow
from ssatk.ssa.workflows import (
    PairAnalysisConfig, analyze_pairs, iter_event_metrics, estimate_event_rows,
)


def _source():
    times = np.array([0., 1., 2., 3., 8., 9.])
    angles = np.array([.1, .2, .1, .1, .1, .9])
    radius = 7e6
    positions = np.empty((3, len(times), 3))
    positions[0] = [radius, 0, 0]
    positions[1] = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    positions[2] = [-radius, 0, 0]
    velocities = np.empty_like(positions)
    velocities[:2] = [0, 7500, 0]
    velocities[2] = [0, -7500, 0]
    valid = np.ones((3, len(times)), bool)
    valid[1, 2] = False
    source = ArrayStateProvider(StateBatch(("A", "B", "C"), times, positions, velocities,
                                          valid, {"input": "analytic synthetic geometry"}))
    observer = ObserverStates(times, np.zeros((6, 3)), np.zeros((6, 3)))
    return source, observer, angles


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("chunk_size", [1, 2, 6])
def test_analysis_windows_and_metrics_match_closed_form(workers, chunk_size):
    source, observer, angles = _source()
    config = PairAnalysisConfig(angular_limit_rad=.4, pre_padding_s=1,
                                post_padding_s=1, max_gap_s=3)
    result = analyze_pairs(source, observer, config=config, chunk_size=chunk_size, workers=workers)
    # Explicit interval union: [0,1] and [3,3] padded by 1 merge; gap to [8,8].
    assert result.windows == (
        EventWindow((0, 1), 0, 4, 0, 3, 3),
        EventWindow((0, 1), 7, 9, 8, 8, 1),
    )
    assert result.complete and result.sample_count == 6
    assert estimate_event_rows(result, source.times) == 6
    batches = list(iter_event_metrics(result, source, observer, chunk_size=chunk_size))
    times = np.concatenate([batch.times for batch in batches])
    order = np.argsort(times)
    np.testing.assert_array_equal(times[order], source.times)
    valid = np.concatenate([batch.valid for batch in batches])[order]
    trigger = np.concatenate([batch.trigger for batch in batches])[order]
    np.testing.assert_array_equal(valid, [True, True, False, True, True, True])
    np.testing.assert_array_equal(trigger, [True, True, False, True, True, False])
    distance = np.concatenate([batch.metrics["distance_m"] for batch in batches])[order]
    expected = 2 * 7e6 * np.sin(angles / 2)
    expected[2] = np.nan
    # Chord length closed form, 10 nm absolute tolerance in metres.
    np.testing.assert_allclose(distance, expected, atol=1e-8, rtol=0, equal_nan=True)
    cosine = np.concatenate([batch.metrics["velocity_cosine"] for batch in batches])[order]
    np.testing.assert_allclose(cosine[valid], 1, atol=1e-15, rtol=0)


def test_no_padding_preserves_invalid_breaks_and_large_time_gaps():
    source, observer, _ = _source()
    config = PairAnalysisConfig(angular_limit_rad=.4, pre_padding_s=0,
                                post_padding_s=0, max_gap_s=3)
    result = analyze_pairs(source, observer, config=config)
    assert result.windows == (
        EventWindow((0, 1), 0, 1, 0, 1, 2),
        EventWindow((0, 1), 3, 3, 3, 3, 1),
        EventWindow((0, 1), 8, 8, 8, 8, 1),
    )


def test_elevation_and_illumination_gates_do_not_discard_padded_context():
    source, observer, _ = _source()
    # Local east=Y, north=Z, up=X: A zenith, B elevation=pi/2-angle.
    basis = np.broadcast_to([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], (6, 3, 3))
    observer = ObserverStates(observer.times, observer.positions, observer.velocities, basis)

    def illumination(batch):
        mask = np.ones(batch.valid.shape, bool)
        mask[1, batch.times == 1] = False
        return mask

    config = PairAnalysisConfig(angular_limit_rad=1, pre_padding_s=1, post_padding_s=1,
                                max_gap_s=3, min_elevation_rad=1.3)
    result = analyze_pairs(source, observer, config=config, illumination=illumination)
    # t=1 passes geometric elevation but illumination removes the trigger.
    assert sum(window.trigger_count for window in result.windows) == 3
    batches = list(iter_event_metrics(result, source, observer, illumination=illumination))
    np.testing.assert_array_equal(np.concatenate([b.trigger for b in batches]),
                                  [True, False, False, True, True, False])


def test_cancelled_run_reports_only_processed_samples_and_clipped_windows():
    source, observer, _ = _source()
    progress = []
    result = analyze_pairs(source, observer, chunk_size=2,
                           progress=lambda values: progress.append(values),
                           cancelled=lambda: bool(progress))
    assert not result.complete
    assert result.sample_count == 2 and result.end_time == 1
    assert result.windows == (EventWindow((0, 1), 0, 1, 0, 1, 2),)
