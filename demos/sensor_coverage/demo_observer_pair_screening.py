"""Deterministic observer-relative pair screening with a replayable provider."""

from __future__ import annotations

from time import perf_counter

import numpy as np

import ssatk
from ssatk.navigation import GroundStation
from ssatk.plots.plotutils import figsave
from ssatk.ssa.observer import from_ground_station
from ssatk.ssa.pair_screening import screen_pairs
from ssatk.ssa.trajectories import OrbitStateProvider
from ssatk.ssa.workflows import (
    PairAnalysisConfig,
    analyze_pairs,
    estimate_event_rows,
    iter_event_metrics,
)

GALLERY_CATEGORY = "sensor_coverage"


def _provider(times):
    orbits = {
        "A": ssatk.Orbit.fromKeplerianElements(
            7_000_000.0, 0.001, np.radians(51.6), 0.0, 0.0, 0.0, t=times[0]
        ),
        "B": ssatk.Orbit.fromKeplerianElements(
            7_000_000.0, 0.001, np.radians(51.6), 0.0, 0.0, 0.008, t=times[0]
        ),
        "C": ssatk.Orbit.fromKeplerianElements(
            7_150_000.0, 0.01, np.radians(48.0), 0.3, 0.8, 2.0, t=times[0]
        ),
        "D": ssatk.Orbit.fromKeplerianElements(
            7_400_000.0, 0.02, np.radians(63.0), 0.7, 1.4, 1.0, t=times[0]
        ),
    }
    return OrbitStateProvider(
        orbits,
        times,
        propagator=ssatk.KeplerianPropagator(),
        provenance={"example": "synthetic_observer_pair_screening"},
    )


def _benchmark(provider, observer, times):
    batch = next(provider.iter_chunks(len(times)))
    observer_state = observer(times)
    kwargs = dict(
        angular_limit_rad=np.radians(2.0),
        velocity_threshold=0.999,
        valid=np.ones(len(provider.object_ids), dtype=bool),
    )
    measurements = {}
    reference = None
    for strategy in ("brute_force", "velocity", "sightline", "adaptive"):
        start = perf_counter()
        pairs = []
        candidates = 0
        for index in range(len(times)):
            result = screen_pairs(
                batch.positions[:, index],
                batch.velocities[:, index],
                observer_state.positions[index],
                strategy=strategy,
                **kwargs,
            )
            pairs.append(result.pairs)
            candidates += result.candidate_count
        measurements[strategy] = {
            "seconds": perf_counter() - start,
            "candidate_count": candidates,
        }
        if reference is None:
            reference = pairs
        else:
            assert all(np.array_equal(left, right) for left, right in zip(reference, pairs))
    return measurements


def main(make_figures=True, benchmark=False):
    times = 1.4e9 + np.arange(0.0, 1801.0, 60.0)
    provider = _provider(times)
    station = GroundStation(-106.5, 35.1, elevation_m=1600.0, min_elevation_rad=np.radians(5.0))
    observer = lambda block: from_ground_station(station, block)
    config = PairAnalysisConfig(
        angular_limit_rad=np.radians(10.0),
        velocity_threshold=0.98,
        strategy="adaptive",
        pre_padding_s=120.0,
        post_padding_s=120.0,
        max_gap_s=120.0,
        # Access gates are optional; this example leaves them open to focus
        # on pair windows and metric replay.
        min_elevation_rad=-np.pi / 2,
    )
    result = analyze_pairs(provider, observer, config=config, chunk_size=7)
    row_estimate = estimate_event_rows(result, times)
    metric_batches = list(iter_event_metrics(result, provider, observer, chunk_size=7))
    metric_rows = sum(len(batch.times) for batch in metric_batches)
    assert row_estimate == metric_rows

    if benchmark:
        result.diagnostics["benchmark"] = _benchmark(provider, observer, times)
    result.diagnostics["estimated_metric_rows"] = row_estimate
    print(
        f"observer pair screen: samples={result.sample_count}, "
        f"windows={len(result.windows)}, metric_rows={metric_rows}, "
        f"complete={result.complete}"
    )
    if benchmark:
        for name, values in result.diagnostics["benchmark"].items():
            print(f"  {name}: {values['seconds']:.6f} s, candidates={values['candidate_count']}")

    if make_figures:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 3.5))
        if metric_batches:
            window_batches = [
                batch for batch in metric_batches if batch.window_index == 0
            ]
            plot_times = np.concatenate([batch.times for batch in window_batches])
            plot_distance = np.concatenate(
                [batch.metrics["distance_m"] for batch in window_batches]
            )
            axis.plot((plot_times - times[0]) / 60.0, plot_distance / 1e3)
            axis.set_ylabel("A-B separation (km)")
        axis.set_xlabel("Elapsed time (min)")
        axis.set_title("Synthetic observer pair context")
        axis.grid(alpha=0.3)
        # gallery-call: demo_observer_pair_screening.png
        figsave(figure, "demo_observer_pair_screening.png")
        # gallery-end
        plt.close(figure)
    return result


if __name__ == "__main__":
    main(make_figures=True, benchmark=True)
