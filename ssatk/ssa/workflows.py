"""Streaming observer-pair analysis; persistence is a separate concern."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from numbers import Integral
import platform
import hashlib
import time

import numpy as np

from .trajectories import StateBatch, _chunk_size
from .observer import ObserverStates, observer_metrics, earth_occulted
from .pair_screening import screen_pairs
from .event_windows import EventWindowAccumulator
from .pair_geometry import pair_metrics


@dataclass(frozen=True)
class PairAnalysisConfig:
    """Sampled trigger settings, with angles in radians and padding in seconds.

    Velocity cosine uses absolute GCRF velocities, not relative observer
    velocities. Angular separation is inclusive; velocity threshold is strict.
    Access/illumination are opt-in and do not establish detectability.
    """

    angular_limit_rad: float = np.pi / 4
    velocity_threshold: float = 0.98
    strategy: str = "adaptive"
    pre_padding_s: float = 300.0
    post_padding_s: float = 300.0
    max_gap_s: float | None = None
    min_elevation_rad: float | None = None
    occultation_radius_m: float | None = None
    pair_batch_size: int = 65536

    def __post_init__(self):
        for name, low, high in (("angular_limit_rad", 0, np.pi),
                                ("velocity_threshold", -1, 1)):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}].")
            object.__setattr__(self, name, value)
        for name in ("pre_padding_s", "post_padding_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        for name in ("max_gap_s", "occultation_radius_m"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        if self.min_elevation_rad is not None:
            if not np.isfinite(self.min_elevation_rad) or not -np.pi / 2 <= self.min_elevation_rad <= np.pi / 2:
                raise ValueError("min_elevation_rad must be in [-pi/2, pi/2].")
        if self.strategy not in {"adaptive", "velocity", "sightline", "brute_force"}:
            raise ValueError("unsupported candidate strategy.")
        _chunk_size(self.pair_batch_size)


@dataclass(frozen=True)
class PairAnalysisResult:
    """Window-only result; no detailed state histories are retained."""

    object_ids: tuple
    windows: tuple
    config: PairAnalysisConfig
    start_time: float
    end_time: float
    sample_count: int
    complete: bool
    diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EventMetricBatch:
    """One event's bounded sample block, including invalid padded context."""

    window_index: int
    object_ids: tuple
    times: np.ndarray
    metrics: dict
    valid: np.ndarray
    trigger: np.ndarray


def _observer_at(observer, times):
    if callable(observer):
        value = observer(times)
    elif isinstance(observer, ObserverStates):
        indices = np.searchsorted(observer.times, times)
        if np.any(indices >= len(observer.times)) or not np.array_equal(observer.times[indices], times):
            raise ValueError("observer must contain every requested epoch exactly.")
        basis = None if observer.enu_basis is None else observer.enu_basis[indices]
        value = ObserverStates(times, observer.positions[indices],
                               observer.velocities[indices], basis)
    else:
        raise TypeError("observer must be ObserverStates or a time-block callable.")
    if not isinstance(value, ObserverStates) or not np.array_equal(value.times, times):
        raise ValueError("observer provider must return matching ObserverStates epochs.")
    return value


def _usable(batch, observer, config, illumination):
    observed = observer_metrics(batch.positions, batch.velocities, observer)
    valid = batch.valid & observed["valid"]
    if config.min_elevation_rad is not None:
        if observer.enu_basis is None:
            raise ValueError("elevation gating requires an observer ENU basis.")
        valid &= observed["elevation_rad"] >= config.min_elevation_rad
    if config.occultation_radius_m is not None:
        valid &= ~earth_occulted(batch.positions, observer.positions,
                                radius_m=config.occultation_radius_m)
    if illumination is not None:
        mask = np.asarray(illumination(batch), dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("illumination callback must return an (objects, times) mask.")
        valid &= mask
    return valid


def _fingerprint_sample(digest, batch, observer, gated, index):
    # Interleaving by epoch makes the digest independent of chunk partitioning.
    for value in (batch.times[index], batch.positions[:, index],
                  batch.velocities[:, index], batch.valid[:, index],
                  observer.positions[index], observer.velocities[index],
                  gated[:, index]):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    if observer.enu_basis is not None:
        digest.update(np.asarray(observer.enu_basis[index], dtype="<f8").tobytes())


def analyze_pairs(source, observer, *, config=None, chunk_size=300, workers=1,
                  progress=None, cancelled=None, illumination=None):
    """Analyze a provider's sampled states without creating detailed output.

    Providers expose ``object_ids``, ``times``, ``provenance`` and
    ``iter_chunks(chunk_size)``. Epochs must match the declared time vector
    exactly, with no overlaps or omissions. ``progress`` receives counters;
    ``cancelled()`` is checked between chunks. Parallel screening uses ordered
    threads per chunk; aggregation stays serial and deterministic.

    ``illumination(batch)`` is an optional externally computed boolean mask;
    supply the same deterministic callback for detailed metric extraction.
    """
    from .trajectories import gps_times
    from .. import __version__

    config = PairAnalysisConfig() if config is None else config
    if not isinstance(config, PairAnalysisConfig):
        raise TypeError("config must be PairAnalysisConfig.")
    chunk_size = _chunk_size(chunk_size)
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    times = gps_times(source.times)
    ids = tuple(source.object_ids)
    accumulator = EventWindowAccumulator(config.pre_padding_s, config.post_padding_s,
                                         config.max_gap_s)
    count = candidate_count = invalid_count = 0
    strategies = {}
    stopped = False
    digest = hashlib.sha256()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    executor = ThreadPoolExecutor(max_workers=int(workers)) if workers > 1 else None
    try:
        for batch in source.iter_chunks(chunk_size):
            if cancelled is not None and cancelled():
                stopped = True
                break
            if not isinstance(batch, StateBatch) or batch.object_ids != ids:
                raise ValueError("source chunks must be StateBatch with stable object IDs.")
            expected = times[count:count + len(batch.times)]
            if not np.array_equal(expected, batch.times):
                raise ValueError("source chunks must exactly cover declared times in order.")
            obs = _observer_at(observer, batch.times)
            valid = _usable(batch, obs, config, illumination)
            invalid_count += int((~batch.valid).sum())

            def screen(index):
                return screen_pairs(batch.positions[:, index], batch.velocities[:, index],
                                    obs.positions[index], angular_limit_rad=config.angular_limit_rad,
                                    velocity_threshold=config.velocity_threshold,
                                    strategy=config.strategy, valid=valid[:, index],
                                    pair_batch_size=config.pair_batch_size)

            results = map(screen, range(len(batch.times))) if executor is None else executor.map(screen, range(len(batch.times)))
            for index, result in enumerate(results):
                _fingerprint_sample(digest, batch, obs, valid, index)
                accumulator.append(float(batch.times[index]), result.pairs, valid=valid[:, index])
                candidate_count += result.candidate_count
                strategies[result.strategy] = strategies.get(result.strategy, 0) + 1
            count += len(batch.times)
            if progress is not None:
                progress({"sample_count": count, "total_samples": len(times),
                          "candidate_count": candidate_count})
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if not stopped and count != len(times):
        raise ValueError("source ended before covering every declared epoch.")
    end = float(times[count - 1]) if count else float(times[0])
    windows = () if not count else tuple(accumulator.finish(float(times[0]), end))
    provenance = {
        "schema": "ssatk.ssa.pair_analysis.v1",
        "frame": "GCRF", "state_units": ["m", "m/s"], "time_scale": "GPS",
        "ssatk_version": __version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source": dict(getattr(source, "provenance", {})),
        "illumination_gate": illumination is not None,
        "sample_fingerprint_sha256": digest.hexdigest(),
        "declared_times_sha256": hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest(),
        "declared_sample_count": len(times),
        "config": asdict(config),
    }
    return PairAnalysisResult(ids, windows, config, float(times[0]), end, count,
                              not stopped, {"candidate_count": candidate_count,
                                            "invalid_state_samples": invalid_count,
                                            "strategies": strategies,
                                            "workers": int(workers),
                                            "wall_seconds": time.perf_counter() - wall_start,
                                            "process_cpu_seconds": time.process_time() - cpu_start}, provenance)


def iter_event_metrics(result, source, observer, *, chunk_size=300, illumination=None):
    """Replay deterministic states to stream padded metric histories.

    The caller must supply the same source, observer and illumination models.
    A chunk-independent sample fingerprint is checked on iterator exhaustion;
    do not consider streamed output committed until exhaustion succeeds.
    Padding can include invalid or gated-out samples. ``valid`` denotes
    finite state/LOS geometry, while ``trigger`` additionally applies gates.
    """
    if tuple(source.object_ids) != result.object_ids:
        raise ValueError("source object IDs differ from the analysis.")
    if bool(result.provenance.get("illumination_gate")) != (illumination is not None):
        raise ValueError("supply the same illumination gate used in analysis.")
    from .trajectories import gps_times

    replay_times = gps_times(source.times)
    time_digest = hashlib.sha256(np.asarray(replay_times, dtype="<f8").tobytes()).hexdigest()
    if (len(replay_times) != result.provenance["declared_sample_count"]
            or time_digest != result.provenance["declared_times_sha256"]):
        raise ValueError("replayed declared timeline differs from the analysis.")
    grouped = {}
    for index, window in enumerate(result.windows):
        grouped.setdefault(window.pair, []).append((index, window))
    intervals = {}
    for pair, events in grouped.items():
        events.sort(key=lambda item: item[1].start)
        intervals[pair] = (
            events,
            np.asarray([window.start for _, window in events]),
            np.asarray([window.end for _, window in events]),
        )
    digest = hashlib.sha256()
    count = 0
    declared_times = replay_times[:result.sample_count]
    for batch in source.iter_chunks(_chunk_size(chunk_size)):
        if count == result.sample_count:
            if result.complete:
                raise ValueError("replayed source yielded extra epochs beyond its declared timeline.")
            break
        if batch.object_ids != result.object_ids:
            raise ValueError("replayed chunk object IDs differ from the analysis.")
        used = min(len(batch.times), result.sample_count - count)
        if result.complete and used != len(batch.times):
            raise ValueError("replayed source yielded extra epochs beyond its declared timeline.")
        if not np.array_equal(batch.times[:used], declared_times[count:count + used]):
            raise ValueError("replayed chunks do not exactly match declared epochs.")
        obs = _observer_at(observer, batch.times)
        geometry_valid = batch.valid & observer_metrics(batch.positions, batch.velocities, obs)["valid"]
        gated = _usable(batch, obs, result.config, illumination)
        for sample in range(used):
            _fingerprint_sample(digest, batch, obs, gated, sample)
        count += used
        for pair, (events, starts, ends) in intervals.items():
            a, b = pair
            first = np.searchsorted(ends, batch.times[0], side="left")
            last = np.searchsorted(starts, batch.times[used - 1], side="right")
            for index, window in events[first:last]:
                selected = np.flatnonzero((batch.times >= window.start) & (batch.times <= window.end))
                if not selected.size:
                    continue
                metrics = pair_metrics(batch.positions[a, selected], batch.velocities[a, selected],
                                       batch.positions[b, selected], batch.velocities[b, selected],
                                       obs.positions[selected], obs.velocities[selected])
                valid = geometry_valid[a, selected] & geometry_valid[b, selected]
                for key, value in metrics.items():
                    value = np.array(value, copy=True)
                    value[~valid] = np.nan
                    metrics[key] = value
                trigger = (gated[a, selected] & gated[b, selected]
                           & (metrics["angular_separation_rad"] <= result.config.angular_limit_rad)
                           & (metrics["velocity_cosine"] > result.config.velocity_threshold))
                yield EventMetricBatch(index, (result.object_ids[a], result.object_ids[b]),
                                       batch.times[selected], metrics, valid, trigger)
    if count != result.sample_count or digest.hexdigest() != result.provenance["sample_fingerprint_sha256"]:
        raise ValueError("replayed states, observer, epochs or gates differ from the analysis.")


def estimate_event_rows(result, times):
    """Exact sampled row count before extracting detailed histories."""
    from .trajectories import gps_times

    times = gps_times(times)
    return sum(int(np.searchsorted(times, window.end, side="right")
                   - np.searchsorted(times, window.start, side="left"))
               for window in result.windows)


__all__ = ["PairAnalysisConfig", "PairAnalysisResult", "EventMetricBatch",
           "analyze_pairs", "iter_event_metrics", "estimate_event_rows"]
