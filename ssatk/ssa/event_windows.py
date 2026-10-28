"""Streaming aggregation of sampled SSA event pairs."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class EventWindow:
    """One padded window and the trigger samples that produced it."""

    pair: tuple[int, int]
    start: float
    end: float
    trigger_start: float
    trigger_end: float
    trigger_count: int

    def __post_init__(self) -> None:
        pair = tuple(self.pair)
        if len(pair) != 2 or not all(isinstance(index, Integral) for index in pair):
            raise ValueError("pair must contain two integer indices.")
        pair = tuple(sorted((int(pair[0]), int(pair[1]))))
        if pair[0] < 0 or pair[0] == pair[1]:
            raise ValueError("pair must contain two distinct nonnegative indices.")
        values = (self.start, self.end, self.trigger_start, self.trigger_end)
        values = tuple(float(value) for value in values)
        if not np.all(np.isfinite(values)):
            raise ValueError("event-window times must be finite.")
        if values[1] < values[0] or values[3] < values[2]:
            raise ValueError("event-window times must be ordered.")
        count = int(self.trigger_count)
        if count < 1 or count != self.trigger_count:
            raise ValueError("trigger_count must be a positive integer.")
        object.__setattr__(self, "pair", pair)
        for name, value in zip(
            ("start", "end", "trigger_start", "trigger_end"), values
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "trigger_count", count)


class EventWindowAccumulator:
    """Accumulate contiguous sampled triggers without retaining samples.

    ``valid`` is an optional per-object mask for the current sample.  A pair
    triggers only when both indexed objects are valid.  Padding can still
    place an invalid sample inside the resulting context window; the context
    is not a continuous detection interval.
    """

    def __init__(
        self,
        pre_padding_s: float = 300,
        post_padding_s: float = 300,
        max_gap_s: float | None = None,
    ) -> None:
        self.pre_padding_s = _nonnegative(pre_padding_s, "pre_padding_s")
        self.post_padding_s = _nonnegative(post_padding_s, "post_padding_s")
        self.max_gap_s = (
            None if max_gap_s is None else _nonnegative(max_gap_s, "max_gap_s")
        )
        self._active: dict[tuple[int, int], list[float | int]] = {}
        self._windows: dict[tuple[int, int], list[EventWindow]] = {}
        self._last_time: float | None = None

    def append(self, time, pairs, valid=None) -> None:
        """Append one sampled time and its triggered pair rows."""
        time = _finite_time(time)
        pair_array = np.asarray(pairs)
        if pair_array.size == 0:
            pair_array = np.empty((0, 2), dtype=int)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            raise ValueError("pairs must have shape (K, 2).")
        if not np.issubdtype(pair_array.dtype, np.integer):
            try:
                numeric = pair_array.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("pair indices must be integers.") from exc
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
                raise ValueError("pair indices must be integers.")
            pair_array = numeric.astype(np.intp)
        else:
            pair_array = pair_array.astype(np.intp, copy=False)
        if pair_array.size and (
            np.any(pair_array < 0) or np.any(pair_array[:, 0] == pair_array[:, 1])
        ):
            raise ValueError("pairs must contain distinct nonnegative indices.")

        if self._last_time is not None:
            delta = time - self._last_time
            if delta <= 0.0:
                raise ValueError("sample times must be strictly increasing.")
            if self.max_gap_s is not None and delta > self.max_gap_s:
                self._close_all()

        mask = _valid_mask(valid, pair_array)
        triggered = {
            tuple(sorted((int(first), int(second))))
            for first, second in pair_array
            if mask is None
            or (mask if isinstance(mask, bool) else (mask[first] and mask[second]))
        }
        for pair in set(self._active) - triggered:
            self._close(pair)
        for pair in triggered:
            state = self._active.get(pair)
            if state is None:
                self._active[pair] = [time, time, 1]
            else:
                state[1] = time
                state[2] += 1
        self._last_time = time

    def finish(self, start_time=None, end_time=None) -> list[EventWindow]:
        """Return a snapshot, closing pending runs without changing state."""
        start = None if start_time is None else _finite_time(start_time)
        end = None if end_time is None else _finite_time(end_time)
        if start is not None and end is not None and end < start:
            raise ValueError("end_time must be no earlier than start_time.")

        raw = [window for windows in self._windows.values() for window in windows]
        raw.extend(
            EventWindow(pair, state[0], state[1], state[0], state[1], state[2])
            for pair, state in self._active.items()
        )
        padded = [
            (
                window.pair,
                window.start - self.pre_padding_s,
                window.end + self.post_padding_s,
                window.trigger_start,
                window.trigger_end,
                window.trigger_count,
            )
            for window in raw
        ]
        if start is not None or end is not None:
            padded = [
                (
                    pair,
                    max(left, start) if start is not None else left,
                    min(right, end) if end is not None else right,
                    trigger_start,
                    trigger_end,
                    count,
                )
                for pair, left, right, trigger_start, trigger_end, count in padded
                if (end is None or left <= end) and (start is None or right >= start)
            ]

        result: list[EventWindow] = []
        for pair, left, right, trigger_start, trigger_end, count in sorted(
            padded, key=lambda item: (item[0], item[1], item[2])
        ):
            if result and result[-1].pair == pair and left <= result[-1].end:
                previous = result.pop()
                result.append(
                    EventWindow(
                        pair,
                        previous.start,
                        max(previous.end, right),
                        min(previous.trigger_start, trigger_start),
                        max(previous.trigger_end, trigger_end),
                        previous.trigger_count + count,
                    )
                )
            else:
                result.append(
                    EventWindow(
                        pair, left, right, trigger_start, trigger_end, count
                    )
                )
        return result

    def _close(self, pair: tuple[int, int]) -> None:
        start, end, count = self._active.pop(pair)
        window = EventWindow(pair, start, end, start, end, count)
        windows = self._windows.setdefault(pair, [])
        if windows and start - windows[-1].end <= self.pre_padding_s + self.post_padding_s:
            previous = windows.pop()
            windows.append(
                EventWindow(
                    pair,
                    previous.start,
                    end,
                    previous.trigger_start,
                    window.trigger_end,
                    previous.trigger_count + count,
                )
            )
        else:
            windows.append(window)

    def _close_all(self) -> None:
        for pair in list(self._active):
            self._close(pair)


def buffered_event_windows(
    times,
    pairs_by_time,
    pre_padding_s: float = 300,
    post_padding_s: float = 300,
    max_gap_s: float | None = None,
    valid_by_time=None,
) -> list[EventWindow]:
    """Aggregate a complete sequence using the streaming implementation."""
    times = list(times)
    pairs_by_time = list(pairs_by_time)
    if len(times) != len(pairs_by_time):
        raise ValueError("times and pairs_by_time must have the same length.")
    if valid_by_time is None:
        valid_by_time = [None] * len(times)
    else:
        valid_by_time = list(valid_by_time)
        if len(valid_by_time) != len(times):
            raise ValueError("valid_by_time must match times in length.")
    accumulator = EventWindowAccumulator(pre_padding_s, post_padding_s, max_gap_s)
    for time, pairs, valid in zip(times, pairs_by_time, valid_by_time):
        accumulator.append(time, pairs, valid)
    return accumulator.finish()


def _finite_time(value) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("sample times must be finite.")
    return value


def _nonnegative(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _valid_mask(valid, pairs: np.ndarray) -> np.ndarray | bool | None:
    if valid is None:
        return None
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim == 0:
        return bool(mask)
    if mask.ndim != 1:
        raise ValueError("valid must be a scalar or one-dimensional bool array.")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= mask.size)):
        raise ValueError("pairs contain an index outside valid.")
    return mask
