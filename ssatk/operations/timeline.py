"""Deterministic time-tagged mission event scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True, order=True)
class MissionEvent:
    """One time-tagged mission command or state transition."""

    time: float
    name: str = field(compare=False)
    priority: int = 0
    payload: Any = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("event name must be a non-empty string.")
        if not isinstance(self.time, (int, float)) or not math.isfinite(float(self.time)):
            raise ValueError("event time must be finite.")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "priority", int(self.priority))


class MissionTimeline:
    """Immutable event collection with deterministic same-epoch ordering."""

    def __init__(self, events=()):
        self._events = tuple(sorted((event if isinstance(event, MissionEvent) else MissionEvent(**event) for event in events), key=lambda event: (event.time, event.priority, event.name)))
        if any(self._events[index].time == self._events[index + 1].time and self._events[index].priority == self._events[index + 1].priority and self._events[index].name == self._events[index + 1].name for index in range(len(self._events) - 1)):
            raise ValueError("duplicate event time, priority, and name.")

    @property
    def events(self) -> tuple[MissionEvent, ...]:
        return self._events

    def between(self, start: float, stop: float, *, include_start: bool = True, include_stop: bool = True) -> tuple[MissionEvent, ...]:
        """Return events in a closed/open time window in execution order."""
        start, stop = float(start), float(stop)
        if stop < start:
            raise ValueError("stop must be greater than or equal to start.")
        return tuple(
            event for event in self._events
            if (event.time > start or include_start and event.time == start)
            and (event.time < stop or include_stop and event.time == stop)
        )

    def at(self, time: float, *, tolerance: float = 0.0) -> tuple[MissionEvent, ...]:
        """Return events at ``time`` within an optional absolute tolerance."""
        tolerance = float(tolerance)
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        time = float(time)
        return tuple(event for event in self._events if abs(event.time - time) <= tolerance)


__all__ = ["MissionEvent", "MissionTimeline"]
