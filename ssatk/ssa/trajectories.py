"""Chunk providers for GPS-time, GCRF SI catalog states."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from ..time_functions import to_gps


def gps_times(times):
    """Return a nonempty, finite, strictly increasing GPS-second vector."""
    values = np.asarray(to_gps(times), dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("times must be a nonempty finite GPS-second vector.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("times must be strictly increasing.")
    return values


def _chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("chunk_size must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class StateBatch:
    """Catalog states shaped ``(objects, times, 3)`` with a validity mask.

    Object IDs are unique, hashable identifiers. Invalid nonfinite states are
    retained but masked. ``valid`` cannot make a nonfinite state usable.
    """

    object_ids: tuple
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    valid: np.ndarray | None = None
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        ids = tuple(self.object_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("object_ids must be nonempty and unique.")
        times = np.array(gps_times(self.times), copy=True)
        shape = (len(ids), len(times), 3)
        positions = np.array(self.positions, dtype=float, copy=True)
        velocities = np.array(self.velocities, dtype=float, copy=True)
        if positions.shape != shape or velocities.shape != shape:
            raise ValueError(f"positions and velocities must have shape {shape}.")
        finite = np.isfinite(positions).all(axis=-1) & np.isfinite(velocities).all(axis=-1)
        if self.valid is None:
            valid = finite
        else:
            valid = np.array(self.valid, dtype=bool, copy=True)
            if valid.shape != shape[:2]:
                raise ValueError("valid must have shape (objects, times).")
            valid &= finite
        for value in (times, positions, velocities, valid):
            value.flags.writeable = False
        for name, value in (("object_ids", ids), ("times", times),
                            ("positions", positions), ("velocities", velocities),
                            ("valid", valid), ("provenance", dict(self.provenance))):
            object.__setattr__(self, name, value)


class ArrayStateProvider:
    """Replayable chunk source backed by a precomputed :class:`StateBatch`."""

    def __init__(self, batch: StateBatch):
        if not isinstance(batch, StateBatch):
            raise TypeError("batch must be a StateBatch.")
        self.batch = batch
        self.object_ids = batch.object_ids
        self.times = batch.times
        self.provenance = dict(batch.provenance)

    def iter_chunks(self, chunk_size=300):
        """Yield bounded time chunks in chronological order."""
        chunk_size = _chunk_size(chunk_size)
        for start in range(0, len(self.times), chunk_size):
            stop = start + chunk_size
            yield StateBatch(self.object_ids, self.times[start:stop],
                             self.batch.positions[:, start:stop],
                             self.batch.velocities[:, start:stop],
                             self.batch.valid[:, start:stop], self.provenance)


class OrbitStateProvider:
    """Propagate existing SSATK orbit objects one time chunk at a time.

    ``orbits`` maps IDs to SSATK ``Orbit`` objects, including those constructed
    through existing TLE/OMM readers. Supply a propagator explicitly, e.g.
    ``ssatk.SGP4Propagator()`` for TLEs. Exceptions are not silently replaced by
    another propagator. Nonfinite returned states are masked by StateBatch.
    """

    def __init__(self, orbits: Mapping, times, *, propagator, provenance=None):
        if not orbits:
            raise ValueError("orbits must be a nonempty mapping.")
        self.orbits = dict(orbits)
        self.object_ids = tuple(self.orbits)
        self.times = np.array(gps_times(times), copy=True)
        self.times.flags.writeable = False
        self.propagator = propagator
        self.provenance = dict(provenance or {})
        self.provenance["propagator"] = type(propagator).__name__

    def iter_chunks(self, chunk_size=300):
        """Use SSATK's public ``rv`` interface; no new dynamics model."""
        from .. import rv

        chunk_size = _chunk_size(chunk_size)
        for start in range(0, len(self.times), chunk_size):
            times = self.times[start:start + chunk_size]
            positions, velocities = [], []
            for orbit in self.orbits.values():
                r, v = rv(orbit, times, propagator=self.propagator)
                positions.append(np.asarray(r).reshape(len(times), 3))
                velocities.append(np.asarray(v).reshape(len(times), 3))
            yield StateBatch(self.object_ids, times, positions, velocities,
                             provenance=self.provenance)


__all__ = ["StateBatch", "ArrayStateProvider", "OrbitStateProvider"]
