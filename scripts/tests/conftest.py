"""Observe geometry submitted to Matplotlib without private vertex buffers."""

from weakref import WeakKeyDictionary

import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@pytest.fixture
def submitted_vertices(monkeypatch):
    """Record real set_verts calls; preserve rendering and independent oracles.

    Matplotlib 3.11 removed Poly3DCollection._vec. Observe the public input
    boundary instead, including subsequent animation updates. Store only the
    latest vertices per live artist, and return copies for frame comparisons.
    """
    latest = WeakKeyDictionary()
    original = Poly3DCollection.set_verts

    def record(artist, verts, *args, **kwargs):
        result = original(artist, verts, *args, **kwargs)
        latest[artist] = (np.concatenate(verts).copy() if len(verts)
                          else np.empty((0, 3)))
        return result

    monkeypatch.setattr(Poly3DCollection, "set_verts", record)
    return lambda artist: latest[artist].copy()
