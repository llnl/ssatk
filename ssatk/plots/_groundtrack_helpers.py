"""Shared helpers for ground-track dashboard plots."""

from __future__ import annotations

import numpy as np


def as_list(value):
    """Return ``value`` as a list-like container used by ground-track plots."""
    return value if isinstance(value, (list, tuple)) else [value]


def broadcast_time_list(r_list, t):
    """Return a time list matching the number of ground-track position tracks."""
    if isinstance(t, (list, tuple)):
        if len(t) != len(r_list):
            raise ValueError(
                "When passing a list of times, its length must match the number of orbits."
            )
        return list(t)
    return [t for _ in r_list]


def clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0):
    """Insert NaNs at longitude wrap crossings so lines do not jump across maps."""
    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)

    jumps = np.where(np.abs(np.diff(lon_deg)) > threshold)[0]
    if jumps.size == 0:
        return lon_deg, lat_deg

    lon_out = np.insert(lon_deg, jumps + 1, np.nan)
    lat_out = np.insert(lat_deg, jumps + 1, np.nan)
    return lon_out, lat_out


def ensure_nx3(array):
    """Return a position array as shape ``(N, 3)`` or raise ``ValueError``."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Each 'r' must be 2D; got shape {array.shape}")
    if 3 in array.shape:
        if array.shape[1] == 3:
            return array
        if array.shape[0] == 3:
            return array.T
    raise ValueError(f"Each 'r' must have a dimension of size 3; got shape {array.shape}")


def clean_lonlat(lon, lat):
    """Insert NaNs at longitude wrap breaks for continuous ground-track lines."""

    wraps = np.abs(np.diff(lon)) > 180
    lon_nan = np.insert(lon, np.where(wraps)[0] + 1, np.nan)
    lat_nan = np.insert(lat, np.where(wraps)[0] + 1, np.nan)
    return lon_nan, lat_nan


def force_title(ax, text, size, y=1.02):
    """Set a title consistently for 2D and 3D matplotlib axes."""

    ax.set_title("")
    if hasattr(ax, "text2D"):
        ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size)
    else:
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size)
