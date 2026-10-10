"""
Utilities for working with longitude/latitude tracks.

Provides a helper to break line segments at large longitude jumps
(e.g., crossing the dateline), inserting NaNs so matplotlib does not draw
artifact lines across the whole map, while trying to preserve input types.
"""

from __future__ import annotations

from typing import Tuple, Union, Sequence

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def break_plot_line(
    lon: ArrayLike,
    lat: ArrayLike,
    max_jump: float = 179.0,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Insert NaNs into lon/lat where the longitude jumps more than `max_jump` degrees.

    Type behavior:
        - If `lon`/`lat` are numpy arrays -> returns numpy arrays (dtype preserved
          unless conversion to float is needed to hold NaNs).
        - If `lon`/`lat` are lists -> returns lists.
        - Other sequences are converted to numpy arrays.

    Args:
        lon:
            Sequence or numpy array of longitudes (degrees).
        lat:
            Sequence or numpy array of latitudes (same length as `lon`).
        max_jump:
            Threshold in degrees for detecting a "wrap" jump. Any consecutive
            longitude difference with absolute value > max_jump will be treated
            as a break point.

    Returns:
        (lon_fixed, lat_fixed):
            Same *container type* as inputs (list or numpy array), with NaNs
            inserted at break points. When used with matplotlib, lines will
            break at NaNs.

    Example:
        >>> lon = [170, 175, 179, -179, -175, -170]
        >>> lat = [  0,   1,   2,    3,    4,    5]
        >>> lon2, lat2 = break_dateline(lon, lat)
        >>> type(lon2), type(lat2)
        (<class 'list'>, <class 'list'>)
    """
    # Detect container types
    lon_is_array = isinstance(lon, np.ndarray)
    lat_is_array = isinstance(lat, np.ndarray)
    lon_is_list = isinstance(lon, list)
    lat_is_list = isinstance(lat, list)

    # Convert to numpy for processing
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)

    if lon_arr.shape != lat_arr.shape:
        raise ValueError("lon and lat must have the same shape")

    if lon_arr.size < 2:
        # Nothing to break; return in original-ish type
        return _cast_back(lon_arr, lon_is_array, lon_is_list), _cast_back(
            lat_arr, lat_is_array, lat_is_list
        )

    dlon = np.diff(lon_arr.astype(float))  # safe for diff
    breaks = np.where(np.abs(dlon) > max_jump)[0]

    if breaks.size == 0:
        return _cast_back(lon_arr, lon_is_array, lon_is_list), _cast_back(
            lat_arr, lat_is_array, lat_is_list
        )

    # Work in float to allow NaNs if necessary
    lon_float = lon_arr.astype(float, copy=True)
    lat_float = lat_arr.astype(float, copy=True)

    lon_list = lon_float.tolist()
    lat_list = lat_float.tolist()

    # Insert NaNs after each break; go backwards so indices stay valid
    for idx in breaks[::-1]:
        lon_list.insert(idx + 1, np.nan)
        lat_list.insert(idx + 1, np.nan)

    lon_new = np.array(lon_list, dtype=float)
    lat_new = np.array(lat_list, dtype=float)

    return _cast_back(lon_new, lon_is_array, lon_is_list), _cast_back(
        lat_new, lat_is_array, lat_is_list
    )


def _cast_back(arr: np.ndarray, was_array: bool, was_list: bool):
    """
    Helper to convert back to original container type:
      - numpy array if original was array
      - list if original was list
      - numpy array otherwise
    """
    if was_array:
        return arr
    if was_list:
        # Convert numpy array to list of Python floats (or scalars)
        return arr.tolist()
    return arr