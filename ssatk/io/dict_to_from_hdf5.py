"""
Utilities for saving/loading nested Python dictionaries to/from HDF5.

Public API:
    save_dict_to_hdf5(filename, data, ...)
    load_dict_from_hdf5(filename, keys=None)
"""

import os
import pickle
import datetime as _dt
from typing import Any, Mapping, Optional, Set, Union

import h5py
import numpy as np

try:
    from astropy.time import Time as AstroTime
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


# ---------- SAVE ----------


def save_dict_to_hdf5(
    filename: str,
    data: Mapping[str, Any],
    mode: str = "w",
    *,
    pickle_objects: bool = True,
    compression: Union[str, None] = "gzip",
    compression_opts: Union[int, None] = 4,
) -> None:
    """
    Save a (possibly nested) Python dictionary to an HDF5 file.

    Signature:
        save_dict_to_hdf5(filename, data, ...)

    Supported types:
        - dict -> HDF5 group
        - list/tuple of numeric scalars -> HDF5 dataset (auto-converted to numpy array)
        - list/tuple of mixed/non-numeric -> HDF5 group with numeric keys ("0", "1", ...)
        - numpy arrays -> datasets (with compression, if requested)
        - scalars: int/float/bool, numpy scalar types -> scalar datasets (no compression)
        - str -> variable-length UTF-8 datasets
        - bytes/bytearray/memoryview -> bytes datasets
        - datetime.datetime/date/time -> stored as ISO strings
        - astropy.time.Time -> stored as (mjd, scale, format, meta)
        - other objects -> pickled if pickle_objects=True

    Notes:
        - Flat numeric lists/tuples are automatically converted to numpy arrays
          to avoid the "thousands of groups" explosion that occurs when storing
          large lists element-by-element.
    """

    def _store_string_with_type(
        h5group: h5py.Group,
        key: str,
        value: str,
        type_name: str,
    ) -> None:
        if key in h5group:
            del h5group[key]
        dt = h5py.string_dtype(encoding="utf-8")
        ds = h5group.create_dataset(
            key,
            data=np.array(value, dtype=dt),
            dtype=dt,
        )
        ds.attrs["__type__"] = type_name

    def _write_item(
        h5group: h5py.Group,
        key: str,
        value: Any,
    ) -> None:
        # nested dict -> subgroup
        if isinstance(value, Mapping):
            if key in h5group and isinstance(h5group[key], h5py.Group):
                subgroup = h5group[key]
                subgroup.attrs["__is_sequence__"] = False
            else:
                subgroup = h5group.require_group(key)
                subgroup.attrs["__is_sequence__"] = False

            for k, v in value.items():
                if not isinstance(k, str):
                    raise TypeError(f"HDF5 requires string keys; got key={k!r}")
                _write_item(subgroup, k, v)

        # list/tuple
        elif isinstance(value, (list, tuple)):
            # Try to convert flat numeric lists to a numpy array to avoid
            # storing thousands of individual scalar datasets as groups.
            try:
                arr = np.asarray(value)
                if arr.size > 0 and arr.dtype.kind in ('i', 'u', 'f', 'c') and arr.ndim >= 1:
                    if key in h5group:
                        del h5group[key]
                    if arr.shape == ():
                        h5group.create_dataset(key, data=arr)
                    else:
                        h5group.create_dataset(
                            key,
                            data=arr,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                    return
            except (ValueError, TypeError):
                pass

            # Non-numeric or ragged list: fall back to group-per-element
            if key in h5group and isinstance(h5group[key], h5py.Group):
                subgroup = h5group[key]
                subgroup.attrs["__is_sequence__"] = True
            else:
                subgroup = h5group.require_group(key)
                subgroup.attrs["__is_sequence__"] = True

            for child in list(subgroup.keys()):
                del subgroup[child]

            for idx, v in enumerate(value):
                _write_item(subgroup, str(idx), v)

        # astropy Time
        elif _HAS_ASTROPY and isinstance(value, AstroTime):
            if key in h5group:
                del h5group[key]
            t_group = h5group.create_group(key)
            mjd_arr = np.array(value.mjd, dtype="float64")
            t_group.create_dataset("mjd", data=mjd_arr)
            t_group.attrs["__type__"] = "astropy.time.Time"
            t_group.attrs["scale"] = value.scale
            t_group.attrs["format"] = "mjd"

            if getattr(value, "meta", None):
                meta_pickled = pickle.dumps(dict(value.meta), protocol=pickle.HIGHEST_PROTOCOL)
                meta_ds = t_group.create_dataset(
                    "meta",
                    data=np.frombuffer(meta_pickled, dtype="uint8"),
                )
                meta_ds.attrs["pickled"] = True

        # datetime types -> ISO strings with type attrs
        elif isinstance(value, _dt.datetime):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.datetime")
        elif isinstance(value, _dt.date):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.date")
        elif isinstance(value, _dt.time):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.time")

        # numpy array
        elif isinstance(value, np.ndarray):
            if key in h5group:
                del h5group[key]
            if value.shape == ():
                h5group.create_dataset(key, data=value)
            else:
                h5group.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )

        # numeric scalars
        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            if key in h5group:
                del h5group[key]
            h5group.create_dataset(key, data=value)

        # strings -> vlen UTF-8
        elif isinstance(value, str):
            if key in h5group:
                del h5group[key]
            dt = h5py.string_dtype(encoding="utf-8")
            h5group.create_dataset(
                key,
                data=np.array(value, dtype=dt),
                dtype=dt,
            )

        # bytes-like
        elif isinstance(value, (bytes, bytearray, memoryview)):
            if key in h5group:
                del h5group[key]
            arr = np.frombuffer(bytes(value), dtype="uint8")
            ds = h5group.create_dataset(key, data=arr)
            ds.attrs["__bytes__"] = True

        # fallback: pickle
        else:
            if not pickle_objects:
                raise TypeError(
                    f"Unsupported type for key '{key}': {type(value)!r}. "
                    f"Enable pickle_objects=True to store via pickle."
                )
            if key in h5group:
                del h5group[key]
            pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            ds = h5group.create_dataset(
                key,
                data=np.frombuffer(pickled, dtype="uint8"),
            )
            ds.attrs["pickled"] = True
            ds.attrs["python_type"] = str(type(value))
            ds.attrs["__type__"] = "pickle"

    # Fix: os.path.dirname returns "" for bare filenames, which causes makedirs to fail
    parent = os.path.dirname(os.path.abspath(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(filename, mode) as f:
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"HDF5 requires string keys; got key={k!r}")
            _write_item(f, k, v)


# ---------- LOAD ----------


def load_dict_from_hdf5(filename: str, keys: Optional[Set[str]] = None) -> dict:
    """
    Load a dictionary previously stored with save_dict_to_hdf5(filename, data, ...).

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    keys : set of str, optional
        If provided, only the specified top-level keys are loaded from disk.
        All other keys are ignored, avoiding unnecessary decompression of
        large datasets. If None (default), all keys are loaded.

    Behavior:
        - Groups marked with __is_sequence__=True are reconstructed as lists,
          even if empty.
        - Groups marked with __is_sequence__=False are reconstructed as dicts,
          even if empty.
        - Groups without the attribute:
            * if keys are 0..n-1 as strings, treated as lists
            * otherwise, treated as dicts.
        - Supported special types: astropy.time.Time, datetime.*, bytes, pickled objects.
    """

    def _read_item(obj: Union[h5py.Group, h5py.Dataset]) -> Any:
        # Group
        if isinstance(obj, h5py.Group):
            # astropy Time group?
            if "__type__" in obj.attrs and obj.attrs["__type__"] == "astropy.time.Time":
                if not _HAS_ASTROPY:
                    raise ImportError(
                        "astropy is required to load astropy.time.Time objects."
                    )
                mjd = np.array(obj["mjd"][...], dtype="float64")
                scale = obj.attrs["scale"]
                t = AstroTime(mjd, format="mjd", scale=scale)

                if "meta" in obj:
                    meta_ds = obj["meta"]
                    if meta_ds.attrs.get("pickled", False):
                        meta_bytes = bytes(meta_ds[...].tolist())
                        meta = pickle.loads(meta_bytes)
                        t.meta.update(meta)
                return t

            # Explicit sequence or mapping marker
            if "__is_sequence__" in obj.attrs:
                if obj.attrs["__is_sequence__"]:
                    # list-like
                    obj_keys = list(obj.keys())
                    if not obj_keys:
                        return []
                    int_keys = sorted(int(k) for k in obj_keys)
                    return [_read_item(obj[str(i)]) for i in int_keys]
                else:
                    # dict-like
                    out = {}
                    for k in obj.keys():
                        out[k] = _read_item(obj[k])
                    return out

            # No explicit marker: infer from keys
            obj_keys = list(obj.keys())
            if not obj_keys:
                return {}

            try:
                int_keys = sorted(int(k) for k in obj_keys)
                is_seq = int_keys == list(range(len(obj_keys)))
            except ValueError:
                is_seq = False

            if is_seq:
                return [_read_item(obj[str(i)]) for i in range(len(obj_keys))]
            else:
                out = {}
                for k in obj_keys:
                    out[k] = _read_item(obj[k])
                return out

        # Dataset
        ds: h5py.Dataset = obj  # type: ignore[assignment]

        # pickled object?
        if ds.attrs.get("__type__") == "pickle" or ds.attrs.get("pickled", False):
            arr = np.array(ds[...], dtype="uint8")
            return pickle.loads(arr.tobytes())

        # bytes dataset?
        if ds.attrs.get("__bytes__", False):
            arr = np.array(ds[...], dtype="uint8")
            return bytes(arr.tobytes())

        # type-tagged strings (datetime, etc.)
        tname = ds.attrs.get("__type__", None)
        if tname:
            value = ds.asstr()[()]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if tname == "datetime.datetime":
                return _dt.datetime.fromisoformat(value)
            elif tname == "datetime.date":
                return _dt.date.fromisoformat(value)
            elif tname == "datetime.time":
                return _dt.time.fromisoformat(value)

        # plain string dataset
        if h5py.check_string_dtype(ds.dtype) is not None:
            return ds.asstr()[()]

        # numeric / array dataset
        arr = ds[...]
        if arr.shape == ():
            return arr[()]  # numpy scalar
        return arr

    with h5py.File(filename, "r") as f:
        result = {}
        for k in f.keys():
            if keys is not None and k not in keys:
                continue
            result[k] = _read_item(f[k])
        return result
