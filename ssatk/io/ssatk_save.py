"""Universal SSATK save/load helpers selected by file extension."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ssatk._paths import ensure_file_parent, safe_relative_parts
from ssatk.io.datapath import datapath
from ssatk.io.dict_to_from_hdf5 import load_dict_from_hdf5, save_dict_to_hdf5
from ssatk.io.json_utils import load_json, save_json
from ssatk.io.pickle_utils import read_pickle, save_pickle
from ssatk.io.xml_utils import read_xml, save_xml

_DATA_ROOTS = {"auto", "data", "figures", "cwd"}
_HDF5_EXTS = {".h5", ".hdf5", ".hdf"}
_JSON_EXTS = {".json"}
_JSONL_EXTS = {".jsonl", ".ndjson"}
_CSV_EXTS = {".csv", ".tsv"}
_ARRAY_EXTS = {".npy", ".npz"}
_PICKLE_EXTS = {".pkl", ".pickle"}
_XML_EXTS = {".xml"}
_TEXT_EXTS = {".txt", ".log"}
_TABLE_EXTS = {".parquet", ".feather"}
_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".svg", ".pdf", ".eps", ".ps"}
_SUPPORTED_EXTS = (
    _HDF5_EXTS
    | _JSON_EXTS
    | _JSONL_EXTS
    | _CSV_EXTS
    | _ARRAY_EXTS
    | _PICKLE_EXTS
    | _XML_EXTS
    | _TEXT_EXTS
    | _TABLE_EXTS
    | _FIGURE_EXTS
)


def ssatk_save(
    data: Any,
    path: str | Path,
    *,
    key: str | None = None,
    overwrite: bool = True,
    root: str | Path = "auto",
    compression: str | bool | None = None,
    compression_opts: int | None = 4,
    pickle_objects: bool = True,
    allow_pickle: bool = False,
    index: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs,
) -> Path:
    """Save data using the file extension in ``path`` to select the format.

    Bare and relative filenames are rooted under ``~/ssatk_output`` for data
    formats and ``~/ssatk_output/figures`` for figure formats. Absolute paths
    are honored. Use ``root="cwd"`` to write a relative path below the current
    working directory, or pass a path-like ``root`` as a custom output base.

    HDF5/NPZ keys are deterministic: mappings use their own keys, while a
    single non-mapping object defaults to key ``"data"`` unless ``key=`` is
    supplied.
    """
    save_path = _resolve_path(path, root=root, for_save=True)
    suffix = _extension(save_path)
    if suffix not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported save extension '{suffix}'. Supported: {sorted(_SUPPORTED_EXTS)}")
    if key is not None and suffix not in _HDF5_EXTS and suffix != ".npz":
        raise ValueError("key= is only supported for HDF5 and NPZ outputs.")
    if save_path.exists() and not overwrite and suffix not in _HDF5_EXTS:
        raise FileExistsError(f"Refusing to overwrite existing file: {save_path}")

    if suffix in _FIGURE_EXTS:
        _save_figure(data, save_path, overwrite=overwrite, **kwargs)
    elif suffix in _HDF5_EXTS:
        _save_hdf5(
            data,
            save_path,
            key=key,
            overwrite=overwrite,
            compression=_normalize_hdf5_compression(compression),
            compression_opts=compression_opts,
            pickle_objects=pickle_objects,
            metadata=metadata,
        )
    elif suffix in _JSON_EXTS:
        save_json(save_path, data)
    elif suffix in _JSONL_EXTS:
        _save_jsonl(data, save_path, overwrite=overwrite)
    elif suffix in _CSV_EXTS:
        sep = "\t" if suffix == ".tsv" else ","
        _to_dataframe(data).to_csv(save_path, index=index, sep=sep, **kwargs)
    elif suffix == ".npy":
        _save_npy(data, save_path, allow_pickle=allow_pickle, **kwargs)
    elif suffix == ".npz":
        _save_npz(data, save_path, key=key, allow_pickle=allow_pickle, compressed=bool(compression), **kwargs)
    elif suffix in _PICKLE_EXTS:
        save_pickle(data, save_path)
    elif suffix in _XML_EXTS:
        root_tag = kwargs.pop("root_tag", "root")
        save_xml(save_path, data, root_tag=root_tag, **kwargs)
    elif suffix in _TEXT_EXTS:
        _save_text(data, save_path, **kwargs)
    elif suffix == ".parquet":
        _to_dataframe(data).to_parquet(save_path, index=index, **kwargs)
    elif suffix == ".feather":
        _to_dataframe(data).reset_index(drop=not index).to_feather(save_path, **kwargs)
    else:  # pragma: no cover - guarded by _SUPPORTED_EXTS
        raise ValueError(f"Unsupported save extension '{suffix}'.")

    return save_path


def ssatk_read(
    path: str | Path,
    *,
    key: str | None = None,
    root: str | Path = "auto",
    allow_pickle: bool = False,
    as_array: bool = False,
    **kwargs,
) -> Any:
    """Load data using the file extension in ``path`` to select the format."""
    load_path = _resolve_path(path, root=root, for_save=False)
    suffix = _extension(load_path)
    if suffix not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported load extension '{suffix}'. Supported: {sorted(_SUPPORTED_EXTS)}")
    if not load_path.exists():
        raise FileNotFoundError(load_path)
    if key is not None and suffix not in _HDF5_EXTS and suffix != ".npz":
        raise ValueError("key= is only supported for HDF5 and NPZ inputs.")

    if suffix in _HDF5_EXTS:
        return _load_hdf5(load_path, key=key)
    if suffix in _JSON_EXTS:
        return load_json(load_path)
    if suffix in _JSONL_EXTS:
        return _load_jsonl(load_path)
    if suffix in _CSV_EXTS:
        sep = "\t" if suffix == ".tsv" else ","
        return _read_pandas().read_csv(load_path, sep=sep, **kwargs)
    if suffix == ".npy":
        return np.load(load_path, allow_pickle=allow_pickle, **kwargs)
    if suffix == ".npz":
        with np.load(load_path, allow_pickle=allow_pickle, **kwargs) as loaded:
            if key is not None:
                return loaded[key]
            names = list(loaded.files)
            if names == ["data"]:
                return loaded["data"]
            return {name: loaded[name] for name in names}
    if suffix in _PICKLE_EXTS:
        return read_pickle(load_path)
    if suffix in _XML_EXTS:
        return read_xml(load_path, **kwargs)
    if suffix in _TEXT_EXTS:
        if as_array:
            return np.loadtxt(load_path, **kwargs)
        return load_path.read_text(encoding=kwargs.pop("encoding", "utf-8"))
    if suffix == ".parquet":
        return _read_pandas().read_parquet(load_path, **kwargs)
    if suffix == ".feather":
        return _read_pandas().read_feather(load_path, **kwargs)
    if suffix in _FIGURE_EXTS:
        raise TypeError("Figure/image loading is not supported by ssatk_read; use imageio/PIL/matplotlib directly.")
    raise ValueError(f"Unsupported load extension '{suffix}'.")  # pragma: no cover


def supported_save_formats() -> tuple[str, ...]:
    """Return supported file extensions for :func:`ssatk_save`."""
    return tuple(sorted(_SUPPORTED_EXTS))


def _extension(path: Path) -> str:
    suffix = path.suffix.lower()
    if not suffix:
        raise ValueError("Output path must include a file extension so SSATK can choose a format.")
    return suffix


def _resolve_path(path: str | Path, *, root: str | Path = "auto", for_save: bool) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return ensure_file_parent(raw) if for_save else raw

    if isinstance(root, str) and root in _DATA_ROOTS:
        suffix = raw.suffix.lower()
        if root == "cwd":
            resolved = raw
        elif root == "figures" or (root == "auto" and suffix in _FIGURE_EXTS):
            from ssatk.plots.figpath import ssatk_path

            resolved = Path(ssatk_path(raw))
        else:
            resolved = Path(datapath(raw))
    else:
        parts = safe_relative_parts(raw)
        if not parts:
            raise ValueError("path must include a filename")
        resolved = Path(root).expanduser() / Path(*parts)

    return ensure_file_parent(resolved) if for_save else resolved


def _normalize_hdf5_compression(compression: str | bool | None) -> str | None:
    if compression is True:
        return "gzip"
    if compression is False:
        return None
    return compression


def _hdf5_key_parts(key: str | Path | None, *, default: str | None = None) -> list[str]:
    if key is None:
        if default is None:
            raise ValueError("HDF5 key is required")
        key = default
    key_text = str(key).strip("/")
    parts = key_text.split("/") if key_text else []
    if not parts:
        raise ValueError("HDF5 key cannot be empty")
    for part in parts:
        if part in {"", ".", ".."} or any(ord(char) < 32 for char in part):
            raise ValueError(f"Invalid HDF5 key component: {part!r}")
    return parts


def _nested_for_key(parts: list[str], value: Any) -> dict[str, Any]:
    out: Any = value
    for part in reversed(parts):
        out = {part: out}
    return out


def _validate_mapping_keys(mapping: Mapping[str, Any]) -> None:
    for key, value in mapping.items():
        _hdf5_key_parts(str(key))
        if isinstance(value, Mapping):
            _validate_mapping_keys(value)


def _hdf5_key_exists(path: Path, parts: list[str]) -> bool:
    if not path.exists():
        return False
    import h5py

    with h5py.File(path, "r") as handle:
        return "/".join(parts) in handle


def _delete_hdf5_key(path: Path, parts: list[str]) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        joined = "/".join(parts)
        if joined in handle:
            del handle[joined]


def _save_hdf5(
    data: Any,
    path: Path,
    *,
    key: str | None,
    overwrite: bool,
    compression: str | None,
    compression_opts: int | None,
    pickle_objects: bool,
    metadata: Mapping[str, Any] | None,
) -> None:
    is_mapping = isinstance(data, Mapping)
    if key is None and is_mapping:
        _validate_mapping_keys(data)
        if path.exists() and not overwrite:
            root_keys = [_hdf5_key_parts(str(name)) for name in data]
            conflicts = ["/".join(parts) for parts in root_keys if _hdf5_key_exists(path, parts)]
            if conflicts:
                raise FileExistsError(f"HDF5 key(s) already exist in {path}: {conflicts}")
            mode = "a"
        else:
            mode = "w"
        payload = dict(data)
        default_key = None
    else:
        parts = _hdf5_key_parts(key, default="data")
        if _hdf5_key_exists(path, parts):
            if not overwrite:
                raise FileExistsError(f"HDF5 key already exists in {path}: {'/'.join(parts)}")
            _delete_hdf5_key(path, parts)
        payload = _nested_for_key(parts, data)
        mode = "a"
        default_key = "/".join(parts) if key is None else None

    save_dict_to_hdf5(
        str(path),
        payload,
        mode=mode,
        pickle_objects=pickle_objects,
        compression=compression,
        compression_opts=None if compression is None else compression_opts,
    )
    if metadata or default_key:
        import h5py

        with h5py.File(path, "a") as handle:
            if default_key:
                handle.attrs["__ssatk_default_key__"] = default_key
            if metadata:
                for name, value in metadata.items():
                    handle.attrs[str(name)] = value


def _load_hdf5(path: Path, *, key: str | None) -> Any:
    if key is not None:
        parts = _hdf5_key_parts(key)
        top = parts[0]
        loaded = load_dict_from_hdf5(str(path), keys={top})
        if top not in loaded:
            raise KeyError(key)
        value = loaded[top]
        for part in parts[1:]:
            value = value[part]
        return value

    import h5py

    with h5py.File(path, "r") as handle:
        default_key = handle.attrs.get("__ssatk_default_key__")
    if default_key:
        return _load_hdf5(path, key=str(default_key))
    loaded = load_dict_from_hdf5(str(path))
    if set(loaded) == {"data"}:
        return loaded["data"]
    return loaded


def _to_dataframe(data: Any):
    pd = _read_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()

    try:
        from astropy.table import Table
    except ImportError:
        Table = None
    if Table is not None and isinstance(data, Table):
        return data.to_pandas()

    if isinstance(data, Mapping):
        if all(_is_scalar(value) for value in data.values()):
            return pd.DataFrame([data])
        return pd.DataFrame(data)

    array = np.asarray(data)
    if array.ndim == 0:
        return pd.DataFrame({"value": [array.item()]})
    if array.ndim == 1:
        return pd.DataFrame({"value": array})
    if array.ndim == 2:
        return pd.DataFrame(array)
    raise TypeError(f"Cannot save {type(data)!r} with shape {array.shape} as a table format.")


def _read_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a project dependency
        raise ImportError("pandas is required for tabular SSATK save/load formats") from exc
    return pd


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, bool, type(None), np.generic))


def _save_npy(data: Any, path: Path, *, allow_pickle: bool, **kwargs) -> None:
    array = np.asarray(data)
    if array.dtype == object and not allow_pickle:
        raise TypeError("Refusing to write object dtype .npy without allow_pickle=True.")
    np.save(path, array, allow_pickle=allow_pickle, **kwargs)


def _save_npz(data: Any, path: Path, *, key: str | None, allow_pickle: bool, compressed: bool, **kwargs) -> None:
    arrays = _npz_arrays(data, key=key, allow_pickle=allow_pickle)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **arrays, **kwargs)


def _npz_arrays(data: Any, *, key: str | None, allow_pickle: bool) -> dict[str, np.ndarray]:
    if isinstance(data, Mapping):
        prefix = "/".join(_hdf5_key_parts(key)) if key is not None else ""
        arrays = dict(_flatten_npz_mapping(data, prefix=prefix, allow_pickle=allow_pickle))
    else:
        name = "/".join(_hdf5_key_parts(key, default="data"))
        arrays = {name: _array_for_npz(data, allow_pickle=allow_pickle)}
    if not arrays:
        raise ValueError("Cannot write an empty NPZ archive.")
    return arrays


def _flatten_npz_mapping(mapping: Mapping[str, Any], *, prefix: str, allow_pickle: bool):
    for raw_key, value in mapping.items():
        parts = _hdf5_key_parts(str(raw_key))
        name = "/".join(part for part in [prefix, *parts] if part)
        if isinstance(value, Mapping):
            yield from _flatten_npz_mapping(value, prefix=name, allow_pickle=allow_pickle)
        else:
            yield name, _array_for_npz(value, allow_pickle=allow_pickle)


def _array_for_npz(value: Any, *, allow_pickle: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object and not allow_pickle:
        raise TypeError("Refusing to write object dtype .npz member without allow_pickle=True.")
    return array


def _save_jsonl(data: Any, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    records = _json_records(data)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default) + "\n")


def _load_jsonl(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_records(data: Any) -> list[Any]:
    pd = None
    try:
        pd = _read_pandas()
    except ImportError:  # pragma: no cover
        pass
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, Mapping):
        return [data]
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return data.tolist()
        if data.ndim == 2:
            return [row.tolist() for row in data]
    if isinstance(data, list):
        return data
    raise TypeError("JSON Lines output requires a DataFrame, mapping, ndarray, or list of records.")


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_text(data: Any, path: Path, **kwargs) -> None:
    encoding = kwargs.pop("encoding", "utf-8")
    if isinstance(data, bytes):
        path.write_bytes(data)
    elif isinstance(data, str):
        path.write_text(data, encoding=encoding)
    elif isinstance(data, (list, tuple)) and all(isinstance(item, str) for item in data):
        path.write_text("\n".join(data) + "\n", encoding=encoding)
    else:
        np.savetxt(path, np.asarray(data), **kwargs)


def _save_figure(figure: Any, path: Path, *, overwrite: bool, **kwargs) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    from ssatk.plots.plotutils import figsave

    result = figsave(figure, save_path=path, **kwargs)
    if result is None:
        raise RuntimeError(f"Failed to save figure to {path}")


__all__ = ["ssatk_save", "ssatk_read", "supported_save_formats"]
