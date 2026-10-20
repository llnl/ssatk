import inspect
import pickle
from datetime import datetime
from pathlib import Path
import types
import numpy as np
from .datapath import datapath

__all__ = ["h5cache", "h5load"]

try:
    import h5py
except (ImportError, OSError) as e:
    h5py = None
    _H5PY_IMPORT_ERROR = e


_DEFAULT_EXCLUDE_NAMES = {
    "__name__", "__file__", "__package__", "__spec__", "__builtins__",
    "__loader__", "__cached__", "__doc__", "__annotations__",
}


# ---- key encoding (safe for HDF5 paths) ----
def _enc_key(k):
    # percent-escape '%' first, then '/'
    return k.replace("%", "%25").replace("/", "%2F")


def _dec_key(k):
    return k.replace("%2F", "/").replace("%25", "%")


def _is_scalar_number(x):
    return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_))


def _is_nonsaved_symbol(v):
    """
    Skip symbols that are usually "code" not "data":
      - modules (np, inspect, etc.)
      - functions/methods/builtins
      - classes/types (datetime, Path, custom classes)
    """
    if isinstance(v, types.ModuleType):
        return True

    if isinstance(
        v,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
        ),
    ):
        return True

    if isinstance(v, type):
        return True

    return False


def _write_value(h5group, key, value, *, compression="gzip", compression_opts=4):
    """
    Write a single value under h5group using name=key (encoded).
    For dicts -> subgroup recursion
    For numpy/array-like -> dataset
    For str/bytes -> dataset with attrs
    Fallback -> pickle bytes dataset with attrs
    """
    name = _enc_key(key) if isinstance(key, str) else _enc_key(str(key))

    # nested dict -> subgroup
    if isinstance(value, dict):
        grp = h5group.require_group(name)
        grp.attrs["__kind__"] = "dict"
        for k2, v2 in value.items():
            _write_value(
                grp,
                str(k2),
                v2,
                compression=compression,
                compression_opts=compression_opts,
            )
        return

    # strings
    if isinstance(value, str):
        dt = h5py.string_dtype(encoding="utf-8")
        ds = h5group.create_dataset(name, data=value, dtype=dt)
        ds.attrs["__kind__"] = "str"
        return

    # bytes
    if isinstance(value, (bytes, bytearray)):
        arr = np.frombuffer(bytes(value), dtype=np.uint8)
        ds = h5group.create_dataset(name, data=arr)
        ds.attrs["__kind__"] = "bytes"
        return

    # scalar numbers/bools
    if _is_scalar_number(value):
        ds = h5group.create_dataset(name, data=value)
        ds.attrs["__kind__"] = "scalar"
        return

    # numpy arrays / array-like
    try:
        arr = value if isinstance(value, np.ndarray) else np.asarray(value)

        # object arrays are not reliably HDF5-portable; store via pickle fallback
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            raise TypeError("object arrays stored via pickle fallback")

        ds = h5group.create_dataset(
            name,
            data=arr,
            compression=compression if arr.size > 0 else None,
            compression_opts=compression_opts if arr.size > 0 else None,
        )
        ds.attrs["__kind__"] = "ndarray"
        return
    except Exception:
        pass

    # fallback: pickle
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    arr = np.frombuffer(payload, dtype=np.uint8)
    ds = h5group.create_dataset(name, data=arr)
    ds.attrs["__kind__"] = "pickle"
    ds.attrs["__py_type__"] = str(type(value))
    return


def _read_node(node):
    """
    Read from an h5py Group or Dataset and reconstruct the Python object.
    """
    if isinstance(node, h5py.Group):
        out = {}
        for name, child in node.items():
            out[_dec_key(name)] = _read_node(child)
        return out

    # Dataset
    kind = node.attrs.get("__kind__", "")

    if kind == "str":
        v = node[()]
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
        return v

    if kind == "bytes":
        arr = np.array(node[()], dtype=np.uint8).ravel()
        return arr.tobytes()

    if kind == "scalar":
        v = node[()]
        if isinstance(v, np.generic):
            return v.item()
        return v

    if kind == "ndarray":
        return np.array(node[()])

    if kind == "pickle":
        arr = np.array(node[()], dtype=np.uint8).ravel()
        payload = arr.tobytes()
        return pickle.loads(payload)

    # default (older files): best-effort
    v = node[()]
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def h5cache(
    data=None,
    filename=None,
    *,
    exclude_names=None,
    exclude_private=True,
    include_locals=True,
    only_picklable=False,
    compression="gzip",
    compression_opts=4,
    add_timestamp=True,
):
    """
    Save a dictionary to an HDF5 cache at your datapath() location.

    If data is None:
      - captures caller globals (+ locals if include_locals=True)
      - skips modules/functions/classes (so it's mostly "data variables")

    If data is a dict:
      - saves only that dict (after filtering)

    Values:
      dict -> subgroup recursion
      ndarray/array-like -> dataset (compressed)
      str/bytes/scalars -> dataset
      fallback -> pickled bytes dataset
    """
    if h5py is None:
        raise ImportError(
            f"h5py is required for h5cache() HDF5 caches: {_H5PY_IMPORT_ERROR}"
        )

    if exclude_names is None:
        exclude_names = set(_DEFAULT_EXCLUDE_NAMES)
    else:
        exclude_names = set(exclude_names) | set(_DEFAULT_EXCLUDE_NAMES)

    # collect
    if data is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            raise RuntimeError("h5cache(): could not access caller frame.")
        caller = frame.f_back

        collected = dict(caller.f_globals)
        if include_locals:
            collected.update(caller.f_locals)
        data = collected
    else:
        if not isinstance(data, dict):
            raise TypeError("h5cache(data=...): data must be a dict or None.")

    # filter
    filtered = {}
    for k, v in data.items():
        if not isinstance(k, str):
            k = str(k)

        if k in exclude_names:
            continue
        if exclude_private and k.startswith("_"):
            continue

        # skip code-ish symbols so we cache "variables", not imports/definitions
        if _is_nonsaved_symbol(v):
            continue

        if only_picklable:
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                continue

        filtered[k] = v

    # filename
    if filename is None:
        filename = "workspace_cache"
    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{ts}"

    out = datapath(f"{filename}.h5")

    # write
    with h5py.File(out, "w") as f:
        f.attrs["__ssatk_cache__"] = "1"
        f.attrs["__key_encoding__"] = "percent(%25) and slash(%2F)"
        root = f.require_group("vars")
        root.attrs["__kind__"] = "dict"
        for k, v in filtered.items():
            _write_value(
                root,
                k,
                v,
                compression=compression,
                compression_opts=compression_opts,
            )

    return out


def h5load(name_or_path, *, summary=True, max_items=200):
    """
    Load an h5cache() HDF5 cache and return the reconstructed dictionary.

    - If name_or_path exists as a path, use it.
    - Otherwise treat it as a name under datapath(...).

    If summary=True, prints a path summary of the HDF5 contents.
    """
    if h5py is None:
        raise ImportError(f"h5py is required for h5load(): {_H5PY_IMPORT_ERROR}")

    if not isinstance(name_or_path, (str, Path)):
        raise TypeError("h5load(name_or_path): must be str or pathlib.Path")

    p = Path(name_or_path)

    # If it's not an existing path, resolve under datapath
    if not p.exists():
        s = str(name_or_path)
        if not s.lower().endswith((".h5", ".hdf5", ".hdf")):
            s = s + ".h5"
        p = Path(datapath(s))

    def _print_summary(f):
        print(f"[h5load] file: {str(p)}")
        print("[h5load] tree:")

        n = 0

        def _visit(name, obj):
            nonlocal n
            if n >= max_items:
                return
            # name is path relative to root, e.g. 'vars/x'
            if isinstance(obj, h5py.Group):
                kind = obj.attrs.get("__kind__", "group")
                print(f"  /{name}  <group> kind={kind}  n_children={len(obj)}")
            else:
                kind = obj.attrs.get("__kind__", "dataset")
                shape = getattr(obj, "shape", None)
                dtype = getattr(obj, "dtype", None)
                print(f"  /{name}  <dataset> kind={kind}  shape={shape}  dtype={dtype}")
            n += 1

        f.visititems(_visit)

        if n >= max_items:
            print(f"[h5load] ... truncated at {max_items} items")

    with h5py.File(str(p), "r") as f:
        if summary:
            _print_summary(f)

        if "vars" in f:
            return _read_node(f["vars"])
        return _read_node(f)
