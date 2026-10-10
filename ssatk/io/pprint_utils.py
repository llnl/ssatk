# my_utils/pretty.py
# -----------------------------------------------------------------------------
# Pretty-print helpers with HDF5 and dict summary support.
#
# - NumPy-aware pretty printer for general Python objects
# - HDF5 "summary of everything" view: lists every group and dataset,
#   showing group child counts and dataset shape/dtype, plus attribute keys,
#   and a small head/tail preview of dataset values.
# - Dict/mapping summary that previews values, arrays, and nested dicts.
#
# Usage
# -----
# >>> from my_utils.pretty import pprint
# >>> pprint(my_dict)
# >>> pprint("data/sim_results.h5")            # summarize entire file
# >>> with h5py.File("data.h5") as f:
# ...     pprint(f["/subgroup"])               # summarize subtree
#
# Notes
# -----
# - h5py is optional; if not installed, HDF5 features are disabled.
# - ASCII only; no fancy unicode characters.
# -----------------------------------------------------------------------------

import pathlib
import numpy as np
from pprint import pprint as _pprint

try:
    import h5py
except ImportError:  # make h5py optional
    h5py = None


# --------------------------- NumPy printing helpers --------------------------

_np_defaults = dict(
    edgeitems=3,      # values kept from each end when arrays are long
    threshold=20,     # total elements before truncation kicks in
    linewidth=120,    # wrap arrays nicely in most terminals
    precision=6,      # float digits; set None for NumPy default
    suppress=True,    # 1e-10 -> 0.000000
)


def _print_numpy(obj):
    """Pretty-print any Python object, truncating long NumPy arrays."""
    # np.printoptions supports linewidth; np.array2string below uses max_line_width.
    with np.printoptions(**_np_defaults):
        _pprint(obj)


# ---------------------------- HDF5 summary helpers ---------------------------

def _open_hdf5_like(obj):
    """
    Normalize an HDF5-like input into an open handle and a must_close flag.

    Returns
    -------
    handle : h5py.File | h5py.Group | h5py.Dataset
    must_close : bool
    """
    if h5py is None:
        raise RuntimeError("h5py not installed; cannot work with HDF5")

    if isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)):
        return obj, False

    if isinstance(obj, (str, pathlib.Path)):
        # Try to open as HDF5; let OSError bubble up to caller to decide fallback.
        try:
            fh = h5py.File(obj, "r")
        except Exception as e:
            raise OSError(f"Not an HDF5 file or cannot open: {obj}") from e
        return fh, True

    raise TypeError("Object is not an HDF5 file/group/dataset or path to one")


def _indent_for_name(name: str, indent_per_level: int = 2) -> str:
    """
    Compute indentation based on HDF5 path depth.
    Root '/' -> depth 0, '/grp' -> depth 1, '/grp/sub' -> depth 2, etc.
    """
    if name == "/":
        depth = 0
    else:
        depth = max(0, name.count("/") - 1)
    return " " * (indent_per_level * depth)


def _summarize_group(g: "h5py.Group") -> str:
    """
    Summarize a group by counting immediate subgroups and datasets.
    """
    n_groups = 0
    n_dsets = 0
    for k in g.keys():
        try:
            item = g[k]  # does not load data; returns handle
            if isinstance(item, h5py.Group):
                n_groups += 1
            elif isinstance(item, h5py.Dataset):
                n_dsets += 1
        except Exception:
            # If access fails, skip counting that child
            continue
    return f"{g.name or '/'} (Group: {n_groups} groups, {n_dsets} datasets)"


def _array2string(a: np.ndarray) -> str:
    """Format a small numpy array to a compact one-line string."""
    return np.array2string(
        np.asarray(a),
        max_line_width=_np_defaults["linewidth"],
        precision=_np_defaults["precision"],
        suppress_small=_np_defaults["suppress"],
        separator=", ",
        threshold=1_000_000,  # avoid numpy's own ellipsis for our tiny previews
    )


def _preview_dataset_values(d: "h5py.Dataset", head: int = 3, tail: int = 3, small_limit: int = 20) -> str:
    """
    Return a small string preview of dataset values:
    - If scalar: show the singleton value.
    - If total size <= small_limit: show the entire flattened array.
    - Else: show first `head` and last `tail` elements with "..., " in between.
    Notes:
    - For N-D datasets, preview is taken along the first axis with minimal slicing
      to avoid loading large arrays into memory.
    """
    try:
        # Scalar dataset
        if d.shape == ():
            val = d[()]  # scalar
            return _array2string(np.array([val]))[1:-1]  # strip brackets

        # Guard odd shapes
        if not d.shape or d.ndim == 0:
            return ""

        # Small datasets: read everything (but keep it small)
        if d.size <= small_limit:
            data = d[...]
            flat = np.ravel(data)
            return _array2string(flat)[1:-1]  # strip brackets

        # Large datasets
        n0 = d.shape[0]
        if n0 <= 0:
            return ""

        # Build head and tail slices along the first axis while keeping other dims small
        rest_slices = tuple(slice(0, 1) for _ in range(d.ndim - 1))

        h = min(max(head, 0), n0)
        t = 0
        if n0 > h and tail > 0:
            t = min(tail, max(n0 - h, 0))

        head_slice = (slice(0, h),) + rest_slices
        head_block = d[head_slice]
        head_vals = np.ravel(head_block)

        tail_vals = np.array([], dtype=d.dtype)
        if t > 0:
            tail_slice = (slice(n0 - t, n0),) + rest_slices
            tail_block = d[tail_slice]
            tail_vals = np.ravel(tail_block)

        if head_vals.size + tail_vals.size <= small_limit and t == 0:
            return _array2string(head_vals)[1:-1]

        head_str = _array2string(head_vals).strip()
        tail_str = _array2string(tail_vals).strip() if tail_vals.size else ""
        if head_str.startswith("[") and head_str.endswith("]"):
            head_str = head_str[1:-1]
        if tail_str.startswith("[") and tail_str.endswith("]"):
            tail_str = tail_str[1:-1]

        if tail_str:
            return f"{head_str}, ..., {tail_str}"
        else:
            return head_str
    except Exception as e:
        return f"<preview unavailable: {type(e).__name__}>"


def _summarize_dataset(d: "h5py.Dataset") -> str:
    """
    Summarize a dataset by shape and dtype, plus a small head/tail preview.
    """
    try:
        shape = "x".join(str(s) for s in d.shape)
    except Exception:
        shape = "unknown"
    try:
        dtype = str(d.dtype)
    except Exception:
        dtype = "unknown"

    preview = _preview_dataset_values(d)
    if preview:
        return f"{d.name} (Dataset, shape={shape}, dtype={dtype}) preview=[{preview}]"
    else:
        return f"{d.name} (Dataset, shape={shape}, dtype={dtype})"


def _summarize_attrs(obj) -> str:
    """
    Return a one-line summary of attribute keys, if any.
    """
    try:
        keys = sorted(list(obj.attrs.keys()), key=str)
    except Exception:
        keys = []
    if not keys:
        return ""
    return f"@attrs: {keys}"


def _print_hdf5_summary(obj):
    """
    Print a summary of all levels under the given HDF5 object.

    The summary includes every group and dataset reachable from 'obj',
    one line per object, plus a line of attribute keys when present,
    and a small value preview for datasets.
    Traversal uses h5py visititems to cover the full subtree.
    """
    handle, must_close = _open_hdf5_like(obj)
    try:
        # Print the root (or starting node) first
        if isinstance(handle, (h5py.File, h5py.Group)):
            root_group = handle if isinstance(handle, h5py.Group) else handle["/"]
            head_line = _summarize_group(root_group)
            print(head_line)
            attrs_line = _summarize_attrs(handle)
            if attrs_line:
                print(_indent_for_name(root_group.name) + "  " + attrs_line)
        elif isinstance(handle, h5py.Dataset):
            print(_summarize_dataset(handle))
            attrs_line = _summarize_attrs(handle)
            if attrs_line:
                print(_indent_for_name(handle.name) + "  " + attrs_line)

        # Visitor to walk subtree
        def _visitor(name, item):
            if item is handle:
                return
            pad = _indent_for_name(item.name)
            if isinstance(item, h5py.Group):
                print(pad + _summarize_group(item))
                attrs_line = _summarize_attrs(item)
                if attrs_line:
                    print(pad + "  " + attrs_line)
            elif isinstance(item, h5py.Dataset):
                print(pad + _summarize_dataset(item))
                attrs_line = _summarize_attrs(item)
                if attrs_line:
                    print(pad + "  " + attrs_line)
            else:
                try:
                    print(pad + f"{item.name} (unsupported HDF5 type: {type(item).__name__})")
                except Exception:
                    print(pad + f"(unsupported HDF5 type: {type(item).__name__})")

        if isinstance(handle, (h5py.File, h5py.Group)):
            # Restrict traversal to the starting group for Group handles.
            start = handle if isinstance(handle, h5py.Group) else handle["/"]
            start.visititems(_visitor)

    finally:
        if must_close:
            try:
                handle.close()
            except Exception:
                pass


from collections.abc import Mapping, Sequence

# --------------------------- Dict summary helpers ----------------------------

_dict_defaults = dict(
    max_items=20,      # maximum keys shown per mapping
    max_depth=2,       # recursive depth for nested dicts (logical levels)
    key_sort=True,     # sort keys by str(key)
    seq_head=3,        # preview count for sequence head
    seq_tail=3,        # preview count for sequence tail
    str_max=80,        # max characters for string previews
)

def _clip(s: str, n: int) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    if n <= 3:
        return s[:n]
    return s[: n - 3] + "..."

def _array_preview(a: np.ndarray, small_limit: int = 20) -> str:
    """One-line ndarray preview with shape/dtype and head...tail values."""
    a = np.asarray(a)
    shape = "x".join(str(s) for s in a.shape) if a.shape else "scalar"
    dtype = str(a.dtype)
    # Scalar
    if a.shape == ():
        val = _array2string(a.reshape(1,))[1:-1]
        return f"ndarray(shape={shape}, dtype={dtype}) value={val}"
    flat = a.ravel()
    if flat.size <= small_limit:
        vals = _array2string(flat)[1:-1]
        return f"ndarray(shape={shape}, dtype={dtype}) values=[{vals}]"
    head = flat[: _dict_defaults["seq_head"]]
    tail = flat[-_dict_defaults["seq_tail"] :] if _dict_defaults["seq_tail"] > 0 else np.array([], dtype=a.dtype)
    head_s = _array2string(head)[1:-1]
    tail_s = _array2string(tail)[1:-1] if tail.size else ""
    if tail_s:
        pv = f"{head_s}, ..., {tail_s}"
    else:
        pv = head_s
    return f"ndarray(shape={shape}, dtype={dtype}) preview=[{pv}]"

def _seq_preview(seq: Sequence) -> str:
    """One-line preview for lists/tuples (but not str/bytes)."""
    try:
        n = len(seq)
    except Exception:
        return f"{type(seq).__name__}"
    head_n = min(_dict_defaults["seq_head"], n)
    tail_n = min(_dict_defaults["seq_tail"], max(n - head_n, 0))

    # Use index access to avoid copying whole sequences
    head_vals = [_short_value(seq[i]) for i in range(head_n)]
    tail_vals = [_short_value(seq[n - tail_n + i]) for i in range(tail_n)] if tail_n else []

    parts = head_vals + (["..."] if tail_vals else []) + tail_vals
    inner = ", ".join(parts) if parts else ""
    return f"{type(seq).__name__}(len={n}) [{inner}]"

def _short_value(v) -> str:
    """Compact single-line description for a value."""
    if isinstance(v, str):
        return '"' + _clip(v, _dict_defaults["str_max"]) + '"'
    if isinstance(v, (bytes, bytearray)):
        return f"bytes(len={len(v)})"
    if isinstance(v, np.ndarray):
        return _array_preview(v)
    if isinstance(v, Mapping):
        try:
            n = len(v)
        except Exception:
            n = "?"
        return f"dict({n} keys)"
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return _seq_preview(v)
    if v is None:
        return "None"
    # Numbers and numpy scalars
    if isinstance(v, (np.number, int, float, complex, bool)):
        return _array2string(np.array([v]))[1:-1]
    # Fallback to type name
    return f"{type(v).__name__}"

def _indent(level: int) -> str:
    return " " * (2 * level)

def _print_dict_summary(d: Mapping, depth: int = 0, seen_ids=None):
    """
    Print a structured summary of a dictionary (recurses into nested dicts).
    Shows up to max_items per level, sorted by key repr if key_sort is True.
    depth is the logical indent level; each visual level indents by 2 spaces.
    """
    if seen_ids is None:
        seen_ids = set()

    try:
        nkeys = len(d)
    except Exception:
        nkeys = "?"
    print(_indent(depth) + f"dict ({nkeys} keys)")

    # Prevent cycles
    if id(d) in seen_ids:
        print(_indent(depth) + "  <cycle>")
        return
    seen_ids.add(id(d))

    # Stop if logical depth exceeded
    if depth // 2 >= _dict_defaults["max_depth"]:
        return

    # Choose and order keys
    try:
        keys = list(d.keys())
    except Exception:
        keys = []
    if _dict_defaults["key_sort"]:
        try:
            keys.sort(key=lambda k: str(k))
        except Exception:
            pass

    limit = min(len(keys), _dict_defaults["max_items"])
    for k in keys[:limit]:
        v = d[k]
        kstr = _clip(k, 40)
        line = _indent(depth + 1) + f"[{kstr}] -> {_short_value(v)}"
        print(line)
        if isinstance(v, Mapping):
            _print_dict_summary(v, depth + 2, seen_ids=seen_ids)

    if len(keys) > limit:
        print(_indent(depth + 1) + f"... {len(keys) - limit} more keys omitted")

# ------------------------------ public pprint --------------------------------

def pprint(obj):
    """
    Pretty-print any object.

    If 'obj' is (or points to) an HDF5 file | group | dataset:
        Print a summary of all levels reachable from that node.
    If 'obj' is a dict-like mapping:
        Print a compact, recursive summary of keys and values.
    Otherwise:
        Fall back to NumPy-aware pretty printer for general Python objects.
    """
    # HDF5 summary if available and applicable
    if h5py is not None:
        try:
            if isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)) or isinstance(obj, (str, pathlib.Path)):
                _print_hdf5_summary(obj)
                return
        except (TypeError, OSError, RuntimeError):
            pass

    # Dict summary
    if isinstance(obj, Mapping):
        _print_dict_summary(obj)
        return

    # Default NumPy-aware pretty print
    _print_numpy(obj)
