# json_hdf5.py
# JSON <-> HDF5 converter (dict/list/str/int/float/bool/null)
# Requires: h5py, numpy

import json
import string

import h5py
import numpy as np
from ssatk._paths import ensure_file_parent

# ------------------------ Name encoding (reversible) ------------------------

_SAFE_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

def _percent_encode_name(name):
    """
    Encode an arbitrary JSON key to a valid HDF5 path component using percent-utf8.
    Reversible with _percent_decode_name.
    """
    if not isinstance(name, str):
        name = str(name)
    b = name.encode("utf-8", errors="strict")
    out = []
    for ch in b:
        if ch in _SAFE_CHARS:
            out.append(chr(ch))
        else:
            out.append("%%%02X" % ch)
    return "".join(out)

def _percent_decode_name(encoded):
    """
    Reverse of _percent_encode_name.
    """
    out = bytearray()
    i = 0
    s = encoded
    while i < len(s):
        if s[i] == "%" and i + 2 < len(s) and all(ch in string.hexdigits for ch in s[i+1:i+3]):
            out.append(int(s[i+1:i+3], 16))
            i += 3
        else:
            out.append(ord(s[i]))
            i += 1
    return out.decode("utf-8")

# ------------------------ Writers ------------------------

def _write_json_node(h, name, obj):
    """
    Write a JSON value under group h with child name.
    Returns the created object (Group or Dataset).
    """
    if isinstance(obj, dict):
        g = h.create_group(name)
        g.attrs["kind"] = np.bytes_("dict")
        # Preserve key order (Python 3.7+ dicts are ordered)
        for k, v in obj.items():
            child = _percent_encode_name(k)
            _write_json_node(g, child, v)
        return g

    if isinstance(obj, list):
        g = h.create_group(name)
        g.attrs["kind"] = np.bytes_("list")
        g.attrs["length"] = np.int64(len(obj))
        for i, v in enumerate(obj):
            _write_json_node(g, str(i), v)
        return g

    # Scalars
    if obj is None:
        g = h.create_group(name)
        g.attrs["kind"] = np.bytes_("none")
        return g

    if isinstance(obj, bool):
        d = h.create_dataset(name, data=np.bool_(obj))
        d.attrs["kind"] = np.bytes_("bool")
        return d

    if isinstance(obj, int) and not isinstance(obj, bool):
        d = h.create_dataset(name, data=np.int64(obj))
        d.attrs["kind"] = np.bytes_("int")
        return d

    if isinstance(obj, float):
        d = h.create_dataset(name, data=np.float64(obj))
        d.attrs["kind"] = np.bytes_("float")
        return d

    if isinstance(obj, str):
        dt = h5py.string_dtype(encoding="utf-8")
        d = h.create_dataset(name, data=np.array(obj, dtype=dt))
        d.attrs["kind"] = np.bytes_("str")
        return d

    # Fallback: store as JSON string
    dt = h5py.string_dtype(encoding="utf-8")
    d = h.create_dataset(name, data=np.array(json.dumps(obj), dtype=dt))
    d.attrs["kind"] = np.bytes_("json_blob")
    return d

def json_to_hdf5(json_obj, h5_path, root="/"):
    """
    Write a JSON-serializable object to HDF5 file at h5_path.
    """
    h5_path = ensure_file_parent(h5_path)
    with h5py.File(h5_path, "w") as f:
        f.attrs["format"] = np.bytes_("json-hdf5")
        f.attrs["name_encoding"] = np.bytes_("percent-utf8")
        _write_json_node(f, root.strip("/"), json_obj)

def json_file_to_hdf5(json_path, h5_path, root="/"):
    with open(json_path, "r", encoding="utf-8") as fp:
        obj = json.load(fp)
    json_to_hdf5(obj, h5_path, root=root)

# ------------------------ Readers ------------------------

def _read_json_node(hobj):
    """
    Read a JSON value from an HDF5 object (Group or Dataset).
    """
    kind = None
    if "kind" in hobj.attrs:
        v = hobj.attrs["kind"]
        if isinstance(v, bytes):
            kind = v.decode("utf-8")
        elif isinstance(v, np.ndarray) and v.dtype.kind == "S":
            kind = v.astype(str)
        else:
            kind = str(v)
    else:
        # Heuristic: datasets without kind -> try best-effort
        if isinstance(hobj, h5py.Dataset):
            data = hobj[()]
            if isinstance(data, (bytes, np.bytes_)):
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    return data.decode("utf-8", errors="replace")
            if isinstance(data, np.ndarray) and data.shape == ():
                return data.item()
            return data
        # Groups without kind: assume dict
        kind = "dict"

    if isinstance(hobj, h5py.Dataset):
        if kind == "str":
            val = hobj.asstr()[()]
            return val
        if kind == "bool":
            return bool(hobj[()].item())
        if kind == "int":
            return int(hobj[()].item())
        if kind == "float":
            return float(hobj[()].item())
        if kind == "json_blob":
            s = hobj.asstr()[()]
            return json.loads(s)
        # Fallback
        data = hobj[()]
        if isinstance(data, np.ndarray) and data.shape == ():
            return data.item()
        return data

    # Group kinds
    if kind == "none":
        return None

    if kind == "list":
        # Reconstruct in order 0..length-1 if available, else numeric sort
        length = int(hobj.attrs["length"]) if "length" in hobj.attrs else None
        if length is not None:
            out = []
            for i in range(length):
                child = hobj[str(i)]
                out.append(_read_json_node(child))
            return out
        # Fallback: numeric sort of keys
        items = []
        for k in hobj.keys():
            try:
                idx = int(k)
            except ValueError:
                idx = None
            items.append((idx, k))
        items.sort(key=lambda t: (t[0] is None, t[0], t[1]))
        return [_read_json_node(hobj[k]) for _, k in items]

    if kind == "dict":
        d = {}
        for k in hobj.keys():
            decoded = _percent_decode_name(k)
            d[decoded] = _read_json_node(hobj[k])
        return d

    # Unknown kind: attempt best-effort
    return {k: _read_json_node(hobj[k]) for k in hobj.keys()}

def hdf5_to_json(h5_path, root="/"):
    """
    Read HDF5 file and return a JSON-serializable Python object from the given root node.
    """
    with h5py.File(h5_path, "r") as f:
        node_name = root.strip("/")
        if node_name in f:
            return _read_json_node(f[node_name])
        # If root is top-level and not present, accept the only top-level object
        keys = list(f.keys())
        if len(keys) == 1:
            return _read_json_node(f[keys[0]])
        raise KeyError("Root node not found. Available: %r" % keys)

def hdf5_file_to_json(h5_path, json_path, root="/", pretty=True):
    obj = hdf5_to_json(h5_path, root=root)
    json_path = ensure_file_parent(json_path)
    with json_path.open("w", encoding="utf-8") as fp:
        if pretty:
            json.dump(obj, fp, ensure_ascii=False, indent=2, sort_keys=False)
        else:
            json.dump(obj, fp, ensure_ascii=False, separators=(",", ":"))
