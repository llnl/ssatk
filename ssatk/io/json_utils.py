import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.time import Time
from ssatk._paths import ensure_file_parent


def save_json(filename: str, data) -> None:
    """Save data to a JSON file with support for NumPy, datetime, and Astropy Time."""

    def encode(obj):
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "data": obj.tolist()}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (bytes, bytearray)):
            return {"__type__": "bytes", "data": list(obj)}
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}
        if isinstance(obj, Time):
            # jd1/jd2 round-trip exactly; .isot alone is truncated to
            # millisecond precision (ms-level epochs lose sub-ms accuracy).
            return {
                "__type__": "astropy_time",
                "jd1": obj.jd1,
                "jd2": obj.jd2,
                "scale": obj.scale,
            }
        if isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": list(obj)}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Serialize to a string first so a value that can't be encoded raises
    # before anything on disk is touched, then atomically replace the target
    # file so a failure (or a crash mid-write) never leaves a
    # truncated/invalid file where a good one used to be.
    text = json.dumps(data, default=encode, indent=4)
    path = ensure_file_parent(filename)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def load_json(filename: str):
    """Load data from a JSON file and decode special types."""

    def decode(obj):
        if not isinstance(obj, dict) or "__type__" not in obj:
            return obj
        obj_type = obj["__type__"]
        if obj_type == "ndarray":
            return np.array(obj["data"])
        if obj_type == "datetime":
            return datetime.fromisoformat(obj["data"])
        if obj_type == "astropy_time":
            if "jd1" in obj and "jd2" in obj:
                return Time(obj["jd1"], obj["jd2"], format="jd", scale=obj["scale"])
            return Time(obj["data"], scale=obj["scale"])  # older files: isot-only
        if obj_type == "bytes":
            return bytes(obj["data"])
        if obj_type == "set":
            return set(obj["data"])
        if obj_type == "tuple":
            return tuple(obj["data"])
        return obj

    with Path(filename).open("r", encoding="utf-8") as f:
        return json.load(f, object_hook=decode)


def read_json(filename: str):
    return load_json(filename)


def append_json(filename: str, new_data) -> None:
    """Append data into an existing JSON file.
    
    - If root is a dict:
        * For each key in new_data:
            - If key exists and is a list, extend or append.
            - If key exists and is not a list, convert to list and append.
            - If key does not exist, add it.
    - If root is a list, append or extend.
    - If file does not exist, save new_data as-is.
    - If types mismatch, wrap both into a list.
    """
    try:
        data = load_json(filename)
    except FileNotFoundError:
        save_json(filename, new_data)
        return

    if isinstance(data, dict) and isinstance(new_data, dict):
        for key, value in new_data.items():
            if key in data:
                if isinstance(data[key], list):
                    if isinstance(value, list):
                        data[key].extend(value)
                    else:
                        data[key].append(value)
                else:
                    data[key] = [data[key], value]
            else:
                data[key] = value
    elif isinstance(data, list):
        if isinstance(new_data, list):
            data.extend(new_data)
        else:
            data.append(new_data)
    else:
        data = [data, new_data]

    save_json(filename, data)
