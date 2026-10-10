# flake8: noqa: E501
# (The embedded HTML/CSS template below contains inherently long lines that
#  cannot be wrapped without corrupting the generated markup.)
"""Build the self-contained satellite viewer HTML.

Run:  python assemble.py

Inlines scene.js, the vendored libraries (three.min.js and satellite.min.js) and
the four Earth textures into a single standalone HTML file. The texture images
come from the external core data package, not from this source repository.

Layout-agnostic on purpose: inputs are looked up next to this script first,
then in an optional assets/ subfolder, then one directory up. That means the
files can sit flat in an existing package folder or in their own subdirectory,
whichever suits the repo -- no particular directory structure is required.
"""

import base64
import csv
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO

import numpy as np

from ssatk.data import DataPackageNotFoundError, DataResourceNotFoundError, read_data_binary

HERE = os.path.dirname(os.path.abspath(__file__))          # .../ssatk/plots
PKG_ROOT = os.path.dirname(HERE)                           # .../ssatk
REPO_ROOT = os.path.dirname(PKG_ROOT)                      # repository root

# Candidate directories for inputs, in priority order. The code files are
# expected to sit beside this script; the texture archive may live in a
# dedicated data directory instead, so several conventional locations are
# checked. Set SSAPY_VIEWER_DATA to override with an explicit path.
SEARCH_DIRS = [
    d for d in [
        os.environ.get("SSAPY_VIEWER_DATA"),   # explicit override
        HERE,                                  # ssatk/plots/
        os.path.join(PKG_ROOT, "data"),        # ssatk/data/
        os.path.join(REPO_ROOT, "data"),       # <repo>/data/
        os.path.join(HERE, "assets"),          # ssatk/plots/assets/
        PKG_ROOT,                              # ssatk/
        REPO_ROOT,                             # <repo>/
    ] if d
]


def find_input(filename):
    """Locate an input file, or fail saying where we looked."""
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    looked = "\n  ".join(SEARCH_DIRS)
    raise FileNotFoundError(
        "ERROR: could not find required input '{}'.\n"
        "Looked in:\n  {}\n"
        "Place it in one of those directories and re-run."
        .format(filename, looked)
    )


def text(filename):
    with open(find_input(filename), "r", encoding="utf-8") as f:
        return f.read()


def load_textures():
    """Return base64 strings for the four Earth textures.

    Textures ship as individual files in the core data package. This avoids committing a
    duplicated texture archive to SSATK while
    still allowing installed users to build a self-contained HTML viewer.
    """
    files = {
        "day": "earth_day_2048.jpg",
        "night": "earth_night_2048.jpg",
        "specular": "earth_specular_2048.jpg",
        "clouds": "earth_clouds_2048.png",
    }
    return {
        key: base64.b64encode(_read_texture_binary(filename, key)).decode("ascii")
        for key, filename in files.items()
    }


def _load_ssapy_plot_constants():
    """Return the physical values used by the browser scene from SSAPy."""
    from ssapy import constants as ssapy_constants
    from ssatk import constants as toolkit_constants

    def metres(name, toolkit_name):
        value = getattr(ssapy_constants, name, None)
        if value is None:
            value = getattr(toolkit_constants, toolkit_name) * 1000.0
        return float(value) / 1000.0

    return {
        "R_EARTH_KM": metres("WGS84_EARTH_RADIUS", "WGS84_A_KM"),
        "R_MOON_KM": metres("MOON_RADIUS", "MOON_RADIUS_KM"),
        "R_SUN_KM": metres("SUN_RADIUS", "SUN_RADIUS_KM"),
        "AU_KM": metres("AU", "AU_KM"),
        "MOON_MEAN_DISTANCE_KM": metres("LD", "LD_KM"),
        "MU_EARTH_KM3_S2": float(
            getattr(ssapy_constants, "EARTH_MU", toolkit_constants.EARTH_MU)
        ) / 1000.0**3,
    }


def load_star_catalog(mag_limit=6.5, when=None):
    """Return the real catalog sky as compact, JSON-ready arrays."""
    from ssatk.plots.starfield import star_directions

    when = when or datetime.now(timezone.utc)
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    # The viewer scene is TEME (SGP4 output, Earth turned by GMST).
    stars = star_directions(mag_limit=mag_limit, when=when, frame="teme")
    if stars is None:
        raise FileNotFoundError(
            "Accurate satellite-viewer stars require bright_stars.csv from "
            "the appropriate ssatk-data-* package."
        )

    vectors, magnitudes, colors = (np.asarray(value, dtype=float) for value in stars)
    rotation = _gcrf_to_teme_rotation(when)
    vectors = vectors @ rotation.T
    brightness = np.clip(1.60 - 0.15 * magnitudes, 0.16, 1.0)
    sizes = np.clip(4.8 - 0.48 * magnitudes, 1.0, 6.5)
    return {
        "epoch": when.isoformat(timespec="seconds") + "Z",
        "frame": "teme-of-date",
        "v": np.round(vectors, 7).ravel().tolist(),
        "c": np.round(np.clip(colors * brightness[:, None], 0, 1), 3).ravel().tolist(),
        "s": np.round(sizes, 2).tolist(),
    }


def _gcrf_to_teme_rotation(when):
    from astropy.time import Time
    from ssapy.utils import gcrf_to_teme

    return np.asarray(gcrf_to_teme(Time(when, scale="utc")), dtype=float)


def _vector_track_list(values, name):
    """Normalize one vector track or a list of ``(N, 3)`` vector tracks."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim <= 2:
        arrays = [array]
    elif array is not None and array.ndim == 3:
        arrays = list(array)
    elif isinstance(values, (list, tuple)):
        arrays = [np.asarray(item, dtype=float) for item in values]
    else:
        raise ValueError(f"{name} must be a vector or array of shape (N, 3)")

    normalized = []
    for index, item in enumerate(arrays):
        item = np.asarray(item, dtype=float)
        if item.ndim == 1 and item.size == 3:
            item = item.reshape(1, 3)
        if item.ndim != 2 or item.shape[1] != 3:
            raise ValueError(f"{name}[{index}] must have shape (N, 3)")
        normalized.append(item)
    return normalized


def _quaternion_track_list(values):
    """Normalize one quaternion track or a list of ``(N, 4)`` tracks."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim <= 2:
        arrays = [array]
    elif array is not None and array.ndim == 3:
        arrays = list(array)
    elif isinstance(values, (list, tuple)):
        arrays = [np.asarray(item, dtype=float) for item in values]
    else:
        raise ValueError("q must be a quaternion or array of shape (N, 4)")

    normalized = []
    for index, item in enumerate(arrays):
        item = np.asarray(item, dtype=float)
        if item.ndim == 1 and item.size == 4:
            item = item.reshape(1, 4)
        if item.ndim != 2 or item.shape[1] != 4:
            raise ValueError(f"q[{index}] must have shape (N, 4)")
        normalized.append(item)
    return normalized


def _time_track_list(t):
    """Convert SSAPy GPS times or Astropy times to Unix seconds."""
    from astropy.time import Time

    def convert(values):
        if hasattr(values, "unix"):
            return np.asarray(values.unix, dtype=float)
        raw_values = np.asarray(values)
        if np.issubdtype(raw_values.dtype, np.number):
            return np.asarray(Time(raw_values, format="gps").unix, dtype=float)
        return np.asarray(Time(values).unix, dtype=float)

    if hasattr(t, "unix"):
        values = convert(t)
    else:
        # Recent NumPy versions reject ragged lists instead of creating an
        # object array. Preserve nested entries as independent tracks while
        # keeping a flat list of scalar Astropy Times as one track.
        def is_nested(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                return True
            return hasattr(item, "unix") and np.asarray(item.unix).ndim > 0

        ragged = isinstance(t, (list, tuple)) and any(is_nested(item) for item in t)
        if ragged:
            return [convert(item).reshape(-1) for item in t]
        values = convert(t)
    if values.ndim <= 1:
        return [np.atleast_1d(values)]
    return [np.asarray(row, dtype=float).reshape(-1) for row in values]


def _gcrf_state_vectors_to_teme(r, v, t, q=None):
    """Rotate SSAPy GCRF state samples into the viewer's TEME frame."""
    from astropy.time import Time
    from ssapy.utils import gcrf_to_teme
    from ssatk.coordinates.attitude import quaternion_from_matrix, quaternion_multiply

    r_tracks = _vector_track_list(r, "r")
    v_tracks = _vector_track_list(v, "v")
    t_tracks = _time_track_list(t)
    q_tracks = _quaternion_track_list(q) if q is not None else [None] * len(r_tracks)
    if len(v_tracks) != len(r_tracks) or len(t_tracks) != len(r_tracks):
        raise ValueError("r, v, and t must contain the same number of tracks")
    if len(q_tracks) == 1 and len(r_tracks) > 1:
        q_tracks *= len(r_tracks)
    if len(q_tracks) != len(r_tracks):
        raise ValueError("q must contain the same number of tracks as r")

    rotated_r, rotated_v, rotated_q = [], [], []
    for ri, vi, ti, qi in zip(r_tracks, v_tracks, t_tracks, q_tracks):
        if ri.shape != vi.shape or len(ti) != len(ri):
            raise ValueError("r, v, and t lengths do not match")
        matrices = np.asarray(gcrf_to_teme(Time(ti, format="unix")), dtype=float)
        if matrices.ndim == 2:
            matrices = matrices[None, ...]
        rotated_r.append(np.einsum("nij,nj->ni", matrices, ri))
        rotated_v.append(np.einsum("nij,nj->ni", matrices, vi))
        if qi is None:
            rotated_q.append(None)
            continue
        if qi.shape[0] == 1 and len(ri) > 1:
            qi = np.repeat(qi, len(ri), axis=0)
        if qi.shape != (len(ri), 4):
            raise ValueError("q and r lengths do not match")
        rotated_q.append(np.asarray([
            quaternion_multiply(quaternion_from_matrix(matrix), quaternion)
            for matrix, quaternion in zip(matrices, qi)
        ]))
    return rotated_r, rotated_v, t, (rotated_q if q is not None else None)


def _prepare_state_vectors(r, v, t, labels=None, units="m", q=None):
    """Build JSON-ready state-vector tracks for the browser viewer."""
    if units not in {"m", "km"}:
        raise ValueError("units must be 'm' or 'km'; SSAPy.rv output uses 'm'")
    if r is None or v is None or t is None:
        raise ValueError("r, v, and t are required for state-vector viewer input")

    r_tracks = _vector_track_list(r, "r")
    v_tracks = _vector_track_list(v, "v")
    t_tracks = _time_track_list(t)
    if len(v_tracks) != len(r_tracks):
        raise ValueError("r and v must contain the same number of tracks")
    if len(t_tracks) == 1 and len(r_tracks) > 1:
        t_tracks = t_tracks * len(r_tracks)
    if len(t_tracks) != len(r_tracks):
        raise ValueError("r and t must contain the same number of tracks")
    if labels is None:
        labels = ["State vector track"] * len(r_tracks)
    if len(labels) != len(r_tracks):
        raise ValueError("labels must match the number of state-vector tracks")
    if q is None:
        q_tracks = [None] * len(r_tracks)
    else:
        q_tracks = _quaternion_track_list(q)
        if len(q_tracks) == 1 and len(r_tracks) > 1:
            q_tracks = q_tracks * len(r_tracks)
        if len(q_tracks) != len(r_tracks):
            raise ValueError("q must contain the same number of tracks as r")

    scale = 1.0 if units == "km" else 1.0e-3
    tracks = []
    for index, (ri, vi, ti, qi) in enumerate(zip(r_tracks, v_tracks, t_tracks, q_tracks)):
        if ri.shape != vi.shape or len(ti) != len(ri):
            raise ValueError(f"r, v, and t lengths do not match for track {index}")
        if len(ti) == 0 or not np.all(np.isfinite(ri)) or not np.all(np.isfinite(vi)):
            raise ValueError(f"state-vector track {index} must contain finite samples")
        if not np.all(np.isfinite(ti)) or (len(ti) > 1 and np.any(np.diff(ti) <= 0.0)):
            raise ValueError(f"t must be finite and strictly increasing for track {index}")
        track = {
            "name": str(labels[index]),
            "r": (ri * scale).tolist(),
            "v": (vi * scale).tolist(),
            "t": (ti * 1000.0).tolist(),
        }
        if qi is not None:
            if qi.shape[0] == 1 and len(ri) > 1:
                qi = np.repeat(qi, len(ri), axis=0)
            if qi.shape != (len(ri), 4):
                raise ValueError(f"q and r lengths do not match for track {index}")
            norms = np.linalg.norm(qi, axis=1)
            if not np.all(np.isfinite(qi)) or np.any(norms <= np.finfo(float).tiny):
                raise ValueError(f"quaternion track {index} must contain finite, non-zero samples")
            track["q"] = (qi / norms[:, None]).tolist()
        tracks.append(track)
    return tracks


def _propagator_name(propagator):
    """Return the browser-supported propagation model name."""
    if propagator is None:
        return "sgp4"
    if isinstance(propagator, str):
        name = propagator
    else:
        name = getattr(propagator, "name", None) or getattr(propagator, "kind", None)
        name = name or type(propagator).__name__
    normalized = str(name).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"sgp4", "sgp4propagator"}:
        return "sgp4"
    raise ValueError(
        "propagator must be 'sgp4'; pre-propagate with ssapy.rv for other "
        "models."
    )


_TLE_LINE1_FIELDS = {"line1", "tle_line1", "tleline1"}
_TLE_LINE2_FIELDS = {"line2", "tle_line2", "tleline2"}
_TLE_NAME_FIELDS = {
    "name", "object_name", "objectname", "satno", "idonorbit",
    "origobjectid", "norad_cat_id",
}
_TLE_FIELDS = _TLE_NAME_FIELDS | _TLE_LINE1_FIELDS | _TLE_LINE2_FIELDS


def _hdf5_value(value):
    """Return an HDF5 scalar as a JSON-ready Python value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\0 ")
    return value


def _hdf5_column(dataset):
    values = np.asarray(dataset[()])
    if values.shape == ():
        return [_hdf5_value(values[()])]
    return [_hdf5_value(value) for value in values.reshape(-1)]


def _load_hdf5_records(path):
    """Read row groups, column groups, or compound datasets containing TLEs."""
    import h5py

    records = []

    def matching_key(names, aliases):
        return next((name for name in names if name.lower() in aliases), None)

    def collect(_name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.names:
            names = list(obj.dtype.names)
            if matching_key(names, _TLE_LINE1_FIELDS) and matching_key(names, _TLE_LINE2_FIELDS):
                for row in np.asarray(obj[()]).reshape(-1):
                    records.append({
                        name: _hdf5_value(row[name])
                        for name in names if name.lower() in _TLE_FIELDS
                    })
            return

        if not isinstance(obj, h5py.Group):
            return
        names = [
            name for name in obj
            if name.lower() in _TLE_FIELDS and isinstance(obj[name], h5py.Dataset)
        ]
        line1_key = matching_key(names, _TLE_LINE1_FIELDS)
        line2_key = matching_key(names, _TLE_LINE2_FIELDS)
        if not line1_key or not line2_key:
            return

        columns = {name: _hdf5_column(obj[name]) for name in names}
        row_count = len(columns[line1_key])
        if len(columns[line2_key]) != row_count:
            raise ValueError(f"HDF5 TLE columns have different lengths in {obj.name!r}")
        for index in range(row_count):
            records.append({
                name: values[0] if len(values) == 1 else values[index]
                for name, values in columns.items()
                if len(values) in (1, row_count)
            })

    with h5py.File(path, "r") as handle:
        collect("/", handle)
        handle.visititems(collect)
    if not records:
        raise ValueError("HDF5 file contains no groups or datasets with line1/line2 TLE fields")
    return records


def load_satellite_database(path=None):
    """Load a JSON, CSV, or HDF5 satellite catalog for embedding.

    JSON:API exports from ESA DISCOS are accepted as-is; the browser flattens
    their ``attributes`` records and uses NORAD IDs to match available TLEs.
    """
    explicit_path = path is not None
    if path is None:
        from ssatk.io.tle_updater import SATELLITES_JSON

        path = SATELLITES_JSON
    if not os.path.isfile(path):
        if not explicit_path:
            return None
        raise FileNotFoundError(path)
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if extension == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if extension in {".h5", ".hdf5", ".hdf"}:
        return _load_hdf5_records(path)
    raise ValueError(f"Unsupported satellite database format {extension!r}; use JSON, CSV, or HDF5")


def _read_texture_binary(filename, kind):
    """Read a texture from the core data package or return a generated placeholder."""
    try:
        return read_data_binary(filename)
    except (DataPackageNotFoundError, DataResourceNotFoundError):
        pass

    try:
        from ssatk.plots.starfield import find_data_file
        path = find_data_file(filename)
        if path is not None:
            return path.read_bytes()
    except Exception:
        pass

    return _placeholder_texture(kind)


def _placeholder_texture(kind):
    """Small deterministic texture used when optional Earth assets are absent."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # 1x1 black PNG; acceptable for every channel if Pillow is absent.
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

    mode = "RGB" if kind != "clouds" else "RGBA"
    image = Image.new(mode, (64, 32), (7, 25, 58, 255) if mode == "RGBA" else (7, 25, 58))
    draw = ImageDraw.Draw(image)
    if kind == "day":
        draw.rectangle((0, 0, 63, 31), fill=(20, 82, 142))
        draw.ellipse((4, 6, 25, 22), fill=(50, 130, 70))
        draw.ellipse((32, 4, 58, 24), fill=(60, 145, 75))
    elif kind == "night":
        draw.rectangle((0, 0, 63, 31), fill=(2, 8, 24))
        for x, y in [(8, 9), (15, 14), (35, 8), (48, 20), (55, 12)]:
            draw.point((x, y), fill=(255, 210, 120))
    elif kind == "specular":
        draw.rectangle((0, 0, 63, 31), fill=(45, 45, 45))
        draw.ellipse((0, 0, 63, 31), fill=(160, 160, 160))
    else:
        image = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 26, 18), fill=(255, 255, 255, 80))
        draw.ellipse((30, 5, 58, 19), fill=(255, 255, 255, 65))

    buffer = BytesIO()
    if kind == "clouds":
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# NOTE: asset loading, template substitution and the file write all used to
# happen here at module level. Because ssatk/plots/__init__.py
# auto-imports every .py in this folder, that meant a 4.1 MB HTML file was
# read, base64-encoded, assembled and written to disk on EVERY
# `import ssatk.plots` -- every GUI start, every pytest run, every CI
# job -- and it printed "wrote ... (4.11 MB)" each time. That work now lives
# in build() below and only runs when this file is executed directly.

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Satellite 3D Scene -- Real Earth texture + TLE-driven satellites</title>
<style>
  html, body { margin: 0; padding: 0; background: #000; overflow: hidden; height: 100%; }
  #scene-container { position: absolute; inset: 0; }
  #hint {
    position: absolute; top: 14px; left: 18px; color: #cfd8e3; font: 13px/1.4 -apple-system, sans-serif;
    background: rgba(0,0,0,0.35); padding: 8px 12px; border-radius: 8px; pointer-events: none;
  }
  #label-layer {
    position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 5;
  }
  .sat-label {
    position: absolute; transform: translate(10px, -50%); white-space: nowrap;
    color: #eef3fa; font: 11px/1.25 -apple-system, sans-serif; letter-spacing: 0.02em;
    background: rgba(8,12,20,0.66); padding: 3px 7px; border-radius: 5px;
    border: 1px solid rgba(140,170,220,0.28); pointer-events: none;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6); will-change: left, top;
  }
  .sat-label .sat-label-sub {
    display: block; font-size: 9.5px; opacity: 0.72; font-variant-numeric: tabular-nums;
    letter-spacing: 0;
  }
  .sat-label::before {
    content: ''; position: absolute; left: -7px; top: 50%; width: 4px; height: 4px;
    margin-top: -2px; border-radius: 50%; background: #ffe066;
    box-shadow: 0 0 5px 1px rgba(255,224,102,0.7);
  }
  #sat-panel {
    position: absolute; top: 14px; right: 18px; color: #e8edf4; font: 13px/1.5 -apple-system, sans-serif;
    background: rgba(10,14,22,0.72); padding: 12px 14px; border-radius: 10px; width: 280px;
    border: 1px solid rgba(255,255,255,0.08); max-height: 90vh; display: flex; flex-direction: column;
  }

  #time-section {
    flex-shrink: 0; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333c4a;
  }
  #time-clock-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  #time-clock { font-size: 11px; opacity: 0.75; font-variant-numeric: tabular-nums; }
  #time-scale-select {
    background: #171c26; color: #e8edf4; border: 1px solid #333c4a; border-radius: 6px;
    padding: 4px 6px; font: 12px -apple-system, sans-serif;
  }
  #time-pause-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 4px 10px; font: 12px -apple-system, sans-serif; cursor: pointer;
  }
  #time-pause-btn:hover { background: #2b3c5c; }
  #time-step-row { display: flex; gap: 4px; margin-bottom: 6px; }
  .time-step-btn, #time-now-btn {
    flex: 1; background: #171c26; color: #e8edf4; border: 1px solid #333c4a; border-radius: 5px;
    padding: 4px 2px; font: 11px -apple-system, sans-serif; cursor: pointer;
  }
  .time-step-btn:hover, #time-now-btn:hover { background: #1e2532; }
  #time-jump-row { display: flex; gap: 4px; }
  #time-jump-input {
    flex: 1; min-width: 0; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 3px 4px; font: 11px -apple-system, sans-serif;
  }
  #time-jump-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 5px;
    padding: 3px 8px; font: 11px -apple-system, sans-serif; cursor: pointer; flex-shrink: 0;
  }
  #time-jump-btn:hover { background: #2b3c5c; }

  #db-section { flex-shrink: 0; margin-bottom: 10px; }
  #db-load-row {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px; margin-bottom: 6px;
  }
  #db-load-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    width: 100%; min-width: 0; box-sizing: border-box;
    padding: 5px 8px; font: 12px -apple-system, sans-serif; cursor: pointer;
  }
  #db-load-btn:hover { background: #2b3c5c; }
  #db-status { grid-column: 1 / -1; font-size: 11px; opacity: 0.65; overflow-wrap: anywhere; }
  #db-search-input {
    width: 100%; box-sizing: border-box; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 6px; padding: 6px 8px; font: 13px -apple-system, sans-serif; margin-bottom: 6px;
  }
  #db-search-input:disabled { opacity: 0.5; }
  #db-search-results { max-height: 260px; overflow-y: auto; border: 1px solid #262d3d; border-radius: 6px; }
  .db-hint { padding: 8px; font-size: 11px; opacity: 0.6; }
  .db-result-row {
    padding: 5px 8px; font-size: 12px; cursor: pointer; display: flex; gap: 6px;
    border-bottom: 1px solid #1c212e;
  }
  .db-result-row:last-child { border-bottom: none; }
  .db-result-row:hover { background: #1a2333; }
  .db-result-row.active { background: #1c3050; }
  .db-result-check { width: 12px; color: #6fd68a; flex-shrink: 0; }
  .db-result-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  #analysis-section {
    flex-shrink: 0; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333c4a;
  }
  .analysis-toggle {
    display: flex; align-items: center; gap: 7px; font-size: 12px; cursor: pointer; user-select: none;
  }
  .analysis-toggle input { accent-color: #5fd6e6; cursor: pointer; margin: 0; }
  .analysis-swatch {
    width: 18px; height: 3px; border-radius: 2px; background: #5fd6e6; flex-shrink: 0;
    box-shadow: 0 0 4px 0 rgba(95,214,230,0.7);
  }
  .analysis-hint { font-size: 10.5px; opacity: 0.55; }

  #conj-controls { margin-top: 9px; }
  #conj-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 7px; }
  #conj-screen-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 5px 10px; font: 12px -apple-system, sans-serif; cursor: pointer;
  }
  #conj-screen-btn:hover:not(:disabled) { background: #2b3c5c; }
  #conj-screen-btn:disabled { opacity: 0.6; cursor: default; }
  .conj-param { font-size: 11px; opacity: 0.8; display: flex; align-items: center; gap: 3px; }
  .conj-param input {
    width: 46px; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 3px 4px; font: 11px -apple-system, sans-serif;
  }
  #conj-results { max-height: 168px; overflow-y: auto; font-size: 11.5px; }
  .conj-event {
    padding: 5px 8px; border: 1px solid #2a3346; border-left-width: 3px; border-radius: 6px;
    margin-bottom: 5px; cursor: pointer;
  }
  .conj-event:hover { background: #1a2333; }
  .conj-pair { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conj-meta { opacity: 0.72; font-variant-numeric: tabular-nums; margin-top: 1px; }
  .conj-empty { font-size: 11px; opacity: 0.6; padding: 2px 0; }

  #pass-controls { margin-top: 10px; padding-top: 9px; border-top: 1px solid #333c4a; }
  .pass-title { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
  .pass-site-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .pass-site-row input[type=number] { width: 58px; }
  #pass-target {
    flex: 1 1 auto; min-width: 0; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 4px 5px; font: 11px -apple-system, sans-serif;
  }
  #pass-compute-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 4px 10px; font: 12px -apple-system, sans-serif; cursor: pointer; flex-shrink: 0;
  }
  #pass-compute-btn:hover:not(:disabled) { background: #2b3c5c; }
  #pass-compute-btn:disabled { opacity: 0.6; cursor: default; }
  #pass-results { max-height: 168px; overflow-y: auto; font-size: 11.5px; }
  .cloud-toggle-row { margin-top: 4px; margin-bottom: 4px; }
  .cloud-status { font-size: 10px; opacity: 0.6; margin-bottom: 6px; min-height: 12px; overflow-wrap: anywhere; }
  .vis-badge { font-size: 10px; opacity: 0.9; white-space: nowrap; }
  .vis-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }
  .csv-btn {
    margin-top: 6px; background: #1a2230; color: #cfe0f0; border: 1px solid #33465e;
    border-radius: 5px; padding: 4px 10px; font: 11px -apple-system, sans-serif; cursor: pointer;
  }
  .csv-btn:hover:not(:disabled) { background: #223049; }
  .csv-btn:disabled { opacity: 0.4; cursor: default; }

  #legend-details { margin-top: 10px; padding-top: 9px; border-top: 1px solid #333c4a; }
  #legend-details summary { font-size: 12px; cursor: pointer; opacity: 0.85; }
  .legend-grid {
    display: grid; grid-template-columns: auto 1fr; gap: 4px 8px; align-items: center;
    margin-top: 7px; font-size: 11px; opacity: 0.85;
  }
  .lg-swatch { width: 16px; height: 4px; border-radius: 2px; display: inline-block; }
  .legend-note { font-size: 10px; opacity: 0.55; margin-top: 8px; line-height: 1.4; }

  #sat-info { font-size: 12px; line-height: 1.5; overflow-y: auto; flex: 1 1 auto; border-top: 1px solid #333c4a; padding-top: 8px; }
  #sat-info b { color: #ffe066; }
  .sat-info-block { padding: 2px 0; }
  .sat-info-line { padding: 1px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sat-info-sep { border: none; border-top: 1px solid #333c4a; opacity: 0.6; margin: 8px 0; }
</style>
</head>
<body>
<div id="scene-container"></div>
<div id="label-layer"></div>
<div id="hint">Drag to rotate &middot; scroll / pinch to zoom</div>
<div id="sat-panel">
  <div id="time-section">
    <div id="time-clock-row">
      <span id="time-clock">--</span>
      <select id="time-scale-select">
        <option value="1" selected>Real-time</option>
        <option value="60">60x (1 min/s)</option>
        <option value="3600">3,600x (1 hr/s)</option>
        <option value="86400">86,400x (1 day/s)</option>
      </select>
    </div>
    <div id="time-step-row">
      <button class="time-step-btn" type="button" data-step="-86400000">-1d</button>
      <button class="time-step-btn" type="button" data-step="-3600000">-1h</button>
      <button id="time-now-btn" type="button">Now</button>
      <button class="time-step-btn" type="button" data-step="3600000">+1h</button>
      <button class="time-step-btn" type="button" data-step="86400000">+1d</button>
      <button id="time-pause-btn" type="button">Pause</button>
    </div>
    <div id="time-jump-row">
      <input type="datetime-local" id="time-jump-input" step="1">
      <button id="time-jump-btn" type="button">Go (UTC)</button>
    </div>
  </div>
  <div id="db-section">
    <div id="db-load-row">
      <button id="db-load-btn" type="button">Load database</button>
      <span id="db-status" style="opacity:0.5">No bundled database found -- load a catalog</span>
    </div>
    <input type="file" id="db-file-input" accept=".json,.csv" style="display:none">
    <input type="text" id="db-search-input" placeholder="Load a database first..." disabled>
    <div id="db-search-results"></div>
  </div>
  <div id="analysis-section">
    <label class="analysis-toggle">
      <input type="checkbox" id="ground-track-toggle">
      <span class="analysis-swatch"></span>
      Ground tracks
      <span class="analysis-hint">sub-satellite path</span>
    </label>
    <label class="analysis-toggle" title="Use supplied body-to-TEME quaternions when state-vector data includes q=[w,x,y,z].">
      <input type="checkbox" id="attitude-toggle">
      <span class="analysis-swatch" style="background:#c58cff; box-shadow:0 0 4px 0 rgba(197,140,255,0.7)"></span>
      Quaternion attitude
      <span class="analysis-hint">state vectors with q</span>
    </label>
    <button id="labels-toggle" class="csv-btn" type="button" aria-pressed="true">Hide labels</button>
    <div id="conj-controls">
      <div id="conj-row">
        <button id="conj-screen-btn" type="button">Screen conjunctions</button>
        <label class="conj-param">window <input type="number" id="conj-window" value="24" min="1" max="168" step="1">h</label>
        <label class="conj-param">&le; <input type="number" id="conj-threshold" value="10" min="0.1" max="500" step="0.5">km</label>
      </div>
      <div id="conj-results"><div class="conj-empty">Screen to compute closest approaches.</div></div>
      <button id="conj-export-btn" class="csv-btn" type="button" disabled>Export CSV</button>
    </div>
    <div id="pass-controls">
      <div class="pass-title">Passes over a ground site</div>
      <div class="pass-site-row">
        <label class="conj-param" title="Observer latitude in degrees (north positive)">lat <input type="number" id="pass-lat" value="37.68" step="0.01"></label>
        <label class="conj-param" title="Observer longitude in degrees (east positive)">lon <input type="number" id="pass-lon" value="-121.77" step="0.01"></label>
        <label class="conj-param" title="Minimum elevation above the horizon to count as a pass (0 deg = horizon, 90 deg = straight overhead). Below ~10 deg is usually blocked by terrain/buildings and hazy.">min el <input type="number" id="pass-minel" value="10" min="0" max="89" step="1">&deg;</label>
      </div>
      <div class="pass-site-row">
        <select id="pass-target"></select>
        <label class="conj-param">win <input type="number" id="pass-window" value="24" min="1" max="168" step="1">h</label>
        <button id="pass-compute-btn" type="button">Compute</button>
      </div>
      <label class="analysis-toggle cloud-toggle-row">
        <input type="checkbox" id="cloud-toggle">
        Check sky <span class="analysis-hint">Open-Meteo forecast &middot; external request</span>
      </label>
      <div id="cloud-status" class="cloud-status"></div>
      <div id="pass-results"><div class="conj-empty">Set a site and target, then Compute.</div></div>
      <button id="pass-export-btn" class="csv-btn" type="button" disabled>Export CSV</button>
    </div>
    <details id="legend-details">
      <summary>Legend</summary>
      <div class="legend-grid">
        <span class="lg-swatch" style="background:#ffffff"></span><span>orbit path</span>
        <span class="lg-swatch" style="background:#5fd6e6"></span><span>ground track / nadir</span>
        <span class="lg-swatch" style="background:#66ff99"></span><span>ground site / in view</span>
        <span class="lg-swatch" style="background:#6fd68a; border-radius:50%; width:8px; height:8px"></span><span>pass: optically visible</span>
        <span class="lg-swatch" style="background:#7d8796; border-radius:50%; width:8px; height:8px"></span><span>pass: daylight (not visible)</span>
        <span class="lg-swatch" style="background:#4a5568; border-radius:50%; width:8px; height:8px"></span><span>pass: in Earth's shadow</span>
        <span class="lg-swatch" style="background:#ffe066"></span><span>conjunction &lt; threshold</span>
        <span class="lg-swatch" style="background:#ffa53d"></span><span>conjunction &lt; 5 km</span>
        <span class="lg-swatch" style="background:#ff4d4d"></span><span>conjunction &lt; 1 km</span>
      </div>
      <div class="legend-note">Motion is real propagation at the selected time (use the speed control to see it). Positions are TLE-accuracy; conjunctions are geometric miss distance, not collision probability.</div>
    </details>
  </div>
  <div id="sat-info"></div>
</div>
<!-- Viewer libraries are inlined for full offline operation -- no CDN or
     network dependency. See NOTICE. -->
<script>
__THREE_JS__
</script>
<script>
__SATELLITE_JS__
</script>
<script>
const DAY_TEXTURE_DATAURI = "data:image/jpeg;base64,__DAY_B64__";
const NIGHT_TEXTURE_DATAURI = "data:image/jpeg;base64,__NIGHT_B64__";
const SPECULAR_TEXTURE_DATAURI = "data:image/jpeg;base64,__SPEC_B64__";
const CLOUDS_TEXTURE_DATAURI = "data:image/png;base64,__CLOUDS_B64__";
const SSAPY_CONSTANTS = __SSAPY_CONSTANTS__;
const STAR_CATALOG = __STAR_CATALOG__;
const STATE_VECTOR_TRACKS = __STATE_VECTOR_TRACKS__;
const PROPAGATOR = __PROPAGATOR__;
const INITIAL_SIM_TIME_MS = null;
let BUNDLED_SATELLITE_DATABASE = __SATELLITE_DATABASE__;
</script>
<script>
__SCENE_JS__
</script>
</body>
</html>
"""

def build(out_path=None, verbose=True, database_path=None, state_vectors=None,
          propagator="sgp4"):
    """Assemble the self-contained Three.js viewer and write it to disk.

    Everything expensive lives here rather than at module level so that
    importing this module -- which the plots package does automatically --
    costs nothing. Call it explicitly, or run this file as a script.

    Parameters
    ----------
    out_path : str or None
        Destination HTML file. Defaults to the standard SSATK output directory
        under ``~/ssatk_output/figures``.
    verbose : bool
        Print the written path and size, as the old module-level code did.
    database_path : str or None
        Satellite JSON, CSV, or HDF5 file to embed, including ESA DISCOS JSON.
        None (the default) embeds no catalog, and the page keeps its
        file-picker.
    state_vectors : list of dict or None
        JSON-ready propagated state-vector tracks. This is normally supplied by
        :func:`satellite_viewer`; tracks may include body-to-TEME ``q`` samples
        in ``[w, x, y, z]`` order. When absent, no satellite catalog is
        embedded unless ``database_path`` is supplied.
    propagator : str or propagator object
        Browser propagation model for TLE-backed entries. Only ``"sgp4"`` is
        supported. State-vector input is displayed by interpolation and is
        not propagated in the browser.

    Returns
    -------
    str
        The path written.
    """
    propagator = _propagator_name(propagator)
    _tex = load_textures()
    day_b64 = _tex["day"]
    night_b64 = _tex["night"]
    spec_b64 = _tex["specular"]
    clouds_b64 = _tex["clouds"]
    ssapy_constants_json = json.dumps(
        _load_ssapy_plot_constants(), separators=(",", ":")
    )
    stars_json = json.dumps(load_star_catalog(), separators=(",", ":"))
    database_json = json.dumps(
        load_satellite_database(database_path) if database_path is not None else None,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    state_vectors_json = json.dumps(
        state_vectors or [], separators=(",", ":")
    ).replace("<", "\\u003c")

    satellite_js = text("satellite.min.js")
    three_js = text("three.min.js")
    scene_js = text("satellite_viewer_scene.js")

    replacements = {
        "SATELLITE_JS": satellite_js,
        "DAY_B64": day_b64,
        "NIGHT_B64": night_b64,
        "SPEC_B64": spec_b64,
        "CLOUDS_B64": clouds_b64,
        "SSAPY_CONSTANTS": ssapy_constants_json,
        "STAR_CATALOG": stars_json,
        "STATE_VECTOR_TRACKS": state_vectors_json,
        "PROPAGATOR": json.dumps(propagator),
        "SATELLITE_DATABASE": database_json,
        "SCENE_JS": scene_js,
        "THREE_JS": three_js,
    }
    doc = re.sub(
        r"__([A-Z0-9_]+)__",
        lambda match: replacements.get(match.group(1), match.group(0)),
        html,
    )

    if out_path is None:
        from ssatk.plots.figpath import figpath
        out_path = figpath("satellite_3d_scene_threejs.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    if verbose:
        print("wrote {} ({:.2f} MB)".format(out_path, len(doc) / 1e6))
    return out_path


def satellite_viewer(r, v, t, *, labels=None, units="m", q=None, save_path=None,
                     verbose=True, database_path=None, propagator="sgp4"):
    """Write the satellite WebGL viewer from propagated SSAPy state arrays.

    ``r`` and ``v`` are the position and velocity arrays returned by
    ``ssapy.rv`` in GCRF; their default units are metres and metres/second.
    They are rotated into TEME before embedding because the browser's SGP4,
    Earth rotation, and analysis paths use TEME. ``t`` is the corresponding
    Astropy ``Time`` or GPS-second array. Optional ``q`` contains body-to-GCRF
    quaternions in ``[w, x, y, z]`` order, the same frame as ``r`` and ``v``,
    and is rotated into body-to-TEME; the viewer exposes a checkbox to apply
    them to the spacecraft models.
    ``propagator`` is used only if the bundled/catalog TLE path also needs
    browser-side propagation.
    """
    r, v, t, q = _gcrf_state_vectors_to_teme(r, v, t, q=q)
    state_vectors = _prepare_state_vectors(r, v, t, labels=labels, units=units, q=q)
    return build(
        out_path=save_path,
        verbose=verbose,
        database_path=database_path,
        state_vectors=state_vectors,
        propagator=propagator,
    )


if __name__ == "__main__":
    build()
