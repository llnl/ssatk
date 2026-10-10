# --- Standard library ---
import io
import os
import re
from enum import Enum, auto
from numbers import Real
from pathlib import Path

# --- Third-party ---
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import cnames, to_rgb, rgb2hex
from PIL import Image as PILImage

# --- Local modules ---
from ssapy.utils import find_file
from ._textures import earth_texture_path
from ..constants import EARTH_RADIUS, MOON_RADIUS
from ..vectors import rotation_matrix_from_vectors


_SAVE_PATH_ALIAS_KEYS = (
    "save",
    "savefig",
    "save_fig",
    "save_figure",
    "savepath",
    "save_path",
)


def _plot_series_arrays(values):
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
        if array.ndim <= 1:
            return [array.reshape(-1)]
        return [np.asarray(row, dtype=float).reshape(-1) for row in array]
    return [np.asarray(value, dtype=float).reshape(-1) for value in values]


def log_safe_values(values):
    """Return ``values`` with non-positive or non-finite samples masked for log plots."""
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array) & (array > 0.0), array, np.nan)


def positive_dynamic_range(values):
    """Return max/min over positive finite samples, or 1 when no log scale is useful."""
    positive = []
    for array in _plot_series_arrays(values):
        positive_value = array[np.isfinite(array) & (array > 0.0)]
        if positive_value.size:
            positive.append(positive_value)
    if not positive:
        return 1.0
    combined = np.concatenate(positive)
    return float(combined.max() / combined.min())


def should_use_log_scale(values, *, min_dynamic_range=100.0):
    """Return True when positive finite data span enough range for log scaling."""
    return positive_dynamic_range(values) >= min_dynamic_range


def auto_log_lower_bound(values, *, divergence_fraction=1e-4, max_decades=4.0):
    """Choose a non-tiny lower bound for log-scaled difference curves."""
    arrays = _plot_series_arrays(values)
    positive = [
        array[np.isfinite(array) & (array > 0.0)]
        for array in arrays
    ]
    positive = [array for array in positive if array.size]
    if not positive:
        return None

    combined = np.concatenate(positive)
    fallback = float(max(combined.min(), combined.max() * 10.0 ** -float(max_decades)))

    same_size = arrays and all(array.size == arrays[0].size for array in arrays)
    if len(arrays) < 2 or not same_size:
        return fallback

    stack = np.vstack(arrays)
    valid = np.isfinite(stack) & (stack > 0.0)
    row_count = valid.sum(axis=0)
    valid_spread = row_count >= 2
    if not np.any(valid_spread):
        return fallback

    masked = np.where(valid, stack, np.nan)
    row_min = np.full(stack.shape[1], np.nan)
    row_max = np.full(stack.shape[1], np.nan)
    row_min[valid_spread] = np.nanmin(masked[:, valid_spread], axis=0)
    row_max[valid_spread] = np.nanmax(masked[:, valid_spread], axis=0)
    spread = row_max - row_min
    max_spread = np.nanmax(spread[valid_spread])
    if not np.isfinite(max_spread) or max_spread <= 0.0:
        return fallback

    first = np.flatnonzero(valid_spread & (spread >= max_spread * divergence_fraction))
    if not first.size:
        return fallback
    return float(row_min[first[0]])


def apply_auto_log_scale(
    ax,
    values,
    *,
    axis="y",
    min_dynamic_range=100.0,
    lower_bound=None,
    divergence_fraction=1e-4,
    max_decades=4.0,
):
    """Apply log scaling when data span enough range; return whether it was applied."""
    if not should_use_log_scale(values, min_dynamic_range=min_dynamic_range):
        return False

    lower = lower_bound
    if lower is None:
        lower = auto_log_lower_bound(
            values,
            divergence_fraction=divergence_fraction,
            max_decades=max_decades,
        )

    axis_key = str(axis).lower()
    if axis_key == "y":
        ax.set_yscale("log")
        if lower is not None:
            ax.set_ylim(bottom=lower)
    elif axis_key == "x":
        ax.set_xscale("log")
        if lower is not None:
            ax.set_xlim(left=lower)
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return True


class VarType(Enum):
    NONE = auto()
    TIME = auto()
    ARRAY = auto()
    LIST_ARRAYS = auto()
    LIST_LISTS = auto()
    MIXED_LIST = auto()
    OTHER = auto()


def is_list_of_arrays(lst):
    return all(isinstance(item, np.ndarray) for item in lst)


def is_list_of_lists(lst):
    return all(isinstance(item, list) for item in lst)


def check_type(var):
    """
    Classify 'var' into one of VarType cases.
    """
    if var is None:
        return VarType.NONE

    if isinstance(var, Time):
        return VarType.TIME

    if isinstance(var, np.ndarray):
        return VarType.ARRAY

    if isinstance(var, list):
        if len(var) == 0:
            return VarType.OTHER
        if is_list_of_arrays(var):
            if all(isinstance(item.flat[0], Time) for item in var if item.size > 0):
                return VarType.TIME
            return VarType.LIST_ARRAYS
        if is_list_of_lists(var):
            return VarType.LIST_LISTS
        if all(isinstance(item, Time) for item in var):
            return VarType.TIME
        return VarType.MIXED_LIST

    return VarType.OTHER


def _is_orbit_like(value):
    return value is not None and all(hasattr(value, attr) for attr in ("r", "v", "t"))


def _as_gps_seconds_array(t):
    """Return numeric GPS seconds from Astropy Time, numeric, or Time-like arrays."""
    if t is None:
        return None

    if isinstance(t, Time):
        return np.atleast_1d(np.asarray(t.gps, dtype=float))

    if hasattr(t, "gps"):
        return np.atleast_1d(float(t.gps))

    arr = np.asarray(t)
    if arr.dtype == object:
        flat = list(arr.flat)
        if flat and all(isinstance(item, Time) or hasattr(item, "gps") for item in flat):
            return np.asarray([float(item.gps) for item in flat], dtype=float).reshape(arr.shape)

    return np.atleast_1d(np.asarray(t, dtype=float))


def _ensure_nx3(values, name):
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    if arr.ndim == 2 and arr.shape in [(1, 3), (3, 1)]:
        return arr.reshape(1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(f"{name} must be a 3-vector or an (N, 3) array; got shape {arr.shape}")


def _default_orbit_times_gps(orbit, *, n_steps=360, n_orbits=1.0):
    r_arr = np.asarray(getattr(orbit, "r"), dtype=float).squeeze()
    t_attr = _as_gps_seconds_array(getattr(orbit, "t", 0.0))
    start = float(np.ravel(t_attr)[0]) if t_attr is not None and t_attr.size else 0.0

    if r_arr.ndim == 2 and r_arr.shape[1] == 3:
        if t_attr is not None and t_attr.size == r_arr.shape[0]:
            return np.asarray(t_attr, dtype=float).reshape(-1)
        return start + np.arange(r_arr.shape[0], dtype=float)

    try:
        period = float(np.ravel(np.asarray(getattr(orbit, "period")))[0])
    except Exception:
        period = np.nan
    if not np.isfinite(period) or period <= 0:
        return np.asarray([start], dtype=float)

    n_steps = max(2, int(n_steps))
    return start + np.linspace(0.0, period * float(n_orbits), n_steps)


def _sample_orbit_like(orbit, t=None, *, n_steps=360, n_orbits=1.0):
    """Sample an SSAPy-like Orbit object into r, v, and GPS-second arrays."""
    t_gps = _as_gps_seconds_array(t)
    if t_gps is None:
        t_gps = _default_orbit_times_gps(orbit, n_steps=n_steps, n_orbits=n_orbits)
    t_gps = np.asarray(t_gps, dtype=float).reshape(-1)

    sampled = orbit
    if hasattr(orbit, "at"):
        try:
            sampled = orbit.at(t_gps)
        except Exception:
            sampled = orbit

    r_arr = _ensure_nx3(getattr(sampled, "r"), "orbit.r")
    v_arr = _ensure_nx3(getattr(sampled, "v"), "orbit.v")
    sampled_t = _as_gps_seconds_array(getattr(sampled, "t", t_gps))
    sampled_t = np.asarray(sampled_t, dtype=float).reshape(-1)
    if sampled_t.size != len(r_arr):
        if t_gps.size == len(r_arr):
            sampled_t = t_gps
        elif sampled_t.size == 1:
            sampled_t = np.full(len(r_arr), sampled_t[0], dtype=float)
        else:
            sampled_t = np.linspace(t_gps[0], t_gps[-1], len(r_arr))
    return r_arr, v_arr, sampled_t


def _orbit_position_tracks(r, t=None):
    if _is_orbit_like(r):
        r_arr, _, t_gps = _sample_orbit_like(r, t=t)
        return [r_arr], [t_gps]

    if isinstance(r, (list, tuple)) and r and all(_is_orbit_like(item) for item in r):
        if isinstance(t, (list, tuple)) and len(t) == len(r):
            per_orbit_t = t
        else:
            per_orbit_t = [t] * len(r)
        tracks = []
        times = []
        for orbit, orbit_t in zip(r, per_orbit_t):
            r_arr, _, t_gps = _sample_orbit_like(orbit, t=orbit_t)
            tracks.append(r_arr)
            times.append(t_gps)
        return tracks, times

    return None


def _position_scale_to_km(r, units="auto"):
    key = "auto" if units is None else str(units).strip().lower()
    if key in {"km", "kilometer", "kilometers"}:
        return 1.0
    if key in {"m", "meter", "meters"}:
        return 1e-3
    if key != "auto":
        raise ValueError("r_units must be 'auto', 'm', or 'km'")

    norms = np.linalg.norm(_ensure_nx3(r, "r"), axis=1)
    typical = float(np.nanmedian(norms)) if norms.size else 0.0
    # SSAPy native positions are metres (LEO is ~7e6), while Toolkit plotting
    # arrays are usually kilometres.  A value above 1e6 is ambiguous: use the
    # Moon's radius to keep low lunar-orbit metres distinct from L2 kilometres.
    candidate_radius_km = typical * 1e-3
    return 1e-3 if typical > 1e6 and candidate_radius_km >= MOON_RADIUS / 1e3 else 1.0


def _velocity_scale_to_kms(v, units="auto"):
    key = "auto" if units is None else str(units).strip().lower().replace(" ", "")
    if key in {"km/s", "kms", "kmps", "kilometer/second", "kilometers/second"}:
        return 1.0
    if key in {"m/s", "ms", "mps", "meter/second", "meters/second"}:
        return 1e-3
    if key != "auto":
        raise ValueError("v_units must be 'auto', 'm/s', or 'km/s'")

    norms = np.linalg.norm(_ensure_nx3(v, "v"), axis=1)
    typical = float(np.nanmedian(norms)) if norms.size else 0.0
    return 1e-3 if typical > 100.0 else 1.0


def normalize_orbit_trajectory(
    *,
    orbit=None,
    r=None,
    v=None,
    t=None,
    require_velocity=False,
    r_units="auto",
    v_units="auto",
    n_steps=360,
    n_orbits=1.0,
):
    """Normalize SSAPy orbit outputs into ``r_km, v_kms, t`` for Plotly helpers.

    Accepted inputs are an SSAPy-like ``Orbit`` object, a position time series
    from ``ssapy.rv``/``Orbit.at`` in metres, or already-converted km arrays.
    Raw arrays use unit auto-detection by default; explicit ``r_units`` and
    ``v_units`` are available when callers need deterministic conversion.
    """
    if orbit is None and _is_orbit_like(r):
        orbit, r = r, None

    if orbit is not None:
        r_arr, v_arr, t_gps = _sample_orbit_like(orbit, t=t, n_steps=n_steps, n_orbits=n_orbits)
        orbit_r_units = "m" if r_units == "auto" else r_units
        orbit_v_units = "m/s" if v_units == "auto" else v_units
        r_km = r_arr * _position_scale_to_km(r_arr, orbit_r_units)
        v_kms = v_arr * _velocity_scale_to_kms(v_arr, orbit_v_units)
        return r_km, v_kms, Time(t_gps, format="gps")

    if r is None:
        raise ValueError("Provide either orbit= or r= trajectory input.")

    r_arr = _ensure_nx3(r, "r")
    r_km = r_arr * _position_scale_to_km(r_arr, r_units)

    v_kms = None
    if v is not None:
        v_arr = _ensure_nx3(v, "v")
        if len(v_arr) != len(r_arr):
            raise ValueError("v must have the same number of samples as r")
        v_kms = v_arr * _velocity_scale_to_kms(v_arr, v_units)
    elif require_velocity:
        raise ValueError("Velocity input is required; provide v= or an Orbit object.")

    t_gps = _as_gps_seconds_array(t)
    if t_gps is None:
        t_gps = np.zeros(len(r_arr), dtype=float)
    t_gps = np.asarray(t_gps, dtype=float).reshape(-1)
    if t_gps.size == 1 and len(r_arr) > 1:
        t_gps = np.full(len(r_arr), t_gps[0], dtype=float)
    if t_gps.size != len(r_arr):
        raise ValueError("t must be scalar or have the same number of samples as r")

    return r_km, v_kms, Time(t_gps, format="gps")


def valid_orbits(r, t, drop_empty=True, warn=True):
    """
    Normalize r and t into parallel lists of shape-(n,3) ndarrays and astropy Time objects.

    Accepts:
      r:
        - SSAPy-like Orbit object, or list/tuple of Orbit objects
        - (3,), (1,3), (N,3), (B,N,3) ndarray
        - list/tuple of any of the above
      t:
        - None
        - scalar (float/int) interpreted as GPS seconds (broadcast)
        - ndarray of GPS seconds: (N,) or (B,N)
        - astropy.time.Time (scalar or array)
        - list/tuple of scalars/ndarrays/Time, matching number of tracks

    Additionally (if drop_empty=True):
      * removes empty r-tracks (Ni==0)
      * if t is a per-track list/tuple with the same length as the original r_list,
        removes the corresponding t entries too
      * prints a warning when any are removed

    Returns:
      r_list: list[np.ndarray] where each is (Ni,3)
      t_list: list[astropy.time.Time] where each has len Ni
    """

    def _to_track_list_r(r_in):
        # returns list of (N,3) float arrays
        if isinstance(r_in, (list, tuple)):
            out = []
            for item in r_in:
                out.extend(_to_track_list_r(item))
            return out

        arr = np.asarray(r_in, dtype=float).squeeze()

        # Single position vector
        if arr.ndim == 1 and arr.size == 3:
            return [arr.reshape(1, 3)]

        # Row/col vector forms
        if arr.ndim == 2 and arr.shape in [(1, 3), (3, 1)]:
            return [arr.reshape(1, 3)]

        # Standard track
        if arr.ndim == 2 and arr.shape[1] == 3:
            return [arr]

        # Batched tracks
        if arr.ndim == 3 and arr.shape[2] == 3:
            return [arr[k] for k in range(arr.shape[0])]

        raise ValueError(f"valid_orbits: cannot interpret r with shape {arr.shape}")

    def _to_time_track_list(t_in, r_list):
        """
        Return list[Time] matching r_list length.
        """
        n_tracks = len(r_list)

        # None -> dummy gps time arrays (zeros)
        if t_in is None:
            return [Time(np.zeros(len(rr), dtype=float), format="gps") for rr in r_list]

        # Single Time object -> broadcast if needed
        if isinstance(t_in, Time):
            if t_in.isscalar:
                return [Time(np.full(len(rr), t_in.gps, dtype=float), format="gps") for rr in r_list]
            # If array Time:
            if n_tracks == 1:
                if len(t_in) != len(r_list[0]):
                    raise ValueError("valid_orbits: Time length must match r length")
                return [t_in]
            # Broadcast same Time array to all tracks (must match all)
            if not all(len(t_in) == len(rr) for rr in r_list):
                raise ValueError("valid_orbits: single Time array length must match all r tracks")
            return [t_in for _ in r_list]

        # Scalar numeric -> GPS seconds broadcast
        if isinstance(t_in, (int, float, np.integer, np.floating)):
            val = float(t_in)
            return [Time(np.full(len(rr), val, dtype=float), format="gps") for rr in r_list]

        # ndarray of gps seconds
        if isinstance(t_in, np.ndarray):
            arr = np.asarray(t_in)

            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError("valid_orbits: ndarray t must be numeric GPS seconds")

            if arr.ndim == 0:
                val = float(arr)
                return [Time(np.full(len(rr), val, dtype=float), format="gps") for rr in r_list]

            if arr.ndim == 1:
                if n_tracks == 1:
                    if arr.shape[0] != len(r_list[0]):
                        raise ValueError("valid_orbits: t length must match r length")
                    return [Time(arr.astype(float), format="gps")]

                # Broadcast one time vector to all tracks if it matches all
                if not all(arr.shape[0] == len(rr) for rr in r_list):
                    raise ValueError("valid_orbits: single t-array length must match all r tracks")
                tt = Time(arr.astype(float), format="gps")
                return [tt for _ in r_list]

            if arr.ndim == 2:
                # Batched times: (B,N)
                if arr.shape[0] != n_tracks:
                    raise ValueError("valid_orbits: batched t must have same number of tracks as r")
                out = []
                for rr, row in zip(r_list, arr):
                    if row.shape[0] != len(rr):
                        raise ValueError("valid_orbits: each batched t row must match its r track length")
                    out.append(Time(np.asarray(row, dtype=float), format="gps"))
                return out

            raise ValueError("valid_orbits: ndarray t must be 0D, 1D, or 2D")

        # list/tuple: per-track specification, or a single element to broadcast
        if isinstance(t_in, (list, tuple)):
            if len(t_in) == 1 and n_tracks > 1:
                # broadcast single element if possible
                return _to_time_track_list(t_in[0], r_list)

            if len(t_in) != n_tracks:
                raise ValueError("valid_orbits: number of t entries must equal number of r tracks")

            out = []
            for rr, ti in zip(r_list, t_in):
                # recurse per item but ensure it yields exactly 1 track
                ti_list = _to_time_track_list(ti, [rr])
                if len(ti_list) != 1:
                    raise ValueError("valid_orbits: each per-track t entry must map to exactly one Time array")
                out.append(ti_list[0])
            return out

        raise TypeError(f"valid_orbits: unsupported type for t: {type(t_in)}")

    # 1) normalize r.  SSAPy Orbit inputs are sampled first so the rest of
    # this long-standing helper keeps returning the same r/t list structure.
    orbit_tracks = _orbit_position_tracks(r, t)
    if orbit_tracks is not None:
        r_list, t = orbit_tracks
    else:
        r_list = _to_track_list_r(r)

    # 2) optionally drop empty r tracks and corresponding per-track t entries
    if drop_empty:
        empty_idx = [i for i, rr in enumerate(r_list) if len(rr) == 0]
        if empty_idx:
            if warn:
                print(f"valid_orbits warning: removed {len(empty_idx)} empty orbit track(s) at indices {empty_idx}")

            # If t is per-track (same length as r_list), drop corresponding t entries too
            if isinstance(t, (list, tuple)) and len(t) == len(r_list):
                t = [ti for i, ti in enumerate(t) if i not in empty_idx]

            r_list = [rr for i, rr in enumerate(r_list) if i not in empty_idx]

    # If everything is empty, return early (avoids t-shape errors)
    if len(r_list) == 0:
        if warn:
            print("valid_orbits warning: all orbit tracks were empty; returning empty lists.")
        return [], []

    # 3) normalize t against the filtered r_list
    t_list = _to_time_track_list(t, r_list)

    # 4) final length sanity
    for rr, tt in zip(r_list, t_list):
        if len(rr) != len(tt):
            raise ValueError("valid_orbits: length mismatch after normalization")

    # 5) shape print
    try:
        print(f"Returning arrays shaped: {np.shape(r_list)}, {np.shape(t_list)}")
    except Exception as e:
        print(
            "Returning arrays with varying shapes: "
            f"type(r_list)={type(r_list)}, type(t_list)={type(t_list)}, error={e}"
        )

    return r_list, t_list


def load_earth_file():
    earth = PILImage.open(earth_texture_path())
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def _require_ipyvolume():
    try:
        import ipyvolume
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Earth/Moon notebook meshes require the optional 'notebook' dependencies; "
            "install with 'pip install ssatk[notebook]'."
        ) from exc
    return ipyvolume


def drawEarth(time, ngrid=100, R=EARTH_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid : int
        Number of grid points in Earth model.
    R : float
        Earth radius in meters. Default is WGS84 value.
    rfactor : float
        Factor by which to enlarge Earth (for visualization purposes)
    """
    ipv = _require_ipyvolume()

    earth = load_earth_file()

    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Earth rotation angle for t, from UT1 as in SSAPy's groundTrack
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    from .scene_primitives import _greenwich_azimuth_rad
    gst = np.atleast_1d(_greenwich_azimuth_rad(time))

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v, wireframe=False, texture=earth
    )


def draw_earth(time, ngrid=100, R=EARTH_RADIUS, rfactor=1):
    """Draw the textured Earth mesh; snake-case alias for :func:`drawEarth`."""
    return drawEarth(time, ngrid=ngrid, R=R, rfactor=rfactor)


def load_moon_file():
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def drawMoon(time, ngrid=100, R=MOON_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid : int
        Number of grid points in Moon model.
    R : float
        Moon radius in meters.
    rfactor : float
        Factor by which to enlarge Moon (for visualization purposes)
    """
    ipv = _require_ipyvolume()

    moon = load_moon_file()

    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    # Rotate the texture by the Moon's own prime-meridian angle (DE440 lunar
    # principal axes), not Earth's sidereal angle as before.
    angle = _moon_texture_rotation_rad(time)

    u = u - (angle / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v, wireframe=False, texture=moon
    )


def _moon_texture_rotation_rad(time):
    """Azimuth (rad) in the GCRF x-y plane of the Moon's body +X axis
    (0 deg selenographic longitude) at GPS seconds ``time``.

    The texture spins about the GCRF z axis only, so the ~6.7 deg tilt of the
    lunar equator is not drawn; the prime meridian, which faces Earth within
    the optical libration (about 8 deg), is placed correctly.
    """
    from ..coordinates.lunar import lunar_body_rotation

    rotation = lunar_body_rotation(np.atleast_1d(np.asarray(time, dtype=float)))  # GCRF -> body
    x_axis = rotation[:, 0, :]  # body +X expressed in GCRF
    return np.arctan2(x_axis[:, 1], x_axis[:, 0])


def draw_moon(time, ngrid=100, R=MOON_RADIUS, rfactor=1):
    """Draw the textured Moon mesh; snake-case alias for :func:`drawMoon`."""
    return drawMoon(time, ngrid=ngrid, R=R, rfactor=rfactor)


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure, pdf_path):
    """
    Save a Matplotlib figure as a PNG embedded in a PDF file.

    If the specified PDF already exists, append a new page; otherwise create it.
    """
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1

    # Expand user directory if ~ is in the path
    if pdf_path.startswith('~'):
        pdf_path = os.path.expanduser(pdf_path)

    # Temporary PDF path
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"

    if os.path.exists(pdf_path):
        try:
            from pypdf import PdfWriter
        except ImportError as exc:
            raise ImportError(
                "Appending to an existing PDF requires pypdf; "
                "install it with `pip install ssatk[pdf]`."
            ) from exc

    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    png_buffer.seek(0)

    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)

    # Create the temporary PDF with the PNG image
    with PdfPages(temp_pdf_path) as pdf:
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')

    # Merge or move into place
    if os.path.exists(pdf_path):
        merger = PdfWriter()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        os.rename(temp_pdf_path, pdf_path)

    plt.close(figure)
    plt.close(img_fig)

    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")


def _figure_save_path(save_path=None, default_name="figure"):
    if save_path is False:
        return None
    if save_path is None or save_path is True:
        save_path = default_name

    path = Path(save_path).expanduser()
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    from .figpath import figpath
    return figpath(path)


def _save_alias_explicit(value):
    return value is not None and value is not False


def _save_alias_values_equal(left, right):
    if left is True or right is True:
        return left is True and right is True
    if (left is None or left is False) and (right is None or right is False):
        return True
    return left == right or str(left) == str(right)


def _pop_save_path_aliases(kwargs=None, save_path=None):
    """Resolve standard save-path aliases from a keyword dictionary."""
    kwargs = dict(kwargs or {})
    provided = []

    if _save_alias_explicit(save_path):
        provided.append(("save_path", save_path))

    for key in _SAVE_PATH_ALIAS_KEYS:
        if key in kwargs:
            provided.append((key, kwargs.pop(key)))

    if not provided:
        return save_path, kwargs

    first_key, first_value = provided[0]
    for key, value in provided[1:]:
        if not _save_alias_values_equal(first_value, value):
            raise TypeError(
                "Conflicting figure save aliases: "
                f"{first_key}={first_value!r} and {key}={value!r}"
            )

    return provided[-1][1], kwargs


def _raise_unrecognized_kwargs(kwargs, func_name):
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"{func_name}() got unexpected keyword argument(s): {names}")


def figsave(figure, save_path=None, dpi=200, default_name="figure", **save_kwargs):
    """
    Save a Matplotlib figure through the SSATK figure-output policy.

    Behavior:
      * If save_path is None or True -> save under figpath(default_name).
      * If save_path is False -> do not save and return None.
      * Relative paths are rooted under ~/ssatk_output/figures via figpath().
      * Absolute paths are treated as explicit user requests and used directly.
      * If save_path has no extension -> save as JPG ('.jpg' is appended).
      * If save_path ends with '.pdf' (case-insensitive) -> append/write to PDF
        via save_plot_to_pdf.
      * If save_path has any other extension -> use it directly with figure.savefig().
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "figsave")
    save_path = _figure_save_path(save_path, default_name=default_name)
    if save_path is None:
        return None

    # Split into base and extension
    base_name, extension = os.path.splitext(save_path)

    # If no extension was given, default to .jpg
    if extension == "":
        extension = ".jpg"
        save_path = base_name + extension

    # PDF: use custom handler
    if extension.lower() == ".pdf":
        save_plot_to_pdf(figure, save_path)
        return save_path

    # All other extensions: save as-is
    try:
        figure.savefig(save_path, dpi=dpi, bbox_inches=None)
        plt.close(figure)
        print(f"Figure saved at: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")
        return None


ssatk_fig = figsave


def save_plotly_figure(
    figure,
    save_path=None,
    default_name="figure",
    width=1400,
    height=1000,
    scale=1,
    **save_kwargs,
):
    """Save a Plotly figure using the SSATK figure-output policy."""
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "save_plotly_figure")
    save_path = _figure_save_path(save_path, default_name=default_name)
    if save_path is None:
        return None

    path = Path(save_path)
    if not path.suffix:
        path = path.with_suffix(".html")
        save_path = str(path)

    if path.suffix.lower() == ".html":
        figure.write_html(save_path)
    else:
        figure.write_image(save_path, width=width, height=height, scale=scale)
    print(f"Saved -> {save_path}")
    return save_path


def plotly_orbit_trace(r_km, *, name="Orbit", color="#ff4d4d", width=5, go_module=None):
    """Return a Plotly 3D line trace for an orbit trajectory in kilometres."""
    if go_module is None:
        import plotly.graph_objects as go_module
    r_km = _ensure_nx3(r_km, "r_km")
    return go_module.Scatter3d(
        x=r_km[:, 0],
        y=r_km[:, 1],
        z=r_km[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hovertemplate="X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km<extra></extra>",
        showlegend=True,
    )


def display_figure(figname, display='IPython'):
    def open_image(filename):
        if display == 'IPython':
            try:
                from IPython.display import Image as IPythonImage
                from IPython.display import display as ipython_display
            except ImportError as exc:
                raise ImportError(
                    "IPython display requires the optional 'notebook' dependencies; "
                    "install with 'pip install ssatk[notebook]'."
                ) from exc

            img = IPythonImage(filename=filename)
            ipython_display(img)
        elif display == 'PIL':
            img = PILImage.open(filename)
            img.show()
        else:
            raise ValueError("Invalid display option. Please specify 'IPython' or 'PIL'.")

    if os.path.isfile(figname):
        open_image(figname)
        return

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    for ext in image_extensions:
        filename_with_ext = figname + ext
        if os.path.isfile(filename_with_ext):
            open_image(filename_with_ext)
            return

    print("No image file found.")


def make_white(fig, *axes):
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('white')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('black')

    return fig, axes


def make_black(fig, *axes):
    fig.patch.set_facecolor('black')

    for ax in axes:
        ax.set_facecolor('black')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('white')

    return fig, axes


def draw_dashed_circle(ax, normal_vector, radius, dashes, dash_length=0.1, label='Dashed Circle'):
    # Define the circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)

    # Stack the coordinates into a matrix
    circle_points = np.vstack((x_circle, y_circle, z_circle)).T

    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rot = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)

    # Rotate the circle points
    rotated_points = circle_points @ rot.T

    # Create dashed effect
    dash_points = []
    dash_gap = int(len(theta) / dashes)
    for i in range(dashes):
        start_idx = i * dash_gap
        end_idx = start_idx + int(dash_length * len(theta))
        dash_points.append(rotated_points[start_idx:end_idx])

    # Plot the dashed circle in 3D
    for points in dash_points:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k--', label=label)
        label = None  # Only one label


def create_sphere(cx, cy, cz, r, resolution=360):
    """
    Create sphere coordinates with center (cx, cy, cz) and radius r.

    Returns
    -------
    np.ndarray of shape (3, 2*resolution, resolution)
    """
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x, y, z])


def drawSphere(xCenter, yCenter, zCenter, r, res=10j, flatten=True):
    if 'j' not in str(res):
        res = complex(0, res)
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:2 * res, 0:np.pi:res]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    if flatten:
        x = np.squeeze(np.array(x).flatten())
        y = np.squeeze(np.array(y).flatten())
        z = np.squeeze(np.array(z).flatten())
    return (x, y, z)


def darken(color, amount=0.5):
    """
    Darken a color by reducing its lightness.

    Parameters
    ----------
    color : str
        Named color or hex string.
    amount : float or iterable of floats in [0,1]
        0 -> no change, 1 -> black. Iterable returns multiple shades.

    Returns
    -------
    list of RGB tuples in 0..1
    """
    import colorsys

    # Resolve base color
    try:
        base = cnames[color]
    except Exception:
        base = color

    base_rgb = to_rgb(base)  # 0..1
    h, l, s = colorsys.rgb_to_hls(*base_rgb)

    # Normalize amount to iterable
    try:
        iterator = iter(amount)
    except TypeError:
        iterator = [amount]

    out = []
    for a in iterator:
        a = float(a)
        a = min(max(a, 0.0), 1.0)
        new_l = 1 - a * (1 - l)
        out.append(colorsys.hls_to_rgb(h, new_l, s))
    return out


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def generate_rainbow_colors(num_iterations):
    cmap = plt.get_cmap('rainbow')
    colors = [rgb2hex(cmap(i / num_iterations)) for i in range(num_iterations)]
    return colors
