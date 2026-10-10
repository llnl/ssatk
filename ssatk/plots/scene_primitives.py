"""Reusable Plotly scene primitives for SSATK 3D plots.

These helpers provide small, composable building blocks so plot scripts can add
common context — textured Earth, Moon, Sun, stars, Van Allen belts, and magnetic
field lines — without copying geometry code between demos.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ssatk.constants import (
    LD_KM,
    MOON_RADIUS_KM,
    SUN_EARTH_AVERAGE_DISTANCE_KM,
    SUN_RADIUS_KM,
)

MOON_ORBIT_RADIUS_KM = LD_KM
SUN_RADIUS_TO_DISTANCE = SUN_RADIUS_KM / SUN_EARTH_AVERAGE_DISTANCE_KM


def scene_radius_from_positions(*positions, min_radius_km=8000.0, pad=1.15):
    """Return a useful plot radius that contains all finite position arrays."""
    radius = float(min_radius_km)
    for value in positions:
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)
        norms = np.linalg.norm(arr, axis=1)
        finite = norms[np.isfinite(norms)]
        if finite.size:
            radius = max(radius, float(np.nanmax(finite)) * float(pad))
    return radius


def normalized_vector(vector, fallback=(1.0, 0.0, 0.0)):
    """Return a unit vector, using ``fallback`` for missing or zero vectors."""
    value = np.asarray(vector if vector is not None else fallback, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm == 0.0:
        value = np.asarray(fallback, dtype=float)
        norm = np.linalg.norm(value)
    return value / norm


def stabilize_sphere_poles(values, rows=1):
    """Return a copy with duplicated polar rows made longitude-independent.

    UV spheres store many vertices at each pole, one for every longitude.  If a
    texture, city-light layer, or relief field gives those coincident vertices
    different values, WebGL/Plotly can interpolate degenerate polar triangles as
    streaks or bands on the dark side of a planet.  Averaging only the exact pole
    rows keeps the texture elsewhere unchanged while making the geometry stable.
    """
    arr = np.array(values, copy=True)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return arr
    n_rows = max(0, min(int(rows), max(1, arr.shape[0] // 2)))
    for row in range(n_rows):
        arr[row, ...] = np.mean(arr[row, ...], axis=0)
        arr[-row - 1, ...] = np.mean(arr[-row - 1, ...], axis=0)
    return arr


def _coerce_datetime(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return value





def gps_seconds_at(value, index=-1, default=None):
    """Return one GPS-second sample from scalar, array, datetime, or Time input."""
    if value is None:
        return default
    gps = getattr(value, "gps", None)
    if gps is not None:
        arr = np.asarray(gps, dtype=float).reshape(-1)
    else:
        try:
            from astropy.time import Time
            if isinstance(value, str):
                arr = np.asarray(Time(value, scale="utc").gps, dtype=float).reshape(-1)
            elif isinstance(value, datetime):
                arr = np.asarray(Time(value).gps, dtype=float).reshape(-1)
            else:
                arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[int(index) % arr.size])


def earth_rotation_deg_from_time(time=None, *, epoch_jd=None, relative_seconds=0.0):
    """Return the angle (deg) to turn an Earth texture about GCRF +z at ``time``.

    The angle is the GCRF right ascension of the Greenwich meridian under the
    same GCRF->ITRF chain as SSAPy's ``groundTrack`` (precession-nutation,
    GAST at UT1, polar motion; see
    :func:`ssatk.coordinates.frames.greenwich_azimuth_rad`), so a
    texture turned by it puts the prime meridian where SSAPy's ground tracks
    do. ``epoch_jd`` plus ``relative_seconds`` supports older demo helpers that
    propagate relative seconds from a fixed Julian-date epoch.

    This used to return GAST alone. GAST is measured from the true equinox of
    date, not from the GCRF x-axis, so GCRF scenes drew Earth 1242 arcsec
    (38.4 km at the equator) away from SSAPy's ground tracks on 2026-10-09,
    growing by about 50 arcsec per year. A z-rotation still cannot carry the
    0.15 deg (2026) tilt of the true pole from GCRF +z.
    """
    if time is None and epoch_jd is None:
        return 0.0
    if time is None:
        from astropy.time import Time

        gps_seconds = float(Time(float(epoch_jd) + float(relative_seconds) / 86400.0,
                                 format="jd", scale="utc").gps)
    else:
        gps_seconds = gps_seconds_at(time, default=0.0)
    return float(np.degrees(_greenwich_azimuth_rad(gps_seconds)) % 360.0)


def _greenwich_azimuth_rad(gps_seconds):
    """GCRF right ascension (rad) of the Greenwich meridian at GPS seconds."""
    from ssatk.coordinates.frames import greenwich_azimuth_rad

    return greenwich_azimuth_rad(gps_seconds)


def sun_direction_gcrf(gps_seconds):
    """Unit vector(s) from Earth toward the Sun in GCRF at GPS seconds.

    Uses :func:`ssapy.utils.sunPos` (GCRF, within 37 arcsec of astropy's
    ``get_sun`` over 2000-2040). Returns shape ``(3,)`` for a scalar time and
    ``(N, 3)`` for an array. Plot code should take the Sun from here rather
    than carry its own solar formula.
    """
    from ssapy.utils import sunPos

    t = np.asarray(gps_seconds, dtype=float)
    r = np.asarray(sunPos(t), dtype=float)
    if t.ndim == 0:
        r = r.reshape(3)
        return r / np.linalg.norm(r)
    r = r.reshape(3, -1).T
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _gst94_rad(gps_seconds):
    """Greenwich apparent sidereal angle (rad) at GPS seconds, evaluated at UT1.

    Kept for callers that need GAST itself. Do not use it to orient a texture
    in a GCRF scene; use :func:`_greenwich_azimuth_rad`.
    """
    from erfa import gst94
    from ssapy.utils import iers_interp

    gps_seconds = np.asarray(gps_seconds, dtype=float)
    mjd_tt = 44244.0 + (gps_seconds + 51.184) / 86400.0
    d_ut1_tt_mjd, _pmx, _pmy = iers_interp(gps_seconds)
    return gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)


def _gps_seconds(value):
    if value is None:
        return 0.0
    gps = getattr(value, "gps", None)
    if gps is not None:
        return float(np.asarray(gps).flat[0])
    try:
        from astropy.time import Time
        if isinstance(value, str):
            return float(Time(value, scale="utc").gps)
        if isinstance(value, datetime):
            return float(Time(value).gps)
    except Exception:
        pass
    return float(np.asarray(value, dtype=float).flat[0])


def moon_position_gcrf_km(epoch=None):
    """Return the Moon geocentric GCRF/ECI position in kilometres."""
    t_gps = _gps_seconds(epoch)
    try:
        import ssapy.compute
        return np.asarray(ssapy.compute.moonPos(t_gps), dtype=float).reshape(-1)[:3] / 1000.0
    except Exception:
        phase = 2.0 * np.pi * (t_gps / 86_400.0 / 27.321582)
        return MOON_ORBIT_RADIUS_KM * np.array([
            np.cos(phase),
            np.sin(phase) * np.cos(np.radians(5.1)),
            np.sin(phase) * np.sin(np.radians(5.1)),
        ])


def sun_light_position_km(sun_hat=None, distance_km=SUN_EARTH_AVERAGE_DISTANCE_KM):
    """Return the physical Sun position used for lighting and eclipse geometry."""
    return normalized_vector(sun_hat) * float(distance_km)


def light_direction_from_sun(*, target_km=None, sun_position_km=None, sun_hat=None):
    """Return the unit vector from ``target_km`` toward the Sun.

    Rendering may compress the displayed Sun for readability, but body shading
    should use the physical Sun direction.  Passing a 1-AU ``sun_position_km``
    keeps Earth, Moon, and eclipse lighting tied to the same solar geometry.
    """
    if sun_position_km is None:
        return normalized_vector(sun_hat)
    target = np.zeros(3) if target_km is None else np.asarray(target_km, dtype=float)
    return normalized_vector(np.asarray(sun_position_km, dtype=float) - target, fallback=sun_hat)


def earth_trace(*, sun_hat=None, sun_position_km=None, n_lat=120, n_lon=240,
                radius_scale=1.0, center=(0.0, 0.0, 0.0), rotation_deg=None,
                time=None, show_city_lights=False, night_lift_strength=0.0,
                name="Earth"):
    """Return a textured, day/night shaded Earth mesh.

    The implementation wraps the real-texture/procedural-continent mesh used by
    the day/night globe plot.  It is intentionally exposed here as a stable
    building block so other plots do not fall back to arbitrary blue spheres.
    """
    from .globe_orbit_daynight_plotly import _earth_mesh

    if rotation_deg is None:
        rotation_deg = earth_rotation_deg_from_time(time)
    trace = _earth_mesh(
        light_direction_from_sun(target_km=center, sun_position_km=sun_position_km, sun_hat=sun_hat),
        n_lat=int(n_lat),
        n_lon=int(n_lon),
        radius_scale=float(radius_scale),
        center=center,
        rotation_deg=float(rotation_deg),
        show_city_lights=bool(show_city_lights),
        night_lift_strength=float(night_lift_strength),
    )
    trace.name = name
    return trace


def star_traces(*, scene_radius_km, when=None, frame="gcrf", mag_limit=6.5,
                opacity=0.88, fallback_random=True):
    """Return Plotly starfield traces for a 3D scene."""
    from .starfield import starfield_traces

    return starfield_traces(
        float(scene_radius_km),
        when=_coerce_datetime(when),
        frame=frame,
        mag_limit=float(mag_limit),
        opacity=float(opacity),
        fallback_random=bool(fallback_random),
    )


def sun_position_and_radius(
    *,
    scene_radius_km,
    sun_hat=None,
    position_km=None,
    distance_mode="angular",
    distance_km=None,
    distance_factor=2.5,
    radius_mode="angular",
    radius_km=None,
    radius_scale=1.0,
    match_radius_km=None,
    match_distance_km=None,
):
    """Return a Sun display position and radius for 3D scenes.

    ``distance_mode="real"`` places the Sun at one astronomical unit with its
    physical radius.  The default ``"angular"`` keeps Earth-orbit plots usable by
    placing the Sun as a distant background object while preserving the real
    solar angular radius, so the visual size is tied to physics rather than an
    arbitrary fraction of the axes.
    """
    if position_km is None:
        mode = str(distance_mode or "angular").strip().lower()
        if distance_km is not None:
            distance = float(distance_km)
        elif mode in {"real", "physical", "true"}:
            distance = SUN_EARTH_AVERAGE_DISTANCE_KM
        else:
            distance = float(scene_radius_km) * float(distance_factor)
        position = normalized_vector(sun_hat) * distance
    else:
        position = np.asarray(position_km, dtype=float)
        distance = float(np.linalg.norm(position))

    if radius_km is not None:
        radius = float(radius_km)
    else:
        mode = str(radius_mode or distance_mode or "angular").strip().lower()
        if mode in {"real", "physical", "true"}:
            radius = SUN_RADIUS_KM
        elif mode in {"match", "match_moon", "moon", "apparent_moon"} and match_radius_km and match_distance_km:
            radius = distance * float(match_radius_km) / float(match_distance_km) * float(radius_scale)
        else:
            radius = max(distance * SUN_RADIUS_TO_DISTANCE * float(radius_scale), 1.0)
    return position, radius


def sun_traces(*, scene_radius_km, sun_hat=None, position_km=None,
               distance_mode="angular", distance_km=None, distance_factor=2.5,
               radius_mode="angular", radius_km=None, radius_scale=1.0,
               match_radius_km=None, match_distance_km=None, seed=11):
    """Return Sun body/glow traces with physical or angular-correct sizing."""
    from .globe_orbit_daynight_plotly import _sun_sphere_traces

    position, radius = sun_position_and_radius(
        scene_radius_km=scene_radius_km,
        sun_hat=sun_hat,
        position_km=position_km,
        distance_mode=distance_mode,
        distance_km=distance_km,
        distance_factor=distance_factor,
        radius_mode=radius_mode,
        radius_km=radius_km,
        radius_scale=radius_scale,
        match_radius_km=match_radius_km,
        match_distance_km=match_distance_km,
    )
    return _sun_sphere_traces(np.asarray(position, dtype=float), radius, seed=seed)


def moon_trace(*, center_km, sun_hat=None, sun_position_km=None,
               radius_km=MOON_RADIUS_KM, real_center_km=None, mode="lunar",
               eclipse_tint=True, n_lat=90, n_lon=180):
    """Return a textured Moon mesh trace shaded from the Sun position."""
    from .moon_render import moon_mesh_plotly

    center = np.asarray(center_km, dtype=float)
    real_center = center if real_center_km is None else np.asarray(real_center_km, dtype=float)
    moon_sun_hat = light_direction_from_sun(
        target_km=real_center,
        sun_position_km=sun_position_km,
        sun_hat=sun_hat,
    )
    return moon_mesh_plotly(
        center,
        float(radius_km),
        sun_hat=moon_sun_hat,
        real_center_km=real_center,
        mode=mode,
        eclipse_tint=eclipse_tint,
        n_lat=int(n_lat),
        n_lon=int(n_lon),
    )


def van_allen_traces(*, show_inner=True, show_outer=True, n_pts=40):
    """Return simple Van Allen belt traces from the shared layer class."""
    import plotly.graph_objects as go
    from .layers import VanAllenLayer

    fig = go.Figure()
    VanAllenLayer(show_inner=show_inner, show_outer=show_outer, n_pts=n_pts).add_to_plotly(
        fig,
        orbit_state=None,
    )
    return list(fig.data)


def magfield_traces(*, seed_lats=None, max_r_re=None):
    """Return magnetic field-line traces from the shared layer class.

    If optional field-line dependencies are missing, the layer returns no
    traces and emits its normal warning instead of failing the caller's plot.
    """
    import plotly.graph_objects as go
    from .layers import MagfieldLayer

    fig = go.Figure()
    MagfieldLayer(seed_lats=seed_lats, max_r_re=max_r_re).add_to_plotly(
        fig,
        orbit_state=None,
    )
    return list(fig.data)


def add_traces(fig, traces):
    """Add one trace or an iterable of traces to a Plotly figure."""
    if traces is None:
        return fig
    if isinstance(traces, (list, tuple)):
        for trace in traces:
            fig.add_trace(trace)
    else:
        fig.add_trace(traces)
    return fig


def add_earth(fig, **kwargs):
    """Add the shared Earth mesh to a Plotly figure and return the trace."""
    trace = earth_trace(**kwargs)
    fig.add_trace(trace)
    return trace


def add_stars(fig, **kwargs):
    """Add shared starfield traces to a Plotly figure and return the traces."""
    traces = star_traces(**kwargs)
    add_traces(fig, traces)
    return traces


def add_sun(fig, **kwargs):
    """Add shared Sun traces to a Plotly figure and return the traces."""
    traces = sun_traces(**kwargs)
    add_traces(fig, traces)
    return traces


def add_moon(fig, **kwargs):
    """Add the shared Moon mesh to a Plotly figure and return the trace."""
    trace = moon_trace(**kwargs)
    fig.add_trace(trace)
    return trace


def add_van_allen(fig, **kwargs):
    """Add Van Allen belt traces to a Plotly figure and return the traces."""
    traces = van_allen_traces(**kwargs)
    add_traces(fig, traces)
    return traces


def add_magfield(fig, **kwargs):
    """Add magnetic field-line traces to a Plotly figure and return them."""
    traces = magfield_traces(**kwargs)
    add_traces(fig, traces)
    return traces
