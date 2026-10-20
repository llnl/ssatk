"""Backend-aware eclipse event states and provenance.

Reference contact times remain the public event scaffold.  Body states can be
provided either by the validated NASA/GSFC reconstruction or by LLNL SSAPy's
DE430-backed Sun/Moon ephemerides.  In SSAPy modes, every requested UTC sample
is evaluated directly; no interpolation of the reference body vectors is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_brightness import (
    ephemeris_positions,
    earth_shadow_axis_surface_point,  # compatibility symbol; strict code uses selection-owned transforms
    shadow_axis_surface_point,
    _gmst_rad,
)
from ssapy_toolkit.compute.eclipse_runtime import (
    BackendSelection,
    RuntimeCapabilities,
    astropy_gcrf_to_itrf_km,
    astropy_itrf_to_gcrf_km,
    resolve_backend,
    toolkit_gcrf_to_itrf_km,
    toolkit_itrf_to_gcrf_km,
)
from ssapy_toolkit.coordinates.eclipse_lunar_attitude import moon_attitude_for_event
from ssapy_toolkit.compute.eclipse_reference_events import (
    LUNAR_2025,
    RE_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    ReferenceDefinition,
    ReferenceEvent,
    angular_circle_visible_fraction,
    build_reference_event,
    build_solar_local_event,
    contact_aware_jd,
    itrf_surface_point,
    lunar_reference_state,
    solar_besselian_state,
    solar_local_contact_aware_jd,
    solar_local_contacts,
    _unit,
)


@dataclass(frozen=True)
class _ObserverSpec:
    """Private normalized observer record used by the state builder.

    The canonical public observer type lives in :mod:`observer_geometry`,
    which imports event transforms from this module.  Normalizing by protocol
    here avoids a circular import and, importantly, avoids exposing a second
    competing observer model.
    """

    latitude_deg: float
    longitude_east_deg: float
    elevation_m: float = 0.0
    name: str = "WGS-84 observer"
    model: Mapping[str, object] | None = None

    @property
    def elevation_km(self) -> float:
        return float(self.elevation_m) / 1000.0

    def to_dict(self) -> dict[str, object]:
        if self.model is not None:
            return dict(self.model)
        return {
            "latitude_deg": float(self.latitude_deg),
            "longitude_east_deg": float(self.longitude_east_deg),
            "elevation_m": float(self.elevation_m),
            "name": str(self.name),
        }


def _normalize_observer(value) -> _ObserverSpec | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        source = dict(value)
        def get(*names, default=None):
            for name in names:
                if name in source:
                    return source[name]
            return default
    else:
        source = value
        def get(*names, default=None):
            for name in names:
                if hasattr(source, name):
                    return getattr(source, name)
            return default
    latitude = get("latitude_deg", "lat_deg")
    longitude = get("longitude_east_deg", "lon_east_deg", "longitude_deg")
    if latitude is None or longitude is None:
        raise TypeError("observer requires latitude_deg and longitude_east_deg")
    elevation_m = get("elevation_m", default=None)
    if elevation_m is None:
        elevation_m = float(get("elevation_km", "height_km", default=0.0)) * 1000.0
    name = str(get("name", default="WGS-84 observer"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        model = value.to_dict()
    elif isinstance(value, Mapping):
        model = dict(value)
    else:
        model = {
            "latitude_deg": float(latitude),
            "longitude_east_deg": float(longitude),
            "elevation_m": float(elevation_m),
            "name": name,
        }
    return _ObserverSpec(
        latitude_deg=float(latitude),
        longitude_east_deg=float(longitude),
        elevation_m=float(elevation_m),
        name=name,
        model=model,
    )


def _definition(kind: str | ReferenceDefinition) -> ReferenceDefinition:
    if isinstance(kind, ReferenceDefinition):
        return kind
    key = str(kind).lower()
    if key in {"solar", "solar_2024", SOLAR_2024.key}:
        return SOLAR_2024
    if key in {"lunar", "lunar_2025", LUNAR_2025.key}:
        return LUNAR_2025
    raise ValueError("kind must identify the validated 2024 solar or 2025 lunar eclipse")


def _event_selection_metadata(selection: BackendSelection) -> dict[str, object]:
    return {
        "backend_requested": selection.requested,
        "state_source": selection.selected,
        "ephemeris_backend": selection.ephemeris_backend,
        "frame_backend": selection.frame_backend,
        "strict_backend": selection.strict,
        "backend_fallback_reason": selection.fallback_reason or "",
        "time_input_pipeline": selection.time_input,
        "runtime_capabilities": selection.capabilities.to_dict(portable=True),
        "moon_orientation_backend": (
            "llnl-ssapy DE440 binary lunar PCK"
            if selection.selected in {"ssapy", "ssapy-core"}
            else "NAIF IAU_MOON 2009 text-PCK fallback"
        ),
    }


def backend_selection_from_event(event: ReferenceEvent) -> BackendSelection:
    """Reconstruct the selected backend from immutable event provenance.

    Rendering an already-built event must not re-probe the machine or silently
    change state drivers.  This is especially important for serialized events,
    worker processes, and tests using an injected SSAPy-compatible provider.
    """
    metadata = event.metadata
    caps_dict = dict(metadata.get("runtime_capabilities", {}))
    defaults = RuntimeCapabilities(
        llnl_ssapy_importable=False, llnl_ssapy_version=None,
        astropy_importable=False, astropy_version=None,
        ssapy_ephemeris_healthy=False, ssapy_ephemeris_error="not recorded",
        ssapy_toolkit_importable=False, ssapy_toolkit_version=None,
        toolkit_transform_healthy=False, toolkit_transform_error="not recorded",
    )
    caps = RuntimeCapabilities(**{
        field: caps_dict.get(field, getattr(defaults, field))
        for field in defaults.__dataclass_fields__
    })
    requested = str(metadata.get("backend_requested", "reference"))
    selected = str(metadata.get("state_source", "reference"))
    return BackendSelection(
        requested=requested,
        selected=selected,
        ephemeris_backend=str(metadata.get("ephemeris_backend", event.backend)),
        frame_backend=str(metadata.get("frame_backend", event.frame)),
        strict=bool(metadata.get("strict_backend", requested in {"ssapy", "ssapy-core"})),
        fallback_reason=str(metadata.get("backend_fallback_reason", "")) or None,
        time_input=str(metadata.get("time_input_pipeline", "not recorded")),
        capabilities=caps,
    )


def _reference_gcrf_to_itrf_km(points_gcrf_km, jd_utc) -> np.ndarray:
    """Deterministic reference-frame rotation with no optional-package probe."""
    points = np.asarray(points_gcrf_km, dtype=float)
    scalar = points.ndim == 1
    values = points.reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(values), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(values),))
    theta = _gmst_rad(jd)
    ct, st = np.cos(theta), np.sin(theta)
    result = np.column_stack([
        ct*values[:, 0] + st*values[:, 1],
        -st*values[:, 0] + ct*values[:, 1],
        values[:, 2],
    ])
    return result[0] if scalar else result


def _reference_itrf_to_gcrf_km(points_itrf_km, jd_utc) -> np.ndarray:
    """Inverse deterministic reference rotation with no optional fallback."""
    points = np.asarray(points_itrf_km, dtype=float)
    scalar = points.ndim == 1
    values = points.reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(values), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(values),))
    theta = _gmst_rad(jd)
    ct, st = np.cos(theta), np.sin(theta)
    result = np.column_stack([
        ct*values[:, 0] - st*values[:, 1],
        st*values[:, 0] + ct*values[:, 1],
        values[:, 2],
    ])
    return result[0] if scalar else result


def _selection_gcrf_to_itrf_km(selection: BackendSelection, points, jd_utc) -> np.ndarray:
    if selection.selected == "ssapy":
        return toolkit_gcrf_to_itrf_km(points, jd_utc)
    if selection.selected == "ssapy-core":
        return astropy_gcrf_to_itrf_km(points, jd_utc)
    return _reference_gcrf_to_itrf_km(points, jd_utc)


def _selection_itrf_to_gcrf_km(selection: BackendSelection, points, jd_utc) -> np.ndarray:
    if selection.selected == "ssapy":
        return toolkit_itrf_to_gcrf_km(points, jd_utc)
    if selection.selected == "ssapy-core":
        return astropy_itrf_to_gcrf_km(points, jd_utc)
    return _reference_itrf_to_gcrf_km(points, jd_utc)


def event_gcrf_to_itrf_km(event: ReferenceEvent, points, jd_utc) -> np.ndarray:
    """Transform with the immutable frame backend recorded on ``event``.

    No runtime capability probe or fallback occurs here.  Strict SSAPy and
    SSAPy-core events therefore remain strict throughout rendering, while a
    reference event remains byte-for-byte deterministic even on a workstation
    that happens to have Astropy or SSAPy-Toolkit installed.
    """
    return _selection_gcrf_to_itrf_km(backend_selection_from_event(event), points, jd_utc)


def event_itrf_to_gcrf_km(event: ReferenceEvent, points, jd_utc) -> np.ndarray:
    """Inverse transform owned by the backend recorded on ``event``."""
    return _selection_itrf_to_gcrf_km(backend_selection_from_event(event), points, jd_utc)


def _interpolate_serialized_positions(event: ReferenceEvent, jd_utc: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate positions stored on a dynamic serialized event.

    Dynamic arbitrary-event records are self-contained.  They must not query an
    optional discovery package again after serialization.  Cubic splines are
    used when enough states are available; exact stored samples remain exact to
    floating-point tolerance.  Values outside the serialized interval are
    rejected rather than silently extrapolated.
    """
    value = float(jd_utc)
    times = np.asarray(event.jd, dtype=float)
    if value < float(times[0]) - 1.0e-12 or value > float(times[-1]) + 1.0e-12:
        raise ValueError("requested UTC lies outside the serialized dynamic-event interval")
    index = np.flatnonzero(np.isclose(times, value, rtol=0.0, atol=2.0e-13))
    if len(index):
        i = int(index[0])
        return np.asarray(event.sun_km[i], dtype=float), np.asarray(event.moon_km[i], dtype=float)
    try:
        from scipy.interpolate import CubicSpline
        if len(times) >= 4:
            sun = CubicSpline(times, np.asarray(event.sun_km, dtype=float), axis=0)(value)
            moon = CubicSpline(times, np.asarray(event.moon_km, dtype=float), axis=0)(value)
            return np.asarray(sun, dtype=float), np.asarray(moon, dtype=float)
    except Exception:
        pass
    sun = np.array([np.interp(value, times, np.asarray(event.sun_km)[:, axis]) for axis in range(3)])
    moon = np.array([np.interp(value, times, np.asarray(event.moon_km)[:, axis]) for axis in range(3)])
    return sun, moon


def resample_event(event: ReferenceEvent, jd_utc: Sequence[float]) -> ReferenceEvent:
    """Return the same immutable event evaluated on a new in-range timeline."""
    from ssapy_toolkit.compute.eclipse_reference_events import angular_circle_visible_fraction, R_MOON_MEAN_KM
    values = np.asarray(jd_utc, dtype=float).reshape(-1)
    positions = [_interpolate_serialized_positions(event, float(value))
                 if bool(event.metadata.get("dynamic_event", False))
                 else event_positions_gcrf(event, float(value))
                 for value in values]
    sun = np.asarray([item[0] for item in positions])
    moon = np.asarray([item[1] for item in positions])
    visible, separation = [], []
    for sun_i, moon_i in zip(sun, moon):
        if event.mode == "solar":
            ds, dm = float(np.linalg.norm(sun_i)), float(np.linalg.norm(moon_i))
            sep = math.acos(np.clip(np.dot(sun_i/ds, moon_i/dm), -1.0, 1.0))
            visible.append(float(angular_circle_visible_fraction(
                math.asin(R_MOON_MEAN_KM/dm), math.asin(R_SUN_KM/ds), sep)))
            separation.append(math.degrees(sep))
        else:
            to_earth, to_sun = -moon_i, sun_i-moon_i
            de, ds = float(np.linalg.norm(to_earth)), float(np.linalg.norm(to_sun))
            sep = math.acos(np.clip(np.dot(to_earth/de, to_sun/ds), -1.0, 1.0))
            visible.append(float(angular_circle_visible_fraction(
                math.asin(RE_KM*1.01/de), math.asin(R_SUN_KM/ds), sep)))
            separation.append(math.degrees(math.pi-sep))
    return replace(event, jd=values, sun_km=sun, moon_km=moon,
                   center_visibility=np.asarray(visible), separation_deg=np.asarray(separation))


def event_positions_gcrf(event: ReferenceEvent, jd_utc: float) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric GCRF ``(sun, moon)`` for one event instant."""
    if bool(event.metadata.get("dynamic_event", False)):
        return _interpolate_serialized_positions(event, float(jd_utc))
    source = str(event.metadata.get("state_source", "reference"))
    if source in {"ssapy", "ssapy-core"}:
        ephem = ephemeris_positions([float(jd_utc)], backend="ssapy")
        return ephem.sun_gcrf_km[0], ephem.moon_gcrf_km[0]

    if event.mode == "solar":
        state = solar_besselian_state(float(jd_utc))
        points = np.asarray([state.sun_itrf_km, state.moon_itrf_km], dtype=float)
        gcrf = event_itrf_to_gcrf_km(
            event, points, np.asarray([jd_utc, jd_utc], dtype=float)
        )
        return np.asarray(gcrf[0], dtype=float), np.asarray(gcrf[1], dtype=float)
    state = lunar_reference_state(float(jd_utc))
    return np.asarray(state.sun_gcrf_km, dtype=float), np.asarray(state.moon_gcrf_km, dtype=float)


def event_moon_body_to_gcrf(event: ReferenceEvent, jd_utc) -> np.ndarray:
    """Moon body-fixed-to-GCRF attitude owned by the event backend.

    Strict SSAPy modes use the DE440 binary lunar PCK exposed by
    ``get_body("moon").orientation``.  The deterministic reference mode uses
    the documented NAIF IAU_MOON 2009 trigonometric model.  No renderer-level
    fallback or synchronous-look-at approximation is permitted here.
    """
    return np.asarray(moon_attitude_for_event(event, float(jd_utc)).body_to_gcrf, dtype=float)


def _observer_visibility(
    observer_gcrf_km: np.ndarray,
    moon_gcrf_km: np.ndarray,
    sun_gcrf_km: np.ndarray,
    moon_radius_km: float = SOLAR_UMBRA_OPTICAL_RADIUS_KM,
) -> tuple[float, float]:
    to_moon = np.asarray(moon_gcrf_km, dtype=float)-observer_gcrf_km
    to_sun = np.asarray(sun_gcrf_km, dtype=float)-observer_gcrf_km
    dm = float(np.linalg.norm(to_moon))
    ds = float(np.linalg.norm(to_sun))
    separation = math.acos(float(np.clip(
        np.dot(np.ravel(_unit(to_moon)), np.ravel(_unit(to_sun))), -1.0, 1.0)))
    moon_ang = math.asin(np.clip(float(moon_radius_km)/dm, 0.0, 1.0))
    sun_ang = math.asin(np.clip(R_SUN_KM/ds, 0.0, 1.0))
    visible = float(angular_circle_visible_fraction(moon_ang, sun_ang, separation))
    return visible, math.degrees(separation)


def _lunar_center_visibility(
    moon_gcrf_km: np.ndarray,
    sun_gcrf_km: np.ndarray,
    effective_earth_radius_km: float,
) -> tuple[float, float]:
    to_earth = -np.asarray(moon_gcrf_km, dtype=float)
    to_sun = np.asarray(sun_gcrf_km, dtype=float)-moon_gcrf_km
    de = float(np.linalg.norm(to_earth))
    ds = float(np.linalg.norm(to_sun))
    separation = math.acos(np.clip(np.dot(_unit(to_earth), _unit(to_sun)), -1.0, 1.0))
    earth_ang = math.asin(np.clip(effective_earth_radius_km/de, 0.0, 1.0))
    sun_ang = math.asin(np.clip(R_SUN_KM/ds, 0.0, 1.0))
    visible = float(angular_circle_visible_fraction(earth_ang, sun_ang, separation))
    # Lunar plots conventionally report offset from the anti-solar shadow axis.
    shadow_offset = math.degrees(math.pi-separation)
    return visible, shadow_offset


def build_event(
    kind: str | ReferenceDefinition = "solar",
    *,
    backend: str = "auto",
    n_frames: int = 121,
    jd: Sequence[float] | None = None,
    solar_scope: str = "global",
    observer: object | Mapping[str, object] | None = None,
) -> ReferenceEvent:
    """Build a backend-aware real eclipse event.

    Parameters
    ----------
    backend
        ``reference`` uses the validated NASA/GSFC reconstruction.
        ``auto`` selects a healthy full LLNL runtime, then SSAPy core with
        Astropy frames, then the reference state.
        ``ssapy`` is fail-fast and requires both LLNL SSAPy and Toolkit.
        ``ssapy-core`` is fail-fast and requires SSAPy ephemerides; Astropy is
        used for terrestrial frames.
    solar_scope
        ``local`` uses the NASA greatest-eclipse site C1-C4 timeline;
        ``global`` uses P1-P4.
    """
    definition = _definition(kind)
    selection = resolve_backend(backend)
    local_solar = definition.mode == "solar" and str(solar_scope).lower() == "local"
    observer_site = _normalize_observer(observer)
    if local_solar and observer_site is None:
        observer_site = _ObserverSpec(
            SOLAR_GREATEST_SITE_LAT_DEG, SOLAR_GREATEST_SITE_LON_EAST_DEG,
            name="NASA greatest-eclipse reference site",
            model={
                "latitude_deg": SOLAR_GREATEST_SITE_LAT_DEG,
                "longitude_east_deg": SOLAR_GREATEST_SITE_LON_EAST_DEG,
                "elevation_m": 0.0,
                "pressure_hpa": 1010.0,
                "temperature_c": 10.0,
                "apply_refraction": True,
                "name": "NASA greatest-eclipse reference site",
                "horizon": {"source": "flat geometric horizon", "sample_count": 2},
            },
        )

    if selection.selected == "reference":
        if local_solar and jd is None:
            assert observer_site is not None
            event = build_solar_local_event(
                n_frames=max(25, int(n_frames)),
                lat_deg=observer_site.latitude_deg,
                lon_east_deg=observer_site.longitude_east_deg,
                height_km=observer_site.elevation_km,
                observer_name=observer_site.name,
            )
        else:
            event = build_reference_event(definition, n_frames=n_frames, jd=jd)
        metadata = dict(event.metadata)
        metadata.update(_event_selection_metadata(selection))
        metadata["solar_scope"] = "local" if local_solar else "global"
        if observer_site is not None and local_solar:
            contacts = solar_local_contacts(
                observer_site.latitude_deg, observer_site.longitude_east_deg, observer_site.elevation_km
            )
            metadata.update({
                "observer_name": observer_site.name,
                "observer_lat_deg": observer_site.latitude_deg,
                "observer_lon_east_deg": observer_site.longitude_east_deg,
                "observer_height_km": observer_site.elevation_km,
                "observer_elevation_m": observer_site.elevation_m,
                "observer_model": observer_site.to_dict(),
                "observer_contacts_jd": contacts,
                "local_max_jd": contacts.get("MAX", definition.greatest_jd),
                "observer_contact_backend": "NASA/GSFC Besselian reference scaffold",
            })
        return replace(event, metadata=metadata)

    if jd is None:
        times = (
            solar_local_contact_aware_jd(
                max(25, int(n_frames)),
                lat_deg=observer_site.latitude_deg,
                lon_east_deg=observer_site.longitude_east_deg,
                height_km=observer_site.elevation_km,
            )
            if local_solar
            else contact_aware_jd(definition, n_frames=max(25, int(n_frames)))
        )
    else:
        times = np.asarray(jd, dtype=float)

    ephem = ephemeris_positions(times, backend="ssapy")
    sun = np.asarray(ephem.sun_gcrf_km, dtype=float)
    moon = np.asarray(ephem.moon_gcrf_km, dtype=float)
    visible: list[float] = []
    separation: list[float] = []

    reference = build_reference_event(definition, jd=times)
    reference_sun = []
    reference_moon = []
    for value in times:
        rs, rm = event_positions_gcrf(
            replace(reference, metadata={**dict(reference.metadata), "state_source": "reference"}),
            float(value),
        )
        reference_sun.append(rs)
        reference_moon.append(rm)
    reference_sun_arr = np.asarray(reference_sun)
    reference_moon_arr = np.asarray(reference_moon)

    if definition.mode == "solar":
        if local_solar:
            assert observer_site is not None
        fixed_observer_itrf = itrf_surface_point(
            observer_site.latitude_deg if observer_site is not None else SOLAR_GREATEST_SITE_LAT_DEG,
            observer_site.longitude_east_deg if observer_site is not None else SOLAR_GREATEST_SITE_LON_EAST_DEG,
            observer_site.elevation_km if observer_site is not None else 0.0,
        )
        for value, sun_i, moon_i in zip(times, sun, moon):
            if local_solar:
                observer_gcrf = _selection_itrf_to_gcrf_km(
                    selection, fixed_observer_itrf, float(value)
                )
            else:
                moon_itrf, sun_itrf = _selection_gcrf_to_itrf_km(
                    selection,
                    np.asarray([moon_i, sun_i], dtype=float),
                    np.asarray([value, value], dtype=float),
                )
                hit_itrf = shadow_axis_surface_point(
                    moon_itrf, sun_itrf, target_axes_km=(RE_KM, RE_KM, 6_356.752314245),
                )
                hit = (
                    None if hit_itrf is None else
                    _selection_itrf_to_gcrf_km(selection, hit_itrf, float(value))
                )
                observer_gcrf = np.zeros(3) if hit is None else np.asarray(hit, dtype=float)
            vis, sep = _observer_visibility(np.asarray(observer_gcrf), moon_i, sun_i)
            visible.append(vis)
            separation.append(sep)
    else:
        enlargement = float(reference.metadata.get("danjon_shadow_enlargement", 1.02))
        effective_radius = RE_KM*enlargement
        for sun_i, moon_i in zip(sun, moon):
            vis, sep = _lunar_center_visibility(moon_i, sun_i, effective_radius)
            visible.append(vis)
            separation.append(sep)

    metadata = dict(reference.metadata)
    metadata.update(_event_selection_metadata(selection))
    metadata.update({
        "solar_scope": "local" if local_solar else "global",
        "reference_contact_scaffold": definition.source_label,
        "ssapy_sun_reference_rms_km": float(np.sqrt(np.mean(np.sum((sun-reference_sun_arr)**2, axis=1)))),
        "ssapy_moon_reference_rms_km": float(np.sqrt(np.mean(np.sum((moon-reference_moon_arr)**2, axis=1)))),
        "ssapy_sun_reference_max_km": float(np.max(np.linalg.norm(sun-reference_sun_arr, axis=1))),
        "ssapy_moon_reference_max_km": float(np.max(np.linalg.norm(moon-reference_moon_arr, axis=1))),
    })
    if local_solar:
        assert observer_site is not None
        contacts = solar_local_contacts(
            observer_site.latitude_deg, observer_site.longitude_east_deg, observer_site.elevation_km
        )
        metadata.update({
            "animation_scope": f"{observer_site.name}, local C1-C4",
            "observer_name": observer_site.name,
            "observer_lat_deg": observer_site.latitude_deg,
            "observer_lon_east_deg": observer_site.longitude_east_deg,
            "observer_height_km": observer_site.elevation_km,
            "observer_elevation_m": observer_site.elevation_m,
            "observer_model": observer_site.to_dict(),
            "observer_contacts_jd": contacts,
            "local_max_jd": contacts.get("MAX", definition.greatest_jd),
            "observer_contact_backend": "NASA/GSFC Besselian reference scaffold",
        })

    return ReferenceEvent(
        definition=definition,
        jd=np.asarray(times, dtype=float),
        moon_km=moon,
        sun_km=sun,
        frame="GCRF (LLNL SSAPy DE430 state)",
        backend=(
            "llnl-ssapy/DE430 + ssapy-toolkit"
            if selection.selected == "ssapy"
            else "llnl-ssapy/DE430 + Astropy frames"
        ),
        center_visibility=np.asarray(visible, dtype=float),
        separation_deg=np.asarray(separation, dtype=float),
        metadata=metadata,
    )
