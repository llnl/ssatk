"""Eclipse event states for the two state backends.

Published contact times are the event timeline. Body states come either from
the NASA/GSFC reconstruction (``reference``) or from SSAPy's DE430 Sun/Moon
ephemerides (``ssapy``), evaluated directly at every sample. Both backends use
the same SSATK GCRF/ITRF transforms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence
import math

import numpy as np

from ssatk.compute.eclipse_brightness import shadow_axis_surface_point
from ssatk.compute.eclipse_runtime import (
    BackendSelection,
    gcrf_to_itrf_km,
    itrf_to_gcrf_km,
    resolve_backend,
    ssapy_ephemeris_positions_km,
)
from ssatk.coordinates.eclipse_lunar_attitude import moon_attitude_for_event
from ssatk.compute.eclipse_reference_events import (
    LUNAR_2025,
    RE_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    ReferenceDefinition,
    ReferenceEvent,
    RP_KM,
    angular_circle_visible_fraction,
    build_reference_event,
    center_visibility_and_separation,
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
        "moon_orientation_backend": (
            "SSAPy DE440 binary lunar PCK" if selection.selected == "ssapy"
            else "NAIF IAU_MOON 2009 rotation model"
        ),
    }


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
    values = np.asarray(jd_utc, dtype=float).reshape(-1)
    positions = [_interpolate_serialized_positions(event, float(value))
                 if bool(event.metadata.get("dynamic_event", False))
                 else event_positions_gcrf(event, float(value))
                 for value in values]
    sun = np.asarray([item[0] for item in positions])
    moon = np.asarray([item[1] for item in positions])
    visible, separation = (
        solar_visibility_and_separation(values, sun, moon) if event.mode == "solar"
        else center_visibility_and_separation("lunar", sun, moon)
    )
    # event_positions_gcrf returns GCRF even for ITRF-stored reference solar states.
    return replace(event, jd=values, sun_km=sun, moon_km=moon, frame="GCRF",
                   center_visibility=np.asarray(visible), separation_deg=np.asarray(separation))


def event_positions_gcrf(event: ReferenceEvent, jd_utc: float) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric GCRF ``(sun, moon)`` for one event instant."""
    if bool(event.metadata.get("dynamic_event", False)):
        return _interpolate_serialized_positions(event, float(jd_utc))
    source = str(event.metadata.get("state_source", "reference"))
    if source == "ssapy":
        moon, sun = ssapy_ephemeris_positions_km([float(jd_utc)])
        return sun[0], moon[0]

    if event.mode == "solar":
        state = solar_besselian_state(float(jd_utc))
        points = np.asarray([state.sun_itrf_km, state.moon_itrf_km], dtype=float)
        gcrf = itrf_to_gcrf_km(points, float(jd_utc))
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


def solar_visibility_and_separation(times, sun_km, moon_km, *, observer_itrf_km=None):
    """Visible solar fraction and Sun-Moon separation seen from the ground.

    A fixed observer is used when ``observer_itrf_km`` is given; otherwise each
    state is seen from the point where the shadow axis meets the WGS-84
    surface, or from Earth's center while the axis misses Earth.
    """
    visible, separation = [], []
    for value, sun_i, moon_i in zip(np.asarray(times, dtype=float), sun_km, moon_km):
        if observer_itrf_km is not None:
            observer_gcrf = itrf_to_gcrf_km(observer_itrf_km, float(value))
        else:
            moon_itrf, sun_itrf = gcrf_to_itrf_km(np.asarray([moon_i, sun_i]), float(value))
            hit_itrf = shadow_axis_surface_point(moon_itrf, sun_itrf, target_axes_km=(RE_KM, RE_KM, RP_KM))
            observer_gcrf = np.zeros(3) if hit_itrf is None else itrf_to_gcrf_km(hit_itrf, float(value))
        vis, sep = _observer_visibility(np.asarray(observer_gcrf), moon_i, sun_i)
        visible.append(vis)
        separation.append(sep)
    return np.asarray(visible), np.asarray(separation)


def build_event(
    kind: str | ReferenceDefinition = "solar",
    *,
    backend: str = "ssapy",
    n_frames: int = 121,
    jd: Sequence[float] | None = None,
    solar_scope: str = "global",
    observer: object | Mapping[str, object] | None = None,
) -> ReferenceEvent:
    """Build a backend-aware real eclipse event.

    Parameters
    ----------
    backend
        ``ssapy`` evaluates SSAPy DE430 Sun/Moon states at every sample;
        ``reference`` replays the NASA/GSFC Besselian elements (solar) or
        published circumstances (lunar).
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

    moon, sun = ssapy_ephemeris_positions_km(times)
    reference = build_reference_event(definition, jd=times)
    if definition.mode == "solar":
        observer_itrf = itrf_surface_point(
            observer_site.latitude_deg if observer_site is not None else SOLAR_GREATEST_SITE_LAT_DEG,
            observer_site.longitude_east_deg if observer_site is not None else SOLAR_GREATEST_SITE_LON_EAST_DEG,
            observer_site.elevation_km if observer_site is not None else 0.0,
        )
        visible_arr, separation_arr = solar_visibility_and_separation(
            times, sun, moon, observer_itrf_km=observer_itrf if local_solar else None,
        )
    else:
        visible_arr, separation_arr = center_visibility_and_separation("lunar", sun, moon)

    metadata = dict(reference.metadata)
    metadata.update(_event_selection_metadata(selection))
    metadata.update({
        "solar_scope": "local" if local_solar else "global",
        "reference_contact_scaffold": definition.source_label,
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
        frame="GCRF",
        backend=selection.ephemeris_backend,
        center_visibility=visible_arr,
        separation_deg=separation_arr,
        metadata=metadata,
    )
