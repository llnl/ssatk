"""Numerical solar and lunar eclipse discovery from Sun/Moon state vectors.

Discovery searches minima of a finite-cone target margin and root-finds the
external and internal contact equations. ``discover_eclipses`` uses DE430
states; ``discover_with_position_provider`` accepts any provider so other
ephemerides can be compared against it. A discovered event materializes as
the same ``ReferenceEvent`` the ray tracer and renderers consume, with every
body state stored so it stays reproducible after serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence
import json

import numpy as np

from ssatk.compute.eclipse_core import shadow_cone
from ssatk.compute.eclipse_runtime import (
    gcrf_to_itrf_km,
    itrf_to_gcrf_km,
    resolve_backend,
    ephemeris_positions_km,
)
from ssatk.compute.eclipse_brightness import shadow_axis_surface_point
from ssatk.compute.eclipse_state import solar_visibility_and_separation
from ssatk.compute.eclipse_reference_events import (
    LUNAR_DANJON_FACTOR,
    RE_KM,
    RP_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    geodetic_from_itrf,
    itrf_surface_point,
    ReferenceDefinition,
    ReferenceEvent,
    center_visibility_and_separation,
    datetime_to_jd,
    jd_to_datetime,
)


class PositionProvider(Protocol):
    """Geocentric equatorial position provider in kilometres."""

    name: str

    def positions_gcrf_km(self, jd_utc: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sun, moon)`` arrays with shape ``(N, 3)``."""


@dataclass(frozen=True)
class DiscoveredEclipse:
    key: str
    mode: str
    eclipse_type: str
    greatest_jd_utc: float
    contacts_jd_utc: Mapping[str, float]
    discovery_backend: str
    source_label: str
    central: bool = False
    greatest_lat_deg: float | None = None
    greatest_lon_east_deg: float | None = None
    magnitude: float | None = None
    obscuration: float | None = None
    shadow_width_km: float | None = None
    saros_series: int | None = None
    saros_member: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def greatest_utc(self) -> datetime:
        return jd_to_datetime(float(self.greatest_jd_utc))

    def to_definition(self) -> ReferenceDefinition:
        label = self.eclipse_type.replace("-", " ").title()
        title = f"{label} {'Solar' if self.mode == 'solar' else 'Lunar'} Eclipse — {self.greatest_utc:%Y-%m-%d}"
        contacts = {
            str(name): jd_to_datetime(float(value))
            for name, value in sorted(self.contacts_jd_utc.items(), key=lambda item: float(item[1]))
        }
        if "MAX" not in contacts:
            contacts["MAX"] = self.greatest_utc
        return ReferenceDefinition(
            key=self.key,
            mode=self.mode,
            title=title,
            source_label=self.source_label,
            contacts_utc=contacts,
            greatest_utc=self.greatest_utc,
            notes=(
                f"Discovered with {self.discovery_backend}; backend provenance is preserved in the event state.",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "$schema": "ssatk.eclipse.discovery/1",
            "key": self.key,
            "mode": self.mode,
            "eclipse_type": self.eclipse_type,
            "greatest_jd_utc": float(self.greatest_jd_utc),
            "greatest_utc": self.greatest_utc.isoformat().replace("+00:00", "Z"),
            "contacts_jd_utc": {k: float(v) for k, v in self.contacts_jd_utc.items()},
            "contacts_utc": {
                k: jd_to_datetime(float(v)).isoformat().replace("+00:00", "Z")
                for k, v in self.contacts_jd_utc.items()
            },
            "discovery_backend": self.discovery_backend,
            "source_label": self.source_label,
            "central": bool(self.central),
            "greatest_lat_deg": self.greatest_lat_deg,
            "greatest_lon_east_deg": self.greatest_lon_east_deg,
            "magnitude": self.magnitude,
            "obscuration": self.obscuration,
            "shadow_width_km": self.shadow_width_km,
            "saros_series": self.saros_series,
            "saros_member": self.saros_member,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DiscoveredEclipse":
        return cls(
            key=str(payload["key"]),
            mode=str(payload["mode"]),
            eclipse_type=str(payload["eclipse_type"]),
            greatest_jd_utc=float(payload["greatest_jd_utc"]),
            contacts_jd_utc={str(k): float(v) for k, v in dict(payload["contacts_jd_utc"]).items()},
            discovery_backend=str(payload["discovery_backend"]),
            source_label=str(payload["source_label"]),
            central=bool(payload.get("central", False)),
            greatest_lat_deg=_optional_float(payload.get("greatest_lat_deg")),
            greatest_lon_east_deg=_optional_float(payload.get("greatest_lon_east_deg")),
            magnitude=_optional_float(payload.get("magnitude")),
            obscuration=_optional_float(payload.get("obscuration")),
            shadow_width_km=_optional_float(payload.get("shadow_width_km")),
            saros_series=_optional_int(payload.get("saros_series")),
            saros_member=_optional_int(payload.get("saros_member")),
            metadata=dict(payload.get("metadata", {})),
        )


def _optional_float(value):
    return None if value is None else float(value)


def _optional_int(value):
    if value is None:
        return None
    result = int(round(float(value)))
    return None if result < -1_000_000 else result


def _coerce_jd(value: float | str | datetime) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return datetime_to_jd(dt.astimezone(timezone.utc))
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid UTC date/time {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime_to_jd(dt.astimezone(timezone.utc))


def _key(mode: str, jd: float) -> str:
    dt = jd_to_datetime(float(jd))
    return f"{mode}_{dt:%Y_%m_%d}"


def _nonzero_contacts(mapping: Mapping[str, float]) -> dict[str, float]:
    result = {name: float(value) for name, value in mapping.items() if float(value) > 0.0}
    return dict(sorted(result.items(), key=lambda item: item[1]))


def _provider_positions(provider: PositionProvider | Callable, jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(provider, "positions_gcrf_km"):
        sun, moon = provider.positions_gcrf_km(jd)
    else:
        sun, moon = provider(jd)
    sun = np.asarray(sun, dtype=float).reshape(-1, 3)
    moon = np.asarray(moon, dtype=float).reshape(-1, 3)
    if sun.shape != moon.shape or sun.shape[0] != len(jd):
        raise ValueError("position provider must return Sun and Moon arrays with shape (N, 3)")
    return sun, moon


def _solar_shadow_geometry(sun: np.ndarray, moon: np.ndarray) -> tuple[float, float, float, float]:
    """Return axis miss, downstream distance, penumbra radius, signed umbra radius."""
    source_to_occ = moon - sun
    distance = float(np.linalg.norm(source_to_occ))
    direction = source_to_occ / distance
    downstream = max(0.0, -float(np.dot(moon, direction)))
    miss = float(np.linalg.norm(moon + downstream * direction))
    cone = shadow_cone(R_SUN_KM, R_MOON_MEAN_KM, distance)
    return miss, downstream, float(cone.penumbra_radius(downstream)), float(cone.umbra_radius(downstream))


def _lunar_shadow_geometry(sun: np.ndarray, moon: np.ndarray) -> tuple[float, float, float, float]:
    distance = float(np.linalg.norm(sun))
    direction = -sun / distance
    downstream = float(np.dot(moon, direction))
    miss = float(np.linalg.norm(moon - downstream * direction))
    if downstream <= 0.0:
        # At new Moon the Moon is sunward of Earth, so Earth's anti-solar
        # shadow cannot reach it. Negative radii keep the margin positive.
        return miss, downstream, -1.0e9, -1.0e9
    # The Danjon enlargement of Earth's shadow applies to lunar contacts only.
    cone = shadow_cone(R_SUN_KM, RE_KM * LUNAR_DANJON_FACTOR, distance)
    return miss, downstream, float(cone.penumbra_radius(downstream)), float(cone.umbra_radius(downstream))


def _solar_surface_apparent_geometry(
    sun: np.ndarray,
    moon: np.ndarray,
    jd_utc: float,
) -> tuple[float, float, float, float, float]:
    """Return topocentric solar disc radii at the closest WGS-84 surface view.

    The view is selected by minimizing apparent centre separation over the
    ellipsoid.  This is the standard projected-diameter definition for a
    partial eclipse and also supplies the radius sign needed to distinguish a
    hybrid central path.
    """
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for surface eclipse geometry") from exc

    sun_itrf, moon_itrf = gcrf_to_itrf_km(np.asarray([sun, moon]), float(jd_utc))
    axis = moon_itrf - sun_itrf
    axis /= np.linalg.norm(axis)
    downstream = max(0.0, -float(np.dot(moon_itrf, axis)))
    closest = moon_itrf + downstream * axis
    norm = np.linalg.norm(closest)
    initial = (
        np.degrees(np.arcsin(np.clip(closest[2] / norm, -1.0, 1.0))),
        np.degrees(np.arctan2(closest[1], closest[0])),
    ) if norm > 0.0 else (0.0, 0.0)

    def geometry(latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
        observer_itrf = itrf_surface_point(latitude_deg, longitude_deg, 0.0)
        observer = itrf_to_gcrf_km(observer_itrf, float(jd_utc))
        to_sun = np.asarray(sun, dtype=float) - observer
        to_moon = np.asarray(moon, dtype=float) - observer
        sun_distance = float(np.linalg.norm(to_sun))
        moon_distance = float(np.linalg.norm(to_moon))
        separation = float(np.arccos(np.clip(
            np.dot(to_sun, to_moon) / (sun_distance * moon_distance), -1.0, 1.0,
        )))
        return (
            separation,
            float(np.arcsin(np.clip(R_SUN_KM / sun_distance, 0.0, 1.0))),
            float(np.arcsin(np.clip(R_MOON_MEAN_KM / moon_distance, 0.0, 1.0))),
        )

    def objective(values: np.ndarray) -> float:
        latitude = float(np.clip(values[0], -90.0, 90.0))
        longitude = float(values[1])
        return geometry(latitude, longitude)[0]

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1.0e-9, "fatol": 1.0e-15, "maxiter": 1000},
    )
    latitude = float(np.clip(result.x[0], -90.0, 90.0))
    longitude = float(result.x[1])
    separation, sun_radius, moon_radius = geometry(latitude, longitude)
    return moon_radius, sun_radius, separation, latitude, longitude


def _solar_central_apparent_geometry(
    sun: np.ndarray,
    moon: np.ndarray,
    jd_utc: float,
) -> tuple[float, float, float, float, float]:
    """Return apparent radii at the WGS-84 shadow-axis surface intersection."""
    sun_itrf, moon_itrf = gcrf_to_itrf_km(np.asarray([sun, moon]), float(jd_utc))
    point = shadow_axis_surface_point(
        moon_itrf, sun_itrf, target_axes_km=(RE_KM, RE_KM, RP_KM),
    )
    if point is None:
        return _solar_surface_apparent_geometry(sun, moon, jd_utc)
    observer = itrf_to_gcrf_km(point, float(jd_utc))
    to_sun = np.asarray(sun, dtype=float) - observer
    to_moon = np.asarray(moon, dtype=float) - observer
    sun_distance = float(np.linalg.norm(to_sun))
    moon_distance = float(np.linalg.norm(to_moon))
    separation = float(np.arccos(np.clip(
        np.dot(to_sun, to_moon) / (sun_distance * moon_distance), -1.0, 1.0,
    )))
    sun_radius = float(np.arcsin(np.clip(R_SUN_KM / sun_distance, 0.0, 1.0)))
    moon_radius = float(np.arcsin(np.clip(R_MOON_MEAN_KM / moon_distance, 0.0, 1.0)))
    latitude, longitude, _ = geodetic_from_itrf(point)
    return moon_radius, sun_radius, separation, latitude, longitude


def _roots(function: Callable[[float], float], start: float, stop: float, *, samples: int = 721) -> list[float]:
    try:
        from scipy.optimize import brentq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for numerical eclipse discovery") from exc
    grid = np.linspace(float(start), float(stop), int(samples))
    values = np.asarray([function(float(value)) for value in grid], dtype=float)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if abs(float(f_left)) < 1.0e-12:
            root = float(left)
        elif f_left * f_right < 0.0:
            root = float(brentq(function, float(left), float(right), xtol=1.0e-11))
        else:
            continue
        if not roots or abs(root - roots[-1]) > 0.5 / 86400.0:
            roots.append(root)
    return roots


def discover_with_position_provider(
    start: float | str | datetime,
    end: float | str | datetime,
    *,
    mode: str,
    provider: PositionProvider | Callable,
    provider_name: str = "external geocentric position provider",
    coarse_step_hours: float = 6.0,
) -> list[DiscoveredEclipse]:
    """Numerically discover eclipses from arbitrary Sun/Moon state vectors.

    The method searches minima of a finite-cone target-intersection margin and
    then root-finds the external and internal contact equations.  It is used by
    the DE430 backend and is public so downstream propagators can be
    validated without being wrapped as a package-specific backend.
    """
    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for numerical eclipse discovery") from exc
    mode = str(mode).lower()
    if mode not in {"solar", "lunar"}:
        raise ValueError("mode must be 'solar' or 'lunar'")
    start_jd, end_jd = _coerce_jd(start), _coerce_jd(end)
    if not start_jd < end_jd:
        raise ValueError("start must precede end")
    step = float(coarse_step_hours) / 24.0
    grid = np.arange(start_jd, end_jd + 0.5 * step, step)
    sun_grid, moon_grid = _provider_positions(provider, grid)

    def margin_from_vectors(sun, moon):
        if mode == "solar":
            miss, _, pen, _ = _solar_shadow_geometry(sun, moon)
            return miss - (RE_KM + pen)
        miss, _, pen, _ = _lunar_shadow_geometry(sun, moon)
        return miss - (pen + R_MOON_MEAN_KM)

    margins = np.asarray([
        margin_from_vectors(s, m) for s, m in zip(sun_grid, moon_grid)
    ])
    minima = [
        index for index in range(1, len(grid) - 1)
        if margins[index] <= margins[index - 1] and margins[index] < margins[index + 1]
        and margins[index] < 2.5 * RE_KM
    ]
    records: list[DiscoveredEclipse] = []

    def state(jd: float) -> tuple[np.ndarray, np.ndarray]:
        sun, moon = _provider_positions(provider, np.asarray([jd], dtype=float))
        return sun[0], moon[0]

    for index in minima:
        lo, hi = float(grid[index - 1]), float(grid[index + 1])
        midpoint = float(grid[index])
        lo_s, hi_s = (lo-midpoint)*86400.0, (hi-midpoint)*86400.0
        optimized = minimize_scalar(
            lambda offset_s: margin_from_vectors(*state(midpoint + float(offset_s)/86400.0)),
            bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1.0e-3},
        )
        maximum = float(midpoint + optimized.x/86400.0)
        sun_max, moon_max = state(maximum)
        if mode == "solar":
            miss, _, pen, umb = _solar_shadow_geometry(sun_max, moon_max)
            def solar_geometry_at(value: float) -> tuple[float, float, float, float]:
                """Evaluate one ephemeris state once for a contact equation."""
                return _solar_shadow_geometry(*state(float(value)))

            def p_eq(value: float) -> float:
                axis_miss, _, penumbra, _ = solar_geometry_at(value)
                return axis_miss - (RE_KM + penumbra)

            def u_eq(value: float) -> float:
                axis_miss, _, _, umbra = solar_geometry_at(value)
                return axis_miss - (RE_KM + abs(umbra))
            search_lo, search_hi = maximum - 0.45, maximum + 0.45
            p_roots = _roots(p_eq, search_lo, search_hi)
            if len(p_roots) < 2:
                continue
            u_roots = _roots(u_eq, search_lo, search_hi)
            central = len(u_roots) >= 2
            if central:
                central_span = np.linspace(u_roots[0], u_roots[-1], 5)
                radius_signs = []
                for value in central_span:
                    sun_value, moon_value = state(float(value))
                    surface = _solar_surface_apparent_geometry(
                        sun_value, moon_value, float(value),
                    )
                    radius_signs.append(surface[0] - surface[1])
                has_total = any(sign > 0.0 for sign in radius_signs)
                has_annular = any(sign < 0.0 for sign in radius_signs)
                eclipse_type = (
                    "hybrid" if has_total and has_annular
                    else "total" if has_total
                    else "annular"
                )
            else:
                eclipse_type = "partial"
            contacts = {"P1": p_roots[0], "MAX": maximum, "P4": p_roots[-1]}
            if central:
                contacts.update({"U1": u_roots[0], "U4": u_roots[-1]})
            surface_at_max = (
                _solar_central_apparent_geometry(sun_max, moon_max, maximum)
                if central else _solar_surface_apparent_geometry(sun_max, moon_max, maximum)
            )
            moon_radius, sun_radius, separation, surface_lat, surface_lon = surface_at_max
            magnitude = (
                moon_radius / sun_radius if central
                else (moon_radius + sun_radius - separation) / (2.0 * sun_radius)
            )
            metadata = {
                "axis_miss_km": miss,
                "penumbra_radius_at_earth_plane_km": pen,
                "signed_umbra_radius_at_earth_plane_km": umb,
                "surface_greatest_lat_deg": surface_lat,
                "surface_greatest_lon_east_deg": surface_lon,
            }
        else:
            miss, _, pen, umb = _lunar_shadow_geometry(sun_max, moon_max)
            def family_eq(value, family: str):
                m, _, p, u = _lunar_shadow_geometry(*state(float(value)))
                boundary = p + R_MOON_MEAN_KM if family == "P" else (
                    u + R_MOON_MEAN_KM if family == "U" else u - R_MOON_MEAN_KM
                )
                return m - boundary
            search_lo, search_hi = maximum - 0.55, maximum + 0.55
            p_roots = _roots(lambda value: family_eq(value, "P"), search_lo, search_hi)
            if len(p_roots) < 2:
                continue
            u_roots = _roots(lambda value: family_eq(value, "U"), search_lo, search_hi)
            t_roots = _roots(lambda value: family_eq(value, "T"), search_lo, search_hi)
            if len(t_roots) >= 2:
                eclipse_type = "total"
            elif len(u_roots) >= 2:
                eclipse_type = "partial-umbral"
            else:
                eclipse_type = "penumbral"
            contacts = {"P1": p_roots[0], "MAX": maximum, "P4": p_roots[-1]}
            if len(u_roots) >= 2:
                contacts.update({"U1": u_roots[0], "U4": u_roots[-1]})
            if len(t_roots) >= 2:
                contacts.update({"U2": t_roots[0], "U3": t_roots[-1]})
            magnitude = (umb + R_MOON_MEAN_KM - miss) / (2.0 * R_MOON_MEAN_KM)
            central = True
            metadata = {
                "axis_miss_km": miss,
                "penumbra_radius_at_moon_plane_km": pen,
                "umbra_radius_at_moon_plane_km": umb,
            }
        records.append(DiscoveredEclipse(
            key=_key(mode, maximum), mode=mode, eclipse_type=eclipse_type,
            greatest_jd_utc=maximum, contacts_jd_utc=_nonzero_contacts(contacts),
            discovery_backend=provider_name, source_label="Finite-cone numerical discovery",
            central=central, magnitude=float(magnitude), metadata=metadata,
        ))
    # Deduplicate candidates that arose from adjacent coarse minima.
    unique: list[DiscoveredEclipse] = []
    for record in sorted(records, key=lambda item: item.greatest_jd_utc):
        if unique and abs(record.greatest_jd_utc - unique[-1].greatest_jd_utc) < 5.0:
            if len(record.contacts_jd_utc) > len(unique[-1].contacts_jd_utc):
                unique[-1] = record
        else:
            unique.append(record)
    return unique


class _De430Provider:
    name = "DE430 numerical search"

    def positions_gcrf_km(self, jd_utc):
        moon, sun = ephemeris_positions_km(jd_utc)
        return sun, moon


def discover_eclipses(
    start: float | str | datetime,
    end: float | str | datetime,
    *,
    kind: str = "all",
) -> list[DiscoveredEclipse]:
    """Discover all solar and/or lunar eclipses in a UTC interval from DE430 states."""
    start_jd, end_jd = _coerce_jd(start), _coerce_jd(end)
    if not start_jd < end_jd:
        raise ValueError("start must precede end")
    key = str(kind).lower()
    modes = ("solar", "lunar") if key == "all" else (key,)
    if any(mode not in {"solar", "lunar"} for mode in modes):
        raise ValueError("kind must be 'solar', 'lunar', or 'all'")
    records: list[DiscoveredEclipse] = []
    for mode in modes:
        records.extend(discover_with_position_provider(
            start_jd, end_jd, mode=mode, provider=_De430Provider(), provider_name=_De430Provider.name,
        ))
    return sorted(records, key=lambda item: item.greatest_jd_utc)


def _contact_aware_times(record: DiscoveredEclipse, n_frames: int) -> np.ndarray:
    ordered = sorted(record.contacts_jd_utc.items(), key=lambda item: float(item[1]))
    start, stop = float(ordered[0][1]), float(ordered[-1][1])
    count = max(int(n_frames), len(ordered), 17)
    u = np.linspace(0.0, 1.0, count)
    base = start + (stop - start) * (0.5 - 0.5 * np.cos(np.pi * u))
    offsets_s = np.array([-900, -600, -300, -120, -60, -20, 0, 20, 60, 120, 300, 600, 900])
    exact = np.concatenate([
        np.asarray([float(value) for _, value in ordered]),
        record.greatest_jd_utc + offsets_s / 86400.0,
    ])
    exact = exact[(exact >= start) & (exact <= stop)]
    keep = np.ones(base.shape, dtype=bool)
    for value in exact:
        keep &= np.abs(base - value) > 0.05 / 86400.0
    return np.sort(np.unique(np.concatenate([base[keep], exact])))


def build_discovered_event(record: DiscoveredEclipse, *, n_frames: int = 121) -> ReferenceEvent:
    """Materialize a discovered event with DE430 states on a contact-aware timeline."""
    times = _contact_aware_times(record, n_frames)
    selection = resolve_backend("de430")
    moon, sun = ephemeris_positions_km(times)
    visible, separation = (
        solar_visibility_and_separation(times, sun, moon) if record.mode == "solar"
        else center_visibility_and_separation("lunar", sun, moon)
    )
    metadata = {
        "dynamic_event": True,
        "requested_minimum_state_count": int(n_frames),
        "event_search_record": record.to_dict(),
        "event_search_backend": record.discovery_backend,
        "state_source": selection.selected,
        "ephemeris_backend": selection.ephemeris_backend,
        "frame_backend": selection.frame_backend,
        "solar_scope": "global",
        "event_classification": record.eclipse_type,
        "greatest_lat_deg": record.greatest_lat_deg,
        "greatest_lon_east_deg": record.greatest_lon_east_deg,
        "magnitude": record.magnitude,
        "obscuration": record.obscuration,
    }
    return ReferenceEvent(
        definition=record.to_definition(),
        jd=np.asarray(times, dtype=float),
        moon_km=moon,
        sun_km=sun,
        frame="GCRF",
        backend=selection.ephemeris_backend,
        center_visibility=visible,
        separation_deg=separation,
        metadata=metadata,
    )


def write_discovery_catalog(records: Iterable[DiscoveredEclipse], output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "$schema": "ssatk.eclipse.discovery-catalog/1",
        "records": [record.to_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def read_discovery_catalog(path: str | Path) -> list[DiscoveredEclipse]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [DiscoveredEclipse.from_dict(item) for item in payload.get("records", [])]
