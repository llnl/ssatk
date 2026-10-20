"""Provider-driven arbitrary solar and lunar eclipse discovery.

The search layer deliberately separates *discovery* from *rendering*:

* ``ssapy`` / ``ssapy-core`` use LLNL body states and the selected frame
  provider, then solve finite-Sun contact equations numerically.
* ``swisseph`` is an optional, search-only catalogue/validation bridge.  It is
  never installed as a required dependency and its provenance is explicit.
* ``catalog`` exposes only the two bundled NASA/GSFC reference events.
* ``auto`` prefers strict SSAPy, then the optional Swiss Ephemeris bridge, then
  the portable analytical discovery model.  The two bundled reference events
  remain available explicitly through ``backend='catalog'``.

A discovered event can be materialized as the same immutable ``ReferenceEvent``
record consumed by the ray tracer and public renderers.  Dynamic records store
all body states and therefore remain reproducible after serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence
import json
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_brightness import AU_KM
from ssapy_toolkit.compute.eclipse_runtime import (
    BackendUnavailableError,
    resolve_backend,
    ssapy_ephemeris_positions_km,
    toolkit_gcrf_to_itrf_km,
    astropy_gcrf_to_itrf_km,
)
from ssapy_toolkit.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    RP_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    ReferenceDefinition,
    ReferenceEvent,
    angular_circle_visible_fraction,
    datetime_to_jd,
    jd_to_datetime,
)

SearchBackend = str


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
            "$schema": "ssapy-toolkit.eclipse.discovery/2.0",
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


def _load_swe():
    try:
        import swisseph as swe
    except Exception as exc:
        raise BackendUnavailableError(
            "The optional Swiss Ephemeris search bridge is unavailable. "
            "Install pyswisseph under its applicable licence, or use backend='ssapy'."
        ) from exc
    return swe


def swisseph_available() -> bool:
    try:
        _load_swe()
        return True
    except BackendUnavailableError:
        return False


def _swe_solar_type(swe, flags: int) -> str:
    if flags & swe.ECL_ANNULAR_TOTAL:
        return "hybrid"
    if flags & swe.ECL_TOTAL:
        return "total"
    if flags & swe.ECL_ANNULAR:
        return "annular"
    return "partial"


def _swe_lunar_type(swe, flags: int) -> str:
    if flags & swe.ECL_TOTAL:
        return "total"
    if flags & swe.ECL_PARTIAL:
        return "partial-umbral"
    return "penumbral"


def _nonzero_contacts(mapping: Mapping[str, float]) -> dict[str, float]:
    result = {name: float(value) for name, value in mapping.items() if float(value) > 0.0}
    return dict(sorted(result.items(), key=lambda item: item[1]))


def _discover_swisseph(start_jd: float, end_jd: float, mode: str) -> list[DiscoveredEclipse]:
    swe = _load_swe()
    flags = swe.FLG_MOSEPH
    records: list[DiscoveredEclipse] = []
    cursor = float(start_jd) - 1.0e-6
    while cursor <= end_jd:
        if mode == "solar":
            result_flags, times = swe.sol_eclipse_when_glob(
                cursor, flags, swe.ECL_ALLTYPES_SOLAR, False
            )
            maximum = float(times[0])
            if maximum > end_jd:
                break
            if maximum >= start_jd:
                where_flags, geopos, attr = swe.sol_eclipse_where(maximum, flags)
                eclipse_type = _swe_solar_type(swe, int(result_flags))
                contacts = _nonzero_contacts({
                    "P1": times[2], "U1": times[4], "MAX": times[0],
                    "U4": times[5], "P4": times[3],
                    "CENTER_BEGIN": times[6], "CENTER_END": times[7],
                    "HYBRID_TOTAL_BEGIN": times[8], "HYBRID_TOTAL_END": times[9],
                })
                records.append(DiscoveredEclipse(
                    key=_key("solar", maximum), mode="solar", eclipse_type=eclipse_type,
                    greatest_jd_utc=maximum, contacts_jd_utc=contacts,
                    discovery_backend=f"Swiss Ephemeris {getattr(swe, 'version', 'unknown')} / Moshier",
                    source_label="Optional Swiss Ephemeris global-eclipse search bridge",
                    central=bool(result_flags & swe.ECL_CENTRAL),
                    greatest_lon_east_deg=float(geopos[0]),
                    greatest_lat_deg=float(geopos[1]),
                    magnitude=float(attr[8]), obscuration=float(attr[2]),
                    shadow_width_km=abs(float(attr[3])) if eclipse_type != "partial" else None,
                    saros_series=_optional_int(attr[9]), saros_member=_optional_int(attr[10]),
                    metadata={
                        "swisseph_result_flags": int(result_flags),
                        "swisseph_where_flags": int(where_flags),
                        "ephemeris_flag": "FLG_MOSEPH",
                        "licensing_note": "Optional external search bridge; not redistributed or required by this package.",
                    },
                ))
        else:
            result_flags, times = swe.lun_eclipse_when(
                cursor, flags, swe.ECL_ALLTYPES_LUNAR, False
            )
            maximum = float(times[0])
            if maximum > end_jd:
                break
            if maximum >= start_jd:
                _, attr = swe.lun_eclipse_how(maximum, (0.0, 0.0, 0.0), flags)
                contacts = _nonzero_contacts({
                    "P1": times[6], "U1": times[2], "U2": times[4],
                    "MAX": times[0], "U3": times[5], "U4": times[3], "P4": times[7],
                })
                records.append(DiscoveredEclipse(
                    key=_key("lunar", maximum), mode="lunar",
                    eclipse_type=_swe_lunar_type(swe, int(result_flags)),
                    greatest_jd_utc=maximum, contacts_jd_utc=contacts,
                    discovery_backend=f"Swiss Ephemeris {getattr(swe, 'version', 'unknown')} / Moshier",
                    source_label="Optional Swiss Ephemeris global-eclipse search bridge",
                    central=True, magnitude=float(attr[8]),
                    saros_series=_optional_int(attr[9]), saros_member=_optional_int(attr[10]),
                    metadata={
                        "umbral_magnitude": float(attr[0]),
                        "penumbral_magnitude": float(attr[1]),
                        "axis_offset_deg": float(attr[7]),
                        "swisseph_result_flags": int(result_flags),
                        "ephemeris_flag": "FLG_MOSEPH",
                        "licensing_note": "Optional external search bridge; not redistributed or required by this package.",
                    },
                ))
        cursor = maximum + 1.0
    return records


def _catalog_records(start_jd: float, end_jd: float, mode: str) -> list[DiscoveredEclipse]:
    definitions = [SOLAR_2024] if mode == "solar" else [LUNAR_2025]
    records = []
    for definition in definitions:
        if start_jd <= definition.greatest_jd <= end_jd:
            records.append(DiscoveredEclipse(
                key=definition.key,
                mode=definition.mode,
                eclipse_type="total",
                greatest_jd_utc=definition.greatest_jd,
                contacts_jd_utc=definition.contacts_jd,
                discovery_backend="bundled NASA/GSFC reference catalogue",
                source_label=definition.source_label,
                central=True,
                greatest_lat_deg=(25.2866666667 if definition.mode == "solar" else None),
                greatest_lon_east_deg=(-104.1383333333 if definition.mode == "solar" else None),
                magnitude=(1.0566 if definition.mode == "solar" else 1.1784),
                metadata={"catalog_only": True},
            ))
    return records


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
    closest = moon + downstream * direction
    miss = float(np.linalg.norm(closest))
    pen_slope = (R_SUN_KM + R_MOON_MEAN_KM) / math.sqrt(
        max(distance * distance - (R_SUN_KM + R_MOON_MEAN_KM) ** 2, 1.0)
    )
    umb_slope = (R_SUN_KM - R_MOON_MEAN_KM) / math.sqrt(
        max(distance * distance - (R_SUN_KM - R_MOON_MEAN_KM) ** 2, 1.0)
    )
    penumbra = R_MOON_MEAN_KM + downstream * pen_slope
    umbra_signed = R_MOON_MEAN_KM - downstream * umb_slope
    return miss, downstream, penumbra, umbra_signed


def _lunar_shadow_geometry(sun: np.ndarray, moon: np.ndarray) -> tuple[float, float, float, float]:
    distance = float(np.linalg.norm(sun))
    direction = -sun / distance
    downstream = float(np.dot(moon, direction))
    closest = moon - downstream * direction
    miss = float(np.linalg.norm(closest))
    if downstream <= 0.0:
        # At new Moon the Moon is sunward of Earth, so Earth's anti-solar
        # shadow cannot intersect it.  Returning negative cone radii keeps the
        # finite-cone margin strictly positive and prevents false lunar events.
        return miss, downstream, -1.0e9, -1.0e9
    # Danjon enlargement is intentionally explicit rather than hidden in the
    # Earth solid radius.  It is used only for lunar contact prediction.
    earth_shadow_radius = RE_KM * 1.01
    pen_slope = (R_SUN_KM + earth_shadow_radius) / math.sqrt(
        max(distance * distance - (R_SUN_KM + earth_shadow_radius) ** 2, 1.0)
    )
    umb_slope = (R_SUN_KM - earth_shadow_radius) / math.sqrt(
        max(distance * distance - (R_SUN_KM - earth_shadow_radius) ** 2, 1.0)
    )
    penumbra = earth_shadow_radius + downstream * pen_slope
    umbra = earth_shadow_radius - downstream * umb_slope
    return miss, downstream, penumbra, umbra


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
    the strict SSAPy backend and is public so downstream propagators can be
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
            p_eq = lambda value: (
                _solar_shadow_geometry(*state(float(value)))[0]
                - (RE_KM + _solar_shadow_geometry(*state(float(value)))[2])
            )
            u_eq = lambda value: (
                _solar_shadow_geometry(*state(float(value)))[0]
                - (RE_KM + abs(_solar_shadow_geometry(*state(float(value)))[3]))
            )
            search_lo, search_hi = maximum - 0.45, maximum + 0.45
            p_roots = _roots(p_eq, search_lo, search_hi)
            if len(p_roots) < 2:
                continue
            u_roots = _roots(u_eq, search_lo, search_hi)
            central = len(u_roots) >= 2
            if central:
                eclipse_type = "total" if umb > 0.0 else "annular"
            else:
                eclipse_type = "partial"
            contacts = {"P1": p_roots[0], "MAX": maximum, "P4": p_roots[-1]}
            if central:
                contacts.update({"U1": u_roots[0], "U4": u_roots[-1]})
            magnitude = (R_MOON_MEAN_KM / np.linalg.norm(moon_max)) / (
                R_SUN_KM / np.linalg.norm(sun_max)
            )
            metadata = {
                "axis_miss_km": miss,
                "penumbra_radius_at_earth_plane_km": pen,
                "signed_umbra_radius_at_earth_plane_km": umb,
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




def analytic_positions_gcrf_km(jd_utc: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic analytical geocentric Sun/Moon positions.

    This compact osculating-element model retains the dominant lunar evection,
    variation, annual equation, latitude, and range perturbations.  It is a
    dependency-free discovery/preview fallback, not a JPL or SSAPy replacement.
    Returned arrays are ``(sun, moon)`` in kilometres in a mean-equatorial
    J2000-like frame.
    """
    jd = np.asarray(jd_utc, dtype=float).reshape(-1)
    d = jd - 2451543.5
    norm = lambda value: np.mod(value, 360.0)
    sind = lambda value: np.sin(np.radians(value))
    cosd = lambda value: np.cos(np.radians(value))

    def solve(mean_deg, eccentricity):
        mean = np.radians(np.asarray(mean_deg, dtype=float))
        ecc = np.asarray(eccentricity, dtype=float)
        estimate = mean + ecc*np.sin(mean)*(1.0 + ecc*np.cos(mean))
        for _ in range(12):
            delta = (estimate-ecc*np.sin(estimate)-mean)/(1.0-ecc*np.cos(estimate))
            estimate -= delta
            if np.max(np.abs(delta)) < 2.0e-13:
                break
        return estimate

    sun_argp = norm(282.9404 + 4.70935e-5*d)
    sun_ecc = 0.016709 - 1.151e-9*d
    sun_mean = norm(356.0470 + 0.9856002585*d)
    sun_e = solve(sun_mean, sun_ecc)
    sun_xv = np.cos(sun_e)-sun_ecc
    sun_yv = np.sqrt(1.0-sun_ecc*sun_ecc)*np.sin(sun_e)
    sun_true = np.degrees(np.arctan2(sun_yv, sun_xv))
    sun_range_au = np.hypot(sun_xv, sun_yv)
    sun_lon = norm(sun_true+sun_argp)

    node = norm(125.1228-0.0529538083*d)
    inclination = 5.1454
    moon_argp = norm(318.0634+0.1643573223*d)
    moon_a_re = 60.2666
    moon_ecc = 0.054900
    moon_mean = norm(115.3654+13.0649929509*d)
    moon_e = solve(moon_mean, moon_ecc)
    xv = moon_a_re*(np.cos(moon_e)-moon_ecc)
    yv = moon_a_re*np.sqrt(1.0-moon_ecc*moon_ecc)*np.sin(moon_e)
    true_anomaly = np.degrees(np.arctan2(yv, xv))
    radius_re = np.hypot(xv, yv)
    node_r = np.radians(node)
    inc_r = math.radians(inclination)
    vw = np.radians(true_anomaly+moon_argp)
    xh = radius_re*(np.cos(node_r)*np.cos(vw)-np.sin(node_r)*np.sin(vw)*math.cos(inc_r))
    yh = radius_re*(np.sin(node_r)*np.cos(vw)+np.cos(node_r)*np.sin(vw)*math.cos(inc_r))
    zh = radius_re*np.sin(vw)*math.sin(inc_r)
    moon_lon = np.degrees(np.arctan2(yh, xh))
    moon_lat = np.degrees(np.arctan2(zh, np.hypot(xh, yh)))

    moon_mean_lon = norm(moon_mean+moon_argp+node)
    sun_mean_lon = norm(sun_mean+sun_argp)
    elongation = norm(moon_mean_lon-sun_mean_lon)
    argument_latitude = norm(moon_mean_lon-node)
    moon_lon += (
        -1.274*sind(moon_mean-2.0*elongation) + 0.658*sind(2.0*elongation)
        -0.186*sind(sun_mean) - 0.059*sind(2.0*moon_mean-2.0*elongation)
        -0.057*sind(moon_mean-2.0*elongation+sun_mean)
        +0.053*sind(moon_mean+2.0*elongation) + 0.046*sind(2.0*elongation-sun_mean)
        +0.041*sind(moon_mean-sun_mean) - 0.035*sind(elongation)
        -0.031*sind(moon_mean+sun_mean) - 0.015*sind(2.0*argument_latitude-2.0*elongation)
        +0.011*sind(moon_mean-4.0*elongation)
    )
    moon_lat += (
        -0.173*sind(argument_latitude-2.0*elongation)
        -0.055*sind(moon_mean-argument_latitude-2.0*elongation)
        -0.046*sind(moon_mean+argument_latitude-2.0*elongation)
        +0.033*sind(argument_latitude+2.0*elongation)
        +0.017*sind(2.0*moon_mean+argument_latitude)
    )
    radius_re += -0.58*cosd(moon_mean-2.0*elongation)-0.46*cosd(2.0*elongation)

    obliquity = np.radians(23.4393-3.563e-7*d)
    moon_lon_r = np.radians(moon_lon)
    moon_lat_r = np.radians(moon_lat)
    mx = radius_re*np.cos(moon_lon_r)*np.cos(moon_lat_r)
    my_ecliptic = radius_re*np.sin(moon_lon_r)*np.cos(moon_lat_r)
    mz_ecliptic = radius_re*np.sin(moon_lat_r)
    my = my_ecliptic*np.cos(obliquity)-mz_ecliptic*np.sin(obliquity)
    mz = my_ecliptic*np.sin(obliquity)+mz_ecliptic*np.cos(obliquity)

    sun_lon_r = np.radians(sun_lon)
    sx = sun_range_au*np.cos(sun_lon_r)
    sy_ecliptic = sun_range_au*np.sin(sun_lon_r)
    sy = sy_ecliptic*np.cos(obliquity)
    sz = sy_ecliptic*np.sin(obliquity)
    sun = np.column_stack([sx, sy, sz])*AU_KM
    moon = np.column_stack([mx, my, mz])*RE_KM
    return sun, moon


class _AnalyticProvider:
    name = "dependency-free truncated analytical discovery model"

    def positions_gcrf_km(self, jd_utc):
        return analytic_positions_gcrf_km(jd_utc)


def _discover_analytic(start_jd: float, end_jd: float, mode: str) -> list[DiscoveredEclipse]:
    return discover_with_position_provider(
        start_jd, end_jd, mode=mode, provider=_AnalyticProvider(),
        provider_name=_AnalyticProvider.name, coarse_step_hours=3.0,
    )

class _SsapyProvider:
    def __init__(self, selected: str):
        self.selected = selected
        self.name = f"LLNL SSAPy numerical search ({selected})"

    def positions_gcrf_km(self, jd_utc):
        moon, sun = ssapy_ephemeris_positions_km(jd_utc)
        return sun, moon


def _discover_ssapy(start_jd: float, end_jd: float, mode: str, backend: str) -> list[DiscoveredEclipse]:
    selection = resolve_backend(backend)
    if selection.selected not in {"ssapy", "ssapy-core"}:
        raise BackendUnavailableError(
            f"backend={backend!r} did not resolve to an executable LLNL state provider"
        )
    return discover_with_position_provider(
        start_jd, end_jd, mode=mode,
        provider=_SsapyProvider(selection.selected),
        provider_name=f"LLNL SSAPy {selection.ephemeris_backend}",
    )


def discover_eclipses(
    start: float | str | datetime,
    end: float | str | datetime,
    *,
    kind: str = "all",
    backend: SearchBackend = "auto",
) -> list[DiscoveredEclipse]:
    """Discover all eclipses in a UTC interval.

    ``backend='auto'`` prefers a healthy strict SSAPy provider, then the
    optional Swiss Ephemeris bridge, and finally the portable analytical
    discovery model.  The selected backend is recorded in every result.
    """
    start_jd, end_jd = _coerce_jd(start), _coerce_jd(end)
    if not start_jd < end_jd:
        raise ValueError("start must precede end")
    key = str(kind).lower()
    modes = ("solar", "lunar") if key == "all" else (key,)
    if any(mode not in {"solar", "lunar"} for mode in modes):
        raise ValueError("kind must be 'solar', 'lunar', or 'all'")
    requested = str(backend).lower()
    selected = requested
    if requested == "auto":
        try:
            resolution = resolve_backend("auto")
            selected = resolution.selected if resolution.selected in {"ssapy", "ssapy-core"} else ""
        except Exception:
            selected = ""
        if not selected:
            selected = "swisseph" if swisseph_available() else "analytic"
    records: list[DiscoveredEclipse] = []
    for mode in modes:
        if selected in {"ssapy", "ssapy-core"}:
            records.extend(_discover_ssapy(start_jd, end_jd, mode, selected))
        elif selected == "swisseph":
            records.extend(_discover_swisseph(start_jd, end_jd, mode))
        elif selected == "analytic":
            records.extend(_discover_analytic(start_jd, end_jd, mode))
        elif selected in {"catalog", "reference"}:
            records.extend(_catalog_records(start_jd, end_jd, mode))
        else:
            raise ValueError("search backend must be auto, ssapy, ssapy-core, swisseph, analytic, or catalog")
    return sorted(records, key=lambda item: item.greatest_jd_utc)


def swisseph_positions_gcrf_km(jd_utc: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Optional Swiss-Ephemeris geocentric J2000 equatorial positions.

    This function is a catalogue/validation bridge, not a required state
    backend.  No Swiss Ephemeris code or data are redistributed by this
    project.
    """
    swe = _load_swe()
    flags = swe.FLG_MOSEPH | swe.FLG_EQUATORIAL | swe.FLG_XYZ | swe.FLG_J2000
    sun, moon = [], []
    for value in np.asarray(jd_utc, dtype=float).reshape(-1):
        sun_au, _ = swe.calc_ut(float(value), swe.SUN, flags)
        moon_au, _ = swe.calc_ut(float(value), swe.MOON, flags)
        sun.append(np.asarray(sun_au[:3], dtype=float) * AU_KM)
        moon.append(np.asarray(moon_au[:3], dtype=float) * AU_KM)
    return np.asarray(sun), np.asarray(moon)


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


def build_discovered_event(
    record: DiscoveredEclipse,
    *,
    n_frames: int = 121,
    state_backend: str = "auto",
) -> ReferenceEvent:
    """Materialize a discovered event as a renderer-ready immutable state."""
    times = _contact_aware_times(record, n_frames)
    requested = str(state_backend).lower()
    selected = requested
    if requested == "auto":
        if record.discovery_backend.lower().startswith("llnl ssapy"):
            selected = "ssapy"
        elif "analytical" in record.discovery_backend.lower():
            selected = "analytic"
        elif swisseph_available():
            selected = "swisseph"
        else:
            selected = "analytic"
    if selected in {"ssapy", "ssapy-core"}:
        resolution = resolve_backend(selected)
        if resolution.selected not in {"ssapy", "ssapy-core"}:
            raise BackendUnavailableError(f"strict state backend {selected!r} is unavailable")
        moon, sun = ssapy_ephemeris_positions_km(times)
        state_source = resolution.selected
        frame_backend = resolution.frame_backend
        backend_label = resolution.ephemeris_backend
        strict = True
        fallback = ""
    elif selected == "swisseph":
        sun, moon = swisseph_positions_gcrf_km(times)
        swe = _load_swe()
        state_source = "swisseph"
        frame_backend = "deterministic GMST/WGS-84 reference transform"
        backend_label = f"Swiss Ephemeris {getattr(swe, 'version', 'unknown')} Moshier J2000"
        strict = True
        fallback = ""
    elif selected == "analytic":
        sun, moon = analytic_positions_gcrf_km(times)
        state_source = "analytic-discovery"
        frame_backend = "deterministic GMST/WGS-84 reference transform"
        backend_label = "truncated analytical Sun/Moon discovery model"
        strict = False
        fallback = "portable dependency-free preview; not a JPL/SSAPy ephemeris"
    else:
        raise ValueError("state_backend must be auto, ssapy, ssapy-core, swisseph, or analytic")

    visible, separation = [], []
    if record.mode == "solar":
        for sun_i, moon_i in zip(sun, moon):
            ds, dm = float(np.linalg.norm(sun_i)), float(np.linalg.norm(moon_i))
            sep = math.acos(np.clip(np.dot(sun_i / ds, moon_i / dm), -1.0, 1.0))
            visible.append(float(angular_circle_visible_fraction(
                math.asin(R_MOON_MEAN_KM / dm), math.asin(R_SUN_KM / ds), sep
            )))
            separation.append(math.degrees(sep))
    else:
        for sun_i, moon_i in zip(sun, moon):
            to_earth = -moon_i
            to_sun = sun_i - moon_i
            de, ds = float(np.linalg.norm(to_earth)), float(np.linalg.norm(to_sun))
            sep = math.acos(np.clip(np.dot(to_earth / de, to_sun / ds), -1.0, 1.0))
            visible.append(float(angular_circle_visible_fraction(
                math.asin(RE_KM * 1.01 / de), math.asin(R_SUN_KM / ds), sep
            )))
            separation.append(math.degrees(math.pi - sep))
    definition = record.to_definition()
    metadata = {
        "dynamic_event": True,
        "event_search_record": record.to_dict(),
        "event_search_backend": record.discovery_backend,
        "state_source": state_source,
        "ephemeris_backend": backend_label,
        "frame_backend": frame_backend,
        "backend_requested": requested,
        "strict_backend": strict,
        "backend_fallback_reason": fallback,
        "time_input_pipeline": "UTC Julian date supplied to selected discovery/state backend",
        "solar_scope": "global",
        "event_classification": record.eclipse_type,
        "greatest_lat_deg": record.greatest_lat_deg,
        "greatest_lon_east_deg": record.greatest_lon_east_deg,
        "magnitude": record.magnitude,
        "obscuration": record.obscuration,
        "moon_orientation_backend": (
            "LLNL SSAPy DE440 binary lunar PCK" if state_source in {"ssapy", "ssapy-core"}
            else "NAIF IAU_MOON 2009 text-PCK fallback"
        ),
        "state_interpolation": "shape-preserving cubic interpolation over serialized event samples",
    }
    return ReferenceEvent(
        definition=definition,
        jd=np.asarray(times, dtype=float),
        moon_km=np.asarray(moon, dtype=float),
        sun_km=np.asarray(sun, dtype=float),
        frame="GCRF/J2000 geocentric equatorial",
        backend=backend_label,
        center_visibility=np.asarray(visible, dtype=float),
        separation_deg=np.asarray(separation, dtype=float),
        metadata=metadata,
    )


def write_discovery_catalog(records: Iterable[DiscoveredEclipse], output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "$schema": "ssapy-toolkit.eclipse.discovery-catalog/2.0",
        "records": [record.to_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def read_discovery_catalog(path: str | Path) -> list[DiscoveredEclipse]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return [DiscoveredEclipse.from_dict(item) for item in payload.get("records", [])]
