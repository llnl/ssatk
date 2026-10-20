"""Topocentric observer geometry, refraction, terrain horizon, and contacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
from typing import TYPE_CHECKING

import numpy as np

from ssatk.coordinates.eclipse_lunar_geometry import LunarLimbProfile, apparent_position_angle_deg

if TYPE_CHECKING:
    from ssatk.compute.eclipse_reference_events import ReferenceEvent


@dataclass(frozen=True)
class HorizonProfile:
    azimuth_deg: np.ndarray
    altitude_deg: np.ndarray
    source: str = "flat geometric horizon"

    def __post_init__(self) -> None:
        az = np.mod(np.asarray(self.azimuth_deg, dtype=float).reshape(-1), 360.0)
        alt = np.asarray(self.altitude_deg, dtype=float).reshape(-1)
        if len(az) < 2 or len(az) != len(alt):
            raise ValueError("horizon profile requires matching azimuth/altitude arrays")
        if not np.all(np.isfinite(az)) or not np.all(np.isfinite(alt)):
            raise ValueError("horizon values must be finite")
        order = np.argsort(az)
        object.__setattr__(self, "azimuth_deg", az[order])
        object.__setattr__(self, "altitude_deg", alt[order])

    @classmethod
    def flat(cls, altitude_deg: float = 0.0) -> "HorizonProfile":
        return cls(np.array([0.0, 180.0]), np.array([altitude_deg, altitude_deg]))

    @classmethod
    def from_file(cls, path: str | Path) -> "HorizonProfile":
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                return cls(data["azimuth_deg"], data["altitude_deg"], source=str(path))
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
            az_key = fields.get("azimuth_deg") or fields.get("az_deg")
            alt_key = fields.get("altitude_deg") or fields.get("horizon_deg") or fields.get("elevation_deg")
            if az_key is None or alt_key is None:
                raise ValueError("CSV horizon profile needs azimuth_deg and altitude_deg")
            az, alt = [], []
            for row in reader:
                az.append(float(row[az_key]))
                alt.append(float(row[alt_key]))
        return cls(np.asarray(az), np.asarray(alt), source=str(path))

    def altitude_at(self, azimuth_deg) -> np.ndarray | float:
        query = np.asarray(azimuth_deg, dtype=float)
        az = self.azimuth_deg
        alt = self.altitude_deg
        extended_az = np.concatenate([az[-1:] - 360.0, az, az[:1] + 360.0])
        extended_alt = np.concatenate([alt[-1:], alt, alt[:1]])
        result = np.interp(np.mod(query, 360.0), extended_az, extended_alt)
        return float(result) if query.ndim == 0 else result


@dataclass(frozen=True)
class ObserverConfig:
    latitude_deg: float
    longitude_east_deg: float
    elevation_m: float = 0.0
    pressure_hpa: float = 1010.0
    temperature_c: float = 10.0
    relative_humidity_pct: float = 0.0
    wavelength_um: float = 0.55
    apply_refraction: bool = True
    refraction_model: str = "Bennett/Saemundsson + Edlen wavelength scaling"
    weather_source: str = "standard atmosphere defaults"
    horizon: HorizonProfile | None = None
    name: str = "WGS-84 observer"

    def __post_init__(self) -> None:
        if not -90.0 <= float(self.latitude_deg) <= 90.0:
            raise ValueError("observer latitude must be between -90 and +90 degrees")
        if not np.isfinite(float(self.longitude_east_deg)):
            raise ValueError("observer longitude must be finite")
        if float(self.elevation_m) < -500.0:
            raise ValueError("observer elevation is implausibly below the WGS-84 ellipsoid")
        if float(self.pressure_hpa) < 0.0:
            raise ValueError("pressure_hpa cannot be negative")
        if not 0.0 <= float(self.relative_humidity_pct) <= 100.0:
            raise ValueError("relative_humidity_pct must be between 0 and 100")
        if not 0.30 <= float(self.wavelength_um) <= 2.0:
            raise ValueError("wavelength_um must be between 0.30 and 2.0 micrometres")

    @classmethod
    def from_value(cls, value: "ObserverConfig | dict[str, object] | object | None") -> "ObserverConfig | None":
        """Normalize mappings or ObserverConfig-like objects through one API."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            data = dict(value)
            horizon_value = data.pop("horizon", None)
            if isinstance(horizon_value, HorizonProfile):
                horizon = horizon_value
            elif isinstance(horizon_value, dict):
                horizon = HorizonProfile(
                    np.asarray(horizon_value.get("azimuth_deg", [0.0, 180.0]), dtype=float),
                    np.asarray(horizon_value.get("altitude_deg", [0.0, 0.0]), dtype=float),
                    source=str(horizon_value.get("source", "serialized horizon")),
                )
            else:
                horizon = None
            allowed = {
                "latitude_deg", "longitude_east_deg", "elevation_m",
                "pressure_hpa", "temperature_c", "relative_humidity_pct",
                "wavelength_um", "apply_refraction", "refraction_model",
                "weather_source", "name",
            }
            return cls(horizon=horizon, **{key: data[key] for key in allowed if key in data})
        required = ("latitude_deg", "longitude_east_deg")
        if all(hasattr(value, name) for name in required):
            return cls(
                latitude_deg=float(value.latitude_deg),
                longitude_east_deg=float(value.longitude_east_deg),
                elevation_m=float(getattr(value, "elevation_m", 0.0)),
                pressure_hpa=float(getattr(value, "pressure_hpa", 1010.0)),
                temperature_c=float(getattr(value, "temperature_c", 10.0)),
                relative_humidity_pct=float(getattr(value, "relative_humidity_pct", 0.0)),
                wavelength_um=float(getattr(value, "wavelength_um", 0.55)),
                apply_refraction=bool(getattr(value, "apply_refraction", True)),
                refraction_model=str(getattr(value, "refraction_model", "Bennett/Saemundsson + Edlen wavelength scaling")),
                weather_source=str(getattr(value, "weather_source", "standard atmosphere defaults")),
                horizon=getattr(value, "horizon", None),
                name=str(getattr(value, "name", "WGS-84 observer")),
            )
        raise TypeError("observer must be ObserverConfig, mapping, ObserverConfig-like object, or None")

    @property
    def effective_horizon(self) -> HorizonProfile:
        return self.horizon if self.horizon is not None else HorizonProfile.flat()

    def to_dict(self) -> dict[str, object]:
        horizon = self.effective_horizon
        return {
            "latitude_deg": float(self.latitude_deg),
            "longitude_east_deg": float(self.longitude_east_deg),
            "elevation_m": float(self.elevation_m),
            "pressure_hpa": float(self.pressure_hpa),
            "temperature_c": float(self.temperature_c),
            "relative_humidity_pct": float(self.relative_humidity_pct),
            "wavelength_um": float(self.wavelength_um),
            "apply_refraction": bool(self.apply_refraction),
            "refraction_model": str(self.refraction_model),
            "weather_source": str(self.weather_source),
            "name": str(self.name),
            "horizon": {
                "source": horizon.source,
                "sample_count": int(len(horizon.azimuth_deg)),
                "azimuth_deg": np.asarray(horizon.azimuth_deg, dtype=float).tolist(),
                "altitude_deg": np.asarray(horizon.altitude_deg, dtype=float).tolist(),
            },
        }


@dataclass(frozen=True)
class ObserverState:
    jd_utc: float
    observer_gcrf_km: np.ndarray
    up_gcrf: np.ndarray
    north_gcrf: np.ndarray
    east_gcrf: np.ndarray
    sun_altitude_geometric_deg: float
    sun_altitude_apparent_deg: float
    sun_azimuth_deg: float
    moon_altitude_geometric_deg: float
    moon_altitude_apparent_deg: float
    moon_azimuth_deg: float
    horizon_altitude_deg: float
    sun_horizon_clearance_deg: float
    moon_horizon_clearance_deg: float
    photosphere_visible: float
    lunar_limb_position_angle_deg: float
    lunar_limb_radius_km: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("observer_gcrf_km", "up_gcrf", "north_gcrf", "east_gcrf"):
            result[key] = np.asarray(result[key], dtype=float).tolist()
        return result


def atmospheric_refraction_deg(
    geometric_altitude_deg: float,
    *,
    pressure_hpa: float = 1010.0,
    temperature_c: float = 10.0,
    relative_humidity_pct: float = 0.0,
    wavelength_um: float = 0.55,
) -> float:
    """Bennett/Sæmundsson refraction with weather and wavelength scaling.

    The geometric Bennett form is preserved exactly for the historical
    defaults (dry air at 0.55 micrometres).  Humidity applies a small
    refractivity reduction using saturation-vapour pressure, while the
    Edlén dispersion ratio scales non-standard optical/near-IR wavelengths.
    Below -1 degree the approximation is clamped rather than extrapolated.
    """
    if pressure_hpa <= 0.0:
        return 0.0
    h = max(float(geometric_altitude_deg), -1.0)
    denominator = math.tan(math.radians(h + 10.3 / (h + 5.11)))
    if abs(denominator) < 1e-12:
        return 0.0
    arcmin = 1.02 / denominator
    temperature = float(temperature_c)
    humidity = float(np.clip(relative_humidity_pct, 0.0, 100.0))
    saturation_hpa = 6.112 * math.exp(17.62 * temperature / (243.12 + temperature))
    vapour_hpa = humidity / 100.0 * saturation_hpa
    effective_pressure = max(0.0, float(pressure_hpa) - 0.15 * vapour_hpa)
    wavelength = float(np.clip(wavelength_um, 0.30, 2.0))
    def refractivity(lam_um: float) -> float:
        sigma2 = (1.0 / lam_um) ** 2
        return 8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)
    dispersion = refractivity(wavelength) / refractivity(0.55)
    scale = (effective_pressure / 1010.0) * (283.0 / (273.0 + temperature)) * dispersion
    return max(0.0, arcmin * scale / 60.0)


def _observer_axes_itrf(config: ObserverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = math.radians(float(config.latitude_deg))
    lon = math.radians(float(config.longitude_east_deg))
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    return up, north, east


def observer_state(
    event: ReferenceEvent,
    jd_utc: float,
    config: ObserverConfig,
    *,
    sun_gcrf_km: np.ndarray | None = None,
    moon_gcrf_km: np.ndarray | None = None,
    limb_profile: LunarLimbProfile | None = None,
) -> ObserverState:
    from ssatk.compute.eclipse_reference_events import (
        R_SUN_KM,
        angular_circle_visible_fraction,
        itrf_surface_point,
    )
    from ssatk.compute.eclipse_runtime import itrf_to_gcrf_km
    from ssatk.compute.eclipse_state import event_positions_gcrf

    if sun_gcrf_km is None or moon_gcrf_km is None:
        sun_gcrf_km, moon_gcrf_km = event_positions_gcrf(event, float(jd_utc))
    sun = np.asarray(sun_gcrf_km, dtype=float)
    moon = np.asarray(moon_gcrf_km, dtype=float)
    observer_itrf = itrf_surface_point(
        config.latitude_deg,
        config.longitude_east_deg,
        float(config.elevation_m) / 1000.0,
    )
    observer = np.asarray(itrf_to_gcrf_km(observer_itrf, float(jd_utc)), dtype=float)
    axes_itrf = np.vstack(_observer_axes_itrf(config))
    axes_gcrf = np.asarray(
        itrf_to_gcrf_km(axes_itrf, np.full(3, float(jd_utc))),
        dtype=float,
    )
    up, north, east = [row / np.linalg.norm(row) for row in axes_gcrf]

    def alt_az(vector: np.ndarray) -> tuple[float, float]:
        unit = vector / np.linalg.norm(vector)
        altitude = math.degrees(math.asin(np.clip(np.dot(unit, up), -1.0, 1.0)))
        azimuth = math.degrees(math.atan2(np.dot(unit, east), np.dot(unit, north))) % 360.0
        return altitude, azimuth

    to_sun = sun - observer
    to_moon = moon - observer
    sun_alt, sun_az = alt_az(to_sun)
    moon_alt, moon_az = alt_az(to_moon)
    sun_ref = atmospheric_refraction_deg(
        sun_alt, pressure_hpa=config.pressure_hpa, temperature_c=config.temperature_c,
        relative_humidity_pct=config.relative_humidity_pct, wavelength_um=config.wavelength_um
    ) if config.apply_refraction else 0.0
    moon_ref = atmospheric_refraction_deg(
        moon_alt, pressure_hpa=config.pressure_hpa, temperature_c=config.temperature_c,
        relative_humidity_pct=config.relative_humidity_pct, wavelength_um=config.wavelength_um
    ) if config.apply_refraction else 0.0
    horizon_alt = float(config.effective_horizon.altitude_at(sun_az))

    profile = limb_profile or LunarLimbProfile.circular()
    pa = apparent_position_angle_deg(to_sun, to_moon)
    limb_radius = float(profile.radius_km(pa))

    sun_distance = float(np.linalg.norm(to_sun))
    moon_distance = float(np.linalg.norm(to_moon))
    sun_ang = math.asin(np.clip(R_SUN_KM / sun_distance, 0.0, 1.0))
    moon_ang = math.asin(np.clip(limb_radius / moon_distance, 0.0, 1.0))
    separation = math.acos(np.clip(np.dot(to_sun / sun_distance, to_moon / moon_distance), -1.0, 1.0))
    visible = float(angular_circle_visible_fraction(moon_ang, sun_ang, separation))

    return ObserverState(
        jd_utc=float(jd_utc),
        observer_gcrf_km=observer,
        up_gcrf=up,
        north_gcrf=north,
        east_gcrf=east,
        sun_altitude_geometric_deg=sun_alt,
        sun_altitude_apparent_deg=sun_alt + sun_ref,
        sun_azimuth_deg=sun_az,
        moon_altitude_geometric_deg=moon_alt,
        moon_altitude_apparent_deg=moon_alt + moon_ref,
        moon_azimuth_deg=moon_az,
        horizon_altitude_deg=horizon_alt,
        sun_horizon_clearance_deg=sun_alt + sun_ref - horizon_alt,
        moon_horizon_clearance_deg=moon_alt + moon_ref - float(config.effective_horizon.altitude_at(moon_az)),
        photosphere_visible=visible,
        lunar_limb_position_angle_deg=pa,
        lunar_limb_radius_km=limb_radius,
    )


def refine_solar_contacts(
    event: ReferenceEvent,
    config: ObserverConfig,
    *,
    limb_profile: LunarLimbProfile | None = None,
    reference_contacts: dict[str, float] | None = None,
) -> dict[str, float]:
    """Root-find local solar contacts with the event's selected backend.

    External contacts C1/C4 are returned for every visible partial eclipse.
    Internal contacts C2/C3 are returned only when the reference scaffold
    indicates totality/annularity at the site.  ``MAX`` is the instant of
    minimum apparent centre separation, not an arbitrary point on a zero-
    visibility plateau.  Refraction and terrain affect practical visibility,
    but not the geometric disc-contact equation.
    """
    from ssatk.compute.eclipse_reference_events import R_SUN_KM, solar_local_contacts
    from ssatk.compute.eclipse_state import event_positions_gcrf

    try:
        from scipy.optimize import brentq, minimize_scalar
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for contact refinement") from exc

    contacts = dict(reference_contacts or solar_local_contacts(
        float(config.latitude_deg),
        float(config.longitude_east_deg),
        float(config.elevation_m) / 1000.0,
    ))
    if "C1" not in contacts or "C4" not in contacts:
        raise RuntimeError(f"observer {config.name} has no bracketed solar eclipse")
    profile = limb_profile or LunarLimbProfile.circular()

    def geometry(jd: float) -> tuple[float, float, float]:
        sun, moon = event_positions_gcrf(event, jd)
        state = observer_state(
            event, jd, config, sun_gcrf_km=sun, moon_gcrf_km=moon,
            limb_profile=profile,
        )
        observer = state.observer_gcrf_km
        to_sun = sun - observer
        to_moon = moon - observer
        ds = float(np.linalg.norm(to_sun))
        dm = float(np.linalg.norm(to_moon))
        sep = math.acos(np.clip(np.dot(to_sun / ds, to_moon / dm), -1.0, 1.0))
        sun_ang = math.asin(np.clip(R_SUN_KM / ds, 0.0, 1.0))
        moon_ang = math.asin(np.clip(state.lunar_limb_radius_km / dm, 0.0, 1.0))
        return sep, sun_ang, moon_ang

    def equation(jd: float, internal: bool) -> float:
        sep, sun_ang, moon_ang = geometry(jd)
        boundary = abs(moon_ang - sun_ang) if internal else moon_ang + sun_ang
        return sep - boundary

    reference_max = float(contacts.get("MAX", event.greatest_jd))
    references = {name: float(contacts[name]) for name in ("C1", "C2", "C3", "C4") if name in contacts}
    bounds: dict[str, tuple[float, float]] = {
        "C1": (references["C1"] - 1.0 / 24.0, reference_max),
        "C4": (reference_max, references["C4"] + 1.0 / 24.0),
    }
    if "C2" in references and "C3" in references:
        bounds["C2"] = (0.5 * (references["C1"] + references["C2"]), reference_max)
        bounds["C3"] = (reference_max, 0.5 * (references["C3"] + references["C4"]))

    result: dict[str, float] = {}
    for name in ("C1", "C2", "C3", "C4"):
        if name not in references:
            continue
        lo, hi = bounds[name]
        internal = name in {"C2", "C3"}
        grid = np.linspace(lo, hi, 321)
        values = np.array([equation(float(value), internal) for value in grid])
        candidates: list[tuple[float, float]] = []
        for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if abs(float(f_left)) < 1e-15:
                candidates.append((float(left), float(left)))
            elif f_left * f_right < 0.0:
                candidates.append((float(left), float(right)))
        if not candidates:
            raise RuntimeError(f"could not bracket {name} for observer {config.name}")
        reference = references[name]
        pair = min(candidates, key=lambda item: abs(0.5 * (item[0] + item[1]) - reference))
        root = pair[0] if pair[0] == pair[1] else brentq(
            lambda value: equation(value, internal), pair[0], pair[1], xtol=1e-12
        )
        result[name] = float(root)

    midpoint = 0.5 * (result["C1"] + result["C4"])
    lo_s = (result["C1"] - midpoint) * 86400.0
    hi_s = (result["C4"] - midpoint) * 86400.0
    optimized = minimize_scalar(
        lambda offset_s: geometry(midpoint + float(offset_s) / 86400.0)[0],
        bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1.0e-4},
    )
    result["MAX"] = float(midpoint + optimized.x / 86400.0)
    return result


def contact_aware_observer_jd(
    event: ReferenceEvent,
    config: ObserverConfig,
    *,
    n_frames: int = 91,
    contacts: dict[str, float] | None = None,
    limb_profile: LunarLimbProfile | None = None,
) -> np.ndarray:
    """Return a local C1--C4 timeline retaining every solved contact exactly.

    The baseline is cosine-distributed to preserve ingress and egress detail.
    Exact C1/MAX/C4 and, when present, C2/C3 are inserted without rounding.
    A compact cluster around maximum prevents short total phases from being
    skipped at low frame budgets.  Partial-only sites are valid and do not
    receive synthetic internal contacts.
    """
    solved = dict(contacts or refine_solar_contacts(
        event, config, limb_profile=limb_profile,
    ))
    if "C1" not in solved or "C4" not in solved:
        raise ValueError("the observer does not have a bracketed local solar eclipse")
    start, stop = float(solved["C1"]), float(solved["C4"])
    n_frames = max(17, int(n_frames))
    u = np.linspace(0.0, 1.0, n_frames)
    base = start + (stop - start) * (0.5 - 0.5 * np.cos(np.pi * u))
    maximum = float(solved.get("MAX", 0.5 * (start + stop)))
    offsets_s = np.array([-600, -300, -180, -120, -60, -20, 0,
                           20, 60, 120, 180, 300, 600], dtype=float)
    exact = np.concatenate([
        np.asarray([float(value) for _, value in sorted(
            solved.items(), key=lambda item: float(item[1])
        )], dtype=float),
        maximum + offsets_s / 86400.0,
    ])
    exact = exact[(exact >= start) & (exact <= stop)]
    keep = np.ones(base.shape, dtype=bool)
    for value in exact:
        keep &= np.abs(base - value) > 0.05 / 86400.0
    return np.sort(np.unique(np.concatenate([base[keep], exact])))


