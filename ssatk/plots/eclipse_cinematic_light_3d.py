"""V22.2 public cinematic finite-Sun eclipse renderer with SSAPy-Data assets.

The renderer consumes one immutable :class:`ReferenceEvent` backend from the
state layer and never re-probes frame providers during rendering.  It emits a
standalone Three.js document with:

* exact WGS-84 Earth, spherical Moon, and spherical solar photosphere geometry;
* weighted samples distributed across the complete visible solar photosphere;
* first-positive opaque-body clipping for every explanatory light path;
* continuously evaluated finite-disc irradiance in the Earth/Moon fragment shader;
* contact-aware solver states played at a uniform rate in physical UTC time;
* body-following, north-up cameras plus a real greatest-site observer view;
* explicit labels that distinguish bodies, target points, and diagnostic rays;
* capability-negotiated HDR/postprocessing with a basic-render fallback.

The bundled JavaScript, Three.js source, textures, and state payload are embedded
in each HTML file.  Viewing the file requires no Node, npm, CDN, or network
connection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import io
import html
import json
import math

import numpy as np

from ssatk.compute.eclipse_brightness import illumination_fraction, irradiance_fraction
from ssatk.io.eclipse_asset_resolver import resolve_asset, resolve_image
from ssatk.compute.eclipse_photometry import resolve_limb_darkening
from ssatk.compute.eclipse_state import build_event as build_backend_event, event_gcrf_to_itrf_km, event_itrf_to_gcrf_km, event_moon_body_to_gcrf
from ssatk.plots.eclipse_system_raytrace_3d import ASSET_DIR, _scene_coordinates
from ssatk.coordinates.eclipse_observer_geometry import ObserverConfig, observer_state, refine_solar_contacts
from ssatk.coordinates.eclipse_lunar_geometry import LunarLimbProfile
from ssatk.plots.eclipse_ultra_public_3d import (
    PublicPeakState,
    build_public_peak_state,
    photospheric_target_gcrf,
    sampled_photospheric_path_records,
)
from ssatk.plots.eclipse_ultra_public_animation import build_public_animation
from ssatk.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    RP_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    ReferenceDefinition,
    ReferenceEvent,
    geodetic_from_itrf,
    itrf_surface_point,
    jd_to_datetime,
)
from ssatk.compute.eclipse_raytrace import LUNAR_DANJON_EARTH_RADIUS_KM

UNIT_KM = 1_000.0
THREE_DIR = ASSET_DIR
RUNTIME_JS = ASSET_DIR / "eclipse_cinematic_runtime.js"
RENDERER_VERSION = "20.1"
PACKAGE_VERSION = "2.2.2"

# A stable Earth-centered reference is useful in the system and Earth views:
# it gives the body scale an orbital context without pretending to be part of
# the eclipse solution.  The state history itself remains owned by the
# eclipse backend below.
_SSAPY_REFERENCE_ORBIT = {
    "name": "SSAPy geocentric reference orbit",
    "a_km": 42_164.0,
    "e": 0.0,
    "inc_deg": 0.0,
    "argp_deg": 0.0,
    "raan_deg": 0.0,
    "nu_deg": 0.0,
}


@dataclass(frozen=True)
class SSAPyOrbitOverlay:
    """A display-only SSAPy orbit to draw in the cinematic scene.

    ``orbit`` may be any existing SSAPy ``Orbit``.  When it is omitted, the
    built-in geocentric reference orbit is constructed from the elements above.
    """

    orbit: object | None = None
    name: str = "SSAPy orbit"
    color: str = "#5fffa2"
    sample_count: int = 193
    propagator: object | None = None


@dataclass(frozen=True)
class CinematicQuality:
    earth_lat: int
    earth_lon: int
    moon_lat: int
    moon_lon: int
    sun_lat: int
    sun_lon: int
    ray_count: int
    local_span_km: float


QUALITY = {
    # Motion presets keep complete-event HTML practical while preserving the
    # exact event state, body radii, ray origins, and first-surface clipping.
    "motion": CinematicQuality(80, 160, 80, 160, 48, 96, 25, 560_000.0),
    "motion_high": CinematicQuality(120, 240, 120, 240, 72, 144, 37, 620_000.0),
    "high": CinematicQuality(160, 320, 160, 320, 96, 192, 37, 620_000.0),
    "ultra": CinematicQuality(224, 448, 224, 448, 128, 256, 61, 720_000.0),
    "cinema": CinematicQuality(288, 576, 288, 576, 160, 320, 85, 850_000.0),
}

_DEFAULT_SOLVER_STATES = {
    ("solar", "global"): 121,
    ("solar", "local"): 91,
    ("lunar", "global"): 91,
}
_PLAYBACK_SECONDS = {
    ("solar", "global"): 36.0,
    ("solar", "local"): 24.0,
    ("lunar", "global"): 32.0,
}


def _resolve_photometry_lut() -> tuple[dict[str, object], object, object]:
    """Resolve the radiometric lookup table and metadata from SSAPy-Data."""
    binary = resolve_asset("quadratic_visible_delta_lut", policy="strict-data")
    metadata = resolve_asset("quadratic_visible_delta_lut_metadata", policy="strict-data")
    if binary is None or metadata is None:
        raise RuntimeError("SSAPy-Data photometry LUT assets are unavailable")
    try:
        payload = json.loads(Path(metadata.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("photometry correction LUT metadata is missing or invalid") from exc
    expected = int(payload["width"]) * int(payload["height"])
    if Path(binary.path).stat().st_size != expected:
        raise RuntimeError("photometry correction LUT binary has the wrong size")
    return payload, binary, metadata


def _quality(name: str) -> CinematicQuality:
    key = str(name).strip().lower()
    if key not in QUALITY:
        raise ValueError("quality must be 'motion', 'motion_high', 'high', 'ultra', or 'cinema'")
    return QUALITY[key]


def _definition(kind: str | ReferenceDefinition | ReferenceEvent) -> ReferenceDefinition:
    if isinstance(kind, ReferenceEvent):
        return kind.definition
    if isinstance(kind, ReferenceDefinition):
        return kind
    return SOLAR_2024 if str(kind).lower().startswith("solar") else LUNAR_2025


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _resolved_image_uri(logical_name: str, *, max_width: int, quality: int = 90) -> tuple[str, dict[str, object]]:
    """Resolve a PNG/JPEG from SSAPy-Data and embed an optimized derivative.

    The source image remains the auditable SSAPy-Data asset.  WebP is an
    in-memory HTML transport encoding only; no WebP file is required in either
    the Toolkit package or SSAPy-Data repository.
    """
    from PIL import Image

    resolved = resolve_image(logical_name, policy="data-first", required=True)
    with Image.open(resolved.path) as image:
        image = image.convert("RGB")
        if image.width > int(max_width):
            height = max(1, round(image.height * int(max_width) / image.width))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize((int(max_width), height), resampling)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=int(quality), method=6)
    uri = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    provenance = {
        "logical_name": logical_name,
        "source": resolved.source,
        "source_name": Path(resolved.path).name,
        "source_sha256": resolved.sha256,
        "source_bytes": resolved.bytes,
        "embedded_encoding": "image/webp",
        "embedded_bytes": len(buffer.getvalue()),
    }
    return uri, provenance


def _three_module_uri() -> str:
    module_path = THREE_DIR / "eclipse_three_module.min.js"
    core_path = THREE_DIR / "eclipse_three_core.js"
    if not module_path.is_file() or not core_path.is_file():
        raise FileNotFoundError(
            "Vendored Three.js runtime is incomplete; expected flat eclipse_three_* assets in ssatk/plots"
        )
    core_uri = _data_uri(core_path, "text/javascript")
    module = module_path.read_text(encoding="utf-8")
    old = 'from"./three.core.min.js"'
    if old not in module:
        raise RuntimeError("Unexpected Three.js module layout; core import not found")
    module = module.replace(old, f'from"{core_uri}"')
    return "data:text/javascript;base64," + base64.b64encode(
        module.encode("utf-8")
    ).decode("ascii")


def _rotation_rows_itrf_to_scene(state: PublicPeakState) -> np.ndarray:
    """Column-vector rotation matrix from the event-owned ITRF to scene."""
    gcrf_rows = np.asarray(
        event_itrf_to_gcrf_km(
            state.event,
            np.eye(3),
            np.full(3, float(state.jd_utc), dtype=float),
        ),
        dtype=float,
    ).reshape(3, 3)
    # Row-vector body -> GCRF -> scene, then transpose for Three.js columns.
    matrix = (gcrf_rows @ state.basis_world_from_scene).T
    u, _, vt = np.linalg.svd(matrix)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vt
    return matrix


def _rotation_rows_moon_to_scene(state: PublicPeakState) -> np.ndarray:
    body_axes_gcrf = np.asarray(
        event_moon_body_to_gcrf(state.event, state.jd_utc), dtype=float
    ).reshape(3, 3)
    matrix = (body_axes_gcrf.T @ state.basis_world_from_scene).T
    u, _, vt = np.linalg.svd(matrix)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vt
    return matrix


def _to_scene(points_gcrf_km: np.ndarray, state: PublicPeakState) -> np.ndarray:
    return _scene_coordinates(
        np.asarray(points_gcrf_km, dtype=float),
        np.zeros(3),
        state.basis_world_from_scene,
        UNIT_KM,
    )


def _vector_to_scene(vector_gcrf: np.ndarray, state: PublicPeakState) -> np.ndarray:
    return np.asarray(vector_gcrf, dtype=float) @ state.basis_world_from_scene


def _path_pair(points_gcrf_km: np.ndarray, state: PublicPeakState) -> list[list[float]]:
    values = np.asarray(points_gcrf_km, dtype=float).reshape(-1, 3)
    if len(values) < 2:
        return []
    local = _to_scene(np.vstack([values[0], values[-1]]), state)
    return local.tolist()


def _tail_segment(points: np.ndarray, max_length_km: float) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(2, 3)
    delta = values[1] - values[0]
    length = float(np.linalg.norm(delta))
    if length <= max_length_km or length <= 1e-12:
        return values
    start = values[1] - delta / length * float(max_length_km)
    return np.vstack([start, values[1]])


def _sample_paths(
    state: PublicPeakState,
    *,
    n_samples: int,
    local_span_km: float,
    photometry: str,
) -> tuple[list[list[object]], list[list[object]], dict[str, float | int]]:
    """Return weighted full-disc explanatory paths in full and local frames."""
    full: list[list[object]] = []
    local: list[list[object]] = []
    records = sampled_photospheric_path_records(state, n_samples=n_samples, photometry=photometry)
    for record in records:
        values = np.asarray(record.points_gcrf_km, dtype=float).reshape(2, 3)
        full_scene = _to_scene(values, state)
        local_scene = _to_scene(_tail_segment(values, local_span_km), state)
        suffix = [
            bool(record.blocked),
            float(record.weight),
            float(record.disk_radius_fraction),
            str(record.endpoint_body),
        ]
        full.append([full_scene[0].tolist(), full_scene[1].tolist(), *suffix])
        local.append([local_scene[0].tolist(), local_scene[1].tolist(), *suffix])

    sources = np.asarray([r.points_gcrf_km[0] for r in records], dtype=float)
    max_span = 0.0
    if len(sources) > 1:
        for i in range(len(sources)):
            max_span = max(max_span, float(np.max(np.linalg.norm(sources[i + 1 :] - sources[i], axis=1))) if i + 1 < len(sources) else 0.0)
    metrics = {
        "count": len(records),
        "blocked_count": int(sum(r.blocked for r in records)),
        "blocked_weight": float(sum(r.weight for r in records if r.blocked)),
        "photosphere_span_km": max_span,
        "photosphere_span_fraction": max_span / (2.0 * R_SUN_KM),
    }
    return full, local, metrics


def _fixed_observer_payload(
    state: PublicPeakState,
    config: ObserverConfig | None = None,
    limb_profile: LunarLimbProfile | None = None,
) -> dict[str, object]:
    """Topocentric observer payload with elevation, refraction, and horizon."""
    if config is None:
        config = ObserverConfig(
            latitude_deg=float(state.event.metadata.get("observer_lat_deg", SOLAR_GREATEST_SITE_LAT_DEG)),
            longitude_east_deg=float(state.event.metadata.get("observer_lon_east_deg", SOLAR_GREATEST_SITE_LON_EAST_DEG)),
            elevation_m=float(state.event.metadata.get("observer_elevation_m", 0.0)),
            name=str(state.event.metadata.get("observer_name", "NASA greatest-eclipse reference site")),
        )
    obs = observer_state(
        state.event, state.jd_utc, config,
        sun_gcrf_km=np.asarray(state.state.sun_gcrf_km, dtype=float),
        moon_gcrf_km=np.asarray(state.state.moon_gcrf_km, dtype=float),
        limb_profile=limb_profile,
    )
    sun = np.asarray(state.state.sun_gcrf_km, dtype=float)
    moon = np.asarray(state.state.moon_gcrf_km, dtype=float)
    to_sun = sun - obs.observer_gcrf_km
    to_moon = moon - obs.observer_gcrf_km
    sky_center = to_sun / np.linalg.norm(to_sun) + to_moon / np.linalg.norm(to_moon)
    sky_center /= np.linalg.norm(sky_center)
    return {
        "name": config.name,
        "latitude_deg": float(config.latitude_deg),
        "longitude_east_deg": float(config.longitude_east_deg),
        "elevation_m": float(config.elevation_m),
        "position": _to_scene(obs.observer_gcrf_km.reshape(1, 3), state)[0].tolist(),
        "up": _vector_to_scene(obs.up_gcrf, state).tolist(),
        "north": _vector_to_scene(obs.north_gcrf, state).tolist(),
        "east": _vector_to_scene(obs.east_gcrf, state).tolist(),
        "sky_direction": _vector_to_scene(sky_center, state).tolist(),
        "sun_altitude_deg": obs.sun_altitude_apparent_deg,
        "sun_altitude_geometric_deg": obs.sun_altitude_geometric_deg,
        "sun_azimuth_deg": obs.sun_azimuth_deg,
        "moon_altitude_deg": obs.moon_altitude_apparent_deg,
        "moon_altitude_geometric_deg": obs.moon_altitude_geometric_deg,
        "moon_azimuth_deg": obs.moon_azimuth_deg,
        "atmospheric_refraction_enabled": bool(config.apply_refraction),
        "pressure_hpa": float(config.pressure_hpa),
        "temperature_c": float(config.temperature_c),
        "horizon_source": config.effective_horizon.source,
        "horizon_altitude_deg": obs.horizon_altitude_deg,
        "sun_horizon_clearance_deg": obs.sun_horizon_clearance_deg,
        "moon_horizon_clearance_deg": obs.moon_horizon_clearance_deg,
        "photosphere_visible": obs.photosphere_visible,
        "lunar_limb_position_angle_deg": obs.lunar_limb_position_angle_deg,
        "lunar_limb_radius_km": obs.lunar_limb_radius_km,
    }


def _target_payload(state: PublicPeakState, photometry: str = "quadratic-visible") -> dict[str, object]:
    target_gcrf, label = photospheric_target_gcrf(state)
    target_gcrf = np.asarray(target_gcrf, dtype=float)
    result: dict[str, object] = {
        "label": str(label),
        "position": _to_scene(target_gcrf.reshape(1, 3), state)[0].tolist(),
        "latitude_deg": None,
        "longitude_east_deg": None,
        "photosphere_visible": None,
        "irradiance_visible": None,
    }
    if state.event.mode == "solar":
        target_itrf = np.asarray(
            event_gcrf_to_itrf_km(state.event, target_gcrf, state.jd_utc),
            dtype=float,
        )
        radius = float(np.linalg.norm(target_itrf))
        if RP_KM * 0.8 <= radius <= RE_KM * 1.2:
            lat, lon, _ = geodetic_from_itrf(target_itrf)
            result["latitude_deg"] = float(lat)
            result["longitude_east_deg"] = float(lon)
        moon = np.asarray(state.state.moon_gcrf_km, dtype=float)
        sun = np.asarray(state.state.sun_gcrf_km, dtype=float)
        result["photosphere_visible"] = float(
            illumination_fraction(
                target_gcrf - moon,
                R_body_km=R_MOON_MEAN_KM,
                sun_position_km=sun - moon,
            )
        )
        result["irradiance_visible"] = float(
            irradiance_fraction(
                target_gcrf - moon, R_body_km=R_MOON_MEAN_KM,
                sun_position_km=sun - moon, photometry=photometry,
            )
        )
    else:
        sun = np.asarray(state.state.sun_gcrf_km, dtype=float)
        moon = np.asarray(state.state.moon_gcrf_km, dtype=float)
        result["photosphere_visible"] = float(
            illumination_fraction(
                moon,
                R_body_km=LUNAR_DANJON_EARTH_RADIUS_KM,
                sun_position_km=sun,
            )
        )
        result["irradiance_visible"] = float(
            irradiance_fraction(
                moon, R_body_km=LUNAR_DANJON_EARTH_RADIUS_KM,
                sun_position_km=sun, photometry=photometry,
            )
        )
    return result


def _frame_payload(
    state: PublicPeakState,
    *,
    phase: str,
    quality: CinematicQuality,
    scope: str,
    observer_config: ObserverConfig | None = None,
    lunar_limb_profile: LunarLimbProfile | None = None,
    photometry: str = "quadratic-visible",
) -> dict[str, object]:
    full_rays, local_rays, ray_metrics = _sample_paths(
        state,
        n_samples=quality.ray_count,
        local_span_km=quality.local_span_km,
        photometry=photometry,
    )
    s = state.state
    moon_scene = _to_scene(np.asarray(s.moon_gcrf_km).reshape(1, 3), state)[0]
    sun_scene = _to_scene(np.asarray(s.sun_gcrf_km).reshape(1, 3), state)[0]
    umbra = [_path_pair(path, state) for path in s.umbra_gcrf_km]
    penumbra = [_path_pair(path, state) for path in s.penumbra_gcrf_km]
    target = _target_payload(state, photometry=photometry)
    observer = (
        _fixed_observer_payload(state, observer_config, lunar_limb_profile)
        if state.event.mode == "solar" and scope == "local"
        else None
    )
    return {
        "jd_utc": float(state.jd_utc),
        "time": jd_to_datetime(state.jd_utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "phase": str(phase),
        "moon": moon_scene.tolist(),
        "sun": sun_scene.tolist(),
        "earth_rotation": _rotation_rows_itrf_to_scene(state).tolist(),
        "moon_rotation": _rotation_rows_moon_to_scene(state).tolist(),
        "full_rays": full_rays,
        "local_rays": local_rays,
        "ray_metrics": ray_metrics,
        "central": _path_pair(s.central_gcrf_km, state),
        "shadow": _path_pair(s.shadow_axis_gcrf_km, state),
        "umbra": umbra,
        "penumbra": penumbra,
        "earth_moon_km": float(np.linalg.norm(s.moon_gcrf_km)),
        "sun_earth_km": float(np.linalg.norm(s.sun_gcrf_km)),
        "target": target,
        "observer": observer,
    }


def _contact_payload(event) -> list[dict[str, object]]:
    contacts = dict(event.contacts_jd)
    contacts.setdefault("MAX", float(event.greatest_jd))
    ordered = sorted(contacts.items(), key=lambda item: float(item[1]))
    return [
        {
            "name": str(name),
            "jd_utc": float(jd),
            "time": jd_to_datetime(float(jd)).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        for name, jd in ordered
    ]


def _observer_model_from_event(event) -> dict[str, object] | None:
    value = event.metadata.get("observer_model")
    return dict(value) if isinstance(value, dict) else None


def _ssapy_orbit_overlay(
    states: list[PublicPeakState], overlay: SSAPyOrbitOverlay,
) -> dict[str, object]:
    """Build a display-only orbit overlay with SSAPy's public propagation API.

    The orbit is a geocentric reference path, separate from the validated
    eclipse state history.  ``ssapy.rv`` supplies both the path and the
    time-matched marker positions; no propagation equations are duplicated in
    this renderer.  A missing optional runtime component simply disables the
    overlay while leaving the eclipse product usable.
    """
    disabled = {
        "enabled": False,
        "name": str(overlay.name),
        "color": str(overlay.color),
        "reason": "SSAPy orbit overlay unavailable",
    }
    if not states:
        return disabled
    try:
        import ssapy
        from astropy.time import Time
        from ssapy.propagator import KeplerianPropagator
    except ImportError as exc:
        return {**disabled, "reason": f"SSAPy orbit overlay unavailable: {exc}"}

    start_jd = float(states[0].jd_utc)
    start_time = Time(start_jd, format="jd", scale="utc")
    try:
        if overlay.orbit is not None:
            orbit = overlay.orbit
        else:
            elements = _SSAPY_REFERENCE_ORBIT
            a_m = float(elements["a_km"]) * UNIT_KM
            try:
                orbit = ssapy.Orbit.fromKeplerianElements(
                    a_m,
                    float(elements["e"]),
                    math.radians(float(elements["inc_deg"])),
                    math.radians(float(elements["argp_deg"])),
                    math.radians(float(elements["raan_deg"])),
                    math.radians(float(elements["nu_deg"])),
                    t=start_time,
                )
            except TypeError:
                # SSAPy releases before the keyword-time signature accept GPS
                # seconds as the final positional argument.
                orbit = ssapy.Orbit.fromKeplerianElements(
                    a_m,
                    float(elements["e"]),
                    math.radians(float(elements["inc_deg"])),
                    math.radians(float(elements["argp_deg"])),
                    math.radians(float(elements["raan_deg"])),
                    math.radians(float(elements["nu_deg"])),
                    start_time.gps,
                )
        period_seconds = float(np.asarray(orbit.period, dtype=float).reshape(()))
        if not math.isfinite(period_seconds) or period_seconds <= 0.0:
            return {**disabled, "reason": "SSAPy returned an invalid reference-orbit period"}

        path_count = max(16, int(overlay.sample_count))
        path_offsets_s = np.linspace(0.0, period_seconds, path_count)
        marker_jd = np.asarray([float(state.jd_utc) for state in states], dtype=float)
        sample_jd = np.concatenate((
            start_jd + path_offsets_s / 86_400.0,
            marker_jd,
        ))
        sample_time = Time(sample_jd, format="jd", scale="utc")
        propagator = (
            overlay.propagator
            if overlay.propagator is not None else KeplerianPropagator()
        )
        try:
            positions_m, _ = ssapy.rv(
                orbit=orbit, time=sample_time, propagator=propagator,
            )
        except TypeError:
            positions_m, _ = ssapy.rv(orbit, sample_time, propagator)
        positions_km = np.asarray(positions_m, dtype=float).reshape(-1, 3) / UNIT_KM
        if len(positions_km) != len(sample_jd):
            return {**disabled, "reason": "SSAPy returned an unexpected reference-orbit state shape"}
        basis_state = states[0]
        path_scene = _to_scene(positions_km[:path_count], basis_state)
        marker_scene = _to_scene(positions_km[path_count:], basis_state)
        return {
            "enabled": True,
            "name": str(overlay.name),
            "color": str(overlay.color),
            "central_body": "Earth",
            "frame": "GCRF",
            "propagator": type(propagator).__name__,
            "period_seconds": period_seconds,
            "path": path_scene.tolist(),
            "markers": marker_scene.tolist(),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {**disabled, "reason": f"SSAPy orbit overlay unavailable: {exc}"}


def build_cinematic_payload(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    *,
    animate: bool = False,
    quality: str = "ultra",
    n_frames: int | None = None,
    backend: str = "auto",
    solar_scope: str = "global",
    playback_seconds: float | None = None,
    observer_config: ObserverConfig | None = None,
    lunar_limb_profile: LunarLimbProfile | None = None,
    photometry: str = "quadratic-visible",
    orbit_overlays: list[SSAPyOrbitOverlay | object] | None = None,
) -> dict[str, object]:
    q = _quality(quality)
    photometry_law = resolve_limb_darkening(photometry)
    definition = _definition(kind)
    source = kind if isinstance(kind, ReferenceEvent) else definition
    if isinstance(kind, ReferenceEvent):
        solar_scope = str(kind.metadata.get("solar_scope", solar_scope))
        backend = str(kind.metadata.get("state_source", backend))
    scope = str(solar_scope).lower() if definition.mode == "solar" else "global"
    if scope not in {"global", "local"}:
        raise ValueError("solar_scope must be 'global' or 'local'")

    refined_observer_contacts = None
    if definition.mode == "solar" and scope == "local" and observer_config is not None:
        preflight = build_backend_event(
            definition, backend=backend, n_frames=65, solar_scope="global",
        )
        refined_observer_contacts = refine_solar_contacts(
            preflight, observer_config, limb_profile=lunar_limb_profile,
        )

    if animate:
        requested = int(n_frames or _DEFAULT_SOLVER_STATES[(definition.mode, scope)])
        animation = build_public_animation(
            source,
            n_frames=requested,
            solar_scope=scope,
            ray_azimuth=max(48, q.ray_count),
            backend=backend,
            observer=observer_config,
            observer_contacts=refined_observer_contacts,
        )
        states = [
            _frame_payload(
                state, phase=phase, quality=q, scope=scope,
                observer_config=observer_config, lunar_limb_profile=lunar_limb_profile,
                photometry=photometry_law.name,
            )
            for state, phase in zip(animation.states, animation.phase_labels)
        ]
        audit = animation.greatest_state.ssapy_diagnostics
        event = animation.event
        orbit_states = list(animation.states)
    else:
        requested = 1
        peak = build_public_peak_state(
            source,
            ray_azimuth=max(48, q.ray_count),
            backend=backend,
            solar_scope=scope,
            observer=observer_config,
        )
        states = [_frame_payload(
            peak, phase="MAX", quality=q, scope=scope,
            observer_config=observer_config, lunar_limb_profile=lunar_limb_profile,
            photometry=photometry_law.name,
        )]
        audit = peak.ssapy_diagnostics
        event = peak.event
        orbit_states = [peak]

    requested_orbits = (
        [SSAPyOrbitOverlay(name=str(_SSAPY_REFERENCE_ORBIT["name"]))]
        if orbit_overlays is None else list(orbit_overlays)
    )
    normalized_orbits = [
        value if isinstance(value, SSAPyOrbitOverlay)
        else SSAPyOrbitOverlay(orbit=value, name=f"SSAPy orbit {index + 1}")
        for index, value in enumerate(requested_orbits)
    ]
    orbit_payloads = []
    orbit_markers: list[list[list[float]]] = []
    for overlay in normalized_orbits:
        payload = _ssapy_orbit_overlay(orbit_states, overlay)
        orbit_markers.append(payload.pop("markers", []))
        orbit_payloads.append(payload)
    for frame_index, frame in enumerate(states):
        frame["ssapy_orbit_positions"] = [
            markers[frame_index] if frame_index < len(markers) else None
            for markers in orbit_markers
        ]

    selected_backend = str(event.metadata.get("state_source", event.backend))
    frame_backend = str(event.metadata.get("frame_backend", event.frame))
    duration_s = float((states[-1]["jd_utc"] - states[0]["jd_utc"]) * 86400.0)
    default_playback = _PLAYBACK_SECONDS[(definition.mode, scope)] if animate else 0.0
    playback = float(playback_seconds if playback_seconds is not None else default_playback)
    if animate and playback <= 0.0:
        raise ValueError("playback_seconds must be positive for an animation")

    photometry_lut, _, _ = _resolve_photometry_lut()

    labels = {
        "earth": "Earth — opaque WGS-84 body",
        "moon": "Moon — opaque mean-radius body",
        "sun": "Sun — true photosphere",
        "target": "Eclipse evaluation point",
    }
    display_title = definition.title
    if definition.mode == "solar" and scope == "local":
        observer_name = (
            observer_config.name if observer_config is not None
            else "NASA greatest-eclipse observer"
        )
        contact_names = set(event.contacts_jd)
        is_central = {"C2", "C3"}.issubset(contact_names)
        phase_scope = "local C1–C4" if is_central else "local partial C1–MAX–C4"
        scope_label = f"{observer_name} · {phase_scope}"
        if not is_central:
            display_title = definition.title.replace("Total Solar Eclipse", "Partial Solar Eclipse", 1)
    else:
        if animate:
            scope_label = (
                "complete global P1–P4 solar eclipse" if definition.mode == "solar"
                else "complete lunar P1–P4 eclipse"
            )
        else:
            scope_label = "greatest-eclipse physical inspection"
    return {
        "version": RENDERER_VERSION,
        "package_version": PACKAGE_VERSION,
        "mode": definition.mode,
        "scope": scope,
        "scope_label": scope_label,
        "title": display_title,
        "animate": bool(animate),
        "backend": selected_backend,
        "frame_backend": frame_backend,
        "strict_backend": bool(event.metadata.get("strict_backend", False)),
        "backend_audit": audit,
        "backend_provenance": {
            "requested": str(event.metadata.get("backend_requested", backend)),
            "selected": selected_backend,
            "ephemeris": str(event.metadata.get("ephemeris_backend", event.backend)),
            "frame": frame_backend,
            "fallback_reason": str(event.metadata.get("backend_fallback_reason", "")),
        },
        "moon_orientation_backend": str(event.metadata.get(
            "moon_orientation_backend", "NAIF IAU_MOON 2009 text-PCK fallback"
        )),
        "observer_model": (
            observer_config.to_dict() if observer_config is not None
            else _observer_model_from_event(event)
        ),
        "lunar_limb_model": (lunar_limb_profile or LunarLimbProfile.circular()).to_dict(),
        "photometry": photometry_law.to_dict(),
        "photometry_lut": photometry_lut,
        "unit_km": UNIT_KM,
        "event_start_utc": states[0]["time"],
        "event_end_utc": states[-1]["time"],
        "event_duration_seconds": duration_s,
        "playback_seconds": playback,
        "requested_solver_states": requested,
        "solver_state_count": len(states),
        "ray_count": q.ray_count,
        "orbit_overlays": orbit_payloads,
        "contacts": _contact_payload(event),
        "labels": labels,
        "mesh": {
            "earth_lat": q.earth_lat,
            "earth_lon": q.earth_lon,
            "moon_lat": q.moon_lat,
            "moon_lon": q.moon_lon,
            "sun_lat": q.sun_lat,
            "sun_lon": q.sun_lon,
        },
        "radii": {
            "earth_equatorial": RE_KM,
            "earth_polar": RP_KM,
            "earth_shadow": LUNAR_DANJON_EARTH_RADIUS_KM,
            "moon": R_MOON_MEAN_KM,
            "sun": R_SUN_KM,
        },
        "states": states,
    }


def _page_html(payload: dict[str, object]) -> str:
    three_uri = _three_module_uri()
    runtime = RUNTIME_JS.read_text(encoding="utf-8")
    earth_uri, earth_asset = _resolved_image_uri("earth_albedo", max_width=4096, quality=88)
    moon_uri, moon_asset = _resolved_image_uri("moon_albedo", max_width=4096, quality=90)
    payload = dict(payload)
    _, lut_binary, lut_metadata = _resolve_photometry_lut()
    payload["data_assets"] = {
        "earth_albedo": earth_asset,
        "moon_albedo": moon_asset,
        "photometry_lut": lut_binary.to_dict(portable=True),
        "photometry_lut_metadata": lut_metadata.to_dict(portable=True),
    }
    photometry_lut_b64 = base64.b64encode(Path(lut_binary.path).read_bytes()).decode("ascii")
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    title = html.escape(str(payload["title"]))
    mode = str(payload["mode"])
    scope = str(payload["scope"])
    scope_label = html.escape(str(payload["scope_label"]))
    local_button = (
        '<button data-camera="observer">Observer sky</button>'
        if mode == "solar" and scope == "local" else ""
    )
    geometry_note = (
        "Moon blocks the finite solar disc; Earth irradiance is evaluated per fragment."
        if mode == "solar"
        else "Earth blocks the finite solar disc; lunar eclipse color is an explicitly illustrative atmospheric term."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — V22.2 physical eclipse simulation</title>
<style>
:root{{--bg:#01030a;--panel:rgba(6,13,25,.90);--line:#31445e;--text:#eef5ff;--muted:#9aabc0;--gold:#ffe36e}}
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
#viewport{{position:fixed;inset:0}}#gl{{width:100%;height:100%;display:block;touch-action:none}}
#top{{position:fixed;left:18px;right:18px;top:14px;display:flex;align-items:flex-start;justify-content:space-between;gap:18px;pointer-events:none}}
.title{{align-self:flex-start;padding:12px 15px;border:1px solid var(--line);border-radius:12px;background:linear-gradient(135deg,rgba(7,16,30,.94),rgba(3,8,18,.78));box-shadow:0 18px 60px rgba(0,0,0,.38);max-width:min(790px,72vw)}}
h1{{font-size:20px;line-height:1.15;margin:0 0 5px}}.subtitle{{font-size:12.5px;color:var(--muted);line-height:1.43}}
#status{{font-size:11px;color:#b7c8dd;margin-top:5px}}#status.error{{color:#ff8d8d}}#status.warn{{color:#ffd36d}}
.panel{{pointer-events:auto;width:330px;max-height:calc(100vh - 140px);overflow:auto;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel);backdrop-filter:blur(12px);box-shadow:0 18px 60px rgba(0,0,0,.42)}}
.section{{border-top:1px solid rgba(100,130,165,.25);padding-top:10px;margin-top:10px}}.section:first-child{{border-top:0;padding-top:0;margin-top:0}}
.section h2{{font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:#afc2d9;margin:0 0 8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}button{{background:#132239;color:var(--text);border:1px solid #425a78;border-radius:8px;padding:8px 7px;font-size:11px;cursor:pointer}}button:hover{{background:#1c3150;border-color:#6c8db4}}select{{background:#132239;color:var(--text);border:1px solid #425a78;border-radius:7px;padding:5px 7px;font-size:11px;max-width:155px}}
.row{{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0;font-size:11.5px}}input[type=range]{{width:155px;accent-color:#ffd85f}}input[type=checkbox]{{accent-color:#ffd85f}}
.metric{{font-size:11px;display:grid;grid-template-columns:115px 1fr;gap:5px 9px;margin:5px 0}}.metric span:nth-child(odd){{color:var(--muted)}}
#bottom{{position:fixed;left:18px;right:18px;bottom:14px;display:flex;gap:10px;align-items:center;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:rgba(4,10,20,.90);backdrop-filter:blur(12px)}}
#bottom button{{min-width:68px}}#timeline{{flex:1;min-width:120px}}#timeLabel{{font-variant-numeric:tabular-nums;font-size:12px;min-width:178px}}#phase{{font-size:12px;color:var(--gold);min-width:155px;text-align:right}}
.legend{{font-size:11px;color:var(--muted);line-height:1.48}}.sw{{display:inline-block;width:16px;height:3px;margin:0 5px 2px 0;vertical-align:middle;border-radius:3px;box-shadow:0 0 8px currentColor}}
@media(max-width:900px){{.panel{{width:265px}}.title{{max-width:55vw}}#timeLabel{{display:none}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style></head><body>
<div id="viewport"><canvas id="gl" aria-label="Interactive Sun Earth Moon eclipse simulation"></canvas></div>
<div id="top"><div class="title"><h1>{title}</h1><div class="subtitle">V22.2 SSAPy-Data integration release · {scope_label} — opaque Earth and Moon, exact physical bodies, quadratic limb-darkened finite-Sun irradiance, full-disc weighted photospheric sampling, physical UTC playback, body-following north-up cameras, immutable backend provenance, and first-opaque-surface clipping. {html.escape(geometry_note)}</div><div id="status">Loading embedded textures and WebGL renderer…</div></div>
<div class="panel">
<div class="section"><h2>Camera</h2><div class="grid"><button data-camera="system">Earth–Moon system</button><button data-camera="earth">Earth north-up</button><button data-camera="moon">Moon north-up</button><button data-camera="optics">Ray geometry side</button><button data-camera="true">True Sun distance</button><button data-camera="sun">Sun photosphere</button>{local_button}</div></div>
<div class="section"><h2>Light graphics</h2>
<label class="row"><span>Sampled photospheric light</span><input id="raysToggle" type="checkbox" checked></label>
<label class="row"><span>Surface photometry</span><select id="photometrySelect"><option value="limb" selected>Limb-darkened</option><option value="uniform">Uniform disc</option></select></label>
<label class="row"><span>Tangent diagnostics</span><input id="boundaryToggle" type="checkbox"></label>
<label class="row"><span>Atmospheric limb</span><input id="atmosphereToggle" type="checkbox" checked></label>
<label class="row"><span>Stars</span><input id="starToggle" type="checkbox" checked></label>
<label class="row"><span>Physical labels</span><input id="labelsToggle" type="checkbox" checked></label>
<label class="row"><span>Scientific axes</span><input id="scientificToggle" type="checkbox"></label>
<label class="row"><span>SSAPy reference orbit</span><input id="orbitToggle" type="checkbox" checked></label>
</div>
<div class="section"><h2>Current physical state</h2><div class="metric"><span>State backend</span><b id="metricBackend"></b><span>Frame backend</span><b id="metricFrame"></b><span>Event scope</span><b id="metricScope"></b><span>Earth–Moon</span><b id="metricDistance"></b><span>Photosphere samples</span><b id="metricRays"></b><span>Photometry</span><b id="metricPhotometry"></b><span>Blocked light weight</span><b id="metricBlocked"></b><span>Evaluation point</span><b id="metricTarget"></b><span>Photosphere visible</span><b id="metricVisibility"></b><span>Limb-darkened flux</span><b id="metricIrradiance"></b><span>Target coordinates</span><b id="metricLocation"></b><span>Observer Sun/Moon</span><b id="metricObserver"></b><span>Solid bodies</span><b>opaque · depth-writing</b></div></div>
<div class="section"><h2>Visual key</h2><div class="legend"><span class="sw" style="color:#fff2a8;background:#fff2a8"></span>white-gold: weighted photospheric samples reaching a solid surface<br><span class="sw" style="color:#ffad46;background:#ffad46"></span>amber: samples stopped at the eclipse occluder<br><span class="sw" style="color:#e47aff;background:#e47aff"></span>magenta: blocked-light axis, never transmitted light<br><span class="sw" style="color:#ff665c;background:#ff665c"></span>red: umbral/antumbral tangent diagnostic<br><span class="sw" style="color:#6fc8ff;background:#6fc8ff"></span>blue: penumbral tangent diagnostic<br><span class="sw" style="color:#5fffa2;background:#5fffa2"></span>green: SSAPy two-body geocentric reference orbit</div></div>
<div class="section"><h2>Simulation qualification</h2><div class="legend">Earth and Moon are fully opaque. The visible beams are quadrature samples distributed across the complete apparent solar disc; their weights sum to one. The body shader evaluates either geometric uniform-disc overlap or the limb-darkened flux model independently at every fragment, using an exact analytic overlap plus a high-resolution precomputed radiometric correction texture. Contact geometry remains uniform-disc and is not changed by the display photometry selection. Beam thickness, bloom, and flare size are explanatory display aids. The atmosphere, corona, and beams are the only translucent layers. A dedicated body-depth and body-mask pass prevents them from appearing through a solid body.</div></div>
</div></div>
<div id="bottom"><button id="play" aria-label="Play or pause">Play</button><input id="timeline" aria-label="Elapsed physical event time" type="range" min="0" value="0" step="1"><span id="timeLabel"></span><b id="phase"></b></div>
<script type="module">
import * as THREE from "{three_uri}";
const CONFIG={payload_json};
const EARTH_TEXTURE_URI="{earth_uri}";
const MOON_TEXTURE_URI="{moon_uri}";
const PHOTOMETRY_LUT_B64="{photometry_lut_b64}";
{runtime}
</script></body></html>"""


def generate_cinematic_view(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    output_path: str | Path = "eclipse_cinematic_v22_2.html",
    *,
    animate: bool = False,
    quality: str = "ultra",
    n_frames: int | None = None,
    backend: str = "auto",
    solar_scope: str = "global",
    playback_seconds: float | None = None,
    observer_config: ObserverConfig | None = None,
    lunar_limb_profile: LunarLimbProfile | None = None,
    photometry: str = "quadratic-visible",
    orbit_overlays: list[SSAPyOrbitOverlay | object] | None = None,
) -> str:
    payload = build_cinematic_payload(
        kind,
        animate=animate,
        quality=quality,
        n_frames=n_frames,
        backend=backend,
        solar_scope=solar_scope,
        playback_seconds=playback_seconds,
        observer_config=observer_config,
        lunar_limb_profile=lunar_limb_profile,
        photometry=photometry,
        orbit_overlays=orbit_overlays,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_page_html(payload), encoding="utf-8")
    return str(out)


def generate_cinematic_suite(
    output_dir: str | Path,
    *,
    quality: str = "ultra",
    peak_quality: str | None = None,
    backend: str = "auto",
    solar_frames: int = 121,
    lunar_frames: int = 91,
    include_solar_local: bool = True,
    solar_local_frames: int | None = None,
    reuse_existing: bool = False,
) -> dict[str, str]:
    """Generate the complete peak and contact-to-contact cinematic suite.

    ``reuse_existing`` is an explicit resumable-build option. A product is
    reused only when the exact V22.2 target filename already exists and is
    non-empty; otherwise it is regenerated from the requested state and
    quality settings. This makes interrupted release generation restartable
    without silently accepting a differently named legacy artifact.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    peak_quality = str(peak_quality or quality)

    def create(path: Path, *args, **kwargs) -> str:
        if reuse_existing and path.is_file() and path.stat().st_size > 0:
            return str(path)
        return generate_cinematic_view(*args, output_path=path, **kwargs)

    products = {
        "solar_peak": create(
            out / "solar_greatest_ssapy_data_v22_2.html", "solar",
            quality=peak_quality, backend=backend,
        ),
        "lunar_peak": create(
            out / "lunar_greatest_ssapy_data_v22_2.html", "lunar",
            quality=peak_quality, backend=backend,
        ),
        "solar_global_animation": create(
            out / "solar_P1_P4_ssapy_data_v22_2.html", "solar",
            animate=True, quality=quality, n_frames=solar_frames, backend=backend,
            solar_scope="global",
        ),
        "lunar_global_animation": create(
            out / "lunar_P1_P4_ssapy_data_v22_2.html", "lunar",
            animate=True, quality=quality, n_frames=lunar_frames, backend=backend,
        ),
    }
    if include_solar_local:
        products["solar_local_animation"] = create(
            out / "solar_C1_C4_ssapy_data_v22_2.html", "solar",
            animate=True, quality=quality,
            n_frames=(int(solar_local_frames) if solar_local_frames is not None
                      else max(49, solar_frames * 3 // 4)),
            backend=backend, solar_scope="local",
        )
    return products
