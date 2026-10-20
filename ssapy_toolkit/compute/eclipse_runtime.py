"""Optional LLNL SSAPy and SSAPy-Toolkit runtime integration.

This module is deliberately independent of the eclipse geometry modules.  It
owns only package discovery, UTC-to-GPS conversion, body ephemeris calls, and
coordinate transforms.  The separation keeps optional LLNL imports from
becoming mandatory for the reference renderer and makes backend provenance
explicit and testable.

SSAPy 1.1.6 ``Body.position`` callables accept GPS seconds and return GCRF
positions in metres.  SSAPy-Toolkit 1.0.4 coordinate helpers accept positions
in metres and times convertible to GPS seconds.  All public functions here
return kilometres and use UTC Julian Dates at the package boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import import_module, metadata
from typing import Callable, Literal

import numpy as np

from ssapy_toolkit.io.eclipse_asset_resolver import audit_assets
from ssapy_toolkit.io.eclipse_provenance import portable_record

BackendMode = Literal["reference", "auto", "ssapy", "ssapy-core"]


class BackendUnavailableError(RuntimeError):
    """Raised when a caller requests a strict backend that cannot execute."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    llnl_ssapy_importable: bool
    llnl_ssapy_version: str | None
    astropy_importable: bool
    astropy_version: str | None
    ssapy_ephemeris_healthy: bool
    ssapy_ephemeris_error: str | None
    ssapy_toolkit_importable: bool
    ssapy_toolkit_version: str | None
    toolkit_transform_healthy: bool
    toolkit_transform_error: str | None
    ssapy_orientation_healthy: bool = False
    ssapy_orientation_error: str | None = None
    ssapy_orientation_source: str | None = None
    ssapy_data_roots: tuple[str, ...] = ()
    ssapy_data_rendering_ready: bool = False
    ssapy_data_strict_ready: bool = False
    ssapy_data_error: str | None = None

    def to_dict(self, *, portable: bool = False) -> dict[str, object]:
        result = asdict(self)
        return portable_record(result) if portable else result


@dataclass(frozen=True)
class BackendSelection:
    requested: str
    selected: str
    ephemeris_backend: str
    frame_backend: str
    strict: bool
    fallback_reason: str | None
    time_input: str
    capabilities: RuntimeCapabilities

    def to_dict(self, *, portable: bool = False) -> dict[str, object]:
        result = asdict(self)
        result["capabilities"] = self.capabilities.to_dict(portable=portable)
        return portable_record(result) if portable else result


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _astropy_time_class():
    try:
        return import_module("astropy.time").Time
    except Exception:
        return None


@lru_cache(maxsize=1)
def _ssapy_get_body():
    try:
        module = import_module("ssapy.body")
        return module.get_body
    except Exception:
        try:
            module = import_module("ssapy")
            return module.get_body
        except Exception:
            return None


@lru_cache(maxsize=1)
def _ssapy_bodies():
    get_body = _ssapy_get_body()
    if get_body is None:
        raise BackendUnavailableError("llnl-ssapy is not importable")
    # Keep one kernel-backed body object per process.  This avoids repeatedly
    # opening the same DE430/PCK files when rendering many animation frames.
    return get_body("moon"), get_body("sun")


@lru_cache(maxsize=1)
def _ssapy_orientation_bodies():
    get_body = _ssapy_get_body()
    if get_body is None:
        raise BackendUnavailableError("llnl-ssapy is not importable")
    return get_body("earth"), get_body("moon")


def _as_rotation_matrix(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.shape != (3, 3):
        raise ValueError(f"Could not normalize orientation shape {arr.shape} to (3, 3)")
    u, _, vt = np.linalg.svd(arr)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def ssapy_body_to_gcrf_matrices(jd_utc, body: str = "moon") -> np.ndarray:
    """Return SSAPy body-fixed-to-GCRF rotation matrix/matrices.

    SSAPy's ``Body.orientation`` is the GCRF-to-body matrix used by its
    gravity/orientation models.  The public eclipse renderer needs the inverse,
    so each proper-orthogonal matrix is transposed after evaluation with
    numeric GPS seconds.  Moon attitude comes from SSAPy's DE440 binary PCK.
    """
    key = str(body).strip().lower()
    if key not in {"earth", "moon"}:
        raise ValueError("body must be 'earth' or 'moon'")
    jd = np.asarray(jd_utc, dtype=float)
    scalar = jd.ndim == 0
    flat = jd.reshape(-1)
    gps = jd_utc_to_gps_seconds(flat)
    earth_body, moon_body = _ssapy_orientation_bodies()
    orientation = earth_body.orientation if key == "earth" else moon_body.orientation
    matrices = []
    for value in gps:
        gcrf_to_body = _as_rotation_matrix(orientation(float(value)))
        matrices.append(_as_rotation_matrix(gcrf_to_body.T))
    result = np.stack(matrices, axis=0)
    return result[0] if scalar else result.reshape(jd.shape + (3, 3))


def ssapy_moon_orientation_body_to_gcrf(jd_utc) -> np.ndarray:
    """Compatibility name for the strict DE440 Moon attitude provider."""
    return ssapy_body_to_gcrf_matrices(jd_utc, body="moon")


@lru_cache(maxsize=1)
def _toolkit_functions() -> tuple[Callable, Callable | None, Callable] | None:
    roots = ("ssapy_toolkit.coordinates", "ssapy_toolkit.Coordinates")
    for root in roots:
        # Released wheels may re-export the helpers from the package root,
        # while source checkouts expose one module per transform.  Support
        # both layouts without importing the optional package at module load.
        try:
            package = import_module(root)

            def exported(name: str, *, optional: bool = False):
                value = getattr(package, name, None)
                if value is None and optional:
                    return None
                if callable(value):
                    return value
                nested = getattr(value, name, None)
                if callable(nested):
                    return nested
                if optional:
                    return None
                raise TypeError(f"{root}.{name} is not callable")

            return (
                exported("gcrf_to_itrf"),
                exported("gcrf_to_lonlat", optional=True),
                exported("itrf_to_gcrf"),
            )
        except Exception:
            pass
        try:
            g2i_mod = import_module(f"{root}.gcrf_to_itrf")
            i2g_mod = import_module(f"{root}.itrf_to_gcrf")
            try:
                g2l_mod = import_module(f"{root}.gcrf_to_lonlat")
                g2l = g2l_mod.gcrf_to_lonlat
            except Exception:
                g2l = None
            return g2i_mod.gcrf_to_itrf, g2l, i2g_mod.itrf_to_gcrf
        except Exception:
            continue
    return None


def clear_runtime_caches() -> None:
    """Clear backend caches; useful after installing packages in-process/tests."""
    _astropy_time_class.cache_clear()
    _ssapy_get_body.cache_clear()
    _ssapy_bodies.cache_clear()
    _ssapy_orientation_bodies.cache_clear()
    _toolkit_functions.cache_clear()
    runtime_capabilities.cache_clear()


def jd_utc_to_gps_seconds(jd_utc) -> np.ndarray:
    """Convert UTC Julian Dates to SSAPy's GPS-second convention.

    Astropy performs the UTC/GPS leap-second conversion.  A plain
    ``(jd - epoch) * 86400`` expression would be wrong by the accumulated leap
    seconds and is therefore intentionally not used in the LLNL backend.
    """
    Time = _astropy_time_class()
    if Time is None:
        raise BackendUnavailableError(
            "Astropy is required to convert UTC Julian Dates to SSAPy GPS seconds"
        )
    jd = np.asarray(jd_utc, dtype=float)
    time = Time(jd, format="jd", scale="utc")
    return np.asarray(time.gps, dtype=float)


def _as_n_by_3(value, n_expected: int, rows_are_points: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    # (n, 3) and (3, n) are indistinguishable when n == 3. Callers whose
    # source returns one point per row say so; everything else keeps the
    # historical (3, n) preference.
    if rows_are_points and arr.shape == (n_expected, 3):
        return arr
    if arr.shape == (3, n_expected):
        return arr.T
    if arr.shape == (n_expected, 3):
        return arr
    if arr.size == n_expected * 3:
        return arr.reshape(n_expected, 3)
    raise ValueError(
        f"Could not normalize LLNL ephemeris/transform shape {arr.shape} to ({n_expected}, 3)"
    )


def ssapy_ephemeris_positions_km(jd_utc) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(moon_gcrf_km, sun_gcrf_km)`` from LLNL SSAPy.

    SSAPy body positions are evaluated with numeric GPS seconds and converted
    from metres to kilometres exactly once.
    """
    jd = np.asarray(jd_utc, dtype=float).reshape(-1)
    gps = jd_utc_to_gps_seconds(jd)
    moon_body, sun_body = _ssapy_bodies()
    moon = _as_n_by_3(moon_body.position(gps), len(jd)) / 1_000.0
    sun = _as_n_by_3(sun_body.position(gps), len(jd)) / 1_000.0
    if not np.all(np.isfinite(moon)) or not np.all(np.isfinite(sun)):
        raise RuntimeError("SSAPy returned non-finite Sun or Moon positions")
    return moon, sun





def toolkit_gcrf_to_itrf_km(points_gcrf_km, jd_utc) -> np.ndarray:
    functions = _toolkit_functions()
    if functions is None:
        raise BackendUnavailableError("ssapy-toolkit coordinate transforms are unavailable")
    gcrf_to_itrf, _, _ = functions
    points = np.asarray(points_gcrf_km, dtype=float).reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(points), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(points),))
    gps = jd_utc_to_gps_seconds(jd)
    result = gcrf_to_itrf(points * 1_000.0, gps)
    return _as_n_by_3(result, len(points), rows_are_points=True) / 1_000.0


def toolkit_itrf_to_gcrf_km(points_itrf_km, jd_utc) -> np.ndarray:
    functions = _toolkit_functions()
    if functions is None:
        raise BackendUnavailableError("ssapy-toolkit coordinate transforms are unavailable")
    _, _, itrf_to_gcrf = functions
    points = np.asarray(points_itrf_km, dtype=float).reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(points), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(points),))
    gps = jd_utc_to_gps_seconds(jd)
    result = itrf_to_gcrf(points * 1_000.0, gps)
    return _as_n_by_3(result, len(points), rows_are_points=True) / 1_000.0


def astropy_gcrf_to_itrf_km(points_gcrf_km, jd_utc) -> np.ndarray:
    """Strict Astropy GCRS->ITRS transform in kilometres.

    This function never falls back to GMST or SSAPy-Toolkit.  It is used by
    the immutable ``ssapy-core`` backend so renderer behavior cannot change
    merely because another optional package is present.
    """
    Time = _astropy_time_class()
    if Time is None:
        raise BackendUnavailableError("Astropy GCRS/ITRS transforms are unavailable")
    try:
        import astropy.units as u
        from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
    except Exception as exc:  # pragma: no cover - import contract guarded above
        raise BackendUnavailableError(f"Astropy coordinate transforms are unavailable: {exc}") from exc
    points = np.asarray(points_gcrf_km, dtype=float)
    scalar = points.ndim == 1
    values = points.reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(values), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(values),))
    time = Time(jd, format="jd", scale="utc")
    rep = CartesianRepresentation(values[:, 0]*u.km, values[:, 1]*u.km, values[:, 2]*u.km)
    coord = GCRS(rep, obstime=time).transform_to(ITRS(obstime=time))
    result = np.asarray(coord.cartesian.xyz.to_value(u.km), dtype=float).T
    return result[0] if scalar else result


def astropy_itrf_to_gcrf_km(points_itrf_km, jd_utc) -> np.ndarray:
    """Strict Astropy ITRS->GCRS transform in kilometres."""
    Time = _astropy_time_class()
    if Time is None:
        raise BackendUnavailableError("Astropy GCRS/ITRS transforms are unavailable")
    try:
        import astropy.units as u
        from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(f"Astropy coordinate transforms are unavailable: {exc}") from exc
    points = np.asarray(points_itrf_km, dtype=float)
    scalar = points.ndim == 1
    values = points.reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(values), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(values),))
    time = Time(jd, format="jd", scale="utc")
    rep = CartesianRepresentation(values[:, 0]*u.km, values[:, 1]*u.km, values[:, 2]*u.km)
    coord = ITRS(rep, obstime=time).transform_to(GCRS(obstime=time))
    result = np.asarray(coord.cartesian.xyz.to_value(u.km), dtype=float).T
    return result[0] if scalar else result


def toolkit_gcrf_to_lonlat_km(points_gcrf_km, jd_utc):
    functions = _toolkit_functions()
    if functions is None or functions[1] is None:
        raise BackendUnavailableError("ssapy-toolkit gcrf_to_lonlat is unavailable")
    _, gcrf_to_lonlat, _ = functions
    points = np.asarray(points_gcrf_km, dtype=float).reshape(-1, 3)
    jd = np.asarray(jd_utc, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(points), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(points),))
    gps = jd_utc_to_gps_seconds(jd)
    lon, lat, height_m = gcrf_to_lonlat(points * 1_000.0, gps)
    return (
        np.asarray(lon, dtype=float).reshape(-1),
        np.asarray(lat, dtype=float).reshape(-1),
        np.asarray(height_m, dtype=float).reshape(-1) / 1_000.0,
    )


@lru_cache(maxsize=2)
def runtime_capabilities(execute: bool = True) -> RuntimeCapabilities:
    data_error: str | None = None
    try:
        data_audit = audit_assets(policy="data-first")
        data_roots = tuple(str(value) for value in data_audit.get("data_roots", []))
        data_rendering_ready = bool(data_audit.get("rendering_ready", False))
        data_strict_ready = bool(data_audit.get("strict_ssapy_data_ready", False))
        if not data_strict_ready:
            missing = [
                name for name, record in data_audit.get("assets", {}).items()
                if record.get("required_for_strict_ssapy") and record.get("resolved") is None
            ]
            if missing:
                data_error = "missing required SSAPy-Data assets: " + ", ".join(missing)
        # activate_ssapy_data_root is deliberately NOT called here.
        # runtime_capabilities() is a probe, and that function replaces
        # ssapy.utils.datadir with the SSAPy-Data path for the rest of the
        # process. SSAPy's find_file searches only the cwd and datadir, and
        # SSAPy-Data carries no egm* gravity models, so a single probe left
        # EGM2008 unresolvable and broke unrelated SSAPy force models.
        # Callers that need the redirect should request it explicitly.
    except Exception as exc:
        data_audit = {}
        data_roots = ()
        data_rendering_ready = False
        data_strict_ready = False
        data_error = f"{type(exc).__name__}: {exc}"
    astropy_ok = _astropy_time_class() is not None
    ssapy_importable = _ssapy_get_body() is not None
    toolkit_importable = _toolkit_functions() is not None
    ephem_healthy = False
    ephem_error: str | None = None
    transform_healthy = False
    transform_error: str | None = None
    orientation_healthy = False
    orientation_error: str | None = None
    orientation_source: str | None = None

    if execute and ssapy_importable and astropy_ok:
        try:
            moon, sun = ssapy_ephemeris_positions_km(np.asarray([2_460_409.0]))
            if moon.shape != (1, 3) or sun.shape != (1, 3):
                raise RuntimeError("unexpected SSAPy ephemeris shape")
            if not (300_000.0 < np.linalg.norm(moon[0]) < 500_000.0):
                raise RuntimeError("SSAPy Moon distance failed sanity check")
            if not (1.3e8 < np.linalg.norm(sun[0]) < 1.7e8):
                raise RuntimeError("SSAPy Sun distance failed sanity check")
            ephem_healthy = True
        except Exception as exc:
            ephem_error = f"{type(exc).__name__}: {exc}"
    elif ssapy_importable and astropy_ok:
        ephem_healthy = True
    else:
        missing = []
        if not ssapy_importable:
            missing.append("llnl-ssapy")
        if not astropy_ok:
            missing.append("astropy")
        ephem_error = "missing " + ", ".join(missing)


    if execute and toolkit_importable and astropy_ok:
        try:
            sample = np.asarray([
                [6_378.137, 0.0, 0.0],
                [0.0, 6_378.137, 0.0],
                [0.0, 0.0, 6_356.752314245],
            ])
            jd = np.full(3, 2_460_409.0)
            gcrf = toolkit_itrf_to_gcrf_km(sample, jd)
            recovered = toolkit_gcrf_to_itrf_km(gcrf, jd)
            error = float(np.max(np.linalg.norm(recovered - sample, axis=1)))
            if not np.isfinite(error) or error > 1.0e-3:
                raise RuntimeError(f"Toolkit round-trip error {error:.6g} km")
            transform_healthy = True
        except Exception as exc:
            transform_error = f"{type(exc).__name__}: {exc}"
    elif toolkit_importable and astropy_ok:
        transform_healthy = True
    else:
        missing = []
        if not toolkit_importable:
            missing.append("ssapy-toolkit")
        if not astropy_ok:
            missing.append("astropy")
        transform_error = "missing " + ", ".join(missing)

    if execute and ssapy_importable and astropy_ok:
        try:
            moon_orientation = ssapy_body_to_gcrf_matrices(2_460_409.0, "moon")
            earth_orientation = ssapy_body_to_gcrf_matrices(2_460_409.0, "earth")
            for name, matrix in (("Moon", moon_orientation), ("Earth", earth_orientation)):
                orth_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
                determinant = float(np.linalg.det(matrix))
                if orth_error > 1.0e-10 or determinant < 0.999999999:
                    raise RuntimeError(
                        f"{name} orientation is not a proper rotation: "
                        f"orth_error={orth_error:.3g}, det={determinant:.12g}"
                    )
            orientation_healthy = True
            orientation_source = "llnl-ssapy EarthOrientation + DE440 lunar binary PCK"
        except Exception as exc:
            orientation_error = f"{type(exc).__name__}: {exc}"
    elif ssapy_importable and astropy_ok:
        orientation_healthy = True
        orientation_source = "llnl-ssapy orientation API (execution probe disabled)"
    else:
        orientation_error = ephem_error

    return RuntimeCapabilities(
        llnl_ssapy_importable=ssapy_importable,
        llnl_ssapy_version=_version("llnl-ssapy"),
        astropy_importable=astropy_ok,
        astropy_version=_version("astropy"),
        ssapy_ephemeris_healthy=ephem_healthy,
        ssapy_ephemeris_error=ephem_error,
        ssapy_toolkit_importable=toolkit_importable,
        ssapy_toolkit_version=_version("ssapy-toolkit"),
        toolkit_transform_healthy=transform_healthy,
        toolkit_transform_error=transform_error,
        ssapy_orientation_healthy=orientation_healthy,
        ssapy_orientation_error=orientation_error,
        ssapy_orientation_source=orientation_source,
        ssapy_data_roots=data_roots,
        ssapy_data_rendering_ready=data_rendering_ready,
        ssapy_data_strict_ready=data_strict_ready,
        ssapy_data_error=data_error,
    )


def resolve_backend(mode: BackendMode | str = "auto", *, execute_probe: bool = True) -> BackendSelection:
    requested = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "nasa": "reference",
        "validated": "reference",
        "llnl": "ssapy",
        "strict-ssapy": "ssapy",
        "core": "ssapy-core",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"reference", "auto", "ssapy", "ssapy-core"}:
        raise ValueError("backend must be 'reference', 'auto', 'ssapy', or 'ssapy-core'")

    caps = runtime_capabilities(execute=execute_probe and requested != "reference")
    fallback_reason: str | None = None

    if requested == "reference":
        selected = "reference"
    elif requested == "ssapy":
        if not caps.ssapy_data_strict_ready:
            raise BackendUnavailableError(
                "backend='ssapy' requires real de430.bsp and moon_pa_de440_200625.bpc "
                "from llnl-ssapy-data or an SSAPy-Data checkout; Git LFS pointers are rejected. "
                f"{caps.ssapy_data_error or ''}"
            )
        if not caps.ssapy_ephemeris_healthy:
            raise BackendUnavailableError(
                "backend='ssapy' requires a healthy LLNL SSAPy ephemeris: "
                f"{caps.ssapy_ephemeris_error}"
            )
        if not caps.toolkit_transform_healthy:
            raise BackendUnavailableError(
                "backend='ssapy' requires healthy SSAPy-Toolkit transforms: "
                f"{caps.toolkit_transform_error}"
            )
        if not caps.ssapy_orientation_healthy:
            raise BackendUnavailableError(
                "backend='ssapy' requires healthy SSAPy Earth/Moon orientation: "
                f"{caps.ssapy_orientation_error}"
            )
        selected = "ssapy"
    elif requested == "ssapy-core":
        if not caps.ssapy_data_strict_ready:
            raise BackendUnavailableError(
                "backend='ssapy-core' requires real de430.bsp and moon_pa_de440_200625.bpc "
                "from llnl-ssapy-data or an SSAPy-Data checkout; Git LFS pointers are rejected. "
                f"{caps.ssapy_data_error or ''}"
            )
        if not caps.ssapy_ephemeris_healthy:
            raise BackendUnavailableError(
                "backend='ssapy-core' requires a healthy LLNL SSAPy ephemeris: "
                f"{caps.ssapy_ephemeris_error}"
            )
        if not caps.ssapy_orientation_healthy:
            raise BackendUnavailableError(
                "backend='ssapy-core' requires healthy SSAPy Earth/Moon orientation: "
                f"{caps.ssapy_orientation_error}"
            )
        selected = "ssapy-core"
    else:
        if caps.ssapy_data_strict_ready and caps.ssapy_ephemeris_healthy and caps.ssapy_orientation_healthy and caps.toolkit_transform_healthy:
            selected = "ssapy"
        elif caps.ssapy_data_strict_ready and caps.ssapy_ephemeris_healthy and caps.ssapy_orientation_healthy:
            selected = "ssapy-core"
            fallback_reason = (
                "SSAPy ephemerides are healthy, but SSAPy-Toolkit transforms are not; "
                "using Astropy frame transforms"
            )
        else:
            selected = "reference"
            fallback_reason = (
                "LLNL SSAPy could not execute; using the validated NASA/GSFC reference state"
            )

    if selected == "ssapy":
        ephem = "llnl-ssapy/DE430"
        frame = "ssapy-toolkit GCRF<->ITRF"
    elif selected == "ssapy-core":
        ephem = "llnl-ssapy/DE430"
        frame = "Astropy GCRS<->ITRS"
    else:
        ephem = "NASA/GSFC validated reference reconstruction"
        frame = "reference event frame with deterministic WGS-84 transforms"

    time_input = (
        "UTC Julian Date -> Astropy Time.gps -> numeric GPS seconds"
        if selected in {"ssapy", "ssapy-core"}
        else "UTC Julian Date on the validated NASA/GSFC contact scaffold"
    )
    return BackendSelection(
        requested=requested,
        selected=selected,
        ephemeris_backend=ephem,
        frame_backend=frame,
        strict=requested in {"ssapy", "ssapy-core"},
        fallback_reason=fallback_reason,
        time_input=time_input,
        capabilities=caps,
    )