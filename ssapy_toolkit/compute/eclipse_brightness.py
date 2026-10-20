"""Shared eclipse ephemerides, geometry, photometry, and ground-track tools.

The public functions from the original module remain available, but the
implementation now separates *physical geometry* from *display scaling* and
uses finite Sun distances throughout.  When LLNL SSAPy is installed, the
Moon and Sun are read from SSAPy's DE430-backed body ephemerides.  When
``ssapy-toolkit`` is installed, GCRF shadow points are converted to Earth-fixed
longitude/latitude with its coordinate utilities.  A deterministic analytical
fallback is included so the plotting modules remain runnable in lightweight
environments.

All internal distances are kilometres unless a function explicitly says
otherwise.  Times used by the new API are Julian Dates (UTC-like for plotting;
SSAPy/Astropy performs its own time-scale handling when available).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterable, Literal, NamedTuple
import warnings

import numpy as np

from ssapy_toolkit.compute.eclipse_photometry import (
    LimbDarkeningLaw,
    apparent_disk_radiometry,
    limb_darkened_visibility_fraction,
    resolve_limb_darkening,
)
from ssapy_toolkit.compute.eclipse_core import (
    CORE_PROVIDER as ECLIPSE_CORE_PROVIDER,
    ApparentDiskGeometry as CoreApparentDiskGeometry,
    apparent_disk_geometry as _core_apparent_disk_geometry,
    circle_overlap_visible_fraction as _core_circle_overlap_visible_fraction,
    finite_source_irradiance as _core_finite_source_irradiance,
    finite_source_visibility as _core_finite_source_visibility,
    ray_ellipsoid_intersections as _core_ray_ellipsoid_intersections,
    ray_sphere_intersections as _core_ray_sphere_intersections,
)

MU_EARTH_KM3S2 = 398_600.4418
RE_KM = 6_378.137
RP_EARTH_KM = 6_356.752314245
EARTH_AXES_KM = np.array([RE_KM, RE_KM, RP_EARTH_KM], dtype=float)
R_MOON_KM = 1_737.4
R_SUN_KM = 695_700.0
AU_KM = 149_597_870.7
DEFAULT_EPOCH_JD = 2_460_310.5  # 2024-01-01 00:00 UTC


# ---------------------------------------------------------------------------
# Optional high-fidelity backends
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _ssapy_backend():
    """Return ``(Time, get_body)`` or ``None`` without making imports fatal."""
    try:
        from astropy.time import Time  # type: ignore
        from ssapy import get_body  # type: ignore
        return Time, get_body
    except Exception:
        return None


@lru_cache(maxsize=1)
def _ssapy_bodies():
    """Cache the kernel-backed Moon and Sun body objects per process."""
    backend = _ssapy_backend()
    if backend is None:
        return None
    _, get_body = backend
    return get_body("moon"), get_body("sun")


@lru_cache(maxsize=1)
def _toolkit_backend():
    """Return Toolkit coordinate callables across released API layouts.

    The current LLNL package uses ``ssapy_toolkit.coordinates``.  Older
    source checkouts exposed the same modules under a capitalized
    ``Coordinates`` directory, so both are accepted without making the
    optional dependency fatal.
    """
    module_roots = ("ssapy_toolkit.coordinates", "ssapy_toolkit.Coordinates")
    for root in module_roots:
        try:
            module = __import__(root, fromlist=[
                "gcrf_to_itrf", "gcrf_to_lonlat", "itrf_to_gcrf",
            ])
            return (module.gcrf_to_itrf, module.gcrf_to_lonlat,
                    module.itrf_to_gcrf)
        except Exception:
            pass
        try:
            g2i = __import__(f"{root}.gcrf_to_itrf", fromlist=["gcrf_to_itrf"]).gcrf_to_itrf
            g2l = __import__(f"{root}.gcrf_to_lonlat", fromlist=["gcrf_to_lonlat"]).gcrf_to_lonlat
            i2g = __import__(f"{root}.itrf_to_gcrf", fromlist=["itrf_to_gcrf"]).itrf_to_gcrf
            return g2i, g2l, i2g
        except Exception:
            pass
    return None


def available_backends() -> dict[str, bool]:
    """Report which optional packages can be used in the current runtime."""
    return {
        "llnl_ssapy": _ssapy_backend() is not None,
        "ssapy_toolkit": _toolkit_backend() is not None,
    }


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def datetime_to_jd(value: datetime | str | np.datetime64 | float) -> float:
    """Convert a UTC datetime/ISO string to Julian Date; floats pass through."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.datetime64):
        seconds = (value - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return 2_440_587.5 + float(seconds) / 86_400.0
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return 2_440_587.5 + value.timestamp() / 86_400.0


def jd_to_datetime(jd: float | np.ndarray) -> datetime | np.ndarray:
    """Convert Julian Date(s) to timezone-aware UTC ``datetime`` objects."""
    arr = np.asarray(jd, dtype=float)

    def _one(x: float) -> datetime:
        return datetime.fromtimestamp((x - 2_440_587.5) * 86_400.0, tz=timezone.utc)

    if arr.ndim == 0:
        return _one(float(arr))
    return np.array([_one(float(x)) for x in arr.ravel()], dtype=object).reshape(arr.shape)


def _as_jd_array(times: float | datetime | str | np.datetime64 | Iterable) -> np.ndarray:
    if isinstance(times, (float, int, np.floating, np.integer, datetime, str, np.datetime64)):
        return np.asarray([datetime_to_jd(times)], dtype=float)
    values = list(times)
    return np.asarray([datetime_to_jd(v) for v in values], dtype=float)


# ---------------------------------------------------------------------------
# Ephemerides
# ---------------------------------------------------------------------------


def _as_n_by_3(value: np.ndarray, n_expected: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    if arr.shape == (3, n_expected):
        return arr.T
    if arr.shape == (n_expected, 3):
        return arr
    if arr.size == n_expected * 3:
        return arr.reshape(n_expected, 3)
    raise ValueError(f"Could not normalize ephemeris shape {arr.shape} to ({n_expected}, 3)")


def _analytic_sun_position_gcrf(jd: np.ndarray) -> np.ndarray:
    """Compact solar ephemeris fallback, including ecliptic-to-GCRF tilt."""
    d = np.asarray(jd, dtype=float) - 2_451_545.0
    L = np.radians((280.460 + 0.9856474 * d) % 360.0)
    g = np.radians((357.528 + 0.9856003 * d) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2.0 * g)
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    eps = np.radians(23.439291 - 0.0000004 * d)
    x = r_au * np.cos(lam)
    y = r_au * np.cos(eps) * np.sin(lam)
    z = r_au * np.sin(eps) * np.sin(lam)
    return AU_KM * np.stack([x, y, z], axis=-1)


def _analytic_moon_position_gcrf(jd: np.ndarray) -> np.ndarray:
    """Truncated analytical lunar ephemeris used only when SSAPy is absent.

    This includes the dominant longitude, latitude, and distance terms.  It is
    materially better for eclipse searching than a fixed Keplerian plane
    because the node, inclination, and principal solar perturbations evolve
    with epoch, but it is intentionally not presented as a replacement for
    SSAPy's DE430 ephemeris.
    """
    d = np.asarray(jd, dtype=float) - 2_451_545.0
    L0 = np.radians((218.3164477 + 13.17639648 * d) % 360.0)
    Mm = np.radians((134.9633964 + 13.06499295 * d) % 360.0)
    Ms = np.radians((357.5291092 + 0.98560028 * d) % 360.0)
    D = np.radians((297.8501921 + 12.19074912 * d) % 360.0)
    F = np.radians((93.2720950 + 13.22935024 * d) % 360.0)

    lon = L0 + np.radians(
        6.289 * np.sin(Mm)
        + 1.274 * np.sin(2 * D - Mm)
        + 0.658 * np.sin(2 * D)
        + 0.214 * np.sin(2 * Mm)
        - 0.186 * np.sin(Ms)
        - 0.114 * np.sin(2 * F)
        + 0.059 * np.sin(2 * D - 2 * Mm)
        + 0.057 * np.sin(2 * D - Ms - Mm)
        + 0.053 * np.sin(2 * D + Mm)
        + 0.046 * np.sin(2 * D - Ms)
        + 0.041 * np.sin(Ms - Mm)
        - 0.035 * np.sin(D)
        - 0.031 * np.sin(Ms + Mm)
        - 0.015 * np.sin(2 * F - 2 * D)
        + 0.011 * np.sin(Mm - 4 * D)
    )
    lat = np.radians(
        5.128 * np.sin(F)
        + 0.280 * np.sin(Mm + F)
        + 0.277 * np.sin(Mm - F)
        + 0.173 * np.sin(2 * D - F)
        + 0.055 * np.sin(2 * D + F - Mm)
        + 0.046 * np.sin(2 * D - F - Mm)
        + 0.033 * np.sin(2 * D + F)
        + 0.017 * np.sin(2 * Mm + F)
        + 0.009 * np.sin(2 * D + Mm - F)
        + 0.009 * np.sin(2 * D - Mm - F)
        + 0.008 * np.sin(2 * D - Ms - F)
    )
    dist = (
        385_000.56
        - 20_905.0 * np.cos(Mm)
        - 3_699.0 * np.cos(2 * D - Mm)
        - 2_956.0 * np.cos(2 * D)
        - 570.0 * np.cos(2 * Mm)
        + 246.0 * np.cos(2 * Mm - 2 * D)
        - 205.0 * np.cos(Ms - 2 * D)
        - 171.0 * np.cos(Mm + 2 * D)
        - 152.0 * np.cos(Mm + Ms - 2 * D)
    )

    cosb = np.cos(lat)
    xe = dist * cosb * np.cos(lon)
    ye = dist * cosb * np.sin(lon)
    ze = dist * np.sin(lat)
    eps = np.radians(23.439291 - 0.0000004 * d)
    x = xe
    y = ye * np.cos(eps) - ze * np.sin(eps)
    z = ye * np.sin(eps) + ze * np.cos(eps)
    return np.stack([x, y, z], axis=-1)


@dataclass(frozen=True)
class EphemerisResult:
    jd: np.ndarray
    moon_gcrf_km: np.ndarray
    sun_gcrf_km: np.ndarray
    backend: str


def ephemeris_positions(
    times: float | datetime | str | np.datetime64 | Iterable,
    backend: Literal["auto", "ssapy", "analytic"] = "auto",
) -> EphemerisResult:
    """Return geocentric GCRF Moon and Sun positions.

    ``backend='auto'`` prefers LLNL SSAPy and falls back to the analytical
    model.  SSAPy positions are documented in metres, so they are converted to
    kilometres here exactly once.
    """
    jd = _as_jd_array(times)
    if backend in ("auto", "ssapy"):
        try:
            # One authoritative LLNL adapter owns UTC->GPS conversion, body
            # caching, shape normalization, and the metres->kilometres unit
            # boundary.  Keeping that logic in one place prevents the public
            # renderer and audit tool from silently using different APIs.
            from ssapy_toolkit.compute.eclipse_runtime import ssapy_ephemeris_positions_km
            moon, sun = ssapy_ephemeris_positions_km(jd)
            return EphemerisResult(
                jd=jd, moon_gcrf_km=moon, sun_gcrf_km=sun,
                backend="llnl-ssapy/DE430",
            )
        except Exception as exc:
            if backend == "ssapy":
                raise ImportError(
                    "backend='ssapy' requested, but the LLNL SSAPy ephemeris "
                    f"could not execute: {exc}"
                ) from exc

    moon = _analytic_moon_position_gcrf(jd)
    sun = _analytic_sun_position_gcrf(jd)
    return EphemerisResult(jd=jd, moon_gcrf_km=moon, sun_gcrf_km=sun, backend="analytic fallback")


# ---------------------------------------------------------------------------
# Legacy propagation API (kept for downstream compatibility)
# ---------------------------------------------------------------------------


def propagate_eci(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg,
                  n_orbits=1.0, n_steps=1500):
    """Vectorized two-body propagation retained for non-lunar orbit plots."""
    def _solve_kepler(M, ecc, tol=1e-12, max_iter=80):
        E = np.asarray(M, dtype=float).copy()
        for _ in range(max_iter):
            dE = (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
            E -= dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    inc, raan, argp = np.radians([inc_deg, raan_deg, argp_deg])
    nu0 = np.radians(nu0_deg)
    E0 = 2 * np.arctan2(np.sqrt(1-e) * np.sin(nu0/2), np.sqrt(1+e) * np.cos(nu0/2))
    M0 = E0 - e * np.sin(E0)
    T_s = 2 * np.pi * np.sqrt(a_km**3 / MU_EARTH_KM3S2)
    t_s = np.linspace(0, n_orbits * T_s, int(n_steps))
    n_rad_s = np.sqrt(MU_EARTH_KM3S2 / a_km**3)
    E = _solve_kepler(M0 + n_rad_s * t_s, e)
    nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e) * np.cos(E/2))
    r_mag = a_km * (1 - e * np.cos(E))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    R11 = cO*cw - sO*sw*ci; R12 = -cO*sw - sO*cw*ci
    R21 = sO*cw + cO*sw*ci; R22 = -sO*sw + cO*cw*ci
    R31 = sw*si;             R32 = cw*si
    xp, yp = r_mag * np.cos(nu), r_mag * np.sin(nu)
    x = R11*xp + R12*yp; y = R21*xp + R22*yp; z = R31*xp + R32*yp
    return t_s, np.stack([x, y, z], axis=1), T_s


def sun_position_eci(t_s, epoch_jd=2_460_500.0, backend="auto"):
    """Geocentric GCRF Sun position for seconds since ``epoch_jd``."""
    jd = np.asarray(epoch_jd + np.asarray(t_s, dtype=float) / 86_400.0)
    return ephemeris_positions(jd.ravel(), backend=backend).sun_gcrf_km.reshape(jd.shape + (3,))


def sun_direction_eci(t_s, epoch_jd=2_460_500.0, backend="auto"):
    """Unit vector toward the Sun, now with the correct equatorial Z term."""
    p = sun_position_eci(t_s, epoch_jd=epoch_jd, backend=backend)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Apparent-disc overlap photometry
# ---------------------------------------------------------------------------


def _circle_overlap_fraction(r1, r2, d):
    """Compatibility wrapper around :func:`ssapy_toolkit.eclipse.circle_overlap_visible_fraction`."""
    return _core_circle_overlap_visible_fraction(r1, r2, d)


@dataclass(frozen=True)
class ApparentDiskGeometry:
    occluder_angular_radius_rad: np.ndarray
    sun_angular_radius_rad: np.ndarray
    separation_rad: np.ndarray
    occluder_distance_km: np.ndarray
    sun_distance_km: np.ndarray


def apparent_disk_geometry(
    r_eval_km,
    sun_position_from_occluder_km,
    R_body_km,
    R_sun_km=R_SUN_KM,
) -> ApparentDiskGeometry:
    """Exact finite-distance apparent-disc geometry via the core SSAPy kernel."""
    geom: CoreApparentDiskGeometry = _core_apparent_disk_geometry(
        r_eval_km, sun_position_from_occluder_km, R_body_km, R_sun_km
    )
    return ApparentDiskGeometry(
        geom.occluder_angular_radius_rad,
        geom.source_angular_radius_rad,
        geom.separation_rad,
        geom.occluder_distance,
        geom.source_distance,
    )


def _resolve_sun_position(sun_hat, D_km, sun_position_km):
    if sun_position_km is not None:
        return np.asarray(sun_position_km, dtype=float)
    if sun_hat is None:
        raise TypeError("Provide sun_hat or sun_position_km")
    sh = np.asarray(sun_hat, dtype=float)
    norm = np.linalg.norm(sh, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("sun_hat must be nonzero")
    # Preserve the historical unit-vector signature while accepting a full
    # source-position vector as a convenience.
    scale = np.asarray(D_km, dtype=float)[..., None]
    return sh / norm * scale if np.nanmedian(norm) < 10.0 else sh


def illumination_fraction(
    r_eval_km,
    sun_hat=None,
    R_body_km=None,
    R_sun_km=R_SUN_KM,
    D_km=AU_KM,
    *,
    sun_position_km=None,
    return_geometry=False,
):
    """Uniform-disc visible-Sun fraction from the canonical SSAPy kernel."""
    if R_body_km is None:
        raise TypeError("R_body_km is required")
    source = _resolve_sun_position(sun_hat, D_km, sun_position_km)
    geom = apparent_disk_geometry(r_eval_km, source, R_body_km, R_sun_km)
    illum = _core_finite_source_visibility(r_eval_km, source, R_body_km, R_sun_km)
    return (illum, geom) if return_geometry else illum


def irradiance_fraction(
    r_eval_km,
    sun_hat=None,
    R_body_km=None,
    R_sun_km=R_SUN_KM,
    D_km=AU_KM,
    *,
    sun_position_km=None,
    photometry: str | LimbDarkeningLaw = "quadratic-visible",
    quadrature_order: int = 48,
    return_geometry: bool = False,
):
    """Limb-darkened visible solar-flux fraction from the SSAPy kernel."""
    if R_body_km is None:
        raise TypeError("R_body_km is required")
    source = _resolve_sun_position(sun_hat, D_km, sun_position_km)
    geom = apparent_disk_geometry(r_eval_km, source, R_body_km, R_sun_km)
    flux = _core_finite_source_irradiance(
        r_eval_km, source, R_body_km, R_sun_km,
        law=resolve_limb_darkening(photometry),
        quadrature_order=quadrature_order,
    )
    return (flux, geom) if return_geometry else flux


# ---------------------------------------------------------------------------
# Umbra, antumbra, penumbra, and target intersections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowCone:
    axis_hat: np.ndarray
    sun_distance_km: float
    umbra_half_angle_rad: float
    penumbra_half_angle_rad: float
    umbra_apex_distance_km: float
    penumbra_vertex_distance_km: float

    def radii_at(self, axial_distance_km):
        x = np.asarray(axial_distance_km, dtype=float)
        signed_umbra = (self.umbra_apex_distance_km - x) * np.tan(self.umbra_half_angle_rad)
        penumbra = (self.penumbra_vertex_distance_km + x) * np.tan(self.penumbra_half_angle_rad)
        return signed_umbra, np.maximum(penumbra, 0.0)


@dataclass(frozen=True)
class ShadowCrossSection:
    axial_distance_km: np.ndarray
    axis_offset_km: np.ndarray
    signed_umbra_radius_km: np.ndarray
    penumbra_radius_km: np.ndarray
    phase: np.ndarray


def shadow_cone(
    sun_position_from_occluder_km,
    R_occ_km,
    R_sun_km=R_SUN_KM,
) -> ShadowCone:
    """Construct exact common-tangent cone angles for finite spherical bodies."""
    s = np.asarray(sun_position_from_occluder_km, dtype=float)
    if s.shape != (3,):
        raise ValueError("shadow_cone expects one 3-vector")
    D = float(np.linalg.norm(s))
    if D <= R_sun_km + R_occ_km:
        raise ValueError("Sun and occluder spheres overlap or distance is invalid")
    axis = -s / D
    theta_u = float(np.arcsin((R_sun_km - R_occ_km) / D))
    theta_p = float(np.arcsin((R_sun_km + R_occ_km) / D))
    L_u = float(R_occ_km / np.sin(theta_u))
    L_p = float(R_occ_km / np.sin(theta_p))
    return ShadowCone(axis, D, theta_u, theta_p, L_u, L_p)


def shadow_cross_section(
    target_position_from_occluder_km,
    sun_position_from_occluder_km,
    R_occ_km,
    target_radius_km=0.0,
    R_sun_km=R_SUN_KM,
) -> ShadowCrossSection:
    """Axis offset and cone radii at a target centre, with phase classification."""
    target = np.asarray(target_position_from_occluder_km, dtype=float)
    sun = np.asarray(sun_position_from_occluder_km, dtype=float)
    target, sun = np.broadcast_arrays(target, sun)
    flat_t = target.reshape(-1, 3)
    flat_s = sun.reshape(-1, 3)
    x = np.empty(len(flat_t)); rho = np.empty(len(flat_t)); ru = np.empty(len(flat_t)); rp = np.empty(len(flat_t))
    labels: list[str] = []
    for i, (r, sv) in enumerate(zip(flat_t, flat_s)):
        cone = shadow_cone(sv, R_occ_km, R_sun_km)
        x[i] = np.dot(r, cone.axis_hat)
        rho[i] = np.linalg.norm(r - x[i] * cone.axis_hat)
        ru[i], rp[i] = cone.radii_at(x[i])
        if x[i] <= 0 or rho[i] - target_radius_km >= rp[i]:
            labels.append("sunlit")
        elif ru[i] > 0 and rho[i] + target_radius_km <= ru[i]:
            labels.append("total umbra")
        elif ru[i] > 0 and rho[i] - target_radius_km < ru[i]:
            labels.append("partial umbra")
        elif ru[i] <= 0 and rho[i] - target_radius_km < abs(ru[i]):
            labels.append("antumbra")
        else:
            labels.append("penumbra")
    shape = target.shape[:-1]
    return ShadowCrossSection(
        axial_distance_km=x.reshape(shape),
        axis_offset_km=rho.reshape(shape),
        signed_umbra_radius_km=ru.reshape(shape),
        penumbra_radius_km=rp.reshape(shape),
        phase=np.asarray(labels, dtype=object).reshape(shape),
    )


def ray_sphere_intersections(origin, direction, center, radius_km):
    """Sorted non-negative sphere roots from the canonical SSAPy kernel."""
    roots = np.asarray(
        _core_ray_sphere_intersections(origin, direction, radius_km, center=center),
        dtype=float,
    ).reshape(-1)
    return np.sort(roots[np.isfinite(roots) & (roots >= 0.0)])


def ray_ellipsoid_intersections(origin, direction, center, axes_km=EARTH_AXES_KM):
    """Sorted non-negative ellipsoid roots from the canonical SSAPy kernel."""
    roots = np.asarray(
        _core_ray_ellipsoid_intersections(origin, direction, axes_km, center=center),
        dtype=float,
    ).reshape(-1)
    return np.sort(roots[np.isfinite(roots) & (roots >= 0.0)])


def shadow_axis_surface_point(
    occluder_position_gcrf_km,
    sun_position_gcrf_km,
    target_center_gcrf_km=np.zeros(3),
    target_radius_km=RE_KM,
    target_axes_km=None,
):
    """Nearest target-surface hit of the shadow axis, or ``None``."""
    occ = np.asarray(occluder_position_gcrf_km, dtype=float)
    sun = np.asarray(sun_position_gcrf_km, dtype=float)
    target = np.asarray(target_center_gcrf_km, dtype=float)
    d = occ - sun
    d /= np.linalg.norm(d)
    roots = (ray_ellipsoid_intersections(occ, d, target, target_axes_km)
             if target_axes_km is not None
             else ray_sphere_intersections(occ, d, target, target_radius_km))
    if roots.size == 0:
        return None
    return occ + roots[0] * d


def _orthonormal_basis(axis_hat):
    a = np.asarray(axis_hat, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


def cone_sphere_boundary(
    occluder_position_gcrf_km,
    sun_position_gcrf_km,
    target_center_gcrf_km=np.zeros(3),
    target_radius_km=RE_KM,
    target_axes_km=None,
    R_occ_km=R_MOON_KM,
    kind: Literal["umbra", "antumbra", "penumbra"] = "penumbra",
    n_azimuth=180,
    prefer_toolkit: bool = True,
    prefer_astropy: bool = True,
):
    """Sample the true cone/sphere intersection boundary in GCRF.

    Missing azimuths are returned as NaN rows.  Unlike a tangent-plane disc,
    this curve follows the target's spherical surface and naturally becomes
    asymmetric near the limb.
    """
    occ = np.asarray(occluder_position_gcrf_km, dtype=float)
    sun = np.asarray(sun_position_gcrf_km, dtype=float)
    target = np.asarray(target_center_gcrf_km, dtype=float)
    cone = shadow_cone(sun - occ, R_occ_km)
    axis = cone.axis_hat
    u, v = _orthonormal_basis(axis)

    if kind == "penumbra":
        vertex = occ - axis * cone.penumbra_vertex_distance_km
        theta = cone.penumbra_half_angle_rad
        sign = 1.0
    else:
        vertex = occ + axis * cone.umbra_apex_distance_km
        theta = cone.umbra_half_angle_rad
        side = float(np.dot(target - vertex, axis))
        if kind == "umbra" and side >= 0:
            return np.full((n_azimuth, 3), np.nan)
        if kind == "antumbra" and side <= 0:
            return np.full((n_azimuth, 3), np.nan)
        sign = 1.0 if side >= 0 else -1.0

    phi = np.linspace(0.0, 2.0*np.pi, int(n_azimuth), endpoint=False)
    result = np.full((len(phi), 3), np.nan)
    for i, p in enumerate(phi):
        radial = np.cos(p) * u + np.sin(p) * v
        d = sign * np.cos(theta) * axis + np.sin(theta) * radial
        roots = (ray_ellipsoid_intersections(vertex, d, target, target_axes_km)
                 if target_axes_km is not None
                 else ray_sphere_intersections(vertex, d, target, target_radius_km))
        if roots.size:
            # For an umbral apex lying beyond Earth, rays are traced backward
            # from the apex.  The first hit is the night-side entry point; the
            # eclipse footprint is the second, Sun-facing exit point.
            root = roots[-1] if kind == "umbra" and sign < 0 and roots.size > 1 else roots[0]
            result[i] = vertex + root * d
    return result


# ---------------------------------------------------------------------------
# Earth-fixed conversion and ground tracks
# ---------------------------------------------------------------------------


def _gmst_rad(jd):
    T = (np.asarray(jd, dtype=float) - 2_451_545.0) / 36_525.0
    deg = 280.46061837 + 360.98564736629 * (np.asarray(jd, dtype=float) - 2_451_545.0) \
          + 0.000387933 * T*T - T*T*T / 38_710_000.0
    return np.radians(deg % 360.0)


def gcrf_to_itrf_km(points_gcrf_km, times_jd, prefer_toolkit=True, prefer_astropy=True):
    """Convert GCRF positions to ITRF in kilometres.

    Uses ssapy-toolkit's official ``gcrf_to_itrf`` helper when available;
    Astropy GCRS->ITRS is the secondary high-fidelity path, followed by the
    deterministic GMST plotting fallback.
    """
    p = np.asarray(points_gcrf_km, dtype=float)
    scalar = p.ndim == 1
    p2 = p.reshape(-1, 3)
    jd = np.asarray(times_jd, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(p2), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(p2),))
    if prefer_toolkit and _toolkit_backend() is not None and _ssapy_backend() is not None:
        try:
            Time, _ = _ssapy_backend()  # type: ignore[misc]
            gcrf_to_itrf, _, _ = _toolkit_backend()  # type: ignore[misc]
            gps_seconds = np.asarray(Time(jd, format="jd", scale="utc").gps, dtype=float)
            out = np.asarray(gcrf_to_itrf(p2*1_000.0, gps_seconds), dtype=float)/1_000.0
            out = _as_n_by_3(out, len(p2))
            return out[0] if scalar else out
        except Exception as ex:
            warnings.warn(f"ssapy-toolkit GCRF->ITRF failed; using GMST fallback: {ex}", RuntimeWarning)
    if prefer_astropy:
        try:
            import astropy.units as u  # type: ignore
            from astropy.coordinates import CartesianRepresentation, GCRS, ITRS  # type: ignore
            from astropy.time import Time  # type: ignore
            t = Time(jd, format="jd", scale="utc")
            rep = CartesianRepresentation(p2[:, 0]*u.km, p2[:, 1]*u.km, p2[:, 2]*u.km)
            coord = GCRS(rep, obstime=t).transform_to(ITRS(obstime=t))
            out = np.asarray(coord.cartesian.xyz.to_value(u.km), dtype=float).T
            return out[0] if scalar else out
        except Exception:
            pass
    theta = _gmst_rad(jd)
    ct, st = np.cos(theta), np.sin(theta)
    out = np.column_stack([
        ct*p2[:, 0] + st*p2[:, 1],
        -st*p2[:, 0] + ct*p2[:, 1],
        p2[:, 2],
    ])
    return out[0] if scalar else out


def itrf_to_gcrf_km(points_itrf_km, times_jd, prefer_toolkit=True, prefer_astropy=True):
    """Convert fixed ITRF positions to GCRF in kilometres.

    SSAPy Toolkit's official inverse transform is preferred.  Astropy's
    ITRS->GCRS transform is the secondary high-fidelity path, and a
    deterministic inverse-GMST rotation keeps the package usable without
    either optional dependency.
    """
    p = np.asarray(points_itrf_km, dtype=float)
    scalar = p.ndim == 1
    p2 = p.reshape(-1, 3)
    jd = np.asarray(times_jd, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(p2), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(p2),))
    if prefer_toolkit and _toolkit_backend() is not None and _ssapy_backend() is not None:
        try:
            Time, _ = _ssapy_backend()  # type: ignore[misc]
            _, _, itrf_to_gcrf = _toolkit_backend()  # type: ignore[misc]
            gps_seconds = np.asarray(Time(jd, format="jd", scale="utc").gps, dtype=float)
            out = np.asarray(
                itrf_to_gcrf(p2*1_000.0, gps_seconds),
                dtype=float,
            )/1_000.0
            out = _as_n_by_3(out, len(p2))
            return out[0] if scalar else out
        except Exception as ex:
            warnings.warn(f"ssapy-toolkit ITRF->GCRF failed; using Astropy/GMST fallback: {ex}",
                          RuntimeWarning)
    if prefer_astropy:
        try:
            import astropy.units as u  # type: ignore
            from astropy.coordinates import CartesianRepresentation, GCRS, ITRS  # type: ignore
            from astropy.time import Time  # type: ignore
            t = Time(jd, format="jd", scale="utc")
            rep = CartesianRepresentation(p2[:, 0]*u.km, p2[:, 1]*u.km, p2[:, 2]*u.km)
            coord = ITRS(rep, obstime=t).transform_to(GCRS(obstime=t))
            out = np.asarray(coord.cartesian.xyz.to_value(u.km), dtype=float).T
            return out[0] if scalar else out
        except Exception:
            pass
    theta = _gmst_rad(jd)
    ct, st = np.cos(theta), np.sin(theta)
    out = np.column_stack([
        ct*p2[:, 0] - st*p2[:, 1],
        st*p2[:, 0] + ct*p2[:, 1],
        p2[:, 2],
    ])
    return out[0] if scalar else out


def ellipsoid_surface_in_direction(direction, axes_km=EARTH_AXES_KM):
    """Radial intersection of a direction with an axis-aligned ellipsoid."""
    d = np.asarray(direction, dtype=float)
    axes = np.asarray(axes_km, dtype=float)
    norm = np.sqrt(np.sum((d/axes)**2, axis=-1, keepdims=True))
    return d/np.maximum(norm, np.finfo(float).eps)


def gcrf_to_lonlat_km(points_gcrf_km, times_jd, prefer_toolkit=True):
    """Convert GCRF points to geocentric lon/lat/height.

    The official ssapy-toolkit transform is used when available.  The fallback
    applies GMST rotation and a spherical-height approximation, sufficient for
    plotting but not geodetic analysis.
    """
    p = np.asarray(points_gcrf_km, dtype=float)
    scalar = p.ndim == 1
    p2 = p.reshape(-1, 3)
    jd = np.asarray(times_jd, dtype=float)
    if jd.ndim == 0:
        jd = np.full(len(p2), float(jd))
    jd = np.broadcast_to(jd.reshape(-1), (len(p2),))

    if prefer_toolkit and _toolkit_backend() is not None and _ssapy_backend() is not None:
        try:
            Time, _ = _ssapy_backend()  # type: ignore[misc]
            _, gcrf_to_lonlat, _ = _toolkit_backend()  # type: ignore[misc]
            t = np.asarray(Time(jd, format="jd", scale="utc").gps, dtype=float)
            try:
                lon, lat, h_m = gcrf_to_lonlat(p2 * 1_000.0, t)
            except Exception:
                # Some toolkit/SSAPy releases expose coordinate arrays as
                # (3, N), while newer helpers accept (N, 3).
                lon, lat, h_m = gcrf_to_lonlat((p2 * 1_000.0).T, t)
            lon = np.asarray(lon, dtype=float).reshape(-1)
            lat = np.asarray(lat, dtype=float).reshape(-1)
            h = np.asarray(h_m, dtype=float).reshape(-1) / 1_000.0
            if scalar:
                return float(lon[0]), float(lat[0]), float(h[0])
            return lon, lat, h
        except Exception as ex:
            warnings.warn(f"ssapy-toolkit GCRF conversion failed; using GMST fallback: {ex}", RuntimeWarning)

    theta = _gmst_rad(jd)
    ct, st = np.cos(theta), np.sin(theta)
    x = ct * p2[:, 0] + st * p2[:, 1]
    y = -st * p2[:, 0] + ct * p2[:, 1]
    z = p2[:, 2]
    pxy = np.hypot(x, y)
    lon = np.degrees(np.arctan2(y, x))
    # Bowring closed-form geodetic latitude on WGS-84.
    a, b = RE_KM, RP_EARTH_KM
    e2 = 1.0-(b*b)/(a*a)
    ep2 = (a*a-b*b)/(b*b)
    theta_b = np.arctan2(z*a, pxy*b)
    st_b, ct_b = np.sin(theta_b), np.cos(theta_b)
    lat_r = np.arctan2(z+ep2*b*st_b**3, pxy-e2*a*ct_b**3)
    N = a/np.sqrt(1.0-e2*np.sin(lat_r)**2)
    coslat = np.cos(lat_r)
    h = np.where(np.abs(coslat) > 1e-12, pxy/coslat-N, np.abs(z)-b)
    lat = np.degrees(lat_r)
    if scalar:
        return float(lon[0]), float(lat[0]), float(h[0])
    return lon, lat, h


def earth_shadow_axis_surface_point(
    moon_gcrf_km, sun_gcrf_km, time_jd, *,
    prefer_toolkit: bool = True, prefer_astropy: bool = True,
):
    """Shadow-axis hit on the WGS-84 ellipsoid, returned in GCRF.

    The intersection is solved in ITRF so the ellipsoid's polar axis follows
    the actual Earth-fixed frame supplied by SSAPy Toolkit.  This avoids
    treating WGS-84 as though it were permanently aligned with the inertial
    GCRF axes.
    """
    moon_itrf, sun_itrf = gcrf_to_itrf_km(
        np.asarray([moon_gcrf_km, sun_gcrf_km], dtype=float),
        np.asarray([time_jd, time_jd], dtype=float),
        prefer_toolkit=prefer_toolkit, prefer_astropy=prefer_astropy,
    )
    hit_itrf = shadow_axis_surface_point(
        moon_itrf, sun_itrf, target_axes_km=EARTH_AXES_KM)
    if hit_itrf is None:
        return None
    return np.asarray(itrf_to_gcrf_km(
        hit_itrf, time_jd, prefer_toolkit=prefer_toolkit,
        prefer_astropy=prefer_astropy,
    ), dtype=float)


def earth_cone_boundary_gcrf(
    moon_gcrf_km,
    sun_gcrf_km,
    time_jd,
    *,
    kind: Literal["umbra", "antumbra", "penumbra"] = "penumbra",
    n_azimuth=180,
    prefer_toolkit: bool = True,
    prefer_astropy: bool = True,
    event=None,
):
    """Exact Moon-shadow cone/WGS-84 boundary, returned in GCRF.

    When ``event`` is supplied, its immutable backend provenance owns both
    frame transformations.  Legacy callers may still select optional generic
    transforms with ``prefer_toolkit``/``prefer_astropy``.
    """
    points_gcrf = np.asarray([moon_gcrf_km, sun_gcrf_km], dtype=float)
    times = np.asarray([time_jd, time_jd], dtype=float)
    if event is not None:
        from ssapy_toolkit.compute.eclipse_state import event_gcrf_to_itrf_km
        moon_itrf, sun_itrf = event_gcrf_to_itrf_km(event, points_gcrf, times)
    else:
        moon_itrf, sun_itrf = gcrf_to_itrf_km(
            points_gcrf, times,
            prefer_toolkit=prefer_toolkit, prefer_astropy=prefer_astropy,
        )
    boundary_itrf = cone_sphere_boundary(
        moon_itrf, sun_itrf, kind=kind, n_azimuth=n_azimuth,
        target_axes_km=EARTH_AXES_KM,
    )
    valid = np.all(np.isfinite(boundary_itrf), axis=1)
    result = np.full_like(boundary_itrf, np.nan)
    if np.any(valid):
        output_times = np.full(np.sum(valid), float(time_jd))
        if event is not None:
            from ssapy_toolkit.compute.eclipse_state import event_itrf_to_gcrf_km
            result[valid] = event_itrf_to_gcrf_km(
                event, boundary_itrf[valid], output_times
            )
        else:
            result[valid] = itrf_to_gcrf_km(
                boundary_itrf[valid], output_times,
                prefer_toolkit=prefer_toolkit, prefer_astropy=prefer_astropy,
            )
    return result


def solar_shadow_ground_track(jd, moon_gcrf_km, sun_gcrf_km):
    """Return central shadow-axis ground points and lon/lat arrays.

    Rows where the axis misses Earth are NaN.  This is the geometric central
    line; penumbral/umbral boundary curves can be obtained with
    :func:`cone_sphere_boundary`.
    """
    jd = np.asarray(jd, dtype=float).reshape(-1)
    moon = np.asarray(moon_gcrf_km, dtype=float).reshape(-1, 3)
    sun = np.asarray(sun_gcrf_km, dtype=float).reshape(-1, 3)
    points = np.full_like(moon, np.nan)
    for i, (m, s) in enumerate(zip(moon, sun)):
        hit = earth_shadow_axis_surface_point(m, s, jd[i])
        if hit is not None:
            points[i] = hit
    valid = np.all(np.isfinite(points), axis=1)
    lon = np.full(len(jd), np.nan); lat = np.full(len(jd), np.nan); h = np.full(len(jd), np.nan)
    if np.any(valid):
        lon[valid], lat[valid], h[valid] = gcrf_to_lonlat_km(points[valid], jd[valid])
    return points, lon, lat, h


# ---------------------------------------------------------------------------
# Convenience geometry for eclipse searching
# ---------------------------------------------------------------------------


class EclipseMetrics(NamedTuple):
    illumination: np.ndarray
    separation_rad: np.ndarray
    sun_angular_radius_rad: np.ndarray
    occulter_angular_radius_rad: np.ndarray
    shadow_axial_distance_km: np.ndarray
    shadow_axis_offset_km: np.ndarray
    signed_umbra_radius_km: np.ndarray
    penumbra_radius_km: np.ndarray


def eclipse_metrics(mode, moon_gcrf_km, sun_gcrf_km):
    """Vectorized central geometry for ``mode='solar'`` or ``'lunar'``."""
    mode = str(mode).lower()
    moon = np.asarray(moon_gcrf_km, dtype=float)
    sun = np.asarray(sun_gcrf_km, dtype=float)
    if mode == "solar":
        # Observer Earth centre, occluder Moon.
        r_eval = -moon
        sun_from_moon = sun - moon
        illum, geom = illumination_fraction(
            r_eval, R_body_km=R_MOON_KM, sun_position_km=sun_from_moon,
            return_geometry=True,
        )
        cross = shadow_cross_section(-moon, sun-moon, R_MOON_KM, target_radius_km=RE_KM)
    elif mode == "lunar":
        # Observer Moon centre, occluder Earth.
        illum, geom = illumination_fraction(
            moon, R_body_km=RE_KM, sun_position_km=sun,
            return_geometry=True,
        )
        cross = shadow_cross_section(moon, sun, RE_KM, target_radius_km=R_MOON_KM)
    else:
        raise ValueError("mode must be 'solar' or 'lunar'")
    return EclipseMetrics(
        illumination=np.asarray(illum),
        separation_rad=np.asarray(geom.separation_rad),
        sun_angular_radius_rad=np.asarray(geom.sun_angular_radius_rad),
        occulter_angular_radius_rad=np.asarray(geom.occluder_angular_radius_rad),
        shadow_axial_distance_km=np.asarray(cross.axial_distance_km),
        shadow_axis_offset_km=np.asarray(cross.axis_offset_km),
        signed_umbra_radius_km=np.asarray(cross.signed_umbra_radius_km),
        penumbra_radius_km=np.asarray(cross.penumbra_radius_km),
    )


__all__ = [
    "MU_EARTH_KM3S2", "RE_KM", "RP_EARTH_KM", "EARTH_AXES_KM", "R_MOON_KM", "R_SUN_KM", "AU_KM", "DEFAULT_EPOCH_JD",
    "available_backends", "datetime_to_jd", "jd_to_datetime", "ephemeris_positions",
    "propagate_eci", "sun_position_eci", "sun_direction_eci", "illumination_fraction",
    "apparent_disk_geometry", "shadow_cone", "shadow_cross_section",
    "shadow_axis_surface_point", "ray_ellipsoid_intersections", "cone_sphere_boundary",
    "earth_shadow_axis_surface_point", "earth_cone_boundary_gcrf",
    "gcrf_to_itrf_km", "itrf_to_gcrf_km", "ellipsoid_surface_in_direction", "gcrf_to_lonlat_km",
    "solar_shadow_ground_track", "eclipse_metrics", "_circle_overlap_fraction",
]
