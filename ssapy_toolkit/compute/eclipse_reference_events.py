"""NASA-reference eclipse geometry for validated figures and animations.

The built-in solar event is the 2024-04-08 total solar eclipse.  Its
Earth-fixed Sun/Moon geometry is reconstructed from NASA/GSFC polynomial
Besselian elements.  The central line therefore follows the WGS-84 path
published by NASA rather than an unconstrained two-body Moon orbit.

The built-in lunar event is the 2025-03-14 total lunar eclipse.  Its contact
sequence, shadow radii, greatest-eclipse geometry, Sun/Moon apparent sizes,
and Moon track through the shadow are tied to the NASA/GSFC eclipse plot.
A monotone cubic track passes through every published contact exactly.

All distances are kilometres.  Earth-fixed coordinates use an ITRF-like
WGS-84 frame: +X at Greenwich, +Y east at 90 E, +Z north.  This is the right
frame for plotting a moving eclipse path on a non-moving Earth texture.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Mapping, Sequence
import math

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - linear fallback is tested indirectly
    PchipInterpolator = None

RE_KM = 6378.137
RP_KM = 6356.752314245
EARTH_AXES_KM = np.array([RE_KM, RE_KM, RP_KM], dtype=float)
R_SUN_KM = 695700.0
R_MOON_MEAN_KM = 1737.4
AU_KM = 149597870.7


def datetime_to_jd(value: datetime) -> float:
    """Convert an aware UTC datetime to Julian Date (UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.timestamp()/86400.0 + 2440587.5


def jd_to_datetime(jd: float) -> datetime:
    """Convert Julian Date (UTC) to an aware UTC datetime."""
    return datetime.fromtimestamp((float(jd)-2440587.5)*86400.0, tz=timezone.utc)


def utc(year: int, month: int, day: int, hour: int, minute: int,
        second: float = 0.0) -> datetime:
    whole = int(math.floor(second))
    micro = int(round((second-whole)*1_000_000))
    if micro == 1_000_000:
        whole += 1
        micro = 0
    return datetime(year, month, day, hour, minute, whole, micro, tzinfo=timezone.utc)


def _unit(vector, *, name: str = "vector") -> np.ndarray:
    # Ravel first. np.linalg.norm without an axis returns a scalar, so this
    # only ever normalises a single vector -- but callers pass results of
    # frame conversions, which normalise a single point to shape (1, 3).
    # Preserving that shape makes every downstream np.dot(u, v) a matrix
    # product between two (1, 3) arrays, which raises rather than returning
    # a scalar.
    arr = np.ravel(np.asarray(vector, dtype=float))
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have finite nonzero length")
    return arr/norm


def _poly(coeff: Sequence[float], t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t, dtype=float)
    for power, value in enumerate(coeff):
        result += float(value)*t**power
    return result


def _poly_derivative(coeff: Sequence[float], t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t, dtype=float)
    for power, value in enumerate(coeff[1:], start=1):
        result += power*float(value)*t**(power-1)
    return result


def _ray_ellipsoid_roots(origin, direction,
                         axes_km: np.ndarray = EARTH_AXES_KM) -> np.ndarray:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = _unit(direction, name="ray direction")
    # Extended precision avoids losing the ~12,000 km separation between
    # near/far Earth roots when both roots are ~150 million km from the Sun.
    ld = np.longdouble
    axes = np.asarray(axes_km, dtype=ld).reshape(3)
    q = np.ravel(np.asarray(origin, dtype=ld))/axes
    v = np.ravel(np.asarray(direction, dtype=ld))/axes
    aa = np.dot(v, v)
    bb = ld(2.0)*np.dot(q, v)
    cc = np.dot(q, q)-ld(1.0)
    disc = bb*bb-ld(4.0)*aa*cc
    if disc < 0.0:
        return np.empty(0, dtype=float)
    root = np.sqrt(max(disc, ld(0.0)))
    qroot = -ld(0.5)*(bb+np.copysign(root, bb))
    if abs(qroot) < ld(1.0e-40):
        roots = np.array([-bb/(ld(2.0)*aa), -bb/(ld(2.0)*aa)], dtype=ld)
    else:
        roots = np.array([qroot/aa, cc/qroot], dtype=ld)
    return np.sort(np.asarray(roots, dtype=float)[np.isfinite(roots)])


def geodetic_from_itrf(point_km) -> tuple[float, float, float]:
    """Return WGS-84 geodetic latitude, east longitude, and height."""
    x, y, z = np.asarray(point_km, dtype=float).reshape(3)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    a, b = RE_KM, RP_KM
    e2 = 1.0-(b*b)/(a*a)
    if p < 1.0e-14:
        lat = math.copysign(math.pi/2.0, z)
        height = abs(z)-b
        return math.degrees(lat), math.degrees(lon), height
    lat = math.atan2(z, p*(1.0-e2))
    height = 0.0
    for _ in range(12):
        sin_lat = math.sin(lat)
        N = a/math.sqrt(1.0-e2*sin_lat*sin_lat)
        height = p/max(math.cos(lat), 1.0e-15)-N
        lat_new = math.atan2(z, p*(1.0-e2*N/(N+height)))
        if abs(lat_new-lat) < 2.0e-15:
            lat = lat_new
            break
        lat = lat_new
    lon_deg = (math.degrees(lon)+180.0) % 360.0-180.0
    return math.degrees(lat), lon_deg, height


def itrf_surface_point(lat_deg: float, lon_deg: float, height_km: float = 0.0) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    e2 = 1.0-(RP_KM*RP_KM)/(RE_KM*RE_KM)
    N = RE_KM/math.sqrt(1.0-e2*math.sin(lat)**2)
    return np.array([
        (N+height_km)*math.cos(lat)*math.cos(lon),
        (N+height_km)*math.cos(lat)*math.sin(lon),
        (N*(1.0-e2)+height_km)*math.sin(lat),
    ])


def angular_circle_visible_fraction(occulter_radius_rad, sun_radius_rad,
                                    separation_rad) -> np.ndarray:
    """Visible fraction of a circular solar disk after circular occultation."""
    r1, r2, d = np.broadcast_arrays(
        np.asarray(occulter_radius_rad, dtype=float),
        np.asarray(sun_radius_rad, dtype=float),
        np.asarray(separation_rad, dtype=float),
    )
    visible = np.ones_like(d)
    no_overlap = d >= r1+r2
    contained = d <= np.abs(r1-r2)
    visible = np.where(contained & (r1 >= r2), 0.0, visible)
    with np.errstate(divide="ignore", invalid="ignore"):
        annular = 1.0-np.clip((r1/np.maximum(r2, 1.0e-30))**2, 0.0, 1.0)
    visible = np.where(contained & (r1 < r2), annular, visible)
    partial = ~(no_overlap | contained)
    dp = np.where(partial, d, 1.0)
    a = np.clip((dp*dp+r1*r1-r2*r2)/(2.0*dp*np.maximum(r1, 1e-30)), -1.0, 1.0)
    b = np.clip((dp*dp+r2*r2-r1*r1)/(2.0*dp*np.maximum(r2, 1e-30)), -1.0, 1.0)
    term = (-dp+r1+r2)*(dp+r1-r2)*(dp-r1+r2)*(dp+r1+r2)
    area = r1*r1*np.arccos(a)+r2*r2*np.arccos(b)-0.5*np.sqrt(np.clip(term, 0.0, None))
    partial_visible = 1.0-area/(math.pi*np.maximum(r2*r2, 1e-30))
    visible = np.where(partial, partial_visible, visible)
    return np.clip(visible, 0.0, 1.0)


@dataclass(frozen=True)
class ReferenceDefinition:
    key: str
    mode: str
    title: str
    source_label: str
    contacts_utc: Mapping[str, datetime]
    greatest_utc: datetime
    notes: tuple[str, ...] = ()

    @property
    def contacts_jd(self) -> dict[str, float]:
        return {name: datetime_to_jd(value) for name, value in self.contacts_utc.items()}

    @property
    def greatest_jd(self) -> float:
        return datetime_to_jd(self.greatest_utc)


SOLAR_2024 = ReferenceDefinition(
    key="solar_2024_04_08",
    mode="solar",
    title="Total Solar Eclipse — 2024-04-08",
    source_label="NASA/GSFC Besselian elements and eclipse contact plot",
    contacts_utc={
        "P1": utc(2024, 4, 8, 15, 42, 9.4),
        "U1": utc(2024, 4, 8, 16, 38, 46.8),
        "U2": utc(2024, 4, 8, 16, 41, 4.1),
        "P2": utc(2024, 4, 8, 17, 44, 55.0),
        "MAX": utc(2024, 4, 8, 18, 17, 18.3),
        "P3": utc(2024, 4, 8, 18, 49, 10.1),
        "U3": utc(2024, 4, 8, 19, 53, 16.4),
        "U4": utc(2024, 4, 8, 19, 55, 31.6),
        "P4": utc(2024, 4, 8, 20, 52, 16.3),
    },
    greatest_utc=utc(2024, 4, 8, 18, 17, 18.3),
    notes=(
        "Geometry uses the NASA polynomial Besselian set with Delta-T = 70.6 s.",
        "Global contact labels are NASA/GSFC values; the companion NASA plot uses a nearby ephemeris/Delta-T set, so sub-four-second differences are expected.",
    ),
)


LUNAR_2025 = ReferenceDefinition(
    key="lunar_2025_03_14",
    mode="lunar",
    title="Total Lunar Eclipse — 2025-03-14",
    source_label="NASA/GSFC total lunar eclipse plot",
    contacts_utc={
        "P1": utc(2025, 3, 14, 3, 57, 24.0),
        "U1": utc(2025, 3, 14, 5, 9, 33.0),
        "U2": utc(2025, 3, 14, 6, 25, 59.0),
        "MAX": utc(2025, 3, 14, 6, 58, 41.7),
        "U3": utc(2025, 3, 14, 7, 31, 23.0),
        "U4": utc(2025, 3, 14, 8, 47, 48.0),
        "P4": utc(2025, 3, 14, 10, 0, 1.0),
    },
    greatest_utc=utc(2025, 3, 14, 6, 58, 41.7),
    notes=(
        "The effective shadow radii use the NASA Danjon-rule values, including the atmospheric enlargement of Earth's shadow.",
    ),
)


# NASA/GSFC polynomial Besselian elements for 2024-04-08, t0 = 18:00 TDT.
_SOLAR_BESSEL = {
    "x": (-0.318157, 0.5117105, 0.0000326, -0.0000085),
    "y": (0.219747, 0.2709586, -0.0000594, -0.0000047),
    "d": (7.58620, 0.014844, -0.000002),
    "l1": (0.535813, 0.0000618, -0.0000128),
    "l2": (-0.010274, 0.0000615, -0.0000127),
    "mu": (89.59122, 15.004084),
}
_SOLAR_TAN_F1 = 0.0046683
_SOLAR_TAN_F2 = 0.0046450
_SOLAR_DELTA_T_S = 70.6
_SOLAR_K1 = 0.272488
_SOLAR_K2 = 0.272281
SOLAR_UMBRA_OPTICAL_RADIUS_KM = _SOLAR_K2*RE_KM
SOLAR_PENUMBRA_OPTICAL_RADIUS_KM = _SOLAR_K1*RE_KM
SOLAR_GREATEST_SITE_LAT_DEG = 25.0+17.2/60.0
SOLAR_GREATEST_SITE_LON_EAST_DEG = -(104.0+8.3/60.0)
_SOLAR_T0_TDT_JD = datetime_to_jd(utc(2024, 4, 8, 18, 0, 0.0))


@dataclass(frozen=True)
class SolarBesselianState:
    jd_utc: float
    t_hours_tdt: float
    x: float
    y: float
    d_deg: float
    l1: float
    l2: float
    mu_tdt_deg: float
    mu_utc_deg: float
    x_hat_itrf: np.ndarray
    y_hat_itrf: np.ndarray
    z_hat_sunward_itrf: np.ndarray
    moon_itrf_km: np.ndarray
    sun_itrf_km: np.ndarray
    moon_distance_km: float
    sun_distance_km: float
    sun_moon_distance_km: float

    @property
    def shadow_axis_hat_itrf(self) -> np.ndarray:
        return -self.z_hat_sunward_itrf


def solar_besselian_state(jd_utc: float) -> SolarBesselianState:
    """Reconstruct physical Earth-fixed Sun/Moon vectors from Besselian data."""
    jd = float(jd_utc)
    t = (jd+_SOLAR_DELTA_T_S/86400.0-_SOLAR_T0_TDT_JD)*24.0
    values = {name: float(_poly(coeff, t)) for name, coeff in _SOLAR_BESSEL.items()}
    d = math.radians(values["d"])
    # Besselian mu is expressed on the dynamical timescale.  Rotate Earth by
    # the UT/TT difference before treating it as a terrestrial hour angle.
    mu_utc_deg = values["mu"]-0.00417807*_SOLAR_DELTA_T_S
    mu = math.radians(mu_utc_deg)
    z_hat = np.array([
        math.cos(d)*math.cos(mu),
        -math.cos(d)*math.sin(mu),
        math.sin(d),
    ])
    x_hat = np.array([math.sin(mu), math.cos(mu), 0.0])
    y_hat = np.cross(z_hat, x_hat)
    x_hat, y_hat, z_hat = _unit(x_hat), _unit(y_hat), _unit(z_hat)

    sin_f1 = _SOLAR_TAN_F1/math.sqrt(1.0+_SOLAR_TAN_F1**2)
    sin_f2 = _SOLAR_TAN_F2/math.sqrt(1.0+_SOLAR_TAN_F2**2)
    z1 = values["l1"]/_SOLAR_TAN_F1-_SOLAR_K1/sin_f1
    z2 = values["l2"]/_SOLAR_TAN_F2+_SOLAR_K2/sin_f2
    z_moon = 0.5*(z1+z2)
    sun_radius_er = R_SUN_KM/RE_KM
    G1 = (sun_radius_er+_SOLAR_K1)/sin_f1
    G2 = (sun_radius_er-_SOLAR_K2)/sin_f2
    G = 0.5*(G1+G2)

    moon = RE_KM*(values["x"]*x_hat+values["y"]*y_hat+z_moon*z_hat)
    sun = moon+RE_KM*G*z_hat
    return SolarBesselianState(
        jd_utc=jd,
        t_hours_tdt=t,
        x=values["x"], y=values["y"], d_deg=values["d"],
        l1=values["l1"], l2=values["l2"],
        mu_tdt_deg=values["mu"], mu_utc_deg=mu_utc_deg,
        x_hat_itrf=x_hat, y_hat_itrf=y_hat, z_hat_sunward_itrf=z_hat,
        moon_itrf_km=moon, sun_itrf_km=sun,
        moon_distance_km=float(np.linalg.norm(moon)),
        sun_distance_km=float(np.linalg.norm(sun)),
        sun_moon_distance_km=float(G*RE_KM),
    )


def solar_central_line_formula(jd_utc: float) -> tuple[float, float]:
    """NASA/Meeus central-line solution; returns geodetic lat and west lon."""
    jd = float(jd_utc)
    t = (jd+_SOLAR_DELTA_T_S/86400.0-_SOLAR_T0_TDT_JD)*24.0
    X = float(_poly(_SOLAR_BESSEL["x"], t))
    Y = float(_poly(_SOLAR_BESSEL["y"], t))
    d = math.radians(float(_poly(_SOLAR_BESSEL["d"], t)))
    mu = float(_poly(_SOLAR_BESSEL["mu"], t))
    Xp = float(_poly_derivative(_SOLAR_BESSEL["x"], t))
    Yp = float(_poly_derivative(_SOLAR_BESSEL["y"], t))
    omega = 1.0/math.sqrt(1.0-0.006694385*math.cos(d)**2)
    p = _SOLAR_BESSEL["mu"][1]/57.2957795
    b = Yp-p*X*math.sin(d)
    c = Xp+p*Y*math.sin(d)
    del b, c  # retained in the derivation; central coordinates need only H.
    y1 = omega*Y
    b1 = omega*math.sin(d)
    b2 = 0.99664719*omega*math.cos(d)
    B2 = 1.0-X*X-y1*y1
    if B2 < 0.0:
        raise ValueError("The shadow axis does not intersect Earth at this time")
    B = math.sqrt(max(B2, 0.0))
    sin_phi1 = B*b1+y1*b2
    phi1 = math.asin(np.clip(sin_phi1, -1.0, 1.0))
    cos_phi1 = math.cos(phi1)
    sin_H = X/cos_phi1
    cos_H = (B*b2-y1*b1)/cos_phi1
    H_deg = math.degrees(math.atan2(sin_H, cos_H))
    lat = math.degrees(math.atan(1.00336409*math.tan(phi1)))
    west_lon = mu-H_deg-0.00417807*_SOLAR_DELTA_T_S
    west_lon = west_lon % 360.0
    if west_lon > 180.0:
        west_lon -= 360.0
    return lat, west_lon


def solar_central_line_wgs84(jd_utc: float) -> tuple[float, float, np.ndarray] | None:
    """Intersect the physical Besselian shadow axis with WGS-84 Earth."""
    state = solar_besselian_state(jd_utc)
    roots = _ray_ellipsoid_roots(state.moon_itrf_km, state.shadow_axis_hat_itrf)
    roots = roots[roots >= 0.0]
    if not roots.size:
        return None
    point = state.moon_itrf_km+float(roots[0])*state.shadow_axis_hat_itrf
    lat, lon_east, _ = geodetic_from_itrf(point)
    return lat, lon_east, point


def solar_observer_geometry(
    jd_utc: float,
    *,
    lat_deg: float = SOLAR_GREATEST_SITE_LAT_DEG,
    lon_east_deg: float = SOLAR_GREATEST_SITE_LON_EAST_DEG,
    height_km: float = 0.0,
    moon_radius_km: float = SOLAR_UMBRA_OPTICAL_RADIUS_KM,
):
    """Topocentric Sun/Moon apparent geometry for a WGS-84 observer."""
    observer = itrf_surface_point(lat_deg, lon_east_deg, height_km)
    state = solar_besselian_state(jd_utc)
    to_sun = state.sun_itrf_km-observer
    to_moon = state.moon_itrf_km-observer
    sun_hat = _unit(to_sun)
    moon_hat = _unit(to_moon)
    a_sun = math.asin(np.clip(R_SUN_KM/np.linalg.norm(to_sun), 0.0, 1.0))
    a_moon = math.asin(np.clip(float(moon_radius_km)/np.linalg.norm(to_moon), 0.0, 1.0))
    separation = math.acos(np.clip(np.dot(sun_hat, moon_hat), -1.0, 1.0))
    visible = float(angular_circle_visible_fraction(a_moon, a_sun, separation))
    return observer, sun_hat, moon_hat, a_sun, a_moon, separation, visible


def _bisect_root(function, left: float, right: float, *, iterations: int = 70) -> float:
    f_left = float(function(left))
    f_right = float(function(right))
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left*f_right > 0.0:
        raise ValueError("root is not bracketed")
    a, b = float(left), float(right)
    for _ in range(int(iterations)):
        mid = 0.5*(a+b)
        f_mid = float(function(mid))
        if f_left*f_mid <= 0.0:
            b, f_right = mid, f_mid
        else:
            a, f_left = mid, f_mid
    return 0.5*(a+b)


@lru_cache(maxsize=32)
def solar_local_contacts(
    lat_deg: float = SOLAR_GREATEST_SITE_LAT_DEG,
    lon_east_deg: float = SOLAR_GREATEST_SITE_LON_EAST_DEG,
    height_km: float = 0.0,
) -> dict[str, float]:
    """C1-C4 for the rounded NASA greatest-eclipse site coordinates."""
    def condition(jd: float, internal: bool) -> float:
        adopted_radius = (SOLAR_UMBRA_OPTICAL_RADIUS_KM if internal
                          else SOLAR_PENUMBRA_OPTICAL_RADIUS_KM)
        *_, a_sun, a_moon, separation, _ = solar_observer_geometry(
            jd, lat_deg=lat_deg, lon_east_deg=lon_east_deg, height_km=height_km,
            moon_radius_km=adopted_radius,
        )
        radius = abs(a_moon-a_sun) if internal else a_moon+a_sun
        return separation-radius

    start = SOLAR_2024.greatest_jd-3.0/24.0
    stop = SOLAR_2024.greatest_jd+3.0/24.0
    grid = np.linspace(start, stop, 2161)
    result: dict[str, float] = {}
    for internal, names in ((False, ("C1", "C4")), (True, ("C2", "C3"))):
        values = np.asarray([condition(float(jd), internal) for jd in grid])
        roots = []
        for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if f_left == 0.0:
                roots.append(float(left))
            elif f_left*f_right < 0.0:
                roots.append(_bisect_root(lambda value: condition(value, internal), left, right))
        if len(roots) >= 2:
            result[names[0]], result[names[1]] = roots[0], roots[-1]
        elif not internal:
            raise RuntimeError("Observer has no bracketed external solar-eclipse contacts")
    if "C1" in result and "C4" in result:
        try:
            from scipy.optimize import minimize_scalar
            # Optimize in seconds around the interval midpoint.  Directly
            # optimizing a 2.46-million-valued Julian Date loses enough
            # floating-point resolution for SciPy's bounded termination test
            # to stop far from the true minimum on some platforms.
            midpoint = 0.5 * (result["C1"] + result["C4"])
            lower_s = (result["C1"] - midpoint) * 86400.0
            upper_s = (result["C4"] - midpoint) * 86400.0
            optimized = minimize_scalar(
                lambda offset_s: solar_observer_geometry(
                    midpoint + float(offset_s) / 86400.0,
                    lat_deg=lat_deg, lon_east_deg=lon_east_deg,
                    height_km=height_km, moon_radius_km=R_MOON_MEAN_KM,
                )[-2],
                bounds=(lower_s, upper_s), method="bounded",
                options={"xatol": 1.0e-4},
            )
            result["MAX"] = float(midpoint + optimized.x / 86400.0)
        except Exception:
            sample = np.linspace(result["C1"], result["C4"], 1201)
            visibility = [solar_observer_geometry(
                float(value), lat_deg=lat_deg, lon_east_deg=lon_east_deg,
                height_km=height_km, moon_radius_km=R_MOON_MEAN_KM,
            )[-1] for value in sample]
            result["MAX"] = float(sample[int(np.argmin(visibility))])
    return result


def local_contact_label_at_jd(jd_utc: float, tolerance_s: float = 0.7) -> str | None:
    for name, value in solar_local_contacts().items():
        if abs(float(jd_utc)-value)*86400.0 <= tolerance_s:
            return name
    if abs(float(jd_utc)-SOLAR_2024.greatest_jd)*86400.0 <= tolerance_s:
        return "MAX"
    return None


def solar_local_phase_status(jd_utc: float, tolerance_s: float = 0.7) -> str:
    """Human-readable phase for the fixed NASA greatest-eclipse site.

    Global P/U contacts describe the shadow touching Earth as a whole and are
    not valid phase labels for one observer.  This helper keeps local C1-C4
    animations from silently displaying global-contact intervals.
    """
    jd = float(jd_utc)
    contacts = solar_local_contacts()
    ordered = [
        ("C1", contacts["C1"]),
        ("C2", contacts["C2"]),
        ("MAX", SOLAR_2024.greatest_jd),
        ("C3", contacts["C3"]),
        ("C4", contacts["C4"]),
    ]
    for name, value in ordered:
        if abs(jd-value)*86400.0 <= float(tolerance_s):
            return f"{name} at greatest-eclipse site"
    if jd < contacts["C1"]:
        return "before local C1"
    if jd < contacts["C2"]:
        return "partial ingress — C1 → C2"
    if jd < SOLAR_2024.greatest_jd:
        return "totality — C2 → MAX"
    if jd < contacts["C3"]:
        return "totality — MAX → C3"
    if jd < contacts["C4"]:
        return "partial egress — C3 → C4"
    return "after local C4"


def _ra_dec_to_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    return np.array([math.cos(dec)*math.cos(ra), math.cos(dec)*math.sin(ra), math.sin(dec)])


def _unit_to_ra_dec(vector) -> tuple[float, float]:
    v = _unit(vector)
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, math.degrees(math.asin(v[2]))


def _rotation_from_to(source, target) -> np.ndarray:
    """Proper rotation taking one unit vector to another."""
    a, b = _unit(source), _unit(target)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1.0e-15:
        if c > 0.0:
            return np.eye(3)
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, ref))
        return 2.0*np.outer(axis, axis)-np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3)+vx+vx@vx*((1.0-c)/(s*s))


def _approx_sun_equatorial_unit(jd_utc: float) -> np.ndarray:
    """Compact Meeus-style apparent solar direction, corrected at eclipse max."""
    T = (float(jd_utc)-2451545.0)/36525.0
    L0 = (280.46646+36000.76983*T+0.0003032*T*T) % 360.0
    M = math.radians((357.52911+35999.05029*T-0.0001537*T*T) % 360.0)
    C = ((1.914602-0.004817*T-0.000014*T*T)*math.sin(M)
         +(0.019993-0.000101*T)*math.sin(2*M)+0.000289*math.sin(3*M))
    true_lon = L0+C
    omega = math.radians(125.04-1934.136*T)
    apparent_lon = math.radians(true_lon-0.00569-0.00478*math.sin(omega))
    eps0 = 23.439291-0.0130042*T
    eps = math.radians(eps0+0.00256*math.cos(omega))
    return _unit(np.array([
        math.cos(apparent_lon),
        math.cos(eps)*math.sin(apparent_lon),
        math.sin(eps)*math.sin(apparent_lon),
    ]))


# NASA 2025-03-14 greatest-eclipse apparent data.
_LUNAR_SUN_RA_DEG = (23.0+37.0/60.0+46.0/3600.0)*15.0
_LUNAR_SUN_DEC_DEG = -(2.0+24.0/60.0+16.8/3600.0)
_LUNAR_MOON_RA_DEG = (11.0+38.0/60.0+23.0/3600.0)*15.0
_LUNAR_MOON_DEC_DEG = 2.0+40.0/60.0+54.6/3600.0
_LUNAR_SUN_SD_DEG = (16.0+5.2/60.0)/60.0
_LUNAR_MOON_SD_DEG = (14.0+52.8/60.0)/60.0
_LUNAR_P_RADIUS_DEG = 1.1899
_LUNAR_U_RADIUS_DEG = 0.6537
_LUNAR_AXIS_DEG = 0.3171
_LUNAR_P_MAG = 2.2595
_LUNAR_U_MAG = 1.1784
_LUNAR_MOON_HP_DEG = (54.0+36.8/60.0)/60.0
_LUNAR_SUN_DISTANCE_KM = R_SUN_KM/math.sin(math.radians(_LUNAR_SUN_SD_DEG))
_LUNAR_MOON_DISTANCE_KM = RE_KM/math.sin(math.radians(_LUNAR_MOON_HP_DEG))
_LUNAR_SUN_EXACT = _ra_dec_to_unit(_LUNAR_SUN_RA_DEG, _LUNAR_SUN_DEC_DEG)
_LUNAR_MODEL_AT_MAX = _approx_sun_equatorial_unit(LUNAR_2025.greatest_jd)
_LUNAR_SUN_CORRECTION = _rotation_from_to(_LUNAR_MODEL_AT_MAX, _LUNAR_SUN_EXACT)


def _lunar_impact_components_deg() -> tuple[float, float]:
    anti_ra = (_LUNAR_SUN_RA_DEG+180.0) % 360.0
    anti_dec = -_LUNAR_SUN_DEC_DEG
    dra = ((_LUNAR_MOON_RA_DEG-anti_ra+180.0) % 360.0)-180.0
    east = dra*math.cos(math.radians(anti_dec))
    north = _LUNAR_MOON_DEC_DEG-anti_dec
    vector = np.array([east, north], dtype=float)
    vector *= _LUNAR_AXIS_DEG/np.linalg.norm(vector)
    return float(vector[0]), float(vector[1])


_LUNAR_IMPACT_EAST_DEG, _LUNAR_IMPACT_NORTH_DEG = _lunar_impact_components_deg()


def _lunar_contact_track_data() -> tuple[np.ndarray, np.ndarray]:
    b = _LUNAR_AXIS_DEG
    radii = {
        "P1": _LUNAR_P_RADIUS_DEG+_LUNAR_MOON_SD_DEG,
        "U1": _LUNAR_U_RADIUS_DEG+_LUNAR_MOON_SD_DEG,
        "U2": _LUNAR_U_RADIUS_DEG-_LUNAR_MOON_SD_DEG,
        "MAX": b,
        "U3": _LUNAR_U_RADIUS_DEG-_LUNAR_MOON_SD_DEG,
        "U4": _LUNAR_U_RADIUS_DEG+_LUNAR_MOON_SD_DEG,
        "P4": _LUNAR_P_RADIUS_DEG+_LUNAR_MOON_SD_DEG,
    }
    order = ("P1", "U1", "U2", "MAX", "U3", "U4", "P4")
    t = np.array([(datetime_to_jd(LUNAR_2025.contacts_utc[name])-LUNAR_2025.greatest_jd)*24.0
                  for name in order])
    x = []
    for name in order:
        if name == "MAX":
            x.append(0.0)
        else:
            value = math.sqrt(max(radii[name]**2-b**2, 0.0))
            x.append(-value if name in ("P1", "U1", "U2") else value)
    return t, np.asarray(x, dtype=float)


_LUNAR_TRACK_T_HR, _LUNAR_TRACK_X_DEG = _lunar_contact_track_data()
_LUNAR_TRACK_PCHIP = (PchipInterpolator(_LUNAR_TRACK_T_HR, _LUNAR_TRACK_X_DEG,
                                        extrapolate=False)
                      if PchipInterpolator is not None else None)


def lunar_track_x_deg(jd_utc: float | np.ndarray) -> np.ndarray:
    t = (np.asarray(jd_utc, dtype=float)-LUNAR_2025.greatest_jd)*24.0
    t_clip = np.clip(t, _LUNAR_TRACK_T_HR[0], _LUNAR_TRACK_T_HR[-1])
    if _LUNAR_TRACK_PCHIP is not None:
        return np.asarray(_LUNAR_TRACK_PCHIP(t_clip), dtype=float)
    return np.interp(t_clip, _LUNAR_TRACK_T_HR, _LUNAR_TRACK_X_DEG)


@dataclass(frozen=True)
class LunarReferenceState:
    jd_utc: float
    sun_hat_gcrf: np.ndarray
    anti_sun_hat_gcrf: np.ndarray
    moon_hat_gcrf: np.ndarray
    east_hat: np.ndarray
    north_hat: np.ndarray
    impact_vector_deg: np.ndarray
    track_x_deg: float
    shadow_separation_deg: float
    moon_gcrf_km: np.ndarray
    sun_gcrf_km: np.ndarray
    moon_distance_km: float
    sun_distance_km: float
    center_sun_visibility: float


def lunar_reference_state(jd_utc: float) -> LunarReferenceState:
    jd = float(jd_utc)
    sun_hat = _unit(_LUNAR_SUN_CORRECTION@_approx_sun_equatorial_unit(jd))
    anti = -sun_hat
    ra, dec = _unit_to_ra_dec(anti)
    ra_rad = math.radians(ra)
    dec_rad = math.radians(dec)
    east = _unit(np.array([-math.sin(ra_rad), math.cos(ra_rad), 0.0]))
    north = _unit(np.array([
        -math.sin(dec_rad)*math.cos(ra_rad),
        -math.sin(dec_rad)*math.sin(ra_rad),
        math.cos(dec_rad),
    ]))
    impact2 = np.array([_LUNAR_IMPACT_EAST_DEG, _LUNAR_IMPACT_NORTH_DEG])
    impact_hat2 = impact2/np.linalg.norm(impact2)
    track_hat2 = np.array([impact_hat2[1], -impact_hat2[0]])
    x = float(lunar_track_x_deg(jd))
    offset2 = impact2+x*track_hat2
    rho_deg = float(np.linalg.norm(offset2))
    tangent_world = _unit(offset2[0]*east+offset2[1]*north)
    rho = math.radians(rho_deg)
    moon_hat = _unit(math.cos(rho)*anti+math.sin(rho)*tangent_world)
    moon = _LUNAR_MOON_DISTANCE_KM*moon_hat
    sun = _LUNAR_SUN_DISTANCE_KM*sun_hat

    # Center-point direct sunlight, using the Danjon-adjusted effective Earth
    # radius required to reproduce NASA's umbral radius at the Moon.
    effective_earth_radius = math.sin(math.radians(_LUNAR_U_RADIUS_DEG))*_LUNAR_MOON_DISTANCE_KM \
        + _LUNAR_MOON_DISTANCE_KM*R_SUN_KM/_LUNAR_SUN_DISTANCE_KM
    to_earth = -moon
    to_sun = sun-moon
    sep = math.acos(np.clip(np.dot(_unit(to_earth), _unit(to_sun)), -1.0, 1.0))
    a_earth = math.asin(np.clip(effective_earth_radius/np.linalg.norm(to_earth), 0.0, 1.0))
    a_sun = math.asin(np.clip(R_SUN_KM/np.linalg.norm(to_sun), 0.0, 1.0))
    visible = float(angular_circle_visible_fraction(a_earth, a_sun, sep))
    return LunarReferenceState(
        jd_utc=jd,
        sun_hat_gcrf=sun_hat,
        anti_sun_hat_gcrf=anti,
        moon_hat_gcrf=moon_hat,
        east_hat=east,
        north_hat=north,
        impact_vector_deg=impact2,
        track_x_deg=x,
        shadow_separation_deg=rho_deg,
        moon_gcrf_km=moon,
        sun_gcrf_km=sun,
        moon_distance_km=_LUNAR_MOON_DISTANCE_KM,
        sun_distance_km=_LUNAR_SUN_DISTANCE_KM,
        center_sun_visibility=visible,
    )


@dataclass(frozen=True)
class ReferenceEvent:
    definition: ReferenceDefinition
    jd: np.ndarray
    moon_km: np.ndarray
    sun_km: np.ndarray
    frame: str
    backend: str
    center_visibility: np.ndarray
    separation_deg: np.ndarray
    metadata: Mapping[str, float | str]

    @property
    def mode(self) -> str:
        return self.definition.mode

    @property
    def contacts_jd(self) -> dict[str, float]:
        local = self.metadata.get("observer_contacts_jd")
        return dict(local) if isinstance(local, Mapping) else self.definition.contacts_jd

    @property
    def peak_index(self) -> int:
        return int(np.argmin(np.abs(self.jd-self.greatest_jd)))

    @property
    def greatest_jd(self) -> float:
        return float(self.metadata.get("local_max_jd", self.definition.greatest_jd))


def contact_aware_jd(definition: ReferenceDefinition, n_frames: int = 121) -> np.ndarray:
    """Timeline from first to last contact, retaining every published contact."""
    n_frames = max(int(n_frames), len(definition.contacts_utc))
    ordered = sorted(definition.contacts_jd.items(), key=lambda item: item[1])
    start, stop = ordered[0][1], ordered[-1][1]
    # Smooth baseline plus exact contact insertion.  A cosine redistribution
    # gives more samples around the middle without starving ingress/egress.
    u = np.linspace(0.0, 1.0, n_frames)
    shaped = 0.5-0.5*np.cos(np.pi*u)
    base = start+(stop-start)*shaped
    contact_values = np.array([value for _, value in ordered], dtype=float)
    supplemental = np.empty(0, dtype=float)
    if definition.mode == "solar":
        local = solar_local_contacts()
        # The global P1-P4 interval is several hours long while totality at
        # the fixed greatest-eclipse site lasts only 268 s.  Retain exact
        # C1-C4 and add symmetric samples around maximum so the animation
        # cannot skip over the central phase.
        offsets_s = np.array([-900, -600, -300, -120, -60, -20, 0,
                              20, 60, 120, 300, 600, 900], dtype=float)
        supplemental = np.concatenate([
            np.asarray(list(local.values()), dtype=float),
            definition.greatest_jd+offsets_s/86400.0,
        ])
    exact_values = np.concatenate([contact_values, supplemental])
    # Preserve published contacts bit-for-bit.  Remove only nearby baseline
    # samples instead of rounding the complete timeline.
    keep = np.ones(base.shape, dtype=bool)
    for value in exact_values:
        keep &= np.abs(base-value) > 0.05/86400.0
    all_times = np.concatenate([base[keep], exact_values])
    return np.sort(np.unique(all_times))


def build_reference_event(kind: str | ReferenceDefinition = "solar",
                          *, n_frames: int = 121,
                          jd: Sequence[float] | None = None) -> ReferenceEvent:
    """Build one of the two validated, real-world eclipse events."""
    if isinstance(kind, ReferenceDefinition):
        definition = kind
    else:
        key = str(kind).lower()
        if key in ("solar", "solar_2024", SOLAR_2024.key):
            definition = SOLAR_2024
        elif key in ("lunar", "lunar_2025", LUNAR_2025.key):
            definition = LUNAR_2025
        else:
            raise ValueError("kind must be 'solar'/'solar_2024' or 'lunar'/'lunar_2025'")
    times = contact_aware_jd(definition, n_frames=n_frames) if jd is None else np.asarray(jd, dtype=float)
    moon, sun, visible, separation = [], [], [], []
    if definition.mode == "solar":
        for value in times:
            state = solar_besselian_state(float(value))
            moon.append(state.moon_itrf_km)
            sun.append(state.sun_itrf_km)
            central = solar_central_line_wgs84(float(value))
            if central is None:
                # Geocentric apparent overlap is a useful partial-phase proxy.
                to_moon = state.moon_itrf_km
                to_sun = state.sun_itrf_km
                sep = math.acos(np.clip(np.dot(_unit(to_moon), _unit(to_sun)), -1.0, 1.0))
                a_m = math.asin(np.clip(SOLAR_UMBRA_OPTICAL_RADIUS_KM/np.linalg.norm(to_moon), 0.0, 1.0))
                a_s = math.asin(np.clip(R_SUN_KM/np.linalg.norm(to_sun), 0.0, 1.0))
                visible.append(float(angular_circle_visible_fraction(a_m, a_s, sep)))
                separation.append(math.degrees(sep))
            else:
                _, _, observer = central
                to_moon = state.moon_itrf_km-observer
                to_sun = state.sun_itrf_km-observer
                sep = math.acos(np.clip(np.dot(_unit(to_moon), _unit(to_sun)), -1.0, 1.0))
                a_m = math.asin(np.clip(SOLAR_UMBRA_OPTICAL_RADIUS_KM/np.linalg.norm(to_moon), 0.0, 1.0))
                a_s = math.asin(np.clip(R_SUN_KM/np.linalg.norm(to_sun), 0.0, 1.0))
                visible.append(float(angular_circle_visible_fraction(a_m, a_s, sep)))
                separation.append(math.degrees(sep))
        metadata = {
            "delta_t_s": _SOLAR_DELTA_T_S,
            "gamma": 0.3431,
            "magnitude": 1.0566,
            "path_width_at_greatest_km": 197.5,
            "central_duration_at_greatest_s": 268.1,
            "greatest_lat_deg": SOLAR_GREATEST_SITE_LAT_DEG,
            "greatest_lon_east_deg": SOLAR_GREATEST_SITE_LON_EAST_DEG,
            "umbra_optical_radius_km": SOLAR_UMBRA_OPTICAL_RADIUS_KM,
            "penumbra_optical_radius_km": SOLAR_PENUMBRA_OPTICAL_RADIUS_KM,
            **{f"local_{name.lower()}_utc": jd_to_datetime(value).isoformat().replace("+00:00", "Z")
               for name, value in solar_local_contacts().items()},
        }
        frame = "WGS84 Earth-fixed (Besselian reconstruction)"
        backend = "NASA/GSFC Besselian VSOP87/ELP2000-85"
    else:
        for value in times:
            state = lunar_reference_state(float(value))
            moon.append(state.moon_gcrf_km)
            sun.append(state.sun_gcrf_km)
            visible.append(state.center_sun_visibility)
            separation.append(state.shadow_separation_deg)
        geometric_umbra_deg = math.degrees(math.asin(
            (RE_KM-_LUNAR_MOON_DISTANCE_KM*(R_SUN_KM-RE_KM)/_LUNAR_SUN_DISTANCE_KM)
            /_LUNAR_MOON_DISTANCE_KM))
        enlargement = _LUNAR_U_RADIUS_DEG/geometric_umbra_deg
        metadata = {
            "penumbral_magnitude": _LUNAR_P_MAG,
            "umbral_magnitude": _LUNAR_U_MAG,
            "penumbra_radius_deg": _LUNAR_P_RADIUS_DEG,
            "umbra_radius_deg": _LUNAR_U_RADIUS_DEG,
            "axis_offset_deg": _LUNAR_AXIS_DEG,
            "moon_semidiameter_deg": _LUNAR_MOON_SD_DEG,
            "sun_semidiameter_deg": _LUNAR_SUN_SD_DEG,
            "moon_distance_km": _LUNAR_MOON_DISTANCE_KM,
            "sun_distance_km": _LUNAR_SUN_DISTANCE_KM,
            "danjon_shadow_enlargement": enlargement,
            "totality_duration_s": 3924.0,
        }
        frame = "GCRF-like apparent equatorial reference frame"
        backend = "NASA/GSFC contact and Danjon-shadow fit"
    return ReferenceEvent(
        definition=definition,
        jd=np.asarray(times, dtype=float),
        moon_km=np.asarray(moon, dtype=float),
        sun_km=np.asarray(sun, dtype=float),
        frame=frame,
        backend=backend,
        center_visibility=np.asarray(visible, dtype=float),
        separation_deg=np.asarray(separation, dtype=float),
        metadata=metadata,
    )



def solar_local_contact_aware_jd(
    n_frames: int = 65,
    *,
    lat_deg: float = SOLAR_GREATEST_SITE_LAT_DEG,
    lon_east_deg: float = SOLAR_GREATEST_SITE_LON_EAST_DEG,
    height_km: float = 0.0,
) -> np.ndarray:
    """Timeline for the fixed NASA greatest-eclipse site from C1 through C4.

    The four local contacts and greatest eclipse are retained exactly.  Extra
    samples cluster around C2/MAX/C3 so the 268-second total phase cannot be
    skipped even when a compact animation frame budget is requested.
    """
    contacts = solar_local_contacts(lat_deg, lon_east_deg, height_km)
    start, stop = contacts["C1"], contacts["C4"]
    n_frames = max(int(n_frames), 17)
    u = np.linspace(0.0, 1.0, n_frames)
    shaped = 0.5-0.5*np.cos(np.pi*u)
    base = start+(stop-start)*shaped
    offsets_s = np.array([-600, -300, -180, -120, -60, -20, 0,
                          20, 60, 120, 180, 300, 600], dtype=float)
    local_max = float(contacts.get("MAX", SOLAR_2024.greatest_jd))
    contact_values = [contacts[name] for name in ("C1", "C2", "MAX", "C3", "C4") if name in contacts]
    exact = np.concatenate([
        np.asarray(contact_values, dtype=float),
        np.asarray([SOLAR_2024.greatest_jd], dtype=float),
        local_max+offsets_s/86400.0,
    ])
    exact = exact[(exact >= start) & (exact <= stop)]
    keep = np.ones(base.shape, dtype=bool)
    for value in exact:
        keep &= np.abs(base-value) > 0.05/86400.0
    return np.sort(np.unique(np.concatenate([base[keep], exact])))


def build_solar_local_event(
    *, n_frames: int = 65,
    lat_deg: float = SOLAR_GREATEST_SITE_LAT_DEG,
    lon_east_deg: float = SOLAR_GREATEST_SITE_LON_EAST_DEG,
    height_km: float = 0.0,
    observer_name: str = "NASA greatest-eclipse reference site",
) -> "ReferenceEvent":
    """Build the 2024 eclipse as seen from NASA's greatest-eclipse site."""
    event = build_reference_event(SOLAR_2024, jd=solar_local_contact_aware_jd(
        n_frames, lat_deg=lat_deg, lon_east_deg=lon_east_deg, height_km=height_km
    ))
    metadata = dict(event.metadata)
    scope_name = (
        "NASA greatest-eclipse site"
        if observer_name == "NASA greatest-eclipse reference site"
        else observer_name
    )
    metadata.update({
        "animation_scope": f"{scope_name}, local C1-C4",
        "observer_name": observer_name,
        "observer_lat_deg": float(lat_deg),
        "observer_lon_east_deg": float(lon_east_deg),
        "observer_height_km": float(height_km),
        "observer_contacts_jd": solar_local_contacts(lat_deg, lon_east_deg, height_km),
        "local_max_jd": solar_local_contacts(lat_deg, lon_east_deg, height_km).get("MAX", SOLAR_2024.greatest_jd),
    })
    return replace(event, metadata=metadata)

def contact_label_at_jd(definition: ReferenceDefinition, jd: float,
                        tolerance_s: float = 0.7) -> str | None:
    for name, value in definition.contacts_jd.items():
        if abs(float(jd)-value)*86400.0 <= tolerance_s:
            return name
    return None


def reference_summary(event: ReferenceEvent) -> dict:
    result = {
        "key": event.definition.key,
        "title": event.definition.title,
        "mode": event.mode,
        "backend": event.backend,
        "frame": event.frame,
        "greatest_utc": event.definition.greatest_utc.isoformat().replace("+00:00", "Z"),
        "contacts_utc": {k: v.isoformat().replace("+00:00", "Z")
                         for k, v in event.definition.contacts_utc.items()},
        "metadata": dict(event.metadata),
        "frame_count": int(len(event.jd)),
    }
    return result