"""
Brightness of a Lambertian sphere in (or above) Earth's atmosphere.
====================================================================

Three public functions, all sharing the same geometry/spectral engine:

  * lambertian_reflection()        -- reflected light only
                                      (sunshine, earthshine, moonshine)
  * thermal_emission()             -- gray-body self-emission only
  * lambertian_sphere_brightness() -- both combined

Each returns in-band irradiance [W m^-2] and apparent AB magnitudes in an
arbitrary wavelength band (e.g. V, green, SWIR), including the Lambertian-
sphere phase function for every source, Earth umbra/penumbra for direct
sun, and Kasten & Young (1989) airmass extinction when the observer is
inside the atmosphere.

Every physical model parameter (solar constant, Earth/Moon albedos and
radii, OLR, effective temperatures, atmosphere top, AB zero point, ...)
is a keyword argument; the module-level values are only their defaults.

SSAPy consistency
-----------------
Everything position-related comes from SSAPy (LLNL), evaluated at an
`astropy.time.Time` (default provided):

  * Observer:  ssapy.EarthObserver.getRV(time)        -> GCRF meters
  * Sun:       ssapy.utils.sunPos(time)               -> GCRF meters
  * Moon:      ssapy.utils.moonPos(time)              -> GCRF meters
  * Zenith angle: astropy AltAz transform -- the same backend that
    ssapy.compute.altaz itself uses internally (SSAPy's altaz/quickAltAz
    APIs require an Orbit object rather than a bare position vector).

Geometry: all positions GCRS/GCRF, geocenter at the origin.  All phase
angles (Sun-object-observer, Sun-Earth-object, Earth-object-observer,
Sun-Moon-object, Moon-object-observer) come from the full 3-D vectors.

Spectral model
--------------
Planck spectra throughout:

  * Object thermal:   B_lambda(T_object) x emissivity (gray)
  * Solar-spectrum components (sun, moonshine, shortwave earthshine):
        t_sun (5772 K) blackbody scaled to the local total solar
        irradiance (good to ~5% across VIS-SWIR; solar/telluric
        absorption features are not modeled)
  * Earth longwave:   t_earth_lw (255 K) blackbody scaled to the OLR

Magnitudes are on the AB system, from the band-averaged flux density:
m_AB = -2.5 log10(<f_nu> / 3631 Jy).  For the default V band this agrees
with Johnson V to a few hundredths of a magnitude.
"""

from __future__ import annotations

import functools
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    GCRS, AltAz, EarthLocation, get_sun, get_body, CartesianRepresentation,
)

from ssatk.constants import SOLAR_FLUX_1_AU
from ._visibility import line_of_sight_blocked

# ----------------------------------------------------------------------
# Fundamental constants (SI; not model parameters)
# ----------------------------------------------------------------------
SIGMA_SB = 5.670374419e-8              # Stefan-Boltzmann [W m^-2 K^-4]
H_PLANCK = 6.62607015e-34              # [J s]
C_LIGHT  = 2.99792458e8                # [m s^-1]
K_BOLTZ  = 1.380649e-23                # [J K^-1]
AU_M     = 149_597_870_700.0           # [m]

# ----------------------------------------------------------------------
# Default model parameters -- every one of these is a kwarg of the
# public functions below; override there, not here.
# ----------------------------------------------------------------------
SOLAR_CONST      = SOLAR_FLUX_1_AU     # TSI at 1 AU [W m^-2]
T_SUN            = 5772.0              # solar effective temperature [K]
R_SUN            = 6.957e8             # solar radius [m]
R_EARTH          = 6_378_137.0         # Earth equatorial radius [m]
ALBEDO_EARTH     = 0.306               # Earth Bond albedo
OLR_EARTH        = 239.0               # mean outgoing longwave flux [W m^-2]
T_EARTH_LW       = 255.0               # effective temp for OLR spectrum [K]
R_MOON           = 1_737_400.0         # Moon radius [m]
ALBEDO_MOON      = 0.12                # Moon albedo (Lambertian approx.)
ATMOSPHERE_TOP_M = 100_000.0           # Karman line: above this, no extinction
F_NU_AB_ZERO     = 3.631e-23           # AB zero point, 3631 Jy [W m^-2 Hz^-1]

DEFAULT_TIME = Time("2026-06-09T00:00:00", scale="utc")

# Named pass bands: (lambda_lo, lambda_hi) in meters.  Approximate,
# top-hat representations of common filters / detector windows.
BANDS = {
    "U":     (330e-9, 400e-9),
    "B":     (390e-9, 490e-9),
    "blue":  (450e-9, 495e-9),
    "green": (500e-9, 565e-9),
    "V":     (500e-9, 600e-9),
    "red":   (620e-9, 750e-9),
    "R":     (560e-9, 720e-9),
    "I":     (700e-9, 900e-9),
    "NIR":   (750e-9, 1.0e-6),
    "J":     (1.10e-6, 1.35e-6),
    "H":     (1.50e-6, 1.80e-6),
    "K":     (2.00e-6, 2.40e-6),
    "SWIR":  (0.9e-6, 1.7e-6),         # typical InGaAs window
    "SWIR2": (1.7e-6, 2.5e-6),
    "MWIR":  (3.0e-6, 5.0e-6),
    "LWIR":  (8.0e-6, 14.0e-6),
}

# ----------------------------------------------------------------------
# (N,3)-position support: let the scalar brightness functions also accept
# an array of GCRS positions (consistent with ssapy.rv output) plus a
# Time array, looping internally and returning a dict of (N,) arrays.
# `time` MUST be passed as a keyword -- positional arg 2 is `observer`.
# ----------------------------------------------------------------------
def _stack_dicts(dicts):
    """Merge identically-structured result dicts; scalar leaves -> (N,) arrays."""
    out = {}
    for k, v in dicts[0].items():
        out[k] = (_stack_dicts([d[k] for d in dicts]) if isinstance(v, dict)
                  else np.array([d[k] for d in dicts]))
    return out


def _accept_position_array(func):
    """Decorator: if obj_pos_gcrs_m is (N,3), loop the scalar core over rows
    (broadcasting a scalar Time or indexing a Time array) and stack the
    per-epoch result dicts into one dict of (N,) arrays. A single (3,)
    position falls straight through to the original scalar path."""
    @functools.wraps(func)
    def wrapper(obj_pos_gcrs_m, *args, **kwargs):
        pos = np.asarray(obj_pos_gcrs_m, dtype=float)
        if pos.ndim < 2:                       # (3,) -> unchanged scalar path
            return func(obj_pos_gcrs_m, *args, **kwargs)
        time = kwargs.get("time", DEFAULT_TIME)
        scalar_t = getattr(time, "isscalar", np.ndim(time) == 0)
        results = []
        for i in range(pos.shape[0]):
            kw = dict(kwargs)
            if not scalar_t:
                kw["time"] = time[i]
            results.append(func(pos[i], *args, **kw))
        return _stack_dicts(results)
    return wrapper

# ----------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------
def _unit(v):
    return v / np.linalg.norm(v)


def _angle_between(a, b):
    """Angle [rad] between vectors a and b, numerically safe."""
    c = np.dot(_unit(a), _unit(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def lambert_sphere_phase(alpha):
    """
    Phase function of a Lambertian sphere, normalized so that the flux
    scattered toward the observer is

        F = E_inc * A * (R/d)^2 * p(alpha),     p(0) = 2/3,

    with alpha the phase angle [rad] at the object between the illumination
    source and the observer.
    """
    return (2.0 / (3.0 * np.pi)) * (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha))


# ----------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------
def _resolve_band(band):
    """Return (lam_lo, lam_hi) in meters from a name or a 2-sequence.

    Numeric entries may be floats in meters or astropy length Quantities.
    """
    if isinstance(band, str):
        try:
            return BANDS[band]
        except KeyError:
            raise ValueError(
                f"Unknown band '{band}'. Known: {sorted(BANDS)} "
                "or pass (lam_lo, lam_hi) in meters."
            ) from None
    lo, hi = band
    if isinstance(lo, u.Quantity):
        lo = lo.to_value(u.m)
    if isinstance(hi, u.Quantity):
        hi = hi.to_value(u.m)
    lo, hi = float(lo), float(hi)
    if not (0 < lo < hi):
        raise ValueError("Band must satisfy 0 < lam_lo < lam_hi (meters).")
    return lo, hi


def _planck_band_radiance(T, lam_lo, lam_hi, n=2000):
    """In-band blackbody radiance  integral of B_lambda  [W m^-2 sr^-1].

    Integrated on a logarithmic wavelength grid: a linear 600-point grid
    under-sampled the peak of any band spanning more than a decade, so
    0.01 um - 1 mm recovered only 25 % of sigma T^4 / pi at 5772 K.
    """
    lam = np.geomspace(lam_lo, lam_hi, n)
    with np.errstate(over="ignore"):
        x = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * max(T, 1e-3))
        B = (2 * H_PLANCK * C_LIGHT**2 / lam**5) / np.expm1(np.clip(x, None, 700.0))
    return float(np.trapezoid(B, lam))


def _planck_band_fraction(T, lam_lo, lam_hi):
    """Fraction of total blackbody power emitted within [lam_lo, lam_hi]."""
    return _planck_band_radiance(T, lam_lo, lam_hi) / (SIGMA_SB * T**4 / np.pi)


# ----------------------------------------------------------------------
# Shadow (umbra / penumbra) factor for direct sunlight
# ----------------------------------------------------------------------
def sun_visibility_factor(r_obj, r_sun, r_earth=R_EARTH, r_sun_radius=R_SUN):
    """
    Fraction of the solar disk visible from the object, occulted by Earth.
    1 = full sun, 0 = umbra, linear interpolation across the penumbra
    (adequate to a few percent of the penumbral contribution).
    """
    d_sun = np.linalg.norm(r_sun - r_obj)
    theta = _angle_between(-r_obj, r_sun - r_obj)
    ang_earth = np.arcsin(min(1.0, r_earth / np.linalg.norm(r_obj)))
    ang_sun   = np.arcsin(min(1.0, r_sun_radius / d_sun))
    if theta >= ang_earth + ang_sun:        # no overlap
        return 1.0
    if theta <= ang_earth - ang_sun:        # total eclipse (umbra)
        return 0.0
    return (theta - (ang_earth - ang_sun)) / (2.0 * ang_sun)  # penumbra


# ----------------------------------------------------------------------
# Airmass / extinction
# ----------------------------------------------------------------------
def airmass_kasten_young(zenith_deg):
    """Kasten & Young (1989) relative airmass; valid to the horizon."""
    z = np.clip(zenith_deg, 0.0, 90.0)
    return 1.0 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** (-1.6364))


# ----------------------------------------------------------------------
# Shared geometry / extinction / ephemeris setup
# ----------------------------------------------------------------------
def _setup(
    obj_pos_gcrs_m, observer, time, band, k_extinction,
    lon, lat, elevation, r_earth, atmosphere_top_m,
    *, check_line_of_sight=True,
):
    """Resolve observer, ephemerides, band, airmass.  Returns a dict."""
    r_obj = np.asarray(obj_pos_gcrs_m, dtype=float)
    lam_lo, lam_hi = _resolve_band(band)

    try:
        from ssapy import EarthObserver as _SSAPyEarthObserver
    except ImportError:  # ssapy not installed; EarthLocation path still works
        _SSAPyEarthObserver = ()

    if observer is None:
        if lon is None or lat is None:
            raise ValueError(
                "Provide either `observer` or both `lon` and `lat` (degrees)."
            )
        if not _SSAPyEarthObserver:
            raise ImportError("lon/lat observer construction requires ssapy.")
        observer = _SSAPyEarthObserver(lon=lon, lat=lat, elevation=elevation)

    if _SSAPyEarthObserver and isinstance(observer, _SSAPyEarthObserver):
        r_obs, _v_obs = observer.getRV(time)        # SSAPy GCRF position [m]
        r_obs = np.asarray(r_obs, dtype=float).ravel()
        loc = EarthLocation(
            lon=observer.lon * u.deg,
            lat=observer.lat * u.deg,
            height=observer.elevation * u.m,
        )
        # Zenith angle of the object: same astropy AltAz backend that
        # ssapy.compute.altaz itself uses internally (SSAPy's altaz/quickAltAz
        # APIs require an Orbit, not a bare position vector).
        obj_coord = GCRS(CartesianRepresentation(r_obj * u.m), obstime=time)
        altaz = obj_coord.transform_to(AltAz(obstime=time, location=loc))
        zenith_deg = 90.0 - altaz.alt.to_value(u.deg)
        observer_alt_m = loc.height.to_value(u.m)
        below_horizon = altaz.alt.to_value(u.deg) <= 0.0
    elif isinstance(observer, EarthLocation):
        r_obs = observer.get_gcrs(obstime=time).cartesian.xyz.to_value(u.m)
        obj_coord = GCRS(CartesianRepresentation(r_obj * u.m), obstime=time)
        altaz = obj_coord.transform_to(AltAz(obstime=time, location=observer))
        zenith_deg = 90.0 - altaz.alt.to_value(u.deg)
        observer_alt_m = observer.height.to_value(u.m)
        below_horizon = altaz.alt.to_value(u.deg) <= 0.0
    else:
        r_obs = np.asarray(observer, dtype=float)
        observer_alt_m = np.linalg.norm(r_obs) - r_earth
        zenith_deg = np.degrees(_angle_between(r_obs, r_obj - r_obs))
        below_horizon = check_line_of_sight and line_of_sight_blocked(r_obj, r_obs, r_earth)

    if observer_alt_m < atmosphere_top_m and not below_horizon:
        X = airmass_kasten_young(zenith_deg)
        extinction_mag = k_extinction * X
    else:
        X = 0.0
        extinction_mag = 0.0

    # Sun and Moon from SSAPy (GCRF, meters); astropy fallback only if
    # SSAPy is unavailable.
    try:
        from ssapy.utils import sunPos, moonPos
        r_sun  = np.asarray(sunPos(time),  dtype=float).ravel()
        r_moon = np.asarray(moonPos(time), dtype=float).ravel()
    except ImportError:
        r_sun  = get_sun(time).cartesian.xyz.to_value(u.m)
        r_moon = get_body("moon", time).cartesian.xyz.to_value(u.m)

    return {
        "r_obj": r_obj, "r_obs": r_obs, "r_sun": r_sun, "r_moon": r_moon,
        "d_obs": np.linalg.norm(r_obj - r_obs),
        "lam_lo": lam_lo, "lam_hi": lam_hi,
        "band_name": band if isinstance(band, str) else "custom",
        "airmass": X, "extinction_mag": extinction_mag,
        "trans": 0.0 if below_horizon else 10 ** (-0.4 * extinction_mag),
        "below_horizon": bool(below_horizon),
    }


def _ab_mag(F_band, lam_lo, lam_hi, f_nu_ab_zero=F_NU_AB_ZERO):
    """AB magnitude from in-band irradiance via band-averaged f_nu."""
    if F_band <= 0:
        return np.inf
    dnu = C_LIGHT / lam_lo - C_LIGHT / lam_hi
    return -2.5 * np.log10((F_band / dnu) / f_nu_ab_zero)


def _package(g, comp_bolo, comp_band, angles, time, f_nu_ab_zero):
    """Assemble the common output dict from components."""
    F_bolo_total = sum(comp_bolo.values())
    F_band_total = sum(comp_band.values())
    mag = lambda F: _ab_mag(F, g["lam_lo"], g["lam_hi"], f_nu_ab_zero)
    m_ab = mag(F_band_total)
    observed_flux = 0.0 if g["below_horizon"] else F_band_total * g["trans"]
    return {
        "time": time.isot,
        "band": {"name": g["band_name"],
                 "lam_lo_m": g["lam_lo"], "lam_hi_m": g["lam_hi"]},
        "range_m": g["d_obs"],
        "airmass": g["airmass"],
        "extinction_mag": g["extinction_mag"],
        "object_below_horizon": g["below_horizon"],
        "irradiance_bolometric_W_m2": comp_bolo,
        "irradiance_bolometric_total_W_m2": F_bolo_total,
        "irradiance_inband_W_m2": comp_band,
        "irradiance_inband_total_W_m2": F_band_total,
        "irradiance_inband_total_at_observer_W_m2": observed_flux,
        "ab_mag_components": {k: mag(v) for k, v in comp_band.items()},
        "ab_mag_exoatmospheric": m_ab,
        "ab_mag_observed": mag(observed_flux),
        "angles_deg": angles,
    }


# ======================================================================
# 1) Reflected light only
# ======================================================================
@_accept_position_array
def lambertian_reflection(
    obj_pos_gcrs_m,
    observer=None,
    radius_m=1.0,
    albedo=0.3,
    time=DEFAULT_TIME,
    band="V",
    k_extinction=0.16,
    include_sun=True,
    include_earthshine=True,
    include_moonshine=True,
    lon=None,
    lat=None,
    elevation=0.0,
    # ------- physical model parameters (defaults = module constants) ----
    solar_const=SOLAR_CONST,
    t_sun=T_SUN,
    r_sun_radius=R_SUN,
    r_earth=R_EARTH,
    albedo_earth=ALBEDO_EARTH,
    olr_earth=OLR_EARTH,
    t_earth_lw=T_EARTH_LW,
    r_moon=R_MOON,
    albedo_moon=ALBEDO_MOON,
    atmosphere_top_m=ATMOSPHERE_TOP_M,
    f_nu_ab_zero=F_NU_AB_ZERO,
    _geo=None,
):
    """
    Light *reflected* by a Lambertian sphere: direct sunshine, earthshine
    (shortwave + Earth-thermal longwave), and moonshine.  No self-emission.

    Parameters mirror `lambertian_sphere_brightness` (see its docstring);
    every physical model parameter (solar constant, albedos, radii, OLR,
    effective temperatures, atmosphere top, AB zero point) is a kwarg.

    Returns the standard output dict with components
    'sun', 'earthshine', 'moonshine'.
    """
    g = _geo or _setup(obj_pos_gcrs_m, observer, time, band, k_extinction,
                       lon, lat, elevation, r_earth, atmosphere_top_m)
    r_obj, r_obs = g["r_obj"], g["r_obs"]
    r_sun, r_moon_v = g["r_sun"], g["r_moon"]
    lam_lo, lam_hi = g["lam_lo"], g["lam_hi"]

    geom = radius_m**2 / g["d_obs"]**2                    # (R/d)^2 to observer
    d_sun_obj = np.linalg.norm(r_sun - r_obj)
    F_sun_at_obj = solar_const * (AU_M / d_sun_obj) ** 2  # local solar constant

    frac_solar = _planck_band_fraction(t_sun, lam_lo, lam_hi)
    frac_olr   = _planck_band_fraction(t_earth_lw, lam_lo, lam_hi)

    comp_bolo, comp_band, angles = {}, {}, {}

    if include_sun:
        alpha = _angle_between(r_sun - r_obj, r_obs - r_obj)   # phase angle
        vis = sun_visibility_factor(r_obj, r_sun, r_earth, r_sun_radius)
        F = vis * F_sun_at_obj * albedo * geom * lambert_sphere_phase(alpha)
        comp_bolo["sun"] = F
        comp_band["sun"] = F * frac_solar
        angles["phase_sun_obj_obs_deg"] = np.degrees(alpha)
        angles["sun_visibility"] = vis

    if include_earthshine:
        d_earth_obj = np.linalg.norm(r_obj)
        beta = _angle_between(r_sun, r_obj)        # Sun-Earth-object angle
        E_es_sw = (F_sun_at_obj * albedo_earth
                   * (r_earth / d_earth_obj) ** 2
                   * lambert_sphere_phase(beta))
        E_es_lw = olr_earth * (r_earth / d_earth_obj) ** 2   # phase-independent
        gamma = _angle_between(-r_obj, r_obs - r_obj)  # Earth-object-observer
        p_g = lambert_sphere_phase(gamma)
        F_sw = E_es_sw * albedo * geom * p_g
        F_lw = E_es_lw * albedo * geom * p_g           # gray albedo in LW too
        comp_bolo["earthshine"] = F_sw + F_lw
        comp_band["earthshine"] = F_sw * frac_solar + F_lw * frac_olr
        angles["phase_sun_earth_obj_deg"] = np.degrees(beta)
        angles["phase_earth_obj_obs_deg"] = np.degrees(gamma)

    if include_moonshine:
        d_moon_obj = np.linalg.norm(r_moon_v - r_obj)
        F_sun_at_moon = solar_const * (AU_M / np.linalg.norm(r_sun - r_moon_v))**2
        delta = _angle_between(r_sun - r_moon_v, r_obj - r_moon_v)  # Sun-Moon-obj
        E_ms = (F_sun_at_moon * albedo_moon
                * (r_moon / d_moon_obj) ** 2
                * lambert_sphere_phase(delta))
        eps = _angle_between(r_moon_v - r_obj, r_obs - r_obj)       # Moon-obj-obs
        F = E_ms * albedo * geom * lambert_sphere_phase(eps)
        comp_bolo["moonshine"] = F
        comp_band["moonshine"] = F * frac_solar
        angles["phase_sun_moon_obj_deg"] = np.degrees(delta)
        angles["phase_moon_obj_obs_deg"] = np.degrees(eps)

    return _package(g, comp_bolo, comp_band, angles, time, f_nu_ab_zero)


# ======================================================================
# 2) Thermal self-emission only
# ======================================================================
@_accept_position_array
def thermal_emission(
    obj_pos_gcrs_m,
    observer=None,
    temperature_K=300.0,
    radius_m=1.0,
    albedo=0.3,
    emissivity=None,
    time=DEFAULT_TIME,
    band="V",
    k_extinction=0.16,
    lon=None,
    lat=None,
    elevation=0.0,
    # ------- physical model parameters (defaults = module constants) ----
    r_earth=R_EARTH,
    atmosphere_top_m=ATMOSPHERE_TOP_M,
    f_nu_ab_zero=F_NU_AB_ZERO,
    _geo=None,
):
    """
    Gray-body thermal *self-emission* of the sphere.  No reflected light.

    emissivity : float, optional
        Gray emissivity.  Defaults to Kirchhoff's law, 1 - albedo.

    Other parameters mirror `lambertian_sphere_brightness`.  Returns the
    standard output dict with component 'thermal'.
    """
    g = _geo or _setup(obj_pos_gcrs_m, observer, time, band, k_extinction,
                       lon, lat, elevation, r_earth, atmosphere_top_m)
    if emissivity is None:
        emissivity = 1.0 - albedo

    geom = radius_m**2 / g["d_obs"]**2
    # Isotropic sphere: F = eps*sigma*T^4 * 4*pi*R^2 / (4*pi*d^2)
    F_th = emissivity * SIGMA_SB * temperature_K**4 * geom
    frac_obj = _planck_band_fraction(temperature_K, g["lam_lo"], g["lam_hi"])

    return _package(
        g, {"thermal": F_th}, {"thermal": F_th * frac_obj}, {},
        time, f_nu_ab_zero,
    )


# ======================================================================
# 3) Combined: reflection + emission
# ======================================================================
@_accept_position_array
def lambertian_sphere_brightness(
    obj_pos_gcrs_m,
    observer=None,
    temperature_K=300.0,
    radius_m=1.0,
    albedo=0.3,
    emissivity=None,
    time=DEFAULT_TIME,
    band="V",
    k_extinction=0.16,
    include_sun=True,
    include_earthshine=True,
    include_moonshine=True,
    include_thermal=True,
    lon=None,
    lat=None,
    elevation=0.0,
    # ------- physical model parameters (defaults = module constants) ----
    solar_const=SOLAR_CONST,
    t_sun=T_SUN,
    r_sun_radius=R_SUN,
    r_earth=R_EARTH,
    albedo_earth=ALBEDO_EARTH,
    olr_earth=OLR_EARTH,
    t_earth_lw=T_EARTH_LW,
    r_moon=R_MOON,
    albedo_moon=ALBEDO_MOON,
    atmosphere_top_m=ATMOSPHERE_TOP_M,
    f_nu_ab_zero=F_NU_AB_ZERO,
):
    """
    Total brightness of a Lambertian, gray-body sphere near Earth in a
    chosen band: reflected light (via `lambertian_reflection`) plus
    thermal self-emission (via `thermal_emission`).

    Parameters
    ----------
    obj_pos_gcrs_m : array-like (3,)
        Object position in GCRS/GCRF, meters from the geocenter.
    observer : ssapy.EarthObserver, EarthLocation, or array-like (3,), optional
        Observer as an `ssapy.EarthObserver` (GCRF position from
        `EarthObserver.getRV(time)`), an astropy `EarthLocation`, or a raw
        GCRS position vector in meters (e.g. another spacecraft; extinction
        applies only below `atmosphere_top_m`).  May be omitted if `lon`
        and `lat` are given.
    lon, lat, elevation : float, optional
        Geodetic site (degrees East-positive, degrees, meters); builds an
        `ssapy.EarthObserver` internally when `observer` is None.
    temperature_K, radius_m, albedo : float
        Sphere temperature [K], radius [m], gray Bond/Lambert albedo (0-1).
    emissivity : float, optional
        Gray emissivity; defaults to 1 - albedo (Kirchhoff).
    time : astropy.time.Time
        Epoch for the SSAPy ephemerides (default 2026-06-09 00:00 UTC).
    band : str or (float, float)
        Named key from `BANDS` (e.g. 'V', 'green', 'SWIR', 'J', 'LWIR') or
        explicit (lam_lo, lam_hi) in meters (Quantities accepted).
    k_extinction : float
        Extinction coefficient [mag/airmass] *for the chosen band*.
    include_* : bool
        Toggle individual components.

    Model parameters : float
        ``solar_const``, ``t_sun``, ``r_sun_radius``, ``r_earth``,
        ``albedo_earth``, ``olr_earth``, ``t_earth_lw``, ``r_moon``,
        ``albedo_moon``, ``atmosphere_top_m``, and ``f_nu_ab_zero`` default to
        the module-level constants.

    Returns
    -------
    dict : in-band and bolometric irradiances per component ('sun',
        'earthshine', 'moonshine', 'thermal') and totals [W m^-2], AB
        magnitudes per component and total (exoatmospheric and observed),
        airmass, extinction, range, band, and all phase angles [deg].
    """
    g = _setup(obj_pos_gcrs_m, observer, time, band, k_extinction,
               lon, lat, elevation, r_earth, atmosphere_top_m)

    comp_bolo, comp_band, angles = {}, {}, {}

    if include_sun or include_earthshine or include_moonshine:
        refl = lambertian_reflection(
            obj_pos_gcrs_m, radius_m=radius_m, albedo=albedo, time=time,
            band=band, include_sun=include_sun,
            include_earthshine=include_earthshine,
            include_moonshine=include_moonshine,
            solar_const=solar_const, t_sun=t_sun, r_sun_radius=r_sun_radius,
            r_earth=r_earth, albedo_earth=albedo_earth, olr_earth=olr_earth,
            t_earth_lw=t_earth_lw, r_moon=r_moon, albedo_moon=albedo_moon,
            f_nu_ab_zero=f_nu_ab_zero, _geo=g,
        )
        comp_bolo.update(refl["irradiance_bolometric_W_m2"])
        comp_band.update(refl["irradiance_inband_W_m2"])
        angles.update(refl["angles_deg"])

    if include_thermal:
        th = thermal_emission(
            obj_pos_gcrs_m, temperature_K=temperature_K, radius_m=radius_m,
            albedo=albedo, emissivity=emissivity, time=time, band=band,
            f_nu_ab_zero=f_nu_ab_zero, _geo=g,
        )
        comp_bolo.update(th["irradiance_bolometric_W_m2"])
        comp_band.update(th["irradiance_inband_W_m2"])

    return _package(g, comp_bolo, comp_band, angles, time, f_nu_ab_zero)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t = DEFAULT_TIME
    from ssapy import EarthObserver

    # Haleakala site; sphere 800 km directly overhead.
    site = EarthObserver(lon=-156.26, lat=20.71, elevation=3055.0)
    r_site, _ = site.getRV(t)
    r_site = np.asarray(r_site, dtype=float).ravel()
    obj = r_site * (np.linalg.norm(r_site) + 800e3) / np.linalg.norm(r_site)

    common = dict(
        obj_pos_gcrs_m=obj,
        lon=-156.26, lat=20.71, elevation=3055.0,
        radius_m=1.0, albedo=0.30, time=t,
    )

    print("Combined, several bands:")
    for b, k in [("V", 0.16), ("green", 0.18), ("SWIR", 0.08), ("LWIR", 0.0)]:
        out = lambertian_sphere_brightness(
            temperature_K=290.0, band=b, k_extinction=k, **common)
        bi = out["band"]
        print(f"  {bi['name']:>6} m_AB(exo)={out['ab_mag_exoatmospheric']:7.3f} "
              f"m_AB(obs)={out['ab_mag_observed']:7.3f}  "
              + str({kk: round(vv, 2)
                     for kk, vv in out['ab_mag_components'].items()}))

    print("\nReflection only (V):")
    r = lambertian_reflection(band="V", **common)
    print(f"  m_AB(obs) = {r['ab_mag_observed']:.3f}")

    print("Emission only (LWIR, 290 K):")
    e = thermal_emission(temperature_K=290.0, band="LWIR",
                         k_extinction=0.0, **common)
    print(f"  m_AB(obs) = {e['ab_mag_observed']:.3f}")

    # Consistency: reflection + emission == combined (in-band totals)
    cV = lambertian_sphere_brightness(temperature_K=290.0, band="V",
                                      k_extinction=0.16, **common)
    rV = lambertian_reflection(band="V", k_extinction=0.16, **common)
    eV = thermal_emission(temperature_K=290.0, band="V",
                          k_extinction=0.16, **common)
    s = (rV["irradiance_inband_total_W_m2"] + eV["irradiance_inband_total_W_m2"])
    assert np.isclose(s, cV["irradiance_inband_total_W_m2"], rtol=1e-12)
    print("\nConsistency check passed: reflection + emission == combined")
