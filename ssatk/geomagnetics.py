"""
ssatk/geomagnetics.py
------------------------------
Geomagnetic field, radiation-belt and magnetopause physics.

Extracted verbatim from ssatk/plots/magfield_plot_3d.py. The reason
is concrete rather than stylistic: all three magfield_verification_*.py
scripts already imported the *plot* module purely to reach its private
physics helpers (_bfield_batch, _true_magnetic_equator, _aep8_lookup,
get_solar_wind), so validating the physics meant importing Plotly and the
whole rendering stack.

Layering
--------
    time_functions/               julian_date, gmst -- no dependencies
            |
    geomagnetics.py               (this file) ppigrf + geopack -- the physics
            |
    plots/magnetosphere_core.py   geometry, meshes, textures -- no heavy deps
            |
    plots/magfield_plot_3d.py     plotly -- rendering only
    plots/van_allen_plot_3d.py

Every arrow points one way, downward, and that is load-bearing rather than
tidy. This module previously sat *below* magnetosphere_core and imported the
shared constants and geometry upward from it. Reaching into ssatk.plots
runs plots/__init__, which auto-imports every module in the package, including
magfield_plot_3d, which imports back into this module while it is still
half-initialised. `import ssatk.geomagnetics` failed outright as a
result -- while `import ssatk.plots` happened to work, because it
entered the cycle from the other side. Nothing caught it: every test imported
the plots package first.

The constants and geometry (EARTH_RADIUS_KM, the WGS84 axes, _dipole_axis,
_mag_basis, _subsolar_point, _texture_cache_dir) are therefore defined here,
and magnetosphere_core re-exports them under the same names so existing
callers and tests are unaffected.

Note EARTH_RADIUS_KM below is 6371.0, the spherical mean radius -- NOT
ssatk.constants.EARTH_RADIUS_KM, which is the 6378.137 WGS84
equatorial radius. L shells are expressed in the spherical mean, so
interchanging them shifts every field line by 0.1%. The names collide; the
quantities differ.

magnetosphere_core stays dependency-light on purpose: it documents "nothing
here depends on ppigrf, geopack or spacepy, so it imports cleanly", and
folding heavy-dependency physics in would destroy that property.

Not auto-imported: ssatk/__init__.py has every submodule import
commented out, so `import ssatk` does not drag in ppigrf/geopack
(which cost real time loading IGRF coefficients, and are absent for users who
never touch magnetic fields).
"""

from __future__ import annotations

import functools as _functools
from datetime import datetime

import os
from pathlib import Path

import numpy as np

from .constants import (
    EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM,
    EARTH_MEAN_RADIUS_KM,
    EARTH_OBLIQUITY_J2000_DEG,
    WGS84_A_KM,
    WGS84_B_KM,
)

try:
    import ppigrf.ppigrf as _pp
    _HAS_PPIGRF = True
except Exception:
    # Bind the name anyway: modules that re-export the physics import _pp by
    # name, and `from .geomagnetics import _pp` must not fail merely because
    # ppigrf is absent. Guard uses with _HAS_PPIGRF, as the code already does.
    _pp = None
    _HAS_PPIGRF = False

try:
    from geopack import geopack as _gp, t89 as _t89, t96 as _t96
    _HAS_GEOPACK = True
except Exception:
    _HAS_GEOPACK = False

try:
    from .time_functions.gmst import _gmst_rad
    from .time_functions.julian_date import _julian_date
except ImportError:  # script mode
    from ssatk.time_functions.gmst import _gmst_rad
    from ssatk.time_functions.julian_date import _julian_date

# ── Shared constants and geometry ────────────────────────────────────────────
#
# These used to live in plots/magnetosphere_core.py and were imported from
# here, which pointed the dependency the wrong way: the physics layer reached
# up into the plotting package, whose __init__ auto-imports every module in it,
# including one that imports back into this module. `import
# ssatk.geomagnetics` failed outright as a result.
#
# They are defined here now and magnetosphere_core imports them back, so the
# arrow runs physics -> plotting and the cycle cannot form. Nothing in this
# block needs numpy beyond what is already imported, and none of it renders
# anything.

EARTH_RADIUS_KM = EARTH_MEAN_RADIUS_KM
"""Spherical mean Earth radius, km.

Deliberately NOT ssatk.constants.EARTH_RADIUS_KM, which is 6378.137 --
the WGS84 equatorial radius. Magnetospheric work uses the spherical mean: L
shells are expressed in it, and swapping the two shifts every field line and
L value by 0.1%. The two names collide, so they are kept in separate modules
on purpose; use this one for anything magnetic.
"""

_DIPOLE_TILT_DEG = 9.6
_DIPOLE_LON_DEG = -72.0


def _texture_cache_dir():
    """Directory for cached grids and downloaded textures, created on demand."""
    from ssatk._paths import cache_env_dir

    d = Path(cache_env_dir() or str(Path.home() / ".cache" / "ssatk"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dipole_axis():
    """Unit vector along the geomagnetic dipole axis, in geographic frame."""
    tilt = np.radians(_DIPOLE_TILT_DEG)
    lon = np.radians(_DIPOLE_LON_DEG)
    return np.array([np.sin(tilt) * np.cos(lon), np.sin(tilt) * np.sin(lon), np.cos(tilt)])


def _mag_basis(axis):
    """Right-handed basis with `axis` as the third vector.

    The degenerate case matters: when the dipole axis is parallel to z the
    first cross product vanishes, so fall back to crossing with x instead.
    """
    z = np.array([0., 0., 1.])
    e1 = np.cross(axis, z)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, np.array([1., 0., 0.]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    return axis, e1, e2


def _subsolar_point(date):
    """Subsolar latitude/longitude in degrees (Meeus low-precision Sun).

    Includes the equation of centre and the equation of time. A mean-Sun
    approximation that drops them is ~2.7 deg off, which is ~300 km of
    terminator position -- visible against a coastline.
    """
    jd = _julian_date(date)
    n = jd - 2451545.0
    Lm = (280.460 + 0.9856474 * n) % 360.0                        # mean longitude
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)             # mean anomaly
    lam = np.radians(Lm + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))   # ecliptic longitude
    eps = np.radians(EARTH_OBLIQUITY_J2000_DEG - 3.6e-7 * n)     # obliquity
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)))
    gmst_deg = np.degrees(_gmst_rad(date))
    lon = ((ra - gmst_deg + 180.0) % 360.0) - 180.0
    return dec, lon

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_EXTERNAL_MODEL = None
_T89_CACHE = {}
_AEP8_TABLE = None

_M_DIPOLE_NT_RE3 = 31000.0 / (1.0143 ** 3)

_OMNI_COLS = dict(doy=1, hour=2, by_gsm=15, bz_gsm=16, density=23, speed=24,
                  pressure=28, kp=38, dst=40)
_OMNI_FILL = dict(by_gsm=999.0, bz_gsm=999.0, density=999.0, speed=9999.0,
                  pressure=99.0, kp=99.0, dst=99999.0)
_OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
_OMNI_CACHE = {}


def set_external_model(model):
    """Set (or clear with None) the active external field model.

    Use this rather than assigning to a `_EXTERNAL_MODEL` attribute on some
    other module. `from ... import _EXTERNAL_MODEL` binds a copy of the
    reference, so `other_module._EXTERNAL_MODEL = None` rebinds only that
    module's name while the physics here keeps the old value -- silently
    wrong numbers, no error raised. Returns the previous model so callers can
    restore it in a finally block.
    """
    global _EXTERNAL_MODEL
    prev = _EXTERNAL_MODEL
    _EXTERNAL_MODEL = model
    return prev


def get_external_model():
    """Return the active external field model (None = internal IGRF only)."""
    return _EXTERNAL_MODEL


# ---------------------------------------------------------------------------
# IGRF internal field (vectorised synthesis, cached per epoch)
# ---------------------------------------------------------------------------
# Moved as a whole: these definitions are nested inside an `if _HAS_PPIGRF:`
# guard, so extracting functions individually would have silently dropped
# _FastIGRF and _read_shc_cached along with the guard itself.

if _HAS_PPIGRF:

    @_functools.lru_cache(maxsize=8)
    def _read_shc_cached(coeff_fn):
        return _pp.read_shc(coeff_fn)

    class _FastIGRF:
        def __init__(self, date, coeff_fn=None, min_degree=1, max_degree=13):
            if coeff_fn is None:
                coeff_fn = _pp.shc_fn
            g, h = _read_shc_cached(coeff_fn)
            self.keys = g.keys()
            n, m = np.array([k for k in g.columns]).T
            self.n = n.reshape((1, -1)); self.m = m.reshape((1, -1))
            d = np.array([date])
            index = g.index.union(d)
            gi = g.reindex(index).groupby(index).first().interpolate(method='time').loc[d, :]
            hi = h.reindex(index).groupby(index).first().interpolate(method='time').loc[d, :]
            self.gh = np.hstack((gi.values, hi.values)).T
            self.min_degree = min_degree; self.max_degree = max_degree
            self.RE = _pp.RE

        def __call__(self, lon, lat, h):
            lon, lat, h = np.broadcast_arrays(lon, lat, h)
            shape = lon.shape
            lon, lat, h = map(lambda x: np.asarray(x, float).flatten(), [lon, lat, h])
            theta, r, _, __ = _pp.geod2geoc(lat, h, h, h)
            r = r.reshape((-1, 1)); theta = theta.reshape((-1, 1)); phi = lon.reshape((-1, 1))
            n, m = self.n, self.m
            P, dP = _pp.get_legendre(theta, self.keys)
            phi_rad = np.radians(phi)
            cosmphi = np.cos(phi_rad * m); sinmphi = np.sin(phi_rad * m)
            nn, mm = np.tile(n, 2), np.tile(m, 2)
            RE = self.RE
            N_map = ((nn >= self.min_degree) & (nn <= self.max_degree)).astype(int)
            G = N_map * (RE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
            Br = G.dot(self.gh).T
            G = -N_map * (RE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) * RE / r
            Btheta = G.dot(self.gh).T
            G = -N_map * (RE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
                * RE / r / np.sin(np.radians(theta))
            Bphi = G.dot(self.gh).T
            Be = Bphi
            _, __2, Bn, Bu = _pp.geoc2geod(theta.flatten(), r.flatten(),
                                           Btheta.flatten(), Br.flatten())
            return (Be.reshape((1,) + shape), Bn.reshape((1,) + shape), Bu.reshape((1,) + shape))

        def geocentric(self, r_km, theta_deg, phi_deg):
            """
            Exact geocentric spherical-harmonic synthesis.

            Returns (Br, Btheta, Bphi) in nT for geocentric radius / colatitude
            / longitude.  Field-line tracing works in geocentric cartesian
            coordinates, so going through ppigrf's geodetic entry point would
            mean two lossy conversions per evaluation; this skips both.
            """
            r = np.asarray(r_km, float).reshape((-1, 1))
            theta = np.asarray(theta_deg, float).reshape((-1, 1))
            phi = np.asarray(phi_deg, float).reshape((-1, 1))
            n, m = self.n, self.m
            P, dP = _pp.get_legendre(theta, self.keys)
            phi_rad = np.radians(phi)
            cosmphi = np.cos(phi_rad * m); sinmphi = np.sin(phi_rad * m)
            nn, mm = np.tile(n, 2), np.tile(m, 2)
            RE = self.RE
            N_map = ((nn >= self.min_degree) & (nn <= self.max_degree)).astype(int)
            G = N_map * (RE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
            Br = G.dot(self.gh).flatten()
            G = -N_map * (RE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) * RE / r
            Bt = G.dot(self.gh).flatten()
            G = -N_map * (RE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
                * RE / r / np.sin(np.radians(theta))
            Bp = G.dot(self.gh).flatten()
            return Br, Bt, Bp


    _FAST_IGRF_CACHE = {}

    def _fast_igrf_for(date):
        key = (date, _pp.shc_fn)
        ev = _FAST_IGRF_CACHE.get(key)
        if ev is None:
            ev = _FastIGRF(date)
            _FAST_IGRF_CACHE[key] = ev
        return ev


def _geo_to_gsm_matrix(ut):
    """
    GEO -> GSM rotation R, acting on column vectors: v_gsm = R @ v_geo.
    Equivalently the ROWS of R are the GSM axes expressed in GEO, so the
    sunward axis in GEO is R[0]. For arrays of row vectors, GEO -> GSM is
    `rows @ R.T` and GSM -> GEO is `rows @ R`.
    """
    _gp.recalc(ut)
    return np.stack([np.array(_gp.geogsm(*e, 1)) for e in np.eye(3)], axis=1)


class _T89Grid:
    """T89 sampled on a GSM grid, trilinearly interpolated, vectorised."""

    def __init__(self, ut, kp=2, x_min=-30.0, x_max=25.0, half_yz=16.0, step=0.5,
                 model="t89", parmod=None):
        self.ps = _gp.recalc(ut)
        self.iopt = int(np.clip(kp, 0, 6)) + 1
        self.kp = kp
        self.model = model
        self.parmod = parmod
        self.M = _geo_to_gsm_matrix(ut)
        self.x = np.arange(x_min, x_max + step, step)
        self.y = np.arange(-half_yz, half_yz + step, step)
        self.z = np.arange(-half_yz, half_yz + step, step)
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        tag = ("T96 (Pdyn=%.2f, Dst=%.0f, By=%.1f, Bz=%.1f)" % tuple(parmod[:4])
               if model == "t96" else f"T89 (Kp={kp})")
        print(f"  sampling {tag} onto {nx}x{ny}x{nz} GSM grid...", flush=True)
        for i, xv in enumerate(self.x):
            for j, yv in enumerate(self.y):
                for k, zv in enumerate(self.z):
                    r2 = xv*xv + yv*yv + zv*zv
                    if r2 < 2.25 or r2 > 1600.0:
                        continue
                    if self.model == "t96":
                        B[i, j, k] = _t96.t96(self.parmod, self.ps, xv, yv, zv)
                    else:
                        B[i, j, k] = _t89.t89(self.iopt, self.ps, xv, yv, zv)
        self.B = B
        self.step = step

    def __call__(self, pos_geo_km):
        RE = EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM
        p = (np.asarray(pos_geo_km) / RE) @ self.M.T   # GEO -> GSM (rows)
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        gx = (p[:, 0] - self.x[0]) / self.step
        gy = (p[:, 1] - self.y[0]) / self.step
        gz = (p[:, 2] - self.z[0]) / self.step
        i0 = np.clip(np.floor(gx).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
        k0 = np.clip(np.floor(gz).astype(int), 0, nz - 2)
        fx = np.clip(gx - i0, 0, 1)[:, None]
        fy = np.clip(gy - j0, 0, 1)[:, None]
        fz = np.clip(gz - k0, 0, 1)[:, None]
        B = self.B
        c00 = B[i0, j0, k0]*(1-fx) + B[i0+1, j0, k0]*fx
        c01 = B[i0, j0, k0+1]*(1-fx) + B[i0+1, j0, k0+1]*fx
        c10 = B[i0, j0+1, k0]*(1-fx) + B[i0+1, j0+1, k0]*fx
        c11 = B[i0, j0+1, k0+1]*(1-fx) + B[i0+1, j0+1, k0+1]*fx
        c0 = c00*(1-fy) + c10*fy
        c1 = c01*(1-fy) + c11*fy
        return (c0*(1-fz) + c1*fz) @ self.M           # GSM -> GEO (rows)


def _get_external(date, kp=2, step=None, model="t89", solar_wind=None):
    """
    Sampled external-field model.

    T89 is binned by Kp alone.  T96 takes the measured solar wind directly
    (Pdyn, Dst, By, Bz), which is why it does better during disturbed
    conditions: validated against GOES-18 on 2025-11-12 (Kp 8.7) it gives
    49.5 nT RMS versus 66.1 nT for T89 at the observed Kp.
    """
    if not _HAS_GEOPACK:
        return None
    ut = (date - datetime(1970, 1, 1)).total_seconds()
    parmod = None
    if model == "t96":
        sw = solar_wind or get_solar_wind(date)
        if not sw or sw.get("dp_nPa") is None:
            print("  T96 needs OMNI drivers — falling back to T89", flush=True)
            model = "t89"
        else:
            parmod = np.zeros(10)
            parmod[:4] = [sw["dp_nPa"], sw.get("dst") or 0.0,
                          sw.get("by_nT") or 0.0, sw.get("bz_nT") or 0.0]
    if step is None:
        # T96 is ~24x slower per evaluation than T89 (1.6 ms vs 69 us), so it is
        # sampled on a coarser grid.  Measured interpolation error at 1.0 RE is
        # reported in validate_against_goes.py.
        step = 1.0 if model == "t96" else 0.5
    key = (round(ut / 3600.0), kp, step, model,
           None if parmod is None else tuple(np.round(parmod[:4], 3)))
    g = _T89_CACHE.get(key)
    if g is not None:
        return g
    import hashlib as _hl
    tag = _hl.md5(repr(key).encode()).hexdigest()[:16]
    fp = _texture_cache_dir() / f"extfield_{tag}.npz"
    if fp.exists():
        d = np.load(fp)
        g = _T89Grid.__new__(_T89Grid)
        g.ps = float(d["ps"]); g.iopt = int(d["iopt"]); g.kp = kp
        g.M = d["M"]; g.x = d["x"]; g.y = d["y"]; g.z = d["z"]
        g.B = d["B"]; g.step = float(d["step"])
        g.model = model; g.parmod = parmod
        print(f"  reusing cached {model.upper()} grid", flush=True)
    else:
        g = _T89Grid(ut, kp=kp, step=step, model=model, parmod=parmod)
        np.savez_compressed(fp, ps=g.ps, iopt=g.iopt, M=g.M, x=g.x, y=g.y,
                            z=g.z, B=g.B, step=g.step)
    _T89_CACHE[key] = g
    return g


def _get_t89(date, kp=2, step=None):
    """Backward-compatible T89-only wrapper."""
    return _get_external(date, kp=kp, step=step, model="t89")


def _enu_to_cartesian_batch(Be, Bn, Bu, lons_deg, lats_deg):
    lo = np.radians(lons_deg)
    la = np.radians(lats_deg)
    Bx = -np.sin(lo)*Be - np.sin(la)*np.cos(lo)*Bn + np.cos(la)*np.cos(lo)*Bu
    By =  np.cos(lo)*Be - np.sin(la)*np.sin(lo)*Bn + np.cos(la)*np.sin(lo)*Bu
    Bz =                              np.cos(la)*Bn +             np.sin(la)*Bu
    return np.stack([Bx, By, Bz], axis=1)


def _bfield_batch(positions, date, include_external=True):
    """
    IGRF field vector (nT) in geocentric cartesian coordinates.

    Evaluated by direct geocentric synthesis.  The previous implementation
    passed geocentric latitude and spherical altitude to ppigrf's geodetic
    entry point, which mixes reference surfaces and tilts the field vector by
    up to ~0.35 deg — small per call, but it accumulates over the thousands of
    RK4 steps in a single field line.
    """
    r     = np.linalg.norm(positions, axis=1)
    theta = np.degrees(np.arccos(np.clip(positions[:, 2] / r, -1.0, 1.0)))
    phi   = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
    Br, Bt, Bp = _fast_igrf_for(date).geocentric(r, theta, phi)
    th, ph = np.radians(theta), np.radians(phi)
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    B = np.stack([Br*st*cp + Bt*ct*cp - Bp*sp,
                  Br*st*sp + Bt*ct*sp + Bp*cp,
                  Br*ct    - Bt*st], axis=1)
    if include_external and _EXTERNAL_MODEL is not None:
        B = B + _EXTERNAL_MODEL(positions)
    return B


def _bunit_batch(positions, date):
    B    = _bfield_batch(positions, date)
    Bmag = np.linalg.norm(B, axis=1, keepdims=True)
    return np.where(Bmag > 1e-9, B / np.where(Bmag > 0, Bmag, 1.0), 0.0)


def _field_magnitude_along(points, date, chunk=6000):
    n = len(points)
    out = np.empty(n, dtype=float)
    for s in range(0, n, chunk):
        seg = points[s:s+chunk]
        out[s:s+chunk] = np.linalg.norm(_bfield_batch(seg, date), axis=1)
    return out


def _surface_radius_km(positions, extra_km=0.0):
    """Geocentric radius of the WGS84 surface beneath each position."""
    r = np.linalg.norm(positions, axis=1)
    sphi2 = np.clip(positions[:, 2] / r, -1.0, 1.0) ** 2
    a, b = WGS84_A_KM, WGS84_B_KM
    return a * b / np.sqrt(a*a * sphi2 + b*b * (1.0 - sphi2)) + extra_km


def _adaptive_step_km(r_km, step_min, step_max):
    """Step length grows with radius: the field scale length grows ~r, so
    h ~ r^1.5 keeps angular resolution roughly constant along the line."""
    return np.clip(step_min * (r_km / EARTH_RADIUS_KM) ** 1.5, step_min, step_max)


def _trace_batch_rk4(seeds, date, direction=1, step_min=8.0, step_max=250.0,
                     max_steps=6000, max_r_km=None, stop_maglat_deg=None,
                     mag_axis=None, surface_alt_km=0.0, stop_b_nT=None):
    """
    Vectorised adaptive-step RK4 trace of B-field lines.

    A line stops when it reaches the WGS84 surface (or `surface_alt_km` above
    it), leaves `max_r_km`, exceeds `stop_maglat_deg`, or — when `stop_b_nT`
    is supplied — reaches its mirror field strength.  The mirror condition is
    the physical bound for trapped particles and is what shapes the radiation
    belt shells.
    """
    if max_r_km is None:
        max_r_km = 4.5 * EARTH_RADIUS_KM
    n = len(seeds)
    if n == 0:
        return []
    pos = np.array(seeds, dtype=float)
    d = float(direction)
    trails = [[pos[i].copy()] for i in range(n)]
    active = np.ones(n, dtype=bool)
    stop_b = None if stop_b_nT is None else np.asarray(stop_b_nT, dtype=float)

    for _ in range(max_steps):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        p = pos[idx]
        r = np.linalg.norm(p, axis=1)
        h = (d * _adaptive_step_km(r, step_min, step_max))[:, None]
        try:
            B1 = _bfield_batch(p, date)
            b1mag = np.linalg.norm(B1, axis=1)
            k1 = B1 / np.where(b1mag > 0, b1mag, 1.0)[:, None]
            k2 = _bunit_batch(p + h * k1 / 2, date)
            k3 = _bunit_batch(p + h * k2 / 2, date)
            k4 = _bunit_batch(p + h * k3, date)
        except Exception:
            break
        new = p + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_r = np.linalg.norm(new, axis=1)
        surf = _surface_radius_km(new, extra_km=surface_alt_km)

        for k, i in enumerate(idx):
            q, qr = new[k], new_r[k]
            # mirror point: |B| has risen to the mirror field for this shell
            if stop_b is not None and b1mag[k] >= stop_b[i]:
                active[i] = False
                continue
            if qr < surf[k]:
                a0 = pos[i]; ar = np.linalg.norm(a0)
                if ar > surf[k] and abs(ar - qr) > 1e-9:
                    f = (ar - surf[k]) / (ar - qr)
                    q = a0 + f * (q - a0)
                trails[i].append(q.copy())
                active[i] = False
                continue
            trails[i].append(q.copy())
            pos[i] = q
            if qr > max_r_km:
                active[i] = False
                continue
            if stop_maglat_deg is not None and mag_axis is not None:
                ml = np.degrees(np.arcsin(np.clip(np.dot(q, mag_axis) / qr, -1, 1)))
                if abs(ml) >= stop_maglat_deg:
                    active[i] = False
    return [np.array(t) for t in trails]


def _trace_all_closed(seeds, date, **kw):
    n = len(seeds)
    print(f"  forward  ({n} lines)...", flush=True)
    fwd = _trace_batch_rk4(seeds, date, direction=+1, **kw)
    print(f"  backward ({n} lines)...", flush=True)
    bwd = _trace_batch_rk4(seeds, date, direction=-1, **kw)
    lines = []
    for f, b in zip(fwd, bwd):
        if len(b) > 1:
            lines.append(np.concatenate([b[::-1], f[1:]], axis=0))
        else:
            lines.append(f)
    return lines


def _resample_curve(pts, n):
    """Resample a polyline to n points, uniform in arclength."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.repeat(pts[:1] if len(pts) else np.zeros((1, 3)), n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(3)], axis=1)


def _make_seeds_lshell(L_values, date, n_lons=12):
    """
    Seed field lines on the REAL magnetic equator at given McIlwain L.

    Previously seeds were placed on a grid of dipole magnetic latitude, which
    imposes dipole geometry on a field that is not a dipole and gives the drawn
    lines no physical label.  Seeding on the true minimum-|B| surface at fixed L
    means each drawn line corresponds to a real, labelled drift shell.
    """
    RE = EARTH_RADIUS_KM
    axis = _dipole_axis()
    _, e1, e2 = _mag_basis(axis)
    phis = np.linspace(0, 2*np.pi, n_lons, endpoint=False)
    guess = [L*RE*(np.cos(ph)*e1 + np.sin(ph)*e2) for L in L_values for ph in phis]
    eq, B0 = _true_magnetic_equator(guess, date)
    L_real = (_M_DIPOLE_NT_RE3 / np.maximum(B0, 1e-9)) ** (1.0/3.0)
    return [np.asarray(q) for q in eq], L_real


def _make_seeds_magnetic(lats_deg, n_lons=6, alt_km=500.0):
    RE = EARTH_RADIUS_KM + alt_km
    axis, e1, e2 = _mag_basis(_dipole_axis())
    seeds = []
    for lat in lats_deg:
        la = np.radians(lat)
        for lo in np.linspace(0, 2*np.pi, n_lons, endpoint=False):
            seeds.append(RE * (np.cos(la)*np.cos(lo)*e1
                             + np.cos(la)*np.sin(lo)*e2
                             + np.sin(la)*axis))
    return seeds


def _physics_fingerprint():
    """
    Digest of the source of every routine that feeds the belt samples.

    The flux cache used to be keyed on parameters alone, so fixing the GEO->GSM
    transpose left stale samples on disk that were silently reused — the cache
    could not tell that the physics had changed.  Keying on the code itself
    makes any edit to these routines invalidate it automatically.
    """
    import inspect, hashlib
    parts = []
    for obj in (_bfield_batch, _bunit_batch, _surface_radius_km, _adaptive_step_km,
                _trace_batch_rk4, _resample_curve, _true_magnetic_equator,
                _belt_flux_samples, _aep8_lookup, _T89Grid, _geo_to_gsm_matrix):
        try:
            parts.append(inspect.getsource(obj))
        except Exception:
            parts.append(repr(obj))
    parts.append(str(_M_DIPOLE_NT_RE3))
    return hashlib.md5("".join(parts).encode()).hexdigest()[:12]


def _aep8_table_path():
    return _texture_cache_dir() / "aep8_table.npz"


def _build_aep8_table(L_step=0.1, n_bb=26, energies=(10.0, 1.0), model='max'):
    """Tabulate AE-8/AP-8 over (L, B/B0).  Needs spacepy; cached to disk."""
    try:
        import warnings as _w
        _w.filterwarnings("ignore")
        import spacepy.irbempy as _ib
    except Exception as e:
        print(f"  spacepy/IRBEM unavailable ({e}) — cannot build AE8/AP8 table", flush=True)
        return None
    L = np.round(np.arange(1.0, 8.0 + L_step, L_step), 3)
    B = np.round(np.geomspace(1.0, 300.0, n_bb), 4)
    out = {}
    for pk, E in (('p', energies[0]), ('e', energies[1])):
        F = np.zeros((len(L), len(B)))
        for i, Lv in enumerate(L):
            for j, bv in enumerate(B):
                try:
                    F[i, j] = float(_ib.get_AEP8(E, [float(bv), float(Lv)], model=model,
                                                 fluxtype='int', particles=pk))
                except Exception:
                    F[i, j] = 0.0
        out[pk] = np.nan_to_num(F, nan=0.0)     # NaN = outside model domain
    np.savez_compressed(_aep8_table_path(), L=L, B=B, p=out['p'], e=out['e'])
    return dict(L=L, B=B, p=out['p'], e=out['e'])


def _load_aep8_table(allow_build=True):
    global _AEP8_TABLE
    if _AEP8_TABLE is not None:
        return _AEP8_TABLE
    fp = _aep8_table_path()
    if fp.exists():
        d = np.load(fp)
        _AEP8_TABLE = dict(L=d['L'], B=d['B'], p=d['p'], e=d['e'])
        return _AEP8_TABLE
    if allow_build:
        print("  building AE8/AP8 (L, B/B0) table — one-off, ~2 min...", flush=True)
        _AEP8_TABLE = _build_aep8_table()
        return _AEP8_TABLE
    return None


def _aep8_lookup(species, L, b_over_b0):
    """Bilinear interpolation of the tabulated flux in (L, log B/B0)."""
    tab = _load_aep8_table()
    if tab is None:
        return None
    Lg, Bg, F = tab['L'], tab['B'], tab[species]
    li = np.clip(np.interp(L, Lg, np.arange(len(Lg))), 0, len(Lg) - 1)
    bi = np.clip(np.interp(np.log(np.maximum(b_over_b0, 1.0)), np.log(Bg),
                           np.arange(len(Bg))), 0, len(Bg) - 1)
    i0 = np.floor(li).astype(int); i1 = np.minimum(i0 + 1, len(Lg) - 1)
    j0 = np.floor(bi).astype(int); j1 = np.minimum(j0 + 1, len(Bg) - 1)
    fi = li - i0; fj = bi - j0
    return ((F[i0, j0]*(1-fi) + F[i1, j0]*fi)*(1-fj)
          + (F[i0, j1]*(1-fi) + F[i1, j1]*fi)*fj)


def _true_magnetic_equator(seeds, date, iters=2):
    """
    Locate the real minimum-|B| point on the field line through each seed.

    The dipole equatorial plane is not the magnetic equator of the actual
    field; seeding there biases both L and B/B0.  Iterating "trace, take the
    |B| minimum, re-trace from it" converges on the true equatorial crossing.
    """
    p = np.array(seeds, dtype=float)
    for _ in range(iters):
        fw = _trace_batch_rk4(p, date, direction=+1, step_min=20.0, step_max=140.0,
                              max_steps=4000, max_r_km=30*EARTH_RADIUS_KM)
        bw = _trace_batch_rk4(p, date, direction=-1, step_min=20.0, step_max=140.0,
                              max_steps=4000, max_r_km=30*EARTH_RADIUS_KM)
        nxt = []
        for f, b in zip(fw, bw):
            line = np.concatenate([b[::-1], f[1:]], axis=0) if len(b) > 1 else f
            Bm = np.linalg.norm(_bfield_batch(line, date), axis=1)
            nxt.append(line[np.argmin(Bm)])
        p = np.array(nxt)
    B0 = np.linalg.norm(_bfield_batch(p, date), axis=1)
    return p, B0


def _belt_flux_samples(date, axis, L_min=1.05, L_max=7.0, n_L=24, n_azim=24,
                       n_pts=80, pitch_angle_deg=25.0, loss_cone_alt_km=100.0,
                       eq_iters=2):
    """
    Trace shells from the true magnetic equator and evaluate AE-8/AP-8 at every
    point.  Returns (positions_km, flux_protons, flux_electrons).
    """
    RE = EARTH_RADIUS_KM
    _, e1, e2 = _mag_basis(axis)
    Ls = np.linspace(L_min, L_max, n_L)
    phis = np.linspace(0, 2*np.pi, n_azim, endpoint=False)
    seeds = [L*RE*(np.cos(ph)*e1 + np.sin(ph)*e2) for L in Ls for ph in phis]
    print(f"  belts: locating true magnetic equator for {len(seeds)} shells...", flush=True)
    eq, B0 = _true_magnetic_equator(seeds, date, iters=eq_iters)
    L_real = (_M_DIPOLE_NT_RE3 / np.maximum(B0, 1e-9)) ** (1.0/3.0)
    sa = np.sin(np.radians(pitch_angle_deg))**2
    kw = dict(step_min=25.0, step_max=200.0, max_steps=4000,
              max_r_km=30*RE, surface_alt_km=loss_cone_alt_km,
              stop_b_nT=B0/max(sa, 1e-6))
    print(f"  belts: tracing {len(seeds)} shells to mirror points...", flush=True)
    north = _trace_batch_rk4(eq, date, direction=+1, **kw)
    south = _trace_batch_rk4(eq, date, direction=-1, **kw)
    P, Fp, Fe = [], [], []
    for i in range(len(eq)):
        n_c, s_c = north[i], south[i]
        line = np.concatenate([s_c[::-1], n_c[1:]], axis=0) if len(s_c) > 1 else n_c
        if len(line) < 3:
            continue
        line = _resample_curve(line, n_pts)
        Bm = np.linalg.norm(_bfield_batch(line, date), axis=1)
        ratio = Bm / max(B0[i], 1e-9)
        Lv = np.full(len(line), L_real[i])
        fp = _aep8_lookup('p', Lv, ratio)
        fe = _aep8_lookup('e', Lv, ratio)
        if fp is None:
            return None
        P.append(line); Fp.append(fp); Fe.append(fe)
    return np.concatenate(P), np.concatenate(Fp), np.concatenate(Fe)


def _load_omni_year(year):
    """OMNI2 hourly records for a year, cached on disk and in memory."""
    if year in _OMNI_CACHE:
        return _OMNI_CACHE[year]
    import urllib.request
    fp = _texture_cache_dir() / f"omni2_{year}.dat"
    if not fp.exists() or fp.stat().st_size < 1e5:
        try:
            print(f"  downloading OMNI solar-wind record for {year} ...", flush=True)
            with urllib.request.urlopen(_OMNI_URL.format(year=year), timeout=180) as r:
                fp.write_bytes(r.read())
        except Exception as e:
            print(f"  OMNI download failed ({e})", flush=True)
            return None
    rows = []
    for ln in fp.read_text(errors="replace").splitlines():
        parts = ln.split()
        if len(parts) < 41:
            continue
        try:
            rows.append([float(parts[_OMNI_COLS[k]])
                         for k in ("doy", "hour", "by_gsm", "bz_gsm", "density",
                                   "speed", "pressure", "kp", "dst")])
        except Exception:
            continue
    if not rows:
        return None
    _OMNI_CACHE[year] = np.array(rows)
    return _OMNI_CACHE[year]


def get_solar_wind(date, window_hours=24):
    """
    Observed solar-wind drivers around `date`.

    Returns Kp (for T89), Bz_GSM and dynamic pressure (for the Shue
    magnetopause), plus speed, density and Dst for reporting.  Returns None if
    the record is unavailable, in which case callers fall back to nominal
    values and say so.
    """
    a = _load_omni_year(date.year)
    if a is None:
        return None
    doy = date.timetuple().tm_yday
    hr = date.hour + date.minute / 60.0
    dt_h = (a[:, 0] - doy) * 24.0 + (a[:, 1] - hr)
    sel = a[np.abs(dt_h) <= window_hours / 2.0]
    if len(sel) == 0:
        return None

    def clean(col, key):
        v = sel[:, col]
        v = v[v < _OMNI_FILL[key]]
        return float(np.mean(v)) if len(v) else None

    kp = clean(7, "kp")
    out = dict(by_nT=clean(2, "by_gsm"), bz_nT=clean(3, "bz_gsm"),
               density=clean(4, "density"), speed=clean(5, "speed"),
               dp_nPa=clean(6, "pressure"), dst=clean(8, "dst"),
               kp=None if kp is None else kp / 10.0,   # OMNI stores Kp*10
               n_hours=len(sel))
    return out


def _apply_solar_wind(date, kp, sw_bz_nT, sw_dp_nPa, use_omni=True):
    """Resolve drivers, preferring observation over nominal values."""
    if not use_omni:
        return kp, sw_bz_nT, sw_dp_nPa, "nominal (user-specified)"
    sw = get_solar_wind(date)
    if not sw or sw["dp_nPa"] is None or sw["bz_nT"] is None:
        return kp, sw_bz_nT, sw_dp_nPa, "nominal (OMNI unavailable)"
    kp_o = int(round(sw["kp"])) if sw["kp"] is not None else kp
    src = (f"OMNI {date:%Y-%m-%d}: v={sw['speed']:.0f} km/s, n={sw['density']:.1f}/cc, "
           f"Bz={sw['bz_nT']:+.1f} nT, Dp={sw['dp_nPa']:.2f} nPa, Kp={sw['kp']:.1f}")
    print(f"  solar wind from {src}", flush=True)
    return kp_o, sw["bz_nT"], sw["dp_nPa"], src


def _gsm_frame(date):
    """
    GEO -> GSM rotation R (v_gsm = R @ v_geo); rows of R are the GSM axes
    expressed in GEO.

    Uses geopack when present.  Otherwise the frame is constructed from its
    definition: X_gsm points at the Sun (the subsolar direction), and Z_gsm is
    chosen so the geomagnetic dipole lies in the X-Z plane.
    """
    if _HAS_GEOPACK:
        try:
            ut = (date - datetime(1970, 1, 1)).total_seconds()
            return _geo_to_gsm_matrix(ut)
        except Exception:
            pass
    sd, sl = _subsolar_point(date)
    x = np.array([np.cos(np.radians(sd))*np.cos(np.radians(sl)),
                  np.cos(np.radians(sd))*np.sin(np.radians(sl)),
                  np.sin(np.radians(sd))])
    x /= np.linalg.norm(x)
    d = _dipole_axis()
    y = np.cross(d, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=0)      # rows = GSM axes in GEO, matching R


def _sun_direction_geo(date):
    """Unit vector toward the Sun, in the scene's Earth-fixed frame."""
    return _gsm_frame(date)[0]          # GSM +X expressed in GEO


def _shue_magnetopause(date, bz_nT=0.0, dp_nPa=2.0, n_theta=48, n_phi=64,
                       theta_max_deg=150.0):
    """
    Shue et al. (1998) magnetopause:  r = r0 (2/(1+cos theta))^alpha
        r0    = (10.22 + 1.29 tanh(0.184 (Bz + 8.14))) Dp^(-1/6.6)
        alpha = (0.58 - 0.007 Bz)(1 + 0.024 ln Dp)
    with Bz in nT (GSM) and Dp the solar-wind dynamic pressure in nPa.  This is
    the outer boundary of the magnetosphere — the surface the solar wind
    confines the field inside.  Returned in the scene's Earth-fixed frame.
    """
    r0 = (10.22 + 1.29*np.tanh(0.184*(bz_nT + 8.14))) * dp_nPa**(-1.0/6.6)
    alpha = (0.58 - 0.007*bz_nT) * (1.0 + 0.024*np.log(dp_nPa))
    th = np.radians(np.linspace(0.1, theta_max_deg, n_theta))
    ph = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    R = r0 * (2.0/(1.0 + np.cos(TH)))**alpha
    X = R*np.cos(TH); Y = R*np.sin(TH)*np.cos(PH); Z = R*np.sin(TH)*np.sin(PH)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) * EARTH_RADIUS_KM
    geo = pts @ _gsm_frame(date)     # GSM rows -> GEO
    jj, ii = np.meshgrid(np.arange(n_theta-1), np.arange(n_phi-1), indexing='ij')
    v00 = (jj*n_phi + ii).ravel(); v01 = v00+1; v10 = v00+n_phi; v11 = v10+1
    return (geo, np.concatenate([v00, v00]), np.concatenate([v01, v11]),
            np.concatenate([v11, v10]), r0, alpha)


def _classify_line(pts, surface_pad_km=800.0):
    """
    'closed' if both ends return to Earth, 'open' if only one does.

    Open field lines are the ones connected to the solar wind — the polar cap
    and tail lobes — and separating them from closed lines is the single most
    informative thing that can be said about magnetospheric topology.
    """
    r0 = np.linalg.norm(pts[0]); r1 = np.linalg.norm(pts[-1])
    s0 = _surface_radius_km(pts[:1])[0] + surface_pad_km
    s1 = _surface_radius_km(pts[-1:])[0] + surface_pad_km
    n = int(r0 <= s0) + int(r1 <= s1)
    return 'closed' if n == 2 else ('open' if n == 1 else 'detached')
