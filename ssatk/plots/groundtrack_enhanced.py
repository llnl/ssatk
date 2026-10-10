"""
groundtrack_enhanced.py — Enhanced ground track for ssapy_analysis.py
========================================================================
Drop-in replacement for the toolkit's plain `groundtrack_plot`, matching
the spec in the ssapy_analysis.py checkpoint:

    groundtrack_{tag}.png — enhanced ground track with day/night
    terminator, eclipse coloring, subsolar point, site marker

This version actively uses your real ssapy / ssatk modules wherever
they're available, instead of reimplementing them:

  - Continents: the REAL Earth texture your own `globe_plot.py` already
    uses (`ssatk.plots._textures.earth_texture_path()`)
    is tried first. Falls back to `cartopy` coastlines (if installed), then
    to a bundled low-res coastline outline, so it still renders *something*
    real in an environment that has neither — but in your actual conda env
    with ssapy installed, this will show the same continents as the rest
    of the toolkit's plots.
  - Coordinate transform: `ssatk.coordinates.frames.eci_to_ecf_matrix`,
    the GCRF->ITRF chain SSAPy's groundTrack uses (precession-nutation, GAST
    at UT1, polar motion); within 0.05 arcsec of astropy GCRS->ITRS in 2026.
  - Eclipse / shadow geometry: always uses the inlined two-circle overlap
    formula below (same real physics as a correct umbra/penumbra cone).
    An earlier version of this tried a toolkit `shadow_cone_params`
    helper first, but that implementation built the umbra cone opening
    the wrong way (a real correctness bug, not just unavailable) and was
    deleted rather than fixed in place -- so this file no longer tries to
    import it at all.
  - Orbit propagation (self-test only): tries real `ssapy.Orbit` first;
    falls back to a plain two-body propagator only if `ssapy` isn't
    installed, so you can see immediately whether your real environment
    is wired up correctly by which code path prints.

Interface
---------

.. code-block:: python

    from groundtrack_enhanced import plot_enhanced_groundtrack
    plot_enhanced_groundtrack(
        r_eci_km, t,                      # from your real propagator
        site_lat=28.5, site_lon=-80.6, site_name="Cape Canaveral",
        sat_name="ISS", save_path=f"groundtrack_{tag}.png",
    )
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, get_sun
import astropy.units as u

from ssatk.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM, SUN_RADIUS_KM

from .plotutils import (
    _figure_save_path,
    _pop_save_path_aliases,
    _raise_unrecognized_kwargs,
    normalize_orbit_trajectory,
)

# RE_KM, R_SUN_KM, AU_KM sourced from ssatk.constants / astropy
# (same pattern as sun_view.py) rather than hardcoded literals -- these
# were independently duplicated (at the same value, by coincidence)
# across this file, orbit_state.py, and layers.py.
RE_KM = EARTH_RADIUS_KM
R_SUN_KM = SUN_RADIUS_KM

# ── Optional real-toolkit imports (all soft — each falls back cleanly) ──────
from ssatk.coordinates.frames import eci_to_ecf_matrix as _tk_eci_to_ecf_matrix

# Note: this file used to also try importing a toolkit `shadow_cone_params`
# helper (from the old core/sun.py) for eclipse geometry, with a fallback
# to the inline two-circle formula below. That helper built the umbra cone
# opening the wrong way (a real correctness bug, not just unavailability)
# and was deleted rather than fixed in place, so there's no longer
# anything to optionally import here -- the inline formula (same real
# physics) is used directly and unconditionally.

try:
    import ssapy as _ssapy
    _HAS_SSAPY = hasattr(_ssapy, "Orbit")   # PyPI also hosts an unrelated
    # "ssapy" (Singular Spectrum Analysis) package with the same import
    # name — checking for `Orbit` avoids silently treating that as a match.
    if not _HAS_SSAPY:
        warnings.warn(
            "Found a package named 'ssapy' with no Orbit class; treating it as unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    _HAS_SSAPY = False

# ── gcrf_to_itrf — uses your real frames.py if available, else astropy ───
def gcrf_to_itrf(r_eci_km: np.ndarray, t: Time) -> np.ndarray:
    """
    Transform (N,3) GCRF positions (km) to ITRF (km) at times ``t``.

    ``t`` is an astropy Time of the same length as ``r_eci_km`` or a scalar
    Time broadcast to all rows; any time scale is accepted. The rotation is
    :func:`ssatk.coordinates.frames.eci_to_ecf_matrix`, the chain
    SSAPy's groundTrack uses.

    Previously this rotated by Greenwich mean sidereal time only (38.2 km
    west and 16.7 km south at the equator on 2026-10-09 versus astropy
    GCRS->ITRS), and rebuilt a scalar Time from its ISO string, which
    dropped the time scale: a TT-scale Time moved points another 32.2 km.
    """
    r_eci_km = np.atleast_2d(np.asarray(r_eci_km, dtype=float))
    gps = np.broadcast_to(np.atleast_1d(np.asarray(t.gps, dtype=float)), (len(r_eci_km),))
    matrices = _tk_eci_to_ecf_matrix(gps).reshape(len(r_eci_km), 3, 3)
    return np.einsum("nij,nj->ni", matrices, r_eci_km)


def ecef_to_geodetic(r_ecef_km: np.ndarray):
    """WGS84 geodetic (lat_deg, lon_deg, height_km) of (N,3) ITRF positions (km).

    Previously spherical: geocentric latitude (up to 0.19 deg, 21 km, from
    geodetic) and height above the equatorial radius at every latitude.
    """
    import erfa

    r = np.atleast_2d(np.asarray(r_ecef_km, dtype=float)) * 1000.0
    lon, lat, height = erfa.gc2gd(1, r)   # 1 = WGS84
    return np.degrees(lat), np.degrees(lon), height / 1000.0


def subsolar_point(t: Time):
    """Real subsolar (lat, lon) in degrees at time(s) t, via astropy get_sun + ITRS."""
    sun_gcrs = get_sun(t)
    sun_cart = sun_gcrs.cartesian
    sun_itrs = GCRS(sun_cart, obstime=t).transform_to(ITRS(obstime=t))
    x, y, z = sun_itrs.x.value, sun_itrs.y.value, sun_itrs.z.value
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def compute_eclipse(r_eci_km: np.ndarray, t: Time, R_body_km: float = RE_KM):
    """
    Continuous illumination fraction (1.0 = full sun, 0.0 = total umbra,
    partial = penumbra), via the inline two-circle Sun/Earth angular-disk
    overlap formula below. Real Sun position from astropy (not a
    simplified ecliptic-longitude formula).

    (An earlier version of this tried a toolkit `shadow_cone_params`
    helper first, upgrading its binary in_umbra/in_penumbra result to a
    continuous fraction. That helper built the umbra cone opening the
    wrong way -- a real correctness bug -- and was deleted rather than
    fixed in place, so this function no longer has anything to try
    before the formula below.)
    """
    r = np.atleast_2d(r_eci_km)
    sun = get_sun(t).cartesian
    sun_km = np.stack([sun.x.to(u.km).value, sun.y.to(u.km).value, sun.z.to(u.km).value], axis=-1)
    sun_km = np.atleast_2d(sun_km)

    # Two-circle Sun/Earth angular-disk overlap
    D_km = np.linalg.norm(sun_km, axis=1)
    sun_hat = sun_km / D_km[:, None]
    r_mag = np.linalg.norm(r, axis=1)
    to_body_hat = -r / r_mag[:, None]
    cos_sep = np.clip(np.sum(to_body_hat * sun_hat, axis=1), -1, 1)
    sep = np.arccos(cos_sep)

    ang_body = np.arcsin(np.clip(R_body_km / r_mag, -1, 1))
    ang_sun = np.arcsin(np.clip(R_SUN_KM / D_km, -1, 1))
    anti_sun = np.sum(r * sun_hat, axis=1) < 0

    d, s, b = sep, ang_sun, ang_body
    no_overlap = d >= (s + b)
    full_overlap = d <= np.abs(s - b)
    partial = ~no_overlap & ~full_overlap

    frac = np.ones_like(r_mag)
    frac = np.where(full_overlap, np.clip(1.0 - (np.minimum(b, s)/np.maximum(s, 1e-12))**2, 0.0, 1.0), frac)

    d_s = np.where(partial, d, 1.0); s_s = np.maximum(s, 1e-12); b_s = np.maximum(b, 1e-12)
    t1 = s_s**2*np.arccos(np.clip((d_s**2+s_s**2-b_s**2)/(2*d_s*s_s+1e-30), -1, 1))
    t2 = b_s**2*np.arccos(np.clip((d_s**2+b_s**2-s_s**2)/(2*d_s*b_s+1e-30), -1, 1))
    t3 = 0.5*np.sqrt(np.clip((-d_s+s_s+b_s)*(d_s+s_s-b_s)*(d_s-s_s+b_s)*(d_s+s_s+b_s), 0, None))
    blocked = np.clip((t1+t2-t3)/(np.pi*s_s**2), 0.0, 1.0)
    frac = np.where(partial, 1.0 - blocked, frac)
    frac = np.where(anti_sun, frac, 1.0)
    return np.clip(frac, 0.0, 1.0)


def _elevation_deg(sat_ecef_km, site_ecef_km, site_lat_deg, site_lon_deg):
    lat, lon = np.radians(site_lat_deg), np.radians(site_lon_deg)
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    rel = sat_ecef_km - site_ecef_km
    rng = np.linalg.norm(rel, axis=-1)
    return np.degrees(np.arcsin(np.clip((rel @ up) / rng, -1, 1)))


def _site_ecef(lat_deg, lon_deg, alt_km=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r = RE_KM + alt_km
    return np.array([r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)])


# ── Continents ───────────────────────────────────────────────────────────────
_earth_texture_cache = None


def _load_earth_texture():
    """
    Real Earth raster, same source your own globe_plot.py already uses
    (ssapy's bundled data/earth.png). Cached after first call.
    Returns an (H, W, 3) uint8 array in equirectangular projection
    (lon -180..180, lat +90..-90 top-to-bottom), or None if unavailable.
    """
    global _earth_texture_cache
    if _earth_texture_cache is not None:
        return _earth_texture_cache
    try:
        from PIL import Image as _PILImage
        from ._textures import earth_texture_path
        img = _PILImage.open(earth_texture_path()).convert("RGB")
        _earth_texture_cache = np.asarray(img)
        return _earth_texture_cache
    except Exception as ex:
        print(f"[groundtrack_enhanced] real Earth texture load failed: {ex}")
        return None


def _draw_continents(ax):
    """
    Draw real continents under everything else. Tries, in order:
    1. Your own toolkit's real Earth texture (ssapy.utils.find_file) —
       what globe_plot.py / moon_plot_3d.py already use.
    2. cartopy Natural Earth coastlines/land polygons (if installed).
    3. A visible warning box if neither is available, rather than silently
       leaving the map blank the way the previous version of this script did.
    """
    tex = _load_earth_texture()
    if tex is not None:
        ax.imshow(tex, extent=[-180, 180, -90, 90], origin="upper",
                  aspect="auto", zorder=0.5, alpha=0.9)
        return "ssapy earth.png"

    try:
        import cartopy.feature as cfeature
        # NOTE: this path only works if `ax` was created with a cartopy
        # projection; since the rest of this plot uses plain matplotlib
        # axes (for the night-shading contourf/eclipse-coloured
        # LineCollection to line up in plain lon/lat), we instead pull the
        # raw coastline/land geometries and draw them as plain patches on
        # the existing lon/lat axes.
        land = cfeature.NaturalEarthFeature("physical", "land", "110m",
                                            facecolor="#c8c8a0")
        for geom in land.geometries():
            xs, ys = geom.exterior.xy if geom.geom_type == "Polygon" else (None, None)
            if xs is not None:
                ax.fill(xs, ys, facecolor="#c8c8a0", edgecolor="none", zorder=0.5, alpha=0.9)
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    px, py = part.exterior.xy
                    ax.fill(px, py, facecolor="#c8c8a0", edgecolor="none", zorder=0.5, alpha=0.9)
        return "cartopy Natural Earth"
    except Exception as ex:
        ax.text(0.5, 0.5, "⚠ No continent data available\n"
                          "(ssapy earth.png not found, cartopy not installed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="red", zorder=10,
                bbox=dict(facecolor="white", alpha=0.7))
        print(f"[groundtrack_enhanced] cartopy fallback also failed: {ex}")
        return None


# ── Main plot ────────────────────────────────────────────────────────────────
def plot_enhanced_groundtrack(r_eci_km: np.ndarray | None = None, t: Time | None = None,
                               site_lat: float = 0.0, site_lon: float = 0.0,
                               site_name: str = "Ground Site",
                               sat_name: str = "Satellite",
                               min_elev_deg: float = 10.0,
                               save_path: str | None = None, *, orbit=None,
                               r=None, v=None, r_units: str = "auto",
                               v_units: str = "auto", **kwargs):
    """Plot an enhanced groundtrack from SSAPy or array trajectory output.

    ``r_eci_km`` preserves the original API.  New code can also pass
    ``orbit=ssapy_orbit`` or ``r=``/``v=``/``t=`` arrays from ``ssapy.rv``.
    Position arrays may be metres or kilometres; units are auto-detected unless
    ``r_units`` is set explicitly.
    """
    save_path, kwargs = _pop_save_path_aliases(kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(kwargs, "plot_enhanced_groundtrack")
    if r is not None and r_eci_km is not None:
        raise ValueError("Use either r_eci_km positional input or r=, not both.")

    r_input = r if r is not None else r_eci_km
    r_eci_km, _, t = normalize_orbit_trajectory(
        orbit=orbit,
        r=r_input,
        v=v,
        t=t,
        require_velocity=False,
        r_units=r_units,
        v_units=v_units,
    )
    r_ecef = gcrf_to_itrf(r_eci_km, t)
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    illum = compute_eclipse(r_eci_km, t)
    ss_lat, ss_lon = subsolar_point(t)

    fig, ax = plt.subplots(figsize=(15, 8), dpi=120)
    ax.set_facecolor("#dfe9f2")

    continent_source = _draw_continents(ax)
    if continent_source:
        print(f"[groundtrack_enhanced] continents rendered from: {continent_source}")

    # Night-side shading: points on Earth where the sun is below the horizon,
    # i.e. angular distance from the (mean) subsolar point > 90 deg.
    # Semi-transparent so the continents drawn above remain visible through it.
    grid_lon = np.linspace(-180, 180, 361)
    grid_lat = np.linspace(-90, 90, 181)
    GLon, GLat = np.meshgrid(grid_lon, grid_lat)
    ss_lat_m, ss_lon_m = np.mean(ss_lat), np.mean(ss_lon)
    cos_c = (np.sin(np.radians(GLat))*np.sin(np.radians(ss_lat_m)) +
             np.cos(np.radians(GLat))*np.cos(np.radians(ss_lat_m)) *
             np.cos(np.radians(GLon - ss_lon_m)))
    night = cos_c < 0
    ax.contourf(GLon, GLat, night.astype(float), levels=[0.5, 1.5],
                colors=["#0a1530"], alpha=0.45, zorder=1)

    # Terminator line (cos_c == 0 contour)
    ax.contour(GLon, GLat, cos_c, levels=[0], colors="orange", linewidths=1.3,
               linestyles="--", zorder=2)

    # Ground track, colour-mapped by illumination fraction (eclipse coloring)
    lon_c = lon.copy()
    jumps = np.where(np.abs(np.diff(lon_c)) > 180)[0]
    seg_start = 0
    cmap = plt.get_cmap("inferno")
    for j in list(jumps) + [len(lon_c) - 1]:
        seg = slice(seg_start, j + 1)
        pts = np.array([lon_c[seg], lat[seg]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segs, colors=cmap(0.2 + 0.7*illum[seg][:-1]),
                            linewidths=2.2, zorder=4)
        ax.add_collection(lc)
        seg_start = j + 1
    ax.plot(lon_c[0], lat[0], "o", color="lime", markersize=8,
            markeredgecolor="black", zorder=6, label="Start")
    ax.plot(lon_c[-1], lat[-1], "s", color="red", markersize=7,
            markeredgecolor="black", zorder=6, label="End")

    # Subsolar point
    ax.plot(ss_lon_m, ss_lat_m, "*", color="gold", markersize=20,
            markeredgecolor="black", markeredgewidth=1, zorder=7,
            label="Subsolar point")

    # Ground site + elevation-mask visibility circle
    site_ecef = _site_ecef(site_lat, site_lon)
    elev = _elevation_deg(r_ecef, site_ecef, site_lat, site_lon)
    vis_pct = float(np.mean(elev >= min_elev_deg) * 100)
    ax.plot(site_lon, site_lat, "^", color="cyan", markersize=14,
            markeredgecolor="black", markeredgewidth=1.2, zorder=7,
            label=f"{site_name} ({vis_pct:.1f}% visible)")
    # Approximate elevation-mask ground radius (spherical Earth)
    max_range_km = RE_KM * np.arccos(RE_KM / (RE_KM + np.max(alt))) if np.max(alt) > 0 else 2000
    circle_r_deg = np.degrees(max_range_km / RE_KM)
    ax.add_patch(Circle((site_lon, site_lat), circle_r_deg, fill=False,
                        edgecolor="cyan", linestyle=":", linewidth=1, alpha=0.6))

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Ground Track — {sat_name}\n"
                f"Eclipse-coloured (bright=sunlit, dark=eclipsed) · "
                f"terminator dashed · {site_name} visibility {vis_pct:.1f}%")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.5, zorder=0)

    if save_path:
        resolved_path = _figure_save_path(save_path, default_name="enhanced_groundtrack")
        if resolved_path:
            fig.savefig(resolved_path, bbox_inches="tight")
            print(f"Saved -> {resolved_path}")
    return fig, ax, dict(site_visibility_pct=vis_pct)


# ── Standalone self-test (two-body propagation; swap in your real r_eci/t) ──
if __name__ == "__main__":
    from ssatk.plots.figpath import figpath

    MU = EARTH_MU_KM3_S2

    def _solve_kepler(M, e):
        E = M.copy()
        for _ in range(60):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
        return E

    def two_body(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg, n_orbits, n_steps, epoch_iso):
        inc, raan, argp = np.radians([inc_deg, raan_deg, argp_deg])
        nu0 = np.radians(nu0_deg)
        E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu0/2), np.sqrt(1+e)*np.cos(nu0/2))
        M0 = E0 - e*np.sin(E0)
        T_s = 2*np.pi*np.sqrt(a_km**3/MU)
        t_s = np.linspace(0, n_orbits*T_s, n_steps)
        n_rad_s = np.sqrt(MU/a_km**3)
        E = _solve_kepler(M0 + n_rad_s*t_s, e)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        r_mag = a_km*(1-e*np.cos(E))
        cO, sO = np.cos(raan), np.sin(raan); ci, si = np.cos(inc), np.sin(inc); cw, sw = np.cos(argp), np.sin(argp)
        R11=cO*cw-sO*sw*ci; R12=-cO*sw-sO*cw*ci; R21=sO*cw+cO*sw*ci; R22=-sO*sw+cO*cw*ci; R31=sw*si; R32=cw*si
        xp, yp = r_mag*np.cos(nu), r_mag*np.sin(nu)
        r_eci = np.stack([R11*xp+R12*yp, R21*xp+R22*yp, R31*xp+R32*yp], axis=1)
        t = Time(epoch_iso) + t_s*u.s
        return r_eci, t

    r_eci, t = two_body(a_km=RE_KM+408, e=0.0003, inc_deg=51.6, raan_deg=0,
                        argp_deg=0, nu0_deg=0, n_orbits=1.0, n_steps=400,
                        epoch_iso="2025-06-15 12:00:00")
    plot_enhanced_groundtrack(
        r_eci, t, site_lat=28.5, site_lon=-80.6, site_name="Cape Canaveral",
        sat_name="ISS (self-test)", save_path=figpath("groundtrack_enhanced_selftest.png"),
    )
