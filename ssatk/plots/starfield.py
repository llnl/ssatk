"""
starfield.py — star background for SSATK 3D plots
=========================================================
Drop into: ~/ssatk/ssatk/plots/starfield.py

This is the single owner of sky/astrometry code in the toolkit.  The
magnetosphere plots previously carried their own copy, which drifted; they now
import from here.

    # matplotlib (unchanged from before)
    from .starfield import add_starfield
    add_starfield(ax, plot_range, elev=30, azim=45, epoch=t_gps)

    # plotly
    from .starfield import starfield_traces
    fig.add_traces(starfield_traces(sky_radius, when=date, frame="ecef"))

    # raw directions, for anything else
    from .starfield import star_directions
    v, mag, rgb = star_directions(when=date, frame="gcrf")


Reference frames
----------------
`frame` decides how far along the chain the catalogue is carried:

* ``"j2000"``: catalogue positions as-is.
* ``"gcrf"``: proper motion and precession to the mean equator of date; this
  inertial frame is appropriate for trajectory plots.
* ``"ecef"``: rotation by Greenwich Mean Sidereal Time into the Earth-fixed
  frame. Use it with Earth's surface, an IGRF field, or ground locations;
  otherwise the sky can be misaligned by a full rotation.

Nutation (<=17") and polar motion (<0.5") are omitted.  Validated against
astropy's ITRS transform: star directions agree to 8-24 arcsec, GMST to 0.01 s.

Star colours
------------
Computed from the B-V colour index where the catalogue provides it: B-V ->
effective temperature (Ballesteros 2012) -> Planck spectrum -> CIE 1931 ->
sRGB.  Falls back to a per-spectral-class lookup otherwise.  The Yale Bright
Star Catalogue supplies B-V for ~97% of its entries.

Marker size follows the star-atlas convention of scaling with
(mag_limit - mag).  Flux spans a factor of ~250 across this range, so
area-proportional markers would render Sirius as a blob; the size mapping is a
stated plotting convention, unlike the positions and colours.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

_STAR_CACHE = {}
_HYG_PATHS = [
    os.path.expanduser("~/bright_stars.csv"),
    os.path.expanduser("~/SSAPy/ssapy/data/bright_stars.csv"),
    os.path.join(os.path.dirname(__file__), "bright_stars.csv"),
]

_SPECT_COLORS = {
    'O': [0.61, 0.69, 1.00],
    'B': [0.67, 0.75, 1.00],
    'A': [0.79, 0.85, 1.00],
    'F': [0.97, 0.97, 1.00],
    'G': [1.00, 0.96, 0.92],
    'K': [1.00, 0.82, 0.63],
    'M': [1.00, 0.80, 0.44],
}

ARCSEC = np.pi / (180.0 * 3600.0)
GPS_JD_EPOCH = 2_444_244.5


# ===========================================================================
# Time
# ===========================================================================

# Julian date and GMST moved to ssatk.time_functions.
#
# They are pure time-scale arithmetic with no rendering content, and keeping
# them here forced the physics layer to import the plotting package to reach
# them: geomagnetics needs the subsolar point, which needs GMST. Importing
# ssatk.plots pulls in plots/__init__, which auto-imports every module
# in the package including one that imports back into geomagnetics -- a cycle
# that made `import ssatk.geomagnetics` fail outright.
#
# Re-exported under their original private names so existing imports here,
# in magnetosphere_core and in the tests keep working unchanged. Verified
# bit-identical to the previous definitions across epochs from 1582 to 2026.
try:
    from ..time_functions.gmst import _gmst_rad  # noqa: F401
    from ..time_functions.julian_date import _julian_date  # noqa: F401
except ImportError:  # script mode, no package context
    from ssatk.time_functions.gmst import _gmst_rad  # noqa: F401
    from ssatk.time_functions.julian_date import _julian_date  # noqa: F401


def _to_datetime(epoch):
    """
    Accept the epoch formats used across this toolkit.

      datetime          : used directly
      astropy Time      : via .jd
      float             : GPS seconds since 1980-01-06 (OEM t_gps convention)
      decimal year      : e.g. 2025.5, if the value looks like a year
      None              : returns None
    """
    if epoch is None:
        return None
    if isinstance(epoch, datetime):
        return epoch
    if hasattr(epoch, "utc") and hasattr(epoch, "jd"):  # astropy Time, any scale
        return epoch.utc.to_datetime() if np.ndim(epoch.jd) == 0 else epoch[0].utc.to_datetime()
    v = float(np.asarray(epoch).flat[0])
    if 1900.0 < v < 2200.0:                       # decimal year
        year = int(v)
        frac = v - year
        import calendar
        days = 366 if calendar.isleap(year) else 365
        from datetime import timedelta
        return datetime(year, 1, 1) + timedelta(days=frac*days)
    # GPS -> UTC with the leap-second table; GPS - UTC was 18 s only from 2017.
    from astropy.time import Time
    return Time(v, format="gps").utc.to_datetime()


def _jd_to_datetime(jd: float) -> datetime:
    from datetime import timedelta
    return datetime(2000, 1, 1, 12) + timedelta(days=jd - 2451545.0)


# ===========================================================================
# Frames
# ===========================================================================

def _precession_matrix(when) -> np.ndarray:
    """
    IAU 1976 (Lieske) precession, J2000 -> mean equator of date.

    General precession is ~50 arcsec/yr, so by 2026 the accumulated shift is
    ~0.36 deg — several pixels in a rendered sky, and enough to matter when the
    scene is registered against Earth's surface.
    """
    d = when if isinstance(when, datetime) else _to_datetime(when)
    T = (_julian_date(d) - 2451545.0) / 36525.0
    zeta  = (2306.2181*T + 0.30188*T*T + 0.017998*T**3) * ARCSEC
    z     = (2306.2181*T + 1.09468*T*T + 0.018203*T**3) * ARCSEC
    theta = (2004.3109*T - 0.42665*T*T - 0.041833*T**3) * ARCSEC

    def R3(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def R2(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    return R3(-z) @ R2(theta) @ R3(-zeta)


def _apply_frame(ra_hours, dec_deg, pmra_mas, pmdec_mas, when, frame):
    """Catalogue RA/Dec -> unit vectors in the requested frame.

    The catalogue is ICRS at epoch J2000.0, so its directions are already GCRF
    (frame bias 0.02 arcsec). Frames:

    ``"j2000"``  catalogue directions, no proper motion.
    ``"gcrf"``   proper motion to ``when``; no precession.
    ``"teme"``   GCRF rotated by ``ssapy.utils.gcrf_to_teme`` (the frame the
                 satellite viewer, SGP4 and its GMST-rotated Earth use).
    ``"mod"``    IAU 1976 precession to the mean equator and equinox of date.
    ``"ecef"``   GCRF rotated to ITRF with
                 :func:`ssatk.coordinates.frames.eci_to_ecf_matrix`.

    ``"gcrf"`` used to apply precession, i.e. it returned mean-of-date
    directions: 1142 arcsec median (1345 max) from GCRF for the 50 stars
    brighter than mag 2 on 2026-10-09.
    """
    ra = np.radians(np.asarray(ra_hours, float) * 15.0)
    dec = np.radians(np.asarray(dec_deg, float))
    frame = (frame or "j2000").lower()
    if frame not in ("j2000", "gcrf", "teme", "mod", "ecef"):
        raise ValueError(f"unknown star frame {frame!r}; use j2000, gcrf, teme, mod or ecef")

    if frame != "j2000" and when is not None:
        yrs = (_julian_date(when) - 2451545.0) / 365.25
        # pmra is mu_alpha* (already carries cos(dec))
        ra = ra + np.radians(np.asarray(pmra_mas, float)/3.6e6) * yrs / np.cos(dec)
        dec = dec + np.radians(np.asarray(pmdec_mas, float)/3.6e6) * yrs

    v = np.stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], axis=1)
    if when is None or frame in ("j2000", "gcrf"):
        return v

    if frame == "mod":
        return v @ _precession_matrix(when).T

    from astropy.time import Time

    t = Time(when if isinstance(when, datetime) else _to_datetime(when), scale="utc")
    if frame == "teme":
        from ssapy.utils import gcrf_to_teme

        return v @ np.asarray(gcrf_to_teme(t), dtype=float).reshape(3, 3).T
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    return v @ eci_to_ecf_matrix(float(t.gps)).T


def _stars_to_ecef(ra_hours, dec_deg, pmra_mas, pmdec_mas, date,
                   apply_pm=True, apply_prec=True, apply_rot=True):
    """
    Explicit-step form of _apply_frame, kept because the steps are independently
    switchable — needed to isolate one effect at a time when testing, e.g.
    checking that a star at RA = GMST lands on longitude 0 with precession off.
    The `frame` argument cannot express that, since 'ecef' implies precession.
    """
    ra = np.radians(np.asarray(ra_hours, float) * 15.0)
    dec = np.radians(np.asarray(dec_deg, float))
    if apply_pm and date is not None:
        yrs = (_julian_date(date) - 2451545.0) / 365.25
        ra = ra + np.radians(np.asarray(pmra_mas, float)/3.6e6) * yrs / np.cos(dec)
        dec = dec + np.radians(np.asarray(pmdec_mas, float)/3.6e6) * yrs
    v = np.stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], axis=1)
    if apply_prec and date is not None:
        v = v @ _precession_matrix(date).T
    if apply_rot and date is not None:
        g = _gmst_rad(date)
        c, s_ = np.cos(g), np.sin(g)
        v = v @ np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]]).T
    return v


# ===========================================================================
# Colour
# ===========================================================================

def _bv_to_teff(bv):
    """Ballesteros (2012) blackbody relation."""
    bv = np.clip(np.asarray(bv, float), -0.4, 2.0)
    return 4600.0 * (1.0/(0.92*bv + 1.7) + 1.0/(0.92*bv + 0.62))


def _cie_xyz_bar(lam):
    """Wyman/Sloan/Shirley (2013) fits to the CIE 1931 colour matching functions."""
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5*((x-mu)/s)**2)
    x = 1.056*g(lam,599.8,37.9,31.0) + 0.362*g(lam,442.0,16.0,26.7) - 0.065*g(lam,501.1,20.4,26.2)
    y = 0.821*g(lam,568.8,46.9,40.5) + 0.286*g(lam,530.9,16.3,31.1)
    z = 1.217*g(lam,437.0,11.8,36.0) + 0.681*g(lam,459.0,26.0,13.8)
    return x, y, z


def _teff_to_srgb(teff):
    """Planck spectrum -> CIE XYZ -> sRGB, normalised to constant luminance."""
    lam = np.linspace(360.0, 830.0, 236)
    l_m = lam*1e-9
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    T = np.atleast_1d(np.asarray(teff, float))[:, None]
    B = (2*h*c**2 / l_m**5) / (np.exp(h*c/(l_m*kB*T)) - 1.0)
    xb, yb, zb = _cie_xyz_bar(lam)
    X = np.trapezoid(B*xb, lam, axis=1)
    Y = np.trapezoid(B*yb, lam, axis=1)
    Z = np.trapezoid(B*zb, lam, axis=1)
    s = X+Y+Z
    X, Y, Z = X/s, Y/s, Z/s
    M = np.array([[ 3.2406,-1.5372,-0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [ 0.0557,-0.2040, 1.0570]])
    rgb = np.clip(np.stack([X, Y, Z], 1) @ M.T, 0, None)
    rgb = rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-12)
    srgb = np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def _spect_fallback_rgb(spect):
    return _SPECT_COLORS.get((spect or 'G')[:1], _SPECT_COLORS['G'])


# ===========================================================================
# Catalogue
# ===========================================================================

def _catalog_path(catalog_path=None):
    if catalog_path and os.path.exists(os.path.expanduser(str(catalog_path))):
        return os.path.expanduser(str(catalog_path))

    for name in ("bright_stars_mag9.csv", "bright_stars.csv"):
        try:
            path = find_data_file(name)
        except NameError:
            path = None
        if path is not None:
            return str(path)
    for p in _HYG_PATHS:
        if os.path.exists(p):
            return p
    return None


def _load_stars(mag_limit=6.5, when=None, frame="gcrf", catalog_path=None):
    """
    Load, transform and cache the star catalogue.

    Returns a dict with unit vectors `v` (N,3) in the requested frame, plus
    magnitudes, marker sizes and RGB colours — or None if no catalogue is
    installed.
    """
    catalog_key = (
        None if catalog_path is None
        else os.path.abspath(os.path.expanduser(str(catalog_path)))
    )
    key = (mag_limit, None if when is None else _julian_date(when), frame, catalog_key)
    if key in _STAR_CACHE:
        return _STAR_CACHE[key]

    path = _catalog_path(catalog_path)
    if path is None:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        need = [cols.get('ra'), cols.get('dec'), cols.get('mag')]
        if any(c is None for c in need):
            print(f"[starfield] catalogue lacks ra/dec/mag: {list(df.columns)[:8]}")
            return None
        df = df.dropna(subset=need)
        df = df[(df[need[2]] < mag_limit) & (df[need[2]] > -10)]
        n = len(df)

        def col(name, default=0.0):
            c = cols.get(name)
            return df[c].astype(float).values if c is not None else np.full(n, default)

        mag = df[need[2]].astype(float).values
        v = _apply_frame(df[need[0]].astype(float).values,
                         df[need[1]].astype(float).values,
                         col('pmra'), col('pmdec'), when, frame)

        ci = col('ci', np.nan)
        spect = (df[cols['spect']].fillna('G').astype(str).values
                 if 'spect' in cols else np.full(n, 'G'))
        rgb = np.zeros((n, 3))
        have = np.isfinite(ci)
        if have.any():
            rgb[have] = _teff_to_srgb(_bv_to_teff(ci[have]))
        if (~have).any():
            rgb[~have] = np.array([_spect_fallback_rgb(s) for s in spect[~have]])

        out = dict(v=v, mag=mag, rgb=rgb,
                   sizes=np.clip(0.9*(mag_limit - mag)**1.25, 0.4, 5.0),
                   n=n, mag_limit=mag_limit, frame=frame,
                   n_colored=int(have.sum()))
        _STAR_CACHE[key] = out
        return out
    except Exception as e:
        print(f"[starfield] Could not load catalog: {e}")
        return None


def star_directions(mag_limit=6.5, when=None, frame="gcrf"):
    """Unit vectors, magnitudes and RGB colours.  Returns (v, mag, rgb) or None."""
    s = _load_stars(mag_limit=mag_limit, when=_to_datetime(when), frame=frame)
    return None if s is None else (s['v'], s['mag'], s['rgb'])


def moon_fixed_webgl_stars(mag_limit=7.0, radius_km=4.0e6, epoch=None):
    """Return compact WebGL star arrays in the lunar body-fixed frame.

    Uses the same frame as the textured Moon in ``moon_webgl``; see
    :func:`ssatk.coordinates.lunar_body_rotation`.
    """
    from astropy.time import Time
    from ssatk.coordinates.lunar import lunar_body_rotation

    catalogue = star_directions(mag_limit=mag_limit, frame="gcrf")
    if catalogue is None:
        return None
    v, mag, rgb = (np.asarray(value, float) for value in catalogue)
    if epoch is None:
        epoch = Time.now()
    elif isinstance(epoch, (int, float, np.integer, np.floating)):
        epoch = Time(epoch, format="gps")
    else:
        epoch = Time(epoch, scale="tt")
    rotation = lunar_body_rotation(float(np.atleast_1d(epoch.gps)[0]))[0]
    positions = (v @ rotation.T) * radius_km
    brightness = np.clip(1.60 - 0.15 * mag, 0.16, 1.0)
    return {
        "p": positions.astype(np.float32).ravel().round(0).tolist(),
        "c": np.clip(rgb * brightness[:, None], 0, 1).round(3).ravel().tolist(),
        "s": np.clip(4.8 - 0.48 * mag, 1.0, 6.5).round(2).tolist(),
    }


# ===========================================================================
# Plotly
# ===========================================================================

def _hemisphere_mask(vectors, away_from):
    if away_from is None:
        return np.ones(len(vectors), dtype=bool)
    direction = np.asarray(away_from, dtype=float).reshape(3)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return np.ones(len(vectors), dtype=bool)
    return (vectors @ (direction / norm)) < 0.0


def starfield_traces(sky_radius, when=None, frame="ecef", mag_limit=6.5,
                     opacity=0.92, fallback_random=True, catalog_path=None,
                     hemisphere_away_from=None):
    """
    Star markers for a Plotly 3D scene, as a list of traces.

    `sky_radius` should be far outside the subject so the camera orbits inside
    the star sphere; near-side stars then fall behind the near clip plane
    instead of drawing over the foreground.  A factor of ~50x the scene radius
    is enough, and makes parallax negligible.
    """
    import plotly.graph_objects as go
    d = _to_datetime(when)
    s = _load_stars(mag_limit=mag_limit, when=d, frame=frame, catalog_path=catalog_path)
    if s is None:
        if not fallback_random:
            return []
        rng = np.random.default_rng(42)
        n = 4000
        th = rng.uniform(0, 2*np.pi, n)
        ph = np.arccos(rng.uniform(-1, 1, n))
        mags = rng.uniform(1.0, mag_limit, n)
        print("[starfield] catalogue not found — random placeholder sky")
        v = np.column_stack((
            np.sin(ph) * np.cos(th),
            np.sin(ph) * np.sin(th),
            np.cos(ph),
        ))
        mask = _hemisphere_mask(v, hemisphere_away_from)
        v = v[mask]
        mags = mags[mask]
        return [go.Scatter3d(
            x=sky_radius*v[:, 0], y=sky_radius*v[:, 1], z=sky_radius*v[:, 2],
            mode='markers',
            marker=dict(size=np.clip(0.9*(mag_limit-mags)**1.25, 0.4, 5.0),
                        color='white', opacity=0.75),
            hoverinfo='none', showlegend=False, name='Stars')]
    v = s['v']
    mask = _hemisphere_mask(v, hemisphere_away_from)
    v = v[mask]
    rgb = s['rgb'][mask]
    sizes = s['sizes'][mask]
    cstrs = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgb]
    if d is not None:
        print(f"  starfield: {s['n']} stars, {frame.upper()} at "
              f"{d.strftime('%Y-%m-%d %H:%M')} UT "
              f"(GMST {np.degrees(_gmst_rad(d))/15:.3f} h), "
              f"{s['n_colored']} coloured from B-V", flush=True)
    return [go.Scatter3d(
        x=v[:, 0]*sky_radius, y=v[:, 1]*sky_radius, z=v[:, 2]*sky_radius,
        mode='markers',
        marker=dict(size=sizes, color=cstrs, opacity=opacity),
        hoverinfo='none', showlegend=False, name='Stars')]


# ===========================================================================
# Matplotlib
# ===========================================================================

def _camera_direction(elev_deg, azim_deg):
    """Unit vector from camera toward origin."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    cam = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    look = -cam
    return look / np.linalg.norm(look)


def add_starfield(ax, plot_range, elev=30, azim=45,
                  fov=360, mag_limit=6.5,
                  show_milky_way=True,
                  epoch=None,
                  frame="gcrf",
                  depth_variation=False):
    """
    Add a star background to a matplotlib 3D axes.

    Parameters
    ----------
    ax, plot_range   : axes and scene half-extent (sets the sky radius)
    elev, azim       : camera angles, used for the field-of-view cut
    fov              : degrees; 360 keeps the whole sky
    mag_limit        : magnitude cutoff
    show_milky_way   : draw the galactic band
    epoch            : datetime, astropy Time, GPS seconds, or decimal year
    frame            : 'gcrf' (default, inertial) or 'ecef' — use 'ecef' when
                       the scene also contains Earth's surface
    depth_variation  : place stars at 0.5-1.0x the sky radius by magnitude.
                       Off by default: it makes the sky non-spherical and
                       introduces a parallax artifact when the camera rotates,
                       which starfield_verification.py flagged as a known
                       issue.  Kept for reproducing older figures.
    """
    s = _load_stars(mag_limit=mag_limit, when=_to_datetime(epoch), frame=frame)
    if s is None:
        return

    sky_radius = plot_range * 4.0
    v = s['v']

    look = _camera_direction(elev, azim)
    mask = (v @ (-look)) > np.cos(np.radians(fov / 2.0))
    if not mask.any():
        return

    r = np.full(mask.sum(), sky_radius, dtype=float)
    if depth_variation:
        mag = s['mag'][mask]
        span = mag.max() - mag.min()
        r = sky_radius * (0.5 + 0.5*(mag - mag.min())/(span + 1e-6))

    ax.scatter(v[mask, 0]*r, v[mask, 1]*r, v[mask, 2]*r,
               s=s['sizes'][mask]*2.0, c=s['rgb'][mask],
               alpha=0.75, depthshade=False, linewidths=0)

    if show_milky_way:
        _add_milky_way(ax, sky_radius)


def _add_milky_way(ax, sky_radius):
    """Galactic band, as a set of small circles about the galactic pole."""
    gnp = np.array([
        np.cos(np.radians(27.13)) * np.cos(np.radians(192.85)),
        np.cos(np.radians(27.13)) * np.sin(np.radians(192.85)),
        np.sin(np.radians(27.13)),
    ])
    arb = np.array([0., 1., 0.])
    v1 = np.cross(gnp, arb); v1 /= np.linalg.norm(v1)
    theta = np.linspace(0, 2*np.pi, 60)
    for w, a in [(0., 0.08), (0.1, 0.05), (0.2, 0.03), (-0.1, 0.05), (-0.2, 0.03)]:
        n_ = gnp + w * v1
        n_ /= np.linalg.norm(n_)
        b1 = np.cross(n_, arb)
        if np.linalg.norm(b1) < 1e-6:
            b1 = np.cross(n_, np.array([1, 0, 0]))
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(n_, b1); b2 /= np.linalg.norm(b2)
        pts = (np.outer(np.cos(theta), b1) + np.outer(np.sin(theta), b2)) * sky_radius
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='#8899dd', alpha=a, linewidth=0.5)


# ---------------------------------------------------------------------------
# Data-asset resolution (split SSATK data packages)
# ---------------------------------------------------------------------------
# Large binary assets -- the HYG star catalogue, the AE-8/AP-8 flux table,
# planetary textures -- are not carried in this repo. They ship in the split
# ``ssatk-data-*`` distributions (``ssatk-data-core`` holds the star catalogue
# and Earth textures), each installing its own ``ssapy_data_<name>`` package
# with files below ``<package>/data``. See ``ssatk.data``.
#
# Resolution order, first hit wins:
#   1. $SSAPY_DATA                        -- explicit override for CI / odd layouts
#   2. a sibling data checkout            -- useful while developing both repos
#      side by side, before `pip install -e` has been run against the data repo
#   3. the installed split data packages  -- the real, supported user mechanism
#      (then the retired ``ssapy_data`` archive package, if still installed)
#   4. alongside this module              -- legacy in-tree assets, so existing
#                                            working copies keep functioning
#
# Everything degrades: find_data_file() returns None rather than raising, and
# callers fall back (procedural textures, no belt surfaces, a synthetic
# starfield) when an asset is absent. Callers that fall back should say so.


def _ssapy_data_package_dirs():
    """Directories of the installed split data packages, in search order.

    The split packages expose data via importlib.resources. For a normal
    (unzipped) install the ``data`` directories are stable on disk, which is
    what the filename searches below need.
    """
    import importlib.resources as _res
    from pathlib import Path as _P

    from ssatk.data import DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE

    dirs = []
    for package in (*DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE):
        try:
            p = _P(str(_res.files(package) / "data"))
        except (ImportError, TypeError):
            continue
        if p.is_dir():
            dirs.append(p)
    return dirs


def _ssapy_data_package_dir():
    """First installed data-package directory, or None (backward compatible)."""
    dirs = _ssapy_data_package_dirs()
    return dirs[0] if dirs else None


def ssapy_data_dirs():
    """Return the directories searched for data assets, in priority order.

    Public rather than private because it is what you want printed when an
    asset can't be found: "not found" is not actionable, "not found in these
    four places" is. `ssatk-magnetosphere --check` prints it.
    """
    import os as _os
    from pathlib import Path as _Path

    here = _Path(__file__).resolve()
    plots_dir = here.parent          # ssatk/plots
    repo_root = plots_dir.parent.parent   # repository checkout root
    repo_parent = repo_root.parent        # directory holding the sibling repos

    dirs = []

    env = _os.environ.get("SSAPY_DATA")
    if env:
        dirs.append(_Path(env))

    # Sibling checkouts of the core data repository, raw or in packaged layout.
    for parent in (repo_parent, repo_root):
        for name in ("ssatk-data-core", "data-core"):
            dirs.append(parent / name)
            dirs.append(parent / name / "src" / "ssapy_data_core" / "data")

    dirs.extend(_ssapy_data_package_dirs())

    dirs.append(plots_dir)

    seen, unique = set(), []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def find_data_file(name):
    """Locate a data asset by filename, or return None if it isn't anywhere.

    Returns a pathlib.Path. None means "not present" -- callers degrade rather
    than fail, which is why this does not raise.
    """
    for d in ssapy_data_dirs():
        try:
            candidate = d / name
            if candidate.is_file():
                return candidate
        except OSError:
            # A bad $SSAPY_DATA or a stale network mount should not take out
            # the whole search.
            continue
    return None
