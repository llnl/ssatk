"""
test_magfield_physics.py — physics regression suite for the SSATK
magnetosphere plots.

Every check here corresponds to a property that was verified by hand during
development.  They exist because several real bugs in these modules were the
kind that produce a plausible-looking figure:

  * geocentric coordinates passed to a geodetic IGRF entry point
  * a star catalogue plotted in equatorial axes inside an Earth-fixed scene
  * the GEO->GSM rotation applied transposed

The last one survived a round of testing because the test derived "nightside"
from the same transposed matrix — self-consistently wrong.  The frame tests
below therefore check against *independent* references (geopack's own inverse
call, and a solar-position calculation that does not use the matrix at all).

Run:
    pytest -q -m "not slow"          47 checks, ~50 s
    pytest -q -m slow                 6 cold-cache / subprocess checks, ~4 min
    python test_magfield_physics.py   same, without pytest installed (--slow to include)

The `slow` marker is registered in tests/conftest.py.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# Package imports. These were bare flat imports (`import magfield_plot_3d`),
# which only resolved when the interpreter happened to be started from inside
# ssatk/plots/ -- so `pytest tests/test_magfield_physics.py` from the
# repo root failed at collection with ModuleNotFoundError. The fallback keeps
# the "run this file directly" mode advertised in the docstring working.
try:
    from ssatk.plots import magfield_plot_3d as mf
    from ssatk.plots import magnetosphere_core as core
    from ssatk.plots import starfield as sf
except ImportError:
    import magfield_plot_3d as mf
    try:
        from ssatk.plots import magnetosphere_core as core
    except ImportError:
        import magnetosphere_core as core
    import starfield as sf

DATE = datetime(2025, 7, 2)
RE = core.EARTH_RADIUS_KM

_HAS_GEOPACK = getattr(mf, "_HAS_GEOPACK", False)
_HAS_PPIGRF = getattr(mf, "_HAS_PPIGRF", False)
try:
    import spacepy.irbempy as _ib
    _HAS_SPACEPY = True
except Exception:
    _HAS_SPACEPY = False

needs_geopack = pytest.mark.skipif(not _HAS_GEOPACK, reason="geopack not installed")
needs_ppigrf = pytest.mark.skipif(not _HAS_PPIGRF, reason="ppigrf not installed")
needs_spacepy = pytest.mark.skipif(not _HAS_SPACEPY, reason="spacepy/IRBEM not installed")


# ---------------------------------------------------------------------------
# Reference frames  (the class of bug that hurt most)
# ---------------------------------------------------------------------------

@needs_geopack
def test_gsm_matrix_is_a_rotation():
    M = mf._gsm_frame(DATE)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-9), "GEO->GSM must be orthonormal"
    assert abs(np.linalg.det(M) - 1.0) < 1e-9, "must be a proper rotation, not a reflection"


@needs_geopack
def test_gsm_matrix_orientation_against_geopack():
    """M must satisfy v_gsm = M @ v_geo, checked against geopack's own call."""
    from geopack import geopack as gp
    ut = (DATE - datetime(1970, 1, 1)).total_seconds()
    gp.recalc(ut)
    M = mf._gsm_frame(DATE)
    v = np.array([0.3, -0.5, 0.81]); v /= np.linalg.norm(v)
    truth = np.array(gp.geogsm(*v, 1))
    assert np.allclose(M @ v, truth, atol=1e-9), "M @ v_geo must equal geogsm(v_geo)"
    # and the transpose must NOT satisfy it — guards against re-introducing the bug
    assert not np.allclose(M.T @ v, truth, atol=1e-6)


def test_sun_direction_matches_independent_solar_position():
    """
    The sunward axis in GEO must agree with the subsolar point computed from
    an ephemeris that never touches the rotation matrix.  This is the check
    that exposed the transposed frame (it was 1.74 deg out).
    """
    sun = mf._sun_direction_geo(DATE)
    dec, lon = core._subsolar_point(DATE)
    ref = np.array([math.cos(math.radians(dec)) * math.cos(math.radians(lon)),
                    math.cos(math.radians(dec)) * math.sin(math.radians(lon)),
                    math.sin(math.radians(dec))])
    sep = math.degrees(math.acos(float(np.clip(sun @ ref, -1, 1))))
    assert sep < 0.05, f"sunward axis is {sep:.3f} deg from the subsolar point"


@needs_geopack
def test_t89_external_model_evaluates_at_the_right_place():
    """
    Decisive frame test: take a point defined in GSM, express it in GEO, and
    push it through the interpolating model.  The model converts back to
    GSM internally, so the answer must match t89 called directly on the
    original GSM point.  A transposed rotation cannot pass this.
    """
    from geopack import t89
    grid = mf._get_external(DATE, kp=3, model="t89")
    M = grid.M
    for gsm in ([6.0, 1.0, 2.0], [-9.0, -2.0, 1.5], [4.0, -3.0, -2.0]):
        gsm = np.array(gsm, dtype=float)
        geo_km = (gsm @ M) * 6371.2                      # GSM -> GEO (row vector)
        got = grid(geo_km[None, :])[0]                   # GEO -> GSM -> field -> GEO
        want_gsm = np.array(t89.t89(grid.iopt, grid.ps, *gsm))
        want_geo = want_gsm @ M                          # GSM -> GEO
        err = np.linalg.norm(got - want_geo)
        assert err < 0.05 * max(np.linalg.norm(want_geo), 1.0), (
            f"T89 wrapper off by {err:.3f} nT at GSM {gsm}")


@needs_geopack
def test_magnetopause_nose_points_at_the_sun():
    sun = mf._sun_direction_geo(DATE)
    pts, _, _, _, r0, _ = mf._shue_magnetopause(DATE, n_theta=10, n_phi=12)
    nose = pts[np.argmax(pts @ sun)]
    ang = math.degrees(math.acos(float(np.clip(
        (nose / np.linalg.norm(nose)) @ sun, -1, 1))))
    assert ang < 1.0, f"magnetopause nose is {ang:.2f} deg off the Sun line"
    assert abs(np.linalg.norm(nose) / RE - r0) < 0.05


def test_shue_standoff_and_its_response():
    """Nominal standoff ~10-11 RE; southward Bz and higher pressure compress it."""
    r_nom = mf._shue_magnetopause(DATE, bz_nT=0.0, dp_nPa=2.0, n_theta=4, n_phi=4)[4]
    r_south = mf._shue_magnetopause(DATE, bz_nT=-5.0, dp_nPa=2.0, n_theta=4, n_phi=4)[4]
    r_press = mf._shue_magnetopause(DATE, bz_nT=0.0, dp_nPa=8.0, n_theta=4, n_phi=4)[4]
    assert 10.0 < r_nom < 11.0, f"nominal standoff {r_nom:.2f} RE outside 10-11"
    assert r_south < r_nom, "southward Bz must erode the dayside"
    assert r_press < r_nom, "higher dynamic pressure must compress the boundary"


# ---------------------------------------------------------------------------
# Internal field
# ---------------------------------------------------------------------------

@needs_ppigrf
def test_bfield_matches_geocentric_reference():
    """_bfield_batch must equal ppigrf's geocentric synthesis exactly."""
    import ppigrf.ppigrf as pp
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P *= rng.uniform(RE + 300, 6 * RE, 200)[:, None]
    saved = mf.set_external_model(None)
    try:
        got = mf._bfield_batch(P, DATE)
    finally:
        mf.set_external_model(saved)
    r = np.linalg.norm(P, axis=1)
    gclat = np.degrees(np.arcsin(P[:, 2] / r))
    lon = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    Br, Bt, Bp = [np.asarray(x).flatten() for x in pp.igrf_gc(r, 90 - gclat, lon, DATE)]
    th, ph = np.radians(90 - gclat), np.radians(lon)
    want = np.stack([Br*np.sin(th)*np.cos(ph) + Bt*np.cos(th)*np.cos(ph) - Bp*np.sin(ph),
                     Br*np.sin(th)*np.sin(ph) + Bt*np.cos(th)*np.sin(ph) + Bp*np.cos(ph),
                     Br*np.cos(th) - Bt*np.sin(th)], axis=1)
    assert np.abs(got - want).max() < 1e-6


@needs_ppigrf
def test_south_atlantic_anomaly_location():
    """|B| minimum at 500 km must fall in the South Atlantic."""
    lon = np.linspace(-180, 180, 121)
    lat = np.linspace(-70, 70, 57)
    LO, LA = np.meshgrid(lon, lat)
    r = RE + 500.0
    P = np.stack([r*np.cos(np.radians(LA))*np.cos(np.radians(LO)),
                  r*np.cos(np.radians(LA))*np.sin(np.radians(LO)),
                  r*np.sin(np.radians(LA))], axis=-1).reshape(-1, 3)
    saved = mf.set_external_model(None)
    try:
        B = np.linalg.norm(mf._bfield_batch(P, DATE), axis=1).reshape(LA.shape)
    finally:
        mf.set_external_model(saved)
    k = np.unravel_index(np.argmin(B), B.shape)
    assert -75 < LO[k] < -30, f"SAA longitude {LO[k]:.1f} outside the South Atlantic"
    assert -40 < LA[k] < -5, f"SAA latitude {LA[k]:.1f} outside the South Atlantic"
    assert 15e3 < B[k] < 24e3, f"SAA |B| {B[k]:.0f} nT implausible at 500 km"


# ---------------------------------------------------------------------------
# Geometry and integration
# ---------------------------------------------------------------------------

def test_wgs84_flattening():
    mesh = core._build_earth_mesh("none", n_lon=90, n_lat=45)
    r = np.sqrt(np.asarray(mesh.x)**2 + np.asarray(mesh.y)**2 + np.asarray(mesh.z)**2)
    f = 1.0 - r.min() / r.max()
    assert abs(f - 1/298.257223563) < 2e-5, f"flattening {f:.6f}"
    assert abs(r.min() - core.WGS84_B_KM) < 1e-3


def test_surface_radius_is_the_ellipsoid():
    eq = mf._surface_radius_km(np.array([[RE, 0.0, 0.0]]))[0]
    pole = mf._surface_radius_km(np.array([[0.0, 0.0, RE]]))[0]
    assert abs(eq - core.WGS84_A_KM) < 1e-6
    assert abs(pole - core.WGS84_B_KM) < 1e-6


def test_tracer_converges_under_step_halving():
    saved = mf.set_external_model(None)
    try:
        seed = mf._make_seeds_magnetic([45.0], n_lons=1)
        coarse = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=8, step_max=220,
                                     max_steps=9000)[0]
        fine = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=4, step_max=110,
                                   max_steps=20000)[0]
    finally:
        mf.set_external_model(saved)
    assert np.linalg.norm(coarse[-1] - fine[-1]) < 1.0, "RK4 endpoints must agree within 1 km"


@needs_ppigrf
def test_trace_terminates_on_the_ellipsoid():
    saved = mf.set_external_model(None)
    try:
        seed = mf._make_seeds_magnetic([50.0], n_lons=2)
        lines = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=8, step_max=220,
                                    max_steps=9000)
    finally:
        mf.set_external_model(saved)
    for L in lines:
        end = L[-1:]
        assert abs(np.linalg.norm(end) - mf._surface_radius_km(end)[0]) < 0.01


# ---------------------------------------------------------------------------
# Trapped particles
# ---------------------------------------------------------------------------


def test_flux_ratio_matches_closed_form():
    """
    The trapped-density profile is my own derivation, so check it against the
    cases that integrate analytically.  For j(a_eq) ~ sin^n(a_eq) the surviving
    fraction is int_0^amax sin^(n+1) / int_0^(pi/2) sin^(n+1), with
    sin(amax) = sqrt(Beq/B):

        n = 0 :  1 - cos(amax)              = 1 - sqrt(1 - Beq/B)
        n = 1 :  (amax - sin amax cos amax) / (pi/2)
    """
    b = np.array([1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0])
    amax = np.arcsin(np.sqrt(1.0 / b))
    exact0 = 1.0 - np.sqrt(1.0 - 1.0 / b)
    exact1 = (amax - np.sin(amax) * np.cos(amax)) / (np.pi / 2)
    assert np.abs(core._omnidirectional_flux_ratio(b, n=0.0) - exact0).max() < 1e-5
    assert np.abs(core._omnidirectional_flux_ratio(b, n=1.0) - exact1).max() < 1e-5


@needs_ppigrf
def test_mirror_latitude_tracks_dipole_theory():
    """Median IGRF mirror latitude must sit near the analytic dipole value."""
    from scipy.optimize import brentq
    saved = mf.set_external_model(None)
    try:
        axis = core._dipole_axis()
        for alpha, tol in ((25.0, 4.0), (40.0, 4.0)):
            theory = math.degrees(brentq(
                lambda lam: math.sqrt(1 + 3*math.sin(lam)**2)/math.cos(lam)**6
                            - 1/math.sin(math.radians(alpha))**2,
                0.01, math.radians(75)))
            b = mf._igrf_lshell_boundary(2.0, DATE, axis, n_azim=12, n_pts=40,
                                         pitch_angle_deg=alpha)
            ends = np.concatenate([b[:, 0, :], b[:, -1, :]])
            ml = np.abs(np.degrees(np.arcsin(np.clip(
                (ends @ axis) / np.linalg.norm(ends, axis=1), -1, 1))))
            assert abs(np.median(ml) - theory) < tol, (
                f"pitch {alpha}: median {np.median(ml):.1f} vs theory {theory:.1f}")
    finally:
        mf.set_external_model(saved)


@needs_spacepy
def test_aep8_reproduces_the_belt_structure():
    tab = mf._load_aep8_table(allow_build=False)
    if tab is None:
        pytest.skip("AE8/AP8 table not cached")
    L, Fp, Fe = tab['L'], tab['p'], tab['e']
    assert 1.4 <= L[np.argmax(Fp[:, 0])] <= 2.0, "AP-8 proton peak outside L 1.4-2.0"
    assert 3.8 <= L[np.argmax(Fe[:, 0])] <= 4.8, "AE-8 electron peak outside L 3.8-4.8"
    slot = (L > 1.9) & (L < 3.4)
    assert Fe[slot, 0].min() < 0.2 * Fe[:, 0].max(), "slot region not resolved"
    row = Fe[np.argmin(abs(L - 4.4))]
    assert np.all(np.diff(row) <= 0), "flux must fall with B/B0 along a field line"


@needs_spacepy
def test_mcilwain_L_matches_irbem():
    """
    Compared at 2024-07-02, not the plot epoch: IRBEM's bundled IGRF ends at
    2025 and silently clamps to the nearest year ("out of valid range ...
    Using nearest"), which would make this a comparison against a clamped
    reference rather than a real one.
    """
    import spacepy.time as spt, spacepy.coordinates as spc
    date = datetime(2024, 7, 2)
    saved = mf.set_external_model(None)
    try:
        axis = core._dipole_axis()
        _, e1, e2 = core._mag_basis(axis)
        seeds = [L*RE*(math.cos(p)*e1 + math.sin(p)*e2)
                 for L in (2.0, 3.0, 4.0) for p in (0.0, math.pi)]
        eq, B0 = mf._true_magnetic_equator(seeds, date)
    finally:
        mf.set_external_model(saved)
    mine = (mf._M_DIPOLE_NT_RE3 / B0) ** (1/3)
    t = spt.Ticktock([date]*len(eq), 'UTC')
    c = spc.Coords((eq/RE).tolist(), 'GEO', 'car', use_irbem=True); c.ticks = t
    Lm = np.abs(np.array(_ib.get_Lm(t, c, [90.0], extMag='0', intMag='IGRF')['Lm']).flatten())
    rel = np.abs(mine - Lm) / Lm
    assert rel.max() < 0.02, f"L differs from IRBEM by {100*rel.max():.2f}%"


# ---------------------------------------------------------------------------
# Astrometry
# ---------------------------------------------------------------------------

def test_julian_date_and_gmst():
    assert abs(core._julian_date(datetime(2000, 1, 1, 12)) - 2451545.0) < 1e-6
    g = math.degrees(core._gmst_rad(datetime(2000, 1, 1, 12))) / 15.0
    assert abs(g - 18.697375) < 1e-4, f"GMST at J2000 = {g:.6f} h"


def test_precession_rate():
    """Pole motion must match theta = 2004.31 arcsec/century."""
    d = datetime(2026, 7, 2)
    T = (core._julian_date(d) - 2451545.0) / 36525.0
    pole = np.array([0.0, 0.0, 1.0]) @ core._precession_matrix(d).T
    moved = math.degrees(math.acos(float(np.clip(pole @ [0, 0, 1], -1, 1)))) * 3600
    assert abs(moved - 2004.31 * T) < 1.0


def test_star_at_ra_equal_to_gmst_sits_over_greenwich():
    d = datetime(2026, 7, 2)
    g_h = math.degrees(core._gmst_rad(d)) / 15.0
    v = core._stars_to_ecef(np.array([g_h]), np.array([0.0]), np.array([0.0]),
                          np.array([0.0]), d, apply_pm=False, apply_prec=False)[0]
    assert abs(math.degrees(math.atan2(v[1], v[0]))) < 1e-4


def test_celestial_pole_maps_to_earth_rotation_axis():
    """Polaris must sit within ~1 deg of +Z in the Earth-fixed frame."""
    d = datetime(2026, 7, 2)
    v = core._stars_to_ecef(np.array([2.5303]), np.array([89.264]),
                          np.array([44.5]), np.array([-11.9]), d)[0]
    assert math.degrees(math.acos(float(np.clip(v[2], -1, 1)))) < 1.0


def test_star_directions_match_astropy():
    """
    Independent check of the whole J2000 -> Earth-fixed chain against astropy's
    ITRS transform, which shares no code with this implementation.  Residuals
    of order 10-20 arcsec are expected: nutation is deliberately omitted.
    """
    pytest.importorskip("astropy")
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, ITRS
    import astropy.units as u
    when = datetime(2026, 7, 8)
    t = Time("2026-07-08T00:00:00", scale="utc")
    assert abs(math.degrees(core._gmst_rad(when))/15
               - t.sidereal_time("mean", "greenwich").hour) * 3600 < 0.5
    for ra, dec, pmra, pmdec in ((6.7525, -16.716, -546.0, -1223.0),
                                 (18.6156, 38.784, 201.0, 287.0),
                                 (2.5303, 89.264, 44.5, -11.9),
                                 (14.2610, 19.182, -1093.4, -1999.4)):
        mine = core._stars_to_ecef(np.array([ra]), np.array([dec]),
                                   np.array([pmra]), np.array([pmdec]), when)[0]
        c = SkyCoord(ra=ra*15*u.deg, dec=dec*u.deg, distance=1e6*u.pc,
                     pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                     frame="icrs", obstime=Time("J2000")).apply_space_motion(new_obstime=t)
        ref = c.transform_to(ITRS(obstime=t)).cartesian.xyz.value
        ref = ref / np.linalg.norm(ref)
        sep = math.degrees(math.acos(float(np.clip(mine @ ref, -1, 1)))) * 3600
        assert sep < 60.0, f"star direction {sep:.0f} arcsec from astropy"


def test_solar_colour_temperature():
    """B-V = 0.65 must give roughly the solar effective temperature."""
    T = float(core._bv_to_teff(0.65))
    assert 5600 < T < 5950, f"Sun-like B-V gives {T:.0f} K"
    rgb = core._teff_to_srgb([T])[0]
    assert rgb.max() > 0.9 and rgb.min() > 0.7, "solar colour should be near-white"
    hot, cool = core._teff_to_srgb([core._bv_to_teff(0.0)])[0], core._teff_to_srgb([core._bv_to_teff(1.8)])[0]
    assert hot[2] > hot[0], "hot star must be blue-weighted"
    assert cool[0] > cool[2], "cool star must be red-weighted"


# ---------------------------------------------------------------------------
# Cross-module parity and van_allen coverage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    run_slow = "--slow" in sys.argv
    print("running without pytest; add --slow for the cold-cache checks "
          "(~4 min).  'pytest -q -m \"not slow\"' gives better output.\n")
    fails = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = list(getattr(fn, "pytestmark", []))
        skip = False
        cases = [()]
        for mk in marks:
            if mk.name == "skipif" and mk.args and mk.args[0]:
                skip = True
            if mk.name == "slow" and not run_slow:
                skip = True
            if mk.name == "parametrize":
                vals = mk.args[1]
                cases = [(v,) for v in vals]
        if skip:
            skipped += 1
            print(f"SKIP  {name}")
            continue
        for args in cases:
            tag = f"{name}{list(args) if args else ''}"
            try:
                fn(*args)
                print(f"PASS  {tag}")
            except Exception as e:
                fails += 1
                print(f"FAIL  {tag}: {type(e).__name__}: {e}")
    print(f"\n{fails} failure(s), {skipped} skipped")
    sys.exit(1 if fails else 0)
