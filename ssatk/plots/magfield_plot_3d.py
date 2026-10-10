"""
magfield_plot_3d.py — Earth magnetic field line plot for SSATK
=======================================================================
Plotly (WebGL) 3D scene with true depth testing and full interactivity.

High-fidelity model content
---------------------------
  Field lines : real IGRF (ppigrf, degree 13) traced with an adaptive-step
                RK4 integrator — small steps near the surface where the
                field varies fastest, larger steps far out.  Lines are
                terminated exactly on the ellipsoid by interpolation and
                coloured by ``|B|`` along their length.
  Van Allen   : belt shells built from REAL IGRF-traced L-shells rather
                than an axisymmetric dipole, so the inner belt reproduces
                the South Atlantic Anomaly dip.  Dipole L-shell and legacy
                torus geometries remain available.
  Earth       : WGS84 oblate ellipsoid (a = 6378.137, b = 6356.7523 km) on
                a high-resolution mesh with geodetic-correct texture
                mapping.
  Stars       : HYG catalog placed on a sphere far outside the subject, so
                the camera sits *inside* the star sphere.  Near-side stars
                therefore fall behind the near clip plane instead of
                floating in front of Earth, and parallax is negligible.

Dependencies
------------
    pip install 'ssatk[static,geomagnetics]'
"""


from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta
import calendar

import numpy as np

from ssatk._paths import output_root

try:
    from .plotutils import normalize_orbit_trajectory, plotly_orbit_trace
except ImportError:
    from plotutils import normalize_orbit_trajectory, plotly_orbit_trace

# Shared geometry, Earth, sky and texture helpers live in one module so the
# two plot modules cannot drift apart (they did, repeatedly, when duplicated).
# Imported explicitly rather than with * so static analysis can still see
# undefined names — the gate that caught the last round of bugs.
try:
    from .magnetosphere_core import (
    EARTH_RADIUS_KM, PLOTLY_CONFIG, STAR_SPHERE_FACTOR, _ACCENT_WARM, _BG, _CAM_FILL, _FIELD_CMAP,
    _HAS_PLOTLY, _INK_FAINT, _INK_SOFT, _MAG_NORTH, _MAG_SOUTH,
    _atmosphere_traces, _belt_vertex_color, _build_earth_mesh,
    _build_starfield_trace, _camera_eye, _dipole_axis, _dipole_belt_mesh,
    _geo_to_xyz, _mag_basis, _mag_equator_ring, _omnidirectional_flux_ratio,
    _texture_cache_dir, _torus_mesh3d, go)
except ImportError:
    from magnetosphere_core import (
    EARTH_RADIUS_KM, PLOTLY_CONFIG, STAR_SPHERE_FACTOR, _ACCENT_WARM, _BG, _CAM_FILL, _FIELD_CMAP,
    _HAS_PLOTLY, _INK_FAINT, _INK_SOFT, _MAG_NORTH, _MAG_SOUTH,
    _atmosphere_traces, _belt_vertex_color, _build_earth_mesh,
    _build_starfield_trace, _camera_eye, _dipole_axis, _dipole_belt_mesh,
    _geo_to_xyz, _mag_basis, _mag_equator_ring, _omnidirectional_flux_ratio,
    _texture_cache_dir, _torus_mesh3d, go)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
# The geomagnetic field, tracing, radiation-belt and magnetopause code that
# used to live in this file now lives in ssatk/geomagnetics.py. It was
# separated because the magfield_verification_*.py scripts imported THIS
# module purely to reach those private helpers, which meant validating physics
# required importing Plotly and the whole rendering stack.
#
# The names are re-exported below so existing callers keep working unchanged.
#
# IMPORTANT: `_EXTERNAL_MODEL` is deliberately NOT re-exported. It is mutable
# module state, and `from geomagnetics import _EXTERNAL_MODEL` would bind a
# copy of the reference -- so `magfield_plot_3d._EXTERNAL_MODEL = None` would
# rebind only this module's name while the physics kept using the old value,
# silently producing wrong numbers. Use geomagnetics.set_external_model()
# instead; `_EXTERNAL_MODEL` below is a read-only convenience via __getattr__.
# geomagnetics is a sibling module in this package. It previously lived at the
# ssatk package root, which required a sys.path bootstrap here for
# script mode and -- worse -- created a circular import: reaching it pulled in
# ssatk.plots, whose __init__ auto-imports this module, which imports
# back into a half-initialised geomagnetics. A plain sibling import has neither
# problem.
try:
    from .. import geomagnetics as _geo
except ImportError:
    # Running as a script (`python .../magfield_plot_3d.py`) gives no package
    # context, so the relative import fails -- and a bare `import geomagnetics`
    # cannot work either, because geomagnetics.py sits at the ssatk
    # package root, not beside this file. The GUI launches this module exactly
    # that way, so put the repo root on sys.path and import by its real name.
    import sys as _sys_boot
    from pathlib import Path as _Path_boot
    _repo_root = str(_Path_boot(__file__).resolve().parents[2])
    if _repo_root not in _sys_boot.path:
        _sys_boot.path.insert(0, _repo_root)
    from ssatk import geomagnetics as _geo

# Explicit imports rather than a runtime globals() loop: ruff (and IDEs, and
# readers) cannot see names injected dynamically, so the loop produced 16
# spurious F821 "undefined name" errors and hid one real one -- `_pp` was used
# below but never re-exported, which would have raised NameError the first
# time that branch ran.
# Both spellings, so this resolves whether imported as a package submodule
# or executed directly as a script. sys.path was extended above, so the
# absolute form works in script mode.
try:
    from ..geomagnetics import (  # noqa: F401
    _geo_to_gsm_matrix, _get_external, _get_t89, _enu_to_cartesian_batch,
    _bfield_batch, _bunit_batch, _field_magnitude_along, _surface_radius_km,
    _adaptive_step_km, _trace_batch_rk4, _trace_all_closed, _resample_curve,
    _make_seeds_lshell, _make_seeds_magnetic, _physics_fingerprint,
    _aep8_table_path, _build_aep8_table, _load_aep8_table, _aep8_lookup,
    _true_magnetic_equator, _belt_flux_samples, _load_omni_year,
    get_solar_wind, _apply_solar_wind, _gsm_frame, _sun_direction_geo,
    _shue_magnetopause, _classify_line, _T89Grid,
    _M_DIPOLE_NT_RE3, _HAS_PPIGRF, _HAS_GEOPACK, _pp,
    set_external_model, get_external_model,
)
except ImportError:
    from ssatk.geomagnetics import (  # noqa: F401
    _geo_to_gsm_matrix, _get_external, _get_t89, _enu_to_cartesian_batch,
    _bfield_batch, _bunit_batch, _field_magnitude_along, _surface_radius_km,
    _adaptive_step_km, _trace_batch_rk4, _trace_all_closed, _resample_curve,
    _make_seeds_lshell, _make_seeds_magnetic, _physics_fingerprint,
    _aep8_table_path, _build_aep8_table, _load_aep8_table, _aep8_lookup,
    _true_magnetic_equator, _belt_flux_samples, _load_omni_year,
    get_solar_wind, _apply_solar_wind, _gsm_frame, _sun_direction_geo,
    _shue_magnetopause, _classify_line, _T89Grid,
    _M_DIPOLE_NT_RE3, _HAS_PPIGRF, _HAS_GEOPACK, _pp,
    set_external_model, get_external_model,
)


def __getattr__(name):
    """Read-only passthrough for the mutable state that cannot be re-exported."""
    if name == "_EXTERNAL_MODEL":
        return _geo.get_external_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Assigning `magfield_plot_3d._EXTERNAL_MODEL = ...` used to be the way to
# switch the external field off. It now cannot work -- the state lives in
# geomagnetics, and a plain assignment here would create a shadowing attribute
# that silently diverges from the value the physics actually reads. Worse, the
# shadow also defeats the __getattr__ above, so even *reading* it afterwards
# returns the wrong thing.
#
# Rather than let that fail silently (it previously produced field residuals
# computed with the external model still active, with no error raised), make
# the assignment raise with a pointer to the correct call.
import sys as _sys
import types as _types


class _GuardedModule(_types.ModuleType):
    def __setattr__(self, name, value):
        if name == "_EXTERNAL_MODEL":
            raise AttributeError(
                "Assigning magfield_plot_3d._EXTERNAL_MODEL has no effect on the "
                "physics -- the state lives in ssatk.geomagnetics. Use "
                "geomagnetics.set_external_model(value), which returns the "
                "previous model so you can restore it in a finally block."
            )
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _GuardedModule



            # mean radius (used for RE / L-shell units)
          # equatorial radius
    # polar radius

# Star sphere sits this many times the subject radius away.  Anything >= ~10
# puts the camera inside the sphere so near-side stars are clipped; 50 makes
# parallax negligible (~2%).

# Camera eye distance in normalised box units when framing the whole subject.
# ---------------------------------------------------------------------------
# Earth texture.  texture_path="auto" (the default) searches these locations,
# then falls back to downloading NASA Blue Marble into a local cache once.
# Drop your own equirectangular image at any of these paths to override.
# ---------------------------------------------------------------------------

# IGRF 2025 magnetic dip pole positions (lon_deg, lat_deg)

# Editorial palette




# Interactive modebar / zoom behaviour for saved HTML
# ===========================================================================
# Fast IGRF — precompute the epoch coefficient interpolation once
# ===========================================================================
# ppigrf.igrf() re-reads the .shc file and re-interpolates the Gauss
# coefficients to the requested date on EVERY call (~25 ms).  During a trace
# the epoch is fixed and we call it tens of thousands of times, so cache the
# date-dependent work and reuse ppigrf's own Legendre synthesis per point.
# Output is bit-identical to ppigrf.igrf(); measured ~8-9x faster.




# ===========================================================================
# External field — Tsyganenko T89 (magnetopause, ring and tail currents)
# ===========================================================================
# IGRF is an INTERNAL field model.  Beyond a few RE the magnetosphere is
# shaped by external current systems, and internal-only field lines close
# symmetrically instead of stretching into the tail.  Measured with T89
# (Kp=3) the external contribution is 1.7% of the internal field at 4 RE
# dayside, 36% at 6.6 RE nightside, 96% at 10 RE and 249% at 15 RE — so any
# figure drawn past ~5 RE on internal field alone is wrong.
#
# geopack's t89 is scalar Python (~44 us/call), far too slow inside an RK4
# trace.  It is therefore sampled once onto a GSM grid and trilinearly
# interpolated; measured error is 0.30% mean / 0.85% (95th pct) of |B_ext|,
# well inside the model's own uncertainty.












# ===========================================================================
# IGRF field helpers
# ===========================================================================









# ===========================================================================
# Adaptive-step field line tracing
# ===========================================================================











# ===========================================================================
# Star catalog
# ===========================================================================

# ---------------------------------------------------------------------------
# Astrometry — J2000 catalogue to Earth-fixed, and physical star colour.
# The scene is Earth-fixed (Greenwich on +X, matching IGRF and the texture),
# but catalogues are J2000 equatorial.  Plotting RA/Dec straight into this
# frame leaves the sky misaligned by GMST: measured median 68.6 deg, max 80.3.
# ---------------------------------------------------------------------------
     # 240 s per degree

# ---------------------------------------------------------------- colour
# ===========================================================================
# Earth — WGS84 ellipsoid, high-resolution, geodetic texture mapping
# ===========================================================================


# ===========================================================================
# Helpers
# ===========================================================================

# ===========================================================================
# Magnetic geometry
# ===========================================================================


def _igrf_lshell_boundary(L, date, axis, n_azim, n_pts, pitch_angle_deg=25.0,
                          loss_cone_alt_km=100.0, lat_max_deg=None):
    """
    Trace REAL IGRF field lines outward from the magnetic-equator crossing at
    r = L*RE for n_azim magnetic longitudes.

    Each line is followed until the field strength reaches the mirror value
    for an equatorial pitch angle `pitch_angle_deg`,

        B_mirror = B_equator / sin^2(alpha_eq)

    which is where a trapped particle turns around, or until the line drops
    into the atmosphere (`loss_cone_alt_km`) — the loss cone.  Because IGRF is
    not axisymmetric, both bounds vary with longitude, which is what produces
    the South Atlantic Anomaly.  Returns (n_azim, n_pts, 3).
    """
    RE = EARTH_RADIUS_KM
    _, e1, e2 = _mag_basis(axis)
    phis = np.linspace(0, 2*np.pi, n_azim, endpoint=False)
    seeds = [L * RE * (np.cos(p) * e1 + np.sin(p) * e2) for p in phis]
    B_eq = np.linalg.norm(_bfield_batch(np.array(seeds), date), axis=1)
    sa = np.sin(np.radians(pitch_angle_deg)) ** 2
    stop_b = B_eq / max(sa, 1e-6)
    kw = dict(step_min=25.0, step_max=200.0, max_steps=4000,
              max_r_km=L * RE * 3.0, surface_alt_km=loss_cone_alt_km,
              stop_b_nT=stop_b, mag_axis=axis, stop_maglat_deg=lat_max_deg)
    north = _trace_batch_rk4(seeds, date, direction=+1, **kw)
    south = _trace_batch_rk4(seeds, date, direction=-1, **kw)
    out = np.empty((n_azim, n_pts, 3))
    for i in range(n_azim):
        n_c, s_c = north[i], south[i]
        curve = np.concatenate([s_c[::-1], n_c[1:]], axis=0) if len(s_c) > 1 else n_c
        out[i] = _resample_curve(curve, n_pts)
    return out


def _igrf_belt_mesh(L_min, L_max, date, axis, base_rgb, lat_max_deg=None,
                    n_azim=48, n_pts=26, edge_dim=None, pitch_angle_deg=25.0,
                    pad_index=2.0):
    """
    Belt shell bounded by two mirror-point-limited IGRF L-shells, with vertex
    brightness set by the modelled trapped-particle density at each point
    (from B/B_eq and the first adiabatic invariant) rather than by geometry.
    """
    outer = _igrf_lshell_boundary(L_max, date, axis, n_azim, n_pts,
                                  pitch_angle_deg=pitch_angle_deg, lat_max_deg=lat_max_deg)
    inner = _igrf_lshell_boundary(L_min, date, axis, n_azim, n_pts,
                                  pitch_angle_deg=pitch_angle_deg, lat_max_deg=lat_max_deg)
    M = 2 * n_pts
    xs, ys, zs, cols = [], [], [], []
    for a in range(n_azim):
        loop = np.concatenate([outer[a], inner[a][::-1]], axis=0)
        Bmag = np.linalg.norm(_bfield_batch(loop, date), axis=1)
        # equatorial reference = weakest field on each wall of this meridian
        b_out = Bmag[:n_pts] / max(Bmag[:n_pts].min(), 1e-9)
        b_in  = Bmag[n_pts:] / max(Bmag[n_pts:].min(), 1e-9)
        w = np.concatenate([_omnidirectional_flux_ratio(b_out, n=pad_index),
                            _omnidirectional_flux_ratio(b_in,  n=pad_index)])
        for k in range(M):
            xs.append(loop[k, 0]); ys.append(loop[k, 1]); zs.append(loop[k, 2])
            cols.append(_belt_vertex_color(base_rgb, w[k]))
    ti, tj, tk = [], [], []
    for a in range(n_azim):
        a1 = (a + 1) % n_azim
        for k in range(M):
            k1 = (k + 1) % M
            v00 = a*M + k;  v01 = a*M + k1
            v10 = a1*M + k; v11 = a1*M + k1
            ti += [v00, v00]; tj += [v01, v11]; tk += [v11, v10]
    return xs, ys, zs, ti, tj, tk, cols







# ===========================================================================
# Radiation belts from NASA AE-8 / AP-8 trapped flux
# ===========================================================================
# The belts are rendered as iso-flux surfaces of the NASA AE-8 (electron) and
# AP-8 (proton) models, evaluated on field geometry traced from real IGRF.
# AE-8/AP-8 are functions of exactly two coordinates, (L, B/B0), so the model
# is tabulated once over that plane and interpolated — no shape, extent or
# radial profile is asserted here.  The inner/outer belts and the slot between
# them emerge from the model: tabulated equatorial peaks land at L = 1.7 for
# protons > 10 MeV and L = 4.4 for electrons > 1 MeV, with the slot minimum at
# L = 2.2, matching the published belt structure.

# Dipole moment calibrated so L = (M/Bmin)^(1/3) reproduces IRBEM's McIlwain L
# (validated: constant +1.43% bias against irbempy.get_Lm, removed here).


# ---------------------------------------------------------------------------
# Measured accuracy.  Each entry records WHEN it was measured, because these
# numbers move with geomagnetic conditions: the same model gives 10.4% on a
# quiet day and 17.2% on a moderately disturbed one.  Regenerate with
# validate_against_goes.py / validate_tail_and_magnetopause.py /
# validate_belts_against_goes.py and update here.
# ---------------------------------------------------------------------------
VALIDATION = {
    "geo":        ("2026-07-08", "T96 14.5 nT = 17.2%, T89 15.8 nT = 18.8% "
                                 "(Kp 1.9, Dst -30)"),
    "geo_quiet":  ("2025-07-02", "T96 10.4%, T89 9.7% (Kp 1.1)"),
    "geo_storm":  ("2025-11-12", "T96 49 nT = 40%, T89 66 nT = 53% (Kp 8.7)"),
    "sky":        ("2026-07-08", "star directions agree with astropy ITRS to "
                                 "8-24 arcsec; GMST to 0.01 s"),
    # Characterised INSIDE the magnetosphere, which is the only place these
    # models are defined.  Earlier "tail" figures quoted here were taken with
    # 0% of samples inside the modelled magnetopause — i.e. in the
    # magnetosheath, outside T96's domain — so they described the model where
    # it does not apply.  Corrected rather than reused.
    "depth":      ("2026-01-10", "T96 inside the magnetosphere: 17.2% at "
                                 "6.6 RE (GOES), 23.2% at 10.2 RE (MMS)"),
    "sheath":     ("2025-07-03", "outside the magnetopause neither model is "
                                 "defined: T96 ~97%, T89 ~35-44% (MMS, "
                                 "magnetosheath) — not a magnetotail result"),
    "boundary":   ("2026-01-10", "T96: agreement degrades 3.6x across the Shue "
                                 "surface at a 10.2 RE dayside MMS crossing; "
                                 "measured |B| 47->32 nT"),
    "belts":      ("2025-07",    "AE-8 reads ~18x above GOES >2.1 MeV "
                                 "electrons, which vary 82x over 8 days"),
}

















def _flux_isosurfaces(date, axis, grid_n=56, extent_re=7.2, levels=(0.18,),
                      pitch_angle_deg=25.0, opacity=0.30, cache=True, **kw):
    """
    Volumetric belts: AE-8/AP-8 flux sampled onto a Cartesian grid and drawn as
    iso-flux surfaces.  Each surface is a real flux level in /cm2/s, not a
    chosen shape.
    """
    if not _HAS_PPIGRF:
        # The samples are traced through the real field, so without an internal
        # field model there is nothing to trace.  Return empty and let the
        # caller fall back rather than failing deep inside the tracer.
        print("  ppigrf unavailable — cannot trace belt shells", flush=True)
        return []
    if _load_aep8_table() is None:
        # Check the flux model BEFORE tracing: the shells cost ~80 s to trace
        # and were previously computed and then discarded when the lookup
        # turned out to be unavailable.
        print("  AE-8/AP-8 unavailable — skipping belt flux", flush=True)
        return []
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # Every other optional dependency degrades with a message; this one
        # used to raise from inside the belt builder.
        print("  scipy unavailable — cannot build belt flux volume", flush=True)
        return []
    key = (date.strftime('%Y%m%d'), grid_n, extent_re, round(pitch_angle_deg, 1),
           tuple(sorted(kw.items())),
           _geo.get_external_model() is not None,
           _physics_fingerprint())
    import hashlib as _hl
    # stable digest: Python's hash() is salted per process, so a plain hash()
    # here would miss the cache on every new run
    _dig = _hl.md5(repr(key).encode()).hexdigest()[:16]
    fp_cache = _texture_cache_dir() / ("beltflux_" + _dig + ".npz")
    if cache and fp_cache.exists():
        d = np.load(fp_cache)
        P, Fp, Fe = d['P'], d['Fp'], d['Fe']
        print(f"  belts: reusing cached AE8/AP8 samples ({len(P)} points)", flush=True)
    else:
        out = _belt_flux_samples(date, axis, pitch_angle_deg=pitch_angle_deg, **kw)
        if out is None:
            return []
        P, Fp, Fe = out
        if cache:
            np.savez_compressed(fp_cache, P=P, Fp=Fp, Fe=Fe)
    RE = EARTH_RADIUS_KM
    g = np.linspace(-extent_re*RE, extent_re*RE, grid_n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    Q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    tree = cKDTree(P)
    dist, idx = tree.query(Q, k=1)
    cell = (g[1] - g[0])
    far = dist > 1.6 * cell            # outside the traced region
    traces = []
    for species, F, cmap, label in (('p', Fp, 'YlOrBr', 'AP-8 protons &gt;10 MeV'),
                                    ('e', Fe, 'Blues', 'AE-8 electrons &gt;1 MeV')):
        V = F[idx].astype(float)
        V[far] = 0.0
        if V.max() <= 0:
            continue
        vmax = float(V.max())
        # One trace per species carrying all levels (surface_count): each extra
        # Isosurface would repeat the whole coordinate grid in the saved HTML.
        traces.append(go.Isosurface(
            x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], value=V,
            isomin=levels[0]*vmax, isomax=levels[-1]*vmax,
            surface_count=len(levels),
            showscale=(species == 'e'), colorscale=cmap,
            colorbar=dict(title=dict(text='AE-8 e&#8315; mean flux<br>(cm&#8315;&#178; s&#8315;&#185;)',
                                     font=dict(color=_INK_SOFT, size=10)),
                          tickfont=dict(color=_INK_SOFT, size=9), thickness=11,
                          len=0.32, x=0.98, y=0.14, outlinewidth=0,
                          bgcolor='rgba(0,0,0,0)', exponentformat='e'),
            opacity=opacity, flatshading=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo='none', showlegend=True,
            name=(f'{label} &#183; AE-8/AP-8 climatological mean '
                  f'({levels[0]*vmax:.0e} cm&#8315;&#178;s&#8315;&#185; contour)')))
    return traces




# ===========================================================================
# Observed solar-wind drivers (OMNI)
# ===========================================================================
# The magnetopause and the external field were previously driven by separate
# hardcoded guesses — Shue at "nominal" Bz=0/Dp=2 while T89 ran at Kp=3 — so
# one figure depicted two different weather conditions.  Both are now taken
# from the OMNI hourly record for the epoch being plotted, which also fixes
# the drivers simply being wrong for the date (2025-07-02 was Bz=-1.8 nT,
# Dp=2.5 nPa, not 0/2.0).
#
# OMNI2 hourly column indices (0-based) and their fill values.








# ===========================================================================
# Solar-wind frame, magnetopause, and reference orbits
# ===========================================================================







# Reference orbits: the point of the figure for most viewers is where the
# hardware sits relative to the belts.
_REFERENCE_ORBITS = [
    (1.0 + 420.0/EARTH_RADIUS_KM, "LEO / ISS  420 km", '#7fd4ff'),
    (4.164,                       "GPS  20 200 km",    '#9ee37d'),
    (6.611,                       "GEO  35 786 km",    '#ffd24a'),
]


def _orbit_rings(date, n=180, show_labels=True):
    """Circular reference orbits in the geographic equatorial plane."""
    out = []
    t = np.linspace(0, 2*np.pi, n)
    for r_re, label, col in _REFERENCE_ORBITS:
        R = r_re * EARTH_RADIUS_KM
        out.append(go.Scatter3d(
            x=R*np.cos(t), y=R*np.sin(t), z=np.zeros_like(t), mode='lines',
            line=dict(color=col, width=1.6, dash='dot'),
            name=label, hoverinfo='skip', showlegend=True))
        if show_labels:
            out.append(go.Scatter3d(
                x=[R*np.cos(0.72)], y=[R*np.sin(0.72)], z=[0.0],
                mode='text', text=[label.split()[0]],
                textfont=dict(color=col, size=10), hoverinfo='skip',
                showlegend=False))
    return out


def _sun_marker(date, length_km, label="TO SUN \u2192"):
    u = _sun_direction_geo(date)
    return [go.Scatter3d(
        x=[0, u[0]*length_km], y=[0, u[1]*length_km], z=[0, u[2]*length_km],
        mode='lines', line=dict(color='#ffcf6b', width=4),
        name='Sunward direction', hoverinfo='skip', showlegend=True),
        go.Scatter3d(
        x=[u[0]*length_km*1.04], y=[u[1]*length_km*1.04], z=[u[2]*length_km*1.04],
        mode='text', text=[label], textfont=dict(color='#ffcf6b', size=12),
        hoverinfo='skip', showlegend=False)]





# ===========================================================================
# Presets — the common cases in one line
# ===========================================================================
# plot_magfield_3d() exposes every model control, which is a lot of knobs for
# someone who just wants a figure.  These cover the usual intents; each returns
# a dict you can pass straight in, or edit first.

PRESETS = {
    "draft":  dict(fidelity="draft", max_r_re=10.0, show_stars=False,
                   show_magnetopause=False, external_field="t89",
                   width=900, height=700),
    "poster": dict(fidelity="high", max_r_re=18.0, projection="perspective",
                   width=1500, height=1100),
    "public": dict(fidelity="high", max_r_re=18.0, show_orbits=True,
                   show_sun=True, show_magnetopause=True, split_topology=True,
                   sun_shading=True, width=1500, height=1100),
    "measure": dict(fidelity="high", projection="orthographic", show_stars=False,
                    show_atmosphere=False, max_r_re=12.0),
}


def quick_figure(preset="poster", epoch=None, save_path=None, **overrides):
    """
    One-line entry point.

    Examples
    --------
    ::

        from magfield_plot_3d import quick_figure
        quick_figure("public", save_path="magnetosphere")

    Parameters
    ----------
    preset : {'draft', 'poster', 'public', 'measure'}
        ``'draft'`` is a fast preview; ``'poster'`` is for print; ``'public'``
        adds Sun, orbits, and field-line topology; ``'measure'`` is orthographic.
    epoch : datetime or float
        Defaults to the most recent date with a complete OMNI solar-wind record.

    Anything else is passed through to plot_magfield_3d().
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    if epoch is None:
        epoch = latest_driven_epoch()
    return plot_magfield_3d(epoch=epoch, save_path=save_path, **kw)


def latest_driven_epoch(default=datetime(2025, 7, 2)):
    """
    Most recent date with real OMNI solar-wind data.

    The final hourly OMNI record lags real time by roughly three weeks, so
    "today" is not a drivable epoch.  Asking for one silently falls back to
    nominal drivers; this returns the newest date that does not.
    """
    for year in (datetime.now().year, datetime.now().year - 1):
        a = _load_omni_year(year)
        if a is None:
            continue
        real = (a[:, 3] < 999) & (a[:, 5] < 9999) & (a[:, 6] < 99)
        if not real.any():
            continue
        doy = int(a[real, 0].max())
        return datetime(year, 1, 1) + timedelta(days=doy - 1)
    return default


# ===========================================================================
# Public API
# ===========================================================================

def _model_provenance(date, external_field, kp, belt_style, sw_bz_nT, sw_dp_nPa,
                      sw_source='nominal'):
    """
    One line per model actually used, plus the measured accuracy.

    Figures that state only "IGRF" invite the reader to assume more precision
    than the models carry.  The residuals quoted here were measured against
    GOES-18 magnetometer data (validate_against_goes.py) and GOES SEISS
    particle data (validate_belts_against_goes.py).
    """
    try:
        igrf = Path(_pp.shc_fn).stem.replace("_", " ").upper()
    except Exception:
        igrf = "IGRF"
    if external_field and _HAS_GEOPACK:
        ext = ("TSYGANENKO T96 (SOLAR-WIND DRIVEN)"
               if str(external_field).lower() == "t96"
               else f"TSYGANENKO T89 (Kp={kp})")
    else:
        ext = "NO EXTERNAL FIELD"
    lines = [
        f"INTERNAL FIELD  {igrf}, DEGREE 13, EPOCH {date:%Y-%m-%d}",
        f"EXTERNAL FIELD  {ext}",
        f"MAGNETOPAUSE  SHUE ET AL. 1998  (Bz {sw_bz_nT:+.1f} nT, Dp {sw_dp_nPa:.2f} nPa)",
        f"DRIVERS  {sw_source}",
    ]
    if belt_style == "flux":
        lines.append("BELTS  NASA AE-8 / AP-8 MAX, CLIMATOLOGICAL MEAN")
    lines.append("EARTH  WGS84 ELLIPSOID  &#183;  SKY  J2000 CATALOGUE, PRECESSED TO EPOCH")
    for key in ("geo", "geo_storm", "depth", "sheath", "boundary", "sky"):
        when, what = VALIDATION[key]
        lines.append(f"MEASURED [{when}]  {what}")
    if belt_style == "flux":
        when, what = VALIDATION["belts"]
        lines.append(f"MEASURED [{when}]  {what}")
    return "<br>".join(lines)


def plot_magfield_3d(
    *,
    epoch=2025.0,
    lats=None,
    lons=None,
    title="Earth Magnetic Field Lines — IGRF",
    texture_path="auto",
    sun_shading=True,
    night_floor=0.34,
    max_line_points=900,
    elev=25.0,
    azim=-55.0,
    zoom=1.0,
    figsize=(12, 10),
    width=1400,
    height=1000,
    fidelity="high",                # 'high' | 'draft'
        color_by_field=True,
    field_colorscale=_FIELD_CMAP,
    max_r_re=10.0,
    seed_mode="lshell",
    L_values=None,
    polar_seeds=True,
    polar_lats=(74.0, 79.0, -74.0, -79.0),
    projection="perspective",
    external_field="t96",
    kp=2,
    show=False,
    save_path=None,
    save_png=False,
    dpi=150,
    show_van_allen=True,
    belt_style="flux",              # 'igrf' | 'lshell' | 'torus'
    inner_min_RE=1.0,
    inner_max_RE=2.0,
    outer_min_RE=3.0,
    outer_max_RE=6.0,
    inner_lat_max=42.0,
    outer_lat_max=34.0,
    pitch_angle_deg=25.0,
    show_mag_equator=True,
    show_pole_labels=True,
    show_stars=True,
    show_atmosphere=False,
    show_magnetopause=True,
    show_provenance=True,
    show_orbits=True,
    show_sun=True,
    orbit=None,
    r=None,
    v=None,
    t=None,
    r_units="auto",
    v_units="auto",
    orbit_name="SSAPy orbit",
    orbit_color="#ff4d4d",
    orbit_width=5,
    split_topology=True,
    sw_bz_nT=0.0,
    sw_dp_nPa=2.0,
    use_omni=True,
    belt_residence=None,
    show_belt_residence=True,
    **kwargs,
):
    """
    Render IGRF field lines, Van Allen belts and a WGS84 Earth in an
    interactive Plotly 3D scene.

    Parameters
    ----------
    fidelity : {'high', 'draft'}
        ``'high'`` uses fine integration steps, a dense Earth mesh, and
        IGRF-traced belts; ``'draft'`` is the fast preview path.
    belt_style : {'igrf', 'lshell', 'torus'}
        ``'igrf'`` uses real IGRF-traced L-shells; ``'lshell'`` uses an analytic
        dipole crescent; ``'torus'`` uses an analytic circular torus.
    zoom : float
        Values greater than one tighten framing; values below one pull back.
    save_path : path-like
        Writes an interactive HTML file and a PNG when ``save_png`` is true.
    """
    if not _HAS_PPIGRF:
        raise ImportError("ppigrf is required: pip install ppigrf")
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    hi = (str(fidelity).lower() != "draft")

    if isinstance(epoch, (int, float)):
        year = int(epoch); frac = epoch - year
        days = 366 if calendar.isleap(year) else 365
        date = datetime(year, 1, 1) + timedelta(days=int(frac * days))
    else:
        date = epoch.replace(tzinfo=None) if hasattr(epoch, 'tzinfo') else epoch

    print(f"IGRF epoch: {date.strftime('%Y-%m-%d')}  (fidelity={'high' if hi else 'draft'})",
          flush=True)

    if lons is None:
        lons = list(range(0, 360, 30)) if hi else list(range(0, 360, 60))

    kp, sw_bz_nT, sw_dp_nPa, _sw_src = _apply_solar_wind(
        date, kp, sw_bz_nT, sw_dp_nPa, use_omni=use_omni)

    # Set the external field through geomagnetics.set_external_model rather
    # than a `global _EXTERNAL_MODEL` here. The state lives in geomagnetics
    # now, and a global assignment in this module would create a *separate*
    # binding that the physics never reads -- so every plot would silently
    # render with the internal IGRF field only, while still announcing T96.
    # (Note the module __setattr__ guard below does not catch this: `global`
    # writes go straight to __dict__ and never reach __setattr__.)
    _geo.set_external_model(None)
    if external_field and str(external_field).lower() in ("t89", "t96"):
        if _HAS_GEOPACK:
            _geo.set_external_model(
                _get_external(date, kp=kp, model=str(external_field).lower()))
        else:
            print("  geopack not installed — internal (IGRF) field only; "
                  "outer field lines will not show tail stretching", flush=True)

    if str(seed_mode).lower() == "lshell" and lats is None:
        if L_values is None:
            L_values = ([1.5, 2.0, 3.0, 4.0, 5.0, 6.6, 8.0] if hi
                        else [1.5, 3.0, 5.0, 8.0])
        print(f"  seeding on the true magnetic equator at L = {L_values}", flush=True)
        seeds, _seedL = _make_seeds_lshell(L_values, date, n_lons=len(lons))
        seeds = list(seeds)
        if polar_seeds:
            # Equatorial L-shell seeds are closed by construction, so on their
            # own they can never show the magnetotail.  Polar-cap footpoints
            # are what map to the open lobes.
            pc = _make_seeds_magnetic(list(polar_lats), n_lons=max(6, len(lons)//2))
            print(f"  + {len(pc)} polar-cap seeds at {list(polar_lats)} deg magnetic "
                  f"(these map to the open tail lobes)", flush=True)
            seeds += list(pc)
    else:
        if lats is None:
            lats = [15, 25, 35, 45, 55, 63, -15, -25, -35, -45, -55, -63] if hi \
                   else [20, 35, 50, 63, -20, -35, -50, -63]
        seeds = _make_seeds_magnetic(lats, n_lons=len(lons))
    max_r_km = max_r_re * EARTH_RADIUS_KM

    step_min, step_max = (8.0, 220.0) if hi else (25.0, 400.0)
    print(f"Tracing {len(seeds)} field lines (adaptive RK4, "
          f"{step_min:.0f}-{step_max:.0f} km)...", flush=True)
    raw_lines = _trace_all_closed(seeds, date, max_r_km=max_r_km,
                                  step_min=step_min, step_max=step_max,
                                  max_steps=9000 if hi else 4000)
    field_lines = [(i, pts) for i, pts in enumerate(raw_lines) if len(pts) > 2]
    print(f"Done — {len(field_lines)} lines. Building figure...", flush=True)

    RE = EARTH_RADIUS_KM
    axis = _dipole_axis()

    trajectory_r_km = None
    if orbit is not None or r is not None:
        trajectory_r_km, _, _ = normalize_orbit_trajectory(
            orbit=orbit,
            r=r,
            v=v,
            t=t,
            require_velocity=False,
            r_units=r_units,
            v_units=v_units,
        )

    # ------------------------------------------------------------------
    # Scene framing.  The box is sized to the STAR sphere so the stars are
    # inside the axis range (they would otherwise be clipped), and the camera
    # eye distance is scaled down by content/sky so the subject still fills
    # the frame.  This keeps the camera inside the star sphere.
    # ------------------------------------------------------------------
    if field_lines:
        content_r = max(float(np.max(np.linalg.norm(p, axis=1))) for _, p in field_lines)
    else:
        content_r = 2.0 * RE
    if show_van_allen:
        content_r = max(content_r, outer_max_RE * RE * 1.06)
    if trajectory_r_km is not None:
        content_r = max(content_r, float(np.nanmax(np.linalg.norm(trajectory_r_km, axis=1))) * 1.08)
    content_r = max(content_r, 1.5 * RE) * 1.02 / max(float(zoom), 1e-3)

    ortho = str(projection).lower().startswith("ortho")
    if ortho and show_stars:
        # A correct sky needs the camera inside the star sphere so near-side
        # stars fall behind the near clip plane.  Orthographic has no such
        # plane, and Plotly clips to the axis box, so the sphere would have to
        # sit at the same radius as the field lines — tangled in the scene
        # rather than behind it.  Drop the stars instead of faking them.
        print("  orthographic projection: starfield disabled (a finite-radius "
              "sky sphere would sit inside the scene); use projection='perspective' "
              "for stars", flush=True)
        show_stars = False
    if ortho:
        # Orthographic removes the foreshortening of a perspective camera: at
        # the framing used here the camera sits ~1.35 subject radii away, so a
        # feature on the near side renders up to ~6.7x larger than the same
        # feature on the far side.  That distortion makes symmetry impossible
        # to judge by eye, so it is the default for a measurement figure.
        # The cost: Plotly cannot clip the near half of the star sphere in this
        # projection, so stars are culled to the hemisphere away from the
        # initial camera (correct as rendered; rotating far from the initial
        # view will thin the sky).
        box = content_r * 1.02
        sky_radius = box * 0.98
        eye_dist = 1.35
    else:
        sky_radius = STAR_SPHERE_FACTOR * content_r
        box = sky_radius
        eye_dist = _CAM_FILL * content_r / box

    traces = []

    # 1. Starfield — far outside the subject
    if show_stars:
        traces.extend(_build_starfield_trace(sky_radius=sky_radius, date=date))

    # 2. Van Allen belts
    if show_van_allen:
        if belt_style == "flux":
            fx = _flux_isosurfaces(date, axis, pitch_angle_deg=pitch_angle_deg,
                                   n_L=22 if hi else 16, n_azim=24 if hi else 18,
                                   n_pts=80 if hi else 60,
                                   grid_n=48 if hi else 36, eq_iters=2 if hi else 1)
            if fx:
                traces.extend(fx)
            else:
                print("  AE8/AP8 unavailable — falling back to IGRF L-shell belts", flush=True)
                belt_style = "igrf"
        if belt_style == "igrf":
            print("  belts: tracing IGRF L-shells...", flush=True)
            n_az = 48 if hi else 24
            ox, oy, oz, oi, oj, ok, ocol = _igrf_belt_mesh(
                outer_min_RE, outer_max_RE, date, axis, base_rgb=(60, 150, 255),
                lat_max_deg=None, n_azim=n_az, pitch_angle_deg=pitch_angle_deg)
            traces.append(go.Mesh3d(x=ox, y=oy, z=oz, i=oi, j=oj, k=ok,
                vertexcolor=ocol, opacity=0.20, flatshading=False, hoverinfo='none',
                name=f'Outer belt ({outer_min_RE}-{outer_max_RE} RE, IGRF)', showlegend=True))
            ix, iy, iz, ii, ij, ik_, icol = _igrf_belt_mesh(
                inner_min_RE, inner_max_RE, date, axis, base_rgb=(255, 205, 60),
                lat_max_deg=None, n_azim=n_az, pitch_angle_deg=pitch_angle_deg)
            traces.append(go.Mesh3d(x=ix, y=iy, z=iz, i=ii, j=ij, k=ik_,
                vertexcolor=icol, opacity=0.34, flatshading=False, hoverinfo='none',
                name=f'Inner belt ({inner_min_RE}-{inner_max_RE} RE, IGRF)', showlegend=True))
        elif belt_style == "flux":
            pass                        # already added as iso-flux surfaces
        elif belt_style == "lshell":
            ox, oy, oz, oi, oj, ok, ocol = _dipole_belt_mesh(
                outer_min_RE, outer_max_RE, axis, base_rgb=(60, 150, 255),
                lat_max_deg=outer_lat_max)
            traces.append(go.Mesh3d(x=ox, y=oy, z=oz, i=oi, j=oj, k=ok,
                vertexcolor=ocol, opacity=0.20, flatshading=False, hoverinfo='none',
                name=f'Outer belt ({outer_min_RE}-{outer_max_RE} RE)', showlegend=True))
            ix, iy, iz, ii, ij, ik_, icol = _dipole_belt_mesh(
                inner_min_RE, inner_max_RE, axis, base_rgb=(255, 205, 60),
                lat_max_deg=inner_lat_max)
            traces.append(go.Mesh3d(x=ix, y=iy, z=iz, i=ii, j=ij, k=ik_,
                vertexcolor=icol, opacity=0.34, flatshading=False, hoverinfo='none',
                name=f'Inner belt ({inner_min_RE}-{inner_max_RE} RE)', showlegend=True))
        else:
            R_o=(outer_min_RE+outer_max_RE)/2*RE; r_o=(outer_max_RE-outer_min_RE)/2*RE
            ox,oy,oz,oi,oj,ok=_torus_mesh3d(R_o,r_o,axis)
            traces.append(go.Mesh3d(x=ox,y=oy,z=oz,i=oi,j=oj,k=ok,color='rgb(30,140,255)',
                opacity=0.15,flatshading=False,hoverinfo='none',
                name=f'Outer belt ({outer_min_RE}-{outer_max_RE} RE)',showlegend=True))
            R_i=(inner_min_RE+inner_max_RE)/2*RE; r_i=(inner_max_RE-inner_min_RE)/2*RE
            ix,iy,iz,ii,ij,ik_=_torus_mesh3d(R_i,r_i,axis)
            traces.append(go.Mesh3d(x=ix,y=iy,z=iz,i=ii,j=ij,k=ik_,color='rgb(255,200,30)',
                opacity=0.25,flatshading=False,hoverinfo='none',
                name=f'Inner belt ({inner_min_RE}-{inner_max_RE} RE)',showlegend=True))

    # 3. Field lines coloured by |B| — merged into a SINGLE trace with NaN
    #    separators.  One draw call instead of one per line keeps the scene
    #    responsive when rotating/zooming in the browser.
    cap = int(max_line_points if hi else max(200, max_line_points // 2))
    draw = []
    for _, pts in field_lines:
        draw.append(_resample_curve(pts, cap) if len(pts) > cap else pts)

    cmin = cmax = None
    if color_by_field and draw:
        vals = [np.log10(np.clip(_field_magnitude_along(p, date), 1e-3, None)) for p in draw]
        allv = np.concatenate(vals)
        cmin, cmax = float(np.percentile(allv, 2)), float(np.percentile(allv, 98))
        gap = np.array([[np.nan, np.nan, np.nan]])
        groups = {'closed': ([], []), 'open': ([], [])}
        for p, v in zip(draw, vals):
            kind = _classify_line(p) if split_topology else 'closed'
            kind = 'open' if kind != 'closed' else 'closed'
            groups[kind][0].extend([p, gap])
            groups[kind][1].extend([v, np.array([np.nan])])
        style = {'closed': dict(dash='solid',
                                nm='Closed field lines (both ends on Earth)'),
                 'open':   dict(dash='longdash',
                                nm='Open field lines (connected to the solar wind)')}
        first = True
        for kind in ('closed', 'open'):
            segs, cols = groups[kind]
            if not segs:
                continue
            P = np.concatenate(segs, axis=0)
            C = np.concatenate(cols)
            alt = np.linalg.norm(P, axis=1) - EARTH_RADIUS_KM
            cd = np.stack([alt / EARTH_RADIUS_KM, 10.0**C], axis=1)
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                line=dict(color=C, colorscale=field_colorscale, cmin=cmin, cmax=cmax,
                          width=3.5, dash=style[kind]['dash'], showscale=first,
                          colorbar=dict(title=dict(text='|B|  (log&#8321;&#8320; nT)',
                                                   font=dict(color=_INK_SOFT, size=11)),
                                        tickfont=dict(color=_INK_SOFT, size=10),
                                        thickness=12, len=0.34, x=0.98, y=0.62,
                                        outlinewidth=0, bgcolor='rgba(0,0,0,0)')),
                customdata=cd,
                hovertemplate='altitude %{customdata[0]:.2f} R<sub>E</sub><br>'
                              '|B| %{customdata[1]:,.0f} nT<extra></extra>',
                name=style[kind]['nm'], showlegend=True))
            first = False
    else:
        n_lats = len(lats) if lats else 1
        gap = np.array([[np.nan, np.nan, np.nan]])
        by_color = {}
        for (seed_idx, _), p in zip(field_lines, draw):
            t = 0.5 if not lats else float(np.clip((abs(lats[seed_idx % n_lats]) - 20) / 43.0, 0, 1))
            by_color.setdefault(f'rgb({int(t*255)},{int((1-t)*255)},255)', []).append(p)
        for colr, group in by_color.items():
            segs = []
            for p in group:
                segs.append(p); segs.append(gap)
            P = np.concatenate(segs, axis=0)
            traces.append(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                line=dict(color=colr, width=3), hoverinfo='none', showlegend=False))

    # 4. Earth (after belts/lines so it occludes them)
    traces.append(_build_earth_mesh(texture_path,
                                    n_lon=480 if hi else 240,
                                    n_lat=240 if hi else 120,
                                    sun_shading=sun_shading, date=date,
                                    night_floor=night_floor))
    if show_atmosphere:
        traces.extend(_atmosphere_traces())

    if trajectory_r_km is not None:
        traces.append(plotly_orbit_trace(
            trajectory_r_km,
            name=orbit_name,
            color=orbit_color,
            width=orbit_width,
            go_module=go,
        ))

    if show_magnetopause:
        gpts, mi, mj, mk, r0, alpha = _shue_magnetopause(date, bz_nT=sw_bz_nT,
                                                         dp_nPa=sw_dp_nPa)
        traces.append(go.Mesh3d(
            x=gpts[:, 0], y=gpts[:, 1], z=gpts[:, 2], i=mi, j=mj, k=mk,
            color='rgb(120,170,255)', opacity=0.06, flatshading=False,
            hoverinfo='skip',
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
            name=f'Magnetopause (Shue 1998, {r0:.1f} R<sub>E</sub> nose)',
            showlegend=True))

    if show_orbits:
        traces.extend(_orbit_rings(date))

    if show_sun:
        traces.extend(_sun_marker(date, content_r * 0.92))

    # 5. Dipole axis, magnetic equator, dip poles
    ax_len = content_r * 0.95
    traces.append(go.Scatter3d(
        x=[-axis[0]*ax_len, axis[0]*ax_len],
        y=[-axis[1]*ax_len, axis[1]*ax_len],
        z=[-axis[2]*ax_len, axis[2]*ax_len],
        mode='lines', line=dict(color='#66e0ff', width=2, dash='dash'),
        name='Magnetic dipole axis', showlegend=show_van_allen))

    if show_mag_equator:
        ex, ey, ez = _mag_equator_ring(axis, outer_max_RE * RE * 1.04)
        traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='#3a6a8a', width=1.5), name='Magnetic equator',
            hoverinfo='none', showlegend=show_van_allen))

    if show_pole_labels:
        for (lon_d, lat_d), label in ((_MAG_NORTH, 'N'), (_MAG_SOUTH, 'S')):
            px, py, pz = _geo_to_xyz(lon_d, lat_d, r=RE*1.02)
            traces.append(go.Scatter3d(x=[px], y=[py], z=[pz], mode='markers+text',
                marker=dict(size=3, color='#66e0ff'), text=[f'  {label}'],
                textposition='top center', textfont=dict(color=_INK_SOFT, size=11),
                hoverinfo='none', showlegend=False))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    belt_note = {'igrf': f'BELTS: IGRF L-SHELLS BOUNDED BY MIRROR POINTS (PITCH {pitch_angle_deg:.0f}\u00b0)',
                 'lshell': 'BELTS FROM DIPOLE L-SHELL MODEL',
                 'torus': 'BELTS FROM TORUS APPROXIMATION'}.get(belt_style, '')
    _annotations = [
        dict(x=0.01, y=0.045, xref='paper', yref='paper', xanchor='left',
             text=f'IGRF EPOCH&#160;&#160;{date.strftime("%Y-%m-%d")}',
             font=dict(color=_INK_SOFT, size=10, family='Helvetica, Arial'),
             showarrow=False),
        dict(x=0.01, y=0.015, xref='paper', yref='paper', xanchor='left',
             text=f'IGRF DEGREE 13, GEOCENTRIC SYNTHESIS &#183; ADAPTIVE RK4 &#183; WGS84 EARTH &#183; {belt_note}',
             font=dict(color=_INK_FAINT, size=9, family='Helvetica, Arial'),
             showarrow=False),
    ]
    if show_provenance:
        _annotations.append(dict(
            x=0.01, y=0.30, xref='paper', yref='paper',
            xanchor='left', yanchor='top', align='left',
            text=_model_provenance(date, external_field, kp, belt_style,
                                   sw_bz_nT, sw_dp_nPa, _sw_src),
            font=dict(color=_INK_FAINT, size=8.5, family='Helvetica, Arial'),
            showarrow=False))

    if show_belt_residence and belt_residence:
        try:
            t_in  = belt_residence.get("t_inner_s", 0.0)/60.0
            t_out = belt_residence.get("t_outer_s", 0.0)/60.0
            per   = belt_residence.get("period_s", 0.0)/60.0
            _annotations.append(dict(x=0.01, y=0.90, xref='paper', yref='paper',
                xanchor='left',
                text=(f'Time in field per {per:.1f}-min orbit &#183; '
                      f'inner {t_in:.2f} min &#183; outer {t_out:.2f} min'),
                font=dict(color=_ACCENT_WARM, size=10, family='Helvetica, Arial'),
                showarrow=False))
        except Exception:
            pass

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=(f'{title}<br><span style="font-size:12px;color:{_INK_SOFT};">'
                             f'IGRF {date.strftime("%Y")} &#183; field strength colour-mapped '
                             f'&#183; Van Allen belts</span>'),
                       font=dict(color='#f2f2f5', size=20, family='Helvetica, Arial'),
                       x=0.5, y=0.96, xanchor='center'),
            paper_bgcolor=_BG,
            scene=dict(
                bgcolor=_BG,
                xaxis=dict(visible=False, range=[-box, box]),
                yaxis=dict(visible=False, range=[-box, box]),
                zaxis=dict(visible=False, range=[-box, box]),
                camera=dict(eye=_camera_eye(elev, azim, eye_dist),
                            up=dict(x=0, y=0, z=1),
                            projection=dict(type='orthographic' if ortho else 'perspective')),
                aspectmode='cube',
                dragmode='orbit'),
            width=width, height=height,
            margin=dict(l=0, r=0, t=64, b=0),
            showlegend=show_van_allen,
            legend=dict(font=dict(color='#d8d8de', size=11),
                        bgcolor='rgba(10,10,18,0.6)', bordercolor='#2a2a38',
                        borderwidth=1, x=0.01, y=0.82) if show_van_allen else dict(),
            annotations=_annotations,
            uirevision='magfield_camera'),
    )

    if save_path is not None:
        out = Path(str(save_path)).with_suffix('')
        out.parent.mkdir(parents=True, exist_ok=True)
        html_out = out.with_suffix('.html')
        fig.write_html(str(html_out), include_plotlyjs=True,
                       full_html=True, config=PLOTLY_CONFIG)
        print(f"  HTML (interactive): {html_out}", flush=True)
        if save_png:
            try:
                png_out = out.with_suffix('.png')
                fig.write_image(str(png_out), width=width, height=height, scale=2)
                print(f"  PNG: {png_out}", flush=True)
            except Exception as e:
                print(f"  PNG skipped: {e}", flush=True)

    if show:
        fig.show(config=PLOTLY_CONFIG)

    return fig, None


# -- Entry point -------------------------------------------------------------
if __name__ == "__main__":
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(output_root() / "figures")))
    cfg = {}
    gui_cfg_path = os.environ.get("GUI_CONFIG", "")
    if gui_cfg_path and Path(gui_cfg_path).exists():
        try:
            _ns: dict = {}
            exec(compile(Path(gui_cfg_path).read_text(), gui_cfg_path, "exec"), {}, _ns)
            cfg = _ns
            print(f"[magfield_plot_3d] Loaded GUI_CONFIG from {gui_cfg_path}")
        except Exception as e:
            print(f"[magfield_plot_3d] Warning: could not parse GUI_CONFIG ({e}); using defaults.")

    output_dir.mkdir(parents=True, exist_ok=True)

    epoch = cfg.get("epoch")
    if isinstance(epoch, str):
        try:
            from datetime import datetime as _dt
            _d = _dt.strptime(epoch[:19], "%Y-%m-%d %H:%M:%S")
            epoch = _d.year + (_d.timetuple().tm_yday - 1) / 365.25
        except Exception:
            epoch = 2025.0
    elif not isinstance(epoch, (int, float)):
        epoch = 2025.0

    # Any key in the GUI config that names a real parameter is forwarded.
    # Previously only ten keys were read, so most of the model controls
    # (external_field, kp, projection, magnetopause, solar-wind drivers,
    # pitch angle, figure size ...) could not be reached from the GUI at all.
    import inspect as _inspect
    _params = _inspect.signature(plot_magfield_3d).parameters
    kwargs = {k: v for k, v in cfg.items()
              if k in _params and not k.startswith("_") and k != "epoch"}
    kwargs.setdefault("save_path", str(output_dir / "magfield_plot_3d"))
    kwargs.setdefault("show", False)
    unknown = sorted(k for k in cfg
                     if k not in _params and k != "epoch"
                     and not k.startswith("_") and not callable(cfg[k]))
    if unknown:
        print(f"[magfield_plot_3d] ignoring unknown config keys: {unknown}")

    fig, _ = plot_magfield_3d(epoch=epoch, **kwargs)
    print(f"[magfield_plot_3d] Saved -> {output_dir / 'magfield_plot_3d.html'}")

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _check_environment():
    """Report what's present and what's missing, and return an exit code.

    This exists because the failure modes of this module are mostly silent
    degradations rather than crashes -- a missing star catalogue gives you a
    synthetic starfield, no spacepy gives you no belt surfaces, no geopack
    gives you the internal field only. Each is reasonable on its own, but the
    figure doesn't say which happened. --check answers "why does my plot look
    different from the one in the docs" without reading any source.
    """
    lines = []

    lines.append("data files:")
    try:
        from .starfield import find_data_file, ssapy_data_dirs
    except Exception:
        find_data_file = None
        ssapy_data_dirs = None
    if find_data_file is None:
        lines.append("  (starfield.find_data_file unavailable -- cannot search)")
    else:
        for asset, consequence in (
            ("bright_stars.csv", "synthetic starfield instead of the real sky"),
            ("aep8_table.npz", "no radiation-belt flux surfaces"),
        ):
            found = find_data_file(asset)
            if found is None:
                lines.append(f"  MISSING  {asset}  -> {consequence}")
            else:
                lines.append(f"  ok       {asset}  {found}")
        if ssapy_data_dirs is not None:
            lines.append("  searched: " + ", ".join(str(d) for d in ssapy_data_dirs()))

    lines.append("optional packages:")
    for mod, consequence in (
        ("ppigrf", "no IGRF internal field"),
        ("geopack", "no T89/T96 external field, no GEO->GSM conversion"),
        ("spacepy", "cannot build the AE-8/AP-8 table (a cached one still works)"),
    ):
        try:
            __import__(mod)
            lines.append(f"  ok       {mod}")
        except Exception:
            lines.append(f"  MISSING  {mod}  -> {consequence}")

    lines.append("solar-wind drivers:")
    try:
        when = latest_driven_epoch()
        sw = get_solar_wind(when)
        if sw and sw.get("dp_nPa") is not None:
            lines.append(f"  ok       OMNI drivers available to {when:%Y-%m-%d}")
        else:
            lines.append("  MISSING  no OMNI drivers -> nominal solar wind assumed")
    except Exception as exc:
        lines.append(f"  MISSING  OMNI lookup failed ({type(exc).__name__}) "
                     f"-> nominal solar wind assumed")

    print("\n".join(lines))
    return 0


def main(argv=None):
    """CLI entry point: `ssatk-magnetosphere`.

    Returns an exit code rather than calling sys.exit() directly so it stays
    testable; argparse still raises SystemExit on a bad argument, which is the
    documented behaviour callers rely on.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="ssatk-magnetosphere",
        description="Render the 3D geomagnetic field / radiation belts.",
    )
    p.add_argument("--check", action="store_true",
                   help="report data files, optional packages and solar-wind "
                        "drivers, then exit")
    p.add_argument("--preset", choices=sorted(PRESETS), default="poster",
                   help="named parameter set (default: poster)")
    p.add_argument("--out", default=None,
                   help="write the figure here instead of showing it")
    args = p.parse_args(argv)

    if args.check:
        return _check_environment()

    quick_figure(args.preset, save_path=args.out)
    return 0


# NOTE: no `if __name__ == "__main__"` here for main(). This module already has
# one further up that the GUI relies on -- it reads $GUI_CONFIG and renders --
# and a second block would run main() straight after it on every script
# invocation. The CLI is reached through the console-script entry point
# declared in pyproject.toml (ssatk-magnetosphere = ...:main), which calls
# main() directly and needs no __main__ block.
