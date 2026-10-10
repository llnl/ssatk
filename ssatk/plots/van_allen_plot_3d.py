"""
van_allen_plot_3d.py — Van Allen radiation belt plot for SSATK
=======================================================================
Interactive Plotly 3D scene of the inner and outer radiation belts.

Belt geometry (belt_style)
--------------------------
* ``'igrf'``: belt shells bounded by real IGRF-traced L-shells. Because the
  geomagnetic field is not axisymmetric, the inner belt dips toward the South
  Atlantic Anomaly. It falls back automatically if ppigrf is unavailable.
* ``'lshell'``: analytic dipole L-shell crescent (no ppigrf needed).
* ``'torus'``: analytic circular torus.

Belt dimensions (approximate, L-shell based):

* Inner belt: 1.0 - 2.0 RE (proton-dominated, warm).
* Outer belt: 3.0 - 6.0 RE (electron-dominated, cool).


Earth is the WGS84 oblate ellipsoid; stars sit on a far sphere so the camera
orbits inside it and near-side stars never render in front of the planet.

Dependencies
------------
    pip install 'ssatk[static]'    (ppigrf optional, for belt_style='igrf')
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

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
    EARTH_RADIUS_KM, PLOTLY_CONFIG, STAR_SPHERE_FACTOR, _ACCENT_WARM, _BG,
    _CAM_FILL, _HAS_PLOTLY, _INK_FAINT, _INK_SOFT, _MAG_NORTH,
    _MAG_SOUTH, _atmosphere_traces, _build_earth_mesh,
    _build_starfield_trace, _camera_eye, _dipole_axis, _dipole_belt_mesh,
    _geo_to_xyz, _mag_equator_ring, _torus_mesh3d, go)
except ImportError:
    from magnetosphere_core import (
    EARTH_RADIUS_KM, PLOTLY_CONFIG, STAR_SPHERE_FACTOR, _ACCENT_WARM, _BG,
    _CAM_FILL, _HAS_PLOTLY, _INK_FAINT, _INK_SOFT, _MAG_NORTH,
    _MAG_SOUTH, _atmosphere_traces, _build_earth_mesh,
    _build_starfield_trace, _camera_eye, _dipole_axis, _dipole_belt_mesh,
    _geo_to_xyz, _mag_equator_ring, _torus_mesh3d, go)


# ---------------------------------------------------------------------------
# Earth texture.  texture_path="auto" (the default) searches these locations,
# then falls back to downloading NASA Blue Marble into a local cache once.
# Drop your own equirectangular image at any of these paths to override.
# ---------------------------------------------------------------------------

# IGRF 2025 magnetic dip pole positions (lon_deg, lat_deg)

# Editorial palette




# Interactive modebar / zoom behaviour for saved HTML

# ===========================================================================
# Geometry
# ===========================================================================

def _revolve_faces(n_azim, M):
    ti, tj, tk = [], [], []
    for a in range(n_azim):
        a1 = (a + 1) % n_azim
        for k in range(M):
            k1 = (k + 1) % M
            v00 = a*M + k;  v01 = a*M + k1
            v10 = a1*M + k; v11 = a1*M + k1
            ti += [v00, v00]; tj += [v01, v11]; tk += [v11, v10]
    return ti, tj, tk


def _igrf_belt_backend():
    """
    Lazily fetch the IGRF belt builder from magfield_plot_3d.  Imported inside
    the call (not at module import) so the plots package can auto-import both
    modules without a circular-import problem, and so this module still works
    with no ppigrf installed.
    """
    try:
        from .magfield_plot_3d import _igrf_belt_mesh, _HAS_PPIGRF  # type: ignore
    except Exception:
        try:
            from magfield_plot_3d import _igrf_belt_mesh, _HAS_PPIGRF  # type: ignore
        except Exception:
            return None
    return _igrf_belt_mesh if _HAS_PPIGRF else None



# ===========================================================================
# Astrometry — catalogue (J2000) to Earth-fixed, and physical star colour
# ===========================================================================
# The scene is Earth-fixed (Greenwich on +X, matching IGRF and the texture),
# but star catalogues are J2000 equatorial.  Plotting RA/Dec directly into
# this frame — as the previous version did — leaves the sky misaligned by the
# Greenwich Mean Sidereal Time: measured median 68.6 deg, max 80.3 deg.
# The chain below applies proper motion, precession and Earth rotation.


     # 240 s per degree

# ---------------------------------------------------------------- colour
# ===========================================================================
# Earth — WGS84 ellipsoid, high resolution, geodetic texture mapping
# ===========================================================================


# ===========================================================================
# Stars
# ===========================================================================

# ---------------------------------------------------------------------------
# Astrometry — J2000 catalogue to Earth-fixed, and physical star colour.
# The scene is Earth-fixed (Greenwich on +X, matching IGRF and the texture),
# but catalogues are J2000 equatorial.  Plotting RA/Dec straight into this
# frame leaves the sky misaligned by GMST: measured median 68.6 deg, max 80.3.
# ---------------------------------------------------------------------------
# ===========================================================================
# Public API
# ===========================================================================

def plot_van_allen_3d(
    *,
    title="Van Allen Radiation Belts",
    texture_path="auto",
    sun_shading=True,
    night_floor=0.34,
    epoch=2025.0,
    elev=22.0,
    azim=-55.0,
    zoom=1.0,
    width=1400,
    height=1000,
    show=False,
    save_path=None,
    save_png=False,
    fidelity="high",
    belt_style="flux",         # 'igrf' | 'lshell' | 'torus'
    inner_min_RE=1.0,
    inner_max_RE=2.0,
    outer_min_RE=3.0,
    outer_max_RE=6.0,
    inner_lat_max=42.0,
    outer_lat_max=34.0,
    show_mag_equator=True,
    show_pole_labels=True,
    show_stars=True,
    show_atmosphere=False,
    orbit=None,
    r=None,
    v=None,
    t=None,
    r_units="auto",
    v_units="auto",
    orbit_name="SSAPy orbit",
    orbit_color="#ff4d4d",
    orbit_width=5,
    pitch_angle_deg=25.0,
    external_field="t96",
    kp=2,
    belt_residence=None,
    show_belt_residence=True,
    **kwargs,
):
    """
    Render the Van Allen belts in an interactive Plotly 3D scene.

    Parameters
    ----------
    belt_style : {'igrf', 'lshell', 'torus'}
        ``'igrf'`` traces real IGRF L-shells; it falls back to ``'lshell'`` if
        ppigrf is unavailable.
    zoom : float
        Values greater than one tighten framing; values below one pull back.
    save_path : path-like
        Writes an interactive HTML file and a PNG when ``save_png`` is true.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    hi   = (str(fidelity).lower() != "draft")
    RE   = EARTH_RADIUS_KM
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

    # Framing: box sized to the star sphere (so stars aren't clipped), camera
    # eye scaled down by content/sky so the belts still fill the frame.
    content_r  = outer_max_RE * RE * 1.08 / max(float(zoom), 1e-3)
    if trajectory_r_km is not None:
        content_r = max(content_r, float(np.nanmax(np.linalg.norm(trajectory_r_km, axis=1))) * 1.08)
    sky_radius = STAR_SPHERE_FACTOR * content_r
    box        = sky_radius
    eye_dist   = _CAM_FILL * content_r / box

    traces = []

    # Epoch for sky orientation (stars are placed in the Earth-fixed frame)
    import calendar as _cl
    if isinstance(epoch, (int, float)):
        _yr = int(epoch); _fr = epoch - _yr
        _sky_date = datetime(_yr, 1, 1) + timedelta(days=int(_fr * (366 if _cl.isleap(_yr) else 365)))
    else:
        _sky_date = epoch.replace(tzinfo=None) if hasattr(epoch, 'tzinfo') else epoch

    # 1. Stars
    if show_stars:
        traces.extend(_build_starfield_trace(sky_radius, date=_sky_date))

    # 2. Belts (outer first so the inner belt reads in front)
    style = belt_style
    if style == "flux":
        # AE-8 / AP-8 trapped flux iso-surfaces, evaluated on real IGRF(+T89)
        # field geometry.  Belt extent and the slot between the belts come from
        # the model, not from asserted radii.
        try:
            import magfield_plot_3d as _mfx
        except Exception:
            try:
                from . import magfield_plot_3d as _mfx
            except Exception:
                _mfx = None
        _fx = []
        if _mfx is not None and not getattr(_mfx, "_HAS_PPIGRF", False):
            print("  ppigrf unavailable — AE-8/AP-8 belts need it to trace shells",
                  flush=True)
            _mfx = None
        if _mfx is not None:
            if external_field and str(external_field).lower() == "t89" \
                    and getattr(_mfx, "_HAS_GEOPACK", False):
                # set/get_external_model() rather than attribute assignment: the
                # state lives in ssatk.geomagnetics, and assigning it on a
                # module that merely re-exports the physics is a silent no-op.
                _mfx.set_external_model(_mfx._get_external(_sky_date, kp=kp, model="t89"))
            _fx = _mfx._flux_isosurfaces(_sky_date, axis,
                                         pitch_angle_deg=pitch_angle_deg,
                                         n_L=22 if hi else 16, n_azim=24 if hi else 18,
                                         n_pts=80 if hi else 60,
                                         grid_n=48 if hi else 36, eq_iters=2 if hi else 1)
            _mfx.set_external_model(None)
        if _fx:
            traces.extend(_fx)
            style = "_done"
        else:
            print("  AE8/AP8 unavailable — falling back to dipole L-shell belts", flush=True)
            style = "lshell"
    igrf_mesh = _igrf_belt_backend() if style == "igrf" else None
    if style == "igrf" and igrf_mesh is None:
        print("  ppigrf unavailable — falling back to dipole L-shell belts", flush=True)
        style = "lshell"

    if style == "_done":
        pass
    elif style == "igrf":
        import calendar as _cal
        if isinstance(epoch, (int, float)):
            yr = int(epoch); fr = epoch - yr
            days = 366 if _cal.isleap(yr) else 365
            date = datetime(yr, 1, 1) + timedelta(days=int(fr*days))
        else:
            date = epoch.replace(tzinfo=None) if hasattr(epoch, 'tzinfo') else epoch
        n_az = 48 if hi else 24
        # Belts reach L=6, where T89 external currents are ~36% of the internal
        # field on the nightside, so trace them in the total field.
        try:
            import magfield_plot_3d as _mf
        except Exception:
            try:
                from . import magfield_plot_3d as _mf
            except Exception:
                _mf = None
        if _mf is not None and external_field and str(external_field).lower() == "t89" \
                and getattr(_mf, "_HAS_GEOPACK", False):
            _mf.set_external_model(_mf._get_external(date, kp=kp, model="t89"))
        print("  belts: tracing IGRF L-shells...", flush=True)
        ox, oy, oz, oi, oj, ok, ocol = igrf_mesh(
            outer_min_RE, outer_max_RE, date, axis, base_rgb=(60, 150, 255),
            lat_max_deg=None, n_azim=n_az, pitch_angle_deg=pitch_angle_deg)
        traces.append(go.Mesh3d(x=ox, y=oy, z=oz, i=oi, j=oj, k=ok,
            vertexcolor=ocol, opacity=0.22, flatshading=False, hoverinfo='none',
            name=f'Outer belt ({outer_min_RE}-{outer_max_RE} RE, IGRF)', showlegend=True))
        ix, iy, iz, ii, ij, ik_, icol = igrf_mesh(
            inner_min_RE, inner_max_RE, date, axis, base_rgb=(255, 205, 60),
            lat_max_deg=None, n_azim=n_az, pitch_angle_deg=pitch_angle_deg)
        traces.append(go.Mesh3d(x=ix, y=iy, z=iz, i=ii, j=ij, k=ik_,
            vertexcolor=icol, opacity=0.36, flatshading=False, hoverinfo='none',
            name=f'Inner belt ({inner_min_RE}-{inner_max_RE} RE, IGRF)', showlegend=True))
    elif style == "lshell":
        n_az = 96 if hi else 48
        ox, oy, oz, oi, oj, ok, ocol = _dipole_belt_mesh(
            outer_min_RE, outer_max_RE, axis, base_rgb=(60, 150, 255),
            lat_max_deg=outer_lat_max, n_azim=n_az)
        traces.append(go.Mesh3d(x=ox, y=oy, z=oz, i=oi, j=oj, k=ok,
            vertexcolor=ocol, opacity=0.22, flatshading=False, hoverinfo='none',
            name=f'Outer belt ({outer_min_RE}-{outer_max_RE} RE)', showlegend=True))
        ix, iy, iz, ii, ij, ik_, icol = _dipole_belt_mesh(
            inner_min_RE, inner_max_RE, axis, base_rgb=(255, 205, 60),
            lat_max_deg=inner_lat_max, n_azim=n_az)
        traces.append(go.Mesh3d(x=ix, y=iy, z=iz, i=ii, j=ij, k=ik_,
            vertexcolor=icol, opacity=0.36, flatshading=False, hoverinfo='none',
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

    try:
        try:
            from . import magfield_plot_3d as _mf0
        except ImportError:
            import magfield_plot_3d as _mf0
        _mf0.set_external_model(None)
    except Exception:
        pass

    # 3. Earth (after belts so it occludes them)
    _edate = locals().get('date', None)
    traces.append(_build_earth_mesh(texture_path,
                                    n_lon=480 if hi else 240,
                                    n_lat=240 if hi else 120,
                                    sun_shading=sun_shading, date=_edate,
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

    # 4. Dipole axis
    ax_len = outer_max_RE * RE * 1.1
    traces.append(go.Scatter3d(
        x=[-axis[0]*ax_len, axis[0]*ax_len],
        y=[-axis[1]*ax_len, axis[1]*ax_len],
        z=[-axis[2]*ax_len, axis[2]*ax_len],
        mode='lines', line=dict(color='#66e0ff', width=2, dash='dash'),
        name='Magnetic dipole axis', showlegend=True))

    # 5. Magnetic equator
    if show_mag_equator:
        ex, ey, ez = _mag_equator_ring(axis, outer_max_RE*RE*1.05)
        traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='#3a6a8a', width=1.5), name='Magnetic equator',
            hoverinfo='none', showlegend=True))

    # 6. Dip poles
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
    src = {'_done': 'BELTS: NASA AE-8/AP-8 CLIMATOLOGICAL MEAN FLUX ON IGRF FIELD GEOMETRY '
                    '&#183; MEASURED ~18x ABOVE GOES &gt;2.1 MeV ELECTRONS, WHICH VARY 82x IN 8 DAYS',
           'igrf': f'IGRF L-SHELLS BOUNDED BY MIRROR POINTS, PITCH {pitch_angle_deg:.0f}\u00b0 (SHOWS SOUTH ATLANTIC ANOMALY)',
           'lshell': 'BELT SHELLS FROM DIPOLE L-SHELL MODEL',
           'torus': 'BELT SHELLS FROM TORUS APPROXIMATION'}[style]
    _annotations = [dict(
        x=0.01, y=0.02, xref='paper', yref='paper', xanchor='left',
        text=f'{src} &#183; INNER: PROTONS &#183; OUTER: ELECTRONS &#183; WGS84 EARTH',
        font=dict(color=_INK_FAINT, size=9, family='Helvetica, Arial'),
        showarrow=False)]
    if show_belt_residence and belt_residence:
        try:
            t_in  = belt_residence.get("t_inner_s", 0.0)/60.0
            t_out = belt_residence.get("t_outer_s", 0.0)/60.0
            per   = belt_residence.get("period_s", 0.0)/60.0
            _annotations.append(dict(x=0.99, y=0.06, xref='paper', yref='paper',
                xanchor='right',
                text=(f'Time in belts per {per:.1f}-min orbit &#183; '
                      f'inner {t_in:.2f} min &#183; outer {t_out:.2f} min'),
                font=dict(color=_ACCENT_WARM, size=10, family='Helvetica, Arial'),
                showarrow=False))
        except Exception:
            pass

    sub = ('AE-8/AP-8 mean-flux contour' if style == '_done'
           else 'IGRF-traced L-shells' if style == 'igrf'
           else 'dipole L-shell model' if style == 'lshell' else 'torus model')
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=(f'{title}<br><span style="font-size:12px;color:{_INK_SOFT};">'
                             f'{sub} &#183; aligned to IGRF 2025 magnetic axis</span>'),
                       font=dict(color='#f2f2f5', size=20, family='Helvetica, Arial'),
                       x=0.5, y=0.96, xanchor='center'),
            paper_bgcolor=_BG,
            scene=dict(
                bgcolor=_BG,
                xaxis=dict(visible=False, range=[-box, box]),
                yaxis=dict(visible=False, range=[-box, box]),
                zaxis=dict(visible=False, range=[-box, box]),
                camera=dict(eye=_camera_eye(elev, azim, eye_dist),
                            up=dict(x=0, y=0, z=1)),
                aspectmode='cube',
                dragmode='orbit'),
            width=width, height=height,
            margin=dict(l=0, r=100, t=64, b=0),
            legend=dict(font=dict(color='#d8d8de', size=11),
                        bgcolor='rgba(10,10,18,0.6)', bordercolor='#2a2a38',
                        borderwidth=1, x=0.01, y=0.95),
            annotations=_annotations,
            uirevision='van_allen_camera'),
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
                print(f"  PNG:  {png_out}", flush=True)
            except Exception as e:
                print(f"  PNG skipped: {e}", flush=True)

    if show:
        fig.show(config=PLOTLY_CONFIG)

    return fig


# -- Entry point -------------------------------------------------------------
# Mirrors magfield_plot_3d so the GUI can drive either module the same way:
# every key in GUI_CONFIG that names a real parameter is forwarded.
if __name__ == "__main__":
    import inspect as _inspect

    output_dir = Path(os.environ.get(
        "OUTPUT_DIR", str(output_root() / "figures")))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {}
    gui_cfg_path = os.environ.get("GUI_CONFIG", "")
    if gui_cfg_path and Path(gui_cfg_path).exists():
        try:
            _ns: dict = {}
            exec(compile(Path(gui_cfg_path).read_text(), gui_cfg_path, "exec"), {}, _ns)
            cfg = _ns
            print(f"[van_allen_plot_3d] Loaded GUI_CONFIG from {gui_cfg_path}")
        except Exception as e:
            print(f"[van_allen_plot_3d] Warning: could not parse GUI_CONFIG ({e})")

    epoch = cfg.get("epoch")
    if isinstance(epoch, str):
        try:
            _d = datetime.strptime(epoch[:19], "%Y-%m-%d %H:%M:%S")
            epoch = _d.year + (_d.timetuple().tm_yday - 1) / 365.25
        except Exception:
            epoch = 2025.0
    elif not isinstance(epoch, (int, float)):
        epoch = 2025.0

    _params = _inspect.signature(plot_van_allen_3d).parameters
    kwargs = {k: val for k, val in cfg.items()
              if k in _params and not k.startswith("_") and k != "epoch"}
    kwargs.setdefault("save_path", str(output_dir / "van_allen_plot_3d"))
    kwargs.setdefault("show", False)
    unknown = sorted(k for k in cfg if k not in _params and k != "epoch"
                     and not k.startswith("_") and not callable(cfg[k]))
    if unknown:
        print(f"[van_allen_plot_3d] ignoring unknown config keys: {unknown}")

    plot_van_allen_3d(epoch=epoch, **kwargs)
    print(f"[van_allen_plot_3d] Saved -> {output_dir / 'van_allen_plot_3d.html'}")
