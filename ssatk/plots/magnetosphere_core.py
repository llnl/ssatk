"""
magnetosphere_core.py — geometry, Earth, sky and texture helpers shared by the
SSATK magnetosphere plots.

These routines used to be duplicated in magfield_plot_3d.py and
van_allen_plot_3d.py. That duplication silently drifted three separate times:
van_allen kept a crude subsolar-point calculation and a cosmetic belt gradient
for several revisions after magfield was corrected, and later kept a
quadrature with a 2e-3 error after magfield's was fixed. A parity test caught
each one, but the only structural fix is a single definition, which is what
this module is.

Nothing here depends on ppigrf, geopack or spacepy, so it imports cleanly in a
minimal environment.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np

try:
    from .scene_primitives import stabilize_sphere_poles
except ImportError:
    from scene_primitives import stabilize_sphere_poles

# Sky and astrometry live in starfield.py, which the rest of the toolkit
# already uses for matplotlib plots.  Re-exported here so existing callers keep
# working; there is only one definition.
try:
    from .starfield import (ARCSEC, _HYG_PATHS, _SPECT_COLORS, _bv_to_teff,
                            _cie_xyz_bar, _gmst_rad, _julian_date,
                            _precession_matrix, _spect_fallback_rgb,
                            _stars_to_ecef, _teff_to_srgb, star_directions,
                            starfield_traces)
except ImportError:
    from starfield import (ARCSEC, _HYG_PATHS, _SPECT_COLORS, _bv_to_teff,
                           _cie_xyz_bar, _gmst_rad, _julian_date,
                           _precession_matrix, _spect_fallback_rgb,
                           _stars_to_ecef, _teff_to_srgb, star_directions,
                           starfield_traces)


# Constants and geometry moved to ssatk.geomagnetics, one layer below
# this module, so the physics layer no longer imports the plotting package.
# Re-exported here under the same names -- references, not copies -- so the
# plot modules and the tests that patch them keep working.
#
# EARTH_RADIUS_KM here is 6371.0, the spherical mean radius used for magnetic
# work -- NOT constants.EARTH_RADIUS_KM, which is 6378.137 WGS84 equatorial.
try:
    from ..geomagnetics import (  # noqa: F401
        EARTH_RADIUS_KM, WGS84_A_KM, WGS84_B_KM,
        _DIPOLE_LON_DEG, _DIPOLE_TILT_DEG,
        _dipole_axis, _mag_basis, _subsolar_point, _texture_cache_dir,
    )
except ImportError:
    from ssatk.geomagnetics import (  # noqa: F401
        EARTH_RADIUS_KM, WGS84_A_KM, WGS84_B_KM,
        _DIPOLE_LON_DEG, _DIPOLE_TILT_DEG,
        _dipole_axis, _mag_basis, _subsolar_point, _texture_cache_dir,
    )

# Re-exported for the plot modules and tests; declared so static analysis
# knows they are intentional pass-throughs rather than dead imports.
__all__ = [
    "ARCSEC", "_HYG_PATHS", "_SPECT_COLORS", "_bv_to_teff", "_cie_xyz_bar",
    "_gmst_rad", "_julian_date", "_precession_matrix", "_spect_fallback_rgb",
    "_stars_to_ecef", "_teff_to_srgb", "star_directions", "starfield_traces",
    "_build_starfield_trace",
]


def _build_starfield_trace(sky_radius, date=None, mag_limit=6.5):
    """Earth-fixed starfield for the Plotly scenes.  Thin wrapper over
    starfield.starfield_traces(frame='ecef')."""
    return starfield_traces(sky_radius, when=date, frame="ecef",
                            mag_limit=mag_limit)


try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


STAR_SPHERE_FACTOR = 50.0


_CAM_FILL = 1.35


_TEXTURE_SEARCH = [
    "~/earth_texture.jpg",
    "~/blue_marble.jpg",
    "~/ssatk/assets/earth_texture.jpg",
    "~/SSAPy-Toolkit/assets/earth_texture.jpg",   # checkouts made before the repo rename
    "~/SSAPy/ssapy/data/earth_texture.jpg",
    "./assets/earth_texture.jpg",
]


EARTH_TEXTURE_URLS = [
    # NASA Blue Marble, public domain (Earth Observatory)
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    "world.topo.bathy.200412.3x5400x2700.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
    "land_shallow_topo_2048.jpg",
]




def _download_earth_texture(timeout=90):
    """Fetch NASA Blue Marble once into the cache.  Returns a Path or None."""
    import urllib.request
    dest = _texture_cache_dir() / "earth_texture.jpg"
    if dest.exists() and dest.stat().st_size > 50_000:
        return dest
    for url in EARTH_TEXTURE_URLS:
        try:
            print(f"  downloading Earth texture: {url.rsplit('/', 1)[-1]} ...", flush=True)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if tmp.stat().st_size > 50_000:
                tmp.replace(dest)
                print(f"  cached -> {dest}", flush=True)
                return dest
        except Exception as e:
            print(f"  texture download failed ({e}); trying next source", flush=True)
    return None


def _resolve_texture_path(texture_path, allow_download=True):
    """Turn texture_path (None | 'auto' | path) into a concrete file, if any."""
    if texture_path is False or str(texture_path).lower() in ("none", "off", "flat"):
        return None
    if texture_path is not None and str(texture_path).lower() != "auto":
        p = Path(str(texture_path)).expanduser()
        if p.exists():
            return p
        print(f"  texture not found at {p}; falling back to auto", flush=True)
    for cand in _TEXTURE_SEARCH:
        p = Path(cand).expanduser()
        if p.exists():
            return p
    cached = _texture_cache_dir() / "earth_texture.jpg"
    if cached.exists() and cached.stat().st_size > 50_000:
        return cached
    if allow_download:
        return _download_earth_texture()
    return None


_MAG_NORTH = (136.0, 85.5)


_MAG_SOUTH = (135.0, -64.0)


_BG          = '#07070d'


_INK_SOFT    = '#8a8a99'


_INK_FAINT   = '#5a5a68'


_ACCENT_WARM = '#FFD24A'


_FIELD_CMAP  = 'Plasma'


PLOTLY_CONFIG = dict(
    scrollZoom=True,
    displaylogo=False,
    responsive=True,
    displayModeBar=True,
    toImageButtonOptions=dict(format='png', scale=2),
)


def _load_texture_rgb(texture_path):
    """Load an equirectangular texture as an (H, W, 3) uint8 array."""
    if texture_path is None:
        return None
    p = Path(str(texture_path))
    if not p.exists():
        return None
    try:
        if p.suffix.lower() == '.npz':
            z = np.load(p)
            arr = z[list(z.keys())[0]]
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            return arr[..., :3]
        if not _HAS_PIL:
            print("  Pillow not installed — cannot read image textures", flush=True)
            return None
        return np.array(_PIL_Image.open(p).convert('RGB'), dtype=np.uint8)
    except Exception as e:
        print(f"  texture load failed: {e}", flush=True)
        return None


def _sample_equirect_bilinear(tex, lat_deg, lon_deg, prefilter_to=None):
    """
    Sample an equirectangular texture at given geodetic lat/lon.

    The texture is first area-averaged down to roughly twice the mesh
    resolution (a mip step, so fine coastlines do not alias into noise),
    then bilinearly interpolated.  Longitude wraps at the seam.
    """
    if prefilter_to is not None and _HAS_PIL:
        tw, th = prefilter_to
        if tex.shape[1] > tw or tex.shape[0] > th:
            tex = np.array(_PIL_Image.fromarray(tex).resize((int(tw), int(th)),
                                                            _PIL_Image.BOX), dtype=np.uint8)
    th_, tw_, _ = tex.shape
    fr = (90.0 - lat_deg) / 180.0 * (th_ - 1)
    fc = (lon_deg + 180.0) / 360.0 * tw_
    r0 = np.clip(np.floor(fr).astype(int), 0, th_ - 1)
    r1 = np.clip(r0 + 1, 0, th_ - 1)
    c0 = np.floor(fc).astype(int) % tw_
    c1 = (c0 + 1) % tw_
    wr = (fr - np.floor(fr))[..., None]
    wc = (fc - np.floor(fc))[..., None]
    t = tex.astype(np.float32)
    top = t[r0, c0] * (1 - wc) + t[r0, c1] * wc
    bot = t[r1, c0] * (1 - wc) + t[r1, c1] * wc
    return np.clip(top * (1 - wr) + bot * wr, 0, 255).astype(np.uint8)


def _build_earth_mesh(texture_path, n_lon=480, n_lat=240, sun_shading=False,
                      date=None, night_floor=0.30, allow_download=True):
    """
    WGS84 oblate ellipsoid with a real equirectangular texture.

    Vertices lie on a geodetic latitude grid and are converted with the proper
    prime-vertical radius, so the surface is the true reference ellipsoid
    rather than a sphere.  Colours are per-vertex uint8 (compact in the saved
    HTML) sampled from the texture with area prefilter + bilinear
    interpolation.
    """
    a, b = WGS84_A_KM, WGS84_B_KM
    e2   = 1.0 - (b * b) / (a * a)

    lat = np.linspace(90.0, -90.0, n_lat)      # north -> south (texture row order)
    lon = np.linspace(-180.0, 180.0, n_lon)    # texture left -> right
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')

    slat = np.sin(np.radians(LAT)); clat = np.cos(np.radians(LAT))
    N = a / np.sqrt(1.0 - e2 * slat**2)
    X = N * clat * np.cos(np.radians(LON))
    Y = N * clat * np.sin(np.radians(LON))
    Z = (N * (1.0 - e2)) * slat

    # vectorised quad -> triangle indexing
    jj, ii = np.meshgrid(np.arange(n_lat - 1), np.arange(n_lon - 1), indexing='ij')
    v00 = (jj * n_lon + ii).ravel()
    v01 = v00 + 1
    v10 = v00 + n_lon
    v11 = v10 + 1
    ti = np.concatenate([v00, v00])
    tj = np.concatenate([v01, v11])
    tk = np.concatenate([v11, v10])

    resolved = _resolve_texture_path(texture_path, allow_download=allow_download)
    tex = _load_texture_rgb(resolved)
    if tex is not None:
        colors = _sample_equirect_bilinear(tex, LAT, LON,
                                           prefilter_to=(n_lon * 2, n_lat * 2))
        print(f"  Earth texture: {Path(resolved).name} "
              f"({tex.shape[1]}x{tex.shape[0]}) -> {n_lon}x{n_lat} WGS84 mesh", flush=True)
    else:
        base = np.array([28, 74, 120], dtype=np.float32)
        colors = np.tile(base, (n_lat, n_lon, 1)).astype(np.uint8)
        print("  no Earth texture available — using flat colour", flush=True)

    baked_sun_shading = bool(sun_shading and date is not None)
    if baked_sun_shading:
        sd, sl = _subsolar_point(date)
        sun = np.array([np.cos(np.radians(sd)) * np.cos(np.radians(sl)),
                        np.cos(np.radians(sd)) * np.sin(np.radians(sl)),
                        np.sin(np.radians(sd))])
        nrm = np.stack([clat * np.cos(np.radians(LON)),
                        clat * np.sin(np.radians(LON)), slat], axis=-1)
        cosang = np.clip((nrm * sun).sum(-1), 0.0, 1.0)
        shade = night_floor + (1.0 - night_floor) * cosang
        colors = np.clip(colors.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    colors = stabilize_sphere_poles(colors)

    lighting = (
        dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
        if baked_sun_shading
        else dict(ambient=0.95, diffuse=0.16, specular=0.02, roughness=0.95)
    )

    return go.Mesh3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), i=ti, j=tj, k=tk,
        vertexcolor=colors.reshape(-1, 3),
        showscale=False, hoverinfo='none',
        lighting=lighting,
        name='Earth', showlegend=False,
    )


def _ellipsoid_shell(alt_km, n_lon=96, n_lat=48):
    """Vertices/faces for a WGS84-shaped shell at a given altitude."""
    a, b = WGS84_A_KM + alt_km, WGS84_B_KM + alt_km
    lat = np.radians(np.linspace(90.0, -90.0, n_lat))
    lon = np.radians(np.linspace(-180.0, 180.0, n_lon))
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    X = a * np.cos(LAT) * np.cos(LON)
    Y = a * np.cos(LAT) * np.sin(LON)
    Z = b * np.sin(LAT)
    jj, ii = np.meshgrid(np.arange(n_lat-1), np.arange(n_lon-1), indexing='ij')
    v00 = (jj*n_lon + ii).ravel(); v01 = v00+1; v10 = v00+n_lon; v11 = v10+1
    return (X.ravel(), Y.ravel(), Z.ravel(),
            np.concatenate([v00, v00]), np.concatenate([v01, v11]),
            np.concatenate([v11, v10]))


def _atmosphere_traces(n_shells=4, top_km=60.0, scale_height_km=8.5,
                       tau_vertical=0.10, color=(120, 170, 255)):
    """
    Rayleigh haze from a real atmospheric density profile.

    Density falls as exp(-h/H) with H = 8.5 km, and the vertical Rayleigh
    optical depth at 550 nm is about 0.10.  Each shell is given the opacity
    implied by the fraction of the column it contains, so the limb brightening
    comes from path length through a correctly weighted atmosphere rather than
    from stacked shells chosen to look right.

    Note on scale: the optically significant atmosphere is ~60 km deep, under
    1% of Earth's radius.  At the default framing that is well under a pixel,
    which is why this is off by default — it only becomes meaningful on a
    close-up view.
    """
    out = []
    edges = np.linspace(0.0, top_km, n_shells + 1)
    for k in range(n_shells):
        h0, h1 = edges[k], edges[k + 1]
        frac = (np.exp(-h0 / scale_height_km) - np.exp(-h1 / scale_height_km))
        tau = tau_vertical * frac
        op = float(np.clip(1.0 - np.exp(-tau), 0.0, 1.0))
        if op < 1e-4:
            continue
        x, y, z, i, j, kk = _ellipsoid_shell(0.5 * (h0 + h1))
        out.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=kk,
                             color=f'rgb({color[0]},{color[1]},{color[2]})',
                             opacity=op, flatshading=False, hoverinfo='none',
                             showlegend=False,
                             lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                             name='Atmosphere'))
    return out


def _camera_eye(elev, azim, dist):
    e = np.radians(elev); a = np.radians(azim)
    return dict(x=float(dist*np.cos(e)*np.cos(a)),
                y=float(dist*np.cos(e)*np.sin(a)),
                z=float(dist*np.sin(e)))


def _mpl_to_plotly_camera(elev, azim, dist=2.0):
    return _camera_eye(elev, azim, dist)


def _geo_to_xyz(lon_deg, lat_deg, r=EARTH_RADIUS_KM):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return (r * np.cos(la) * np.cos(lo),
            r * np.cos(la) * np.sin(lo),
            r * np.sin(la))








def _rotation_z_to_axis(axis):
    z  = np.array([0.0, 0.0, 1.0])
    v  = np.cross(z, axis)
    s  = np.linalg.norm(v)
    c  = np.dot(z, axis)
    if s < 1e-10:
        return np.eye(3) if c > 0 else np.diag([1.0, 1.0, -1.0])
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)




def _belt_vertex_color(base_rgb, weight, floor=0.10):
    """Map a normalised particle-density weight to a vertex colour."""
    t = floor + (1.0 - floor) * float(np.clip(weight, 0.0, 1.0))
    r, g, b = [int(np.clip(ch * t, 0, 255)) for ch in base_rgb]
    return f'rgb({r},{g},{b})'


def _omnidirectional_flux_ratio(b_over_beq, n=2.0):
    """
    Relative omnidirectional trapped-particle flux at a point on a field line
    where the field has risen to `b_over_beq` times its equatorial value.

    Conservation of the first adiabatic invariant maps a particle's equatorial
    pitch angle a_eq to a local pitch angle a via sin^2(a) = (B/B_eq) sin^2(a_eq).
    Only particles with sin^2(a_eq) <= B_eq/B are still present; the rest have
    already mirrored below.  Integrating an equatorial distribution
    j(a_eq) ~ sin^n(a_eq) over the surviving population and over solid angle
    gives the density profile along the line.  This replaces the arbitrary
    "bright core" falloff used previously — the brightness is now the actual
    modelled particle density, not a decoration.
    """
    b = np.maximum(np.asarray(b_over_beq, dtype=float), 1.0)
    a_max = np.arcsin(np.clip(np.sqrt(1.0 / b), 0.0, 1.0))     # loss boundary
    # Integrate each row on its own 0..a_max grid.  Masking a shared grid
    # instead quantises the upper limit to the nearest node, which cost ~2e-3
    # against the closed-form solution.
    u = np.linspace(0.0, 1.0, 512)[None, :]
    a = a_max[:, None] * u
    num = np.trapezoid(np.sin(a) ** (n + 1.0), a, axis=1)
    a_full = (np.pi / 2) * u
    den = np.trapezoid(np.sin(a_full) ** (n + 1.0), a_full, axis=1)
    return num / np.maximum(den, 1e-30)


def _lshell_cross_section(L_min, L_max, lat_max_deg, n_lat):
    """Analytic dipole L-shell meridional boundary (belt_style='lshell')."""
    RE  = EARTH_RADIUS_KM
    lat = np.radians(np.linspace(-lat_max_deg, lat_max_deg, n_lat))
    c2  = np.cos(lat)**2
    rho = np.concatenate([L_max*RE*np.cos(lat)**3, (L_min*RE*np.cos(lat)**3)[::-1]])
    z   = np.concatenate([L_max*RE*c2*np.sin(lat), (L_min*RE*c2*np.sin(lat))[::-1]])
    lat_core = 1.0 - np.abs(np.linspace(-1, 1, n_lat))
    core = np.concatenate([lat_core, lat_core[::-1] * 0.7])
    return rho, z, core


def _dipole_belt_mesh(L_min, L_max, axis, base_rgb, lat_max_deg=40.0,
                      n_lat=48, n_azim=96, edge_dim=None, pitch_angle_deg=25.0,
                      pad_index=2.0):
    """
    Analytic dipole L-shell belt.  Brightness uses the same adiabatic-invariant
    density profile as the IGRF path, with B/B_eq from the dipole relation
    B/B_eq = sqrt(1 + 3 sin^2(lat)) / cos^6(lat).
    """
    RE = EARTH_RADIUS_KM
    lat = np.radians(np.linspace(-lat_max_deg, lat_max_deg, n_lat))
    c2 = np.cos(lat)**2
    ratio = np.sqrt(1.0 + 3.0*np.sin(lat)**2) / np.cos(lat)**6
    w_line = _omnidirectional_flux_ratio(ratio, n=pad_index)
    rho = np.concatenate([L_max*RE*np.cos(lat)**3, (L_min*RE*np.cos(lat)**3)[::-1]])
    z   = np.concatenate([L_max*RE*c2*np.sin(lat), (L_min*RE*c2*np.sin(lat))[::-1]])
    w   = np.concatenate([w_line, w_line[::-1]])
    M = len(rho)
    phi = np.linspace(0, 2*np.pi, n_azim, endpoint=False)
    R = _rotation_z_to_axis(axis)
    xs, ys, zs, cols = [], [], [], []
    for a in range(n_azim):
        ca, sa = np.cos(phi[a]), np.sin(phi[a])
        for k in range(M):
            p = R @ np.array([rho[k]*ca, rho[k]*sa, z[k]])
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
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


def _torus_mesh3d(R_km, r_km, axis, n_major=80, n_minor=40):
    """Legacy plain-torus belt (belt_style='torus')."""
    u  = np.linspace(0, 2*np.pi, n_major, endpoint=False)
    v  = np.linspace(0, 2*np.pi, n_minor, endpoint=False)
    U, V = np.meshgrid(u, v)
    X = (R_km + r_km*np.cos(V))*np.cos(U)
    Y = (R_km + r_km*np.cos(V))*np.sin(U)
    Z = r_km*np.sin(V)
    R = _rotation_z_to_axis(axis)
    pts = R @ np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    xs, ys, zs = pts[0], pts[1], pts[2]
    nm, nM = n_minor, n_major
    ti, tj, tk = [], [], []
    for j in range(nm):
        for i in range(nM):
            j1=(j+1)%nm; i1=(i+1)%nM
            v00=j*nM+i; v01=j*nM+i1; v10=j1*nM+i; v11=j1*nM+i1
            ti+=[v00,v00]; tj+=[v01,v11]; tk+=[v11,v10]
    return xs, ys, zs, ti, tj, tk


def _mag_equator_ring(axis, radius_km, n=240):
    _, e1, e2 = _mag_basis(axis)
    t = np.linspace(0, 2*np.pi, n)
    pts = radius_km * (np.cos(t)[:, None]*e1 + np.sin(t)[:, None]*e2)
    return pts[:, 0], pts[:, 1], pts[:, 2]
