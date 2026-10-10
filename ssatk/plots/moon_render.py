"""
moon_render.py — shared, high-quality Moon mesh for Plotly scenes
======================================================================
Used by eclipse_space_view_plotly.py (and anything else that needs a Moon:
globe_orbit_daynight_plotly.py's future lunar-orbit views, moon_plot_3d.py,
etc). Split out on its own so there's exactly one Moon renderer in the
toolkit instead of a slightly-different copy in every file that draws one.

What actually changes visual quality vs. the old _moon_mesh_plotly():

1. REAL TEXTURE FIRST, same pattern as globe_orbit_daynight_plotly's
   _load_real_earth_texture: try ssapy.utils.find_file("moon", ext=".png")
   before falling back to anything procedural. This is the single biggest
   quality jump when a real asset is present — no amount of procedural
   crater-painting matches an actual lunar photomosaic.

2. REAL PER-VERTEX DIFFUSE LIGHTING, not flat ambient=1/diffuse=0 over a
   painted albedo. The old version made craters *look* like paint swatches
   because only color changed, never brightness-by-surface-angle. This
   computes an actual outward normal at every vertex (from the displaced,
   bumpy sphere, via finite differences across the lat/lon grid — not the
   undisplaced sphere normal, so the bumps themselves cast the shading)
   and shades each point by max(0, normal . sun_hat). That's what makes
   craters read as 3D relief with a lit rim and a shadowed bowl, the way
   every real Moon photo looks, instead of a flat texture.

3. MARIA vs HIGHLANDS, not just "some albedo dips". The real Moon's most
   recognizable feature from Earth is the dark basaltic maria (Sea of
   Tranquility etc.) against brighter cratered highlands — large-scale
   low-frequency blobs, distinct from the small high-frequency craters.
   Previously there was only one texture scale (craters), so the disk
   read as uniformly grey rather than having the familiar patchy look.

4. Same real per-vertex Earth-shadow eclipse shading as before for lunar
   mode (illumination_fraction evaluated at each vertex's true physical
   position), just layered on top of the new base shading multiplicatively
   instead of being the only source of brightness variation.

Usage
-----
::

    from moon_render import moon_mesh_plotly
    fig.add_trace(moon_mesh_plotly(center, radius, sun_hat=sun_hat,
                                   real_center_km=moon_r_km, mode="lunar"))
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

from ssatk.constants import AU_KM, EARTH_RADIUS_KM, MOON_MEAN_RADIUS_KM, SUN_NOMINAL_RADIUS_KM

try:
    from .scene_primitives import stabilize_sphere_poles
except ImportError:
    from scene_primitives import stabilize_sphere_poles

try:
    from .eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
except ImportError:
    try:
        from eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
    except ImportError:
        illumination_fraction = None
        R_SUN_KM = SUN_NOMINAL_RADIUS_KM

R_MOON_KM = MOON_MEAN_RADIUS_KM

_moon_texture_cache = None


def _display_lunar_albedo(rgb):
    """Map a dark lunar photomosaic into a readable display albedo.

    Some packaged Moon textures are stored as low-exposure imagery.  Using those
    RGB values directly makes even an almost-full Moon render nearly black in
    Plotly.  Percentile-stretch the luminance, then remap it to a lunar-grey
    diagnostic albedo while preserving maria/crater contrast.
    """
    rgb = np.asarray(rgb, dtype=float) / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    low, high = np.nanpercentile(luminance, [1.0, 99.5])
    if not np.isfinite(high - low) or high <= low:
        stretched = np.clip(luminance, 0.0, 1.0)
    else:
        stretched = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    albedo = 0.34 + 0.58 * stretched
    return np.repeat(albedo[..., None], 3, axis=-1) * np.array([1.0, 0.985, 0.955])


def _load_real_moon_texture(n_lat, n_lon):
    """
    Real lunar photomosaic, same discovery pattern as the Earth texture in
    globe_orbit_daynight_plotly.py: ssapy.utils.find_file + PIL. This is
    what the checkpoint's Section 6 flagged as unconfirmed on the real
    machine — this is that check, applied to the Moon instead of Earth.
    """
    global _moon_texture_cache
    if _moon_texture_cache is not None and _moon_texture_cache.shape[:2] == (n_lat, n_lon):
        return _moon_texture_cache
    try:
        from ssapy.utils import find_file
        from PIL import Image
        img = Image.open(find_file("moon", ext=".png")).convert("RGB")
        img = img.resize((n_lon, n_lat), Image.LANCZOS)
        _moon_texture_cache = np.array(img)
        print("[moon_render] Moon base: real ssapy texture found")
        return _moon_texture_cache
    except Exception as ex:
        print(f"[moon_render] Real Moon texture not found ({ex}) — "
              f"using procedural maria+crater model instead.")
        return None


def _smooth(field, sigma=1.4):
    """Light Gaussian smoothing (wrapping in longitude) so procedural
    albedo/relief never shows hard tile/facet edges once displaced and
    lit — this is what turns a blocky, grid-like sphere (visible facet
    boundaries wherever two adjacent random blotches meet head-on) into
    a continuous, photographic-looking surface. Falls back to an
    unsmoothed field if scipy isn't available."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma, mode="wrap")
    except Exception:
        return field


def _procedural_moon_albedo(Lat, Lon, seed=3):
    """Two-scale albedo: large dark maria + small bright/dark crater fields,
    matching how the real Moon's near side actually looks patchy rather
    than uniformly grey."""
    rng = np.random.default_rng(seed)

    # --- Maria: a handful of large, soft, irregular dark blobs, mostly
    # confined to one hemisphere (real maria are lopsided — concentrated
    # on the historically Earth-facing side) ---
    albedo = np.full_like(Lat, 0.78)
    n_maria = 8
    for _ in range(n_maria):
        clat = rng.uniform(-45, 45)
        clon = rng.uniform(-60, 60)          # biased to one hemisphere
        spread_lat = rng.uniform(10, 26)
        spread_lon = rng.uniform(14, 32)
        dlat = (Lat - clat)
        dlon = (Lon - clon + 180) % 360 - 180
        d2 = (dlat / spread_lat) ** 2 + (dlon / spread_lon) ** 2
        albedo -= 0.22 * np.exp(-d2 / 2)

    # --- Crater fields on top: large + medium, then many small ---
    for _ in range(60):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(4, 11)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        albedo -= 0.12 * bowl ** 3
    for _ in range(500):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(0.4, 2.2)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        albedo -= 0.10 * bowl ** 3

    albedo += rng.normal(0, 0.01, Lat.shape)  # fine regolith speckle
    albedo = _smooth(albedo, sigma=0.45)       # just enough to kill hard blotch edges
    return np.clip(albedo, 0.30, 1.0)


def _procedural_moon_relief(Lat, Lon, seed=3):
    """Radial bump field: bright rim + dark bowl per crater, at two size
    scales, plus fine roughness noise — this is what gets displaced into
    actual 3D geometry (not just painted) so the lighting step below has
    real bumps to shade."""
    rng = np.random.default_rng(seed)
    relief = np.zeros_like(Lat)
    for _ in range(60):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(4, 11)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        relief -= 0.65 * bowl ** 3
        rim = np.clip(1 - np.abs(d - rad * 0.92) / (rad * 0.1), 0, 1)
        relief += 0.45 * rim
    for _ in range(500):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        rad = rng.uniform(0.4, 2.2)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        bowl = np.clip(1 - d / rad, 0, 1)
        relief -= 0.5 * bowl ** 3
        rim = np.clip(1 - np.abs(d - rad * 0.9) / (rad * 0.2), 0, 1)
        relief += 0.35 * rim
    fine_noise = np.zeros_like(Lat)
    for _ in range(25):
        clat, clon = rng.uniform(-90, 90), rng.uniform(-180, 180)
        spread = rng.uniform(2, 6)
        d = np.sqrt((Lat - clat) ** 2 + ((Lon - clon + 180) % 360 - 180) ** 2)
        fine_noise += rng.uniform(-0.15, 0.15) * np.exp(-(d ** 2) / (2 * spread ** 2))
    relief += fine_noise
    # Smooth before displacement — this is what removes any visible
    # facet/grid pattern from the mesh: unsmoothed relief has sharp
    # per-blotch edges that, once turned into real geometry and lit,
    # show up as a faint quad grid across the sphere (exactly the
    # artifact in the reference screenshot). A touch of blur keeps the
    # crater shapes but rounds their edges into the surrounding terrain.
    return _smooth(relief, sigma=0.4)


def _vertex_normals(X, Y, Z, center=None):
    """Return outward normals of a displaced grid.

    Finite differences are translation-invariant, but the final outward-facing
    orientation test must use body-local coordinates.  A Moon plotted hundreds
    of thousands of kilometres from Earth otherwise orients normals using the
    Earth-centered position vector and can render an almost-full Moon as dark.
    """
    Xu = np.gradient(X, axis=1); Yu = np.gradient(Y, axis=1); Zu = np.gradient(Z, axis=1)
    Xv = np.gradient(X, axis=0); Yv = np.gradient(Y, axis=0); Zv = np.gradient(Z, axis=0)
    nx = Yu * Zv - Zu * Yv
    ny = Zu * Xv - Xu * Zv
    nz = Xu * Yv - Yu * Xv
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-12
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    # Orient outward (dot with the sphere's own radial direction should be
    # positive; flip any that came out pointing inward from the cross
    # product's arbitrary handedness)
    if center is None:
        Xr, Yr, Zr = X, Y, Z
    else:
        cx, cy, cz = np.asarray(center, dtype=float)
        Xr, Yr, Zr = X - cx, Y - cy, Z - cz
    R = np.sqrt(Xr ** 2 + Yr ** 2 + Zr ** 2) + 1e-12
    rad_dot = (nx * Xr + ny * Yr + nz * Zr) / R
    flip = rad_dot < 0
    nx, ny, nz = np.where(flip, -nx, nx), np.where(flip, -ny, ny), np.where(flip, -nz, nz)
    return nx, ny, nz


def moon_mesh_plotly(center, radius, sun_hat=None, seed=3,
                     real_center_km=None, mode="lunar",
                     eclipse_tint=True, n_lat=180, n_lon=360):
    """
    Real-texture-first, properly-lit Moon mesh.

    center, radius : display-space sphere placement (km, display units —
                      same size_boost convention as the rest of the toolkit)
    sun_hat         : unit vector toward the Sun, for real diffuse shading
                      and (mode="lunar") the Earth-shadow eclipse check
    real_center_km  : TRUE physical Moon-center position relative to Earth
                      (only needed for mode="lunar" eclipse shading)
    mode            : "lunar" (Moon may be in Earth's shadow) or "solar"
                      (Moon is just normally sunlit; it's Earth being
                      shadowed elsewhere in the scene)
    eclipse_tint    : apply the warm/red multiplicative bias during deep
                      shadow (mode="lunar" only) — set False if the caller
                      wants to apply its own tint instead
    """
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, lat)
    latr, lonr = np.radians(Lat), np.radians(Lon)
    nx0, ny0, nz0 = np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)

    tex = _load_real_moon_texture(n_lat, n_lon)
    if tex is not None:
        tex = np.roll(tex, n_lon // 2, axis=1)  # same antimeridian fix as Earth
        base_rgb = _display_lunar_albedo(tex)
        # Still add a light procedural relief for lighting, even with a
        # real texture — real texture gives *color*, but most Moon photo
        # textures are pre-shaded/flat-lit, so without geometric bumps the
        # sphere would look like a printed ball, not a rocky one.
        relief = _procedural_moon_relief(Lat, Lon, seed=seed) * 0.5
    else:
        base_rgb = None
        relief = _procedural_moon_relief(Lat, Lon, seed=seed)

    R_disp = stabilize_sphere_poles(radius * (1 + relief * 0.009))
    X = center[0] + R_disp * nx0
    Y = center[1] + R_disp * ny0
    Z = center[2] + R_disp * nz0
    nxr, nyr, nzr = _vertex_normals(X, Y, Z, center=center)

    if base_rgb is None:
        albedo = _procedural_moon_albedo(Lat, Lon, seed=seed)
        base_rgb = np.repeat(albedo[..., None], 3, axis=-1) * np.array([1.0, 0.98, 0.94])

    # Real per-vertex diffuse lighting from the bumpy normals — this is
    # what makes craters look sculpted instead of painted. Ambient floor
    # so the unlit portion of a gibbous/crescent Moon isn't pure black.
    if sun_hat is not None:
        diffuse = np.clip(nxr * sun_hat[0] + nyr * sun_hat[1] + nzr * sun_hat[2], 0, 1)
        shading = np.clip(0.14 + 0.86 * diffuse ** 0.80, 0.14, 1.0)
        # Opposition effect: real regolith backscatters extra strongly
        # right where the viewer, Sun, and surface point are nearly
        # aligned (why a full Moon looks noticeably brighter overall
        # than the diffuse-only model predicts, not just "more lit
        # area"). Approximated with the same fixed view-ish reference
        # used for the Sun's limb darkening elsewhere in the toolkit —
        # imperfect (doesn't track the actual camera), but keeps the lit
        # face from looking flat and grey the way a pure Lambertian
        # sphere does.
        opp = np.clip(diffuse - 0.92, 0, 1) / 0.08
        shading = np.clip(shading + 0.18 * opp, 0.14, 1.15)
    else:
        shading = np.ones_like(Lat)

    rgb = np.clip(base_rgb * shading[..., None], 0, 1)

    if mode == "lunar" and real_center_km is not None and sun_hat is not None \
            and illumination_fraction is not None:
        real_surface = np.stack([nx0, ny0, nz0], axis=-1) * R_MOON_KM
        real_positions = real_center_km[None, None, :] + real_surface
        flat_pos = real_positions.reshape(-1, 3)
        flat_sun = np.tile(sun_hat, (flat_pos.shape[0], 1))
        illum = illumination_fraction(flat_pos, flat_sun, R_body_km=EARTH_RADIUS_KM,
                                      R_sun_km=R_SUN_KM, D_km=AU_KM).reshape(Lat.shape)
        floor = 0.12
        eclipse_brightness = np.clip(floor + (1 - floor) * illum, floor, 1.0)
        rgb = rgb * eclipse_brightness[..., None]
        if eclipse_tint:
            red_mix = np.clip((0.35 - illum) / 0.35, 0, 1) ** 1.5
            warm = np.array([1.15, 0.55, 0.42])
            tint = (1 - red_mix[..., None]) + warm[None, None, :] * red_mix[..., None]
            rgb = np.clip(rgb * tint, 0, 1)

    rgb = stabilize_sphere_poles(rgb)

    vertexcolor = [f"rgb({int(rgb[r, c, 0]*255)},{int(rgb[r, c, 1]*255)},{int(rgb[r, c, 2]*255)})"
                  for r in range(n_lat) for c in range(n_lon)]

    ii, jj, kk = [], [], []
    for r in range(n_lat - 1):
        for c in range(n_lon):
            cn = (c + 1) % n_lon
            v0 = r * n_lon + c; v1 = r * n_lon + cn; v2 = (r + 1) * n_lon + c; v3 = (r + 1) * n_lon + cn
            ii += [v0, v1]; jj += [v1, v3]; kk += [v2, v2]

    return go.Mesh3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), i=ii, j=jj, k=kk,
        vertexcolor=vertexcolor, flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name="Moon", hoverinfo="skip", showlegend=False,
    )
