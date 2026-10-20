"""High-quality, fixed-camera rendering for validated eclipse animations."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageFilter

from ssapy_toolkit.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    RP_KM,
    R_SUN_KM,
    R_MOON_MEAN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    SOLAR_PENUMBRA_OPTICAL_RADIUS_KM,
    ReferenceDefinition,
    ReferenceEvent,
    angular_circle_visible_fraction,
    geodetic_from_itrf,
    itrf_surface_point,
    jd_to_datetime,
    lunar_reference_state,
    solar_besselian_state,
    solar_central_line_wgs84,
    solar_observer_geometry,
    solar_local_contacts,
    _unit,
    _LUNAR_MOON_DISTANCE_KM,
    _LUNAR_MOON_SD_DEG,
    _LUNAR_P_RADIUS_DEG,
    _LUNAR_U_RADIUS_DEG,
)
from ssapy_toolkit.compute.eclipse_brightness import gcrf_to_itrf_km
from ssapy_toolkit.io.eclipse_asset_resolver import resolve_asset, resolve_image
from ssapy_toolkit.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    LUNAR_OPTICAL_MOON_RADIUS_KM,
    ReferenceRayBundle,
    solar_footprint_points,
    solar_footprint_segments,
    trace_reference_rays,
)

ASSET_DIR = Path(__file__).resolve().parent
SOLAR_TOTALITY_DURATION_REFERENCE_S = 268.1

def _resolved_grid_path(logical_name: str, packaged_name: str) -> Path:
    """Resolve immutable precomputed grids from SSAPy-Data with package fallback."""
    try:
        record = resolve_asset(logical_name, policy="data-first", required=False)
        if record is not None:
            return Path(record.path)
    except Exception:
        pass
    return ASSET_DIR / packaged_name

SOLAR_TOTALITY_DURATION_GRID_PATH = _resolved_grid_path(
    "solar_2024_totality_duration_grid", "solar_2024_totality_duration_grid.npz"
)
SOLAR_PARTIAL_VISIBILITY_GRID_PATH = _resolved_grid_path(
    "solar_2024_partial_visibility_grid", "solar_2024_partial_visibility_grid.npz"
)
LUNAR_GEOGRAPHIC_VISIBILITY_GRID_PATH = _resolved_grid_path(
    "lunar_2025_geographic_visibility_grid", "lunar_2025_geographic_visibility_grid.npz"
)

# Discrete, physically evaluated regions used by the scientific corridor map.
# Every location inside the corridor reaches complete photospheric coverage;
# these levels are percentages of the event's maximum C2-C3 duration, not
# percentages of obscuration.  Extra resolution near 100% makes the long,
# high-duration core visible instead of collapsing it into one broad color.
SOLAR_TOTALITY_PERCENT_BAND_EDGES = np.asarray(
    [0.0, 25.0, 50.0, 75.0, 90.0, 97.5, 100.0], dtype=float,
)
SOLAR_TOTALITY_PERCENT_BAND_COLORS = (
    "#06162b",  # 0-25% — shortest total phase near either corridor edge
    "#0a3f66",  # 25-50%
    "#0b7394",  # 50-75%
    "#18a8b1",  # 75-90%
    "#77d7bd",  # 90-97.5%
    "#ffe56f",  # 97.5-100% — longest totality near the central line
)

# Maximum photospheric-area obscuration outside the totality corridor.  These
# values answer the geographic question "who saw a partial eclipse, and how
# deep was it?".  They are deliberately different from NASA eclipse magnitude,
# which is a diameter fraction; magnitude contours are drawn separately.
SOLAR_PARTIAL_OBSCURATION_BAND_EDGES = np.asarray(
    [0.0, 20.0, 40.0, 60.0, 80.0, 99.999], dtype=float,
)
SOLAR_PARTIAL_OBSCURATION_BAND_COLORS = (
    "#10284e",
    "#174a73",
    "#1d6d8a",
    "#31949d",
    "#78bbae",
)

# Geographic lunar-eclipse visibility categories.  The upper lunar limb must
# be above the geometric WGS-84 horizon; atmospheric refraction is not added.
LUNAR_VISIBILITY_CATEGORY_COLORS = (
    "#05070c",   # no eclipse visible
    "#263b59",   # penumbral/partial phases only
    "#765a76",   # some totality visible
    "#ad755d",   # all totality visible, but not the complete P1-P4 event
    "#e0c894",   # complete P1-P4 eclipse visible
)
LUNAR_VISIBILITY_CATEGORY_LABELS = (
    "No eclipse visible",
    "Penumbral / partial phases only",
    "Part of totality visible",
    "All totality visible",
    "Entire P1-P4 eclipse visible",
)


def _srgb_to_linear(rgb):
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)


def _linear_to_srgb(rgb):
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1.0/2.4)-0.055)


@lru_cache(maxsize=8)
def _load_texture(name: str) -> np.ndarray:
    logical = {"earth": "earth_albedo", "moon": "moon_albedo"}.get(name)
    if logical is None:
        raise ValueError("name must be 'earth' or 'moon'")
    resolved = resolve_image(logical, policy="data-first", required=True)
    with Image.open(resolved.path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sample_equirectangular(texture: np.ndarray, lat_deg, lon_deg) -> np.ndarray:
    """Periodic bilinear sampling of a north-up -180..+180 texture."""
    tex = np.asarray(texture)
    h, w = tex.shape[:2]
    lat = np.clip(np.asarray(lat_deg, dtype=float), -90.0, 90.0)
    lon = (np.asarray(lon_deg, dtype=float)+180.0) % 360.0-180.0
    x = (lon+180.0)/360.0*w
    y = (90.0-lat)/180.0*(h-1)
    x0 = np.floor(x).astype(np.int64) % w
    x1 = (x0+1) % w
    y0 = np.floor(y).astype(np.int64)
    y1 = np.clip(y0+1, 0, h-1)
    fx = (x-np.floor(x))[..., None]
    fy = (y-np.floor(y))[..., None]
    c00 = tex[y0, x0].astype(float)
    c10 = tex[y0, x1].astype(float)
    c01 = tex[y1, x0].astype(float)
    c11 = tex[y1, x1].astype(float)
    top = c00*(1.0-fx)+c10*fx
    bottom = c01*(1.0-fx)+c11*fx
    return (top*(1.0-fy)+bottom*fy)/255.0


def _camera_basis(view_hat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = _unit(view_hat, name="view direction")
    north = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(north, view))) > 0.94:
        north = np.array([0.0, 1.0, 0.0])
    right = _unit(np.cross(north, view), name="camera right")
    up = _unit(np.cross(view, right), name="camera up")
    return view, right, up


def _disk_grid(size: int):
    """Return a camera-plane disk grid in raster row order.

    Image row zero is the top of the returned RGBA array.  Therefore the
    first Y value must point along +camera-up, not -camera-up.  The previous
    ascending Y grid mapped the top row to geographic/lunar south and flipped
    every Earth and Moon disk vertically even though both textures are
    north-up.
    """
    n = int(size)
    x = np.linspace(-1.075, 1.075, n)
    y = np.linspace(1.075, -1.075, n)  # top raster row is +camera-up / north
    X, Y = np.meshgrid(x, y)
    r2 = X*X+Y*Y
    mask = r2 <= 1.0
    Z = np.sqrt(np.clip(1.0-r2, 0.0, 1.0))
    return X, Y, Z, mask


def _geodetic_arrays(points):
    p = np.asarray(points, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.degrees(np.arctan2(y, x))
    rxy = np.hypot(x, y)
    e2 = 1.0-(RP_KM*RP_KM)/(RE_KM*RE_KM)
    lat = np.arctan2(z, rxy*(1.0-e2))
    for _ in range(5):
        sinlat = np.sin(lat)
        N = RE_KM/np.sqrt(1.0-e2*sinlat*sinlat)
        h = rxy/np.maximum(np.cos(lat), 1.0e-14)-N
        lat = np.arctan2(z, rxy*(1.0-e2*N/np.maximum(N+h, 1.0e-12)))
    return np.degrees(lat), (lon+180.0) % 360.0-180.0


def render_earth_disk(jd_utc: float, *, size: int = 420,
                      view_hat=None, supersample: int = 2) -> np.ndarray:
    """Render a high-fidelity orthographic WGS-84 Earth disk.

    The surface is obtained from exact orthographic ray/ellipsoid
    intersections rather than radially projecting a circular unit sphere.
    Rendering is supersampled and Lanczos-downsampled so the WGS-84 limb,
    finite-Sun shadow, coastlines, and atmosphere remain smooth in the compact
    scientific-summary panel.  The packaged cloud-free SSAPy surface texture
    is retained; no date-inaccurate synthetic cloud field is introduced.
    """
    output_size = max(32, int(size))
    ss = int(np.clip(int(supersample), 1, 4))
    work_size = output_size*ss

    state = solar_besselian_state(jd_utc)
    if view_hat is None:
        # Follow the instantaneous Sun direction in ITRF.  This keeps the
        # displayed hemisphere genuinely Sun-facing while Earth-fixed surface
        # features rotate beneath the camera over the multi-hour event.
        view_hat = state.sun_itrf_km
    view, right, up = _camera_basis(view_hat)
    X, Y, _, _ = _disk_grid(work_size)

    # Exact orthographic ray intersection with the oblate WGS-84 ellipsoid.
    # X/Y are expressed in equatorial-radius units in the camera plane.  The
    # projected silhouette is therefore allowed to be the correct slightly
    # flattened ellipse instead of being forced into a perfect circle.
    plane = (X[..., None]*right + Y[..., None]*up)*RE_KM
    inv_axes2 = 1.0/(EARTH_AXES_KM*EARTH_AXES_KM)
    aa = float(np.sum(view*view*inv_axes2))
    bb = 2.0*np.sum(plane*view*inv_axes2, axis=-1)
    cc = np.sum(plane*plane*inv_axes2, axis=-1)-1.0
    disc = bb*bb-4.0*aa*cc
    mask = disc >= 0.0
    root = np.sqrt(np.clip(disc, 0.0, None))
    # +root is the surface nearest a camera located at +view infinity.
    t_front = (-bb+root)/(2.0*aa)
    surface = plane+t_front[..., None]*view

    normal = surface*inv_axes2
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-15)
    lat, lon = _geodetic_arrays(surface)
    base_srgb = _sample_equirectangular(_load_texture("earth"), lat, lon)

    # Preserve the full-resolution texture's natural color separation after
    # it is projected into a small disk.  This is a restrained saturation and
    # local-contrast adjustment, not invented terrain or cloud geometry.
    tex_luma = np.sum(base_srgb*np.array([0.2126, 0.7152, 0.0722]), axis=-1)
    base_srgb = np.clip(
        tex_luma[..., None]+1.075*(base_srgb-tex_luma[..., None]), 0.0, 1.0,
    )
    base_linear = _srgb_to_linear(base_srgb)

    to_sun = state.sun_itrf_km-surface
    to_sun /= np.maximum(np.linalg.norm(to_sun, axis=-1, keepdims=True), 1.0e-15)
    mu0_signed = np.sum(normal*to_sun, axis=-1)
    mu0 = np.clip(mu0_signed, 0.0, 1.0)
    to_view = np.broadcast_to(view, surface.shape)
    mu_view = np.clip(np.sum(normal*to_view, axis=-1), 0.0, 1.0)

    to_moon = state.moon_itrf_km-surface
    dist_moon = np.linalg.norm(to_moon, axis=-1)
    dist_sun = np.linalg.norm(state.sun_itrf_km-surface, axis=-1)
    moon_hat = to_moon/np.maximum(dist_moon[..., None], 1.0e-12)
    sun_hat = (state.sun_itrf_km-surface)/np.maximum(dist_sun[..., None], 1.0e-12)
    separation = np.arccos(np.clip(np.sum(moon_hat*sun_hat, axis=-1), -1.0, 1.0))
    a_moon = np.arcsin(np.clip(SOLAR_UMBRA_OPTICAL_RADIUS_KM/dist_moon, 0.0, 1.0))
    a_sun = np.arcsin(np.clip(R_SUN_KM/dist_sun, 0.0, 1.0))
    visibility = angular_circle_visible_fraction(a_moon, a_sun, separation)

    # Linear-light direct illumination plus a small physically ordered sky
    # term.  Supersampling resolves the narrow 2024 umbra without a jagged or
    # blocky edge on the compact summary Earth.
    direct = np.clip(mu0, 0.0, 1.0)**0.96*visibility
    sky = 0.0030+0.0085*np.sqrt(mu0)*(0.20+0.80*np.sqrt(visibility))
    linear = base_linear*(sky[..., None]+1.10*direct[..., None])

    # View-dependent ocean glint inferred only from blue-dominant pixels in
    # the real packaged texture.  A narrow and broad lobe produce a natural
    # highlight without making continents reflective.
    r, g, b = base_srgb[..., 0], base_srgb[..., 1], base_srgb[..., 2]
    ocean = np.clip(8.0*(b-np.maximum(r, g)), 0.0, 1.0)*(b > 0.08)
    half = to_sun+to_view
    half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1.0e-12)
    ndoth = np.clip(np.sum(normal*half, axis=-1), 0.0, 1.0)
    glint = 0.94*ndoth**320+0.06*ndoth**110
    linear += (0.11*ocean*glint*visibility*mu0)[..., None]*np.array([0.46, 0.68, 1.0])

    # Camera-relative atmospheric scattering kept deliberately thin.  The
    # previous broad, saturated blue rim overwhelmed the high-resolution
    # surface texture and made the Earth read like a toy globe.  This term is
    # concentrated close to the true limb and remains subordinate to the
    # finite-disc surface illumination.
    limb = np.clip(1.0-mu_view, 0.0, 1.0)**4.35
    twilight = np.exp(-(mu0_signed/0.047)**2)
    atmosphere = limb[..., None]*(
        np.array([0.030, 0.145, 0.62])*(0.18+0.82*np.sqrt(mu0)[..., None])
        +np.array([1.0, 0.18, 0.030])*0.070*twilight[..., None]
    )
    linear += atmosphere*(0.13+0.87*np.sqrt(visibility))[..., None]

    # Highlight-preserving exposure and restrained micro-contrast.  No
    # synthetic terrain or cloud field is introduced: all geographic detail
    # comes from the packaged 5400 x 2700 LLNL/SSAPy surface mosaic.
    linear = np.clip(linear*1.10, 0.0, 1.35)
    linear = linear/(1.0+0.070*linear)
    srgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))

    rgba = np.zeros((*X.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(srgb*255.0).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    rgba[~mask, :3] = 0

    # An exterior atmosphere is evaluated from the exact projected-ellipsoid
    # quadratic.  ``projected_radius`` equals one on the WGS-84 silhouette,
    # so the halo follows the oblate limb rather than a generic circular blur.
    projected_q = np.sum(plane*plane*inv_axes2, axis=-1)-bb*bb/(4.0*aa)
    projected_radius = np.sqrt(np.clip(projected_q, 0.0, None))
    outside = (~mask) & (projected_radius <= 1.032)
    halo_profile = np.exp(-((projected_radius-1.0015)/0.0105)**2)
    sun_center_hat = state.sun_itrf_km/np.linalg.norm(state.sun_itrf_km)
    front_illumination = np.clip(float(np.dot(view, sun_center_hat)), 0.0, 1.0)
    halo_strength = 0.52+0.48*front_illumination
    halo_alpha = np.clip(42.0*halo_profile*halo_strength, 0.0, 42.0)
    halo_rgb = np.array([0.08, 0.29, 0.80])
    rgba[outside, :3] = np.rint(halo_rgb*255.0).astype(np.uint8)
    rgba[outside, 3] = np.rint(halo_alpha[outside]).astype(np.uint8)

    # Downsampling is the antialiasing stage for the limb, coastline detail,
    # and the roughly 200-km umbra.  A mild unsharp mask is applied only to RGB;
    # alpha remains the smooth Lanczos result so no bright outline is created.
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    image = Image.fromarray(rgba, mode="RGBA")
    if work_size != output_size:
        image = image.resize((output_size, output_size), resampling)
    red, green, blue, alpha = image.split()
    rgb_image = Image.merge("RGB", (red, green, blue)).filter(
        ImageFilter.UnsharpMask(radius=0.82, percent=92, threshold=2),
    )
    image = Image.merge("RGBA", (*rgb_image.split(), alpha))
    return np.asarray(image, dtype=np.uint8)

def _moon_body_axes(moon_center) -> np.ndarray:
    x_axis = _unit(-np.asarray(moon_center, dtype=float), name="Earth-facing lunar axis")
    north = np.array([0.0, 0.0, 1.0])
    z_axis = north-x_axis*np.dot(north, x_axis)
    if np.linalg.norm(z_axis) < 1.0e-10:
        north = np.array([0.0, 1.0, 0.0])
        z_axis = north-x_axis*np.dot(north, x_axis)
    z_axis = _unit(z_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def render_moon_disk(jd_utc: float, *, size: int = 460,
                     view_hat=None) -> np.ndarray:
    """Render the textured lunar disk with NASA Danjon shadow geometry."""
    state = lunar_reference_state(jd_utc)
    if view_hat is None:
        # Fixed Earth-based camera; the orientation remains stable throughout.
        peak = lunar_reference_state(LUNAR_2025.greatest_jd)
        view_hat = -peak.moon_gcrf_km
    view, right, up = _camera_basis(view_hat)
    X, Y, Z, mask = _disk_grid(size)
    normal = X[..., None]*right+Y[..., None]*up+Z[..., None]*view
    axes = _moon_body_axes(state.moon_gcrf_km)
    body = normal@axes
    lat = np.degrees(np.arcsin(np.clip(body[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(body[..., 1], body[..., 0]))
    base_srgb = _sample_equirectangular(_load_texture("moon"), lat, lon)
    base_linear = _srgb_to_linear(base_srgb)*1.22

    surface = state.moon_gcrf_km+LUNAR_OPTICAL_MOON_RADIUS_KM*normal
    to_sun = state.sun_gcrf_km-surface
    to_sun /= np.linalg.norm(to_sun, axis=-1, keepdims=True)
    mu0 = np.clip(np.sum(normal*to_sun, axis=-1), 0.0, 1.0)
    mu = np.clip(np.sum(normal*view, axis=-1), 0.0, 1.0)
    ls = np.where(mu0 > 0.0, 2.0*mu0/np.maximum(mu0+mu, 1.0e-6), 0.0)
    reflectance = np.clip(0.82*ls+0.18*mu0, 0.0, 1.2)

    to_earth = -surface
    dist_earth = np.linalg.norm(to_earth, axis=-1)
    dist_sun = np.linalg.norm(state.sun_gcrf_km-surface, axis=-1)
    earth_hat = to_earth/np.maximum(dist_earth[..., None], 1.0e-12)
    sun_hat = (state.sun_gcrf_km-surface)/np.maximum(dist_sun[..., None], 1.0e-12)
    separation = np.arccos(np.clip(np.sum(earth_hat*sun_hat, axis=-1), -1.0, 1.0))
    a_earth = np.arcsin(np.clip(LUNAR_DANJON_EARTH_RADIUS_KM/dist_earth, 0.0, 1.0))
    a_sun = np.arcsin(np.clip(R_SUN_KM/dist_sun, 0.0, 1.0))
    visibility = angular_circle_visible_fraction(a_earth, a_sun, separation)
    direct = reflectance*visibility
    linear = base_linear*(0.0045+0.9955*direct[..., None])

    blocked = np.clip(1.0-visibility, 0.0, 1.0)
    umbra_depth = np.clip((a_earth-a_sun-separation)/np.maximum(2.0*a_sun, 1.0e-12), 0.0, 1.0)
    refracted = blocked**1.22*(0.085+0.220*np.exp(-2.35*umbra_depth))
    linear += base_linear*refracted[..., None]*np.array([1.0, 0.155, 0.036])
    # A faint blue Earthshine floor keeps maria visible without inventing
    # direct sunlight on the blocked hemisphere.
    linear += base_linear*(0.0018+0.0035*(1.0-mu0))[..., None]*np.array([0.58, 0.72, 1.0])
    srgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    rgba = np.zeros((*X.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(srgb*255.0).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    rgba[~mask, :3] = 0
    return rgba


def solar_site_geometry(jd_utc: float, *, lat_deg: float | None = None,
                        lon_east_deg: float | None = None):
    """Compatibility wrapper around the validated WGS-84 observer model."""
    kwargs = {}
    if lat_deg is not None:
        kwargs["lat_deg"] = float(lat_deg)
    if lon_east_deg is not None:
        kwargs["lon_east_deg"] = float(lon_east_deg)
    return solar_observer_geometry(jd_utc, **kwargs)



@lru_cache(maxsize=8)
def _solar_granulation_field(size: int) -> np.ndarray:
    """Deterministic multiscale photospheric texture without periodic banding."""
    from scipy.ndimage import gaussian_filter

    n = int(size)
    rng = np.random.default_rng(20240408+n)
    raw = rng.standard_normal((n, n))
    fine = gaussian_filter(raw, sigma=max(0.65, n/520.0), mode="wrap")
    medium = gaussian_filter(raw, sigma=max(2.2, n/115.0), mode="wrap")
    coarse = gaussian_filter(raw, sigma=max(8.0, n/24.0), mode="wrap")

    def normalized(field):
        value = field-np.mean(field)
        return value/max(float(np.std(value)), 1.0e-12)

    # Fine granules dominate, with restrained supergranular variation.
    return 0.021*normalized(fine)+0.012*normalized(medium)+0.007*normalized(coarse)

def render_solar_apparent_disk(jd_utc: float, *, size: int = 380,
                               fixed_basis=None) -> tuple[np.ndarray, float]:
    """Observer view from NASA's greatest-eclipse site, including corona."""
    _, sun_hat, moon_hat, a_sun, a_moon, sep, visible = solar_site_geometry(jd_utc)
    # The apparent-image origin must follow the CURRENT Sun center.  The old
    # animation projected the Moon into a tangent plane fixed at maximum
    # eclipse, so ordinary diurnal motion displaced both Sun and Moon across
    # the raster but only the Moon was drawn.  That produced a large false
    # crescent during mathematically total frames.  Keep only the screen-up
    # convention from the fixed basis; rebuild its tangent plane about the
    # current Sun direction.
    view = _unit(sun_hat, name="current apparent Sun direction")
    if fixed_basis is None:
        _, right, up = _camera_basis(view)
    else:
        up_reference = np.asarray(fixed_basis[2], dtype=float)
        up_projected = up_reference-view*np.dot(up_reference, view)
        if np.linalg.norm(up_projected) < 1.0e-10:
            _, right, up = _camera_basis(view)
        else:
            up = _unit(up_projected, name="fixed apparent-image up")
            right = _unit(np.cross(up, view), name="fixed apparent-image right")
            up = _unit(np.cross(view, right), name="fixed apparent-image up")
    denom = max(float(np.dot(moon_hat, view)), 1.0e-12)
    off_x = math.atan2(float(np.dot(moon_hat, right)), denom)/a_sun
    off_y = math.atan2(float(np.dot(moon_hat, up)), denom)/a_sun
    moon_r = a_moon/a_sun
    extent = 2.1
    q = np.linspace(-extent, extent, int(size))
    X, Y = np.meshgrid(q, q)
    R = np.hypot(X, Y)
    sun_mask = R <= 1.0
    moon_mask = np.hypot(X-off_x, Y-off_y) <= moon_r
    z = np.sqrt(np.clip(1.0-R*R, 0.0, 1.0))
    limb = 0.47+0.53*z**0.72
    # Multiscale stochastic granulation avoids the woven/moiré artifact made
    # by the former intersecting sine waves.  The seed is fixed, so frames do
    # not shimmer as the eclipse advances.
    granulation = _solar_granulation_field(int(size))
    brightness = np.clip(limb*(1.0+granulation), 0.24, 1.08)
    rgb = np.zeros((*X.shape, 3), dtype=float)
    rgb[..., 0] = 1.00*brightness
    rgb[..., 1] = 0.79*brightness**1.03
    rgb[..., 2] = 0.39*brightness**1.12
    rgb[~sun_mask] = 0.0
    rgb[moon_mask] = np.array([0.0015, 0.0018, 0.0025])

    # Corona appears only when the photosphere is almost completely hidden.
    if visible < 0.035:
        rr = np.maximum(R, 1.0e-6)
        corona = np.exp(-3.4*np.clip(rr-0.96, 0.0, None))/(rr**0.75)
        corona *= (rr >= 0.94)
        angle = np.arctan2(Y, X)
        streamers = 0.58+0.42*(0.5+0.5*np.cos(4*angle+0.7*np.sin(3*angle)))
        corona *= streamers
        corona = np.clip(corona, 0.0, 1.35)
        outside = ~sun_mask
        rgb[outside] += corona[outside, None]*np.array([0.72, 0.80, 1.0])
        rgb[moon_mask] = np.array([0.0, 0.0, 0.0])
    alpha = np.clip(np.max(rgb, axis=-1)*1.6, 0.0, 1.0)
    alpha[sun_mask | moon_mask] = 1.0
    rgba = np.zeros((*X.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0)*255.0).astype(np.uint8)
    rgba[..., 3] = np.rint(alpha*255.0).astype(np.uint8)
    return rgba, visible


def _clip_polyline_x(points, xmin: float, xmax: float) -> np.ndarray:
    """Clip a monotone-in-x polyline to a local display window."""
    p = np.asarray(points, dtype=float)
    out: list[np.ndarray] = []
    for a, b in zip(p[:-1], p[1:]):
        x0, x1 = float(a[0]), float(b[0])
        if max(x0, x1) < xmin or min(x0, x1) > xmax:
            continue
        lo, hi = 0.0, 1.0
        dx = x1-x0
        if abs(dx) > 1.0e-15:
            t_a = (xmin-x0)/dx
            t_b = (xmax-x0)/dx
            lo = max(lo, min(t_a, t_b))
            hi = min(hi, max(t_a, t_b))
        if hi < lo:
            continue
        pa = a+lo*(b-a)
        pb = a+hi*(b-a)
        if not out or np.linalg.norm(out[-1]-pa) > 1.0e-8:
            out.append(pa)
        out.append(pb)
    return np.asarray(out, dtype=float)


def project_ray_bundle_2d(bundle: ReferenceRayBundle):
    occ = bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km
    target = bundle.earth_center_km if bundle.mode == "solar" else bundle.moon_center_km
    axis = bundle.axis_hat
    rel0 = bundle.umbra_tangent_points_km[0]-occ
    transverse = rel0-axis*np.dot(rel0, axis)
    u = _unit(transverse)
    def project(points):
        q = np.asarray(points)-occ
        return np.column_stack([q@axis, q@u])
    return project, float(np.dot(target-occ, axis)), u



def draw_ray_diagram(ax, bundle: ReferenceRayBundle):
    """Draw a physically scaled local cross-section of the finite-Sun rays.

    The Sun remains at its real off-frame distance.  Both axes are kilometres
    and use an equal data aspect, so body radii, ray slopes, and Earth-Moon
    separation are not visually rescaled.
    """
    import matplotlib.patches as patches

    project, target_x, _ = project_ray_bundle_2d(bundle)
    xmin = -0.055*target_x
    xmax = 1.055*target_x
    ymax = max(12_000.0, 1.62*max(
        RE_KM, bundle.target_radius_km, bundle.penumbra_optical_radius_km,
    ))

    def pair(paths):
        projected = [(project(path.points_km), path) for path in paths]
        values = [float(item[0][1, 1]) for item in projected]
        return projected[int(np.argmax(values))], projected[int(np.argmin(values))]

    # Physical penumbra and umbra/antumbra volumes.
    (p_hi, _), (p_lo, _) = pair(bundle.penumbra)
    p_hi = _clip_polyline_x(p_hi, xmin, xmax)
    p_lo = _clip_polyline_x(p_lo, xmin, xmax)
    if len(p_hi) and len(p_lo):
        poly = np.vstack([p_hi, p_lo[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color="#4f8fc7", alpha=0.11, zorder=1)

    (u_hi, _), (u_lo, _) = pair(bundle.umbra)
    u_hi = _clip_polyline_x(u_hi, xmin, xmax)
    u_lo = _clip_polyline_x(u_lo, xmin, xmax)
    if len(u_hi) and len(u_lo):
        poly = np.vstack([u_hi, u_lo[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color="#7d1624", alpha=0.27, zorder=2)

    for paths, color, width, label in (
        (bundle.penumbra, "#8dc7f2", 1.65, "Penumbral tangent"),
        (bundle.umbra, "#ff7766", 1.90, "Umbral / antumbral tangent"),
    ):
        for index, (points, _) in enumerate(pair(paths)):
            clipped = _clip_polyline_x(points, xmin, xmax)
            if len(clipped):
                ax.plot(clipped[:, 0], clipped[:, 1], color=color, lw=width,
                        label=label if index == 0 else None, zorder=5)

    central = _clip_polyline_x(project(bundle.central.points_km), xmin, xmax)
    if len(central):
        central_label = (
            "Axial sunlight — first opaque-surface stop"
            if bundle.mode == "solar"
            else "Axial sunlight — first opaque-surface stop"
        )
        ax.plot(central[:, 0], central[:, 1], color="#ffe36e", lw=2.9,
                label=central_label, zorder=7)

    if bundle.mode == "solar":
        occ_radius = bundle.solid_occluder_radius_km
        target_radius = RE_KM
        occ_color, target_color = "#9b9b9b", "#1f67a6"
        occ_label = "Moon\nmean solid radius"
        target_label = "Earth\nWGS-84 surface"
    else:
        occ_radius = RE_KM
        target_radius = bundle.target_radius_km
        occ_color, target_color = "#1f67a6", "#a6a39c"
        occ_label = "Earth\nsolid WGS-84"
        target_label = "Moon\napparent limb"

    ax.add_patch(patches.Circle((0.0, 0.0), occ_radius, facecolor=occ_color,
                                edgecolor="#f2f5f8", lw=0.9, zorder=8))
    ax.add_patch(patches.Circle((target_x, 0.0), target_radius, facecolor=target_color,
                                edgecolor="#f2f5f8", lw=0.9, zorder=8))
    if bundle.mode == "lunar":
        ax.add_patch(patches.Circle(
            (0.0, 0.0), bundle.umbra_optical_radius_km,
            facecolor="none", edgecolor="#7fafff", lw=1.1,
            linestyle=":", zorder=9,
        ))
        ax.plot([], [], color="#7fafff", lw=1.1, linestyle=":",
                label="Danjon effective atmospheric limb")

    # Keep body labels away from the x-axis caption and the scientific note.
    ax.annotate(occ_label, xy=(0.0, -occ_radius), xytext=(8, -7),
                textcoords="offset points", color="white", ha="left",
                va="top", fontsize=8.1, annotation_clip=False)
    ax.annotate(target_label, xy=(target_x, target_radius), xytext=(-8, 7),
                textcoords="offset points", color="white", ha="right",
                va="bottom", fontsize=8.1, annotation_clip=False)

    sun_distance = np.linalg.norm(bundle.sun_center_km-bundle.earth_center_km)
    axial_stop = ("Moon mean solid surface" if bundle.mode == "solar"
                  else "WGS-84 Earth surface")
    ax.text(
        0.012, 0.875,
        f"Sun off-frame: {sun_distance/1e6:.3f} million km  •  axial stop: {axial_stop}",
        transform=ax.transAxes, color="#ffdd70", fontsize=8.2,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#111725",
                  edgecolor="#4e5b70", alpha=0.92),
    )
    ax.text(
        0.50, 0.035,
        "Equal X/Y physical scale • all rays originate on the real photosphere • no segment passes through an opaque body",
        transform=ax.transAxes, color="#9fb0c5", fontsize=7.3,
        ha="center", va="bottom",
    )

    ax.axhline(0.0, color="#65758d", lw=0.65, alpha=0.50, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#02040a")
    ax.set_xlabel("Distance along eclipse axis [km]", color="#c9d3e3", fontsize=9)
    ax.set_ylabel("Transverse distance [km]", color="#c9d3e3", fontsize=9)
    ax.tick_params(colors="#97a7bb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#35445a")
    ax.grid(color="#2f3d52", alpha=0.32, lw=0.55)

    # A custom legend describes both the boundary rays and the filled optical
    # regions.  It lives in the title band, entirely outside the data area.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([], [], color="#ffe36e", lw=2.9,
               label="Axial sunlight (first-surface stop)"),
        Line2D([], [], color="#ff7766", lw=1.9,
               label="Umbral / antumbral tangent"),
        Line2D([], [], color="#8dc7f2", lw=1.65,
               label="Penumbral tangent"),
        Patch(facecolor="#7d1624", alpha=0.42, edgecolor="none",
              label="Umbra / antumbra volume"),
        Patch(facecolor="#4f8fc7", alpha=0.26, edgecolor="none",
              label="Penumbra volume"),
    ]
    if bundle.mode == "lunar":
        legend_handles.append(Line2D([], [], color="#7fafff", lw=1.1,
                                     linestyle=":",
                                     label="Danjon atmospheric limb"))
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=7.15,
              frameon=True, facecolor="#02040a", edgecolor="#46556c",
              framealpha=0.94, labelcolor="white", borderpad=0.42,
              handlelength=2.2, columnspacing=1.05)
    title = ("How the Moon's shadow reaches Earth" if bundle.mode == "solar"
             else "How Earth's shadow reaches the Moon")
    ax.set_title(f"{title}\nFinite-Sun geometry at true local scale",
                 color="white", fontsize=10.5, pad=39, fontweight="semibold")



def draw_true_distance_locator(ax, bundle: ReferenceRayBundle):
    """One-dimensional true-distance ruler from the Sun to the Earth-Moon system."""
    sun = bundle.sun_center_km
    earth = bundle.earth_center_km
    moon = bundle.moon_center_km
    axis = _unit((earth if bundle.mode == "solar" else moon)-sun)
    x_sun = 0.0
    x_earth = float(np.dot(earth-sun, axis))/1.0e6
    x_moon = float(np.dot(moon-sun, axis))/1.0e6
    xmax = max(x_earth, x_moon)+3.1

    ax.plot([0.0, xmax], [0.0, 0.0], color="#526078", lw=1.0)
    sun_r = R_SUN_KM/1.0e6
    theta = np.linspace(0.0, 2.0*np.pi, 160)
    ax.fill(x_sun+sun_r*np.cos(theta), sun_r*np.sin(theta),
            color="#ffb23b", alpha=0.98, zorder=3)
    ax.scatter([x_earth], [0.0], s=24, color="#5db2ff", zorder=5,
               edgecolor="#d7edff", linewidth=0.5)
    ax.scatter([x_moon], [0.0], s=14, color="#d9d9d9", zorder=6,
               edgecolor="white", linewidth=0.45)

    ax.annotate("Sun — actual radius", (x_sun, 0), xytext=(10, 12),
                textcoords="offset points", color="#ffd47a", fontsize=8.2)
    ax.annotate("Earth locator", (x_earth, 0), xytext=(-7, 13),
                textcoords="offset points", color="#8bc7ff", fontsize=8.0,
                ha="right")
    ax.annotate("Moon locator", (x_moon, 0), xytext=(-7, -17),
                textcoords="offset points", color="#eeeeee", fontsize=8.0,
                ha="right")

    order = "Sun → Moon → Earth" if bundle.mode == "solar" else "Sun → Earth → Moon"
    earth_moon = np.linalg.norm(moon-earth)
    ax.text(0.50, 0.96,
            f"{order}   •   Sun–Earth {np.linalg.norm(sun-earth)/1e6:.3f} million km"
            f"   •   Earth–Moon {earth_moon:,.0f} km",
            transform=ax.transAxes, color="#d7e0ec", fontsize=8.3,
            ha="center", va="top")
    ax.text(0.50, 0.10,
            "Earth and Moon are sub-pixel at this scale; their symbols are locators, not enlarged bodies",
            transform=ax.transAxes, color="#8898ad", fontsize=7.4,
            ha="center", va="bottom")

    ax.set_xlim(-1.2, xmax+1.8)
    ax.set_ylim(-1.25, 1.25)
    ax.set_facecolor("#010207")
    ax.set_yticks([])
    ax.set_xlabel("Distance from the Sun [million km]", color="#9eacc0", fontsize=8,
                  labelpad=2)
    ax.tick_params(axis="x", colors="#8796aa", labelsize=7, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _split_longitudes(lon, lat):
    """Split map polylines at antimeridian and polar projection jumps.

    Near a pole, a physically short geodesic step can span tens of degrees of
    longitude in an equirectangular image.  Drawing that as one straight map
    segment creates the long horizontal streak that made the interactive
    footprint look scrambled.  The underlying 3-D geometry remains continuous;
    only the 2-D map polyline is broken at the projection singularity.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    segments = []
    start = 0
    dlon = np.abs(np.diff(lon))
    polar = ((np.maximum(np.abs(lat[:-1]), np.abs(lat[1:])) > 75.0)
             & (dlon > 25.0))
    jumps = np.where((dlon > 150.0) | polar)[0]
    for stop in np.r_[jumps+1, len(lon)]:
        if stop-start >= 2:
            segments.append((lon[start:stop], lat[start:stop]))
        start = stop
    return segments


@lru_cache(maxsize=1)
def solar_central_path_cached():
    start = SOLAR_2024.contacts_jd["U1"]-10.0/1440.0
    stop = SOLAR_2024.contacts_jd["U4"]+10.0/1440.0
    times = np.arange(start, stop+0.5/1440.0, 1.0/1440.0)
    lat, lon = [], []
    for jd in times:
        point = solar_central_line_wgs84(float(jd))
        if point is not None:
            lat.append(point[0]); lon.append(point[1])
    return np.asarray(lon), np.asarray(lat)



@lru_cache(maxsize=1)
def solar_umbra_corridor_cached():
    """Return physical left/right limits of the 2024 totality corridor.

    The corridor is sampled from the same finite-Sun tangent-ray footprints
    used by the 3-D renderer.  At each central-line sample, the two edge
    points are chosen along the local cross-track direction on WGS-84.
    """
    start = SOLAR_2024.contacts_jd["U1"]
    stop = SOLAR_2024.contacts_jd["U4"]
    times = np.arange(start, stop+2.0/1440.0, 2.0/1440.0)
    out = {name: [] for name in (
        "jd", "center_lon", "center_lat",
        "left_lon", "left_lat", "right_lon", "right_lat",
        "left_xyz", "right_xyz",
    )}
    delta_jd = 20.0/86400.0
    axes2 = np.asarray(EARTH_AXES_KM, dtype=float)**2
    for jd in times:
        center = solar_central_line_wgs84(float(jd))
        footprint = solar_footprint_points(float(jd), family="umbra", n_azimuth=144)
        before = solar_central_line_wgs84(float(jd-delta_jd))
        after = solar_central_line_wgs84(float(jd+delta_jd))
        if center is None or before is None or after is None or len(footprint) < 8:
            continue
        center_xyz = np.asarray(center[2], dtype=float)
        normal = center_xyz/axes2
        normal = _unit(normal, name="WGS-84 surface normal")
        along = np.asarray(after[2], dtype=float)-np.asarray(before[2], dtype=float)
        along = along-normal*np.dot(along, normal)
        if np.linalg.norm(along) < 1.0e-9:
            continue
        along = _unit(along, name="eclipse along-track direction")
        cross_track = _unit(np.cross(normal, along), name="eclipse cross-track direction")
        points = np.asarray(footprint, dtype=float)
        scores = (points-center_xyz[None, :])@cross_track
        left_xyz = points[int(np.argmax(scores))]
        right_xyz = points[int(np.argmin(scores))]
        left = geodetic_from_itrf(left_xyz)
        right = geodetic_from_itrf(right_xyz)
        out["jd"].append(float(jd))
        out["center_lon"].append(float(center[1])); out["center_lat"].append(float(center[0]))
        out["left_lon"].append(float(left[1])); out["left_lat"].append(float(left[0]))
        out["right_lon"].append(float(right[1])); out["right_lat"].append(float(right[0]))
        out["left_xyz"].append(left_xyz); out["right_xyz"].append(right_xyz)
    result = {}
    for key, values in out.items():
        result[key] = np.asarray(values, dtype=float)
    return result


def _solar_internal_contact_margin(jd_utc: float, lat_deg: float,
                                   lon_east_deg: float) -> float:
    """Signed C2/C3 contact margin for one WGS-84 observer.

    Negative values mean that the adopted k2 lunar limb fully covers the
    photosphere.  The same Besselian reconstruction and topocentric angular
    geometry used for the greatest-site contacts are used here.
    """
    *_, a_sun, a_moon, separation, _ = solar_observer_geometry(
        float(jd_utc), lat_deg=float(lat_deg), lon_east_deg=float(lon_east_deg),
        moon_radius_km=SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    )
    return float(separation-abs(a_moon-a_sun))


def _bisect_totality_contact(lat_deg: float, lon_east_deg: float,
                             left: float, right: float,
                             *, iterations: int = 48) -> float:
    f_left = _solar_internal_contact_margin(left, lat_deg, lon_east_deg)
    f_right = _solar_internal_contact_margin(right, lat_deg, lon_east_deg)
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left*f_right > 0.0:
        raise ValueError("totality contact is not bracketed")
    a, b = float(left), float(right)
    for _ in range(int(iterations)):
        mid = 0.5*(a+b)
        f_mid = _solar_internal_contact_margin(mid, lat_deg, lon_east_deg)
        if f_left*f_mid <= 0.0:
            b, f_right = mid, f_mid
        else:
            a, f_left = mid, f_mid
    return 0.5*(a+b)


def solar_totality_duration_seconds(lat_deg: float, lon_east_deg: float,
                                     near_jd: float) -> float:
    """Return exact local C2-C3 duration near a corridor sample.

    The observer is fixed on WGS-84.  Roots are found against the adopted
    NASA k2 lunar limb and the finite solar photosphere.  ``near_jd`` is the
    central-line time adjacent to the site and is used only to bracket roots.
    """
    jd0 = float(near_jd)
    f0 = _solar_internal_contact_margin(jd0, lat_deg, lon_east_deg)
    if f0 >= -2.0e-13:
        return 0.0
    step = 5.0/86400.0
    max_steps = 180  # 15 minutes on each side; far wider than this event needs.

    inside = jd0
    c2 = None
    for k in range(1, max_steps+1):
        outside = jd0-k*step
        if _solar_internal_contact_margin(outside, lat_deg, lon_east_deg) >= 0.0:
            c2 = _bisect_totality_contact(lat_deg, lon_east_deg, outside, inside)
            break
        inside = outside

    inside = jd0
    c3 = None
    for k in range(1, max_steps+1):
        outside = jd0+k*step
        if _solar_internal_contact_margin(outside, lat_deg, lon_east_deg) >= 0.0:
            c3 = _bisect_totality_contact(lat_deg, lon_east_deg, inside, outside)
            break
        inside = outside

    if c2 is None or c3 is None or c3 <= c2:
        return 0.0
    return float((c3-c2)*86400.0)


def _build_solar_totality_duration_grid(*, n_cross: int = 13) -> dict[str, np.ndarray]:
    """Build a WGS-84 corridor mesh shaded by local totality duration.

    Values are exact C2-C3 durations at fixed observers, normalized to the
    published 268.1-second event maximum.  The cross-track mesh follows the
    finite-Sun umbral corridor; it is not a generic decorative gradient.
    """
    corridor = solar_umbra_corridor_cached()
    fractions = np.linspace(-1.0, 1.0, max(5, int(n_cross)))
    n_along = len(corridor["jd"])
    lon_grid = np.empty((n_along, len(fractions)), dtype=float)
    lat_grid = np.empty_like(lon_grid)
    duration_s = np.zeros_like(lon_grid)

    axes = np.asarray(EARTH_AXES_KM, dtype=float)
    for i, jd in enumerate(corridor["jd"]):
        center_xyz = itrf_surface_point(
            corridor["center_lat"][i], corridor["center_lon"][i], 0.0,
        )
        for j, fraction in enumerate(fractions):
            if fraction < 0.0:
                weight = -fraction
                edge_xyz = np.asarray(corridor["right_xyz"][i], dtype=float)
            else:
                weight = fraction
                edge_xyz = np.asarray(corridor["left_xyz"][i], dtype=float)
            point = (1.0-weight)*center_xyz+weight*edge_xyz
            # Project the short cross-track chord back onto WGS-84 before
            # evaluating the fixed observer.  This avoids treating degrees of
            # latitude/longitude as a Euclidean plane.
            point *= 1.0/math.sqrt(float(np.sum((point/axes)**2)))
            lat, lon, _ = geodetic_from_itrf(point)
            lat_grid[i, j] = lat
            lon_grid[i, j] = lon
            if abs(fraction) < 1.0-1.0e-12:
                duration_s[i, j] = solar_totality_duration_seconds(lat, lon, float(jd))

    percent = np.clip(100.0*duration_s/SOLAR_TOTALITY_DURATION_REFERENCE_S,
                      0.0, 100.0)
    return {
        "jd": corridor["jd"],
        "cross_fraction": fractions,
        "lon": lon_grid,
        "lat": lat_grid,
        "duration_s": duration_s,
        "percent_of_max": percent,
        "reference_duration_s": np.asarray(SOLAR_TOTALITY_DURATION_REFERENCE_S),
    }


@lru_cache(maxsize=1)
def solar_totality_duration_grid_cached() -> dict[str, np.ndarray]:
    """Load the packaged duration field, rebuilding it only when absent."""
    path = SOLAR_TOTALITY_DURATION_GRID_PATH
    if path.is_file():
        with np.load(path) as payload:
            result = {key: np.asarray(payload[key]) for key in payload.files}
    else:
        result = _build_solar_totality_duration_grid()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **result)
    # Backward-compatible descriptive aliases used by dashboard consumers.
    result.setdefault("longitude", result["lon"])
    result.setdefault("latitude", result["lat"])
    result.setdefault("duration_percent", result["percent_of_max"])
    return result


@lru_cache(maxsize=16)
def solar_totality_band_polygons(threshold_percent: float) -> tuple[np.ndarray, ...]:
    """Return map polygons enclosing a relative-duration threshold.

    Boundaries are interpolated directly within the exact C2-C3 duration mesh,
    then joined along the eclipse path.  This avoids raster contour artifacts in
    the interactive dashboard and keeps every band inside the physical corridor.
    """
    threshold = float(threshold_percent)
    if not 0.0 < threshold < 100.0:
        raise ValueError("threshold_percent must lie strictly between 0 and 100")
    field = solar_totality_duration_grid_cached()
    fractions = np.asarray(field["cross_fraction"], dtype=float)
    percent = np.asarray(field["percent_of_max"], dtype=float)
    lon = np.asarray(field["lon"], dtype=float)
    lat = np.asarray(field["lat"], dtype=float)

    left_points: list[np.ndarray | None] = []
    right_points: list[np.ndarray | None] = []
    for row_percent, row_lon, row_lat in zip(percent, lon, lat):
        valid = np.flatnonzero(row_percent >= threshold)
        if not valid.size:
            left_points.append(None); right_points.append(None)
            continue
        first, last = int(valid[0]), int(valid[-1])

        def interpolate_boundary(inner: int, outer: int) -> np.ndarray:
            p0, p1 = float(row_percent[outer]), float(row_percent[inner])
            if abs(p1-p0) < 1.0e-12:
                weight = 1.0
            else:
                weight = np.clip((threshold-p0)/(p1-p0), 0.0, 1.0)
            lo = row_lon[outer]+weight*(row_lon[inner]-row_lon[outer])
            la = row_lat[outer]+weight*(row_lat[inner]-row_lat[outer])
            return np.array([lo, la], dtype=float)

        left = (np.array([row_lon[first], row_lat[first]], dtype=float)
                if first == 0 else interpolate_boundary(first, first-1))
        right = (np.array([row_lon[last], row_lat[last]], dtype=float)
                 if last == len(fractions)-1 else interpolate_boundary(last, last+1))
        left_points.append(left); right_points.append(right)

    polygons: list[np.ndarray] = []
    start = None
    for index, point in enumerate(left_points+[None]):
        if point is not None and start is None:
            start = index
        elif point is None and start is not None:
            stop = index
            if stop-start >= 2:
                left = np.vstack(left_points[start:stop])
                right = np.vstack(right_points[start:stop])
                polygon = np.vstack([left, right[::-1], left[:1]])
                polygons.append(polygon)
            start = None
    return tuple(polygons)


def solar_totality_duration_raster_cached(nx: int = 270, ny: int = 152):
    """Interpolate the physical duration mesh to a regular map raster."""
    field = solar_totality_duration_grid_cached()
    x = np.linspace(-160.0, -25.0, int(nx))
    y = np.linspace(-8.0, 68.0, int(ny))
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([field["lon"].ravel(), field["lat"].ravel()])
    values = field["percent_of_max"].ravel()
    try:
        from scipy.interpolate import griddata
        Z = griddata(points, values, (X, Y), method="linear")
    except Exception:
        # Nearest-neighbour fallback keeps the dashboard usable without SciPy.
        Z = np.full_like(X, np.nan, dtype=float)
        for row in range(Y.shape[0]):
            d2 = ((points[:, 0][None, :]-X[row, :, None])**2
                  +(points[:, 1][None, :]-Y[row, :, None])**2)
            nearest = np.argmin(d2, axis=1)
            Z[row] = values[nearest]

    # Linear interpolation fills the convex hull; mask back to the actual
    # finite-Sun corridor polygon so the color field never spills outside it.
    corridor = solar_umbra_corridor_cached()
    polygon = np.column_stack([
        np.concatenate([corridor["left_lon"], corridor["right_lon"][::-1]]),
        np.concatenate([corridor["left_lat"], corridor["right_lat"][::-1]]),
    ])
    try:
        from matplotlib.path import Path as MplPath
        inside = MplPath(polygon).contains_points(
            np.column_stack([X.ravel(), Y.ravel()]), radius=1.0e-9,
        ).reshape(X.shape)
        Z = np.where(inside, Z, np.nan)
    except Exception:
        pass
    return x, y, Z



def _wgs84_observer_grid(step_deg: float = 0.75):
    """Return a regular geodetic WGS-84 observer grid and surface normals."""
    step = float(step_deg)
    if not (0.1 <= step <= 5.0):
        raise ValueError("step_deg must be between 0.1 and 5 degrees")
    longitude = np.arange(-180.0, 180.0 + 0.25*step, step, dtype=float)
    latitude = np.arange(-90.0 + step, 90.0, step, dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    e2 = 1.0-(RP_KM*RP_KM)/(RE_KM*RE_KM)
    N = RE_KM/np.sqrt(1.0-e2*np.sin(lat_rad)**2)
    observers = np.stack([
        N*np.cos(lat_rad)*np.cos(lon_rad),
        N*np.cos(lat_rad)*np.sin(lon_rad),
        N*(1.0-e2)*np.sin(lat_rad),
    ], axis=-1)
    normals = observers/np.asarray([RE_KM**2, RE_KM**2, RP_KM**2])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return longitude, latitude, observers, normals


def _build_solar_partial_visibility_grid(
    *, step_deg: float = 0.75, time_step_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """Evaluate the complete partial-eclipse visibility region on WGS-84.

    At each fixed observer, the finite apparent Sun and NASA-k1 lunar disks are
    sampled from global P1 through P4.  The stored obscuration is the maximum
    *photospheric area* covered while any part of the solar disk is above the
    geometric horizon.  ``max_magnitude`` is NASA's diameter-based quantity and
    is retained for 0.2/0.4/0.6/0.8 reference contours.
    """
    longitude, latitude, observers_grid, normals_grid = _wgs84_observer_grid(step_deg)
    shape = observers_grid.shape[:2]
    observers = observers_grid.reshape(-1, 3)
    normals = normals_grid.reshape(-1, 3)
    start = SOLAR_2024.contacts_jd["P1"]
    stop = SOLAR_2024.contacts_jd["P4"]
    step_jd = float(time_step_s)/86400.0
    times = np.arange(start, stop+0.25*step_jd, step_jd, dtype=float)
    times = np.sort(np.unique(np.concatenate([
        times, np.asarray(list(SOLAR_2024.contacts_jd.values()), dtype=float),
    ])))

    max_obscuration = np.zeros(len(observers), dtype=float)
    max_magnitude = np.zeros(len(observers), dtype=float)
    max_jd = np.full(len(observers), np.nan, dtype=float)
    sun_altitude_at_max_deg = np.full(len(observers), np.nan, dtype=float)

    for jd in times:
        state = solar_besselian_state(float(jd))
        to_sun = state.sun_itrf_km[None, :]-observers
        to_moon = state.moon_itrf_km[None, :]-observers
        sun_distance = np.linalg.norm(to_sun, axis=1)
        moon_distance = np.linalg.norm(to_moon, axis=1)
        sun_hat = to_sun/sun_distance[:, None]
        moon_hat = to_moon/moon_distance[:, None]
        sun_radius = np.arcsin(np.clip(R_SUN_KM/sun_distance, 0.0, 1.0))
        moon_radius = np.arcsin(np.clip(
            SOLAR_PENUMBRA_OPTICAL_RADIUS_KM/moon_distance, 0.0, 1.0,
        ))
        separation = np.arccos(np.clip(
            np.sum(sun_hat*moon_hat, axis=1), -1.0, 1.0,
        ))
        obscuration = 1.0-angular_circle_visible_fraction(
            moon_radius, sun_radius, separation,
        )
        magnitude = (sun_radius+moon_radius-separation)/(2.0*sun_radius)
        altitude = np.arcsin(np.clip(np.sum(sun_hat*normals, axis=1), -1.0, 1.0))
        # A location counts as seeing the eclipse when at least part of the
        # solar photosphere is above the ideal geometric horizon.  Refraction
        # is intentionally excluded and stated in every key/caption.
        above_horizon = altitude >= -sun_radius
        obscuration = np.where(above_horizon, obscuration, 0.0)
        magnitude = np.where(above_horizon, np.clip(magnitude, 0.0, None), 0.0)
        update = obscuration > max_obscuration
        max_obscuration[update] = obscuration[update]
        max_jd[update] = float(jd)
        sun_altitude_at_max_deg[update] = np.degrees(altitude[update])
        max_magnitude = np.maximum(max_magnitude, magnitude)

    return {
        "longitude": longitude,
        "latitude": latitude,
        "max_obscuration_percent": (100.0*max_obscuration).reshape(shape),
        "max_magnitude": max_magnitude.reshape(shape),
        "max_jd": max_jd.reshape(shape),
        "sun_altitude_at_max_deg": sun_altitude_at_max_deg.reshape(shape),
        "grid_step_deg": np.asarray(float(step_deg)),
        "time_step_s": np.asarray(float(time_step_s)),
    }


@lru_cache(maxsize=1)
def solar_partial_visibility_grid_cached() -> dict[str, np.ndarray]:
    """Load or build the global 2024 partial-eclipse visibility field."""
    path = SOLAR_PARTIAL_VISIBILITY_GRID_PATH
    if path.is_file():
        with np.load(path) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    result = _build_solar_partial_visibility_grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def solar_maximum_local_eclipse(
    lat_deg: float, lon_east_deg: float, *, time_step_s: float = 20.0,
) -> dict[str, float]:
    """Maximum visible solar obscuration and magnitude at one WGS-84 site."""
    observer = itrf_surface_point(float(lat_deg), float(lon_east_deg), 0.0)
    normal = observer/np.asarray([RE_KM**2, RE_KM**2, RP_KM**2])
    normal = _unit(normal, name="observer surface normal")
    start = SOLAR_2024.contacts_jd["P1"]
    stop = SOLAR_2024.contacts_jd["P4"]
    step_jd = float(time_step_s)/86400.0
    times = np.arange(start, stop+0.25*step_jd, step_jd, dtype=float)
    times = np.sort(np.unique(np.concatenate([
        times, np.asarray(list(SOLAR_2024.contacts_jd.values()), dtype=float),
    ])))
    best = {
        "max_obscuration_percent": 0.0,
        "max_magnitude": 0.0,
        "max_jd": float("nan"),
        "max_magnitude_jd": float("nan"),
        "sun_altitude_deg": float("nan"),
    }
    for jd in times:
        state = solar_besselian_state(float(jd))
        to_sun = state.sun_itrf_km-observer
        to_moon = state.moon_itrf_km-observer
        ds = float(np.linalg.norm(to_sun))
        dm = float(np.linalg.norm(to_moon))
        sun_hat = to_sun/ds
        moon_hat = to_moon/dm
        a_sun = math.asin(np.clip(R_SUN_KM/ds, 0.0, 1.0))
        a_moon = math.asin(np.clip(SOLAR_PENUMBRA_OPTICAL_RADIUS_KM/dm, 0.0, 1.0))
        separation = math.acos(np.clip(np.dot(sun_hat, moon_hat), -1.0, 1.0))
        altitude = math.asin(np.clip(np.dot(sun_hat, normal), -1.0, 1.0))
        if altitude < -a_sun:
            continue
        visible = float(angular_circle_visible_fraction(a_moon, a_sun, separation))
        obscuration = 100.0*(1.0-visible)
        magnitude = max(0.0, (a_sun+a_moon-separation)/(2.0*a_sun))
        if magnitude > best["max_magnitude"]:
            best["max_magnitude"] = float(magnitude)
            best["max_magnitude_jd"] = float(jd)
        if obscuration > best["max_obscuration_percent"]:
            best.update(
                max_obscuration_percent=float(obscuration),
                max_jd=float(jd),
                sun_altitude_deg=math.degrees(altitude),
            )
    return best


def _build_lunar_geographic_visibility_grid(
    *, step_deg: float = 1.0, time_step_s: float = 120.0,
) -> dict[str, np.ndarray]:
    """Classify where the 2025 lunar eclipse was above the WGS-84 horizon."""
    longitude, latitude, observers_grid, normals_grid = _wgs84_observer_grid(step_deg)
    shape = observers_grid.shape[:2]
    observers = observers_grid.reshape(-1, 3)
    normals = normals_grid.reshape(-1, 3)
    start = LUNAR_2025.contacts_jd["P1"]
    stop = LUNAR_2025.contacts_jd["P4"]
    step_jd = float(time_step_s)/86400.0
    times = np.arange(start, stop+0.25*step_jd, step_jd, dtype=float)
    times = np.sort(np.unique(np.concatenate([
        times, np.asarray(list(LUNAR_2025.contacts_jd.values()), dtype=float),
    ])))

    any_event = np.zeros(len(observers), dtype=bool)
    all_event = np.ones(len(observers), dtype=bool)
    any_totality = np.zeros(len(observers), dtype=bool)
    all_totality = np.ones(len(observers), dtype=bool)
    visible_samples = np.zeros(len(observers), dtype=np.int32)
    totality_samples = 0
    contact_upper_limb_altitude: dict[str, np.ndarray] = {}

    contact_lookup = {float(value): name for name, value in LUNAR_2025.contacts_jd.items()}
    for jd in times:
        state = lunar_reference_state(float(jd))
        moon_itrf = gcrf_to_itrf_km(
            state.moon_gcrf_km, float(jd), prefer_toolkit=False,
        )
        to_moon = moon_itrf[None, :]-observers
        distance = np.linalg.norm(to_moon, axis=1)
        moon_hat = to_moon/distance[:, None]
        altitude = np.arcsin(np.clip(np.sum(moon_hat*normals, axis=1), -1.0, 1.0))
        angular_radius = np.arcsin(np.clip(R_MOON_MEAN_KM/distance, 0.0, 1.0))
        visible = altitude >= -angular_radius
        any_event |= visible
        all_event &= visible
        visible_samples += visible.astype(np.int32)
        if LUNAR_2025.contacts_jd["U2"] <= jd <= LUNAR_2025.contacts_jd["U3"]:
            any_totality |= visible
            all_totality &= visible
            totality_samples += 1
        name = contact_lookup.get(float(jd))
        if name is not None:
            contact_upper_limb_altitude[name] = np.degrees(
                altitude+angular_radius,
            ).reshape(shape)

    category = np.zeros(len(observers), dtype=np.uint8)
    category[any_event] = 1
    category[any_totality] = 2
    category[all_totality] = 3
    category[all_event] = 4
    result: dict[str, np.ndarray] = {
        "longitude": longitude,
        "latitude": latitude,
        "category": category.reshape(shape),
        "event_visible_fraction": (visible_samples/float(len(times))).reshape(shape),
        "grid_step_deg": np.asarray(float(step_deg)),
        "time_step_s": np.asarray(float(time_step_s)),
        "totality_sample_count": np.asarray(int(totality_samples)),
    }
    for name, altitude in contact_upper_limb_altitude.items():
        result[f"upper_limb_altitude_{name}_deg"] = altitude
    return result


@lru_cache(maxsize=1)
def lunar_geographic_visibility_grid_cached() -> dict[str, np.ndarray]:
    """Load or build the 2025 lunar-eclipse geographic visibility field."""
    path = LUNAR_GEOGRAPHIC_VISIBILITY_GRID_PATH
    if path.is_file():
        with np.load(path) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    result = _build_lunar_geographic_visibility_grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def _draw_map_background(ax, *, alpha: float = 0.68):
    earth = _load_texture("earth")
    ax.imshow(
        earth, extent=(-180, 180, -90, 90), origin="upper", alpha=float(alpha),
        interpolation="lanczos", resample=True, zorder=0,
    )
    veil = np.zeros((2, 2, 4), dtype=float)
    veil[..., :3] = np.array([0.010, 0.022, 0.042])
    veil[..., 3] = 0.18
    ax.imshow(veil, extent=(-180, 180, -90, 90), origin="upper", zorder=0.5)


def draw_solar_partial_visibility_map(
    ax, *, map_extent: tuple[float, float, float, float] = (-180.0, 30.0, -10.0, 85.0),
    show_magnitude_contours: bool = True,
):
    """Draw the complete region that saw any part of the 2024 solar eclipse."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    xmin, xmax, ymin, ymax = map(float, map_extent)
    _draw_map_background(ax, alpha=0.64)
    field = solar_partial_visibility_grid_cached()
    longitude = np.asarray(field["longitude"], dtype=float)
    latitude = np.asarray(field["latitude"], dtype=float)
    obscuration = np.asarray(field["max_obscuration_percent"], dtype=float)
    magnitude = np.asarray(field["max_magnitude"], dtype=float)
    partial = np.where((obscuration > 0.0) & (obscuration < 99.999), obscuration, np.nan)
    colors = tuple(SOLAR_PARTIAL_OBSCURATION_BAND_COLORS)
    edges = np.asarray(SOLAR_PARTIAL_OBSCURATION_BAND_EDGES, dtype=float)
    ax.pcolormesh(
        longitude, latitude, partial,
        cmap=ListedColormap(colors, name="solar_partial_obscuration"),
        norm=BoundaryNorm(edges, len(colors), clip=True),
        shading="nearest", alpha=0.88, zorder=2.0, rasterized=True,
    )
    # Any-eclipse limit.  A small nonzero magnitude is used instead of exactly
    # zero so contour interpolation does not wander through the large flat
    # no-eclipse domain.
    try:
        ax.contour(
            longitude, latitude, magnitude, levels=[1.0e-5],
            colors=["#a9d8ff"], linewidths=1.15, linestyles="-", zorder=3.0,
        )
    except Exception:
        pass
    if show_magnitude_contours:
        contours = ax.contour(
            longitude, latitude, magnitude,
            levels=[0.2, 0.4, 0.6, 0.8], colors=["#e4f5ff"],
            linewidths=0.75, alpha=0.94, zorder=3.3,
        )
        ax.clabel(
            contours, fmt=lambda value: f"mag {value:.1f}", inline=True,
            inline_spacing=2, fontsize=5.8, colors="#e9f7ff",
        )

    corridor = solar_umbra_corridor_cached()
    if len(corridor["jd"]):
        for longitude_edge, latitude_edge in (
            (corridor["left_lon"], corridor["left_lat"]),
            (corridor["right_lon"], corridor["right_lat"]),
        ):
            ax.plot(
                longitude_edge, latitude_edge, color="#ffe16f", lw=1.3,
                alpha=0.99, zorder=5.0,
            )
    central_lon, central_lat = solar_central_path_cached()
    for seg_lon, seg_lat in _split_longitudes(central_lon, central_lat):
        ax.plot(seg_lon, seg_lat, color="#06101a", lw=3.3, zorder=5.1)
        ax.plot(seg_lon, seg_lat, color="white", lw=1.15, zorder=5.2)
    ax.scatter(
        [SOLAR_GREATEST_SITE_LON_EAST_DEG], [SOLAR_GREATEST_SITE_LAT_DEG],
        marker="D", s=25, facecolor="#05070d", edgecolor="white",
        linewidth=0.75, zorder=6.0,
    )

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    x_step = 30.0 if xmax-xmin > 120.0 else 20.0
    y_step = 15.0 if ymax-ymin > 70.0 else 10.0
    ax.set_xticks(np.arange(math.ceil(xmin/x_step)*x_step, xmax+0.1, x_step))
    ax.set_yticks(np.arange(math.ceil(ymin/y_step)*y_step, ymax+0.1, y_step))
    ax.set_xlabel("Longitude [deg east]", color="#b5c2d3", fontsize=7.2)
    ax.set_ylabel("Latitude [deg]", color="#b5c2d3", fontsize=7.2)
    ax.tick_params(colors="#b2bfd0", labelsize=6.4, length=2.2)
    ax.grid(color="#e4edf7", alpha=0.18, lw=0.48)
    for spine in ax.spines.values():
        spine.set_color("#52637a")
    ax.set_title(
        "Where the partial eclipse was visible\nmaximum solar-photosphere area covered",
        color="white", fontsize=8.4, pad=4, fontweight="semibold",
    )


def draw_solar_combined_visibility_key(ax):
    """Draw a compact, non-overlapping key for partial and total regions."""
    from matplotlib.patches import Rectangle

    ax.set_facecolor("#06101b")
    ax.set_xlim(0.0, 12.0); ax.set_ylim(0.0, 2.0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#52637a"); spine.set_linewidth(0.7)
    ax.text(0.18, 1.80, "PARTIAL ECLIPSE — maximum photospheric area obscured",
            color="#e4edf7", fontsize=6.6, fontweight="bold", va="center")
    partial_edges = SOLAR_PARTIAL_OBSCURATION_BAND_EDGES
    for index, color in enumerate(SOLAR_PARTIAL_OBSCURATION_BAND_COLORS):
        x = 0.18+index*1.42
        ax.add_patch(Rectangle((x, 1.13), 1.28, 0.42, facecolor=color,
                               edgecolor="#d9f3fb", linewidth=0.5))
        ax.text(x+0.64, 1.34,
                f"{partial_edges[index]:g}-{partial_edges[index+1]:g}%",
                color="white", fontsize=5.75, ha="center", va="center")
    ax.text(7.50, 1.34,
            "white contours = NASA-style eclipse magnitude 0.2 / 0.4 / 0.6 / 0.8",
            color="#c5d4e4", fontsize=5.45, ha="left", va="center")

    ax.text(0.18, 0.87, "TOTAL ECLIPSE CORRIDOR — local C2-C3 duration",
            color="#e4edf7", fontsize=6.6, fontweight="bold", va="center")
    total_edges = SOLAR_TOTALITY_PERCENT_BAND_EDGES
    reference_s = SOLAR_TOTALITY_DURATION_REFERENCE_S
    for index, color in enumerate(SOLAR_TOTALITY_PERCENT_BAND_COLORS):
        x = 0.18+index*1.42
        ax.add_patch(Rectangle((x, 0.20), 1.28, 0.42, facecolor=color,
                               edgecolor="#d9f3fb", linewidth=0.5))
        low, high = total_edges[index], total_edges[index+1]
        ax.text(x+0.64, 0.41,
                f"{low:g}-{high:g}%\n{_duration_tick_label(reference_s*low/100)}-{_duration_tick_label(reference_s*high/100)}",
                color="white", fontsize=5.0, ha="center", va="center", linespacing=0.95)
    ax.text(8.82, 0.41,
            "Every warm-colored point reached 100% coverage.\nHorizon test is geometric; atmospheric refraction is excluded.",
            color="#c5d4e4", fontsize=5.35, ha="left", va="center", linespacing=1.12)


def draw_lunar_geographic_visibility_map(
    ax, *, key_ax=None,
    map_extent: tuple[float, float, float, float] = (-180.0, 180.0, -75.0, 75.0),
):
    """Draw where the 2025 lunar eclipse was visible above the horizon."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    xmin, xmax, ymin, ymax = map(float, map_extent)
    _draw_map_background(ax, alpha=0.55)
    field = lunar_geographic_visibility_grid_cached()
    longitude = np.asarray(field["longitude"], dtype=float)
    latitude = np.asarray(field["latitude"], dtype=float)
    category = np.asarray(field["category"], dtype=float)
    ax.pcolormesh(
        longitude, latitude, category,
        cmap=ListedColormap(LUNAR_VISIBILITY_CATEGORY_COLORS, name="lunar_visibility"),
        norm=BoundaryNorm(np.arange(-0.5, 5.5, 1.0), 5),
        shading="nearest", alpha=0.84, zorder=2.0, rasterized=True,
    )
    for name, color, dash in (
        ("P1", "#7ed6ff", "--"), ("U2", "#ffb36b", "-"),
        ("U3", "#ffb36b", "-"), ("P4", "#7ed6ff", "--"),
    ):
        key = f"upper_limb_altitude_{name}_deg"
        if key in field:
            ax.contour(
                longitude, latitude, np.asarray(field[key], dtype=float),
                levels=[0.0], colors=[color], linewidths=0.72,
                linestyles=dash, alpha=0.92, zorder=3.2,
            )
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.set_xlabel("Longitude [deg east]", color="#b5c2d3", fontsize=7.0)
    ax.set_ylabel("Latitude [deg]", color="#b5c2d3", fontsize=7.0)
    ax.tick_params(colors="#b2bfd0", labelsize=6.2, length=2.1)
    ax.grid(color="#e4edf7", alpha=0.17, lw=0.46)
    for spine in ax.spines.values():
        spine.set_color("#52637a")
    ax.set_title(
        "Where the lunar eclipse was visible\nMoon upper limb above geometric horizon",
        color="white", fontsize=8.6, pad=4, fontweight="semibold",
    )

    handles = [Patch(facecolor=color, edgecolor="#dbe8f4", linewidth=0.35,
                     label=label)
               for color, label in zip(
                   LUNAR_VISIBILITY_CATEGORY_COLORS,
                   LUNAR_VISIBILITY_CATEGORY_LABELS,
               )]
    target = key_ax if key_ax is not None else ax
    if key_ax is not None:
        key_ax.set_facecolor("#06101b"); key_ax.set_xticks([]); key_ax.set_yticks([])
        for spine in key_ax.spines.values():
            spine.set_color("#52637a"); spine.set_linewidth(0.7)
        key_ax.legend(
            handles=handles, loc="center", ncol=3, fontsize=5.45,
            facecolor="#07101a", edgecolor="#66758b", framealpha=0.98,
            labelcolor="white", handlelength=1.35, columnspacing=0.65,
            borderpad=0.35,
        )
        key_ax.text(
            0.5, 0.04,
            "Blue dashed curves: P1/P4 moonrise or moonset limits  •  orange curves: U2/U3 totality limits  •  no atmospheric refraction",
            transform=key_ax.transAxes, color="#aebdd0", fontsize=4.8,
            ha="center", va="bottom",
        )
    else:
        target.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28),
            ncol=3, fontsize=5.4, facecolor="#07101a", edgecolor="#66758b",
            framealpha=0.98, labelcolor="white", handlelength=1.3,
        )

def _points_lat_lon(points):
    lat, lon = [], []
    for point in np.asarray(points):
        la, lo, _ = geodetic_from_itrf(point)
        lat.append(la); lon.append(lo)
    return np.asarray(lon), np.asarray(lat)



def _duration_tick_label(seconds: float) -> str:
    value = max(float(seconds), 0.0)
    minutes = int(value // 60.0)
    remainder = int(round(value - 60.0*minutes))
    if remainder == 60:
        minutes += 1
        remainder = 0
    return f"{minutes}:{remainder:02d}"


def draw_solar_map(
    ax,
    jd_utc: float,
    *,
    footprint_azimuth: int = 160,
    key_ax=None,
    key_layout: str = "side",
    map_extent: tuple[float, float, float, float] | None = None,
    show_penumbra: bool = True,
    show_umbra: bool = True,
    show_key: bool = True,
):
    """Plot discrete WGS-84 regions of local C2-C3 totality duration.

    Every point inside the corridor reaches complete photospheric coverage.
    The colored regions therefore encode *percent of the event's maximum
    totality duration* (268.1 s), not percent obscuration.  Region boundaries
    are interpolated directly within the exact fixed-observer duration mesh and
    remain clipped to the physical finite-Sun umbral corridor.  ``key_ax`` is
    outside the geographic axes so the eclipse can never pass underneath it.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Polygon

    if map_extent is None:
        xmin, xmax, ymin, ymax = -160.0, -25.0, -8.0, 68.0
    else:
        if len(map_extent) != 4:
            raise ValueError("map_extent must be (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = map(float, map_extent)

    # Use the full packaged 5400x2700 SSAPy surface instead of pre-reducing it
    # to 1600x800.  Matplotlib clips to the requested geographic extent, so the
    # North American scientific-summary crop keeps substantially more coastline
    # and terrain detail without changing the map geometry.
    earth = _load_texture("earth")
    ax.imshow(
        earth, extent=(-180, 180, -90, 90), origin="upper", alpha=0.68,
        interpolation="lanczos", resample=True, zorder=0,
    )
    veil = np.zeros((2, 2, 4), dtype=float)
    veil[..., :3] = np.array([0.010, 0.022, 0.042])
    veil[..., 3] = 0.18
    ax.imshow(veil, extent=(-180, 180, -90, 90), origin="upper", zorder=0.5)

    corridor = solar_umbra_corridor_cached()
    lon, lat = solar_central_path_cached()
    duration_field = solar_totality_duration_grid_cached()
    reference_s = float(duration_field["reference_duration_s"])

    band_edges = np.asarray(SOLAR_TOTALITY_PERCENT_BAND_EDGES, dtype=float)
    band_colors = tuple(SOLAR_TOTALITY_PERCENT_BAND_COLORS)
    band_labels = (
        "0-25%", "25-50%", "50-75%",
        "75-90%", "90-97.5%", "97.5-100%",
    )
    band_cmap = ListedColormap(band_colors, name="totality_duration_levels_v5")
    band_norm = BoundaryNorm(band_edges, band_cmap.N, clip=True)
    band_artist = ScalarMappable(norm=band_norm, cmap=band_cmap)
    band_artist.set_array(np.asarray([0.0, 100.0]))

    if len(corridor["jd"]):
        # Fill the entire physical corridor with the lowest band, then overlay
        # nested >=threshold polygons.  This yields true discrete regions and
        # avoids broadening the approximately 197.5-km WGS-84 corridor merely
        # to make it visible on a continental map.
        outer_polygon = np.column_stack([
            np.concatenate([corridor["left_lon"], corridor["right_lon"][::-1]]),
            np.concatenate([corridor["left_lat"], corridor["right_lat"][::-1]]),
        ])
        outer_patch = Polygon(
            outer_polygon, closed=True, facecolor=band_colors[0],
            edgecolor="none", alpha=0.94, zorder=3.0,
        )
        outer_patch.set_gid("totality-band-0-25")
        ax.add_patch(outer_patch)

        for band_index, threshold in enumerate(band_edges[1:-1], start=1):
            for polygon in solar_totality_band_polygons(float(threshold)):
                patch = Polygon(
                    polygon, closed=True, facecolor=band_colors[band_index],
                    edgecolor="none", alpha=0.96,
                    zorder=3.0+0.025*band_index,
                )
                patch.set_gid(
                    f"totality-band-{band_edges[band_index]:g}-{band_edges[band_index+1]:g}"
                )
                ax.add_patch(patch)
                # Thin light boundaries make the categorical regions readable
                # even in the small scientific-summary panel.
                ax.plot(
                    polygon[:, 0], polygon[:, 1], color="#07121f",
                    lw=0.56 if threshold < 90.0 else 0.72,
                    alpha=0.88, zorder=4.05,
                )

        ax.plot(
            corridor["left_lon"], corridor["left_lat"], color="#d6f4ff",
            lw=1.48, alpha=0.99, zorder=4.6, label="Corridor edge",
        )
        ax.plot(
            corridor["right_lon"], corridor["right_lat"], color="#d6f4ff",
            lw=1.48, alpha=0.99, zorder=4.6,
        )

    for index, (lo, la) in enumerate(_split_longitudes(lon, lat)):
        ax.plot(
            lo, la, color="#02070d", lw=3.4, zorder=5.0,
            solid_capstyle="round",
        )
        ax.plot(
            lo, la, color="#ffffff", lw=1.35, zorder=5.2,
            label="Central line" if index == 0 else None,
            solid_capstyle="round",
        )

    families = []
    if show_penumbra:
        families.append(("penumbra", "#ffd35f", 1.35, "Instantaneous penumbral limit"))
    if show_umbra:
        families.append(("umbra", "#ff4f61", 2.10, "Instantaneous umbral limit"))
    for family, color, width, label in families:
        points = solar_footprint_points(
            jd_utc, family=family, n_azimuth=max(96, int(footprint_azimuth)),
        )
        if len(points):
            lo, la = _points_lat_lon(points)
            first = True
            for seg_lon, seg_lat in _split_longitudes(lo, la):
                ax.plot(
                    seg_lon, seg_lat, color=color, lw=width, alpha=0.97,
                    zorder=6, label=label if first else None,
                )
                first = False

    center = solar_central_line_wgs84(jd_utc)
    if center is not None:
        ax.scatter(
            [center[1]], [center[0]], s=48, facecolor="#ffffff",
            edgecolor="#ff3f4f", lw=1.2, zorder=8, label="Shadow axis",
        )
    ax.scatter(
        [SOLAR_GREATEST_SITE_LON_EAST_DEG], [SOLAR_GREATEST_SITE_LAT_DEG],
        marker="D", s=35, facecolor="#06080d", edgecolor="#ffffff",
        lw=0.9, zorder=8, label="NASA greatest-eclipse site",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if (xmax-xmin) <= 90.0:
        x_step = 15.0
    else:
        x_step = 30.0
    if (ymax-ymin) <= 50.0:
        y_step = 10.0
    else:
        y_step = 15.0
    ax.set_xticks(np.arange(math.ceil(xmin/x_step)*x_step, xmax+0.1, x_step))
    ax.set_yticks(np.arange(math.ceil(ymin/y_step)*y_step, ymax+0.1, y_step))
    ax.set_xlabel("Longitude [deg east]", color="#b5c2d3", fontsize=8.0)
    ax.set_ylabel("Latitude [deg]", color="#b5c2d3", fontsize=8.0)
    ax.tick_params(colors="#b2bfd0", labelsize=7.5, length=2.5)
    ax.grid(color="#e4edf7", alpha=0.18, lw=0.55)
    for spine in ax.spines.values():
        spine.set_color("#52637a")
    ax.set_title(
        "WGS-84 local totality duration levels — C2-C3 (% of 268.1 s maximum)",
        color="white", fontsize=9.45, pad=5, fontweight="semibold",
    )

    if not show_key:
        return

    handles, labels = ax.get_legend_handles_labels()
    handles = [Patch(
        facecolor=band_colors[3], edgecolor="#d6f4ff", linewidth=0.7,
        label="Shaded totality-duration regions",
    )] + handles
    labels = ["Shaded totality-duration regions"] + labels
    band_centers = 0.5*(band_edges[:-1]+band_edges[1:])
    duration_edge_labels = [_duration_tick_label(reference_s*pct/100.0)
                            for pct in band_edges]

    def _band_text(low_pct: float, high_pct: float) -> str:
        low_s = reference_s*low_pct/100.0
        high_s = reference_s*high_pct/100.0
        return (f"{low_pct:g}-{high_pct:g}%\n"
                f"{_duration_tick_label(low_s)}-{_duration_tick_label(high_s)}")

    def _draw_band_strip(cax, *, compact: bool = False):
        from matplotlib.patches import Rectangle
        # Equal-width swatches are deliberate.  The map itself preserves the
        # physical width of every duration region; a proportional-width key
        # made the 90-97.5% and 97.5-100% labels nearly unreadable even though
        # those high-duration regions are scientifically important.
        n_bands = len(band_colors)
        cax.set_xlim(0.0, float(n_bands))
        cax.set_ylim(0.0, 1.0)
        cax.set_facecolor("#07101a")
        cax.set_yticks([])
        cax.set_xticks([])
        for spine in cax.spines.values():
            spine.set_color("#66758b")
            spine.set_linewidth(0.6)
        for index, (low, high, color) in enumerate(
                zip(band_edges[:-1], band_edges[1:], band_colors)):
            rect = Rectangle((float(index), 0.0), 1.0, 1.0,
                             facecolor=color, edgecolor="#d9f3fb",
                             linewidth=0.55, alpha=0.98)
            cax.add_patch(rect)
            fontsize = 5.65 if compact else 6.65
            cax.text(index+0.5, 0.50, _band_text(low, high),
                     ha="center", va="center", color="white",
                     fontsize=fontsize, linespacing=1.05,
                     fontweight="semibold" if index >= 4 else "normal")
        cax.text(0.5*n_bands, -0.34,
                 "Equal-width key swatches  •  top: percent of maximum duration  •  bottom: local C2-C3 minutes:seconds",
                 ha="center", va="top", color="#aebdd0",
                 fontsize=5.05 if compact else 5.9, clip_on=False)

    if key_ax is None:
        ax.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.17),
            fontsize=6.0, ncol=3, facecolor="#07101a", edgecolor="#66758b",
            framealpha=0.97, labelcolor="white", handlelength=1.8,
            columnspacing=0.8, borderpad=0.38,
        )
        cax = ax.inset_axes([0.075, -0.33, 0.85, 0.075])
        _draw_band_strip(cax, compact=True)
        return

    layout = str(key_layout).lower()
    if layout not in {"side", "horizontal", "auto"}:
        raise ValueError("key_layout must be 'side', 'horizontal', or 'auto'")
    if layout == "auto":
        bounds = key_ax.get_position()
        layout = "horizontal" if bounds.width > 1.7*bounds.height else "side"

    key_ax.set_facecolor("#06101b")
    key_ax.set_xticks([])
    key_ax.set_yticks([])
    for spine in key_ax.spines.values():
        spine.set_color("#52637a")
        spine.set_linewidth(0.7)

    if layout == "horizontal":
        key_ax.text(
            0.015, 0.95, "MAP KEY — shaded bands are totality-duration levels",
            transform=key_ax.transAxes, color="#e4edf7", fontsize=7.1,
            fontweight="bold", ha="left", va="top",
        )
        key_ax.legend(
            handles[1:], labels[1:], loc="upper center", bbox_to_anchor=(0.61, 0.985),
            fontsize=6.15, ncol=3, facecolor="#07101a", edgecolor="#66758b",
            framealpha=0.97, labelcolor="white", handlelength=1.55,
            handletextpad=0.36, borderpad=0.30, labelspacing=0.28,
            columnspacing=0.62,
        )
        key_ax.text(
            0.5, 0.48,
            "Every colored location reaches 100% photospheric coverage; color measures how long totality lasts, not obscuration.",
            transform=key_ax.transAxes, color="#cdd9e7", fontsize=6.25,
            ha="center", va="center",
        )
        cax = key_ax.inset_axes([0.045, 0.095, 0.91, 0.27])
        _draw_band_strip(cax, compact=True)
    else:
        key_ax.text(
            0.5, 0.97, "TOTALITY-DURATION LEVELS", transform=key_ax.transAxes,
            color="#e4edf7", fontsize=6.8, fontweight="bold",
            ha="center", va="top",
        )
        key_ax.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(0.03, 0.90),
            fontsize=5.55, ncol=1, facecolor="#07101a", edgecolor="#66758b",
            framealpha=0.97, labelcolor="white", handlelength=1.50,
            handletextpad=0.38, borderpad=0.32, labelspacing=0.36,
        )
        cax = key_ax.inset_axes([0.12, 0.075, 0.76, 0.25])
        _draw_band_strip(cax, compact=True)
        key_ax.text(
            0.5, 0.015,
            "All colored locations are fully total; the band records local C2-C3 duration.",
            transform=key_ax.transAxes, color="#aebbd0", fontsize=4.75,
            ha="center", va="bottom",
        )

def draw_lunar_shadow_plane(ax, jd_utc: float):
    """Draw the NASA Danjon shadow circles and the Moon's exact contact track."""
    import matplotlib.patches as patches

    state = lunar_reference_state(jd_utc)
    impact = state.impact_vector_deg
    impact_hat = impact/np.linalg.norm(impact)
    track_hat = np.array([impact_hat[1], -impact_hat[0]])
    center = impact+state.track_x_deg*track_hat

    ax.add_patch(patches.Circle((0, 0), _LUNAR_P_RADIUS_DEG,
                                facecolor="#9fb0c8", edgecolor="#d8e0eb",
                                alpha=0.15, lw=1.25, label="Penumbral radius"))
    ax.add_patch(patches.Circle((0, 0), _LUNAR_U_RADIUS_DEG,
                                facecolor="#67141b", edgecolor="#ef6659",
                                alpha=0.48, lw=1.35, label="Umbral radius"))
    ax.add_patch(patches.Circle(center, _LUNAR_MOON_SD_DEG,
                                facecolor="#d8d3ca", edgecolor="white",
                                alpha=0.92, lw=1.0, label="Moon at current UTC"))

    xs = np.linspace(-1.62, 1.62, 240)
    track = impact[None, :]+xs[:, None]*track_hat[None, :]
    ax.plot(track[:, 0], track[:, 1], color="#74d2ff", lw=1.35, ls="--",
            label="Moon-center track")

    label_sign = {"P1": 1, "U1": -1, "U2": 1, "MAX": -1,
                  "U3": 1, "U4": -1, "P4": 1}
    label_along = {"P1": -0.03, "U1": -0.03, "U2": -0.12, "MAX": 0.0,
                   "U3": 0.12, "U4": 0.03, "P4": 0.03}
    normal_hat = np.array([-track_hat[1], track_hat[0]])
    for name in ("P1", "U1", "U2", "MAX", "U3", "U4", "P4"):
        st = lunar_reference_state(LUNAR_2025.contacts_jd[name])
        point = impact+st.track_x_deg*track_hat
        ax.scatter([point[0]], [point[1]], s=17, color="#ffffff",
                   edgecolor="#27364c", linewidths=0.5, zorder=6)
        normal_scale = 0.15 if name in ("U2", "MAX", "U3") else 0.115
        offset = (normal_hat*(normal_scale*label_sign[name])
                  +track_hat*label_along[name])
        ax.text(point[0]+offset[0], point[1]+offset[1], name,
                color="#e4eef8", fontsize=6.7, ha="center", va="center",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.13", facecolor="#02040a",
                          edgecolor="#43516a", lw=0.35, alpha=0.80))

    ax.set_aspect("equal")
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.38, 1.38)
    ax.set_facecolor("#02040a")
    ax.set_xlabel("East / west angular offset [deg]", color="#a7b5c8", fontsize=8)
    ax.set_ylabel("North / south angular offset [deg]", color="#a7b5c8", fontsize=8)
    ax.tick_params(colors="#98a7ba", labelsize=7.3)
    ax.grid(color="#41506a", alpha=0.32, lw=0.55)
    for spine in ax.spines.values():
        spine.set_color("#46556a")
    ax.set_title("Moon's path through Earth's shadow\nPenumbral, partial, and total contacts",
                 color="white", fontsize=10, pad=5, fontweight="semibold")
    ax.legend(loc="lower left", fontsize=6.7, ncol=2,
              facecolor="#07101a", edgecolor="#66758b", framealpha=0.86,
              labelcolor="white", handlelength=2.0, columnspacing=0.9)


def solar_site_visibility_curve(jd_values):
    return np.asarray([solar_site_geometry(float(jd))[-1] for jd in jd_values])



def draw_light_curve(ax, event: ReferenceEvent, jd_utc: float):
    """Draw a contact-labelled, percent-scale geometric visibility curve."""
    t_hours = (event.jd-event.greatest_jd)*24.0
    current = (float(jd_utc)-event.greatest_jd)*24.0
    if event.mode == "solar":
        y_fraction = solar_site_visibility_curve(event.jd)
        title = "Direct sunlight visible at the observer"
        color = "#ffd45d"
    else:
        y_fraction = event.center_visibility
        title = "Direct sunlight reaching the Moon"
        color = "#efb9a5"
    y = 100.0*np.asarray(y_fraction)

    # Phase shading is tied to the exact contact definitions.
    if event.mode == "solar":
        # This curve is evaluated at the fixed NASA greatest-eclipse observer,
        # so its labels must always use that observer's C1-C4 contacts.  Global
        # P/U contacts describe the shadow touching Earth and are not the
        # contacts represented by this site-specific curve.
        local = solar_local_contacts()
        c1, c2, c3, c4 = [
            (local[name]-event.greatest_jd)*24.0 for name in ("C1", "C2", "C3", "C4")
        ]
        ax.axvspan(c1, c4, color="#d9b84d", alpha=0.055, lw=0)
        ax.axvspan(c2, c3, color="#67c5ff", alpha=0.12, lw=0)
        contacts_to_draw = {"C1": local["C1"], "C2": local["C2"],
                            "MAX": event.greatest_jd, "C3": local["C3"],
                            "C4": local["C4"]}
        subtitle = "C1–C4 fixed observer • C2–C3 = 268.0 s"
    else:
        contacts_to_draw = dict(event.contacts_jd)
        if event.mode == "lunar":
            p1, u1, u2, u3, u4, p4 = [
                (contacts_to_draw[name]-event.greatest_jd)*24.0
                for name in ("P1", "U1", "U2", "U3", "U4", "P4")
            ]
            ax.axvspan(p1, p4, color="#9fb0c8", alpha=0.045, lw=0)
            ax.axvspan(u1, u4, color="#b54b43", alpha=0.075, lw=0)
            ax.axvspan(u2, u3, color="#e0665b", alpha=0.12, lw=0)
            subtitle = "NASA contacts • penumbral / partial / total"
        else:
            subtitle = "Global Earth-contact sequence"

    ax.plot(t_hours, y, color=color, lw=2.2, zorder=4)
    ax.fill_between(t_hours, 0, y, color=color, alpha=0.10, zorder=2)
    current_y = float(np.interp(current, t_hours, y))
    ax.axvline(current, color="#ffffff", lw=1.2, alpha=0.95, zorder=6)
    ax.scatter([current], [current_y], s=28, facecolor="#ffffff",
               edgecolor=color, linewidth=0.9, zorder=7)

    ordered_contacts = sorted(contacts_to_draw.items(), key=lambda item: item[1])
    for rank, (name, value) in enumerate(ordered_contacts):
        x = (value-event.greatest_jd)*24.0
        ax.axvline(x, color="#64758d", lw=0.65, alpha=0.62, zorder=1)
        y_text = 104.0 if rank % 2 == 0 else 97.0
        align = "left" if rank == 0 else ("right" if rank == len(ordered_contacts)-1 else "center")
        ax.text(x, y_text, name, color="#c5d1df", fontsize=6.7,
                ha=align, va="bottom", clip_on=True)

    ax.set_xlim(t_hours.min(), t_hours.max())
    ax.set_ylim(-3.0, 111.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_facecolor("#02040a")
    ax.set_xlabel("Hours from greatest eclipse", color="#a7b5c8", fontsize=8.3)
    ax.set_ylabel("Unobscured solar disk", color="#a7b5c8", fontsize=8.3)
    ax.tick_params(colors="#98a7ba", labelsize=7.5)
    ax.grid(color="#41506a", alpha=0.28, lw=0.55)
    for spine in ax.spines.values():
        spine.set_color("#46556a")
    ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=8.55,
                 pad=5, fontweight="semibold")



def draw_north_indicator(ax, *, label: str = "N"):
    """Mark camera-up explicitly so body orientation is unambiguous."""
    ax.annotate(
        label, xy=(0.105, 0.93), xytext=(0.105, 0.78),
        xycoords="axes fraction", textcoords="axes fraction",
        color="#ffffff", fontsize=8.5, fontweight="bold",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.0),
        annotation_clip=False,
    )

def contact_status(definition: ReferenceDefinition, jd_utc: float) -> str:
    ordered = sorted(definition.contacts_jd.items(), key=lambda item: item[1])
    nearest = min(ordered, key=lambda item: abs(float(jd_utc)-item[1]))
    if abs(float(jd_utc)-nearest[1])*86400.0 < 0.8:
        return nearest[0]
    previous = [name for name, value in ordered if value <= jd_utc]
    following = [name for name, value in ordered if value > jd_utc]
    if previous and following:
        return f"{previous[-1]} → {following[0]}"
    return "before P1" if not previous else "after P4"
