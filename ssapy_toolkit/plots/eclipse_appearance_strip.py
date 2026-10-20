"""High-resolution 2-D eclipse appearance renderers.

Unlike the original fixed-geometry thumbnails, every renderer accepts the
actual angular-radius or umbra/penumbra geometry for its event.  The lunar
penumbral gradient is derived from the same two-circle overlap equation used
by the 3-D meshes; the solar renderer accepts the event's real Moon/Sun
angular-size ratio and supports total, annular, and partial appearances.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

try:
    from ssapy_toolkit.compute.eclipse_brightness import (
        RE_KM, R_MOON_KM, _circle_overlap_fraction, illumination_fraction, irradiance_fraction,
    )
except ImportError:
    from ssapy_toolkit.compute.eclipse_brightness import (
        RE_KM, R_MOON_KM, _circle_overlap_fraction, illumination_fraction, irradiance_fraction,
    )

try:
    from ssapy_toolkit.plots.moon_render import (
        _load_real_moon_texture, _srgb_to_linear, _linear_to_srgb,
    )
except ImportError:
    try:
        from ssapy_toolkit.plots.moon_render import (
            _load_real_moon_texture, _srgb_to_linear, _linear_to_srgb,
        )
    except ImportError:
        _load_real_moon_texture = None

        def _srgb_to_linear(rgb):
            x = np.clip(np.asarray(rgb, dtype=float), 0, 1)
            return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

        def _linear_to_srgb(rgb):
            x = np.clip(np.asarray(rgb, dtype=float), 0, 1)
            return np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1/2.4)-0.055)

W, H = 320, 320
SS = 2
_MOON_CACHE = {}
_SUN_CACHE = {}


def _grid(extent=1.0):
    xs = np.linspace(-extent, extent, W * SS)
    ys = np.linspace(extent, -extent, H * SS)
    return np.meshgrid(xs, ys)


def _soft_noise(shape, seed, scales=(7, 18, 42)):
    """Deterministic multi-scale image noise without a hard SciPy dependency."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter
        for scale in scales:
            field = rng.normal(size=shape)
            out += gaussian_filter(field, sigma=scale, mode="wrap") / np.sqrt(scale)
    except Exception:
        out = rng.normal(size=shape)
    out -= np.nanmin(out)
    peak = np.nanmax(out)
    return out / peak if peak > 0 else out


def _moon_albedo(X, Y, seed=7):
    """Procedural maria, craters, relief normals, and limb shading."""
    rng = np.random.default_rng(seed)
    R = np.hypot(X, Y)
    albedo = np.full_like(X, 0.76)
    height = np.zeros_like(X)

    # Large near-side maria.
    maria = [(-0.28, 0.18, 0.25, 0.20), (0.18, 0.30, 0.22, 0.17),
             (0.32, -0.05, 0.20, 0.25), (-0.02, -0.28, 0.28, 0.17),
             (-0.42, -0.18, 0.16, 0.15), (0.04, 0.05, 0.15, 0.13)]
    for cx, cy, sx, sy in maria:
        d2 = ((X-cx)/sx)**2 + ((Y-cy)/sy)**2
        albedo -= 0.18 * np.exp(-0.5*d2)
        height -= 0.035 * np.exp(-0.5*d2)

    # Crater bowls + rims at three scales.
    for count, rlo, rhi, depth in [(20, 0.055, 0.15, 0.55),
                                   (120, 0.016, 0.055, 0.38),
                                   (420, 0.004, 0.018, 0.18)]:
        for _ in range(count):
            cx, cy = rng.uniform(-0.94, 0.94, 2)
            if cx*cx + cy*cy > 0.93**2:
                continue
            cr = rng.uniform(rlo, rhi)
            d = np.hypot(X-cx, Y-cy)
            bowl = np.clip(1-d/cr, 0, 1)
            rim = np.clip(1-np.abs(d-cr*0.90)/(cr*0.18), 0, 1)
            height -= depth*bowl**3
            height += depth*0.68*rim
            albedo -= depth*0.08*bowl**3
            albedo += depth*0.035*rim

    albedo += (rng.random(X.shape)-0.5)*0.018
    gy, gx = np.gradient(height)
    nx, ny, nz = -gx, -gy, np.ones_like(height)*2.4
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    light = np.array([-0.28, 0.38, 0.88]); light /= np.linalg.norm(light)
    diffuse = np.clip((nx*light[0]+ny*light[1]+nz*light[2])/norm, 0, 1)
    sphere_z = np.sqrt(np.clip(1-R*R, 0, 1))
    limb = 0.60 + 0.40*sphere_z
    shade = np.clip(0.48 + 0.72*diffuse, 0.25, 1.25)
    return np.clip(albedo*shade*limb, 0, 1), R


def _sample_texture(texture, lon_deg, lat_deg):
    """Periodic bilinear equirectangular sampling."""
    tex = np.asarray(texture, dtype=float)/255.0
    h, w = tex.shape[:2]
    x = ((np.asarray(lon_deg)+180.0)/360.0*w) % w
    y = np.clip((90.0-np.asarray(lat_deg))/180.0*(h-1), 0, h-1)
    x0 = np.floor(x).astype(int) % w
    y0 = np.floor(y).astype(int)
    x1 = (x0+1) % w
    y1 = np.minimum(y0+1, h-1)
    fx = (x-x0)[..., None]
    fy = (y-y0)[..., None]
    return ((tex[y0, x0]*(1-fx)+tex[y0, x1]*fx)*(1-fy)
            +(tex[y1, x0]*(1-fx)+tex[y1, x1]*fx)*fy)


def _moon_disc_base_rgb(X, Y, seed=7):
    """Map the SSAPy photomosaic onto the Earth-facing lunar hemisphere."""
    R = np.hypot(X, Y)
    z = np.sqrt(np.clip(1.0-R*R, 0, 1))
    if _load_real_moon_texture is not None:
        texture = _load_real_moon_texture(512, 1024)
    else:
        texture = None
    if texture is not None:
        # Body +X faces Earth; screen right is body +Y and screen up is +Z.
        lon = np.degrees(np.arctan2(X, z))
        lat = np.degrees(np.arcsin(np.clip(Y, -1, 1)))
        rgb = _sample_texture(texture, lon, lat)
        # The texture is an albedo/photomosaic, not a self-luminous disc.
        # A restrained centre-to-limb term keeps the spherical limb readable.
        rgb *= (0.88+0.12*z)[..., None]
        return np.clip(rgb, 0, 1), R
    albedo, R = _moon_albedo(X, Y, seed=seed)
    return np.repeat(albedo[..., None], 3, axis=-1), R


def _cached_moon_albedo(seed=7):
    key = int(seed)
    if key not in _MOON_CACHE:
        X, Y = _grid()
        _MOON_CACHE[key] = (X, Y, *_moon_disc_base_rgb(X, Y, seed=key))
    return _MOON_CACHE[key]


def _cached_sun_texture(extent=1.8, seed=11):
    key = (float(extent), int(seed))
    if key not in _SUN_CACHE:
        X, Y = _grid(extent=extent)
        R = np.hypot(X, Y)
        _SUN_CACHE[key] = (X, Y, R, _sun_texture(X, Y, R, seed=seed))
    return _SUN_CACHE[key]


def render_lunar_panel(
    shadow_offset_x,
    shadow_offset_y=0.0,
    *,
    umbra_radius=2.64,
    penumbra_radius=4.70,
    moon_center_km=None,
    sun_position_km=None,
    orientation_u_hat=None,
    blood_moon=True,
    atmosphere_floor=0.045,
    seed=7,
):
    """Render the Moon under an event-specific Earth shadow.

    The default offset/radius interface is in apparent Moon-radius units.
    Passing geocentric ``moon_center_km`` and ``sun_position_km`` activates the
    highest-fidelity path: every visible lunar-surface pixel is reconstructed
    in 3-D and evaluated with the finite-distance Earth/Sun apparent-disc
    model.  This preserves the curved umbral/penumbral boundary across the
    Moon instead of approximating the Moon as one flat target plane.
    """
    X, Y, base_rgb, R = _cached_moon_albedo(seed=seed)
    moon_mask = R <= 1.0
    geom = None

    if moon_center_km is not None and sun_position_km is not None:
        moon_center = np.asarray(moon_center_km, dtype=float)
        sun_position = np.asarray(sun_position_km, dtype=float)
        to_earth = -moon_center/np.linalg.norm(moon_center)
        if orientation_u_hat is None:
            ref = np.array([0.0, 0.0, 1.0]) if abs(to_earth[2]) < 0.92 else np.array([1.0, 0.0, 0.0])
            u_hat = np.cross(ref, to_earth)
        else:
            u_hat = np.asarray(orientation_u_hat, dtype=float)
            u_hat = u_hat-to_earth*np.dot(u_hat, to_earth)
        if np.linalg.norm(u_hat) < 1e-12:
            ref = np.array([1.0, 0.0, 0.0])
            u_hat = np.cross(ref, to_earth)
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(to_earth, u_hat)
        z = np.sqrt(np.clip(1.0-X*X-Y*Y, 0.0, 1.0))
        surface = R_MOON_KM * (
            X[..., None]*u_hat[None, None, :]
            + Y[..., None]*v_hat[None, None, :]
            + z[..., None]*to_earth[None, None, :]
        )
        positions = moon_center[None, None, :]+surface
        illum, geom = irradiance_fraction(
            positions.reshape(-1, 3), R_body_km=RE_KM,
            sun_position_km=np.broadcast_to(sun_position, positions.reshape(-1, 3).shape),
            return_geometry=True, photometry="quadratic-visible", quadrature_order=48,
        )
        illum = illum.reshape(X.shape)
    else:
        ru = float(max(umbra_radius, 0.0))
        rp = float(max(penumbra_radius, ru + 1e-9))
        occ_r = 0.5*(rp + ru)
        sun_r = 0.5*(rp - ru)
        distance = np.hypot(X-float(shadow_offset_x), Y-float(shadow_offset_y))
        illum = _circle_overlap_fraction(occ_r, sun_r, distance)

    direct = np.clip(illum, 0, 1)
    if geom is not None:
        a_occ = np.asarray(geom.occluder_angular_radius_rad).reshape(X.shape)
        a_sun = np.asarray(geom.sun_angular_radius_rad).reshape(X.shape)
        sep = np.asarray(geom.separation_rad).reshape(X.shape)
        depth = np.clip((a_occ-a_sun-sep)/np.maximum(2*a_sun, 1e-12), 0, 1)
    else:
        depth = np.clip((ru-distance)/max(2*sun_r, 1e-12), 0, 1)

    base_linear = _srgb_to_linear(base_rgb)*1.10
    rgb_linear = base_linear*(0.006+0.994*direct[..., None])
    if blood_moon:
        refracted = (1-direct)**1.55*(float(atmosphere_floor)+0.14*np.exp(-2.5*depth))
        rgb_linear += base_linear*refracted[..., None]*np.array([1.00, 0.16, 0.035])
    rgb = _linear_to_srgb(np.clip(rgb_linear, 0, 1))
    rgb[~moon_mask] = 0
    rgba = np.dstack([rgb, moon_mask.astype(float)])
    return Image.fromarray((rgba*255).astype(np.uint8), "RGBA").resize((W, H), Image.Resampling.LANCZOS)


def _sun_texture(X, Y, R, seed=11):
    z = np.sqrt(np.clip(1-R*R, 0, 1))
    mu = np.clip(z, 0, 1)
    # Quadratic limb darkening with multi-scale granulation.
    limb = 0.50 + 0.38*mu + 0.12*mu*mu
    gran = _soft_noise(X.shape, seed)
    gran = (gran-0.5)*0.22
    # A few active-region knots.
    rng = np.random.default_rng(seed+1)
    active = np.zeros_like(X)
    for _ in range(12):
        cx, cy = rng.uniform(-0.78, 0.78, 2)
        if cx*cx+cy*cy > 0.78**2:
            continue
        sx = rng.uniform(0.018, 0.065)
        active += rng.uniform(-0.18, 0.12)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sx*sx))
    intensity = np.clip(limb + gran + active, 0.18, 1.12)
    low = np.array([0.82, 0.22, 0.015])
    high = np.array([1.00, 0.93, 0.60])
    t = np.clip((intensity-0.18)/(1.12-0.18), 0, 1)[..., None]
    return low*(1-t) + high*t


def _corona_layer(extent, center_xy, moon_r, seed=19):
    """Asymmetric streamer corona with transparent background."""
    size = W*SS
    X, Y = _grid(extent)
    dx, dy = X-center_xy[0], Y-center_xy[1]
    R = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    rng = np.random.default_rng(seed)
    angular = np.zeros_like(theta)
    for _ in range(15):
        a = rng.uniform(-np.pi, np.pi)
        width = rng.uniform(0.025, 0.15)
        delta = np.angle(np.exp(1j*(theta-a)))
        angular += rng.uniform(0.2, 1.0)*np.exp(-(delta/width)**2)
    angular /= max(float(angular.max()), 1e-12)
    radial = np.exp(-np.clip(R-moon_r, 0, None)/0.34) / np.maximum(R, moon_r*0.7)**0.55
    inner = 1/(1+np.exp(-(R-moon_r)/0.012))
    alpha = np.clip((0.18+0.82*angular)*radial*inner, 0, 0.92)
    alpha[R > 1.75] = 0
    rgb = np.ones(X.shape+(3,))*np.array([0.95, 0.97, 1.00])
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray((rgba*255).astype(np.uint8), "RGBA")


def render_solar_panel(
    moon_offset_x,
    moon_offset_y=0.0,
    *,
    moon_radius=0.99,
    corona=False,
    extent=1.8,
    seed=11,
):
    """Render an event-specific solar eclipse.

    ``moon_radius`` is the apparent lunar radius divided by the apparent solar
    radius.  Values above one can produce totality; values below one produce
    annularity.  The Moon's centre offset is also measured in solar radii.
    """
    X, Y, R, sun_rgb_cached = _cached_sun_texture(extent=extent, seed=seed)
    sun_mask = R <= 1.0
    sun_rgb = sun_rgb_cached.copy()
    sun_rgb[~sun_mask] = 0

    dmoon = np.hypot(X-float(moon_offset_x), Y-float(moon_offset_y))
    covered = dmoon <= float(moon_radius)
    img = sun_rgb.copy()
    img[covered] = 0
    alpha = (sun_mask | covered).astype(float)
    rgba = np.dstack([img, alpha])
    base = Image.fromarray((np.clip(rgba, 0, 1)*255).astype(np.uint8), "RGBA")

    if corona:
        corona_img = _corona_layer(extent, (float(moon_offset_x), float(moon_offset_y)), float(moon_radius), seed+8)
        base = Image.alpha_composite(corona_img, base)
        # Restore a perfectly crisp lunar silhouette over the corona.
        draw = ImageDraw.Draw(base)
        scale = W*SS/(2*extent)
        cx = (float(moon_offset_x)+extent)*scale
        cy = (extent-float(moon_offset_y))*scale
        mr = float(moon_radius)*scale
        draw.ellipse((cx-mr, cy-mr, cx+mr, cy+mr), fill=(0, 0, 0, 255))

        # Chromospheric red rim appears only in the total/near-total regime.
        if moon_radius >= 0.995:
            rim = Image.new("RGBA", base.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(rim)
            scale = W*SS/(2*extent)
            cx = (float(moon_offset_x)+extent)*scale
            cy = (extent-float(moon_offset_y))*scale
            rr = float(moon_radius)*scale
            dr.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=(255, 85, 35, 150), width=max(1, SS))
            base = Image.alpha_composite(base, rim.filter(ImageFilter.GaussianBlur(0.7*SS)))

    return base.resize((W, H), Image.Resampling.LANCZOS)


def make_strip(
    kind="lunar",
    n_panels=9,
    save_path=None,
    *,
    offsets=None,
    lunar_umbra_radius=2.64,
    lunar_penumbra_radius=4.70,
    solar_moon_radius=0.99,
):
    """Create a standalone appearance strip, optionally with supplied geometry."""
    kind = str(kind).lower()
    if offsets is None:
        if kind == "lunar":
            p = np.linspace(-1, 1, n_panels)
            offsets = np.sign(p)*np.abs(p)**0.62*lunar_penumbra_radius*1.08
        else:
            offsets = np.linspace(-1.55, 1.55, n_panels)
    offsets = np.asarray(offsets, dtype=float)
    fig, axes = plt.subplots(1, len(offsets), figsize=(1.75*len(offsets), 2.25), facecolor="#02030a")
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.025, left=0.008, right=0.992, top=0.985, bottom=0.015)
    for ax, off in zip(axes, offsets):
        ax.set_facecolor("#02030a")
        if kind == "lunar":
            panel = render_lunar_panel(off, umbra_radius=lunar_umbra_radius,
                                       penumbra_radius=lunar_penumbra_radius)
        elif kind == "solar":
            deep = abs(off) <= abs(1-solar_moon_radius)+0.035
            panel = render_solar_panel(off, moon_radius=solar_moon_radius, corona=deep)
        else:
            raise ValueError("kind must be 'lunar' or 'solar'")
        ax.imshow(panel)
        ax.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    return fig


if __name__ == "__main__":
    from ssapy_toolkit.plots.figpath import figpath

    make_strip("lunar", 9, figpath("lunar_eclipse_strip.png"))
    make_strip("solar", 11, figpath("solar_eclipse_strip.png"))
    print("done")