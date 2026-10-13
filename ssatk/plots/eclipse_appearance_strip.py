"""Illustrative eclipse phase strips using the installed lunar surface map.

Lunar thumbnails use a fixed near-side orientation (no epoch or libration).
Shadow transitions, red tint and solar corona are display models, not
calibrated photometry. Solar occulting discs remain black silhouettes.
"""

from .rendering import is_low_spec, _with_low_spec
from ._body_surfaces import surface_map

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

W, H = 260, 260  # per-panel raster size (rendered at 2x then downsampled)
SS = 2            # supersample factor for anti-aliased disk edges


def _grid(extent=1.0):
    """extent=1.0 (default) means the canvas edge sits exactly at the
    disk's own radius — fine for the Moon, which never needs anything
    drawn beyond its own edge, but wrong for the Sun during totality: a
    corona ring at radius >1.0 was silently clipped off-canvas entirely
    no matter how bright it was drawn, since there was no canvas space
    beyond radius 1.0 to begin with. Larger extent gives that headroom."""
    xs = np.linspace(-extent, extent, W * SS)
    ys = np.linspace(-extent, extent, H * SS)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def _moon_albedo(X, Y, seed=7):
    """Near-side lunar map on a sphere; schematic orientation, no libration."""
    R = np.hypot(X, Y)
    if is_low_spec():
        return np.ones_like(X), R
    # Camera along body +X, lunar north at the top of the raster.
    z = np.sqrt(np.maximum(0, 1-R*R))
    lon = np.arctan2(X, z)
    lat = np.arcsin(np.clip(-Y, -1, 1))
    image = surface_map("moon", 1024, 512)
    cols = ((lon/(2*np.pi)+.5)*1024).astype(int) % 1024
    rows = np.clip(((.5-lat/np.pi)*512).astype(int), 0, 511)
    return image[rows, cols].mean(axis=-1), R


@_with_low_spec
def render_lunar_panel(shadow_offset_x):
    """
    shadow_offset_x: horizontal position of Earth's shadow circle center,
    in disk-radius units. 0 = centered on Moon (totality), large = far away
    (full moon, no eclipse).
    """
    X, Y = _grid()
    albedo, R = _moon_albedo(X, Y)
    moon_mask = R <= 1.0

    # Real umbra + penumbra boundary, in Moon-radii units, from the same
    # angular geometry eclipse_demo.py's illumination_fraction() uses —
    # NOT an arbitrary "shadow radius + narrow sigmoid width" (an earlier
    # version of this function used exactly that, with a transition so
    # narrow that anything past ~2.65 rendered as essentially fully
    # bright — but real dimming continues all the way out to ~4.7
    # moon-radii, the true penumbra outer edge set by Earth's angular
    # radius PLUS the Sun's, not Earth's alone). That earlier version
    # made every near-totality frame in this range silently render wrong
    # — often close to full brightness — even though the real
    # illumination at that offset could be under 5%.
    umbra_r = 2.6413889393780616      # (R_earth - R_sun)-angle boundary
    penumbra_r = 4.701085435464585    # (R_earth + R_sun)-angle boundary
    d = np.sqrt((X - shadow_offset_x) ** 2 + Y**2)
    t = np.clip((d - umbra_r) / (penumbra_r - umbra_r), 0, 1)
    smooth_t = t * t * (3 - 2 * t)   # smoothstep — no kink at either edge

    # Brightness multiplier: floor raised from 0.055 to 0.16 — the darker
    # floor rendered as an almost-black red blob (mean RGB ~88,1,0 at
    # totality), nowhere near as vivid as a real blood moon photo. Real
    # totality is genuinely dim compared to a full moon, but still a
    # clearly warm red/orange disk, not a near-black one.
    floor = 0.16
    bright_mul = floor + (1.0 - floor) * smooth_t

    # Wavelength-dependent suppression -> red survives more than
    # green/blue, with red's own floor raised too (0.55 -> 0.72) so
    # totality stays visibly orange-red rather than relying entirely on
    # this factor to rescue brightness.
    r_ch = albedo * (0.72 + 0.28 * bright_mul)
    g_ch = albedo * bright_mul ** 1.4
    b_ch = albedo * bright_mul ** 2.0

    img = np.stack([r_ch, g_ch, b_ch], axis=-1)
    img = np.clip(img, 0, 1)
    img[~moon_mask] = 0.0

    rgba = np.dstack([img, moon_mask.astype(float)])
    arr = (rgba * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGBA")
    im = im.resize((W, H), Image.LANCZOS)
    return im


def render_solar_panel(moon_offset_x, corona=False, extent=1.8):
    """
    moon_offset_x: horizontal offset of the Moon's black disk relative to
    the Sun's disk center, in Sun-radius units. 0 = fully aligned.

    extent: half-width of the rendered canvas in Sun-radius units. Kept
    larger than 1.0 (the Sun's own radius) specifically so a corona ring
    has actual canvas space to render into — at extent=1.0 (matching the
    Moon panel's convention, where nothing is ever drawn past the disk's
    own edge) any corona ring beyond radius 1.0 was being silently
    clipped off-canvas by the image bounds themselves, no matter how
    bright it was drawn. This was the real reason the corona looked like
    a faint thin ring instead of the bright halo real eclipse photos
    show — not just a brightness/alpha tuning problem.
    """
    X, Y = _grid(extent=extent)
    R = np.sqrt(X**2 + Y**2)
    sun_mask = R <= 1.0

    moon_r = 0.98  # near-exact angular match -> total eclipse possible

    z = np.clip(1 - R**2, 0, 1) ** 0.5
    limb = 0.65 + 0.35 * z
    base_color = np.stack(
        [np.full_like(X, 1.00), np.full_like(X, 0.86), np.full_like(X, 0.55)],
        axis=-1,
    )
    sun_rgb = base_color * limb[..., None]

    d = np.sqrt((X - moon_offset_x) ** 2 + Y**2)
    covered = d <= moon_r

    img = sun_rgb.copy()
    img[covered] = 0.0

    arr = np.clip(img, 0, 1)
    # Opaque wherever the Sun's own disk is (whether lit gold or covered
    # black by the Moon's silhouette) — transparent everywhere else, so
    # panels sit cleanly on the black axes background instead of visibly
    # touching squares.
    alpha = sun_mask.astype(float)
    rgba = np.dstack([arr, alpha])
    im = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")

    def _px(v):
        """Map a grid coordinate to a pixel coordinate under this
        canvas's extent, along one axis."""
        return (v + extent) / (2 * extent) * (W * SS)

    if corona:
        glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow, "RGBA")
        cx, cy = _px(moon_offset_x), _px(0)
        # Much brighter, larger, and denser ring stack than before — the
        # old version (max alpha 170, out to 1.55x) was also being
        # clipped by the canvas edge on top of being faint. Real eclipse
        # photos show a corona that reads as a genuinely bright glow
        # filling a large fraction of the frame around the disk, fading
        # smoothly outward rather than stopping abruptly.
        for rad, alpha_v in [(1.7, 20), (1.5, 35), (1.32, 55), (1.16, 90),
                             (1.05, 150), (0.98, 230)]:
            rr = rad * (W * SS / (2 * extent))
            draw.ellipse(
                [cx - rr, cy - rr, cx + rr, cy + rr],
                outline=(255, 255, 250, alpha_v),
                width=max(2, int(SS * 6)),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=SS * 3.5))
        im = Image.alpha_composite(im, glow)
        # Re-draw the Moon's disk crisp and fully opaque on top after the
        # blur, so the corona glow doesn't show through the Moon itself.
        draw2 = ImageDraw.Draw(im)
        mr = moon_r * (W * SS / (2 * extent))
        draw2.ellipse([cx - mr, cy - mr, cx + mr, cy + mr], fill=(0, 0, 0, 255))

    im = im.resize((W, H), Image.LANCZOS)
    return im


@_with_low_spec
def make_strip(kind="lunar", n_panels=9, save_path=None):
    """
    kind: 'lunar' or 'solar'. Sweeps the occulter across the strip and
    renders each panel with the real overlap model above.
    """
    if kind == "lunar":
        lin = np.linspace(-1, 1, n_panels)
        offsets = np.sign(lin) * (np.abs(lin) ** 0.6) * 4.2
    else:
        offsets = np.linspace(-1.55, 1.55, n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(1.9 * n_panels, 2.6), facecolor="black")
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.98, bottom=0.02)

    for ax, off in zip(axes, offsets):
        ax.set_facecolor("black")
        if kind == "lunar":
            panel = render_lunar_panel(off)
        else:
            is_total = abs(off) < 0.05
            panel = render_solar_panel(off, corona=is_total)
        ax.imshow(panel)
        ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="black")
    return fig


if __name__ == "__main__":
    from ssatk.plots.figpath import figpath

    make_strip("lunar", 9, figpath("lunar_eclipse_strip.png"))
    make_strip("solar", 11, figpath("solar_eclipse_strip.png"))
    print("done")
