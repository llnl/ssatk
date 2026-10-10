"""
Offline texture baker for the WebGL Moon.

Writes to ~/.ssatk/moon, not into SSATK or the lunar data package. The
horizon set at full resolution is tens of megabytes and must never enter
either repo; this module is the committed artefact, the textures are
regenerated locally. Override with $SSATK_CACHE.

With no source arguments, the baker downloads NASA SVS LOLA elevation and
LROC colour products into that cache. NASA source products remain subject to
NASA's media usage guidelines.

Outputs
  moon_normal.png         tangent-space normals from LOLA, true scale
  moon_horizon_[0-3].png  horizon elevation angle, 4 bearings per RGBA
  moon_horizon_meta.json  decode parameters

WHY A HORIZON MAP
-----------------
Ray marching many steps per pixel to decide shadowing is not a practical
per-frame budget. This moves the march offline: for each
texel and each of N compass bearings, store the highest elevation angle
terrain reaches along that bearing. The shader computes the Sun's local
elevation and azimuth and does one lookup.

Geometry is spherical. For a point at radius r0 and another at angular
distance psi at radius r1, the elevation of the second above the first's
local horizontal is

    atan2(r1*cos(psi) - r0, r1*sin(psi))

carrying lunar curvature exactly, as the 3D march does. A flat-terrain
horizon map over-shadows, never letting the surface curve away.

RESOLUTION
----------
The horizon grid should remain fine enough that applying a texel-centre
horizon to nearby points does not dominate the shadow edge.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np

R_MOON_KM = 1737.4
ANG_LO_DEG, ANG_HI_DEG = -10.0, 70.0     # 8-bit encode range, 0.314 deg/step
DEM_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16_uint.tif"
ALBEDO_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_8k.tif"


def cache_dir():
    """
    Where the baked textures live.

    Deliberately not inside SSATK or the lunar data package: this is tens of
    megabytes of regenerable data and must never enter either repo. Not
    AppData either -- Explorer hides it, which makes the output hard to
    find for no benefit. A leading dot does not hide a directory on
    Windows, so ~/.ssatk is visible there and still conventional
    on Linux and macOS.

    Override with $SSATK_CACHE, or per run with --outdir.
    """
    from ssatk._paths import cache_env_dir

    env = cache_env_dir()
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".ssatk", "moon")


def download_source(url, directory):
    """Download a NASA source into the cache once and return its path."""
    path = os.path.join(directory, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        partial = path + ".part"
        try:
            urllib.request.urlretrieve(url, partial)
            os.replace(partial, path)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
    return path


def load_dem(path):
    """
    Accepts either NASA's raw uint16 LOLA TIFF or an elevation .npz.

    The TIFF is the 16 ppd source and preserves the most detail in the normal
    map. An .npz must hold elevations in km relative to R_MOON_KM, in an
    ``elev_km`` array or as its first array. The horizon map is independently
    resized below.
    """
    path = os.path.expanduser(path)
    if path.lower().endswith((".tif", ".tiff")):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        arr = np.asarray(Image.open(path)).astype(np.float64)
        dem = (arr - 20000.0) / 2000.0          # uint16 half-metres, +20000
        lo, hi = dem.min(), dem.max()
        if not (-11.0 < lo < -7.0 and 8.0 < hi < 13.0):
            raise ValueError(f"decoded elevations {lo:.2f} to {hi:.2f} km are "
                             "outside LOLA's known range")
        return dem
    with np.load(path) as z:
        key = "elev_km" if "elev_km" in z else z.files[0]
        return np.asarray(z[key], dtype=np.float64)


def bake_normals(dem):
    """
    Tangent-space normal map at true scale, +Y north.

    Returns full XYZ; only XY is written to disk. Storing z is what wastes
    the encoding: near flat ground z sits close to 1.0, so 8-bit quantisation
    loses shallow terrain. Dropping z and reconstructing it in the shader as
    sqrt(1 - x^2 - y^2) puts the whole 8-bit range on the two components
    that actually vary.

    Polar 1/cos(lat) amplification is clamped at cos(lat) >= 0.05; a smaller
    floor permits an extreme slope boost within a degree of the pole -- a
    grid artefact, not lunar terrain.
    """
    H, W = dem.shape
    latn = np.radians(np.linspace(90, -90, H))[:, None]
    # Rows run north to south, so the row gradient is minus the northward slope.
    s_lat = -np.gradient(dem, axis=0) / (np.pi / H) / R_MOON_KM
    s_lon = (np.gradient(dem, axis=1) / (2 * np.pi / W)
             / (R_MOON_KM * np.maximum(np.cos(latn), 0.05)))
    nx, ny, nz = -s_lon, -s_lat, np.ones_like(s_lon)
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return np.stack([nx / n, ny / n, nz / n], axis=-1)


def bake_horizon(dem, n_az=16, n_step=128, max_km=260.0, shape=None):
    """
    Horizon elevation per texel per bearing, quantised to uint8.

    Quantised inside the loop, so the bearings are kept as uint8 rather than
    as a float64 stack: at 2880x1440x16 that is 66 MB instead of 531 MB.

    Bearing 0 is north, increasing eastward.
    """
    H, W = dem.shape
    oh, ow = shape or (H, W)

    lat = np.radians(np.linspace(90, -90, oh))[:, None] * np.ones((1, ow))
    lon = np.radians(np.linspace(-180, 180, ow, endpoint=False))[None, :] * np.ones((oh, 1))

    def sample(la, lo):
        r = np.clip(((np.pi / 2 - la) / np.pi * (H - 1)).astype(np.int32), 0, H - 1)
        c = (((lo + np.pi) / (2 * np.pi) * W).astype(np.int32)) % W
        return dem[r, c]

    r0 = R_MOON_KM + sample(lat, lon)
    psis = np.linspace(max_km / n_step, max_km, n_step) / R_MOON_KM
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    out = np.zeros((n_az, oh, ow), dtype=np.uint8)
    span = ANG_HI_DEG - ANG_LO_DEG
    for a in range(n_az):
        az = 2 * np.pi * a / n_az
        best = np.full((oh, ow), -np.pi / 2)
        for psi in psis:
            sin_p, cos_p = np.sin(psi), np.cos(psi)
            la2 = np.arcsin(np.clip(sin_lat * cos_p
                                    + cos_lat * sin_p * np.cos(az), -1, 1))
            lo2 = lon + np.arctan2(np.sin(az) * sin_p * cos_lat,
                                   cos_p - sin_lat * np.sin(la2))
            r1 = R_MOON_KM + sample(la2, lo2)
            np.maximum(best, np.arctan2(r1 * cos_p - r0, r1 * sin_p), out=best)
        enc = (np.degrees(best) - ANG_LO_DEG) / span * 255.0
        out[a] = np.clip(np.round(enc), 0, 255).astype(np.uint8)
        print(f"  bearing {a + 1}/{n_az}", end="\r", flush=True)
    print()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=None,
                    help="LOLA source: either ldem_16_uint.tif (16 ppd, "
                         "preferred -- normals come off the native grid) or "
                         "an elevation npz; defaults to NASA SVS")
    ap.add_argument("--outdir", default=None,
                    help="defaults to the platform cache dir, never a repo")
    ap.add_argument("--n-az", type=int, default=16)
    ap.add_argument("--n-step", type=int, default=128)
    ap.add_argument("--albedo", default=None,
                    help="LROC colour mosaic (TIFF/PNG) to re-encode as the "
                         "WebGL albedo texture; defaults to NASA SVS. Lossy is fine for colour, "
                         "unlike normals and horizon where a quantisation "
                         "level is a slope or a shadow edge")
    ap.add_argument("--albedo-quality", type=int, default=92)
    ap.add_argument("--albedo-max", type=int, default=8192,
                    help="cap the long edge. 8192 is fine on desktop GPUs; "
                         "drop to 4096 if the viewer must run on mobile or "
                         "integrated parts")
    ap.add_argument("--horizon-width", type=int, default=1440,
                    help="horizon map width; height is half")
    args = ap.parse_args(argv)

    outdir = os.path.expanduser(args.outdir) if args.outdir else cache_dir()
    os.makedirs(outdir, exist_ok=True)
    print(f"cache: {outdir}")

    dem_path = os.path.expanduser(args.dem) if args.dem else download_source(DEM_URL, outdir)
    albedo_path = os.path.expanduser(args.albedo) if args.albedo else download_source(ALBEDO_URL, outdir)
    dem = load_dem(dem_path)
    print(f"DEM {dem.shape[1]}x{dem.shape[0]}, {dem.min():.1f} to {dem.max():.1f} km")

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    nrm = bake_normals(dem)
    # XY only; the shader rebuilds z = sqrt(1 - x^2 - y^2). Blue is left at
    # zero so the file stays a plain RGB PNG.
    enc = np.zeros(nrm.shape[:2] + (3,), dtype=np.uint8)
    enc[..., :2] = np.clip(np.round((nrm[..., :2] * 0.5 + 0.5) * 255), 0, 255)
    p = os.path.join(outdir, "moon_normal.png")
    Image.fromarray(enc).save(p, optimize=True)
    print(f"normals -> {os.path.basename(p)}  {os.path.getsize(p) / 1e6:.1f} MB "
          f"({dem.shape[1]}x{dem.shape[0]}, XY only)")

    src = Image.open(albedo_path).convert("RGB")
    w, h = src.size
    if max(w, h) > args.albedo_max:
        k = args.albedo_max / float(max(w, h))
        src = src.resize((int(w * k), int(h * k)), Image.LANCZOS)
    p = os.path.join(outdir, "moon_albedo.jpg")
    src.save(p, quality=args.albedo_quality, optimize=True, progressive=True)
    print(f"albedo  -> {os.path.basename(p)}  "
          f"{os.path.getsize(p) / 1e6:.1f} MB ({src.size[0]}x{src.size[1]}, "
          f"q{args.albedo_quality})")

    hw = args.horizon_width
    hz = bake_horizon(dem, n_az=args.n_az, n_step=args.n_step,
                      shape=(hw // 2, hw))

    total = 0
    for k in range(args.n_az // 4):
        rgba = np.stack([hz[4 * k + j] for j in range(4)], axis=-1)
        p = os.path.join(outdir, f"moon_horizon_{k}.png")
        Image.fromarray(rgba, mode="RGBA").save(p, optimize=True)
        total += os.path.getsize(p)
        print(f"horizon {k} -> {os.path.basename(p)}  "
              f"{os.path.getsize(p) / 1e6:.1f} MB")

    meta = dict(n_az=args.n_az, width=int(hz.shape[2]), height=int(hz.shape[1]),
                angle_lo_deg=ANG_LO_DEG, angle_hi_deg=ANG_HI_DEG,
                max_km=260.0, n_step=args.n_step)
    with open(os.path.join(outdir, "moon_horizon_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"horizon {hz.shape[2]}x{hz.shape[1]}x{args.n_az}, "
          f"{total / 1e6:.1f} MB total")
    return outdir


if __name__ == "__main__":
    main()
