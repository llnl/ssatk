#!/usr/bin/env python3
"""Prepare real reference data and Earth/Moon maps for the published gallery.

Run from a source checkout after installing the gallery extras. Inputs and
baked textures stay in SSATK_CACHE, outside the repository and published site.
OREKIT_DATA_DIR must name the dedicated Orekit data directory used at runtime.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from ssatk.io import moon_maps
from ssatk.plots._textures import viewer_earth_day_path


# NASA Earth Observatory, Blue Marble Next Generation, January 2004.
# Global equirectangular map: west to east (-180..180), north at the top.
EARTH_DAY_URL = (
    "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/"
    "bmng-topography/january/world.topo.200401.3x21600x10800.jpg"
)
EARTH_DAY_SHA256 = "da0b85bfeaeb2287a322b0a69375e96387e5e1da8ededd7a51352a5507c4821c"


# Fixed NASA/JPL products and an immutable Orekit reference-data revision.
# These hashes identify the exact inputs reviewed for the 2026 benchmark epoch.
INPUTS = (
    ("moon", moon_maps.DEM_URL,
     "45a2b32d56e81ed30db07fead8abc842b249b6511219d9ca2c53f81bc2dc5d62"),
    ("moon", moon_maps.ALBEDO_URL,
     "4af8b0cd4d50c30851359d98e7e72040240dd8d03256b58b345b5b76e9edb4ef"),
    ("orekit", "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de430/linux_p1550p2650.430",
     "0deb23ca9269496fcbab9f84bec9f3f090e263bfb99c62428b212790721de126"),
    ("orekit", "https://raw.githubusercontent.com/CS-SI/Orekit/"
     "d6d2f739a14345e2e023d52ecec68a8aeb753585/src/test/resources/regular-data/UTC-TAI.history",
     "518db19dfd8ee6847dd3397ac8baf626894794c572184eabf0c178cfaa4df36d"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, directory: Path, expected: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    # Orekit 10.3.1 JPLEphemeridesLoader recognizes lnx[mp]YYYY*.4xx,
    # whereas JPL distributes this same little-endian file as linux_*.430.
    if name == "linux_p1550p2650.430":
        name = "lnxp1550p2650.430"
    target = directory / name
    if target.is_file() and _sha256(target) == expected:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    try:
        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        if _sha256(partial) != expected:
            raise ValueError(f"Reference-data checksum mismatch: {url}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def prepare_earth_texture() -> Path:
    """Cache an 8K day map from the checksummed 21600 x 10800 NASA source.

    Downsample without rotating, flipping, or inventing geographic detail.
    The source has baked relief shading and represents January 2004, not live
    weather or the viewer's simulation epoch. No source assets enter Git.
    """
    from PIL import Image

    target = viewer_earth_day_path()
    source = _download(EARTH_DAY_URL, target.parent, EARTH_DAY_SHA256)
    partial = target.with_suffix(".jpg.part")
    # This particular trusted, hash-verified NASA product exceeds Pillow's
    # default image-size limit. Bound the override to its documented size and
    # restore the caller's limit even if decoding fails.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 21600 * 10800
    try:
        with Image.open(source) as image:
            if image.size != (21600, 10800):
                raise ValueError("Unexpected NASA Earth map dimensions")
            # Decode JPEG at half resolution first to limit peak memory.
            image.draft("RGB", (8192, 4096))
            image.resize((8192, 4096), Image.Resampling.LANCZOS).save(
                partial, format="JPEG", quality=93, subsampling=0,
            )
        partial.replace(target)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        partial.unlink(missing_ok=True)
    print(f"Prepared {target} (8192 x 4096, NASA Earth Observatory)", flush=True)
    return target


def main() -> None:
    directories = {
        "moon": Path(moon_maps.cache_dir()).expanduser().resolve(),
        "orekit": Path(os.environ["OREKIT_DATA_DIR"]).expanduser().resolve(),
    }
    for kind, url, expected in INPUTS:
        _download(url, directories[kind], expected)
    prepare_earth_texture()
    # Preserve the native 8192x4096 LROC colour map and 5760x2880 LOLA
    # normals. The independent shadow-horizon grid remains 720x360.
    moon_maps.main([
        "--outdir", str(directories["moon"]),
        "--dem", str(directories["moon"] / moon_maps.DEM_URL.rsplit("/", 1)[-1]),
        "--albedo", str(directories["moon"] / moon_maps.ALBEDO_URL.rsplit("/", 1)[-1]),
        "--albedo-max", "8192", "--horizon-width", "720", "--n-step", "128",
    ])


if __name__ == "__main__":
    main()
