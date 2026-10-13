#!/usr/bin/env python3
"""Prepare real reference data and lunar maps for the published gallery.

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


def main() -> None:
    directories = {
        "moon": Path(moon_maps.cache_dir()).expanduser().resolve(),
        "orekit": Path(os.environ["OREKIT_DATA_DIR"]).expanduser().resolve(),
    }
    for kind, url, expected in INPUTS:
        _download(url, directories[kind], expected)
    # Normals retain the native 5760x2880 LOLA grid. The gallery uses a 4K
    # albedo and 720x360 horizon grid to limit browser memory and build time.
    moon_maps.main([
        "--outdir", str(directories["moon"]),
        "--dem", str(directories["moon"] / moon_maps.DEM_URL.rsplit("/", 1)[-1]),
        "--albedo", str(directories["moon"] / moon_maps.ALBEDO_URL.rsplit("/", 1)[-1]),
        "--albedo-max", "4096", "--horizon-width", "720", "--n-step", "128",
    ])


if __name__ == "__main__":
    main()
