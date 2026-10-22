"""Locate the equirectangular Earth texture used by the 2-D maps and globes.

Historically every plot asked SSAPy for ``earth.png`` (5400 x 2700 Blue
Marble, longitude -180..180 left to right, north up). That file lived in
SSAPy's Git LFS tree and legacy installations, but none
of the split ``ssa-data-*`` 0.0.1 wheels ships it. ``ssa-data-core`` does
ship ``earth_day_2048.jpg`` (2048 x 1024) in the same projection and
orientation (normalized cross-correlation 0.866 at zero longitude shift
against ``earth.png``, peak at zero shift, no vertical flip), so it is the
fallback here.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

EARTH_TEXTURE_NAMES = ("earth.png", "earth_day_2048.jpg")
GALLERY_TEXTURE_SIZE = (8192, 4096)


@lru_cache(maxsize=8)
def _check_gallery_texture(path, modified, length):
    from PIL import Image

    with Image.open(path) as image:
        if image.size != GALLERY_TEXTURE_SIZE:
            raise ValueError(
                f"Gallery texture {path} is {image.width} x {image.height}; "
                "expected 8192 x 4096. Rebuild with "
                "python -m scripts.prepare_gallery_data."
            )
        # JPEG's verify() does not decode the raster; load it once so a
        # truncated 8K file cannot pass the gate and break an exported viewer.
        image.load()


def require_gallery_texture(path, body):
    """Reject missing, damaged, or stale lower-resolution gallery maps.

    Validation is cached by file identity so replacing a cached image is
    checked again. This function never downloads or modifies data.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Gallery {body.title()} texture is missing: {path}. "
            "Run python -m scripts.prepare_gallery_data before building the gallery."
        ) from exc
    _check_gallery_texture(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    return path


def moon_texture_path() -> Path:
    """Prepared 8K gallery albedo, otherwise the installed lunar photomosaic."""
    from .rendering import gallery_max_textures
    if gallery_max_textures():
        from ssatk.io.moon_maps import cache_dir
        return require_gallery_texture(Path(cache_dir()) / "moon_albedo.jpg", "moon")
    from ssapy.utils import find_file
    try:
        return Path(find_file("moon", ext=".png"))
    except (FileNotFoundError, TypeError):
        from .starfield import find_data_file
        path = find_data_file("moon.png")
        if path is not None:
            return Path(path)
    raise FileNotFoundError("No lunar surface map found; install ssa-data-lunar (moon.png).")


def viewer_earth_day_path() -> Path:
    """Cache location of the optional 8K viewer texture (may not exist).

    ``scripts.prepare_gallery_data`` builds this from NASA Blue Marble.
    Ordinary viewer exports never download assets implicitly.
    """
    from ssatk._paths import cache_env_dir

    root = Path(cache_env_dir() or Path.home() / ".ssatk")
    return root / "earth" / "earth_day_8192.jpg"


def earth_texture_path() -> Path:
    """Return a filesystem path to the best available Earth day texture.

    Gallery rendering requires the prepared 8192 x 4096 NASA map and never
    falls back to a smaller installed image. No downloads occur here.

    Search order: ``earth.png`` via :func:`ssapy.utils.find_file` (working
    directory, then installed data packages), then ``earth_day_2048.jpg``
    from the installed split data packages.

    Raises
    ------
    FileNotFoundError
        If neither texture is installed.
    """
    from .rendering import gallery_max_textures
    if gallery_max_textures():
        return require_gallery_texture(viewer_earth_day_path(), "earth")
    try:
        from ssapy.utils import find_file

        return Path(find_file("earth", ext=".png"))
    except (ImportError, FileNotFoundError, TypeError):
        pass

    from ssatk.plots.starfield import find_data_file

    for name in EARTH_TEXTURE_NAMES:
        path = find_data_file(name)
        if path is not None:
            return Path(path)

    raise FileNotFoundError(
        "No Earth texture found: looked for "
        + " and ".join(EARTH_TEXTURE_NAMES)
        + " in the working directory and the installed ssa-data-* packages. "
        "Install ssa-data-core."
    )
