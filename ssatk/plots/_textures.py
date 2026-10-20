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

EARTH_TEXTURE_NAMES = ("earth.png", "earth_day_2048.jpg")


def moon_texture_path() -> Path:
    """Installed north-up lunar photomosaic; no implicit network downloads."""
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

    Search order: ``earth.png`` via :func:`ssapy.utils.find_file` (working
    directory, then installed data packages), then ``earth_day_2048.jpg``
    from the installed split data packages.

    Raises
    ------
    FileNotFoundError
        If neither texture is installed.
    """
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
