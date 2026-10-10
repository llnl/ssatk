"""Access data packaged outside SSATK.

SSATK keeps source code separate from bulky datasets and generated
media. Datasets ship in the split ``ssatk-data-*`` distributions, each of which
installs its own import package with files below ``<package>/data``:

* ``ssatk-data-core`` → ``ssapy_data_core`` (required)
* ``ssatk-data-gravity`` → ``ssapy_data_gravity`` (required)
* ``ssatk-data-lunar`` → ``ssapy_data_lunar`` (required)
* ``ssatk-data-lunar-gravity`` → ``ssapy_data_lunar_gravity`` (required)
* ``ssatk-data-propulsion`` → ``ssapy_data_propulsion`` (``[propulsion]`` extra)
* ``ssatk-data-benchmarks`` → ``ssapy_data_benchmarks`` (``[benchmarks]`` extra)

When no ``package`` is given, the helpers below search these packages in order
(then any user-supplied legacy package, if present)
and return the first match. This module wraps ``importlib.resources`` so
toolkit functions read those files from a normal wheel install without Git LFS,
git submodules, or runtime GitHub pulls.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
from importlib.resources import as_file, files
try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python 3.10
    from importlib.abc import Traversable
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Iterator

DEFAULT_DATA_PACKAGES = (
    "ssapy_data_core",
    "ssapy_data_lunar",
    "ssapy_data_gravity",
    "ssapy_data_lunar_gravity",
    "ssapy_data_propulsion",
    "ssapy_data_benchmarks",
)
"""Split data import packages, searched in this order when ``package`` is None."""

LEGACY_DATA_PACKAGE = "ssapy_data"
"""Import name of the pre-split single data package, searched last if installed."""

DEFAULT_DATA_PACKAGE = DEFAULT_DATA_PACKAGES[0]
"""Kept for backward compatibility; prefer ``package=None`` (search all)."""

DEFAULT_DATA_ROOT = "data"

_INSTALL_HINTS = {
    "propulsion": "pip install 'ssatk[propulsion]'  (ssatk-data-propulsion)",
    "benchmarks": "pip install 'ssatk[benchmarks]'  (ssatk-data-benchmarks)",
}
_DEFAULT_INSTALL_HINT = (
    "pip install ssatk-data-core ssatk-data-gravity ssatk-data-lunar "
    "ssatk-data-lunar-gravity"
)


class DataPackageNotFoundError(ModuleNotFoundError):
    """Raised when the configured SSAPy data package is not installed."""


class DataResourceNotFoundError(FileNotFoundError):
    """Raised when a requested resource is absent from the data package."""


def data_package_available(package: str | None = None) -> bool:
    """Return ``True`` when the named data package (or, by default, any) is installed."""

    names = (package,) if package else (*DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE)
    return any(find_spec(name) is not None for name in names)


def installed_data_packages() -> tuple[str, ...]:
    """Return the split (and legacy) data import packages that are installed."""

    return tuple(
        name
        for name in (*DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE)
        if find_spec(name) is not None
    )


def data_resource(
    relative_path: str | PathLike[str] = "",
    *,
    package: str | None = None,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    must_exist: bool = True,
) -> Traversable:
    """Return an ``importlib.resources`` object for packaged data.

    Parameters
    ----------
    relative_path
        POSIX-style path below ``data_root`` inside the data package. Absolute
        paths and ``..`` traversal are rejected.
    package
        Import package that owns the resource. ``None`` (the default) searches
        :data:`DEFAULT_DATA_PACKAGES`, then :data:`LEGACY_DATA_PACKAGE`, and
        returns the first package that contains ``relative_path``.
    data_root
        Directory inside ``package`` that contains data resources.
    must_exist
        If ``True``, raise :class:`DataResourceNotFoundError` when the resource
        is missing from every searched package.
    """

    root_parts = _safe_parts(data_root)
    rel_parts = _safe_parts(relative_path)
    candidates = (package,) if package else (*DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE)

    installed = []
    for name in candidates:
        try:
            resource = _package_root(name)
        except DataPackageNotFoundError:
            continue
        installed.append(name)
        for part in (*root_parts, *rel_parts):
            resource = resource.joinpath(part)
        if not must_exist or resource.exists():
            return resource

    requested = _display_path(data_root, relative_path)
    if not installed:
        searched = package or ", ".join(candidates)
        raise DataPackageNotFoundError(
            f"No SSATK data package is installed (searched: {searched}). "
            f"Install one with: {_install_hint(rel_parts)}"
        )
    raise DataResourceNotFoundError(
        f"Data resource '{requested}' was not found in installed data package(s) "
        f"{', '.join(installed)}. If it ships in an optional dataset, install it with: "
        f"{_install_hint(rel_parts)}"
    )


def resource_package(
    relative_path: str | PathLike[str],
    *,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
) -> str:
    """Return the import package that :func:`data_resource` resolves ``relative_path`` to."""

    rel_parts = _safe_parts(relative_path)
    for name in (*DEFAULT_DATA_PACKAGES, LEGACY_DATA_PACKAGE):
        try:
            resource = _package_root(name)
        except DataPackageNotFoundError:
            continue
        for part in (*_safe_parts(data_root), *rel_parts):
            resource = resource.joinpath(part)
        if resource.exists():
            return name
    raise DataResourceNotFoundError(
        f"Data resource '{_display_path(data_root, relative_path)}' is not in any installed data package."
    )


@contextmanager
def data_path(
    relative_path: str | PathLike[str],
    *,
    package: str | None = None,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
) -> Iterator[Path]:
    """Yield a filesystem path for a packaged data file.

    Use this when a downstream library requires a real path instead of a file
    object. The yielded path may be a temporary extraction path for zipped wheels,
    so callers should use it only inside the context manager.
    """

    resource = data_resource(relative_path, package=package, data_root=data_root)
    if not resource.is_file():
        requested = _display_path(data_root, relative_path)
        raise DataResourceNotFoundError(
            f"Data resource '{requested}' in package '{package}' is not a file."
        )

    with as_file(resource) as path:
        yield path


@contextmanager
def open_data(
    relative_path: str | PathLike[str],
    mode: str = "rb",
    *,
    package: str | None = None,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    encoding: str | None = None,
):
    """Open a packaged data file.

    Parameters mirror :meth:`importlib.resources.abc.Traversable.open`.
    """

    resource = data_resource(relative_path, package=package, data_root=data_root)
    if not resource.is_file():
        requested = _display_path(data_root, relative_path)
        raise DataResourceNotFoundError(
            f"Data resource '{requested}' in package '{package}' is not a file."
        )

    with resource.open(mode, encoding=encoding) as file_handle:
        yield file_handle


def read_data_text(
    relative_path: str | PathLike[str],
    *,
    package: str | None = None,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    encoding: str = "utf-8",
) -> str:
    """Read a packaged text data file."""

    with open_data(
        relative_path,
        "r",
        package=package,
        data_root=data_root,
        encoding=encoding,
    ) as file_handle:
        return file_handle.read()


def read_data_binary(
    relative_path: str | PathLike[str],
    *,
    package: str | None = None,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
) -> bytes:
    """Read a packaged binary data file."""

    with open_data(relative_path, "rb", package=package, data_root=data_root) as file_handle:
        return file_handle.read()


def _package_root(package: str) -> Traversable:
    try:
        return files(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        raise DataPackageNotFoundError(
            f"Data package '{package}' is not installed. Install the ssatk-data-* "
            "distribution that provides the required resource, then retry."
        ) from exc


def _install_hint(relative_parts: tuple[str, ...]) -> str:
    if relative_parts and relative_parts[0] in _INSTALL_HINTS:
        return _INSTALL_HINTS[relative_parts[0]]
    return _DEFAULT_INSTALL_HINT


def _safe_parts(path: str | PathLike[str]) -> tuple[str, ...]:
    path_string = str(path)
    if path_string in {"", "."}:
        return ()

    pure_path = PurePosixPath(path_string)
    if pure_path.is_absolute():
        raise ValueError(f"Packaged data paths must be relative, got '{path_string}'.")

    parts = tuple(part for part in pure_path.parts if part not in {"", "."})
    if ".." in parts:
        raise ValueError(f"Packaged data paths cannot contain '..', got '{path_string}'.")
    return parts


def _display_path(data_root: str | PathLike[str], relative_path: str | PathLike[str]) -> str:
    return "/".join((*_safe_parts(data_root), *_safe_parts(relative_path)))
