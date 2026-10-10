"""Helpers for copying public sibling-module attributes into package namespaces."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import MutableMapping
from pathlib import Path


_IMPORTED_PUBLIC_NAME_DENYLIST = {
    "Any",
    "Callable",
    "Iterable",
    "Path",
    "Sequence",
    "Time",
    "annotations",
    "dataclass",
    "datetime",
    "field",
    "np",
    "os",
    "plt",
    "sys",
    "warnings",
}


def _default_public_names(module) -> list[str]:
    """Return public names defined by ``module`` when it does not declare ``__all__``."""
    names = []
    for attr in dir(module):
        if attr.startswith("_") or attr in _IMPORTED_PUBLIC_NAME_DENYLIST:
            continue

        value = getattr(module, attr)
        if inspect.ismodule(value):
            continue
        if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ != module.__name__:
            continue
        names.append(attr)
    return names


def import_public_modules(
    package_name: str,
    package_file: str,
    namespace: MutableMapping[str, object],
    *,
    skip: set[str] | None = None,
) -> None:
    """Import sibling modules and copy their public attributes into a package namespace."""
    skip = set() if skip is None else skip
    package_dir = Path(package_file).resolve().parent
    exported_names = {
        name for name in namespace if not name.startswith("_") and name != "import_public_modules"
    }
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py" or path.stem.startswith("_") or path.stem in skip:
            continue

        module = importlib.import_module(f"{package_name}.{path.stem}")
        public_names = getattr(module, "__all__", None)
        if public_names is None:
            public_names = _default_public_names(module)
        for attr in public_names:
            namespace[attr] = getattr(module, attr)
            exported_names.add(attr)

    namespace.pop("import_public_modules", None)
    namespace["__all__"] = sorted(name for name in exported_names if name in namespace)
