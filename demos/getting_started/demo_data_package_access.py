#!/usr/bin/env python3
"""Demonstrate reading data from an external SSAPy data package."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

from ssatk.data import (
    DataPackageNotFoundError,
    data_package_available,
    data_path,
    read_data_text,
)


DEMO_PACKAGE = "demo_ssapy_data"
DEMO_RESOURCE = "catalogs/sample_catalog.txt"
MISSING_PACKAGE = "demo_ssapy_data_missing_offline"


def _create_demo_package(root: Path) -> None:
    package_root = root / DEMO_PACKAGE
    data_root = package_root / "data" / "catalogs"
    data_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text('"""Demo-only SSAPy data package."""\n', encoding="utf-8")
    (data_root / "sample_catalog.txt").write_text(
        "object_id,semi_major_axis_km\nDEMO-1,42164\n",
        encoding="utf-8",
    )


def main(verbose: bool = True):
    """Run the packaged-data access demo without writing repository artifacts."""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _create_demo_package(root)
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        try:
            text = read_data_text(DEMO_RESOURCE, package=DEMO_PACKAGE)
            with data_path(DEMO_RESOURCE, package=DEMO_PACKAGE) as path:
                path_exists = path.exists()
                path_name = path.name
        finally:
            sys.path.remove(str(root))
            sys.modules.pop(DEMO_PACKAGE, None)

    missing_available = data_package_available(MISSING_PACKAGE)
    missing_error = ""
    try:
        read_data_text(DEMO_RESOURCE, package=MISSING_PACKAGE)
    except DataPackageNotFoundError as exc:
        missing_error = str(exc)

    result = {
        "title": "Packaged Data Access",
        "description": "Reads a sample data resource from a wheel-style package and reports graceful missing-package behavior.",
        "package": DEMO_PACKAGE,
        "resource": DEMO_RESOURCE,
        "text": text,
        "path_exists": path_exists,
        "path_name": path_name,
        "missing_package_available": missing_available,
        "missing_error": missing_error,
    }

    if verbose:
        print(f"Read {DEMO_RESOURCE} from package {DEMO_PACKAGE}:")
        print(text.strip())
        print()
        print(f"Missing package available: {missing_available}")
        print(f"Graceful missing-package message: {missing_error}")

    return result


if __name__ == "__main__":
    main()
