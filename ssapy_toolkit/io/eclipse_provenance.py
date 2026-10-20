"""Portable serialization helpers for sealed eclipse records."""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any
import os
import re

_PACKAGE_ROOT = Path(__file__).resolve().parent
_RELEASE_ROOT = _PACKAGE_ROOT.parent


def _portable_path(value: str) -> str:
    text = str(value)
    is_windows = bool(re.match(r"^[A-Za-z]:[\\/]", text))
    if not (text.startswith("/") or is_windows):
        return text
    try:
        path = Path(text).expanduser().resolve()
    except Exception:
        return f"external://{PurePath(text).name}"
    for root, prefix in ((_PACKAGE_ROOT, "package://ssapy_toolkit.eclipse"), (_RELEASE_ROOT, "release://")):
        try:
            rel = path.relative_to(root).as_posix()
            return prefix.rstrip("/") + "/" + rel
        except Exception:
            pass
    parts = {part.lower() for part in path.parts}
    if "ssapy_data" in parts or "ssapy-data" in parts or "ssapy_data_overlay" in parts:
        return "ssapy-data://" + path.name
    return f"external://{path.name}"


def portable_record(value: Any) -> Any:
    """Recursively replace machine-local absolute paths with logical URIs."""
    if isinstance(value, dict):
        return {str(key): portable_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_record(item) for item in value]
    if isinstance(value, Path):
        return _portable_path(str(value))
    if isinstance(value, str):
        return _portable_path(value)
    return value


__all__ = ["portable_record"]
