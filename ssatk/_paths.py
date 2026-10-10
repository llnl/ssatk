"""Shared path normalization helpers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OUTPUT_DIR_NAME = "ssatk_output"
SSATK_OUTPUT_ENV = "SSATK_OUTPUT_DIR"
HOME_OUTPUT_DIR = Path.home() / DEFAULT_OUTPUT_DIR_NAME


def safe_relative_parts(path: str | Path) -> list[str]:
    """Normalize a user path into safe relative path components."""
    user_path = Path(path)
    parts: list[str] = []
    for part in user_path.parts:
        if part in (user_path.anchor, "/", "\\", ""):
            continue
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return parts


def ensure_file_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return it as a Path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


CACHE_ENV = "SSATK_CACHE"
LEGACY_CACHE_ENVS = ("SSAPY_TOOLKIT_CACHE", "ssapy_toolkit_CACHE")


def cache_env_dir() -> str | None:
    """Cache directory from ``$SSATK_CACHE`` (or the pre-rename names), if set."""
    for name in (CACHE_ENV, *LEGACY_CACHE_ENVS):
        value = os.environ.get(name, "").strip()
        if value:
            return os.path.expanduser(value)
    return None


def output_root() -> Path:
    """Return the shared SSATK output root."""
    override = os.environ.get(SSATK_OUTPUT_ENV)
    return Path(override).expanduser() if override else HOME_OUTPUT_DIR
