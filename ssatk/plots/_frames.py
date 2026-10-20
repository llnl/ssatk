"""Shared plot frame normalization helpers."""

from __future__ import annotations


_FRAME_ALIASES = {
    "gcrf": "gcrf",
    "gcrs": "gcrf",
    "itrf": "itrf",
    "itrs": "itrf",
    "lunar": "lunar",
    "lunar_fixed": "lunar",
    "lunar fixed": "lunar",
    "lunar_centered": "lunar",
    "lunar centered": "lunar",
    "lunarearthfixed": "lunar axis",
    "lunarearth": "lunar axis",
    "lunar axis": "lunar axis",
    "lunar_axis": "lunar axis",
    "lunaraxis": "lunar axis",
}


def normalize_orbit_frame(frame: str) -> str | None:
    """Normalize supported orbit plot frame aliases."""

    return _FRAME_ALIASES.get(frame.lower())
