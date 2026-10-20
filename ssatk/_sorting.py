"""Shared sorting helpers."""

from __future__ import annotations

import re


def natural_key(value: str):
    """Split text into lowercase text and integer tokens for natural sorting."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    ]
