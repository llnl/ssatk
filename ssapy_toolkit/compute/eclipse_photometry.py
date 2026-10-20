"""Finite-source photometry re-exported from the canonical SSAPy kernel.

The implementation lives in :mod:`ssapy_toolkit.compute.eclipse_core`.
All finite-source geometry and radiometry code is owned by SSAPy-Toolkit;
core SSAPy is not modified by this release.
"""
from __future__ import annotations

from ssapy_toolkit.compute.eclipse_core import (
    CORE_PROVIDER,
    LimbDarkeningLaw,
    UNIFORM_DISC,
    VISIBLE_LINEAR,
    VISIBLE_QUADRATIC,
    apparent_disk_radiometry,
    finite_source_irradiance,
    limb_darkened_visibility_fraction,
    quadrature_sample_weight,
    resolve_limb_darkening,
)

__all__ = [
    "CORE_PROVIDER",
    "LimbDarkeningLaw",
    "UNIFORM_DISC",
    "VISIBLE_LINEAR",
    "VISIBLE_QUADRATIC",
    "resolve_limb_darkening",
    "limb_darkened_visibility_fraction",
    "finite_source_irradiance",
    "apparent_disk_radiometry",
    "quadrature_sample_weight",
]
