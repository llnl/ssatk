"""Limb-darkening correction texture used by the cinematic WebGL shader."""
from __future__ import annotations

import numpy as np

from ssatk.compute.eclipse_core import limb_darkened_visibility_fraction

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_Q_MAX = 4.5
DEFAULT_S_MAX = 5.5
DEFAULT_DELTA_MIN = -0.10
DEFAULT_DELTA_MAX = 0.04


def build_delta_lut(
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    q_max: float = DEFAULT_Q_MAX,
    s_max: float = DEFAULT_S_MAX,
    delta_min: float = DEFAULT_DELTA_MIN,
    delta_max: float = DEFAULT_DELTA_MAX,
    quadrature_order: int = 96,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return an 8-bit LUT of limb-darkened minus uniform visibility.

    Storing the small smooth correction rather than the full visibility gives
    substantially better precision at 8 bits: the shader adds the interpolated
    correction to its exact analytic uniform-disc result.
    """
    width, height = int(width), int(height)
    if width < 64 or height < 64:
        raise ValueError("LUT dimensions must be at least 64 by 64")
    q = np.linspace(0.0, float(q_max), height)
    s = np.linspace(0.0, float(s_max), width)
    values = np.empty((height, width), dtype=float)
    uniform_values = np.empty_like(values)
    for row, q_value in enumerate(q):
        uniform = limb_darkened_visibility_fraction(q_value, 1.0, s, law="uniform-disc")
        limb = limb_darkened_visibility_fraction(
            q_value, 1.0, s, law="quadratic-visible", quadrature_order=quadrature_order,
        )
        uniform_values[row] = uniform
        values[row] = limb-uniform
    observed_min = float(values.min())
    observed_max = float(values.max())
    if observed_min < delta_min or observed_max > delta_max:
        raise ValueError(
            f"delta range [{observed_min}, {observed_max}] exceeds encoded "
            f"range [{delta_min}, {delta_max}]"
        )
    normalized = np.clip((values-delta_min)/(delta_max-delta_min), 0.0, 1.0)
    encoded = np.rint(normalized*255.0).astype(np.uint8)
    decoded = encoded.astype(float)/255.0*(delta_max-delta_min)+delta_min
    reconstructed = np.clip(uniform_values+decoded, 0.0, 1.0)
    exact = np.clip(uniform_values+values, 0.0, 1.0)
    error = np.abs(reconstructed-exact)
    metadata = {
        "law": "quadratic-visible",
        "width": width,
        "height": height,
        "q_max": float(q_max),
        "s_max": float(s_max),
        "delta_min": float(delta_min),
        "delta_max": float(delta_max),
        "quadrature_order": int(quadrature_order),
        "observed_delta_min": observed_min,
        "observed_delta_max": observed_max,
        "maximum_quantization_error": float(error.max()),
        "mean_quantization_error": float(error.mean()),
    }
    return encoded, metadata
