"""Build and validate the compact V19 limb-darkening correction LUT."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ssapy_toolkit.compute.eclipse_photometry import limb_darkened_visibility_fraction

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
        "schema": "ssapy-toolkit.eclipse.photometry-lut/2.0",
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


def write_delta_lut(binary_path: str | Path, metadata_path: str | Path, **kwargs) -> dict[str, object]:
    encoded, metadata = build_delta_lut(**kwargs)
    binary = Path(binary_path)
    meta = Path(metadata_path)
    binary.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(encoded.tobytes(order="C"))
    meta.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("metadata")
    args = parser.parse_args()
    print(json.dumps(write_delta_lut(args.binary, args.metadata), indent=2))


def decode_delta_lut(encoded: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    """Decode an 8-bit correction texture back to floating-point deltas."""
    values = np.asarray(encoded, dtype=np.uint8)
    expected = (int(metadata["height"]), int(metadata["width"]))
    if values.shape != expected:
        raise ValueError(f"encoded LUT shape {values.shape} does not match metadata {expected}")
    lo = float(metadata["delta_min"])
    hi = float(metadata["delta_max"])
    return values.astype(float) / 255.0 * (hi - lo) + lo


def sample_delta_lut(
    encoded: np.ndarray,
    metadata: dict[str, object],
    q,
    s,
) -> np.ndarray:
    """Bilinearly sample the correction LUT using shader-equivalent coordinates."""
    table = decode_delta_lut(encoded, metadata)
    q_values, s_values = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(s, dtype=float))
    q_max = float(metadata["q_max"])
    s_max = float(metadata["s_max"])
    h, w = table.shape
    x = np.clip(s_values / s_max, 0.0, 1.0) * (w - 1)
    y = np.clip(q_values / q_max, 0.0, 1.0) * (h - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    tx = x - x0; ty = y - y0
    a = table[y0, x0] * (1.0 - tx) + table[y0, x1] * tx
    b = table[y1, x0] * (1.0 - tx) + table[y1, x1] * tx
    return a * (1.0 - ty) + b * ty


def visibility_from_lut(
    encoded: np.ndarray,
    metadata: dict[str, object],
    q,
    s,
) -> np.ndarray:
    """Reconstruct quadratic-visible flux from exact uniform overlap plus LUT correction."""
    q_values, s_values = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(s, dtype=float))
    uniform = limb_darkened_visibility_fraction(q_values, 1.0, s_values, law="uniform-disc")
    in_range = (
        (q_values >= 0.0) & (q_values <= float(metadata["q_max"])) &
        (s_values >= 0.0) & (s_values <= float(metadata["s_max"]))
    )
    correction = sample_delta_lut(encoded, metadata, q_values, s_values)
    return np.clip(uniform + np.where(in_range, correction, 0.0), 0.0, 1.0)
