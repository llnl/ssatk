"""Plotly mesh utilities for physically faithful body rendering.

Plotly/WebGL renders large indexed meshes differently across browser/GPU
combinations.  In particular, a single ``Mesh3d`` with more than roughly
65k vertices can overflow 16-bit element-index paths on older or software
WebGL implementations and produce folded, spiked, or partially missing
bodies.  Eclipse products must not depend on that implementation detail.

This module splits an indexed mesh into compact, independently indexed
chunks while preserving the exact input vertices and per-vertex colours.
It also supplies explicit equal-unit scene aspect ratios so a kilometre in X,
Y, and Z always has the same displayed length.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

# Conservative enough for WebGL1 / software-rendered Chromium while leaving
# ample room for Plotly's internal duplicated vertices.
DEFAULT_MAX_VERTICES = 48_000
DEFAULT_MAX_FACES = 16_000


def rgb_strings(rgb: np.ndarray) -> list[str]:
    """Convert Nx3 linear/sRGB values in [0, 1] to Plotly colour strings."""
    values = np.rint(np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0) * 255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in values]


def indexed_mesh_chunks(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertexcolor: Sequence[str] | np.ndarray | None = None,
    color: str | None = None,
    name: str,
    showlegend: bool = True,
    legendgroup: str | None = None,
    legendgrouptitle: Mapping[str, object] | None = None,
    hovertemplate: str | None = None,
    hoverinfo: str | None = None,
    flatshading: bool = False,
    lighting: Mapping[str, object] | None = None,
    lightposition: Mapping[str, float] | None = None,
    opacity: float = 1.0,
    visible: bool | str = True,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    max_faces: int = DEFAULT_MAX_FACES,
) -> list[go.Mesh3d]:
    """Return a list of safe ``Mesh3d`` chunks for one indexed surface.

    The geometry is not resampled, deformed, or simplified.  Each face batch
    is remapped to a compact local vertex array, so every WebGL draw call uses
    small indices.  Shared edge vertices are duplicated only at chunk
    boundaries, which does not change the rendered surface.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return [go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=name,
                          showlegend=showlegend, legendgroup=legendgroup,
                          visible=visible, hoverinfo="skip")]
    if np.min(f) < 0 or np.max(f) >= len(v):
        raise ValueError("faces contain an out-of-range vertex index")
    if max_vertices < 4 or max_faces < 1:
        raise ValueError("max_vertices/max_faces are too small")

    colors = None if vertexcolor is None else np.asarray(vertexcolor, dtype=object).reshape(-1)
    if colors is not None and len(colors) != len(v):
        raise ValueError("vertexcolor length must match vertices")

    traces: list[go.Mesh3d] = []
    start = 0
    while start < len(f):
        # Start with a face-count bound, then shrink until the unique vertex
        # count is safely below the WebGL bound.
        stop = min(start + max_faces, len(f))
        while True:
            batch = f[start:stop]
            unique, inverse = np.unique(batch.reshape(-1), return_inverse=True)
            if len(unique) <= max_vertices or stop <= start + 1:
                break
            stop = start + max(1, (stop - start) // 2)
        local_faces = inverse.reshape(-1, 3).astype(np.int32)
        local_vertices = v[unique]
        kwargs: dict[str, object] = {
            "x": local_vertices[:, 0],
            "y": local_vertices[:, 1],
            "z": local_vertices[:, 2],
            "i": local_faces[:, 0],
            "j": local_faces[:, 1],
            "k": local_faces[:, 2],
            "name": name,
            "showlegend": bool(showlegend and not traces),
            "legendgroup": legendgroup,
            "flatshading": flatshading,
            "opacity": float(opacity),
            "visible": visible,
        }
        if colors is not None:
            kwargs["vertexcolor"] = colors[unique].tolist()
        elif color is not None:
            kwargs["color"] = color
        if legendgrouptitle is not None and not traces:
            kwargs["legendgrouptitle"] = dict(legendgrouptitle)
        if hovertemplate is not None:
            kwargs["hovertemplate"] = hovertemplate
        elif hoverinfo is not None:
            kwargs["hoverinfo"] = hoverinfo
        if lighting is not None:
            kwargs["lighting"] = dict(lighting)
        if lightposition is not None:
            kwargs["lightposition"] = dict(lightposition)
        traces.append(go.Mesh3d(**kwargs))
        start = stop
    return traces


def equal_unit_aspect(ranges: Sequence[Sequence[float]]) -> dict[str, float]:
    """Aspect ratio that preserves equal data units for three axis ranges."""
    spans = np.asarray([abs(float(r[1]) - float(r[0])) for r in ranges], dtype=float)
    if np.any(~np.isfinite(spans)) or np.any(spans <= 0.0):
        raise ValueError("axis ranges must be finite and non-degenerate")
    scale = float(np.max(spans))
    ratios = np.maximum(spans / scale, 1.0e-6)
    return {"x": float(ratios[0]), "y": float(ratios[1]), "z": float(ratios[2])}


def radial_extent(vertices: np.ndarray, center: Iterable[float]) -> tuple[float, float]:
    """Minimum and maximum radius of vertices about ``center``."""
    values = np.asarray(vertices, dtype=float).reshape(-1, 3)
    c = np.asarray(tuple(center), dtype=float).reshape(3)
    radii = np.linalg.norm(values - c, axis=1)
    return float(np.min(radii)), float(np.max(radii))
