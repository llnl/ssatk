"""Pure finite-source occultation geometry for SSATK.

This module contains the reusable numerical kernel used by the eclipse plots.
It has no plotting, file-system, or data-package dependencies.
All distance arguments are unit-agnostic: every radius and position supplied to
one call must use the same units. Angles are radians.

"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

import numpy as np

ArrayLike = np.ndarray | Iterable[float] | float
PhotometryKind = Literal["uniform", "linear", "quadratic"]


@dataclass(frozen=True)
class ApparentDiskGeometry:
    """Apparent angular geometry of a finite source and circular occluder."""

    occluder_angular_radius_rad: np.ndarray
    source_angular_radius_rad: np.ndarray
    separation_rad: np.ndarray
    occluder_distance: np.ndarray
    source_distance: np.ndarray


@dataclass(frozen=True)
class LimbDarkeningLaw:
    """Radially symmetric center-to-limb intensity law.

    ``mu`` is the cosine of the angle between the surface normal and the
    line of sight.  ``mu=1`` is the apparent source center and ``mu=0`` the
    apparent limb.
    """

    name: str
    kind: PhotometryKind = "uniform"
    coefficients: tuple[float, ...] = ()
    wavelength_nm: float | None = None
    description: str = ""

    def intensity(self, mu: ArrayLike) -> np.ndarray:
        values = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
        one_minus = 1.0 - values
        if self.kind == "uniform":
            out = np.ones_like(values)
        elif self.kind == "linear":
            if len(self.coefficients) != 1:
                raise ValueError("linear limb darkening requires one coefficient")
            (u,) = self.coefficients
            out = 1.0 - float(u) * one_minus
        elif self.kind == "quadratic":
            if len(self.coefficients) != 2:
                raise ValueError("quadratic limb darkening requires two coefficients")
            u1, u2 = map(float, self.coefficients)
            out = 1.0 - u1 * one_minus - u2 * one_minus * one_minus
        else:  # pragma: no cover
            raise ValueError(f"unsupported limb-darkening kind {self.kind!r}")
        return np.clip(out, 0.0, None)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "coefficients": list(self.coefficients),
            "wavelength_nm": self.wavelength_nm,
            "description": self.description,
        }


UNIFORM_DISC = LimbDarkeningLaw(
    name="uniform-disc",
    kind="uniform",
    description="Uniform source surface brightness; appropriate for contact geometry.",
)
VISIBLE_LINEAR = LimbDarkeningLaw(
    name="linear-visible",
    kind="linear",
    coefficients=(0.60,),
    wavelength_nm=550.0,
    description="Broadband visible linear center-to-limb approximation.",
)
VISIBLE_QUADRATIC = LimbDarkeningLaw(
    name="quadratic-visible",
    kind="quadratic",
    coefficients=(0.47, 0.28),
    wavelength_nm=550.0,
    description="Broadband visible quadratic center-to-limb approximation.",
)

_LAWS = {
    "uniform": UNIFORM_DISC,
    "uniform-disc": UNIFORM_DISC,
    "linear": VISIBLE_LINEAR,
    "linear-visible": VISIBLE_LINEAR,
    "quadratic": VISIBLE_QUADRATIC,
    "quadratic-visible": VISIBLE_QUADRATIC,
    "limb-darkened": VISIBLE_QUADRATIC,
}


def resolve_limb_darkening(
    model: str | LimbDarkeningLaw | None = None,
    *,
    coefficients: Iterable[float] | None = None,
    wavelength_nm: float | None = None,
) -> LimbDarkeningLaw:
    """Resolve a named or user-supplied center-to-limb law."""

    if isinstance(model, LimbDarkeningLaw):
        if coefficients is not None or wavelength_nm is not None:
            raise ValueError("do not override an explicit LimbDarkeningLaw")
        return model
    key = "quadratic-visible" if model is None else str(model).strip().lower()
    if coefficients is None:
        try:
            return _LAWS[key]
        except KeyError as exc:
            raise ValueError(
                "photometry must be uniform-disc, linear-visible, or quadratic-visible"
            ) from exc
    coeffs = tuple(float(value) for value in coefficients)
    if key in {"uniform", "uniform-disc"}:
        kind: PhotometryKind = "uniform"
    elif key in {"linear", "linear-visible"}:
        kind = "linear"
    elif key in {"quadratic", "quadratic-visible", "limb-darkened"}:
        kind = "quadratic"
    else:
        raise ValueError(f"cannot infer limb-darkening kind from {model!r}")
    return LimbDarkeningLaw(
        name=f"custom-{kind}",
        kind=kind,
        coefficients=coeffs,
        wavelength_nm=wavelength_nm,
        description="User-supplied center-to-limb law.",
    )


def _unit(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError(f"{name} must be nonzero")
    return values / norm


def apparent_angular_radius(radius: ArrayLike, distance: ArrayLike) -> np.ndarray:
    """Angular radius of a sphere, with safe containment clipping."""

    r, d = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(distance, dtype=float))
    if np.any(r < 0.0):
        raise ValueError("radius must be non-negative")
    if np.any(d <= 0.0):
        raise ValueError("distance must be positive")
    return np.arcsin(np.clip(r / d, 0.0, 1.0))


def circle_overlap_visible_fraction(
    occluder_radius: ArrayLike,
    source_radius: ArrayLike,
    separation: ArrayLike,
) -> np.ndarray:
    """Visible fraction of the second circular disc after occultation.

    Parameters use any common angular unit.  The result is the unobscured area
    of the source disc divided by its full area.
    """

    r1, r2, d = np.broadcast_arrays(
        np.asarray(occluder_radius, dtype=float),
        np.asarray(source_radius, dtype=float),
        np.asarray(separation, dtype=float),
    )
    if np.any(r1 < 0.0) or np.any(r2 < 0.0) or np.any(d < 0.0):
        raise ValueError("radii and separation must be non-negative")
    visible = np.ones_like(d)
    valid_source = r2 > 0.0
    no_overlap = d >= r1 + r2
    contained = d <= np.abs(r1 - r2)
    visible = np.where(contained & (r1 >= r2) & valid_source, 0.0, visible)
    with np.errstate(divide="ignore", invalid="ignore"):
        annular = 1.0 - np.clip((r1 / np.where(valid_source, r2, 1.0)) ** 2, 0.0, 1.0)
    visible = np.where(contained & (r1 < r2) & valid_source, annular, visible)

    partial = valid_source & ~no_overlap & ~contained
    dp = np.where(partial, np.maximum(d, np.finfo(float).tiny), 1.0)
    a = np.where(partial, r1, 1.0)
    b = np.where(partial, r2, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg1 = np.clip((dp * dp + a * a - b * b) / (2.0 * dp * a), -1.0, 1.0)
        arg2 = np.clip((dp * dp + b * b - a * a) / (2.0 * dp * b), -1.0, 1.0)
        radicand = (-dp + a + b) * (dp + a - b) * (dp - a + b) * (dp + a + b)
        overlap = (
            a * a * np.arccos(arg1)
            + b * b * np.arccos(arg2)
            - 0.5 * np.sqrt(np.clip(radicand, 0.0, None))
        )
        partial_visible = 1.0 - overlap / (np.pi * b * b)
    visible = np.where(partial, partial_visible, visible)
    # A point source is either strictly inside the occluder or fully visible.
    point_visible = np.where(d < r1, 0.0, 1.0)
    return np.clip(np.where(valid_source, visible, point_visible), 0.0, 1.0)


def apparent_disk_geometry(
    evaluation_position_from_occluder: ArrayLike,
    source_position_from_occluder: ArrayLike,
    occluder_radius: float,
    source_radius: float,
) -> ApparentDiskGeometry:
    """Return apparent source/occluder angular radii and center separation."""

    evaluation = np.asarray(evaluation_position_from_occluder, dtype=float)
    source = np.asarray(source_position_from_occluder, dtype=float)
    evaluation, source = np.broadcast_arrays(evaluation, source)
    if evaluation.shape[-1] != 3:
        raise ValueError("position arrays must end in dimension 3")
    occ_distance = np.linalg.norm(evaluation, axis=-1)
    to_source = source - evaluation
    source_distance = np.linalg.norm(to_source, axis=-1)
    if np.any(occ_distance <= 0.0) or np.any(source_distance <= 0.0):
        raise ValueError("evaluation point must differ from source and occluder centers")
    occ_direction = -evaluation / occ_distance[..., None]
    source_direction = to_source / source_distance[..., None]
    separation = np.arccos(np.clip(np.sum(occ_direction * source_direction, axis=-1), -1.0, 1.0))
    return ApparentDiskGeometry(
        occluder_angular_radius_rad=apparent_angular_radius(occluder_radius, occ_distance),
        source_angular_radius_rad=apparent_angular_radius(source_radius, source_distance),
        separation_rad=separation,
        occluder_distance=occ_distance,
        source_distance=source_distance,
    )


def finite_source_visibility(
    evaluation_position_from_occluder: ArrayLike,
    source_position_from_occluder: ArrayLike,
    occluder_radius: float,
    source_radius: float,
) -> np.ndarray:
    """Uniform-disc visible source fraction at one or many evaluation points."""

    geometry = apparent_disk_geometry(
        evaluation_position_from_occluder,
        source_position_from_occluder,
        occluder_radius,
        source_radius,
    )
    return circle_overlap_visible_fraction(
        geometry.occluder_angular_radius_rad,
        geometry.source_angular_radius_rad,
        geometry.separation_rad,
    )


@lru_cache(maxsize=16)
def _radial_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _blocked_ring_fraction(rho: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    rho_b, q_b, s_b = np.broadcast_arrays(rho, q, s)
    blocked = np.zeros_like(rho_b, dtype=float)
    centered = np.abs(s_b) <= 1.0e-14
    blocked = np.where(centered & (rho_b <= q_b), 1.0, blocked)
    general = ~centered
    denom = 2.0 * np.maximum(rho_b * s_b, np.finfo(float).tiny)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        c = (rho_b * rho_b + s_b * s_b - q_b * q_b) / denom
    blocked = np.where(general & (c <= -1.0), 1.0, blocked)
    partial = general & (c > -1.0) & (c < 1.0)
    blocked = np.where(partial, np.arccos(np.clip(c, -1.0, 1.0)) / np.pi, blocked)
    return np.clip(blocked, 0.0, 1.0)


def limb_darkened_visibility_fraction(
    occluder_angular_radius_rad: ArrayLike,
    source_angular_radius_rad: ArrayLike,
    separation_rad: ArrayLike,
    *,
    law: str | LimbDarkeningLaw | None = None,
    quadrature_order: int = 48,
) -> np.ndarray:
    """Visible source-flux fraction under a radial limb-darkening law."""

    model = resolve_limb_darkening(law)
    r_occ, r_source, sep = np.broadcast_arrays(
        np.asarray(occluder_angular_radius_rad, dtype=float),
        np.asarray(source_angular_radius_rad, dtype=float),
        np.asarray(separation_rad, dtype=float),
    )
    if model.kind == "uniform":
        return circle_overlap_visible_fraction(r_occ, r_source, sep)
    valid = np.isfinite(r_occ) & np.isfinite(r_source) & np.isfinite(sep) & (r_source > 0.0)
    q = np.where(valid, np.maximum(r_occ, 0.0) / np.maximum(r_source, 1.0e-30), 0.0)
    s = np.where(valid, np.maximum(sep, 0.0) / np.maximum(r_source, 1.0e-30), 0.0)
    result = np.ones(r_occ.shape, dtype=float)
    no_overlap = valid & (s >= q + 1.0)
    total = valid & (q >= 1.0) & (s <= q - 1.0)
    result = np.where(total, 0.0, result)
    active = valid & ~no_overlap & ~total
    if not np.any(active):
        return np.where(valid, result, 1.0)

    rho, weights = _radial_quadrature(int(quadrature_order))
    reshape = (1,) * r_occ.ndim + (len(rho),)
    rho_view = rho.reshape(reshape)
    mu = np.sqrt(np.clip(1.0 - rho_view * rho_view, 0.0, 1.0))
    radial_weight = weights.reshape(reshape) * rho_view * model.intensity(mu)
    blocked = _blocked_ring_fraction(rho_view, q[..., None], s[..., None])
    total_flux = np.sum(radial_weight, axis=-1)
    visible_flux = np.sum(radial_weight * (1.0 - blocked), axis=-1)
    result = np.where(active, visible_flux / np.maximum(total_flux, np.finfo(float).tiny), result)
    return np.clip(np.where(valid, result, 1.0), 0.0, 1.0)


def finite_source_irradiance(
    evaluation_position_from_occluder: ArrayLike,
    source_position_from_occluder: ArrayLike,
    occluder_radius: float,
    source_radius: float,
    *,
    law: str | LimbDarkeningLaw | None = None,
    quadrature_order: int = 48,
) -> np.ndarray:
    """Visible finite-source irradiance fraction at physical points."""

    geometry = apparent_disk_geometry(
        evaluation_position_from_occluder,
        source_position_from_occluder,
        occluder_radius,
        source_radius,
    )
    return limb_darkened_visibility_fraction(
        geometry.occluder_angular_radius_rad,
        geometry.source_angular_radius_rad,
        geometry.separation_rad,
        law=law,
        quadrature_order=quadrature_order,
    )


def ray_sphere_intersections(
    origin: ArrayLike,
    direction: ArrayLike,
    radius: float,
    *,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Stable parametric ray roots for a sphere, sorted ascending.

    A closest-approach/chord construction avoids catastrophic cancellation
    when the ray origin is astronomical distances from a small target.
    Returned roots satisfy ``origin + t * unit(direction)``; misses are ``nan``.
    """

    o, d, c = np.broadcast_arrays(
        np.asarray(origin, dtype=float),
        np.asarray(direction, dtype=float),
        np.asarray(center, dtype=float),
    )
    if o.shape[-1] != 3:
        raise ValueError("origin, direction, and center must end in dimension 3")
    d = _unit(d, name="ray direction")
    oc = o - c
    along = -np.sum(oc * d, axis=-1)
    closest = oc + along[..., None] * d
    miss2 = np.sum(closest * closest, axis=-1)
    radius2 = float(radius) ** 2
    hit = miss2 <= radius2 * (1.0 + 4.0 * np.finfo(float).eps)
    half = np.sqrt(np.clip(radius2 - miss2, 0.0, None))
    roots = np.stack([along - half, along + half], axis=-1)
    return np.where(hit[..., None], np.sort(roots, axis=-1), np.nan)


def ray_ellipsoid_intersections(
    origin: ArrayLike,
    direction: ArrayLike,
    axes: ArrayLike,
    *,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Stable parametric ray roots for an axis-aligned ellipsoid."""

    o, d, c = np.broadcast_arrays(
        np.asarray(origin, dtype=float),
        np.asarray(direction, dtype=float),
        np.asarray(center, dtype=float),
    )
    axis = np.asarray(axes, dtype=float)
    if o.shape[-1] != 3 or axis.shape != (3,):
        raise ValueError("vectors must end in dimension 3 and axes must have shape (3,)")
    if np.any(axis <= 0.0):
        raise ValueError("ellipsoid axes must be positive")
    d = _unit(d, name="ray direction")
    q = (o - c) / axis
    v = d / axis
    vv = np.sum(v * v, axis=-1)
    along = -np.sum(q * v, axis=-1) / vv
    closest = q + along[..., None] * v
    miss2 = np.sum(closest * closest, axis=-1)
    hit = miss2 <= 1.0 + 2.0e-12
    half = np.sqrt(np.clip(1.0 - miss2, 0.0, None) / vv)
    roots = np.stack([along - half, along + half], axis=-1)
    return np.where(hit[..., None], np.sort(roots, axis=-1), np.nan)


def first_positive_intersection(roots: ArrayLike, *, minimum: float = 0.0) -> np.ndarray:
    """Return the first finite root at or beyond ``minimum``; otherwise ``nan``."""

    values = np.asarray(roots, dtype=float)
    if values.shape[-1] < 1:
        raise ValueError("roots must have a final root dimension")
    candidates = np.where(np.isfinite(values) & (values >= float(minimum)), values, np.inf)
    first = np.min(candidates, axis=-1)
    return np.where(np.isfinite(first), first, np.nan)


@dataclass(frozen=True)
class ShadowCone:
    """Finite-source umbra/antumbra and penumbra geometry."""

    source_radius: float
    occluder_radius: float
    source_occluder_distance: float
    umbra_slope: float
    penumbra_slope: float
    umbra_apex_distance: float

    def umbra_radius(self, downstream_distance: ArrayLike, *, signed: bool = True) -> np.ndarray:
        values = self.occluder_radius - self.umbra_slope * np.asarray(downstream_distance, dtype=float)
        return values if signed else np.abs(values)

    def penumbra_radius(self, downstream_distance: ArrayLike) -> np.ndarray:
        return self.occluder_radius + self.penumbra_slope * np.asarray(downstream_distance, dtype=float)


@dataclass(frozen=True)
class ShadowCrossSection:
    downstream_distance: np.ndarray
    umbra_radius: np.ndarray
    penumbra_radius: np.ndarray
    regime: np.ndarray


def shadow_cone(source_radius: float, occluder_radius: float, source_occluder_distance: float) -> ShadowCone:
    """Construct exact similar-triangle shadow slopes for spherical bodies."""

    rs, ro, distance = map(float, (source_radius, occluder_radius, source_occluder_distance))
    if rs <= 0.0 or ro <= 0.0 or distance <= 0.0:
        raise ValueError("radii and source-occluder distance must be positive")
    umbra_slope = (rs - ro) / distance
    penumbra_slope = (rs + ro) / distance
    apex = np.inf if abs(umbra_slope) <= np.finfo(float).tiny else ro / umbra_slope
    return ShadowCone(rs, ro, distance, umbra_slope, penumbra_slope, float(apex))


def shadow_cross_section(cone: ShadowCone, downstream_distance: ArrayLike) -> ShadowCrossSection:
    """Evaluate umbral/antumbral and penumbral radii downstream of the occluder."""

    distance = np.asarray(downstream_distance, dtype=float)
    signed_umbra = cone.umbra_radius(distance, signed=True)
    regime = np.where(signed_umbra >= 0.0, "umbra", "antumbra")
    return ShadowCrossSection(
        downstream_distance=distance,
        umbra_radius=np.abs(signed_umbra),
        penumbra_radius=cone.penumbra_radius(distance),
        regime=regime,
    )


def orthonormal_basis(axis: ArrayLike, *, reference: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable right-handed ``(axis, u, v)`` basis."""

    w = _unit(np.asarray(axis, dtype=float), name="axis")
    if w.shape != (3,):
        raise ValueError("axis must be a single 3-vector")
    if reference is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    else:
        ref = _unit(np.asarray(reference, dtype=float), name="reference")
    u = np.cross(ref, w)
    if np.linalg.norm(u) <= 1.0e-12:
        ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(ref, w)
    u = _unit(u, name="basis u")
    v = _unit(np.cross(w, u), name="basis v")
    return w, u, v


@dataclass(frozen=True)
class PhotosphereSamples:
    """Deterministic projected-disc quadrature samples on a spherical source."""

    points: np.ndarray
    weights: np.ndarray
    mu: np.ndarray
    disk_xy: np.ndarray


def sample_spherical_photosphere(
    source_center: ArrayLike,
    source_radius: float,
    target_center: ArrayLike,
    *,
    count: int = 61,
    law: str | LimbDarkeningLaw | None = None,
) -> PhotosphereSamples:
    """Sample the target-facing apparent disc of a spherical light source.

    Samples use a deterministic golden-angle distribution that is uniform in
    projected disc area.  Returned weights include the selected limb-darkening
    law and sum exactly to one (within floating-point arithmetic).
    """

    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    center = np.asarray(source_center, dtype=float)
    target = np.asarray(target_center, dtype=float)
    if center.shape != (3,) or target.shape != (3,):
        raise ValueError("source_center and target_center must be 3-vectors")
    forward, u, v = orthonormal_basis(target - center)
    i = np.arange(count, dtype=float)
    rho = np.sqrt((i + 0.5) / count)
    theta = i * (np.pi * (3.0 - np.sqrt(5.0)))
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    mu = np.sqrt(np.clip(1.0 - rho * rho, 0.0, 1.0))
    normals = x[:, None] * u + y[:, None] * v + mu[:, None] * forward
    points = center[None, :] + float(source_radius) * normals
    weights = resolve_limb_darkening(law).intensity(mu)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("photospheric quadrature weights sum to zero")
    weights = weights / total
    return PhotosphereSamples(
        points=points,
        weights=weights,
        mu=mu,
        disk_xy=np.column_stack([x, y]),
    )


# Compatibility aliases used by the existing eclipse renderer.
_circle_overlap_fraction = circle_overlap_visible_fraction


__all__ = [
    "ApparentDiskGeometry",
    "LimbDarkeningLaw",
    "UNIFORM_DISC",
    "VISIBLE_LINEAR",
    "VISIBLE_QUADRATIC",
    "ShadowCone",
    "ShadowCrossSection",
    "PhotosphereSamples",
    "resolve_limb_darkening",
    "apparent_angular_radius",
    "circle_overlap_visible_fraction",
    "apparent_disk_geometry",
    "finite_source_visibility",
    "limb_darkened_visibility_fraction",
    "finite_source_irradiance",
    "ray_sphere_intersections",
    "ray_ellipsoid_intersections",
    "first_positive_intersection",
    "shadow_cone",
    "shadow_cross_section",
    "orthonormal_basis",
    "sample_spherical_photosphere",
]
