"""Minimal conjunction screening and encounter-plane probability tools.

Trajectories use relative seconds and GCRF SI coordinates. Relative states are
defined consistently as object B minus object A. The collision probability is
the numerical encounter-plane Gaussian disk integral described by Patera (2001),
not an implementation of Patera's published algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ConjunctionCandidate:
    """One coarse interval whose linearized miss distance is below a threshold."""

    t_start: float
    t_end: float
    t_min: float
    minimum_distance: float

    def __post_init__(self):
        start, end = float(self.t_start), float(self.t_end)
        minimum_time = float(self.t_min)
        distance = float(self.minimum_distance)
        if not all(np.isfinite(value) for value in (start, end, minimum_time, distance)):
            raise ValueError("candidate times and distance must be finite.")
        if end < start:
            raise ValueError("candidate bracket end must be no earlier than its start.")
        if not start <= minimum_time <= end:
            raise ValueError("candidate minimum time must lie inside its bracket.")
        if distance < 0.0:
            raise ValueError("candidate minimum distance must be nonnegative.")
        object.__setattr__(self, "t_start", start)
        object.__setattr__(self, "t_end", end)
        object.__setattr__(self, "t_min", minimum_time)
        object.__setattr__(self, "minimum_distance", distance)

    @property
    def bracket(self) -> tuple[float, float]:
        return self.t_start, self.t_end

    @property
    def miss_distance(self) -> float:
        return self.minimum_distance


@dataclass(frozen=True)
class ClosestApproach:
    """Refined closest approach in the B-minus-A relative convention."""

    tca: float
    miss_distance: float
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    bracket: tuple[float, float]

    def __post_init__(self):
        position = np.array(self.relative_position, dtype=float, copy=True)
        velocity = np.array(self.relative_velocity, dtype=float, copy=True)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("relative position and velocity must have shape (3,).")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("relative position and velocity must be finite.")
        if not np.isfinite(float(self.tca)) or not np.isfinite(float(self.miss_distance)):
            raise ValueError("tca and miss_distance must be finite.")
        if len(self.bracket) != 2 or not np.all(np.isfinite(self.bracket)):
            raise ValueError("bracket must contain two finite times.")
        if self.bracket[1] < self.bracket[0]:
            raise ValueError("bracket end must be no earlier than its start.")
        position.flags.writeable = False
        velocity.flags.writeable = False
        object.__setattr__(self, "relative_position", position)
        object.__setattr__(self, "relative_velocity", velocity)
        object.__setattr__(self, "tca", float(self.tca))
        object.__setattr__(self, "miss_distance", float(self.miss_distance))
        object.__setattr__(self, "bracket", (float(self.bracket[0]), float(self.bracket[1])))

    @property
    def time(self) -> float:
        return self.tca

    @property
    def distance(self) -> float:
        return self.miss_distance


@dataclass(frozen=True)
class CatalogConjunctionEvent:
    """A refined conjunction for an ordered pair of catalog object IDs."""

    object_id_a: object
    object_id_b: object
    closest_approach: ClosestApproach


@dataclass(frozen=True)
class _Trajectory:
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray


def _trajectory(value) -> _Trajectory:
    try:
        t = np.asarray(value.t, dtype=float)
        r = np.asarray(value.r, dtype=float)
        v = np.asarray(value.v, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trajectory must expose finite t, r, and v arrays.") from exc
    if t.ndim != 1 or t.size < 2 or r.shape != (t.size, 3) or v.shape != (t.size, 3):
        raise ValueError("trajectory t must be 1-D and r/v must have shape (N, 3).")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError("trajectory t, r, and v must be finite.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("trajectory times must be strictly increasing.")
    return _Trajectory(t, r, v)


def _time_grid(first: _Trajectory, second: _Trajectory) -> np.ndarray:
    start = max(first.t[0], second.t[0])
    end = min(first.t[-1], second.t[-1])
    if end < start:
        return np.empty(0, dtype=float)
    if end == start:
        return np.array([start], dtype=float)
    return np.unique(
        np.concatenate(([start, end], first.t[(first.t > start) & (first.t < end)], second.t[(second.t > start) & (second.t < end)]))
    )


def _linear_position(trajectory: _Trajectory, time: float) -> np.ndarray:
    index = min(np.searchsorted(trajectory.t, time, side="right") - 1, trajectory.t.size - 2)
    fraction = (time - trajectory.t[index]) / (trajectory.t[index + 1] - trajectory.t[index])
    return trajectory.r[index] + fraction * (trajectory.r[index + 1] - trajectory.r[index])


def _hermite_state(trajectory: _Trajectory, time: float) -> tuple[np.ndarray, np.ndarray]:
    index = min(np.searchsorted(trajectory.t, time, side="right") - 1, trajectory.t.size - 2)
    h = trajectory.t[index + 1] - trajectory.t[index]
    s = (time - trajectory.t[index]) / h
    p0, p1 = trajectory.r[index], trajectory.r[index + 1]
    v0, v1 = trajectory.v[index], trajectory.v[index + 1]
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    position = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    velocity = (
        ((6.0 * s**2 - 6.0 * s) / h) * p0
        + (3.0 * s**2 - 4.0 * s + 1.0) * v0
        + ((-6.0 * s**2 + 6.0 * s) / h) * p1
        + (3.0 * s**2 - 2.0 * s) * v1
    )
    return position, velocity


def _scalar_nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _linear_candidate(first, second, start, end, threshold):
    relative_start = _linear_position(second, start) - _linear_position(first, start)
    relative_end = _linear_position(second, end) - _linear_position(first, end)
    if end == start:
        minimum_time = start
        minimum = float(np.linalg.norm(relative_start))
    else:
        relative_velocity = (relative_end - relative_start) / (end - start)
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        tau = 0.0 if speed_squared == 0.0 else np.clip(
            -float(np.dot(relative_start, relative_velocity)) / speed_squared,
            0.0,
            end - start,
        )
        minimum_time = start + tau
        minimum = float(np.linalg.norm(relative_start + tau * relative_velocity))
    if minimum <= threshold:
        return ConjunctionCandidate(start, end, minimum_time, minimum)
    return None


def coarse_conjunction_screen(
    trajectory_a,
    trajectory_b,
    threshold,
) -> tuple[ConjunctionCandidate, ...]:
    """Screen overlapping trajectories using linear interval minima.

    Each returned bracket is one interval of the overlapping union time grid
    whose analytic constant-relative-velocity minimum is at or below
    ``threshold`` metres.
    """

    threshold = _scalar_nonnegative(threshold, "threshold")
    first, second = _trajectory(trajectory_a), _trajectory(trajectory_b)
    grid = _time_grid(first, second)
    if grid.size == 1:
        candidate = _linear_candidate(first, second, grid[0], grid[0], threshold)
        return () if candidate is None else (candidate,)
    candidates = []
    for start, end in pairwise(grid):
        candidate = _linear_candidate(first, second, start, end, threshold)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def refine_closest_approach(
    trajectory_a,
    trajectory_b,
    bracket: ConjunctionCandidate | tuple[float, float],
    *,
    xatol: float = 1.0e-9,
) -> ClosestApproach:
    """Refine a candidate with piecewise cubic Hermite interpolation."""

    first, second = _trajectory(trajectory_a), _trajectory(trajectory_b)
    if isinstance(bracket, ConjunctionCandidate):
        bounds = bracket.bracket
    else:
        try:
            if len(bracket) != 2:
                raise ValueError
            bounds = (float(bracket[0]), float(bracket[1]))
        except (IndexError, TypeError, ValueError):
            raise ValueError("bracket must contain two times.") from None
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
        raise ValueError("bracket must contain an ordered pair of finite times.")
    overlap_start = max(first.t[0], second.t[0])
    overlap_end = min(first.t[-1], second.t[-1])
    if bounds[0] < overlap_start or bounds[1] > overlap_end:
        raise ValueError("bracket must lie within the trajectory overlap.")
    xatol = float(xatol)
    if not np.isfinite(xatol) or xatol <= 0.0:
        raise ValueError("xatol must be finite and positive.")

    def state(time):
        position_a, velocity_a = _hermite_state(first, time)
        position_b, velocity_b = _hermite_state(second, time)
        return position_b - position_a, velocity_b - velocity_a

    def objective(time):
        return float(np.linalg.norm(state(float(time))[0]))

    internal_knots = np.concatenate(
        (
            first.t[(first.t > bounds[0]) & (first.t < bounds[1])],
            second.t[(second.t > bounds[0]) & (second.t < bounds[1])],
        )
    )
    split_points = np.unique(np.concatenate(([bounds[0], bounds[1]], internal_knots)))
    candidates = list(split_points)
    for start, end in pairwise(split_points):
        duration = end - start
        if duration == 0.0:
            continue
        position_start, velocity_start = state(start)
        position_end, velocity_end = state(end)
        coefficients = np.array(
            [
                position_start,
                duration * velocity_start,
                -3.0 * position_start
                + 3.0 * position_end
                - 2.0 * duration * velocity_start
                - duration * velocity_end,
                2.0 * position_start
                - 2.0 * position_end
                + duration * velocity_start
                + duration * velocity_end,
            ]
        )
        squared_distance = np.zeros(7, dtype=float)
        for coordinate in coefficients.T:
            product = np.polynomial.polynomial.polymul(coordinate, coordinate)
            squared_distance[: product.size] += product
        derivative = np.polynomial.polynomial.polyder(
            np.trim_zeros(squared_distance, trim="b")
        )
        if derivative.size > 1 and np.any(np.abs(derivative) > 0.0):
            roots = np.polynomial.polynomial.polyroots(derivative)
            root_tolerance = min(1.0e-7, max(1.0e-12, xatol / max(duration, 1.0)))
            for root in roots:
                if abs(root.imag) <= root_tolerance * max(1.0, abs(root.real)):
                    parameter = float(root.real)
                    if -root_tolerance <= parameter <= 1.0 + root_tolerance:
                        parameter = float(np.clip(parameter, 0.0, 1.0))
                        candidates.append(start + parameter * duration)
    tca = min(candidates, key=objective)
    relative_position, relative_velocity = state(tca)
    return ClosestApproach(tca, float(np.linalg.norm(relative_position)), relative_position, relative_velocity, bounds)


def catalog_conjunction_screen(
    catalog: Mapping,
    threshold,
    *,
    xatol: float = 1.0e-9,
) -> tuple[CatalogConjunctionEvent, ...]:
    """Return refined conjunctions from an insertion-ordered trajectory mapping.

    Object pairs retain mapping insertion order: the first ID is object A and
    the second is B, so each event uses the existing B-minus-A convention.
    Results are ordered by that pair order and then TCA. Adjacent qualifying
    brackets are one event when the pair remains within ``threshold`` at their
    shared boundary.
    """

    threshold = _scalar_nonnegative(threshold, "threshold")
    xatol = float(xatol)
    if not np.isfinite(xatol) or xatol <= 0.0:
        raise ValueError("xatol must be finite and positive.")
    if not isinstance(catalog, Mapping):
        raise TypeError("catalog must be a mapping of object IDs to trajectories.")
    items = tuple((object_id, _trajectory(trajectory)) for object_id, trajectory in catalog.items())
    if len(items) < 2:
        return ()

    segment_objects = np.concatenate(
        [np.full(trajectory.t.size - 1, index, dtype=int) for index, (_, trajectory) in enumerate(items)]
    )
    segment_starts = np.concatenate([trajectory.t[:-1] for _, trajectory in items])
    segment_ends = np.concatenate([trajectory.t[1:] for _, trajectory in items])
    position_starts = np.concatenate([trajectory.r[:-1] for _, trajectory in items])
    position_ends = np.concatenate([trajectory.r[1:] for _, trajectory in items])
    centers = 0.5 * (position_starts + position_ends)
    radii = 0.5 * np.linalg.norm(position_ends - position_starts, axis=1)
    segment_pairs = cKDTree(centers).query_pairs(
        threshold + 2.0 * float(np.max(radii)), output_type="ndarray"
    )
    candidates_by_pair = {}
    for segment_a, segment_b in segment_pairs:
        first, second = sorted((segment_objects[segment_a], segment_objects[segment_b]))
        if first == second:
            continue
        start = max(segment_starts[segment_a], segment_starts[segment_b])
        end = min(segment_ends[segment_a], segment_ends[segment_b])
        if end < start or np.linalg.norm(centers[segment_a] - centers[segment_b]) > (
            radii[segment_a] + radii[segment_b] + threshold
        ):
            continue
        candidate = _linear_candidate(items[first][1], items[second][1], start, end, threshold)
        if candidate is not None:
            candidates_by_pair.setdefault((first, second), []).append(candidate)

    events = []
    for (first, second), candidates in sorted(candidates_by_pair.items()):
        brackets = []
        for candidate in sorted(candidates, key=lambda value: value.bracket):
            if brackets and candidate.t_start <= brackets[-1][1]:
                position_a, _ = _hermite_state(items[first][1], candidate.t_start)
                position_b, _ = _hermite_state(items[second][1], candidate.t_start)
                if np.linalg.norm(position_b - position_a) <= threshold:
                    brackets[-1] = (brackets[-1][0], max(brackets[-1][1], candidate.t_end))
                    continue
            brackets.append(candidate.bracket)
        for bracket in brackets:
            closest = refine_closest_approach(items[first][1], items[second][1], bracket, xatol=xatol)
            if closest.miss_distance <= threshold:
                events.append(CatalogConjunctionEvent(items[first][0], items[second][0], closest))
    return tuple(events)


def encounter_frame(relative_position, relative_velocity, *, speed_tolerance: float = 0.0) -> np.ndarray:
    """Return an orthonormal 3x2 basis for the encounter plane.

    The third axis is the relative-velocity direction. The first axis follows
    the projected B-minus-A miss vector when defined; otherwise a deterministic
    least-aligned Cartesian axis supplies the first direction.
    """

    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("relative position and velocity must be finite three-vectors.")
    speed = float(np.linalg.norm(velocity))
    speed_tolerance = _scalar_nonnegative(speed_tolerance, "speed_tolerance")
    if speed <= speed_tolerance:
        raise ValueError("relative speed is too small to define an encounter plane.")
    normal = velocity / speed
    # The double cross product stays perpendicular near collinearity.
    projected = np.cross(normal, np.cross(position, normal))
    projected_norm = float(np.linalg.norm(projected))
    # A projection at roundoff scale does not define a reliable miss direction.
    projection_tolerance = 8.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(position)))
    if projected_norm > projection_tolerance:
        first = projected / projected_norm
    else:
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = axis - normal * float(np.dot(normal, axis))
        first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    basis = np.column_stack((first, second))
    return basis


def _covariance(value, name):
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 3x3 covariance in m^2.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(covariance) < -1.0e-12):
        raise ValueError(f"{name} must be positive semidefinite.")
    return covariance


def relative_encounter_covariance(
    covariance_a,
    covariance_b,
    frame,
    *,
    cross_covariance=None,
) -> np.ndarray:
    """Project relative B-minus-A position covariance into an encounter plane."""

    basis = np.asarray(frame, dtype=float)
    if basis.shape != (3, 2) or not np.all(np.isfinite(basis)):
        raise ValueError("frame must be a finite 3x2 basis.")
    if not np.allclose(basis.T @ basis, np.eye(2), rtol=0.0, atol=1.0e-10):
        raise ValueError("frame columns must be orthonormal.")
    covariance_a = _covariance(covariance_a, "covariance_a")
    covariance_b = _covariance(covariance_b, "covariance_b")
    relative = covariance_a + covariance_b
    if cross_covariance is not None:
        cross_covariance = np.asarray(cross_covariance, dtype=float)
        if cross_covariance.shape != (3, 3) or not np.all(np.isfinite(cross_covariance)):
            raise ValueError("cross_covariance must be a finite 3x3 matrix in m^2.")
        joint = np.block(
            [[covariance_a, cross_covariance], [cross_covariance.T, covariance_b]]
        )
        if np.any(np.linalg.eigvalsh(joint) < -1.0e-10):
            raise ValueError("joint covariance with cross_covariance must be positive semidefinite.")
        relative = relative - cross_covariance - cross_covariance.T
    relative = 0.5 * (relative + relative.T)
    if np.any(np.linalg.eigvalsh(relative) < -1.0e-10):
        raise ValueError("relative covariance must be positive semidefinite.")
    projected = basis.T @ relative @ basis
    return 0.5 * (projected + projected.T)


def probability_of_collision(
    mean,
    covariance,
    hard_body_radius,
    *,
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-10,
) -> float:
    """Integrate a 2-D Gaussian over a circular hard-body disk.

    This is a numerical encounter-plane Gaussian integration in the Patera
    (2001) context. ``mean`` is in metres and ``covariance`` in m^2.
    """

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    radius = _scalar_nonnegative(hard_body_radius, "hard_body_radius")
    if mean.shape != (2,) or not np.all(np.isfinite(mean)):
        raise ValueError("mean must be a finite two-vector in metres.")
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite 2x2 matrix in m^2.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("covariance must be symmetric.")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive-definite.") from exc
    epsabs, epsrel = float(epsabs), float(epsrel)
    if not np.isfinite(epsabs) or epsabs <= 0.0 or not np.isfinite(epsrel) or epsrel <= 0.0:
        raise ValueError("epsabs and epsrel must be finite and positive.")
    if radius == 0.0:
        return 0.0
    normalization = np.exp(-np.log(2.0 * np.pi) - np.log(cholesky[0, 0]) - np.log(cholesky[1, 1]))

    def angular_integrand(angle):
        unit = np.array([np.cos(angle), np.sin(angle)])

        def radial_integrand(distance):
            whitened = np.linalg.solve(cholesky, distance * unit - mean)
            return distance * normalization * np.exp(-0.5 * float(np.dot(whitened, whitened)))

        return quad(radial_integrand, 0.0, radius, epsabs=epsabs, epsrel=epsrel)[0]

    probability = quad(angular_integrand, 0.0, 2.0 * np.pi, epsabs=epsabs, epsrel=epsrel)[0]
    return float(np.clip(probability, 0.0, 1.0))


__all__ = [
    "CatalogConjunctionEvent",
    "ClosestApproach",
    "ConjunctionCandidate",
    "catalog_conjunction_screen",
    "coarse_conjunction_screen",
    "encounter_frame",
    "probability_of_collision",
    "refine_closest_approach",
    "relative_encounter_covariance",
]
