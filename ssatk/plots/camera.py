"""Deterministic animation shots in the coordinates of the displayed scene.

Cameras change presentation only; they never modify the input trajectory.
Distances and offsets use the renderer's scene units (see camera_modes.rst).
"""

from dataclasses import dataclass as _dataclass
import numpy as np

__all__ = ["AnimationCamera"]


@_dataclass(frozen=True)
class AnimationCamera:
    """A shared shot: overview, follow, chase, or orbit.

    ``target`` selects a trajectory when several are supplied. ``distance``
    is eye-to-target distance; ``offset`` gives a fixed scene-axis vector for
    follow (and the initial orbit angle). ``turns`` sets the orbit sweep over
    the full input time span, not per frame. Chase follows the sampled path
    tangent, not an inferred spacecraft attitude or inertial velocity.
    """

    mode: str = "overview"
    target: int = 0
    distance: float | None = None
    offset: tuple | None = None
    up: tuple = (0.0, 0.0, 1.0)
    elevation: float = 25.0
    azimuth: float = 35.0
    turns: float = 1.0

    def __post_init__(self):
        if self.mode not in {"overview", "follow", "chase", "orbit"}:
            raise ValueError("camera mode must be overview, follow, chase, or orbit")
        if (
            isinstance(self.target, bool)
            or not isinstance(self.target, (int, np.integer))
            or self.target < 0
        ):
            raise ValueError("camera target must be a nonnegative trajectory index")
        if self.distance is not None and (
            not np.isfinite(self.distance) or self.distance <= 0
        ):
            raise ValueError("camera distance must be positive and finite")
        if self.up is None:
            raise ValueError("camera up must be a finite nonzero three-vector")
        for name in ("offset", "up"):
            value = getattr(self, name)
            if value is not None:
                a = np.asarray(value, float)
                if (
                    a.shape != (3,)
                    or not np.isfinite(a).all()
                    or np.linalg.norm(a) == 0
                ):
                    raise ValueError(
                        f"camera {name} must be a finite nonzero three-vector"
                    )
                object.__setattr__(self, name, tuple(a))
        if not np.isfinite([self.elevation, self.azimuth, self.turns]).all():
            raise ValueError("camera angles and turns must be finite")
        if not -90 <= self.elevation <= 90:
            raise ValueError("camera elevation must lie between -90 and 90 degrees")
        if self.offset is not None and self.distance is not None:
            raise ValueError("specify camera offset or distance, not both")
        if self.offset is not None and self.mode == "chase":
            raise ValueError("chase uses distance; a fixed offset belongs to follow")


def _camera_spec(camera):
    if camera is None:
        return AnimationCamera()
    if isinstance(camera, AnimationCamera):
        return camera
    if isinstance(camera, str):
        return AnimationCamera(mode=camera)
    if isinstance(camera, dict):
        return AnimationCamera(**camera)
    raise TypeError(
        "camera must be a shot name, AnimationCamera, or configuration dict"
    )


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class _CameraPath:
    """Eye and target vectors; all geometry and times are caller-owned units."""

    def __init__(
        self,
        camera,
        positions,
        bounds,
        *,
        times=None,
        default_distance=None,
        phase_range=None,
    ):
        self.spec = spec = _camera_spec(camera)
        p = np.asarray(positions, float)
        bounds = np.asarray(bounds, float)
        if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
            raise ValueError("camera positions must be a finite nonempty (N, 3) track")
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[1] < bounds[0])
        ):
            raise ValueError("camera bounds must be finite lower/upper three-vectors")
        t = (
            np.arange(len(p), dtype=float)
            if times is None
            else np.asarray(times, float)
        )
        if t.shape != (len(p),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("camera times must be finite and strictly increasing")
        start, stop = (t[0], t[-1]) if phase_range is None else phase_range
        phase = (t - start) / (stop - start) if stop > start else np.zeros(len(t))
        center = bounds.mean(axis=0)
        self.radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) / 2), 1e-12)
        self.up = _unit(np.asarray(spec.up))
        el, az = np.deg2rad([spec.elevation, spec.azimuth])
        direction = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        distance = spec.distance
        if distance is None:
            distance = (
                3.2 * self.radius
                if spec.mode in {"overview", "orbit"}
                else (default_distance if default_distance is not None else self.radius)
            )
        offset = (
            direction * distance if spec.offset is None else np.asarray(spec.offset)
        )
        self.distance = float(np.linalg.norm(offset))
        self.targets = (
            np.tile(center, (len(p), 1))
            if spec.mode in {"overview", "orbit"}
            else p.copy()
        )
        offsets = np.tile(offset, (len(p), 1))
        if spec.mode == "orbit":
            # Rodrigues rotation about the requested scene up axis.
            angle = 2 * np.pi * spec.turns * phase[:, None]
            offsets = (
                offset * np.cos(angle)
                + np.cross(self.up, offset) * np.sin(angle)
                + self.up * np.dot(self.up, offset) * (1 - np.cos(angle))
            )
        elif spec.mode == "chase":
            if len(p) < 2:
                raise ValueError("chase camera requires at least two moving samples")
            tangent = np.gradient(p, t - t[0], axis=0)
            moving = np.linalg.norm(tangent, axis=1) > 1e-12
            if not moving.any():
                raise ValueError(
                    "chase camera requires a moving target; use follow for a stationary target"
                )
            last = tangent[np.flatnonzero(moving)[0]]
            for i in range(len(p)):
                if moving[i]:
                    last = tangent[i]
                tangent[i] = last
            offsets = _unit(-_unit(tangent) + 0.3 * self.up) * self.distance
        self.eyes = self.targets + offsets
        self.ups = np.tile(self.up, (len(p), 1))
        direction = _unit(offsets)
        for i in np.flatnonzero(np.abs(direction @ self.up) > 1 - 1e-10):
            self.ups[i] = np.eye(3)[np.argmin(np.abs(direction[i]))]

    def mpl(self, ax, index):
        """Apply an equal-scale orthographic view; no perspective unit ambiguity."""
        target, eye = self.targets[index], self.eyes[index]
        direction = _unit(eye - target)
        # In overview/orbit all corners of the bounding cube fit at every angle.
        half = (
            1.08 * self.radius
            if self.spec.mode in {"overview", "orbit"}
            else self.distance / 2
        )
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), target):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.8)
        ax.set_proj_type("ortho")
        elev = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))
        azim = np.degrees(np.arctan2(direction[1], direction[0]))
        # Matplotlib's roll aligns the projected requested up vector.
        right = np.array([-np.sin(np.deg2rad(azim)), np.cos(np.deg2rad(azim)), 0.0])
        screen_up = np.cross(direction, right)
        roll = np.degrees(
            np.arctan2(
                np.dot(self.ups[index], right), np.dot(self.ups[index], screen_up)
            )
        )
        ax.view_init(elev=elev, azim=azim, roll=roll)
        return elev, azim


def _mpl_camera(camera, ax, positions, *, times=None, default_distance=None):
    bounds = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).T
    return _CameraPath(
        camera, positions, bounds, times=times, default_distance=default_distance
    )


def _plotly_camera(fig, camera, positions, *, times=None, scene="scene"):
    """Attach full camera states to each frame using cube-normalized coordinates.

    Keep fixed equal data ranges so the physical aspect ratio and camera
    translation agree. A moving target is supplied explicitly by each plot,
    never guessed from surface vertices or labels.
    """
    spec = _camera_spec(camera)
    if spec.target != 0:
        raise ValueError("this plot has one camera target (index 0)")
    layout = fig.layout[scene]
    if camera is None or camera == "overview":
        # Keep each diagnostic's deliberate viewing side (e.g. lunar eclipse
        # geometry must not put the Moon behind Earth).
        eye = layout.camera.eye
        if all(eye[k] is not None for k in "xyz"):
            from dataclasses import replace

            direction = _unit(np.array([eye[k] for k in "xyz"]))
            spec = replace(
                spec,
                elevation=float(np.degrees(np.arcsin(direction[2]))),
                azimuth=float(np.degrees(np.arctan2(direction[1], direction[0]))),
            )
    ranges = np.asarray(
        [getattr(layout, axis).range for axis in ("xaxis", "yaxis", "zaxis")], float
    )
    if ranges.shape != (3, 2) or not np.isfinite(ranges).all():
        raise ValueError("animated camera requires explicit finite scene axis ranges")
    center = ranges.mean(axis=1)
    half = max(float(np.max(np.diff(ranges, axis=1))) / 2, 1e-12)
    bounds = np.array([center - half, center + half])
    path = _CameraPath(spec, positions, bounds, times=times)
    if len(fig.frames) != len(path.eyes):
        raise ValueError("camera target samples must match animation frames")
    layout.aspectmode = "cube"
    for axis, c in zip(("xaxis", "yaxis", "zaxis"), center):
        getattr(layout, axis).update(range=[c - half, c + half], autorange=False)

    def state(i):
        # Plotly cube coordinates span [-.5, .5] on each axis.
        eye = (path.eyes[i] - center) / (2 * half)
        target = (path.targets[i] - center) / (2 * half)
        return dict(
            eye=dict(zip("xyz", eye)),
            center=dict(zip("xyz", target)),
            up=dict(zip("xyz", path.ups[i])),
            projection=dict(type="perspective"),
        )

    layout.camera = state(0)
    for i, frame in enumerate(fig.frames):
        frame.layout[scene] = dict(camera=state(i))
    return path
