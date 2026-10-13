"""View-dependent clipping of plotted geometry by opaque spherical bodies.

The sphere, orbit, and camera coordinates must use the same frame and units.
Clipping is a presentation operation; input states and artist data are restored
after each draw. Matplotlib has no per-pixel depth buffer, so sorting whole
artists cannot correctly render an orbit that passes both behind and in front
of a body. These helpers split its segments at the actual sphere/limb crossings.
"""
from dataclasses import dataclass
from types import MethodType
import weakref

import numpy as np

__all__ = ["set_orbit_occlusion"]


def _dot(a, b):
    return np.sum(a*b, axis=-1)


def _visible(points, bodies, *, direction=None, eye=None):
    """Visibility along a ray towards an orthographic or perspective camera."""
    points = np.asarray(points, dtype=float)
    visible = np.all(np.isfinite(points), axis=-1)
    for center, radius in bodies:
        q = points-center
        ray = np.broadcast_to(np.asarray(direction, dtype=float), points.shape) if eye is None else eye-points
        length = _dot(ray, ray)
        # Closest point on the ray, bounded at the eye for perspective views.
        along = np.divide(-_dot(q, ray), length, out=np.zeros_like(length), where=length > 0)
        along = np.maximum(along, 0) if eye is None else np.clip(along, 0, 1)
        closest = q+along[..., None]*ray
        # A tangent is visible. A relative roundoff margin also retains points
        # on the near surface (e.g. ground stations).
        visible &= _dot(closest, closest) >= radius**2*(1-32*np.finfo(float).eps)
    return visible


def _roots(a, b, c):
    """Vectorized real quadratic roots; roots outside [0,1] are harmless cuts."""
    scale = np.maximum.reduce((np.abs(a), np.abs(b), np.abs(c)))
    eps = 32*np.finfo(float).eps*scale
    quadratic = np.abs(a) > eps
    discriminant = b*b-4*a*c
    valid = quadratic & (discriminant >= 0)
    # Stable form preserves roots close to a segment endpoint.
    q = -.5*(b+np.copysign(np.sqrt(np.maximum(discriminant, 0)), b))
    first = np.divide(q, a, out=np.ones_like(a), where=valid)
    second = np.divide(c, q, out=first.copy(), where=valid & (q != 0))
    linear = ~quadratic & (np.abs(b) > eps)
    first[linear] = -c[linear]/b[linear]
    return np.clip(np.column_stack((first, second)), 0, 1)


def _clip_segments(segments, bodies, *, direction=None, eye=None):
    """Return visible subsegments and their original indices (for colours).

    Cuts are analytic sphere and shadow-cylinder/cone intersections, not
    sample-point tests. Thus a coarse segment with two visible endpoints can
    still have its hidden middle removed. NaN-separated tracks stay separate.
    """
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    indices = np.flatnonzero(np.all(np.isfinite(segments), axis=(1, 2)))
    segments = segments[indices]
    for center, radius in bodies:
        if not len(segments):
            break
        start = segments[:, 0]
        delta = segments[:, 1]-start
        q = start-center
        dd = _dot(delta, delta)
        cuts = [np.zeros((len(start), 1)), np.ones((len(start), 1)),
                _roots(dd, 2*_dot(q, delta), _dot(q, q)-radius**2)]
        if eye is None:
            normal = np.asarray(direction)/np.linalg.norm(direction)
            qp = q-_dot(q, normal)[:, None]*normal
            dp = delta-_dot(delta, normal)[:, None]*normal
            cuts.append(_roots(_dot(dp, dp), 2*_dot(qp, dp), _dot(qp, qp)-radius**2))
        else:
            # Tangent cone with apex at eye. Its intersection with a line is
            # quadratic. Sphere roots delimit endpoints inside the body.
            v, c = start-eye, np.asarray(center)-eye
            k = _dot(c, c)-radius**2
            vc, dc = _dot(v, c), _dot(delta, c)
            cuts.append(_roots(dc*dc-k*dd, 2*(vc*dc-k*_dot(v, delta)), vc*vc-k*_dot(v, v)))
        cuts = np.sort(np.concatenate(cuts, axis=1), axis=1)
        lo, hi = cuts[:, :-1], cuts[:, 1:]
        middle = start[:, None, :]+((lo+hi)/2)[..., None]*delta[:, None, :]
        keep = (hi-lo > 64*np.finfo(float).eps) & _visible(
            middle, [(center, radius)], direction=direction, eye=eye)
        row, col = np.nonzero(keep)
        segments = np.stack((start[row]+lo[row, col, None]*delta[row],
                             start[row]+hi[row, col, None]*delta[row]), axis=1)
        indices = indices[row]
    return segments, indices


@dataclass
class _Body:
    artist: object
    center: np.ndarray
    radius: float


def set_orbit_occlusion(ax, enabled=True):
    """Enable (default) or disable solid-body orbit masking on an SSATK axes.

    Applies to every draw, including saved animations and interactive camera
    rotations. Earth/Moon plotters register their displayed spherical surfaces
    automatically. Disabling permits an explicit see-through diagnostic view.
    Map panels and non-spatial diagnostics are unaffected. Custom 3-D Line3D,
    Line3DCollection and scatter artists on these axes are handled as well.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be True or False")
    if not hasattr(ax, "_ssatk_occlusion"):
        ax._ssatk_occlusion = _Scene(ax)
    ax._ssatk_occlusion.enabled = enabled
    ax.stale = True
    return ax


def _register_body(ax, artist, center, radius, view=None):
    if not hasattr(ax, "_ssatk_occlusion"):
        set_orbit_occlusion(ax)
    scene = ax._ssatk_occlusion
    scene.view = view
    body = _Body(weakref.ref(artist), np.asarray(center, dtype=float), float(radius))
    scene.bodies.append(body)
    return body


def _projected_orbit(artist, xyz, view="xy"):
    """Attach the missing depth coordinate to a 2-D spatial artist."""
    artist._ssatk_orbit_xyz = np.asarray(xyz, dtype=float)
    artist._ssatk_orbit_view = view
    return artist


def _camera(ax, view):
    if not hasattr(ax, "get_zlim"):
        a, b = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
        return dict(direction=np.eye(3)[3-a-b])
    # Derive camera rays from the actual projection. This includes roll,
    # vertical_axis, unequal/inverted limits, and perspective focal length.
    inverse = np.linalg.inv(ax.get_proj())
    homogeneous = inverse @ np.array([0., 0., -1., 0.])
    if abs(homogeneous[3]) < 1e-12:
        return dict(direction=homogeneous[:3]/np.linalg.norm(homogeneous[:3]))
    return dict(eye=homogeneous[:3]/homogeneous[3])


class _Scene:
    """Per-axes draw adapter; never patches Matplotlib classes or global state."""
    def __init__(self, ax):
        self.bodies, self.view, self.enabled = [], None, True
        original = ax.draw

        def draw(axes, renderer):
            # Removed/cleared/hidden surfaces must no longer occult tracks.
            self.bodies[:] = [b for b in self.bodies if b.artist() is not None
                              and b.artist().axes is axes]
            active = [b for b in self.bodies if b.artist().get_visible()]
            if not active:
                return original(renderer)
            camera = _camera(axes, self.view)
            bodies = [(b.center, b.radius) for b in active] if self.enabled else []
            restore = []
            try:
                if hasattr(axes, "computed_zorder"):
                    old = axes.computed_zorder
                    restore.append(lambda: setattr(axes, "computed_zorder", old))
                    axes.computed_zorder = False
                if "eye" in camera:
                    active.sort(key=lambda b: -np.linalg.norm(b.center-camera["eye"]))
                else:
                    active.sort(key=lambda b: b.center @ camera["direction"])
                for i, body in enumerate(active):
                    order = 1+i/(len(active)+1)*.5
                    _order(body.artist(), order, restore)
                    for image in getattr(body.artist(), "_ssatk_body_images", []):
                        _order(image, order, restore)
                    for halo in getattr(body.artist(), "_ssatk_body_backgrounds", []):
                        _order(halo, order-.001, restore)
                body_artists = {b.artist() for b in active}
                for artist in list(axes.lines)+list(axes.collections):
                    if (artist in body_artists or not artist.get_visible()
                            or getattr(artist, "_ssatk_occlusion_exempt", False)):
                        continue
                    _mask_artist(artist, bodies, camera, restore)
                return original(renderer)
            finally:
                for undo in reversed(restore):
                    undo()

        ax.draw = MethodType(draw, ax)


def _order(artist, order, restore):
    previous = artist.get_zorder()
    restore.append(lambda: artist.set_zorder(previous))
    artist.set_zorder(order)


def _mask_artist(artist, bodies, camera, restore):
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection, Path3DCollection

    projected = getattr(artist, "_ssatk_orbit_xyz", None)
    spatial = isinstance(artist, (Line3D, Line3DCollection, Path3DCollection))
    if not spatial and projected is None:
        return
    view = getattr(artist, "_ssatk_orbit_view", "xy")
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
    _order(artist, 2+np.arctan(artist.get_zorder())/10, restore)
    if isinstance(artist, PathCollection):
        if spatial:
            original = artist._offsets3d
            points = np.column_stack(original).astype(float)
            restore.append(lambda: setattr(artist, "_offsets3d", original))
        else:
            original = artist.get_offsets().copy()
            points = projected.copy()
            restore.append(lambda: artist.set_offsets(original))
        points[~_visible(points, bodies, **camera)] = np.nan
        if spatial:
            artist._offsets3d = tuple(points.T)
        else:
            artist.set_offsets(points[:, axes])
    elif isinstance(artist, LineCollection):
        original = artist._segments3d if spatial else artist.get_segments()
        colors = artist.get_colors().copy()
        restore.append(lambda: artist.set_segments(original))
        restore.append(lambda: artist.set_color(colors))
        paths = original if spatial else projected
        if isinstance(paths, np.ndarray) and paths.ndim == 3 and paths.shape[1:] == (2, 3):
            segments, owners = paths, np.arange(len(paths))
        else:
            segments, owners = [], []
            for i, path in enumerate(paths):
                path = np.asarray(path)
                if len(path) > 1:
                    segments.extend(np.stack((path[:-1], path[1:]), axis=1))
                    owners.extend([i]*(len(path)-1))
        clipped, indices = _clip_segments(segments, bodies, **camera)
        artist.set_segments(clipped if spatial else clipped[..., axes])
        if len(colors) and len(indices):
            artist.set_color(colors[np.asarray(owners)[indices] % len(colors)])
    elif isinstance(artist, Line2D):
        original = artist.get_data_3d() if spatial else artist.get_data()
        setter = artist.set_data_3d if spatial else artist.set_data
        restore.append(lambda: setter(*original))
        points = np.column_stack(original).astype(float) if spatial else projected.copy()
        if artist.get_linestyle() not in {"None", "", " "} and artist.get_linewidth() > 0:
            segments = np.stack((points[:-1], points[1:]), axis=1)
            clipped, _ = _clip_segments(segments, bodies, **camera)
            # Separate subsegments with NaNs; never join across a hidden arc.
            line = np.full((len(clipped), 3, 3), np.nan)
            line[:, :2] = clipped
            line = line.reshape(-1, 3)
            if artist.get_marker() not in {"None", "", " ", None}:
                # Keep markers at supplied samples only, not at limb cuts.
                from matplotlib.lines import Line2D as _Line
                from mpl_toolkits.mplot3d.art3d import Line3D as _Line3D
                overlay = (_Line3D if spatial else _Line)(
                    *(points if spatial else points[:, axes]).T,
                    linestyle="None", marker=artist.get_marker(),
                    markersize=artist.get_markersize(), color=artist.get_color(),
                    markerfacecolor=artist.get_markerfacecolor(),
                    markeredgecolor=artist.get_markeredgecolor(),
                    alpha=artist.get_alpha(), zorder=artist.get_zorder())
                artist.axes.add_line(overlay)
                if not spatial:
                    _projected_orbit(overlay, points, view)
                restore.append(overlay.remove)
                _mask_artist(overlay, bodies, camera, restore)
                marker = artist.get_marker()
                restore.append(lambda: artist.set_marker(marker))
                artist.set_marker("None")
            points = line
        else:
            points[~_visible(points, bodies, **camera)] = np.nan
        setter(*(points if spatial else points[:, axes]).T)
