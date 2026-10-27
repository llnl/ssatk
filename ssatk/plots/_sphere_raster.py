"""Per-pixel mapped spheres for Matplotlib's actual 3-D camera projection."""
import numpy as np
from matplotlib.artist import Artist


def sphere_intersections(points, center, radius, *, direction=None, eye=None):
    """Front intersections and signed ray clearance, in scene coordinates.

    ``points`` lie on a camera plane. Clearance is positive for intersecting
    rays; the returned tangent extension outside the limb is used only for
    subpixel edge coverage. Perspective intersections behind the eye are hidden.
    """
    if eye is None:
        toward_eye = np.asarray(direction, dtype=float)
        toward_eye = toward_eye/np.linalg.norm(toward_eye)
        q = points-center
        closest = q-np.sum(q*toward_eye, axis=-1)[..., None]*toward_eye
        distance = np.linalg.norm(closest, axis=-1)
        front = closest+np.sqrt(np.maximum(0, radius**2-distance**2))[..., None]*toward_eye
        return center+front, radius-distance
    rays = points-eye
    rays /= np.linalg.norm(rays, axis=-1)[..., None]
    q = np.asarray(eye)-center
    along = np.sum(q*rays, axis=-1)
    closest = q-along[..., None]*rays
    distance = np.linalg.norm(closest, axis=-1)
    root = np.sqrt(np.maximum(0, radius**2-distance**2))
    travel = -along-root
    travel = np.where(travel >= 0, travel, -along+root)
    clearance = np.where(travel >= 0, radius-distance, -radius)
    return np.asarray(eye)+travel[..., None]*rays, clearance


class MappedSphere(Artist):
    """Opaque sphere sampled at output pixels, with no polygon color tiles.

    The existing occultation adapter handles orbit/body ordering. Drawing uses
    the axes' projection matrix, so perspective, roll, limits and camera motion
    match the surrounding 3-D artists. The source map is captured at creation.
    """
    def __init__(self, ax, texture, *, zorder=10):
        super().__init__()
        self.texture = texture
        self.set_zorder(zorder)
        ax.add_artist(self)
        self.set_clip_path(ax.patch)
        self._view_geometry_key = None
        self._view_geometry = None

    def set_surface(self, center, radius, rotation, sun_hat, ambient, diffuse):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.rotation = np.asarray(rotation)
        self.sun_hat, self.ambient, self.diffuse = sun_hat, ambient, diffuse
        self.stale = True

    def draw(self, renderer):
        if not self.get_visible():
            return
        from scipy.ndimage import map_coordinates
        from .occultation import _camera

        ax = self.axes
        matrix = ax.get_proj()
        bounds = np.array([[ax.bbox.x0, ax.bbox.y0], [ax.bbox.x1, ax.bbox.y1]])
        scale = renderer.get_image_magnification()
        key = (
            matrix.shape, matrix.tobytes(), tuple(bounds.ravel()), float(scale),
            tuple(self.center), float(self.radius),
        )
        if key != self._view_geometry_key:
            corners = np.array(np.meshgrid(*[[-1., 1.]]*3)).reshape(3, -1).T
            corners = self.center+self.radius*corners
            projected = np.column_stack((corners, np.ones(8))) @ matrix.T
            # A box entirely in front of the camera bounds the projected sphere.
            if np.all(projected[:, 3] > 0):
                pixels = ax.transData.transform(projected[:, :2]/projected[:, 3, None])
                bounds[0] = np.maximum(bounds[0], pixels.min(axis=0)-1)
                bounds[1] = np.minimum(bounds[1], pixels.max(axis=0)+1)
            lower = np.floor(bounds[0]).astype(int)
            upper = np.ceil(bounds[1]).astype(int)
            width, height = np.ceil((upper-lower)*scale).astype(int)
            if width < 2 or height < 2:
                self._view_geometry = None
            else:
                x, y = np.meshgrid(lower[0]+(np.arange(width)+.5)/scale,
                                   lower[1]+(np.arange(height)+.5)/scale)
                xy = ax.transData.inverted().transform(
                    np.column_stack((x.ravel(), y.ravel()))
                )
                clip = np.column_stack((xy, -np.ones(len(xy)), np.ones(len(xy))))
                world = clip @ np.linalg.inv(matrix).T
                points = (world[:, :3]/world[:, 3, None]).reshape(height, width, 3)
                camera = _camera(ax, None)
                hits, clearance = sphere_intersections(
                    points, self.center, self.radius, **camera
                )
                self._view_geometry = (lower, points, hits, clearance, camera)
            self._view_geometry_key = key
        if self._view_geometry is None:
            return
        lower, points, hits, clearance, camera = self._view_geometry
        height, width = clearance.shape
        # Linear coverage over the boundary pixel avoids a jagged silhouette.
        dy, dx = np.gradient(clearance)
        footprint = np.maximum(np.hypot(dx, dy), np.finfo(float).eps*self.radius)
        alpha = np.clip(.5+clearance/footprint, 0, 1)
        normals = hits-self.center
        normals /= np.maximum(np.linalg.norm(normals, axis=-1)[..., None], np.finfo(float).tiny)
        fixed = normals @ self.rotation
        lon = np.arctan2(fixed[..., 1], fixed[..., 0])
        lat = np.arcsin(np.clip(fixed[..., 2], -1, 1))
        rows, cols = self.texture.shape[:2]
        col = (lon+np.pi)/(2*np.pi)*cols-.5
        row = np.clip((np.pi/2-lat)/np.pi*rows-.5, 0, rows-1)
        rgb = np.stack([map_coordinates(self.texture[..., channel], [row, col],
                        order=1, output=np.float64, mode="grid-wrap")
                        for channel in range(3)], axis=-1)
        if self.sun_hat is not None:
            rgb *= (self.ambient+self.diffuse*np.maximum(0, normals @ self.sun_hat))[..., None]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        rgba[..., 3] = np.round(alpha*255).astype(np.uint8)
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        try:
            renderer.draw_image(gc, lower[0], lower[1], rgba)
        finally:
            gc.restore()
        self.stale = False
