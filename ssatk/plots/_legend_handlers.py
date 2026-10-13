"""Shared matplotlib legend handlers for SSATK plots."""

from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend_handler import HandlerPatch
from matplotlib.image import BboxImage
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox


class BodyMapPatch(Patch):
    """Legend proxy carrying the same north-up map used on a body surface."""

    def __init__(self, image, **kwargs):
        super().__init__(**kwargs)
        self.image = image


class BodyMapHandler(HandlerPatch):
    """Fit a full equirectangular map inside the rectangular legend key."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        border = super().create_artists(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        )[0]
        border.set_facecolor("none")
        bounds = TransformedBbox(Bbox.from_bounds(-xdescent, -ydescent, width, height), trans)
        image = BboxImage(bounds, origin="upper", interpolation="bilinear")
        image.set_data(orig_handle.image)
        return [image, border]


def body_legend_handle(body, *, edgecolor="black"):
    """Map key for textured bodies; matching solid key for low/missing maps."""
    from ._body_surfaces import BODY_COLORS, surface_map, texture_path

    path = texture_path(body)
    style = dict(facecolor=BODY_COLORS[body], edgecolor=edgecolor, label=body.title())
    if path is None:
        return Patch(**style)
    return BodyMapPatch(surface_map(body, 128, 64, path), **style)


class GradientLineHandler(HandlerBase):
    """Legend handler that draws a short rainbow-gradient line."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        num_segments = 10
        x = np.linspace(xdescent, xdescent + width, num_segments + 1)
        y = ydescent + height / 2
        segments = [((x[i], y), (x[i + 1], y)) for i in range(num_segments)]
        colors = cm.rainbow(np.linspace(0, 1, num_segments))
        line_collection = LineCollection(segments, colors=colors, linewidth=2)
        line_collection.set_transform(trans)
        return [line_collection]
