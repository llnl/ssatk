"""Shared matplotlib legend handlers for SSATK plots."""

from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase


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
