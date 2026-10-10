#!/usr/bin/env python3
"""
Generate a dense set of rotated unit-vector plots and optionally stitch them
into a GIF.

The final GIF is saved under figpath("") within the
shared ``~/ssatk_output/figures`` output root.
Intermediate frame images are created in a temporary directory and removed
after the GIF is written, so they do not remain on disk or appear in the
demo gallery.

Pytest-safe behavior:
- does not generate all frames by default under pytest
- does not write GIF by default under pytest
- uses a much smaller grid under pytest
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from IPython.display import clear_output
except Exception:
    def clear_output(wait=True):
        return None

from ssatk.vectors import rotate_vector
from ssatk.plots.figpath import figpath
from ssatk.plots.write_gifs import write_gif

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(make_figures=None, make_gif=None, fast=None):
    """
    Run the rotate-vector demo.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    make_gif : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        GIF path and run configuration metadata.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_gif is None:
        make_gif = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    v_unit = np.array([1, 0, 0])

    if fast:
        theta_step = 60
        phi_step = 90
        fps = 6
    else:
        theta_step = 10
        phi_step = 10
        fps = 20

    num_theta = int(180 / theta_step) + 1
    num_phi = int(360 / phi_step) + 1
    total = num_theta * num_phi

    gif_path = None
    frame_count = 0

    if make_figures:
        with tempfile.TemporaryDirectory(prefix="rotate_vector_frames_") as tmpdir:
            tmpdir = Path(tmpdir)
            frames = []

            i = 0
            for theta in range(0, 181, theta_step):
                for phi in range(0, 361, phi_step):
                    clear_output(wait=True)

                    frame_base = tmpdir / f"{i:06d}"
                    rotate_vector(v_unit, theta, phi, save_path=str(frame_base))

                    frame_path = str(frame_base)
                    if not frame_path.lower().endswith(".jpg"):
                        frame_path += ".jpg"

                    frames.append(frame_path)
                    i += 1
                    print(f"Rendered {i}/{total} frames")

            frame_count = len(frames)

            if make_gif and frames:
                gif_base_jpg_like = figpath(
                    f"figures/rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}"
                )
                if gif_base_jpg_like.lower().endswith(".jpg"):
                    gif_path = gif_base_jpg_like[:-4] + ".gif"
                else:
                    gif_path = gif_base_jpg_like + ".gif"

                write_gif(gif_name=gif_path, frames=frames, fps=fps)

                print("\nGIF written to:", gif_path)
                print("Total frames:", len(frames))
    else:
        i = 0
        for theta in range(0, 181, theta_step):
            for phi in range(0, 361, phi_step):
                clear_output(wait=True)
                try:
                    rotate_vector(v_unit, theta, phi)
                except TypeError:
                    pass
                i += 1
                print(f"Rendered {i}/{total} frames")
        frame_count = i

    return {
        "frames": [],
        "gif_path": gif_path,
        "total_frames": total,
        "rendered_frames": frame_count,
        "theta_step": theta_step,
        "phi_step": phi_step,
    }


if __name__ == "__main__":
    main(make_figures=True, make_gif=True, fast=False)
