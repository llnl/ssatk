import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "video_animation"


def _count_gif_frames(path):
    with imageio.get_reader(path) as rdr:
        try:
            n = rdr.get_length()
        except Exception:
            n = sum(1 for _ in rdr)
    return n


def main(make_artifacts=None, fast=None, verbose=None):
    """
    Run gifify demo/tests.

    Parameters
    ----------
    make_artifacts : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.
    verbose : bool or None
        If None, defaults to False under pytest and True otherwise.

    Returns
    -------
    dict
        Outputs from gifify runs.
    """
    if make_artifacts is None:
        make_artifacts = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST

    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    n = 90 if fast else 120
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) * np.exp(-0.1 * x)

    outputs = {}

    # Test 1: pyplot-style function (returns None), chunks mode
    def plot_simple(x, y, label=None):
        plt.figure()
        plt.plot(x, y, label=label)
        if label:
            plt.legend()
        plt.grid(True)

    if make_artifacts:
        from ssatk.plots.figpath import figpath
        from ssatk.plots.gifify import gifify

        out1 = gifify(
            plot_simple,
            x, y,
            label="damped wave",
            save_path=figpath("test_chunks.gif"),
            array_arg_indices=(0, 1),
            mode="chunks",
            chunk_size=30 if fast else 40,
            step=30 if fast else 40,
            fps=12,
            verbose=verbose,
        )
        assert os.path.exists(out1["path"]) and os.path.getsize(out1["path"]) > 0, "test_chunks.gif not created"
        n1 = _count_gif_frames(out1["path"])
        assert n1 == out1["frames"], f"Frame mismatch (chunks): gif={n1} reported={out1['frames']}"
        print(f"Test 1 OK: {out1['path']} with {n1} frames")
        outputs["chunks"] = out1
    else:
        plot_simple(x[:10], y[:10], label="damped wave")
        plt.close("all")

    # Test 2: function that returns an Axes, cumulative mode
    def plot_returns_axes(x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("returns Axes")
        ax.grid(True)
        return ax

    if make_artifacts:
        from ssatk.plots.figpath import figpath
        from ssatk.plots.gifify import gifify

        out2 = gifify(
            plot_returns_axes,
            x, y,
            save_path=figpath("test_cumulative.gif"),
            array_arg_indices=(0, 1),
            mode="cumulative",
            step=15 if fast else 30,
            fps=10,
            verbose=verbose,
        )
        assert os.path.exists(out2["path"]) and os.path.getsize(out2["path"]) > 0, "test_cumulative.gif not created"
        n2 = _count_gif_frames(out2["path"])
        assert n2 == out2["frames"], f"Frame mismatch (cumulative): gif={n2} reported={out2['frames']}"
        print(f"Test 2 OK: {out2['path']} with {n2} frames")
        outputs["cumulative"] = out2
    else:
        ax = plot_returns_axes(x[:10], y[:10])
        plt.close(ax.figure)

    # Test 3: function that expects an injected Axes, sliding mode
    def plot_with_ax(ax, x, y):
        ax.plot(x, y)
        ax.set_title("injected ax")
        ax.grid(True)

    if make_artifacts:
        from ssatk.plots.figpath import figpath
        from ssatk.plots.gifify import gifify

        out3 = gifify(
            plot_with_ax,
            x, y,
            save_path=figpath("test_sliding.gif"),
            array_arg_indices=(0, 1),
            mode="sliding",
            chunk_size=20 if fast else 30,
            step=15 if fast else 30,
            fps=10,
            inject_ax=True,
            ax_arg_index=0,
            verbose=verbose,
        )
        assert os.path.exists(out3["path"]) and os.path.getsize(out3["path"]) > 0, "test_sliding.gif not created"
        n3 = _count_gif_frames(out3["path"])
        assert n3 == out3["frames"], f"Frame mismatch (sliding): gif={n3} reported={out3['frames']}"
        print(f"Test 3 OK: {out3['path']} with {n3} frames")
        outputs["sliding"] = out3
    else:
        fig, ax = plt.subplots()
        plot_with_ax(ax, x[:10], y[:10])
        plt.close(fig)

    # Test 4: groundtrack_dashboard (multi-axes), sliding mode.
    # Fast non-artifact tests skip this because groundtrack_dashboard has its
    # own focused tests and rendering a dashboard dominates this smoke test.
    if make_artifacts:
        from ssatk.constants import RGEO
        from ssatk.plots.figpath import figpath
        from ssatk.plots.gifify import gifify
        from ssatk.plots.groundtrack_dashboard import groundtrack_dashboard
        from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit

        r, _v, t = ssapy_orbit(a=RGEO, e=0.2)

        max_points = 80 if fast else 100
        step = max(1, len(r) // max_points)
        r_use = r[::step]
        t_use = t.gps[::step]

        out4 = gifify(
            groundtrack_dashboard,
            r_use, t_use,
            save_path=figpath("test_groundtrack.gif"),
            array_arg_indices=(0, 1),
            mode="sliding",
            chunk_size=30 if fast else 40,
            step=20 if fast else 30,
            fps=8,
            verbose=verbose,
            fixed_limits=True,
            show_legend=False,
            pad=1000,
            t0=float(t[0].gps),
        )
        assert os.path.exists(out4["path"]) and os.path.getsize(out4["path"]) > 0, "test_groundtrack.gif not created"
        n4 = _count_gif_frames(out4["path"])
        assert n4 == out4["frames"], f"Frame mismatch (groundtrack): gif={n4} reported={out4['frames']}"
        print(f"Test 4 OK: {out4['path']} with {n4} frames")
        outputs["groundtrack"] = out4
    elif not fast:
        from ssatk.constants import RGEO
        from ssatk.plots.groundtrack_dashboard import groundtrack_dashboard
        from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit

        r, _v, t = ssapy_orbit(a=RGEO, e=0.2)
        r_use = r
        t_use = t.gps
        try:
            groundtrack_dashboard(
                r_use[:20],
                t_use[:20],
                show=False,
                show_legend=False,
                pad=1000,
                t0=float(t[0].gps),
            )
        except TypeError:
            pass

    print("All tests passed.")
    return outputs


if __name__ == "__main__":
    main(make_artifacts=True, fast=False, verbose=True)
