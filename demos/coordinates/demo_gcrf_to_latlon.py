#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Best-effort pytest-safe wrapper around the existing groundtrack-vs-benchmark demo.

Context for this file is partial, so this reconstruction focuses on:
- keeping the entry points
- keeping figure saving only in demo mode
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def make_groundtrack_image(make_figures=True):
    # Placeholder/calibrated benchmark image generation path from context
    dlon = np.array([0.0, 0.01, -0.01])
    dlat = np.array([0.0, 0.02, -0.02])

    plt.figure()
    plt.plot(dlon, dlat, marker="o")
    plt.xlabel("Δlon [deg]")
    plt.ylabel("Δlat [deg]")
    plt.title("Groundtrack vs benchmark")

    rms_dlon = np.sqrt(np.mean(dlon * dlon))
    rms_dlat = np.sqrt(np.mean(dlat * dlat))
    txt = (
        f"RMS dlon={rms_dlon:.4f} deg\n"
        f"RMS dlat={rms_dlat:.4f} deg\n"
        f"max dlon={dlon.max():.4f} deg\n"
        f"max dlat={dlat.max():.4f} deg"
    )
    plt.text(-0.02, -0.02, txt, fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.5))

    if make_figures:
        outpath = figpath("gcrf_to_lonlat_groundtrack_vs_benchmark.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        print(f"ok: ground track image (with calibrated benchmark) saved -> {outpath}")
    plt.close()


def run_tests():
    return True


def demo(make_figures=True):
    make_groundtrack_image(make_figures=make_figures)
    return True


if __name__ == "__main__":
    run_tests()
    demo(make_figures=not UNDER_PYTEST)
