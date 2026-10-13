import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from ssatk import Orbit, rv, AccelKepler, KeplerianPropagator, RK78Propagator

from ssatk.constants import RGEO
from ssatk.orbital_mechanics.ellipse_fit import ellipse_fit
from ssatk.plots.figpath import figpath
from ssatk.plots.groundtrack_dashboard import groundtrack_dashboard
from ssatk.plots.groundtrack_plot import groundtrack_plot
from ssatk.plots.groundtrack_video import groundtrack_video

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "orbital_mechanics"
TITLE = "SSATK transfer and Kepler propagation"
DESCRIPTION = "Compare SSATK's fitted transfer with its Kepler propagation API in position, ground track, altitude, and speed."


def main(make_figures=None, make_video=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_video is None:
        make_video = False
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-1-15")
    r1 = [0, 3 * RGEO, 0]
    r2 = [0, -RGEO / 6, -RGEO]

    # use positional P1/P2
    transfer = ellipse_fit(r1, r2, time_of_departure=t0)

    # Downsample in fast mode
    if fast:
        step = max(1, len(transfer["t_abs"]) // 120)
        t_abs = np.asarray(transfer["t_abs"])[::step]
        r_tf = np.asarray(transfer["r"])[::step]
    else:
        t_abs = np.asarray(transfer["t_abs"])
        r_tf = np.asarray(transfer["r"])

    orbit = Orbit(r=np.asarray(transfer["r"])[0], v=np.asarray(transfer["v"])[0], t=t0)

    if not fast:
        _r_tmp, _v_tmp = rv(
            orbit=orbit,
            time=transfer["t_abs"],
            propagator=RK78Propagator(AccelKepler(), h=1),
        )

    r_reference, v_reference = rv(
        orbit=orbit,
        time=t_abs,
        propagator=KeplerianPropagator(),
    )

    r_ssa = np.asarray(r_reference)
    dr = r_ssa - r_tf
    dr_norm_km = np.linalg.norm(dr, axis=1) / 1000.0
    t_hours = (t_abs - t_abs[0]) / 3600.0

    if make_figures:
        # gallery-call: test_transfer_vs_ssatk_diff.jpg
        plt.figure()
        plt.plot(t_hours, dr_norm_km)
        plt.xlabel("Time since departure [hours]")
        plt.ylabel("Position difference |Δr| [km]")
        plt.title("SSATK fitted transfer vs Kepler propagation: |Δr|(t)")
        out_path = figpath("test_transfer_vs_ssatk_diff.jpg")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        # gallery-end
        print(f"Saved: {out_path}")
        plt.close()

        # gallery-call: test_transfer_vs_ssatk_diff_groundtrack.jpg
        groundtrack_plot(
            r=[r_ssa, r_tf],
            t=t_abs,
            save_path=figpath("test_transfer_vs_ssatk_diff_groundtrack.jpg"),
        )
        # gallery-end

        try:
            # gallery-call: test_transfer_vs_ssatk_diff_dashboard.jpg
            groundtrack_dashboard(
                r=[r_ssa, r_tf],
                t=t_abs,
                save_path=figpath("test_transfer_vs_ssatk_diff_dashboard.jpg"),
                show=False,
            )
            # gallery-end
        except Exception as err:
            print(f"groundtrack_dashboard failed: {err}")

    if make_video:
        groundtrack_video(
            r=[r_ssa, r_tf],
            t=t_abs,
            save_path=figpath("test_transfer_vs_ssatk_diff_groundtrack.mp4"),
        )

    return {
        "transfer": transfer,
        "r_tf": r_tf,
        "r_ssatk": r_ssa,
        "dr": dr,
        "dr_norm_km": dr_norm_km,
        "t_hours": t_hours,
    }


if __name__ == "__main__":
    main(make_figures=True, make_video=True, fast=False)
