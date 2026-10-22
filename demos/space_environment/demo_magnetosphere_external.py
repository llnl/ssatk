#!/usr/bin/env python3
"""
Demo: reproducible Earth magnetosphere with the Tsyganenko 1989 (T89) field.

The external model is driven by a fixed Kp=3 value.  The magnetopause uses
fixed Shue et al. (1998) drivers (Bz=-2 nT and dynamic pressure=2 nPa), so
the output does not depend on the local OMNI cache or a network connection.

T96 is not selected here because the current plotting API resolves its solar-
wind drivers internally through ``get_solar_wind``; it cannot receive a
complete caller-provided driver record.  Selecting T96 with ``use_omni=False``
would therefore fall back to T89 after an unavailable OMNI lookup.

Run
---
    python -m demos.space_environment.demo_magnetosphere_external
"""

GALLERY_CATEGORY = "space_environment"
TITLE = "Magnetosphere with fixed external-field drivers"
DESCRIPTION = (
    "IGRF 2025 plus T89 at Kp=3, with a Shue magnetopause at Bz=-2 nT "
    "and 2 nPa dynamic pressure. Belts illustrate L shells; no AE8/AP8 flux."
)

import os
import sys

from ssatk.plots.figpath import figpath
from ssatk.plots.magfield_plot_3d import plot_magfield_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    """Render the fixed-driver T89 example, or report a truthful skip."""
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if not make_figures:
        return {"figure": None, "skipped": True, "reason": "figures_disabled"}

    # T89 is provided by the optional geomagnetics extra.  The shared plotter
    # otherwise degrades to internal IGRF, which would make this external-field
    # demo misleading; skip instead of publishing that degraded result.
    from ssatk import geomagnetics

    if not geomagnetics._HAS_GEOPACK:
        reason = "geopack is unavailable; T89 external field requires ssatk[geomagnetics]"
        print(f"Skipped: {reason}")
        return {"figure": None, "skipped": True, "reason": reason}

    try:
        # gallery-call: demo_magnetosphere_external.html | demo_magnetosphere_external.png
        fig, stats = plot_magfield_3d(
            epoch=2025.0,
            fidelity="draft" if fast else "high",
            external_field="t89",
            kp=3,
            use_omni=False,
            sw_bz_nT=-2.0,
            sw_dp_nPa=2.0,
            belt_style="igrf",
            show_magnetopause=True,
            # The generic provenance panel contains validation claims from
            # other model configurations; this demo states its inputs directly.
            show_provenance=False,
            title=("Earth magnetosphere — T89 (Kp=3), fixed solar-wind drivers; "
                   "illustrative IGRF L-shell belts"),
            texture_path="auto",
            elev=8,
            azim=-15,
            max_r_re=10.0,
            show_van_allen=True,
            save_path=figpath("demo_magnetosphere_external"),
        )
        # gallery-end
    except ImportError as exc:
        print(f"Skipped: {exc}")
        return {"figure": None, "skipped": True, "reason": str(exc)}

    # A PNG preview belongs on the gallery card; the HTML remains interactive.
    fig.write_image(figpath("demo_magnetosphere_external.png"), width=1200, height=900)
    print("Saved: demo_magnetosphere_external.html and demo_magnetosphere_external.png")
    return {"figure": fig, "stats": stats, "skipped": False}


if __name__ == "__main__":
    main(make_figures=True)
