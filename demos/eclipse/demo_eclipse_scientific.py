#!/usr/bin/env python3
"""
Demo: finite-Sun solar and lunar eclipses.

Builds the 2024-04-08 total solar eclipse and the 2025-03-14 total lunar
eclipse, validates them, and renders each eclipse product: the scientific
summary with its interactive 3-D view, the observer circumstances for
Dallas, Texas, a contact-to-contact WebGL animation, and a Plotly scene.
It then discovers the 2026-08-12 total solar eclipse from JPL DE430 states
and renders that event the same way.
"""

GALLERY_CATEGORY = "eclipse"

import os
import sys
from pathlib import Path

from ssatk.eclipse import (
    EventRequest,
    ObserverConfig,
    RenderRequest,
    build_discovered_event,
    build_event,
    discover_eclipses,
    render_product,
    validate_event,
)
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
DALLAS = ObserverConfig(latitude_deg=32.7767, longitude_east_deg=-96.7970, name="Dallas, Texas")


def _report(event):
    validation = validate_event(event)
    print(f"{event.definition.title}: validation {'passed' if validation.passed else 'FAILED'}, "
          f"{len(event.jd)} states, backend {event.metadata.get('state_source')}")
    return validation


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    samples = 25 if fast else 81
    quality = "motion" if fast else "high"

    solar = build_event(EventRequest(kind="solar", backend="reference", sample_count=samples))
    lunar = build_event(EventRequest(kind="lunar", backend="reference", sample_count=samples))
    records = discover_eclipses("2026-08-01", "2026-08-20", kind="solar")
    discovered = build_discovered_event(records[0], n_frames=samples)
    outputs = {
        "solar_event": solar,
        "lunar_event": lunar,
        "discovered_event": discovered,
        "validations": {event.definition.key: _report(event) for event in (solar, lunar, discovered)},
        "files": [],
    }
    if not make_figures:
        return outputs

    # gallery-call: solar_2024_04_08_scientific_summary.png | solar_2024_04_08_greatest_interactive_3d.html
    solar_suite = render_product(RenderRequest(
        product="scientific-suite", output=figpath("eclipse/solar_2024"), event=solar, quality=quality,
    ))
    # gallery-end
    # gallery-call: lunar_2025_03_14_*
    lunar_suite = render_product(RenderRequest(
        product="scientific-suite", output=figpath("eclipse/lunar_2025"), event=lunar, quality=quality,
    ))
    # gallery-end
    # gallery-call: solar_2024_04_08_32p776700N_96p797000W_*
    observer = render_product(RenderRequest(
        product="observer-suite", output=figpath("eclipse/solar_2024_dallas"),
        event=EventRequest(kind="solar", backend="reference", sample_count=samples,
                           solar_scope="local", observer=DALLAS),
        timezone_name="America/Chicago",
    ))
    # gallery-end
    # gallery-call: demo_eclipse_2024_cinematic_animation.html
    animation = render_product(RenderRequest(
        product="cinematic", output=figpath("eclipse/demo_eclipse_2024_cinematic_animation.html"),
        event=EventRequest(kind="solar", backend="de430", sample_count=samples), quality=quality, animate=True,
    ))
    # gallery-end
    # gallery-call: demo_eclipse_2025_lunar_plotly.html
    plotly_view = render_product(RenderRequest(
        product="plotly", output=figpath("eclipse/demo_eclipse_2025_lunar_plotly.html"),
        event=EventRequest(kind="lunar", backend="de430", sample_count=samples), quality=quality,
    ))
    # gallery-end
    # gallery-call: solar_2026_08_12_*
    discovered_suite = render_product(RenderRequest(
        product="scientific-suite", output=figpath("eclipse/solar_2026_discovered"), event=discovered,
        quality=quality,
    ))
    # gallery-end
    for suite in (solar_suite, lunar_suite, discovered_suite):
        outputs["files"].extend(Path(suite[key]) for key in ("scientific_summary_png", "interactive_3d_html"))
    outputs["files"].extend(Path(path) for path in observer["outputs"].values() if path)
    outputs["files"].extend([Path(animation), Path(plotly_view)])
    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
