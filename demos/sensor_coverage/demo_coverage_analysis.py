"""
Demo script for ssatk coverage_analysis.

Answers the question a ground station actually asks. Not "is this satellite
overhead" -- "how often can I see it, for how long, and what is the worst gap
between contacts?" Two figures:

  site_analysis_<sat>.png
      One propagation window seen from one site: elevation and azimuth against
      time, the satellite's own illumination state, and which windows are
      genuinely observable. That last panel is the logical AND of three
      conditions -- satellite above the elevation mask, satellite sunlit, site
      in darkness -- rather than a single collapsed percentage, because a pass
      that fails any one of them is not a pass you can use optically.

  coverage_metrics_<sat>.png
      The pass structure behind a coverage percentage, computed at every point
      on Earth: mean pass duration, longest gap between contacts, contacts per
      day, and total contact minutes per day. "% of time visible" cannot tell
      one clean usable pass from a dozen useless slivers at the same duty
      cycle; these four can. All are masked to zero outside the orbit's real
      inclination band rather than filled with a misleading low value.

HST is the example because its 28.5 degree inclination puts a hard edge on the
coverage map at roughly +/-50 degrees latitude -- visible proof the masking is
real geometry and not a colour-scale artefact. The site is Cape Canaveral,
which sits inside that band.

Runs offline. The TLE comes from the satellite list bundled with the toolkit,
so no Space-Track credentials and no network are needed. For current elements,
refresh explicitly with ssatk.io.tle_updater.

Pytest-safe mode:
- coarse coverage grid, so the 3-day propagation stays quick
- does not save figures by default
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import sys

import matplotlib.pyplot as plt

from ssatk.plots.coverage_analysis import (
    SATELLITES,
    analyse_satellite,
    build_orbit,
)

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

SITE_NAME = "Cape Canaveral"
SITE_LAT = 28.39
SITE_LON = -80.60
SITE_ALT_M = 0.0

SAT_NAME = "HST"

# 10 degrees is the conventional mask for a real site: below it a pass is
# usually unusable because of terrain, buildings and atmospheric refraction.
# The coverage grid uses 5 degrees because it asks a different question --
# geometric visibility anywhere on Earth, not usability from one location.
MIN_ELEV_SITE_DEG = 10.0
MIN_ELEV_COVERAGE_DEG = 5.0

# Sun below -6 degrees is civil twilight: dark enough that a sunlit satellite
# is actually observable against the sky.
SUN_ELEV_DARK_DEG = -6.0


def _find_satellite(name):
    """Pick one satellite out of the bundled list by name."""
    for sat in SATELLITES:
        if sat.get("name", "").upper() == name.upper():
            return sat
    available = ", ".join(sorted(s.get("name", "?") for s in SATELLITES)[:8])
    raise LookupError(f"{name!r} not in the satellite list; first few: {available}")


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # The coverage grid propagates 3 days at 30 s and evaluates topocentric
    # elevation at every cell, so cost scales with n_lat * n_lon. The coarse
    # grid exercises every code path in a fraction of the time; the full grid
    # is what produces a figure worth looking at.
    n_lat, n_lon = (12, 24) if fast else (36, 72)

    sat = _find_satellite(SAT_NAME)
    orbit = build_orbit(sat)

    out_dir = "."
    if make_figures:
        from ssatk.plots.coverage_analysis import _analysis_output_dir
        out_dir = _analysis_output_dir(SITE_NAME, SITE_LAT, SITE_LON)

    result = analyse_satellite(
        sat_name=sat["name"],
        orbit=orbit,
        site_name=SITE_NAME,
        lat=SITE_LAT,
        lon=SITE_LON,
        alt_m=SITE_ALT_M,
        min_el_sat=MIN_ELEV_SITE_DEG,
        sun_el_dark=SUN_ELEV_DARK_DEG,
        min_el_cov=MIN_ELEV_COVERAGE_DEG,
        n_lat=n_lat,
        n_lon=n_lon,
        out_dir=out_dir,
    )

    if not make_figures:
        plt.close("all")

    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False)
