"""
Demo script for ssatk coverage_analysis.

Shows three days of geometric HST access from WGS84 sea-level grid cells,
with an example site at Cape Canaveral. TLE input uses SSAPy's SGP4Propagator.
The displayed maps are raw 2-degree cells with 15-second time sampling and
interpolated rise/set crossings. A contact shorter than the sample spacing
can be missed. Window-edge contacts/gaps are clipped, not extrapolated.

The site plot's optical candidates require elevation >10 degrees, full solar
illumination and geometric Sun elevation <-6 degrees. This is a configurable
geometric screen, not a weather, brightness or optical-detectability model.
Map access uses a separate >5-degree mask and includes day and night.

Runs offline using archived packaged HST elements, not a current tracking
prediction. CSV grid values and a JSON record of the exact TLE, epoch and
settings accompany the two PNGs. No figures or files are written when
make_figures=False.
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import csv
import json
from pathlib import Path
import sys


from ssatk.plots.coverage_analysis import (
    SATELLITES,
    analyse_satellite,
    build_orbit,
)

UNDER_PYTEST = (
    "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
)

SITE_NAME = "Cape Canaveral"
SITE_LAT = 28.39
SITE_LON = -80.60
SITE_ALT_M = 0.0

SAT_NAME = "HST"

# Illustrative geometric masks; neither includes local terrain or weather.
MIN_ELEV_SITE_DEG = 10.0
MIN_ELEV_COVERAGE_DEG = 5.0

# Civil-twilight cutoff used as an illustrative optical-candidate criterion.
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

    # Published maps retain 2-degree cells even in the gallery's fast mode.
    n_lat, n_lon = (30, 60) if fast and not make_figures else (90, 180)

    sat = _find_satellite(SAT_NAME)
    orbit = build_orbit(sat)

    out_dir = "."
    if make_figures:
        from ssatk.plots.coverage_analysis import _analysis_output_dir

        out_dir = _analysis_output_dir(SITE_NAME, SITE_LAT, SITE_LON)

    # gallery-call: *.png
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
        make_figures=make_figures,
    )
    # gallery-end

    if make_figures:
        keys = [
            "frac",
            "mean_pass_grid",
            "max_pass_grid",
            "max_gap_grid",
            "passes_day_grid",
            "cum_contact_grid",
        ]
        with (Path(out_dir) / "coverage_grid_hst.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["latitude_deg", "longitude_deg", *keys])
            for i, la in enumerate(result["lat_grid"]):
                for j, lo in enumerate(result["lon_grid"]):
                    writer.writerow([la, lo, *(result[key][i, j] for key in keys)])
        from astropy.time import Time

        metadata = {
            "satellite": sat,
            "epoch_utc": Time(orbit.t, format="gps").utc.isot,
            "propagator": result["propagator"],
            "earth_model": "WGS84 sea level",
            "duration_days": result["coverage_days"],
            "step_seconds": result["coverage_step_s"],
            "latitude_cells": n_lat,
            "longitude_cells": n_lon,
            "minimum_elevation_deg": MIN_ELEV_COVERAGE_DEG,
            "global_mean_area_weighted": result["global_mean"],
            "site_coverage": result["site_cov"],
            "site": {
                "name": SITE_NAME,
                "lat_deg": SITE_LAT,
                "lon_deg": SITE_LON,
                "height_m": SITE_ALT_M,
            },
            "site_pass_statistics": {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in result["site_pass_stats"].items()
            },
            "limits": "Finite-window geometric access; interpolated crossings; sub-step contacts can be missed. No terrain, refraction, weather, or detectability model.",
        }
        if metadata["site_pass_statistics"]["n_passes"] == 0:
            metadata["site_pass_statistics"]["time_to_first_min"] = None
        (Path(out_dir) / "coverage_metadata_hst.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )

    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False)
