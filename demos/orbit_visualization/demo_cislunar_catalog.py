"""Rebuild a diverse, screened selection from LLNL's public GDO catalog.

The four labels describe 180-day ephemeris arcs, not certified periodic
families. Reproduce the bounded search with scripts/select_gdo_families.py.
Only required HTTP ranges are downloaded; raw inputs are discarded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.error import URLError


from demos.orbit_visualization._gdo_families import (
    EPOCH_TT, MAX_SAMPLES, SOURCE, STEP_SECONDS, describe_track, read_track,
)
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.figpath import figpath
from ssatk.plots.plotutils import figsave

GALLERY_CATEGORY = "orbit_visualization"
TITLE = "GDO cislunar family candidates"
DESCRIPTION = (
    "One SSATK cislunar scene containing L4-like, L5-like, DRO-like and prograde lunar tracks, selected "
    "by 180-day rotating-frame behavior. L1/L2 NRHO coverage is explicitly "
    "unfilled in this bounded search. Temporary public inputs are discarded."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
# IDs, ETags and canonical little-endian state digests from the reproducible
# search. Ordinary gallery builds fetch these four tracks, not the entire CSV.
SELECTION = (
    (771530, "L4-like", '"216a2d90-62cc84cc00d40"', "7f150f42aa18ee8ac399d76ded1b7c6f2214b41f06ec4061ad25fb3dd44e4c1d"),
    (562904, "L5-like", '"253b4b90-62cc8524b1e80"', "fe0eea063ac04cb794c200a909f5a304e6b01d912e2503d4c69206ee92f98bf3"),
    (776801, "DRO-like", '"188d1480-62cc85536ccc0"', "2ac8c33a314c974bf824cacf47ea7c4c46c384fb5491c0de2517124422d7fe3d"),
    (546929, "Prograde lunar", '"28d298a0-62cc83778aa00"', "f0e93c2d14110b1dec875b1c29acab8c8d875a154d5a92a816e0b3aa4687e646"),
)
SEARCH = {
    "catalog_rows": 1000000, "unique_trajectories_screened": 172,
    "initial_screen": "Nearest 40 six-year survivors (lifetime >5.9 years) to each of L4, L5, Moon, approximate L1 and L2 at the initial epoch; deduplicated",
    "csv_etag": '"2b122cd1-62f74f9cdf480"',
    "duration_days": 180,
    "ranking": "Smallest longitude span for L4/L5; smallest lunar radius q95/q05 for the other accepted morphologies; ID breaks ties",
    "criteria_source": "demos/orbit_visualization/_gdo_families.py",
    "reproduce": "PYTHONPATH=. python scripts/select_gdo_families.py --output selection-audit.json",
    "unfilled": ["L1 halo/NRHO candidate", "L2 halo/NRHO candidate"],
    "coverage_limit": "No match among these candidates does not establish absence from GDO. Long-lived initial-position screening can miss unstable or differently phased families.",
}


def load_catalog_sample():
    tracks, sources = [], []
    for orbit_id, family, etag, expected_digest in SELECTION:
        r, v, t, source = read_track(orbit_id, etag)
        if source["sample_sha256"] != expected_digest:
            raise ValueError(f"ID {orbit_id} checksum changed; review before updating its pin")
        *_, metrics = describe_track(r, v, t)
        if family not in metrics["candidate_labels"]:
            raise ValueError(f"ID {orbit_id} no longer passes the {family} screen")
        tracks.append({"id": orbit_id, "family": family, "t": t, "r": r,
                       "metrics": metrics})
        sources.append({**source, "label": family, "metrics": metrics})
    provenance = {
        "source": SOURCE, "dataset_citation": "https://doi.org/10.3847/2515-5172/adf8ec",
        "conventions_source": "https://arxiv.org/html/2512.11064v1",
        "selection_search": SEARCH, "source_files": sources,
        "state_frame": "GCRF", "position_units": "m", "velocity_units": "m/s",
        "plot_frame": "Moon-centered Earth-Moon rotating axes; SSATK GEO distance units",
        "classification_units": "Instantaneous Earth-Moon distance (LD)",
        "visualization_routine": "ssatk.plots.orbit_plot(view=cislunar_3d, c=black)",
        "epoch_tt": EPOCH_TT, "sample_step_seconds": STEP_SECONDS,
        "samples_per_orbit": MAX_SAMPLES,
        "time_convention": "Published epoch and hourly cadence; timestamps absent from HDF5 arrays",
        "interpretation": "Finite-arc morphological candidates; no periodic-family or stability certification",
        "raw_downloads_retained": False,
    }
    return tracks, provenance


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if not make_figures:
        return {"skipped": True, "reason": "figures_disabled; no download attempted"}
    try:
        tracks, provenance = load_catalog_sample()
    except (URLError, TimeoutError) as exc:
        print(f"Skipped cislunar catalog demo: public data unavailable: {exc}")
        return {"skipped": True, "reason": str(exc)}
    # SSATK owns scene geometry, frame conversion, mapped bodies and styling.
    # Fast and full builds retain all four 180-day arcs in the same scene.
    # gallery-call: demo_cislunar_catalog.jpg
    fig, ax = orbit_plot(
        r=[track["r"] for track in tracks], t=[track["t"] for track in tracks],
        view="cislunar_3d", c="black", figsize=(16, 12), fontsize=13,
        title="GDO cislunar family candidates · four trajectories · 180 days",
        legend=True,
    )
    body_legend = ax.get_legend()
    paths = [artist for artist in ax.collections if artist.get_label() == "Orbit Path"]
    labels = [f"{track['family']} · GDO {track['id']}" for track in tracks]
    ax.legend(paths, labels, loc="upper right", facecolor="black", edgecolor="white",
              labelcolor="white", fontsize=12, markerscale=4)
    ax.add_artist(body_legend)
    fig.subplots_adjust(left=.02, right=.98, bottom=.08, top=.90)
    fig.text(.06, .035,
             "Morphology candidates, not certified periodic families. L1/L2 NRHO: no match in the 172-trajectory screen.",
             color="white", fontsize=11)
    if figsave(fig, "demo_cislunar_catalog.jpg") is None:
        raise RuntimeError("Could not save the SSATK cislunar scene")
    # gallery-end
    Path(figpath("demo_cislunar_catalog_provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"skipped": False, "provenance": provenance}


if __name__ == "__main__":
    main(make_figures=True)
