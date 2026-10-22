"""Rebuild cislunar views from a pinned subset of LLNL's public orbit catalog.

Only required HTTP byte ranges are downloaded. Temporary input blocks are
removed on success or failure; the output retains figures and provenance only.
The first 100 catalog IDs are a deterministic sample, not the seminar's
unidentified selection of long-lived orbits. These are published numerical
trajectories, not new integrations or an independent validation of the catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import URLError

import h5py
import numpy as np
from astropy.time import Time

from demos.orbit_visualization._remote_hdf5 import TemporaryHTTPFile
from ssatk.plots.figpath import figpath
from ssatk.plots.orbit_plot import orbit_plot

GALLERY_CATEGORY = "orbit_visualization"
TITLE = "100 public LLNL cislunar trajectories"
DESCRIPTION = (
    "Catalog IDs 0–99, first 30 days or until the recorded trajectory ends. "
    "Hourly GCRF samples, with Earth–Moon rotating views computed by SSATK. "
    "Pinned public inputs are downloaded temporarily and discarded after use."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
SOURCE = "https://gdo-cislunar.llnl.gov/"
BASE_URL = SOURCE + "cislunar_data_six_year_1.0GEO_to_18.2GEO/"
SOURCES = (
    (0, '"6a0e40e0-62cc82b12d600"'),
    (50, '"62962a30-62cc82bbaaec0"'),
)
# Digest of IDs and canonical little-endian float64 r/v samples, in ID order.
SAMPLE_SHA256 = "afd6f289f697d6068422a8058a150d18c79b5ae8e2ac5e823038f6414b8c85ed"
# Dataset convention: https://arxiv.org/html/2512.11064v1, Sections III.1/III.4.
# HDF5 r/v arrays do not carry timestamps; record this convention explicitly.
EPOCH_TT = "1980-01-01T00:00:00"
STEP_SECONDS = 3600.0
MAX_SAMPLES = 721


def load_catalog_sample():
    """Return the pinned GCRF positions/times and input provenance."""
    tracks, times, sources = [], [], []
    digest = hashlib.sha256()
    epoch_gps = Time(EPOCH_TT, scale="tt").gps
    for first_id, etag in SOURCES:
        url = BASE_URL + f"orb_id_{first_id}_to_{first_id + 49}.h5"
        with TemporaryHTTPFile(url, expected_etag=etag) as remote:
            with h5py.File(remote, "r") as data:
                for orbit_id in range(first_id, first_id + 50):
                    record = data[str(orbit_id)]
                    n = min(MAX_SAMPLES, len(record["r"]))
                    r = np.asarray(record["r"][:n], dtype="<f8")
                    v = np.asarray(record["v"][:n], dtype="<f8")
                    if n < 2 or r.shape != (n, 3) or v.shape != r.shape:
                        raise ValueError(f"Invalid catalog state shape for ID {orbit_id}")
                    if not np.isfinite(r).all() or not np.isfinite(v).all():
                        raise ValueError(f"Nonfinite catalog states for ID {orbit_id}")
                    digest.update(str(orbit_id).encode("ascii") + b"\0")
                    digest.update(r.tobytes())
                    digest.update(v.tobytes())
                    tracks.append(r)
                    times.append(epoch_gps + np.arange(n) * STEP_SECONDS)
            sources.append({"url": url, "etag": etag,
                            "downloaded_bytes": remote.bytes_downloaded})
    if digest.hexdigest() != SAMPLE_SHA256:
        raise ValueError("LLNL catalog sample checksum changed; review the source before updating the pin")
    provenance = {
        "source": SOURCE,
        "dataset_citation": "https://doi.org/10.3847/2515-5172/adf8ec",
        "conventions_source": "https://arxiv.org/html/2512.11064v1",
        "orbit_ids": list(range(100)),
        "sample_sha256": digest.hexdigest(),
        "source_files": sources,
        "state_frame": "GCRF", "position_units": "m",
        "epoch_tt": EPOCH_TT, "sample_step_seconds": STEP_SECONDS,
        "time_convention": "Published dataset convention; timestamps are absent from these HDF5 arrays",
        "samples_per_orbit": [len(r) for r in tracks],
        "maximum_duration_days": 30,
        "raw_downloads_retained": False,
    }
    return tracks, times, provenance


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if not make_figures:
        return {"skipped": True, "reason": "figures_disabled; no download attempted"}
    try:
        r, t, provenance = load_catalog_sample()
    except (URLError, TimeoutError) as exc:
        print(f"Skipped cislunar catalog demo: public data unavailable: {exc}")
        return {"skipped": True, "reason": str(exc)}

    title = "LLNL catalog IDs 0–99 · up to 30 days, hourly samples\n1980-01-01 TT start · published trajectories"
    # Fast and full modes preserve the same pinned sample and time coverage.
    # gallery-call: demo_cislunar_catalog.jpg
    orbit_plot(r=r, t=t, view="cislunar", c="black", title=title,
               save="demo_cislunar_catalog.jpg")
    # gallery-end
    # gallery-call: demo_cislunar_catalog_xy.jpg
    orbit_plot(r=r, t=t, view="cislunar_xy", c="black", title=title,
               save="demo_cislunar_catalog_xy.jpg")
    # gallery-end
    path = Path(figpath("demo_cislunar_catalog_provenance.json"))
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"skipped": False, "provenance": provenance}


if __name__ == "__main__":
    main(make_figures=True)
