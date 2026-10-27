"""Audit the GDO demo selection without retaining downloaded catalog data.

Run from the repository root with PYTHONPATH=. This optional, more expensive
search streams the descriptive CSV, then reads 180-day arcs for the nearest
40 six-year survivors to each of L4, L5, Moon, L1 and L2 at the initial epoch.
It is a bounded candidate search, not an exhaustive dynamical classification
of the million trajectories. The gallery itself reads only its pinned IDs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from astropy.time import Time
from ssapy import get_body

from demos.orbit_visualization._gdo_families import (
    EPOCH_TT, FAMILIES, SOURCE, describe_track, read_track, source_url,
)
from ssatk.coordinates.lunar import gcrf_to_lunar

CSV_ETAG = '"2b122cd1-62f74f9cdf480"'
CENTERS = {"L4": [.5, np.sqrt(3) / 2, 0], "L5": [.5, -np.sqrt(3) / 2, 0],
           "Moon": [1, 0, 0], "L1 vicinity": [.84, 0, 0], "L2 vicinity": [1.16, 0, 0]}


def candidate_ids():
    """Initial-position screen only; no initial-position family labels."""
    epoch = Time(EPOCH_TT, scale="tt").gps
    moon = get_body("moon")
    distance = np.linalg.norm(moon.position(epoch))
    nearest = {key: [] for key in CENTERS}
    rows = 0
    url = SOURCE + "descriptive_data.csv"
    with urlopen(Request(url, headers={"If-Match": CSV_ETAG}), timeout=30) as response:
        if response.headers.get("ETag") != CSV_ETAG:
            raise ValueError("Descriptive catalog changed; review before updating its pin")
        for chunk in pd.read_csv(response, chunksize=25000):
            rows += len(chunk)
            survivors = chunk[chunk.lifetime > 5.9]
            if survivors.empty:
                continue
            r = np.array([np.fromstring(value.strip("[]"), sep=" ") for value in survivors.r0])
            xyz = gcrf_to_lunar(r, epoch) / distance
            for name, center in CENTERS.items():
                distances = np.linalg.norm(xyz - center, axis=1)
                nearest[name].extend((float(distances[j]), int(survivors.iloc[j].orb_id))
                                     for j in np.argsort(distances)[:40])
                nearest[name] = sorted(nearest[name])[:40]
    return sorted({oid for values in nearest.values() for _, oid in values}), {
        "csv_url": url, "csv_etag": CSV_ETAG, "catalog_rows": rows,
        "lifetime_cut_years": 5.9, "nearest_per_region": 40,
        "initial_centers_ld": CENTERS, "nearest_ids": {k: [i for _, i in v] for k, v in nearest.items()},
    }


def inspect(orbit_id):
    with urlopen(Request(source_url(orbit_id), method="HEAD"), timeout=30) as response:
        etag = response.headers["ETag"]
    r, v, t, source = read_track(orbit_id, etag)
    *_, metrics = describe_track(r, v, t)
    return {**source, "metrics": metrics}


def rank(record, family):
    m = record["metrics"]
    if family.startswith(("L4", "L5")):
        low, high = m["longitude_minmax_deg"]
        return high - low, record["orbit_id"]
    return m["moon_radius_ratio"], record["orbit_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Metadata-only JSON audit destination")
    args = parser.parse_args()
    ids, search = candidate_ids()
    print(f"Screening {len(ids)} unique candidates", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(inspect, oid) for oid in ids]
        for future in as_completed(futures):
            records.append(future.result())
            print(f"Checked {len(records)}/{len(ids)}", flush=True)
    chosen = {}
    for family in FAMILIES:
        matches = [r for r in records if family in r["metrics"]["candidate_labels"]]
        chosen[family] = min(matches, key=lambda r: rank(r, family)) if matches else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"search": search, "selected": chosen,
                                      "candidates": sorted(records, key=lambda r: r["orbit_id"]),
                                      "raw_downloads_retained": False}, indent=2) + "\n")
    print({name: r["orbit_id"] if r else None for name, r in chosen.items()})


if __name__ == "__main__":
    main()
