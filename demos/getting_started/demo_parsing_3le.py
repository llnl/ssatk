#!/usr/bin/env python3

import os
import sys

from ssatk.io.demo_data import ensure_demo_data_file
from ssatk.io.read_3le import read_3le
from ssatk.io.read_3le_by_bit import read_3le_by_bit
from ssatk.io.tle_iter_pairs import tle_iter_pairs
from ssatk.io.tle_prop_to_time import tle_prop_to_time
from ssatk.io.pprint_utils import pprint

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(verbose=False, fast=False, allow_download=None):
    if allow_download is None:
        allow_download = not UNDER_PYTEST

    tle_path = ensure_demo_data_file("full_catalog_3le.txt", allow_download=allow_download)
    if tle_path is None:
        print("Skipping demo_parsing_3le: missing optional 3LE catalog data")
        return {
            "data": None,
            "skipped": True,
            "reason": "missing_data_file",
            "tle_path": None,
        }

    tle_path = str(tle_path)

    print(f"DATA: {tle_path}")

    data = read_3le(tle_path, verbose=False)
    if verbose:
        pprint(data)

    pair_iter = tle_iter_pairs(tle_path)
    first_pairs = []
    for idx, triple in enumerate(pair_iter):
        first_pairs.append(triple)
        if fast and idx >= 2:
            break

    from ssapy.orbit import Orbit

    orbs = []
    for name, line1, line2 in first_pairs:
        orb = Orbit.fromTLETuple((line1, line2))
        orbs.append(orb)

    orbits_at_t, names, R, V = tle_prop_to_time(
        "2025-01-01T00:00:00",
        tle_path,
        validate_checksum=False,
        truncate=False,
        return_arrays=True,
    )

    return {
        "data": data,
        "sample_pairs": first_pairs,
        "sample_orbits": orbs,
        "propagated_orbits": orbits_at_t,
        "skipped": False,
        "tle_path": tle_path,
    }


if __name__ == "__main__":
    main(verbose=True, fast=False)
