#!/usr/bin/env python3
"""
Hardcoded groundtrack validation using ssatk (+ssapy), no file I/O.

Uses only your utilities:
- ssatk.get_times, gcrf_to_lonlat
- ssatk.ssapy_wrapper.{best_prop, keplerian_prop}
- ssapy.Orbit, ssapy.rv
- ssatk.coordinates.geodetic.lonlat_distance
"""

import os
import sys
import numpy as np

from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.time_functions.get_times import get_times
from ssatk.coordinates.geodetic import gcrf_to_lonlat, lonlat_distance
from ssatk.ssapy_wrappers.ssapy_props import keplerian_prop, best_prop, threebody_prop

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(fast=None):
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")
    times = get_times(duration=(30 if fast else 24, "minute" if fast else "hour"), freq=(30 if fast else 60, "minute" if fast else "s"), t0=t0)

    orbit = Orbit.fromKeplerianElements(
        a=42164e3,
        e=0.01,
        i=np.radians(10.0),
        pa=0.0,
        raan=0.0,
        trueAnomaly=0.0,
        t=t0,
    )

    reference_prop = threebody_prop() if fast else best_prop()
    r_ref, v_ref = rv(orbit, times, propagator=reference_prop)
    r_test, v_test = rv(orbit, times, propagator=keplerian_prop())

    # gcrf_to_lonlat returns 3 values
    lon_ref, lat_ref, h_ref = gcrf_to_lonlat(r_ref, times)
    lon_test, lat_test, h_test = gcrf_to_lonlat(r_test, times)

    errs_km = lonlat_distance(lon_ref, lat_ref, lon_test, lat_test) / 1e3

    print(f"Mean error [km]: {np.mean(errs_km):.3f}")
    print(f"Max error  [km]: {np.max(errs_km):.3f}")

    n_preview = min(5, len(times))
    print("\nPreview (first few samples):")
    for i in range(n_preview):
        print(
            f"{times.iso[i]}  "
            f"ref(lat,lon)=({lat_ref[i]: .4f},{lon_ref[i]: .4f})  "
            f"test=({lat_test[i]: .4f},{lon_test[i]: .4f})  "
            f"err_km={errs_km[i]:.2f}"
        )

    return {
        "errs_km": errs_km,
        "lat_ref": lat_ref,
        "lon_ref": lon_ref,
        "h_ref": h_ref,
        "lat_test": lat_test,
        "lon_test": lon_test,
        "h_test": h_test,
    }


if __name__ == "__main__":
    main()
