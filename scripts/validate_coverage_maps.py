"""Independent orbit/ENU/event reference for the actual published coverage grid."""

from pathlib import Path
import json
import numpy as np
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, CartesianRepresentation
import astropy.units as u
from scipy.optimize import brentq
from scipy.ndimage import label, find_objects
from sgp4.api import Satrec
from ssatk.compute.coverage import access_statistics

# Reference routines below use direct SGP4, Astropy ITRS, and explicit ENU,
# not the SSATK grid, site, frame-transform or propagation-selection helpers.


def itrf(seconds):
    t = Time(epoch + np.atleast_1d(seconds), format="gps")
    error, r, _ = record.sgp4_array(t.utc.jd1, t.utc.jd2)
    assert not error.any()
    return (
        TEME(CartesianRepresentation(r.T * u.km), obstime=t)
        .transform_to(ITRS(obstime=t))
        .cartesian.xyz.to_value(u.m)
        .T
    )


def site_normal(lat, lon):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    phi, lam = np.radians([lat, lon])
    c, s = np.cos(phi), np.sin(phi)
    n = a / np.sqrt(1 - e2 * s * s)
    site = np.array([n * c * np.cos(lam), n * c * np.sin(lam), n * (1 - e2) * s])
    normal = np.array([c * np.cos(lam), c * np.sin(lam), s])
    return site, normal


def elevation(points, site, normal):
    dr = points - site
    return np.degrees(
        np.arcsin(
            np.clip(
                np.einsum("ij,j->i", dr, normal) / np.linalg.norm(dr, axis=1), -1, 1
            )
        )
    )


def reference_intervals(el, seconds, refine=None):
    # Connected above-mask runs, extended to their independently bracketed roots.
    segments = []
    for component in find_objects(label(el > 5)[0]):
        start, stop = component[0].start, component[0].stop
        if start == 0:
            left = seconds[0]
        elif refine:
            left = brentq(refine, seconds[start - 1], seconds[start], xtol=1e-5)
        else:
            left = np.interp(
                5, el[start - 1 : start + 1], seconds[start - 1 : start + 1]
            )
        if stop == len(seconds):
            right = seconds[-1]
        elif refine:
            right = brentq(refine, seconds[stop - 1], seconds[stop], xtol=1e-5)
        else:
            right = np.interp(
                5, el[stop - 1 : stop + 1][::-1], seconds[stop - 1 : stop + 1][::-1]
            )
        if right > left:
            segments.append([left, right])
    return np.asarray(segments, dtype=float).reshape(-1, 2)


def reference_metrics(intervals):
    spans = intervals[:, 1] - intervals[:, 0]
    gaps = np.diff(np.r_[0.0, intervals.ravel(), end])[::2]
    return np.array(
        [
            spans.sum() / end,
            spans.mean() / 60 if len(spans) else 0,
            spans.max() / 60 if len(spans) else 0,
            gaps.max() / 60,
            len(spans) / 3,
            spans.sum() / end * 1440,
        ]
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    folder = args.data_dir
    metadata = json.loads((folder / "coverage_metadata_hst.json").read_text())
    if (
        metadata["duration_days"],
        metadata["step_seconds"],
        metadata["minimum_elevation_deg"],
    ) != (3.0, 15.0, 5.0):
        raise ValueError(
            "This reference run expects the HST demo's 3-day, 15-second, 5-degree configuration"
        )
    global record, epoch, end
    tle = metadata["satellite"]
    record = Satrec.twoline2rv(tle["line1"], tle["line2"])
    epoch = Time(record.jdsatepoch, record.jdsatepochF, format="jd", scale="utc").gps
    end = 3 * 86400.0
    seconds = np.arange(end + 1)
    print("Computing one-second native-SGP4 / Astropy-ITRS reference", flush=True)
    positions = itrf(seconds)
    rows = []
    for name, lat, lon in [
        ("Cape Canaveral", 28.39, -80.60),
        ("Equator", 0, 0),
        ("North edge 41", 41, 30),
        ("North edge 43", 43, 30),
        ("North edge 44", 44, 30),
        ("Near tangent north", 45, 30),
        ("No access north", 60, 30),
        ("South edge", -43, -121),
    ]:
        site, normal = site_normal(lat, lon)
        el = elevation(positions, site, normal)
        exact = reference_intervals(
            el, seconds, lambda x: elevation(itrf(x), site, normal)[0] - 5
        )
        metrics = reference_metrics(exact)
        convergence = {}
        for step in [30, 15, 5, 1]:
            actual = access_statistics(el[::step], seconds[::step], 5)
            matches = actual["n_passes"] == len(exact)
            convergence[str(step)] = dict(
                contacts=actual["n_passes"],
                counts_match=matches,
                max_event_error_s=float(
                    np.max(np.abs(actual["intervals_seconds"] - exact))
                )
                if matches and len(exact)
                else None,
                contact_budget_error_s=float(
                    actual["contact_seconds"] - metrics[0] * end
                ),
                max_gap_error_s=float(actual["max_gap_min"] * 60 - metrics[3] * 60),
            )
        rows.append(
            dict(
                name=name,
                latitude_deg=lat,
                longitude_deg=lon,
                reference_contacts=len(exact),
                reference_intervals_s=exact.tolist(),
                convergence=convergence,
            )
        )
        print(name, convergence["15"], flush=True)
    # Compare all published cell metrics, using the direct-SGP4 trajectory and
    # explicit ellipsoid/ENU equations. No SSATK coverage/grid helper is called here.
    data = np.genfromtxt(folder / "coverage_grid_hst.csv", delimiter=",", names=True)
    coarse = seconds[::15]
    points = positions[::15]
    max_error = np.zeros(6)
    count_mismatches = 0
    for k, row in enumerate(data):
        site, normal = site_normal(row["latitude_deg"], row["longitude_deg"])
        ref = reference_metrics(
            reference_intervals(elevation(points, site, normal), coarse)
        )
        actual = np.array([row[name] for name in data.dtype.names[2:]])
        max_error = np.maximum(max_error, np.abs(actual - ref))
        count_mismatches += int(abs(actual[4] - ref[4]) > 1e-8)
        if k % 4000 == 0:
            print("Validated map cells", k, flush=True)
    assert count_mismatches == 0
    # Shared SGP4 backend with independent frame/geometry/event implementations:
    # this tolerance is a comparison allowance, not flight-tracking accuracy.
    assert np.all(max_error < [1e-7, 1e-4, 1e-4, 1e-4, 1e-8, 1e-4]), max_error
    result = dict(
        reference="Direct python-sgp4 + Astropy TEME-to-ITRS + explicit WGS84/ENU + scipy brentq events",
        limits="Same SGP4 dynamical model; not independent flight-tracking truth. One-second bracketing cannot rule out sub-second grazing contacts.",
        source_tle=tle,
        map_cells=len(data),
        map_metric_names=list(data.dtype.names[2:]),
        map_max_absolute_errors=max_error.tolist(),
        map_contact_count_mismatches=count_mismatches,
        convergence=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "convergence"}, indent=2))


if __name__ == "__main__":
    main()
