"""Reproducible morphology screens, not periodic-family certification.

All cuts below are demo selection heuristics in instantaneous Earth–Moon
distance units (LD). A finite ephemeris arc cannot establish a CR3BP family.
"""
from __future__ import annotations

import hashlib

import h5py
import numpy as np
from astropy.time import Time
from scipy.signal import find_peaks
from ssatk import get_body

from demos.orbit_visualization._remote_hdf5 import TemporaryHTTPFile
from ssatk.coordinates.lunar import gcrf_to_lunar, gcrf_to_lunar_fixed

SOURCE = "https://gdo-cislunar.llnl.gov/"
BASE_URL = SOURCE + "cislunar_data_six_year_1.0GEO_to_18.2GEO/"
EPOCH_TT = "1980-01-01T00:00:00"
STEP_SECONDS = 3600.0
MAX_SAMPLES = 4321  # 180 days, with both endpoints.
FAMILIES = ("L4-like", "L5-like", "DRO-like", "Prograde lunar",
            "L1 halo/NRHO candidate", "L2 halo/NRHO candidate")


def source_url(orbit_id):
    first = orbit_id // 50 * 50
    return BASE_URL + f"orb_id_{first}_to_{first + 49}.h5"


def read_track(orbit_id, etag):
    """Read one pinned 180-day arc; discard downloaded blocks before returning."""
    url = source_url(orbit_id)
    with TemporaryHTTPFile(url, expected_etag=etag, block_size=256 * 1024) as remote:
        with h5py.File(remote, "r") as data:
            record = data[str(orbit_id)]
            r = np.asarray(record["r"][:MAX_SAMPLES], dtype="<f8")
            v = np.asarray(record["v"][:MAX_SAMPLES], dtype="<f8")
        downloaded = remote.bytes_downloaded
    if r.shape != (MAX_SAMPLES, 3) or v.shape != r.shape:
        raise ValueError(f"ID {orbit_id} lacks the complete 180-day state window")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError(f"Nonfinite states for ID {orbit_id}")
    digest = hashlib.sha256(str(orbit_id).encode("ascii") + b"\0")
    digest.update(r.tobytes())
    digest.update(v.tobytes())
    t = Time(EPOCH_TT, scale="tt").gps + np.arange(len(r)) * STEP_SECONDS
    return r, v, t, {"orbit_id": orbit_id, "url": url, "etag": etag,
                     "sample_sha256": digest.hexdigest(), "downloaded_bytes": downloaded}


def describe_track(r, v, t):
    """Return rotating coordinates and numerical evidence for candidate labels.

    Longitude is geometric Earth-centered angle relative to the Moon, not
    osculating mean longitude. Moon-centered winding and angular momentum
    use rotating velocities, including the frame-rate term.
    """
    moon = get_body("moon")
    distance = np.linalg.norm(moon.position(t).T, axis=1)
    earth = gcrf_to_lunar(r, t) / distance[:, None]
    lunar_m, lunar_v = gcrf_to_lunar_fixed(r, t, v=v)
    lunar = lunar_m / distance[:, None]
    longitude = np.rad2deg(np.unwrap(np.arctan2(earth[:, 1], earth[:, 0])))
    winding = np.unwrap(np.arctan2(lunar[:, 1], lunar[:, 0]))
    radius = np.linalg.norm(lunar, axis=1)
    q05, q95 = np.quantile(radius, [.05, .95])
    h = lunar[:, 0] * lunar_v[:, 1] - lunar[:, 1] * lunar_v[:, 0]
    # Reject small numerical extrema: peaks need 0.005 LD prominence and
    # at least 24 hourly samples of separation.
    perilunes, _ = find_peaks(-radius, prominence=.005, distance=24)
    metrics = {
        "duration_days": float((t[-1] - t[0]) / 86400),
        "longitude_minmax_deg": [float(longitude.min()), float(longitude.max())],
        "longitude_mean_deg": float(longitude.mean()),
        "earth_rho_q05q95_ld": np.quantile(np.linalg.norm(earth[:, :2], axis=1), [.05, .95]).tolist(),
        "moon_radius_minmax_ld": [float(radius.min()), float(radius.max())],
        "moon_radius_q05q95_ld": [float(q05), float(q95)],
        "moon_radius_ratio": float(q95 / q05),
        "moon_winding_turns": float((winding[-1] - winding[0]) / (2 * np.pi)),
        "retrograde_fraction": float(np.mean(h < 0)),
        "positive_x_fraction": float(np.mean(lunar[:, 0] > 0)),
        "z_range_ld": float(np.ptp(lunar[:, 2])),
        "perilune_count": len(perilunes),
    }
    for label, sign in (("L4", 1), ("L5", -1)):
        delta = earth - [0.5, sign * np.sqrt(3) / 2, 0]
        metrics[f"{label}_distance_q50q95_ld"] = np.quantile(np.linalg.norm(delta, axis=1), [.5, .95]).tolist()
    metrics["candidate_labels"] = classify(metrics)
    return earth, lunar, longitude, radius, metrics


def classify(m):
    """Conservative finite-arc morphology cuts; thresholds are not family definitions."""
    if m["duration_days"] < 180:
        return []
    labels = []
    low, high = m["longitude_minmax_deg"]
    rho05, rho95 = m["earth_rho_q05q95_ld"]
    for label, sign in (("L4", 1), ("L5", -1)):
        d50, d95 = m[f"{label}_distance_q50q95_ld"]
        if (high - low < 180 and 30 < sign * m["longitude_mean_deg"] < 90
                and .8 < rho05 < rho95 < 1.2 and d50 < .5 and d95 < .8
                and (low > 0 if sign == 1 else high < 0)):
            labels.append(label + "-like")
    dmin, dmax = m["moon_radius_minmax_ld"]
    d05, d95 = m["moon_radius_q05q95_ld"]
    turns = m["moon_winding_turns"]
    if (.04 < d05 < d95 < .35 and dmax < .4 and m["moon_radius_ratio"] < 6
            and turns < -3 and m["retrograde_fraction"] > .9):
        labels.append("DRO-like")
    if dmax < .4 and turns > 3 and m["retrograde_fraction"] < .1:
        labels.append("Prograde lunar")
    # This is only a search flag. Even a match needs further dynamical analysis
    # before asserting a halo or near-rectilinear halo orbit (NRHO) family.
    if (dmin > .009 and dmax < .35 and dmax / dmin >= 5
            and m["z_range_ld"] > .01 and m["perilune_count"] >= 3):
        if m["positive_x_fraction"] < .3:
            labels.append("L1 halo/NRHO candidate")
        if m["positive_x_fraction"] > .7:
            labels.append("L2 halo/NRHO candidate")
    return labels
