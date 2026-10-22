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

import matplotlib.pyplot as plt
import numpy as np
from ssapy import get_body

from demos.orbit_visualization._gdo_families import (
    EPOCH_TT, MAX_SAMPLES, SOURCE, STEP_SECONDS, describe_track, read_track,
)
from ssatk.constants import EARTH_RADIUS, MOON_RADIUS
from ssatk.plots._body_surfaces import BodySurface, body_rotation
from ssatk.plots.figpath import figpath
from ssatk.plots.plotutils import figsave

GALLERY_CATEGORY = "orbit_visualization"
TITLE = "GDO cislunar family candidates"
DESCRIPTION = (
    "Separate L4-like, L5-like, DRO-like and prograde lunar examples, selected "
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
COLORS = ("#166b9b", "#ad4f90", "#bc5b22", "#148170")
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
        earth, lunar, longitude, radius, metrics = describe_track(r, v, t)
        if family not in metrics["candidate_labels"]:
            raise ValueError(f"ID {orbit_id} no longer passes the {family} screen")
        tracks.append({"id": orbit_id, "family": family, "t": t, "earth": earth,
                       "lunar": lunar, "longitude": longitude, "radius": radius,
                       "metrics": metrics})
        sources.append({**source, "label": family, "metrics": metrics})
    provenance = {
        "source": SOURCE, "dataset_citation": "https://doi.org/10.3847/2515-5172/adf8ec",
        "conventions_source": "https://arxiv.org/html/2512.11064v1",
        "selection_search": SEARCH, "source_files": sources,
        "state_frame": "GCRF", "position_units": "m", "velocity_units": "m/s",
        "plot_frame": "Earth-Moon rotating axes, normalized by instantaneous Earth-Moon distance (LD)",
        "epoch_tt": EPOCH_TT, "sample_step_seconds": STEP_SECONDS,
        "samples_per_orbit": MAX_SAMPLES,
        "time_convention": "Published epoch and hourly cadence; timestamps absent from HDF5 arrays",
        "interpretation": "Finite-arc morphological candidates; no periodic-family or stability certification",
        "raw_downloads_retained": False,
    }
    return tracks, provenance


def _style_axes(ax):
    ax.set_xlabel("x / LD", labelpad=3)
    ax.set_ylabel("y / LD", labelpad=3)
    ax.set_zlabel("z / LD", labelpad=3)
    ax.tick_params(labelsize=8)
    ax.set_proj_type("ortho")
    ax.view_init(elev=25, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.grid(True, alpha=.2)


def plot_atlas(tracks):
    fig = plt.figure(figsize=(13, 11), facecolor="white")
    fig.suptitle("GDO cislunar family candidates", fontsize=22, x=.07, ha="left", y=.98)
    fig.text(.07, .945, "Four distinct trajectories • 180-day behavior screens • Earth–Moon rotating axes", fontsize=11)
    epoch = tracks[0]["t"][0]
    moon = get_body("moon")
    distance = np.linalg.norm(moon.position(epoch))
    for j, (track, color) in enumerate(zip(tracks, COLORS)):
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        local = j >= 2
        points = track["lunar"][:721] if local else track["earth"]
        label = "Retrograde lunar (DRO-like)" if j == 2 else track["family"]
        ax.set_title(f"{label}  ·  ID {track['id']}\n{'30 days shown; 180 days screened' if local else '180 days shown and screened'}", loc="left", fontsize=12, pad=12)
        ax.plot(*points.T, color=color, linewidth=.75, alpha=.85)
        ax.scatter(*points[0], color=color, s=24, marker="o", label="Start")
        center = np.zeros(3) if local else np.array([1., 0., 0.])
        BodySurface(ax, "moon", MOON_RADIUS / distance, center=center,
                    rotation=body_rotation("moon", epoch, "lunar"))
        bounds = np.vstack((points, center))
        if not local:
            BodySurface(ax, "earth", EARTH_RADIUS / distance,
                        rotation=body_rotation("earth", epoch, "lunar"))
            lp = [.5, (1 if j == 0 else -1) * np.sqrt(3) / 2, 0]
            ax.scatter(*lp, marker="+", s=65, color="#444444")
            ax.text(*lp, "  L4" if j == 0 else "  L5", fontsize=9)
            ax.text(0, 0, -.08, "Earth", fontsize=8)
            ax.text(1, 0, -.08, "Moon", fontsize=8)
            bounds = np.vstack((bounds, [0, 0, 0]))
        extent = np.max(np.ptp(bounds, axis=0)) * 1.15
        middle = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
            setter(c - extent / 2, c + extent / 2)
        ax.set_box_aspect((1, 1, 1))
        _style_axes(ax)
    fig.subplots_adjust(left=.03, right=.97, bottom=.11, top=.88, hspace=.26, wspace=.08)
    fig.text(.07, .064, "L1 / L2 NRHO: no accepted candidate in the 172-trajectory screen. This is not an exhaustive catalog search.", fontsize=10)
    fig.text(.07, .039, "Candidate shapes, not certified periodic families. Bodies use physical radii and map orientation at the 1980-01-01 TT start.", fontsize=9, color="#555555")
    return fig


def plot_evidence(tracks):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    fig.suptitle("Why these GDO trajectories were selected", fontsize=21, x=.07, ha="left", y=.97)
    days = np.arange(MAX_SAMPLES) / 24
    for track, color in zip(tracks[:2], COLORS[:2]):
        axes[0, 0].plot(days, track["longitude"], color=color, lw=1, label=f"{track['family']} · {track['id']}")
    for angle in (-60, 60):
        axes[0, 0].axhline(angle, color="#aaaaaa", ls=":", lw=1)
    axes[0, 0].set(title="Bounded longitude near L4 and L5", ylabel="Earth-centered angle from Moon [deg]")
    axes[0, 0].legend(fontsize=9)
    for track, color in zip(tracks[2:], COLORS[2:]):
        axes[0, 1].plot(days, track["radius"], color=color, lw=.6, alpha=.8, label=f"{track['family']} · {track['id']}")
        p = track["lunar"]
        winding = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
        axes[1, 0].plot(days, (winding - winding[0]) / (2 * np.pi), color=color, lw=1.3, label=track["family"])
    axes[0, 1].set(title="Bounded lunar distance", ylabel="Moon-centered radius / LD")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].set(title="Opposite directions around the Moon", ylabel="Signed turns in rotating XY plane")
    axes[1, 0].legend(fontsize=9)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Days from 1980-01-01 TT")
        ax.set_xlim(0, 180)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0, 1, "Coverage and limits", fontsize=15, va="top", transform=ax.transAxes)
    lines = ["✓  L4-like and L5-like libration", "✓  Retrograde lunar (DRO-like) and prograde lunar",
             "—  L1 halo / NRHO: unfilled", "—  L2 halo / NRHO: unfilled", "",
             "1,000,000 descriptive records streamed; 172 unique", "long-lived candidates checked over 180 days.", "",
             "Initial proximity only proposes candidates; sustained", "rotating-frame behavior determines these labels.", "",
             "The screen can miss unstable or differently phased", "families. No match does not mean absent from GDO."]
    ax.text(0, .88, "\n".join(lines), fontsize=11, linespacing=1.55, va="top", transform=ax.transAxes)
    fig.subplots_adjust(left=.08, right=.97, bottom=.10, top=.87, hspace=.37, wspace=.27)
    fig.text(.08, .035, "Hourly published states • ephemeral downloads • pinned source ETags and state SHA256 • criteria and IDs in provenance", fontsize=9, color="#555555")
    return fig


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
    # Fast and full builds use identical pinned states and classification windows.
    # gallery-call: demo_cislunar_catalog.jpg
    if figsave(plot_atlas(tracks), "demo_cislunar_catalog.jpg") is None:
        raise RuntimeError("Could not save the cislunar family atlas")
    # gallery-end
    # gallery-call: demo_cislunar_catalog_xy.jpg
    if figsave(plot_evidence(tracks), "demo_cislunar_catalog_xy.jpg") is None:
        raise RuntimeError("Could not save the cislunar selection evidence")
    # gallery-end
    Path(figpath("demo_cislunar_catalog_provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"skipped": False, "provenance": provenance}


if __name__ == "__main__":
    main(make_figures=True)
