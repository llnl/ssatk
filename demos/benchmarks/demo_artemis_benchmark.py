#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from ssapy import Orbit, rv, SciPyPropagator, AccelKepler

from ssatk.io.ssatk_data import ssatk_data
from ssatk.plots.figpath import figpath
from ssatk.io.demo_data import ensure_demo_data_file
from ssatk.data import DataResourceNotFoundError, read_data_text
from ssatk.plots.orbit_plot import orbit_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "benchmarks"


def _find_csv(allow_download=True):
    return ensure_demo_data_file(
        "data/benchmarks/artemis2_orion_state_vectors.csv",
        allow_download=allow_download,
    )


def _load_orion_csv(csv_path: Path):
    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    required = [
        "JDTDB",
        "Calendar_Date_TDB",
        "X_km", "Y_km", "Z_km",
        "VX_km_s", "VY_km_s", "VZ_km_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cal = df["Calendar_Date_TDB"].astype(str).str.replace("A.D. ", "", regex=False)
    cal_dt = [datetime.strptime(x, "%Y-%b-%d %H:%M:%S.%f") for x in cal.tolist()]
    t = Time(cal_dt, scale="tdb")

    r_m = df[["X_km", "Y_km", "Z_km"]].to_numpy(dtype=float) * 1e3
    v_m_s = df[["VX_km_s", "VY_km_s", "VZ_km_s"]].to_numpy(dtype=float) * 1e3
    return t, r_m, v_m_s


def _make_high_fidelity_propagator():
    return SciPyPropagator(AccelKepler())


def _load_maneuvers():
    try:
        return json.loads(read_data_text("benchmarks/artemis_ii_maneuvers.json"))
    except (DataResourceNotFoundError, ModuleNotFoundError):
        return None


def _match_executed_maneuvers(t_ref, metadata):
    if not metadata:
        return [], []
    launch = Time(metadata["launch_utc"], scale="utc")
    matched = []
    indices = []
    for event in metadata["events"]:
        if event["status"] != "executed":
            continue
        event_time = launch + TimeDelta(event["met_s"], format="sec")
        index = int(np.argmin(np.abs(t_ref.tdb.gps - event_time.tdb.gps)))
        indices.append(index)
        matched.append(
            {
                **event,
                "match_status": event.get("data_status", "timing_only"),
                "utc": event_time.utc.isot,
                "matched_index": index,
                "sample_utc": t_ref[index].utc.isot,
                "sample_offset_s": float(t_ref[index].tdb.gps - event_time.tdb.gps),
            }
        )
    return sorted(set(indices)), matched


def _propagate_segment(r0, v0, t0, t_eval, propagator):
    orb = Orbit(r=np.asarray(r0, float), v=np.asarray(v0, float), t=t0.gps)
    r_prop, v_prop = rv(orb, time=t_eval, propagator=propagator)
    r_prop = np.asarray(r_prop, dtype=float).reshape((-1, 3))
    v_prop = np.asarray(v_prop, dtype=float).reshape((-1, 3))
    return r_prop, v_prop


def main(
    make_figures=None,
    fast=None,
    verbose=None,
    sync_threshold_km=50.0,
    allow_download=None,
    match_burns=True,
):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST
    if allow_download is None:
        allow_download = not UNDER_PYTEST

    csv_path = _find_csv(allow_download=allow_download)
    if csv_path is None:
        print("Skipping demo_artemis_benchmark: missing optional Artemis II state-vector data")
        return {
            "skipped": True,
            "reason": "missing_data_file",
            "csv_path": None,
        }

    t_ref, r_ref, v_ref = _load_orion_csv(csv_path)

    if fast:
        t_ref = t_ref[::2]
        r_ref = r_ref[::2]
        v_ref = v_ref[::2]

    propagator = _make_high_fidelity_propagator()
    sync_threshold_m = float(sync_threshold_km) * 1e3
    maneuver_metadata = _load_maneuvers() if match_burns else None
    match_burns = bool(match_burns and maneuver_metadata)
    maneuver_indices, matched_maneuvers = _match_executed_maneuvers(t_ref, maneuver_metadata)
    maneuver_index_set = set(maneuver_indices)
    burn_data_counts = {}
    for event in matched_maneuvers:
        status = event.get("data_status", "timing_only")
        burn_data_counts[status] = burn_data_counts.get(status, 0) + 1

    n = len(t_ref)
    r_model = np.zeros_like(r_ref)
    v_model = np.zeros_like(v_ref)
    r_model[0] = r_ref[0]
    v_model[0] = v_ref[0]

    sync_indices = [0]

    for i in range(n - 1):
        r_seg, v_seg = _propagate_segment(
            r_ref[i],
            v_ref[i],
            t_ref[i],
            t_ref[i:i+2],
            propagator,
        )

        r_model[i + 1] = r_seg[-1]
        v_model[i + 1] = v_seg[-1]

        dr = np.linalg.norm(r_model[i + 1] - r_ref[i + 1])

        if (match_burns and i + 1 in maneuver_index_set) or (
            not match_burns and dr > sync_threshold_m
        ):
            r_model[i + 1] = r_ref[i + 1]
            v_model[i + 1] = v_ref[i + 1]
            sync_indices.append(i + 1)

    dr_vec = r_model - r_ref
    dv_vec = v_model - v_ref
    dr_norm_m = np.linalg.norm(dr_vec, axis=1)
    dv_norm_m_s = np.linalg.norm(dv_vec, axis=1)

    result = {
        "csv_path": str(csv_path),
        "skipped": False,
        "times": t_ref,
        "r_truth": r_ref,
        "v_truth": v_ref,
        "r_model": r_model,
        "v_model": v_model,
        "dr_norm_m": dr_norm_m,
        "dv_norm_m_s": dv_norm_m_s,
        "sync_indices": np.asarray(sync_indices, dtype=int),
        "sync_times": t_ref[sync_indices],
        "rms_position_error_m": float(np.sqrt(np.mean(dr_norm_m**2))),
        "max_position_error_m": float(np.max(dr_norm_m)),
        "rms_velocity_error_m_s": float(np.sqrt(np.mean(dv_norm_m_s**2))),
        "max_velocity_error_m_s": float(np.max(dv_norm_m_s)),
        "n_syncs": int(len(sync_indices) - 1),
        "sync_threshold_km": float(sync_threshold_km),
        "sync_mode": "executed_maneuvers" if match_burns else "position_threshold",
        "maneuvers": matched_maneuvers,
        "burn_data_counts": burn_data_counts,
    }

    if make_figures:
        result_path = Path(ssatk_data("data/benchmarks/artemis_benchmark_results.json"))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "benchmark": "Artemis II / Orion",
                    "source": str(csv_path),
                    "model": "SSAPy SciPyPropagator(AccelKepler) with auto-sync",
                    "time_scale": "TDB",
                    "sample_count": int(n),
                    "rms_position_error_m": result["rms_position_error_m"],
                    "max_position_error_m": result["max_position_error_m"],
                    "rms_velocity_error_m_s": result["rms_velocity_error_m_s"],
                    "max_velocity_error_m_s": result["max_velocity_error_m_s"],
                    "n_syncs": result["n_syncs"],
                    "sync_threshold_km": result["sync_threshold_km"],
                    "sync_mode": result["sync_mode"],
                    "maneuvers": result["maneuvers"],
                    "burn_data_counts": result["burn_data_counts"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result["results_path"] = str(result_path)

    if verbose:
        print("Artemis / Orion benchmark")
        print(f"CSV: {csv_path}")
        print(f"Samples: {len(t_ref)}")
        print("Time scale assumed: TDB")
        print(f"Sync threshold [km]: {sync_threshold_km:.3f}")
        print(f"Number of sync events: {result['n_syncs']}")
        print(f"Burn data matches: {result['burn_data_counts']}")
        print(f"RMS position error [m]: {result['rms_position_error_m']:.3f}")
        print(f"Max position error [m]: {result['max_position_error_m']:.3f}")
        print(f"RMS velocity error [m/s]: {result['rms_velocity_error_m_s']:.6f}")
        print(f"Max velocity error [m/s]: {result['max_velocity_error_m_s']:.6f}")

    if make_figures:
        hours = (t_ref.gps - t_ref[0].gps) / 3600.0

        out1 = Path(figpath("benchmarks/artemis_benchmark_position_error"))
        if out1.suffix == "":
            out1 = out1.with_suffix(".png")
        out1.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dr_norm_m / 1e3, label="|Δr|")
        if len(sync_indices) > 1:
            ax.scatter(hours[sync_indices], dr_norm_m[sync_indices] / 1e3, color="red", label="sync")
        ax.set_xlabel("Time since start [hr]")
        ax.set_ylabel("Position error [km]")
        ax.set_title("Artemis benchmark: position error with auto-sync")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out1, dpi=200, bbox_inches="tight")
        plt.close(fig)

        out2 = Path(figpath("benchmarks/artemis_benchmark_velocity_error"))
        if out2.suffix == "":
            out2 = out2.with_suffix(".png")
        out2.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dv_norm_m_s, label="|Δv|")
        if len(sync_indices) > 1:
            ax.scatter(hours[sync_indices], dv_norm_m_s[sync_indices], color="red", label="sync")
        ax.set_xlabel("Time since start [hr]")
        ax.set_ylabel("Velocity error [m/s]")
        ax.set_title("Artemis benchmark: velocity error with auto-sync")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig)

        out4 = Path(figpath("benchmarks/artemis_benchmark_burn_events"))
        if out4.suffix == "":
            out4 = out4.with_suffix(".png")
        out4.parent.mkdir(parents=True, exist_ok=True)

        fig4, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax_pos.plot(hours, dr_norm_m / 1e3, color="#16324f", label="Position error")
        ax_vel.plot(hours, dv_norm_m_s, color="#b24c3a", label="Velocity error")
        for event in matched_maneuvers:
            index = int(event["matched_index"])
            event_hour = float(hours[index])
            label = event["name"].replace("_", " ")
            ax_pos.axvline(event_hour, color="#4c956c", alpha=0.65, lw=1.2)
            ax_vel.axvline(event_hour, color="#4c956c", alpha=0.65, lw=1.2)
            ax_pos.text(event_hour, ax_pos.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)
        ax_pos.set_ylabel("Position error [km]")
        ax_vel.set_ylabel("Velocity error [m/s]")
        ax_vel.set_xlabel("Time since launch [hr]")
        ax_pos.set_title("Artemis benchmark: executed burn matches and propagation residuals")
        for axis in (ax_pos, ax_vel):
            axis.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(out4, dpi=200, bbox_inches="tight")
        plt.close(fig4)

        out3 = Path(figpath("benchmarks/artemis_benchmark_cislunar_context"))
        if out3.suffix == "":
            out3 = out3.with_suffix(".png")
        out3.parent.mkdir(parents=True, exist_ok=True)

        step = 4 if not fast else 2
        r_truth_plot = r_ref[::step]
        r_model_plot = r_model[::step]
        t_plot = t_ref.gps[::step]

        fig3d, ax3d = orbit_plot(
            [r_truth_plot, r_model_plot],
            t=[t_plot, t_plot],
            view="cislunar_3d",
            figsize=(8, 8),
            fontsize=12,
            save_path=str(out3),
            show=False,
            legend=True,
            title="Artemis benchmark: truth vs model",
            c="white",
        )
        plt.close(fig3d)

        result["position_error_plot"] = str(out1)
        result["velocity_error_plot"] = str(out2)
        result["burn_events_plot"] = str(out4)
        result["cislunar_context_plot"] = str(out3)

    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True)
