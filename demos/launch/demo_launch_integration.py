"""Numerical convergence of the existing synthetic staging and pitch references."""

import argparse
import json
from pathlib import Path

import numpy as np

from demos.launch.demo_launch_events import run_reference as staging_reference
from demos.launch.demo_launch_guidance import run_reference as pitch_reference
from ssatk.launch.preflight import collect_software_baseline

GALLERY_INCLUDE = False


def _samples(record):
    samples = {}
    for arc in record["truth"]:
        for t, r, v, q, omega, mass in zip(arc["time_gps_s"], arc["position_gcrf_m"],
                arc["velocity_gcrf_m_s"], arc["q_body_to_gcrf"], arc["omega_body_rad_s"], arc["mass_kg"]):
            # At a duplicated boundary retain the post-reset state.
            samples[float(t)] = np.r_[r, v, q, omega, mass]
    return samples


def _compare(first, second):
    left, right = _samples(first), _samples(second)
    times = sorted(left.keys() & right.keys())
    left_rows, right_rows = [left[t] for t in times], [right[t] for t in times]
    first_events, second_events = first["events"], second["events"]
    assert [(e["event"], e["pre_phase"], e["post_phase"]) for e in first_events] == [
        (e["event"], e["pre_phase"], e["post_phase"]) for e in second_events], "event topology changed"
    for first_event, second_event in zip(first_events, second_events):
        assert len(first_event["children"]) == len(second_event["children"]), "separation topology changed"
        for rows, event in ((left_rows, first_event), (right_rows, second_event)):
            for snapshot in (event["pre_state"], event["post_state"], *event["children"]):
                rows.append(np.r_[snapshot["position_gcrf_m"], snapshot["velocity_gcrf_m_s"],
                                  snapshot["q_body_to_gcrf"], snapshot["omega_body_rad_s"], snapshot["mass_kg"]])
    a, b = np.array(left_rows), np.array(right_rows)
    q_error = np.minimum(np.linalg.norm(a[:, 6:10]-b[:, 6:10], axis=1),
                         np.linalg.norm(a[:, 6:10]+b[:, 6:10], axis=1))
    return {"common_samples": len(times),
            "position_m": float(np.max(np.linalg.norm(a[:, :3]-b[:, :3], axis=1))),
            "velocity_m_s": float(np.max(np.linalg.norm(a[:, 3:6]-b[:, 3:6], axis=1))),
            "quaternion": float(np.max(q_error)),
            "body_rate_rad_s": float(np.max(np.linalg.norm(a[:, 10:13]-b[:, 10:13], axis=1))),
            "mass_kg": float(np.max(np.abs(a[:, 13]-b[:, 13]))),
            "event_time_s": float(max(abs(a["time_gps_s"]-b["time_gps_s"])
                                      for a, b in zip(first_events, second_events)))}


def run_reference():
    """R1/R4: rocket equation, sampled pitch, state health and numerical refinement.

    Acceptance budgets are SI except dimensionless quaternion distance and
    norm error. They apply only to these synthetic models and fixed seed.
    """
    atol = np.array([1e-6]*3 + [1e-9]*3 + [1e-11]*4 + [1e-10]*3 + [1e-10])
    budgets = {"position_m": 1e-4, "velocity_m_s": 1e-6, "quaternion": 1e-7,
               "body_rate_rad_s": 1e-7, "mass_kg": 1e-8, "event_time_s": 1e-7}
    cases = []
    for name, reference, base_step, output_samples in (
            ("staging", staging_reference, .25, 141), ("pitch", pitch_reference, .1, 151)):
        runs = []
        for label, rtol, scale, step in (("baseline", 1e-10, 1., base_step),
                ("tighter_tolerance", 1e-12, .01, base_step),
                ("smaller_step", 1e-12, .01, base_step/2),
                ("confirmation", 1e-13, .001, base_step/4)):
            solver = {"method": "DOP853", "rtol": rtol, "atol": atol*scale, "max_step": step}
            record = reference(solver=solver)
            diagnostics = [d for arc in record["truth"] for d in arc["integration_diagnostics"]]
            norm_error = max(max(d["max_output_quaternion_norm_error"],
                                 d["max_endpoint_quaternion_norm_error"]) for d in diagnostics)
            assert norm_error < 1e-8, "raw quaternion norm error, dimensionless"
            assert all(d["finite_output_states"] for d in diagnostics), "nonfinite physical state"
            assert all(d["max_step_s"] <= step+1e-12 for d in diagnostics), "step bound, s"
            runs.append({"label": label, "solver": solver, "record": record,
                         "nfev": sum(arc["nfev"] for arc in record["truth"]),
                         "accepted_steps": sum(d["accepted_steps"] for d in diagnostics),
                         "max_raw_quaternion_norm_error": norm_error,
                         "max_output_quaternion_correction": max(d["max_output_quaternion_correction"] for d in diagnostics)})
        comparisons = []
        for first, second in ((runs[0], runs[1]), (runs[1], runs[2]), (runs[2], runs[3]), (runs[0], runs[3])):
            errors = _compare(first["record"], second["record"])
            assert errors["common_samples"] >= output_samples, "missing common requested output epochs"
            assert all(errors[key] <= limit for key, limit in budgets.items()), f"{name} convergence: {errors}"
            comparisons.append({"from": first["label"], "to": second["label"], "errors": errors})
        cases.append({"name": name, "runs": runs, "comparisons": comparisons})
    return {"scope": "STEP 10 SYNTHETIC NUMERICAL INTEGRATION STUDY", "status": "PASS",
            "qualification_status": "NOT RUN", "software": collect_software_baseline(),
            "state_order": ["r_m (3)", "v_m_s (3)", "q (4)", "omega_rad_s (3)", "mass_kg (1)"],
            "comparison_budgets": budgets, "raw_quaternion_norm_budget": 1e-8, "cases": cases,
            "limits": ["Numerical evidence for the declared synthetic staging and nominal pitch models only",
                       "Tolerances are local error scales; reference residuals and refinement constrain global outputs",
                       "State health is sampled at output and accepted endpoints, not bounded continuously",
                       "No reset-crossing STM is provided; sensitivities need reset and event-time terms",
                       "No measured vehicle calibration, stiff/slosh/flexible dynamics or mission qualification"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON evidence file, never overwritten")
    args = parser.parse_args(argv)
    encoded = json.dumps(run_reference(), indent=2, allow_nan=False, default=lambda v: v.tolist())+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
