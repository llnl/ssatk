"""Retain the launch verification ladder, including unavailable evidence."""

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from ssatk.launch.preflight import collect_software_baseline, verify_data_resources
from ssatk.propagators_6dof import propagate_6dof

GALLERY_INCLUDE = False


def _metric(name, value, unit, tolerance, reference):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative residual.")
    return dict(name=name, value=value, unit=unit, tolerance=tolerance,
                reference=reference, status="PASS" if value <= tolerance else "FAIL")


def mechanics_reference():
    """Constant acceleration and spherical torque-free spin, with analytic SI states."""
    times = np.linspace(0., 8., 33)
    position, velocity = np.array([7e6, 0., 0.]), np.array([1., 2., 3.])
    acceleration, omega = np.array([.2, -.1, .3]), np.array([1., 2., -1.])
    rate = np.linalg.norm(omega)
    expected_r = position + times[:, None]*velocity + .5*times[:, None]**2*acceleration
    expected_v = velocity + times[:, None]*acceleration
    expected_q = np.column_stack((np.cos(rate*times/2), np.sin(rate*times[:, None]/2)*omega/rate))
    runs = [propagate_6dof(
        times=times, r0=position, v0=velocity, omega0=omega, inertia=2*np.eye(3), mu=0.,
        acceleration=lambda t, r, v, q, omega: acceleration,
        rtol=rtol, atol=atol, max_step=step,
    ) for rtol, atol, step in ((1e-10, 1e-11, .125), (1e-12, 1e-13, .0625))]
    metrics = []
    for label, run in zip(("baseline", "refined"), runs):
        # Unit quaternions q and -q describe the same rotation; chord gives its angle.
        chord = np.minimum(np.linalg.norm(run.q-expected_q, axis=1), np.linalg.norm(run.q+expected_q, axis=1))
        for name, value, unit, budget in (
            ("position", np.max(np.linalg.norm(run.r-expected_r, axis=1)), "m", 1e-6),
            ("velocity", np.max(np.linalg.norm(run.v-expected_v, axis=1)), "m/s", 1e-9),
            ("attitude", np.max(4*np.arcsin(np.clip(chord/2, 0, 1))), "rad", 1e-8),
            ("body_rate", np.max(np.linalg.norm(run.omega-omega, axis=1)), "rad/s", 1e-12),
        ):
            metrics.append(_metric(f"{label}_{name}", value, unit, budget, "constant-acceleration / spherical free-spin closed form"))
    return {"metrics": metrics, "inputs": {"time_s": times, "r0_m": position, "v0_m_s": velocity,
            "acceleration_m_s2": acceleration, "omega0_rad_s": omega, "inertia_kg_m2": 2*np.eye(3),
            "mu_m3_s2": 0., "q0": [1., 0., 0., 0.], "frame": "nonrotating inertial"},
            "solver": [{"method": "DOP853", "rtol": r, "atol": a, "max_step_s": s}
                       for r, a, s in ((1e-10, 1e-11, .125), (1e-12, 1e-13, .0625))],
            "reference": {"r_m": expected_r, "v_m_s": expected_v, "q": expected_q},
            "runs": [{"r_m": run.r, "v_m_s": run.v, "q": run.q, "omega_rad_s": run.omega} for run in runs]}


def frame_reference():
    """Repeat the existing independent Astropy pad comparison with retained residuals."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers
    from ssatk.data import data_path
    from ssatk.launch.preflight import launch_initial_state

    epoch = Time("2020-05-30T19:22:00", scale="utc")
    state = launch_initial_state("Kennedy Space Center LC-39A", epoch, heading_deg=90.)
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    location = EarthLocation.from_geodetic(state["longitude_deg"]*u.deg,
                                         state["latitude_deg"]*u.deg, state["height_m"]*u.m)
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        r, v = location.get_gcrs_posvel(epoch)
    dr = np.asarray(state["gcrf_position_m"])-r.xyz.to_value(u.m)
    dv = np.asarray(state["gcrf_velocity_m_s"])-v.xyz.to_value(u.m/u.s)
    return {"inputs": state, "position_residual_gcrf_m": dr, "velocity_residual_gcrf_m_s": dv,
            "metrics": [_metric("position_component_max", np.max(np.abs(dr)), "m", 1., "Astropy ITRS to GCRS"),
                        _metric("velocity_component_max", np.max(np.abs(dv)), "m/s", 1e-3, "Astropy ITRS to GCRS")],
            "limits": ["Same packaged EOP; Astropy uses merged IERS-B and omits dX/dY, SSATK uses IERS-A with dX/dY",
                       "Budgets from test_launch_initialization; frame/model differences, not surveyed pad uncertainty"]}


def _launch_case(name):
    module = importlib.import_module(f"demos.launch.demo_launch_{name}")
    record = module.run_reference()
    metrics = []
    if name == "propulsion":
        metrics = [_metric("impulse", abs(record["impulse_n_s"]-11.), "N s", 1e-9, "fixed-nozzle force integral")]
        for key, unit in (("mass", "kg"), ("velocity", "m_s")):
            metrics.append(_metric(key, record[f"maximum_{key}_error_{unit}"], unit.replace("_", "/"),
                                   1e-9, "constant flow and ideal rocket equation"))
    elif name == "changing_mass":
        metrics = [_metric("mass", abs(record["consumed_mass_kg"]-2.), "kg", 1e-9, "constant axial flow"),
                   _metric("angular_momentum", record["maximum_angular_momentum_residual_kg_m2_s"],
                           "kg m2/s", 1e-9, "zero axial angular-momentum flux balance")]
    elif name == "aerodynamics":
        metrics = [_metric("force", record["maximum_force_error_n"], "N", 1e-6, "ideal gas and body coefficient formula"),
                   _metric("moment", record["maximum_moment_error_n_m"], "N m", 1e-6, "body coefficient formula")]
    elif name == "integration":
        units = {"position_m": "m", "velocity_m_s": "m/s", "quaternion": "1",
                 "body_rate_rad_s": "rad/s", "mass_kg": "kg", "event_time_s": "s", "attitude_rad": "rad"}
        for case in record["cases"]:
            for index, comparison in enumerate(case["comparisons"]):
                for key, budget in record["comparison_budgets"].items():
                    metrics.append(_metric(f"{case['name']}_{index}_{key}", comparison["errors"][key],
                                           units[key], budget, f"numerical refinement: {comparison['from']} to {comparison['to']}"))
            for index, run in enumerate(case["runs"]):
                evidence = run["record"]
                if case["name"] == "staging":
                    metrics.append(_metric(f"staging_{index}_velocity", evidence["velocity_error_m_s"],
                                           "m/s", 1e-6, "two-burn ideal rocket equation"))
                else:
                    metrics.append(_metric(f"pitch_{index}_mass", abs(evidence["mass_error_kg"]),
                                           "kg", 1e-8, "constant propellant flow"))
                for event_index, event in enumerate(evidence["events"]):
                    conservation = event.get("conservation")
                    if conservation is not None:
                        for key, unit in (("mass_kg", "kg"), ("linear_momentum_kg_m_s", "kg m/s"),
                                          ("angular_momentum_kg_m2_s", "kg m2/s")):
                            metrics.append(_metric(f"{case['name']}_{index}_event_{event_index}_{key}",
                                                   np.max(np.abs(conservation["residuals"][key])), unit,
                                                   conservation["tolerances"][key], "material momentum / mass reset balance"))
    elif name == "slosh":
        for index, unit in enumerate(("m", "m/s", "m", "m/s")):
            metrics.append(_metric(record["held_out"]["columns"][index], record["held_out"]["max_errors"][index],
                                   unit, record["held_out"]["budgets"][index], "independent two-mass spring/damper closed form"))
    return {"metrics": metrics, "evidence": record}


def _handoff(perturbed):
    from demos.launch.demo_launch_orbit_handoff import run_reference

    record = run_reference(perturbed=perturbed)
    comparison = record["comparison"]
    metrics = [_metric("position", comparison["overlap_position_m"], "m", comparison["position_budget_m"], "native SSAPy matched-force coast"),
               _metric("velocity", comparison["overlap_velocity_m_s"], "m/s", comparison["velocity_budget_m_s"], "native SSAPy matched-force coast"),
               _metric("force_assembly", comparison["acceleration_assembly_m_s2"], "m/s2", 1e-12, "single-count force sum")]
    if not perturbed:
        for key, unit, budget in (("position_m", "m", 1e-3), ("velocity_m_s", "m/s", 1e-6)):
            metrics.append(_metric(f"kepler_{key}", comparison["kepler_reference"][key], unit, budget, "Keplerian closed-form orbit"))
    return {"metrics": metrics, "evidence": record,
            "limits": ["Native SSAPy shares force implementations with the launch wrapper; this is integration/handoff evidence"]}


def run_verification(output_dir, *, external=True):
    """Write fresh evidence per case and a complete ladder, never hiding unavailable rows.

    This is a source-checkout verification command. The optional external
    checks reuse scripts.external_validation_report without installing tools.
    Set ``external=False`` to retain them as NOT RUN without loading any bridge.
    """
    from ssapy.ephemeris import get_planetary_ephemeris

    if sys.flags.optimize:
        raise RuntimeError("Run without -O: existing numerical references require their assertions.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, family, runner in (
        ("mechanics", "basic mechanics", mechanics_reference),
        ("frames", "frames and environment", frame_reference),
        *((name, family, lambda name=name: _launch_case(name)) for name, family in (
            ("propulsion", "propulsion"), ("aerodynamics", "frames and environment"),
            ("changing_mass", "changing mass rotation"))),
        ("slosh", "spacecraft coupling", lambda: _launch_case("slosh")),
        ("integration", "event sequencing and mission sequences", lambda: _launch_case("integration")),
        ("handoff_two_body", "orbit and finite burns", lambda: _handoff(False)),
        ("handoff_perturbed", "orbit and finite burns", lambda: _handoff(True)),
    ):
        if name == "integration" and external:
            from scripts.external_validation_report import run_external_validation

            print("Running existing external-tool comparisons", flush=True)
            external_report = run_external_validation(output_path=output_dir/"external.json", fast=True)
            for result in external_report["cases"]:
                status = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "UNAVAILABLE"}[result["status"]]
                cases.append(dict(id=result["id"], family="independent package comparison", comparison_mode="identical_physics",
                                  status=status, metrics=[{**m, "status": "PASS" if m["pass"] else "FAIL"}
                                                          for m in result.get("acceptance", [])],
                                  evidence="external.json", detail=result.get("skip", result.get("error")),
                                  sha256=hashlib.sha256((output_dir/"external.json").read_bytes()).hexdigest()))
        print(f"Running {name}", flush=True)
        case = dict(id=name, family=family, comparison_mode="identical_physics")
        try:
            result = runner()
            metrics = result["metrics"]
            case.update(status="PASS" if metrics and all(m["status"] == "PASS" for m in metrics) else "FAIL", metrics=metrics)
        except AssertionError as error:
            result = {"error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
            case.update(status="FAIL", metrics=[], **result)
        except Exception as error:
            result = {"error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
            case.update(status="ERROR", metrics=[], **result)
        evidence = output_dir / f"{name}.json"
        evidence.write_text(json.dumps(result, indent=2, allow_nan=False, default=lambda v: v.tolist())+"\n", encoding="utf-8")
        case["evidence"] = evidence.name
        case["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
        cases.append(case)

    if not external:
        for name in ("orekit_two_body", "gmat_two_body", "gmat_finite_burn", "basilisk_6dof"):
            cases.append(dict(id=name, family="independent package comparison", comparison_mode="identical_physics",
                              status="NOT RUN", metrics=[], reason="External comparisons explicitly disabled."))

    for name, family, mode, reason in (
        ("nasa_flight_cases", "NASA flight cases", "identical_physics",
         "No pinned dropped/tumbling-body or corrected Atmospheric Case 17 input and reference histories are packaged."),
        ("launch_package", "launch package comparison", "identical_physics",
         "No matched RocketPy, OpenRocket or JSBSim launch case with declared common envelope is supplied."),
        ("external_slosh", "spacecraft coupling", "identical_physics",
         "Basilisk rigid-body comparison does not cover slosh; no matched Basilisk/42 modal reference is supplied."),
        ("best_supported", "practical model comparison", "best_supported",
         "No preregistered vehicle, independently configured comparator, measured inputs or practical output budgets are supplied. Perturbed handoff is not this comparison."),
        ("flight_validation", "flight validation", "measured_observables",
         "Step 14: independent synchronized telemetry, instrument uncertainties and held-out cases are unavailable."),
    ):
        cases.append(dict(id=name, family=family, comparison_mode=mode, status="UNAVAILABLE", metrics=[], reason=reason))
    ephemeris = get_planetary_ephemeris()
    kernel = {"de440": "de440s.bsp", "de430": "de430_1900_2150.bsp"}[ephemeris]
    resources = verify_data_resources([{"path": path} for path in
                                       ("environment/eop/finals2000A.all", "egm84.egm", "egm84.egm.cof", kernel)], strict=False)
    statuses = {case["status"] for case in cases} | {resource["status"] for resource in resources}
    status = next((s for s in ("ERROR", "FAIL", "BLOCKED", "UNAVAILABLE", "NOT RUN") if s in statuses), "PASS")
    root = Path(__file__).resolve().parents[2]
    report = {"schema": "ssatk.launch_verification.v1", "status": status,
              "scope": "Synthetic launch verification ladder; incomplete independent/observed validation",
              "qualification_status": "NOT RUN", "software": collect_software_baseline({"ssatk": root}),
              "planetary_ephemeris": ephemeris,
              "resources": resources, "cases": cases,
              "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                for directory in ("ssatk", "demos", "scripts", "tests")
                                for p in sorted((root/directory).rglob("*.py"))},
              "summary": {s: sum(c["status"] == s for c in cases) for s in ("PASS", "FAIL", "ERROR", "UNAVAILABLE", "NOT RUN")},
              "limits": ["Synthetic budgets reuse component references; they are not mission acceptance tolerances",
                         "Numerical refinement, model discrepancy and measurement uncertainty are distinct",
                         "No cross-tool superiority, pad-to-orbit solution or flight qualification is established"]}
    (output_dir/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for JSON evidence; never overwritten")
    parser.add_argument("--no-external", action="store_true", help="Record external comparisons as NOT RUN without loading bridges")
    args = parser.parse_args(argv)
    report = run_verification(args.output, external=not args.no_external)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
