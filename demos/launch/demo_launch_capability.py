"""Freeze and reproduce the demonstrated launch capability, without external bridges."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import time
import uuid

from demos.launch.demo_launch_calibration import run_reference, sample_dataset
from demos.launch.demo_launch_verification import run_verification
from ssatk.launch.preflight import collect_software_baseline, verify_data_resources

GALLERY_INCLUDE = False
ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_bundle(directory):
    """Check the complete byte manifest; integrity alone is not numerical validation."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ssatk.launch_capability_manifest.v1" or not manifest.get("files"):
        raise ValueError("expected a nonempty launch capability manifest")
    files = manifest["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("bundle files differ from the manifest")
    for name, expected in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("bundle paths must remain inside the bundle")
        payload = path.read_bytes()
        if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError(f"bundle fingerprint mismatch: {name}")
    return json.loads((directory / "capability.json").read_text(encoding="utf-8"))


def build_bundle(output, *, replay=None):
    """Run existing references and freeze source, inputs, outputs and missing evidence.

    Requires a source checkout with every change staged (or committed). Replay
    checks the recorded Git tree, Python/dependency versions and data hashes
    before rerunning; scientific acceptance uses the original reference budgets,
    not byte equality of floating-point results or timing metadata.
    """
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("write the capability bundle outside the source checkout")
    if output.exists():
        raise FileExistsError(output)
    if _git("diff", "--name-only").strip() or _git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("stage or commit all source changes before freezing the capability")
    source = {"repository": "https://github.com/LLNL/ssatk",
              "revision": _git("rev-parse", "HEAD").decode().strip(),
              "tree": _git("write-tree").decode().strip(), "patch": "source.patch"}
    patch = _git("diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", "HEAD")
    packages = tuple(sorted({d.metadata["Name"] for d in distributions() if d.metadata["Name"]}))
    software = collect_software_baseline(packages=packages)
    software["python"].pop("executable", None)
    dataset = sample_dataset()
    previous = None
    if replay is not None:
        replay = Path(replay).resolve()
        previous = verify_bundle(replay)
        if source["tree"] != previous["source"]["tree"]:
            raise ValueError("replay requires the recorded source tree; apply source.patch to the recorded revision")
        for field in ("python", "packages"):
            if software[field] != previous["software"][field]:
                raise ValueError(f"replay requires the recorded {field}")
        verify_data_resources(previous["resources"])
        dataset = json.loads((replay / "calibration-input.json").read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=False)
    _write(output / "calibration-input.json", dataset)
    (output / "source.patch").write_bytes(patch)
    (output / "requirements.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in software["packages"].items()
                if name.lower().replace("_", "-") != "ssatk"), encoding="utf-8",
    )
    started = time.perf_counter()
    with (output / "run.log").open("x", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        verification = run_verification(output / "verification", external=False)
        calibration = run_reference(dataset=dataset)
    elapsed = time.perf_counter() - started
    _write(output / "calibration.json", calibration)
    # A changed checkout cannot describe the code that produced all these outputs.
    if (_git("write-tree").decode().strip() != source["tree"] or _git("diff", "--name-only").strip()
            or _git("ls-files", "--others", "--exclude-standard").strip()):
        raise RuntimeError("source changed during execution; incomplete bundle retained")

    cases = [{**case, "evidence": "verification/" + case["evidence"]} if "evidence" in case else dict(case)
             for case in verification["cases"]]
    for case in calibration["cases"]:
        cases.append({"id": "calibration_" + case["id"], "family": "synthetic measurement calibration",
                      "status": case["status"], "evidence": "calibration.json",
                      "detail": "Injected sensor fault must remain FAIL" if case["id"] == "failed_sensor"
                                else "Rocket equation and rotating sensor; no flight measurements"})
    for name, reason in (
        ("leo_mission", "No approved vehicle/mission dataset or pad-to-orbit acceptance run."),
        ("geo_mission", "No transfer, circularization and final GEO acceptance run."),
        ("mission_dispersions", "No mission uncertainty distributions or probability requirements."),
        ("browser_and_all_missions", "Outside this numerical capability; no full-gallery or browser claim."),
    ):
        cases.append(dict(id=name, status="NOT RUN", reason=reason))
    numerical = [c for c in verification["cases"] if "evidence" in c]
    fitted = calibration["parameters"]
    calibration_checks = {
        "fit": calibration["calibration_status"] == "PASS",
        "thrust_reference": abs(fitted["thrust_n"] - 10.) <= 2e-6,
        "bias_reference": abs(fitted["velocity_bias_m_s"] - .04) <= 2e-7,
        "held_out": [c["status"] for c in calibration["cases"]] == ["PASS", "PASS", "FAIL"],
        "measurement_limit": calibration["measurement_evidence_status"] == "UNAVAILABLE",
    }
    demonstrated = (bool(numerical) and all(c["status"] == "PASS" for c in numerical)
                    and all(r["status"] == "PASS" for r in verification["resources"])
                    and all(calibration_checks.values()))
    report = {
        "schema": "ssatk.launch_capability.v1", "run_id": str(uuid.uuid4()),
        "status": "PASS" if demonstrated else "FAIL",
        "scope": "Reproducible synthetic component, event/control, upper-stage/coast and calibration cases",
        "mission_class": "Specified Earth-centred upper-stage and isolated launch subsystem examples",
        "fidelity": "Rigid 6-DoF with bounded changing-mass assumptions; optional constant-mass linear slosh",
        "data_conditions": "Packaged EOP, gravity and ephemerides; prescribed synthetic vehicle/sensor inputs",
        "mission_acceptance_status": "BLOCKED", "qualification_status": "NOT RUN",
        "measurement_validation_status": "UNAVAILABLE",
        "source": source, "software": software, "resources": verification["resources"],
        "randomness": {"sampled_guidance_seed": 9, "other_cases": "deterministic; no Monte Carlo claim"},
        "elapsed_wall_s": elapsed, "timing_scope": "Serial numerical examples only; no speed comparison",
        "cases": cases, "calibration_checks": calibration_checks,
        "calibration_reference": {"thrust_n": 10., "thrust_tolerance_n": 2e-6,
                                  "velocity_bias_m_s": .04, "bias_tolerance_m_s": 2e-7},
        "limits": ["PASS describes only the named software demonstrations, including rejection of the injected fault",
                   "Per-case FAIL, UNAVAILABLE and NOT RUN evidence remains visible; no waiver is inferred",
                   "No end-to-end LEO/GEO launch, measured-flight accuracy or flight qualification",
                   "No external bridge was loaded; no package ranking or performance improvement is claimed",
                   "Numerical convergence, parameter uncertainty and model discrepancy are distinct",
                   "Hashes establish integrity, not physical accuracy or authenticity"],
    }
    if previous is not None:
        original = {c["id"]: c for c in previous["cases"]}
        repeat = {c["id"]: c for c in cases}
        same_cases = original.keys() == repeat.keys() and all(
            original[name]["status"] == case["status"] and
            [{k: v for k, v in m.items() if k != "value"} for m in original[name].get("metrics", [])] ==
            [{k: v for k, v in m.items() if k != "value"} for m in case.get("metrics", [])]
            for name, case in repeat.items())
        report["reproduction"] = {"reference_run_id": previous["run_id"],
                                  "status": "PASS" if demonstrated and same_cases else "FAIL",
                                  "criterion": "Same case outcomes and original independent-reference budgets; not byte-identical histories"}
        if report["reproduction"]["status"] != "PASS":
            report["status"] = "FAIL"
    _write(output / "capability.json", report)
    (output / "CAPABILITY.md").write_text(
        "# Reproducible launch capability\n\n" + report["scope"] + ".\n\n"
        + f"Demonstrated-case result: **{report['status']}**. Mission acceptance: **BLOCKED**. "
        "Measured validation: **UNAVAILABLE**. Qualification: **NOT RUN**.\n\n"
        + report["mission_class"] + ". " + report["fidelity"] + ".\n\n"
        + report["data_conditions"] + ".\n\n"
        + "| Case | Status | Evidence / reason |\n| --- | --- | --- |\n"
        + "".join(f"| {c['id']} | {c['status']} | "
                  + (f"[{c['evidence']}]({c['evidence']})" if "evidence" in c else c.get("reason", ""))
                  + " |\n" for c in cases)
        + "\n" + "\n".join("- " + limit for limit in report["limits"])
        + "\n\nInputs, solver settings, seeds, state/residual histories and reference budgets are in the linked JSON evidence. "
        "See docs/launch_capability.rst in the recorded source for installation and replay commands.\n",
        encoding="utf-8",
    )
    _write(output / "manifest.json", {
        "schema": "ssatk.launch_capability_manifest.v1",
        "files": {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(output.rglob("*")) if p.is_file()},
    })
    verify_bundle(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New bundle directory outside the source checkout")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", type=Path, help="Verify saved byte hashes without running physics")
    mode.add_argument("--replay", type=Path, help="Rerun a verified bundle with its pinned source/environment/inputs")
    args = parser.parse_args(argv)
    if args.verify is not None:
        if args.output is not None:
            parser.error("--verify cannot be combined with --output")
        verify_bundle(args.verify)
        print("Bundle integrity PASS; no numerical rerun performed")
        return 0
    if args.output is None:
        parser.error("--output is required when running cases")
    report = build_bundle(args.output, replay=args.replay)
    print(f"Demonstrated cases: {report['status']}; mission BLOCKED; measured validation UNAVAILABLE; qualification NOT RUN")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
