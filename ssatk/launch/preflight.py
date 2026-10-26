"""Reproducible preflight checks for launch-simulation gates G0--G2.

This module validates the inputs needed before an ascent is propagated.  It
does not choose mission requirements or claim that a preflight pass is a
completed trajectory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from base64 import urlsafe_b64decode
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution
from importlib.metadata import version as _distribution_version
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import numpy as np
from astropy import units as u
from astropy.coordinates import ITRS, EarthLocation
from astropy.time import Time
from astropy.utils import iers

from ..data import data_path, resource_package
from .sites import launch_pads
from .vehicles import launch_vehicles

__all__ = [
    "LAUNCH_CONVENTIONS",
    "MissionContractError",
    "ResourceVerificationError",
    "RUN_STATUSES",
    "collect_software_baseline",
    "launch_site_state",
    "load_mission_contract",
    "run_launch_preflight",
    "validate_mission_contract",
    "verify_data_resources",
]

LAUNCH_CONVENTIONS = {
    "position_frame": "GCRF",
    "velocity_frame": "GCRF",
    "earth_fixed_frame": "ITRF",
    "epoch_scale": "UTC",
    "propagation_time": "GPS seconds",
    "state_units": "SI",
    "angles": "east-positive degrees",
    "quaternion": "body-to-GCRF [w, x, y, z]",
    "angular_rate": "body relative to GCRF, expressed in body, rad/s",
}

RUN_STATUSES = frozenset({"NOT RUN", "BLOCKED", "ERROR", "FAIL", "PASS", "WAIVED"})
RISK_STATUSES = frozenset({"OPEN", "BLOCKED", "MITIGATED", "ACCEPTED"})

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "mission_id",
        "description",
        "profile",
        "launch_site",
        "launch_height_m",
        "epoch_utc",
        "vehicle",
        "payload_mass_kg",
        "phases",
        "transitions",
        "termination_conditions",
        "failure_behavior",
        "requirements",
        "resources",
        "software",
        "source_revisions",
        "conventions",
        "risks",
    }
)
_OPTIONAL_TOP_LEVEL_FIELDS = frozenset({"description"})
_REQUIREMENT_FIELDS = frozenset(
    {"value", "unit", "tolerance", "comparison", "check", "evidence_field", "source"}
)
_RESOURCE_FIELDS = frozenset(
    {"path", "sha256", "source", "license", "units", "frame", "validity"}
)
_RISK_FIELDS = frozenset({"id", "description", "control", "status"})
_SOFTWARE_PINS = frozenset(
    {
        "python",
        "llnl-ssapy",
        "ssatk",
        "ssa-data-core",
        "ssa-data-gravity",
        "ssa-data-lunar",
        "ssa-data-lunar-gravity",
        "astropy",
        "astropy-iers-data",
    }
)
_SOURCE_REVISIONS = frozenset({"llnl-ssapy", "ssatk", "ssa-data-core"})
_COMMON_REQUIREMENTS = frozenset(
    {
        "max_dynamic_pressure",
        "max_heating_rate",
        "max_acceleration",
        "max_angle_of_attack",
        "minimum_propellant_reserve",
        "maximum_navigation_position_error",
        "maximum_mission_duration",
        "frame_position_closure",
        "frame_velocity_closure",
    }
)
_PROFILE_REQUIREMENTS = {
    "LEO": frozenset(
        {
            "target_perigee_altitude",
            "target_apogee_altitude",
            "target_eccentricity",
            "target_inclination",
        }
    ),
    "GEO": frozenset(
        {
            "target_gto_perigee_altitude",
            "target_gto_apogee_altitude",
            "target_gto_inclination",
            "target_transfer_delta_v",
            "target_circularization_delta_v",
            "target_geo_altitude",
            "target_geo_eccentricity",
            "target_geo_inclination",
        }
    ),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class MissionContractError(ValueError):
    """Raised when a launch mission contract is incomplete or ambiguous."""


class ResourceVerificationError(ValueError):
    """Raised when a required packaged resource fails strict verification."""


def load_mission_contract(source: str | Path) -> dict[str, Any]:
    """Load and validate one JSON launch mission contract."""

    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissionContractError(f"{path}: invalid JSON: {exc}") from exc
    return validate_mission_contract(payload)


def validate_mission_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached JSON record after validating the strict version-1 contract."""

    if not isinstance(contract, Mapping):
        raise MissionContractError("mission contract must be a JSON object")
    if any(not isinstance(key, str) for key in contract):
        raise MissionContractError("mission contract field names must be strings")

    errors: list[str] = []
    keys = set(contract)
    missing = _TOP_LEVEL_FIELDS - _OPTIONAL_TOP_LEVEL_FIELDS - keys
    unknown = keys - _TOP_LEVEL_FIELDS
    if missing:
        errors.append(f"missing top-level fields: {', '.join(sorted(missing))}")
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(sorted(unknown))}")

    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        errors.append("schema_version must be 1")
    for field in ("mission_id", "profile", "launch_site", "epoch_utc", "vehicle"):
        _require_text(contract.get(field), field, errors)

    profile = contract.get("profile")
    if isinstance(profile, str) and profile not in _PROFILE_REQUIREMENTS:
        errors.append("profile must be 'LEO' or 'GEO'")

    site = contract.get("launch_site")
    if isinstance(site, str) and site not in launch_pads:
        errors.append(f"launch_site is not in the Toolkit catalog: {site!r}")
    vehicle = contract.get("vehicle")
    if isinstance(vehicle, str) and vehicle not in launch_vehicles:
        errors.append(f"vehicle is not in the Toolkit catalog: {vehicle!r}")

    _require_finite(contract.get("launch_height_m"), "launch_height_m", errors)
    _require_finite(contract.get("payload_mass_kg"), "payload_mass_kg", errors, minimum=0.0)
    _validate_epoch(contract.get("epoch_utc"), errors)
    _validate_phases(contract.get("phases"), contract.get("transitions"), errors)
    _validate_string_list(contract.get("termination_conditions"), "termination_conditions", errors)
    _validate_failure_behavior(
        contract.get("failure_behavior"), contract.get("termination_conditions"), errors
    )
    _validate_requirements(contract.get("requirements"), profile, errors)
    _validate_resources(contract.get("resources"), errors)
    _validate_software(contract.get("software"), errors)
    _validate_revisions(contract.get("source_revisions"), errors)
    if contract.get("conventions") != LAUNCH_CONVENTIONS:
        errors.append("conventions must exactly match LAUNCH_CONVENTIONS")
    _validate_risks(contract.get("risks"), errors)

    if errors:
        raise MissionContractError("Invalid launch mission contract:\n- " + "\n- ".join(errors))
    try:
        return json.loads(json.dumps(dict(contract), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MissionContractError(f"mission contract must contain only finite JSON values: {exc}") from exc


def collect_software_baseline(
    repositories: Mapping[str, str | Path] | None = None,
    *,
    packages: tuple[str, ...] = tuple(sorted(_SOFTWARE_PINS - {"python"})),
) -> dict[str, Any]:
    """Collect runtime versions and optional Git revisions without changing them."""

    installed: dict[str, str | None] = {}
    for name in packages:
        try:
            installed[name] = _distribution_version(name)
        except PackageNotFoundError:
            installed[name] = None

    revisions = {}
    for name, path in (repositories or {}).items():
        revisions[name] = _git_state(Path(path))

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": installed,
        "repositories": revisions,
    }


def verify_data_resources(
    resources: list[Mapping[str, Any]], *, strict: bool = True
) -> list[dict[str, Any]]:
    """Verify wheel RECORD payloads and any supplied contract SHA-256 hashes."""
    results = []
    for requested in resources:
        path = requested["path"]
        declared_hash = requested.get("sha256")
        try:
            package = resource_package(path)
            metadata = distribution(package.replace("_", "-"))
            manifest_entry = next(
                (entry for entry in metadata.files or () if str(entry) == f"{package}/data/{path}"),
                None,
            )
            if manifest_entry is None or manifest_entry.hash is None or manifest_entry.hash.mode != "sha256":
                results.append(
                    {"path": path, "status": "BLOCKED", "detail": "resource has no SHA-256 in the wheel RECORD"}
                )
                continue
            with data_path(path, package=package) as resolved:
                actual_hash, actual_bytes, is_lfs_pointer = _hash_file(resolved)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            results.append({"path": path, "status": "BLOCKED", "detail": str(exc)})
            continue

        manifest_hash = urlsafe_b64decode(manifest_entry.hash.value + "===").hex()
        manifest_bytes = manifest_entry.size
        problems = []
        if is_lfs_pointer:
            problems.append("resource is a Git LFS pointer, not its payload")
        if declared_hash is not None and declared_hash.lower() != manifest_hash:
            problems.append("contract SHA-256 does not match the package manifest")
        if actual_hash != manifest_hash:
            problems.append("resource SHA-256 does not match the package manifest")
        if actual_bytes != manifest_bytes:
            problems.append("resource byte count does not match the package manifest")
        results.append(
            {
                "path": path,
                "status": "FAIL" if problems else "PASS",
                "sha256": actual_hash,
                "bytes": actual_bytes,
                "detail": "; ".join(problems) if problems else "manifest and payload match",
            }
        )

    failures = [item for item in results if item["status"] != "PASS"]
    if strict and failures:
        details = "; ".join(f"{item['path']}: {item['detail']}" for item in failures)
        raise ResourceVerificationError(details)
    return results


def launch_site_state(site: str, epoch_utc: str | Time, *, height_m: float = 0.0) -> dict[str, Any]:
    """Build a stationary pad state in ITRF and its rotating GCRF state.

    The returned velocity is the inertial velocity caused by Earth rotation.
    Astropy network access is disabled for the transform so the result depends
    only on the installed IERS data package.
    """

    if site not in launch_pads:
        raise KeyError(f"unknown launch site {site!r}")
    height_m = _finite_number(height_m, "height_m")
    epoch = epoch_utc if isinstance(epoch_utc, Time) else Time(epoch_utc, scale="utc")
    if not epoch.isscalar:
        raise ValueError("epoch_utc must be scalar")
    epoch = epoch.utc

    site_record = launch_pads[site]
    longitude_deg = float(site_record["longitude"])
    latitude_deg = float(site_record["latitude"])
    location = EarthLocation.from_geodetic(
        longitude_deg * u.deg,
        latitude_deg * u.deg,
        height_m * u.m,
    )

    with iers.conf.set_temp("auto_download", False):
        itrf = location.get_itrs(obstime=epoch)
        gcrf = location.get_gcrs(obstime=epoch)
        closed = gcrf.transform_to(ITRS(obstime=epoch))

    itrf_position = itrf.cartesian.xyz.to_value(u.m)
    gcrf_position = gcrf.cartesian.xyz.to_value(u.m)
    gcrf_velocity = gcrf.velocity.d_xyz.to_value(u.m / u.s)
    closed_position = closed.cartesian.xyz.to_value(u.m)
    closed_velocity = closed.velocity.d_xyz.to_value(u.m / u.s)

    return {
        "site": site,
        "longitude_deg": longitude_deg,
        "latitude_deg": latitude_deg,
        "height_m": height_m,
        "epoch_utc": epoch.isot,
        "epoch_gps_s": float(epoch.gps),
        "itrf_position_m": itrf_position.tolist(),
        "itrf_velocity_m_s": [0.0, 0.0, 0.0],
        "gcrf_position_m": gcrf_position.tolist(),
        "gcrf_velocity_m_s": gcrf_velocity.tolist(),
        "closure": {
            "position_m": float(np.linalg.norm(closed_position - itrf_position)),
            "velocity_m_s": float(np.linalg.norm(closed_velocity)),
        },
        "conventions": dict(LAUNCH_CONVENTIONS),
    }


def run_launch_preflight(
    contract: Mapping[str, Any],
    *,
    repositories: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate launch gates G0--G2 and return a JSON-serializable evidence record."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "scope": "PREFLIGHT ONLY",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT RUN",
        "gates": {},
    }
    try:
        contract = validate_mission_contract(contract)
    except MissionContractError as exc:
        report["status"] = "BLOCKED"
        report["gates"] = {
            "G0": {"status": "BLOCKED", "detail": str(exc)},
            "G1": {"status": "NOT RUN", "detail": "G0 did not pass"},
            "G2": {"status": "NOT RUN", "detail": "G0 did not pass"},
        }
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        return report

    report["mission_id"] = contract["mission_id"]
    report["contract"] = contract
    report["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    report["requirement_checks"] = {
        name: {"status": "NOT RUN", "detail": "mission dynamics were not propagated"}
        for name in contract["requirements"]
    }
    report["gates"]["G0"] = {
        "status": "PASS",
        "detail": "mission targets, limits, frames, units, checks, and tolerances are explicit",
    }

    packages = tuple(name for name in contract["software"] if name != "python")
    baseline = collect_software_baseline(repositories, packages=packages)
    software_checks = _verify_software(contract["software"], baseline)
    revision_checks = _verify_revisions(contract["source_revisions"], baseline["repositories"])
    resource_checks = verify_data_resources(contract["resources"], strict=False)
    report["baseline"] = baseline
    report["software_checks"] = software_checks
    report["revision_checks"] = revision_checks
    report["resource_checks"] = resource_checks
    g1_checks = [*software_checks, *revision_checks, *resource_checks]
    report["gates"]["G1"] = {
        "status": _combined_status(item["status"] for item in g1_checks),
        "detail": "pinned software, source revisions, and required resource hashes checked",
    }

    try:
        state = launch_site_state(
            contract["launch_site"],
            contract["epoch_utc"],
            height_m=contract["launch_height_m"],
        )
        position_requirement = contract["requirements"]["frame_position_closure"]
        velocity_requirement = contract["requirements"]["frame_velocity_closure"]
        frame_checks = [
            _residual_check("frame_position_closure", state["closure"]["position_m"], position_requirement),
            _residual_check("frame_velocity_closure", state["closure"]["velocity_m_s"], velocity_requirement),
        ]
        report["initial_state"] = state
        report["frame_checks"] = frame_checks
        report["requirement_checks"].update({item["name"]: item for item in frame_checks})
        g2_status = _combined_status([
            report["gates"]["G1"]["status"], *(item["status"] for item in frame_checks)
        ])
        g2_detail = "schema, dependency/resource resolution and initial-state checks; not an ascent"
    except (KeyError, OSError, TypeError, ValueError) as exc:
        g2_status = "ERROR"
        g2_detail = f"{type(exc).__name__}: {exc}"
    report["gates"]["G2"] = {"status": g2_status, "detail": g2_detail}

    report["status"] = _combined_status(gate["status"] for gate in report["gates"].values())
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def _validate_epoch(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        return
    try:
        epoch = Time(value, scale="utc")
        if not epoch.isscalar:
            errors.append("epoch_utc must be scalar")
    except (TypeError, ValueError) as exc:
        errors.append(f"epoch_utc is invalid: {exc}")


def _validate_phases(phases: Any, transitions: Any, errors: list[str]) -> None:
    if not _validate_string_list(phases, "phases", errors, minimum=2):
        return
    if len(set(phases)) != len(phases):
        errors.append("phases must be unique")
    if not isinstance(transitions, list) or not transitions:
        errors.append("transitions must be a non-empty list")
        return
    valid = set(phases)
    parsed = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, list) or len(transition) != 2:
            errors.append(f"transitions[{index}] must be [from_phase, to_phase]")
            continue
        if any(not isinstance(phase, str) or phase not in valid for phase in transition):
            errors.append(f"transitions[{index}] references an unknown phase")
            continue
        parsed.append(tuple(transition))
    for pair in itertools.pairwise(phases):
        if pair not in parsed:
            errors.append(f"missing ordered transition {pair[0]!r} -> {pair[1]!r}")


def _validate_failure_behavior(behavior: Any, conditions: Any, errors: list[str]) -> None:
    if not isinstance(behavior, Mapping):
        errors.append("failure_behavior must map each termination condition to FAIL, ERROR or BLOCKED")
        return
    if isinstance(conditions, list) and all(isinstance(condition, str) for condition in conditions):
        if set(behavior) != set(conditions):
            errors.append("failure_behavior must cover exactly the termination_conditions")
        if len(set(conditions)) != len(conditions):
            errors.append("termination_conditions must be unique")
    for condition, status in behavior.items():
        if not isinstance(status, str) or status not in {"FAIL", "ERROR", "BLOCKED"}:
            errors.append(f"failure_behavior.{condition} must be FAIL, ERROR or BLOCKED")


def _validate_requirements(requirements: Any, profile: Any, errors: list[str]) -> None:
    if not isinstance(requirements, Mapping):
        errors.append("requirements must be an object keyed by requirement id")
        return
    required = set(_COMMON_REQUIREMENTS)
    required.update(_PROFILE_REQUIREMENTS.get(profile, ()) if isinstance(profile, str) else ())
    if any(not isinstance(name, str) for name in requirements):
        errors.append("requirement ids must be strings")
        return
    missing = required - set(requirements)
    if missing:
        errors.append(f"missing requirements: {', '.join(sorted(missing))}")
    for name, spec in requirements.items():
        where = f"requirements.{name}"
        if not isinstance(spec, Mapping):
            errors.append(f"{where} must be an object")
            continue
        _exact_fields(spec, _REQUIREMENT_FIELDS, where, errors)
        _require_finite(spec.get("value"), f"{where}.value", errors)
        _require_finite(spec.get("tolerance"), f"{where}.tolerance", errors, minimum=0.0)
        comparison = spec.get("comparison")
        if not isinstance(comparison, str) or comparison not in {"at_most", "at_least", "within"}:
            errors.append(f"{where}.comparison must be at_most, at_least or within")
        if name in required:
            if name.startswith("max"):
                expected = "at_most"
            elif name.startswith("min"):
                expected = "at_least"
            else:
                expected = "within"
            if comparison != expected:
                errors.append(f"{where}.comparison must be {expected}")
        for field in ("unit", "check", "evidence_field", "source"):
            _require_text(spec.get(field), f"{where}.{field}", errors)


def _validate_resources(resources: Any, errors: list[str]) -> None:
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty list")
        return
    paths = []
    for index, resource in enumerate(resources):
        where = f"resources[{index}]"
        if not isinstance(resource, Mapping):
            errors.append(f"{where} must be an object")
            continue
        _exact_fields(resource, _RESOURCE_FIELDS, where, errors)
        for field in _RESOURCE_FIELDS - {"sha256"}:
            _require_text(resource.get(field), f"{where}.{field}", errors)
        digest = resource.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest.lower()) is None:
            errors.append(f"{where}.sha256 must be 64 hexadecimal characters")
        path = resource.get("path")
        if isinstance(path, str) and path:
            pure_path = PurePosixPath(path)
            if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in path:
                errors.append(f"{where}.path must be a safe POSIX path below the selected data package data root")
            paths.append(path)
    if len(paths) != len(set(paths)):
        errors.append("resource paths must be unique")


def _validate_software(software: Any, errors: list[str]) -> None:
    if not isinstance(software, Mapping):
        errors.append("software must be an object of exact version pins")
        return
    missing = _SOFTWARE_PINS - set(software)
    if missing:
        errors.append(f"missing software pins: {', '.join(sorted(missing))}")
    for name, value in software.items():
        _require_text(value, f"software.{name}", errors)
        if isinstance(value, str) and any(token in value for token in (">", "<", "*", ",")):
            errors.append(f"software.{name} must be an exact version, not a range")


def _validate_revisions(revisions: Any, errors: list[str]) -> None:
    if not isinstance(revisions, Mapping):
        errors.append("source_revisions must be an object")
        return
    missing = _SOURCE_REVISIONS - set(revisions)
    if missing:
        errors.append(f"missing source revisions: {', '.join(sorted(missing))}")
    for name, value in revisions.items():
        _require_text(value, f"source_revisions.{name}", errors)


def _validate_risks(risks: Any, errors: list[str]) -> None:
    if not isinstance(risks, list) or not risks:
        errors.append("risks must be a non-empty list")
        return
    ids = []
    for index, risk in enumerate(risks):
        where = f"risks[{index}]"
        if not isinstance(risk, Mapping):
            errors.append(f"{where} must be an object")
            continue
        _exact_fields(risk, _RISK_FIELDS, where, errors)
        for field in ("id", "description", "control"):
            _require_text(risk.get(field), f"{where}.{field}", errors)
        if risk.get("status") not in RISK_STATUSES:
            errors.append(f"{where}.status must be one of {sorted(RISK_STATUSES)}")
        ids.append(risk.get("id"))
    if len(ids) != len(set(ids)):
        errors.append("risk ids must be unique")


def _validate_string_list(value: Any, name: str, errors: list[str], *, minimum: int = 1) -> bool:
    if not isinstance(value, list) or len(value) < minimum:
        errors.append(f"{name} must contain at least {minimum} entries")
        return False
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{name} entries must be non-empty strings")
        return False
    return True


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], where: str, errors: list[str]) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        errors.append(f"{where} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        errors.append(f"{where} has unknown fields: {', '.join(sorted(unknown))}")


def _require_text(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")
    elif value.strip().upper() == "UNSET":
        errors.append(f"{name} must replace UNSET with an authoritative input")


def _require_finite(value: Any, name: str, errors: list[str], *, minimum: float | None = None) -> None:
    try:
        number = _finite_number(value, name)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return
    if minimum is not None and number < minimum:
        errors.append(f"{name} must be at least {minimum}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _git_state(repository: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(repository.resolve()), "revision": None, "dirty": None}
    if not repository.is_dir():
        state["error"] = "repository path does not exist"
        return state
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state
    state.update({"revision": revision, "dirty": bool(dirty)})
    return state


def _verify_software(expected: Mapping[str, str], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    actual = {"python": baseline["python"]["version"], **baseline["packages"]}
    checks = []
    for name, expected_version in expected.items():
        actual_version = actual.get(name)
        status = "BLOCKED" if actual_version is None else "PASS" if actual_version == expected_version else "FAIL"
        checks.append(
            {"name": name, "expected": expected_version, "actual": actual_version, "status": status}
        )
    return checks


def _verify_revisions(
    expected: Mapping[str, str], actual: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    checks = []
    for name, expected_revision in expected.items():
        state = actual.get(name)
        if state is None or state.get("revision") is None:
            status = "BLOCKED"
            actual_revision = None
            detail = "repository revision was not available"
        elif state["revision"] != expected_revision:
            status = "FAIL"
            actual_revision = state["revision"]
            detail = "revision does not match the contract"
        elif state.get("dirty"):
            status = "BLOCKED"
            actual_revision = state["revision"]
            detail = "repository has uncommitted changes"
        else:
            status = "PASS"
            actual_revision = state["revision"]
            detail = "revision matches and worktree is clean"
        checks.append(
            {
                "name": name,
                "expected": expected_revision,
                "actual": actual_revision,
                "status": status,
                "detail": detail,
            }
        )
    return checks


def _hash_file(path: Path) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    byte_count = 0
    prefix = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if not prefix:
                prefix = block[: len(_LFS_PREFIX)]
            byte_count += len(block)
            digest.update(block)
    return digest.hexdigest(), byte_count, prefix == _LFS_PREFIX


def _residual_check(name: str, residual: float, requirement: Mapping[str, Any]) -> dict[str, Any]:
    target = float(requirement["value"])
    tolerance = float(requirement["tolerance"])
    comparison = requirement["comparison"]
    if comparison == "at_most":
        passed = residual <= target + tolerance
    elif comparison == "at_least":
        passed = residual >= target - tolerance
    else:
        passed = abs(residual - target) <= tolerance
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "value": residual,
        "target": target,
        "tolerance": tolerance,
        "comparison": comparison,
        "unit": requirement["unit"],
        "evidence_field": requirement["evidence_field"],
    }


def _combined_status(statuses) -> str:
    statuses = set(statuses)
    for status in ("ERROR", "FAIL", "BLOCKED", "NOT RUN", "WAIVED"):
        if status in statuses:
            return status
    return "PASS"
