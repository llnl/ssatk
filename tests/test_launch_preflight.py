from __future__ import annotations

import json
import platform
from base64 import urlsafe_b64decode
from copy import deepcopy
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import pytest

from ssatk.launch.preflight import (
    LAUNCH_CONVENTIONS,
    MissionContractError,
    ResourceVerificationError,
    collect_software_baseline,
    launch_site_state,
    run_launch_preflight,
    validate_mission_contract,
    verify_data_resources,
)


def _requirement(value=1.0, tolerance=0.0, unit="test-unit"):
    return {
        "value": value,
        "unit": unit,
        "tolerance": tolerance,
        "check": "synthetic_test_check",
        "evidence_field": "synthetic.test.evidence",
        "source": "synthetic test fixture; not a mission requirement",
    }


def _contract():
    requirement_names = {
        "max_dynamic_pressure",
        "max_heating_rate",
        "max_acceleration",
        "max_angle_of_attack",
        "minimum_propellant_reserve",
        "maximum_navigation_position_error",
        "frame_position_closure",
        "frame_velocity_closure",
        "target_perigee_altitude",
        "target_apogee_altitude",
        "target_eccentricity",
        "target_inclination",
    }
    requirements = {name: _requirement() for name in requirement_names}
    requirements["frame_position_closure"] = _requirement(0.0, 1.0e-6, "m")
    requirements["frame_velocity_closure"] = _requirement(0.0, 1.0e-3, "m/s")

    eop = next(
        entry for entry in distribution("ssa-data-core").files
        if str(entry) == "ssa_data_core/data/environment/eop/finals2000A.all"
    )
    return {
        "schema_version": 1,
        "mission_id": "synthetic_preflight_test",
        "description": "Synthetic schema and frame test; not a flight case.",
        "profile": "LEO",
        "launch_site": "Kennedy Space Center LC-39A",
        "launch_height_m": 0.0,
        "epoch_utc": "2026-10-02T16:00:00",
        "vehicle": "Falcon 9 Full Thrust",
        "payload_mass_kg": 1.0,
        "phases": ["prelaunch", "ascent", "insertion"],
        "transitions": [["prelaunch", "ascent"], ["ascent", "insertion"]],
        "termination_conditions": ["invalid state"],
        "requirements": requirements,
        "resources": [
            {
                "path": "environment/eop/finals2000A.all",
                "sha256": urlsafe_b64decode(eop.hash.value + "===").hex(),
                "source": "ssa_data_core installed wheel RECORD test fixture",
                "license": "source metadata retained by ssa_data_core",
                "units": "mixed fields documented by the data package",
                "frame": "ITRF Earth-orientation inputs",
                "validity": "packaged snapshot domain",
            }
        ],
        "software": {
            "python": platform.python_version(),
            "llnl-ssapy": version("llnl-ssapy"),
            "ssatk": version("ssatk"),
            "ssa-data-core": version("ssa-data-core"),
            "ssa-data-gravity": version("ssa-data-gravity"),
            "ssa-data-lunar": version("ssa-data-lunar"),
            "ssa-data-lunar-gravity": version("ssa-data-lunar-gravity"),
            "astropy": version("astropy"),
            "astropy-iers-data": version("astropy-iers-data"),
        },
        "source_revisions": {
            "llnl-ssapy": "a" * 40,
            "ssatk": "b" * 40,
            "ssa-data-core": "c" * 40,
        },
        "conventions": dict(LAUNCH_CONVENTIONS),
        "risks": [
            {
                "id": "synthetic-inputs",
                "description": "The fixture is not a mission.",
                "control": "Never use it as qualification evidence.",
                "status": "ACCEPTED",
            }
        ],
    }


def test_contract_is_strict_and_requires_traceable_numerical_requirements():
    contract = _contract()
    assert validate_mission_contract(contract)["mission_id"] == "synthetic_preflight_test"

    invalid = deepcopy(contract)
    invalid.pop("payload_mass_kg")
    invalid["surprise"] = True
    invalid["requirements"]["max_acceleration"].pop("source")
    with pytest.raises(MissionContractError) as error:
        validate_mission_contract(invalid)

    message = str(error.value)
    assert "missing top-level fields: payload_mass_kg" in message
    assert "unknown top-level fields: surprise" in message
    assert "requirements.max_acceleration is missing fields: source" in message


def test_launch_site_state_includes_earth_rotation_and_closes_offline():
    state = launch_site_state(
        "Kennedy Space Center LC-39A",
        "2026-10-02T16:00:00",
    )

    assert state["epoch_utc"] == "2026-10-02T16:00:00.000"
    assert np.linalg.norm(state["gcrf_velocity_m_s"]) > 0.0
    assert state["closure"]["position_m"] < 1.0e-6
    assert state["closure"]["velocity_m_s"] < 1.0e-3
    assert state == launch_site_state(
        "Kennedy Space Center LC-39A",
        "2026-10-02T16:00:00",
    )


def test_required_data_resource_matches_manifest_and_payload():
    resource = _contract()["resources"]
    result = verify_data_resources(resource)

    assert result[0]["status"] == "PASS"
    assert result[0]["sha256"] == resource[0]["sha256"]

    wrong = deepcopy(resource)
    wrong[0]["sha256"] = "0" * 64
    assert verify_data_resources(wrong, strict=False)[0]["status"] == "FAIL"
    with pytest.raises(ResourceVerificationError, match="contract SHA-256"):
        verify_data_resources(wrong)


def test_preflight_keeps_missing_revisions_and_dependencies_blocked(tmp_path):
    report = run_launch_preflight(
        _contract(),
        repositories={
            "llnl-ssapy": tmp_path / "ssapy",
            "ssatk": tmp_path / "toolkit",
            "ssa-data-core": tmp_path / "data",
        },
    )

    assert report["scope"] == "PREFLIGHT ONLY"
    assert report["gates"]["G0"]["status"] == "PASS"
    assert report["gates"]["G1"]["status"] == "BLOCKED"
    assert report["gates"]["G2"]["status"] == "PASS"
    assert report["status"] == "BLOCKED"
    assert report["initial_state"]["closure"]["position_m"] < 1.0e-6


def test_unconfigured_leo_template_reports_blocked():
    template_path = Path(__file__).parents[1] / "demos" / "launch" / "leo_preflight_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))

    report = run_launch_preflight(template)

    assert report["status"] == "BLOCKED"
    assert report["gates"]["G0"]["status"] == "BLOCKED"
    assert "launch_site must be a non-empty string" in report["gates"]["G0"]["detail"]


def test_baseline_records_missing_repository_without_mutating_it(tmp_path):
    baseline = collect_software_baseline({"missing": tmp_path / "not-created"}, packages=())

    assert baseline["repositories"]["missing"]["revision"] is None
    assert baseline["repositories"]["missing"]["dirty"] is None
    assert not (tmp_path / "not-created").exists()


def test_real_demo2_launch_preflight_matches_spacepy():
    pytest.importorskip("spacepy", reason="Optional validation extra is required for the independent comparison.")
    from demos.launch.demo_real_launch_preflight import run

    result = run()

    assert result["state"]["site"] == "Kennedy Space Center LC-39A"
    assert result["state"]["epoch_utc"] == "2020-05-30T19:22:00.000"
    assert result["fixed_position_error_m"] <= result["fixed_position_tolerance_m"]
    assert result["speed_error_m_s"] <= result["speed_numerical_budget_m_s"]
    assert result["status"] == "PASS"
    print("\nDemo-2 real-launch preflight - NOT a flown trajectory:")
    print(f"Earth-fixed pad difference: {result['fixed_position_error_m']:.3e} m")
    print(f"Inertial speed: SSATK {result['ssatk_speed_m_s']:.6f}, SpacePy {result['spacepy_speed_m_s']:.6f} m/s")
    print(f"Speed difference / numerical budget: {result['speed_error_m_s']:.6f} / {result['speed_numerical_budget_m_s']:.6f} m/s")
    print(f"Unscored J2000/GCRF position difference: {result['inertial_position_difference_m']:.3f} m")
