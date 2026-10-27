from __future__ import annotations

import json
import platform
import subprocess
from base64 import urlsafe_b64decode
from copy import deepcopy
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from ssatk.coordinates.earth_fixed import itrf_to_gcrf_eop
from ssatk.environment_eop import load_packaged_eop
from ssatk.launch.preflight import (
    EOPCoverageError,
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
        "comparison": "within",
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
        "maximum_mission_duration",
        "frame_position_closure",
        "frame_velocity_closure",
        "target_perigee_altitude",
        "target_apogee_altitude",
        "target_eccentricity",
        "target_inclination",
    }
    requirements = {name: _requirement() for name in requirement_names}
    for name, requirement in requirements.items():
        if name.startswith("max"):
            requirement["comparison"] = "at_most"
        elif name.startswith("min"):
            requirement["comparison"] = "at_least"
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
        "epoch_utc": "2024-10-02T16:00:00",
        "allow_predicted_eop": False,
        "vehicle": "Falcon 9 Full Thrust",
        "payload_mass_kg": 1.0,
        "phases": ["prelaunch", "ascent", "insertion"],
        "transitions": [["prelaunch", "ascent"], ["ascent", "insertion"]],
        "termination_conditions": ["invalid state"],
        "failure_behavior": {"invalid state": "ERROR"},
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


def _init_synthetic_repo(path: Path) -> str:
    """Create a one-commit Git repository at ``path`` and return its HEAD revision.

    Used to give ``run_launch_preflight`` a real, clean, matching repository
    for every pinned ``source_revisions`` entry, so G1 can genuinely PASS in a
    test without depending on the Git state of whatever packages happen to be
    installed in the environment running the test.
    """
    path.mkdir()
    run = lambda *args: subprocess.run(args, cwd=path, check=True, capture_output=True, text=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "preflight-test@example.com")
    run("git", "config", "user.name", "preflight test fixture")
    (path / "README").write_text("Synthetic fixture repository; not a real source checkout.\n")
    run("git", "add", "README")
    run("git", "commit", "-q", "-m", "synthetic fixture commit")
    return run("git", "rev-parse", "HEAD").stdout.strip()


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


_OBSERVED_EPOCH = "2024-06-01T16:00:00"  # polar_motion/ut1 flag 'I' in the packaged EOP table
_PREDICTED_EPOCH = "2026-10-02T16:00:00"  # flag 'P'; inside the table but past the observed data
_UNCOVERED_EPOCH = "2027-09-01T00:00:00"  # just past the end of the packaged EOP table


def _wgs84_ecef(latitude_deg, longitude_deg, height_m):
    """Closed-form geodetic->ECEF conversion, independent of SSATK/Astropy."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    phi = np.radians(latitude_deg)
    lam = np.radians(longitude_deg)
    n = a / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    x = (n + height_m) * np.cos(phi) * np.cos(lam)
    y = (n + height_m) * np.cos(phi) * np.sin(lam)
    z = (n * (1.0 - e2) + height_m) * np.sin(phi)
    return np.array([x, y, z])


def test_launch_site_state_itrf_position_matches_wgs84_closed_form():
    """R1: ITRF position must match the closed-form WGS84 geodetic->ECEF
    formula (a = 6378137 m, f = 1/298.257223563), independent of the
    Astropy/SSATK code path, to 1e-6 m (floating-point precision at this
    magnitude, not pad survey accuracy). Checked at two pads of different
    latitude so a spherical-Earth or flattening mistake cannot cancel out.
    """
    for site in ("Kennedy Space Center LC-39A", "Vandenberg SLC-4E"):
        state = launch_site_state(site, _OBSERVED_EPOCH)
        expected_m = _wgs84_ecef(state["latitude_deg"], state["longitude_deg"], state["height_m"])
        residual_m = np.linalg.norm(np.array(state["itrf_position_m"]) - expected_m)
        assert residual_m == pytest.approx(0.0, abs=1.0e-6)


def test_launch_site_state_velocity_matches_earth_rotation():
    """R1: the pad's inertial velocity must be the rigid-body Earth-rotation
    velocity v = omega x rho for the IERS Conventions (2010) nominal mean
    angular velocity omega = 7.292115e-5 rad/s (Table 1.1), where rho is the
    ITRF distance from the spin axis. rho must come from the ITRF position:
    using the GCRF x/y components instead confounds the spin-axis distance
    with the pad's declination, which is wrong by kilometres once a pad is
    off the equator.

    Also checks that v is perpendicular to r (no radial component for a
    stationary point) and that rotation is eastward (right-handed about
    +z): the position component perpendicular to GCRF z, crossed with v,
    must point within 1 degree of GCRF +z. GCRF z stays within a fraction of
    a degree of the true pole from 2000-2030 (precession/nutation), which is
    why 1 degree is a safe tolerance for this direction check while still
    catching a sign-flipped velocity (which fails by ~180 degrees).
    """
    earth_angular_velocity_rad_s = 7.292115e-5
    for site in ("Kennedy Space Center LC-39A", "Vandenberg SLC-4E"):
        state = launch_site_state(site, _OBSERVED_EPOCH)
        r_itrf = np.array(state["itrf_position_m"])
        r_gcrf = np.array(state["gcrf_position_m"])
        v_gcrf = np.array(state["gcrf_velocity_m_s"])

        rho_m = np.linalg.norm(r_itrf[:2])
        speed_m_s = np.linalg.norm(v_gcrf)
        # Residual is the nominal mean rotation rate versus the instantaneous
        # sidereal rate Astropy derives from the packaged Earth-orientation
        # data; observed residual is about 2e-4 to 4e-4 m/s across several
        # launch pads and epochs, well inside this budget.
        assert speed_m_s == pytest.approx(earth_angular_velocity_rad_s * rho_m, abs=1.0e-3)

        cosine_r_v = abs(np.dot(v_gcrf, r_gcrf)) / (speed_m_s * np.linalg.norm(r_gcrf))
        assert cosine_r_v < 1.0e-6

        z_hat = np.array([0.0, 0.0, 1.0])
        rho_vector_gcrf = r_gcrf - np.dot(r_gcrf, z_hat) * z_hat
        spin_axis_estimate = np.cross(rho_vector_gcrf, v_gcrf)
        spin_axis_estimate /= np.linalg.norm(spin_axis_estimate)
        assert np.dot(spin_axis_estimate, z_hat) > np.cos(np.radians(1.0))

        assert state["closure"]["position_m"] < 1.0e-6
        assert state["closure"]["velocity_m_s"] < 1.0e-3


def test_launch_site_state_matches_independent_eop_rotation():
    """R2: compare against SSATK's independent erfa-based ITRF->GCRF path
    (``itrf_to_gcrf_eop``: IAU 1976/1980 precession-nutation plus GST94),
    fed the same packaged EOP snapshot, instead of the Astropy ITRS<->GCRS
    implementation ``launch_site_state`` uses. The two models differ by
    about 0.6-1.4 m at the surface for 2020-2024 (measured here); a 5 ms
    UT1 error moves a Cape Canaveral-latitude pad by about 2 m, so a 2.0 m
    tolerance still catches a UT1/EOP mistake such as Astropy's silent
    UT1-UTC=0 fallback for an uncovered epoch.
    """
    epoch = Time(_OBSERVED_EPOCH, scale="utc")
    eop = load_packaged_eop()
    for site in ("Kennedy Space Center LC-39A", "Vandenberg SLC-4E"):
        state = launch_site_state(site, epoch)
        itrf_position = np.array([state["itrf_position_m"]])
        independent_gcrf = itrf_to_gcrf_eop(itrf_position, epoch.gps, eop=eop, allow_predicted=False)[0]
        residual_m = np.linalg.norm(independent_gcrf - np.array(state["gcrf_position_m"]))
        assert residual_m == pytest.approx(0.0, abs=2.0)


def test_launch_site_state_predicted_eop_requires_opt_in():
    with pytest.raises(EOPCoverageError, match="predicted"):
        launch_site_state("Kennedy Space Center LC-39A", _PREDICTED_EPOCH)


def test_launch_site_state_predicted_eop_succeeds_with_opt_in():
    state = launch_site_state(
        "Kennedy Space Center LC-39A", _PREDICTED_EPOCH, allow_predicted_eop=True,
    )
    assert state["earth_orientation"]["predicted"] is True
    assert state["earth_orientation"]["allow_predicted"] is True
    assert state["closure"]["position_m"] < 1.0e-6
    assert state["closure"]["velocity_m_s"] < 1.0e-3


def test_launch_site_state_rejects_epoch_outside_eop_table():
    with pytest.raises(EOPCoverageError, match="outside"):
        launch_site_state("Kennedy Space Center LC-39A", _UNCOVERED_EPOCH, allow_predicted_eop=True)


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
    assert report["gates"]["G2"]["status"] == "BLOCKED"
    assert report["status"] == "BLOCKED"


def test_run_launch_preflight_blocks_g2_for_predicted_eop_without_contract_opt_in(tmp_path):
    """A mission whose epoch needs predicted EOP must BLOCK G2, not crash or
    pass, unless the contract explicitly opts in with ``allow_predicted_eop``.

    G2 now also carries forward a BLOCKED/FAILED G1 (prerequisite-aware
    gating), so this test gives every ``source_revisions`` entry a matching,
    clean, synthetic Git repository: without that, G1 would stay BLOCKED on
    missing revisions regardless of the EOP behavior under test, and the
    second assertion below could never observe G2 reach PASS.
    """
    contract = deepcopy(_contract())
    contract["epoch_utc"] = _PREDICTED_EPOCH
    repositories = {}
    for name in ("llnl-ssapy", "ssatk", "ssa-data-core"):
        repo_path = tmp_path / name
        contract["source_revisions"][name] = _init_synthetic_repo(repo_path)
        repositories[name] = repo_path

    blocked = run_launch_preflight(contract, repositories=repositories)
    assert blocked["gates"]["G1"]["status"] == "PASS"
    assert blocked["gates"]["G2"]["status"] == "BLOCKED"
    assert "predicted" in blocked["gates"]["G2"]["detail"]

    contract["allow_predicted_eop"] = True
    passed = run_launch_preflight(contract, repositories=repositories)
    assert passed["gates"]["G1"]["status"] == "PASS"
    assert passed["gates"]["G2"]["status"] == "PASS"
    assert passed["initial_state"]["earth_orientation"]["predicted"] is True


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
