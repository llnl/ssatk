"""Integrated ascent mass balance and independent Cartesian orbit references."""

from copy import deepcopy
import json

import numpy as np
import pytest

from demos.launch.demo_integrated_launch import _requirements, run_reference
from ssatk.constants import EARTH_MU, EARTH_RADIUS, G0
from ssatk.launch import check_launch_requirements


@pytest.fixture(scope="module")
def mission(tmp_path_factory):
    report = run_reference()
    (tmp_path_factory.mktemp("integrated-launch") / "mission.json").write_text(
        json.dumps(report, allow_nan=False, default=lambda value:value.tolist()), encoding="utf-8")
    return report


def _throttle_integral(entry, elapsed):
    """Closed form of a slew-limited exponential, rate 1/s and lag 0.2 s."""
    elapsed = max(0., elapsed)
    start = entry["actual"][3]
    target = entry["activated"][-1]["bounded"][3]
    distance = abs(target-start)
    ramp_time = max(0., distance-.2)
    ramp = min(elapsed, ramp_time)
    decay = max(0., elapsed-ramp_time)
    travel = .5*ramp**2 + distance*decay-min(distance, .2)*.2*(-np.expm1(-decay/.2))
    return start*elapsed+np.sign(target-start)*travel


def test_pad_to_orbit_against_flow_balance_and_cartesian_elements(mission):
    """R1/R4: depletion inventory/upper flow within 1e-4 kg; elements 1e-5 m/1e-12.

    Initial dynamic pressure uses only the declared 5/2 m/s ENU wind,
    within 1e-3 Pa, independently detecting doubled Earth-rotation velocity.
    Constant lower flow gives an analytic depletion epoch within 1e-7 s.
    Check inventory separately: GPS rounding at this flow exceeds 1e-4 kg.
    """
    assert mission["status"] == "PASS", (mission["termination"], mission["orbit_insertion_reached"],
        {name:check for name, check in mission["requirement_checks"].items() if check["status"] != "PASS"})
    events = {event["event"]:event for event in mission["events"] if event["event"] != "__control"}
    epoch = mission["initial_state"]["epoch_gps_s"]
    separation = events["stage_separation"]
    insertion = events["orbit_insertion"]
    np.testing.assert_allclose(mission["history"][0]["mass_kg"], 518870., atol=1e-4, rtol=0)
    np.testing.assert_allclose(separation["pre_state"]["mass_kg"], 518870.-(395700.-395.7), atol=1e-4, rtol=0)
    np.testing.assert_allclose(separation["time_gps_s"], epoch+(395700.-395.7)*282.*G0/7605000.,
                               atol=1e-7, rtol=0)
    start, stop = separation["time_gps_s"], insertion["time_gps_s"]
    integrated = 0.
    for index, entry in enumerate(mission["control"]):
        end = mission["control"][index+1]["time_gps_s"] if index+1 < len(mission["control"]) else stop
        low, high = max(start, entry["time_gps_s"]), min(stop, end)
        if high > low:
            integrated += (_throttle_integral(entry, high-entry["time_gps_s"])
                           - _throttle_integral(entry, low-entry["time_gps_s"]))
    np.testing.assert_allclose(insertion["pre_state"]["mass_kg"], 97570.-934000.*integrated/(348.*G0),
                               atol=1e-4, rtol=0)
    np.testing.assert_allclose(separation["pre_state"]["mass_kg"]-separation["post_state"]["mass_kg"],
                               25995.7, atol=1e-7, rtol=0)
    expected_pressure = .5*101325./(287.05*280.)*(5.**2+2.**2)
    np.testing.assert_allclose(mission["history"][0]["loads"]["constraints.max_dynamic_pressure_pa"],
                               expected_pressure, atol=1e-3, rtol=0)
    final = mission["history"][-1]
    r, v = np.asarray(final["position_gcrf_m"]), np.asarray(final["velocity_gcrf_m_s"])
    radius = np.linalg.norm(r)
    momentum = np.cross(r, v)
    eccentricity = np.linalg.norm(np.cross(v, momentum)/EARTH_MU-r/radius)
    semimajor = -EARTH_MU/(np.dot(v, v)-2*EARTH_MU/radius)
    references = {"orbit.perigee_altitude_m":semimajor*(1-eccentricity)-EARTH_RADIUS,
                  "orbit.apogee_altitude_m":semimajor*(1+eccentricity)-EARTH_RADIUS,
                  "orbit.eccentricity":eccentricity,
                  "orbit.inclination_deg":np.degrees(np.arccos(momentum[2]/np.linalg.norm(momentum)))}
    for field, expected in references.items():
        np.testing.assert_allclose(mission["evidence"][field]["value"], expected,
                                   atol=1e-5 if field.endswith("_m") else 1e-12, rtol=0)
    assert mission["qualification_status"] == "NOT RUN"


def test_100km_radial_flight_is_rejected_by_orbit_acceptance(mission):
    """R1: radial 1 km/s at 100 km has e=1 and zero perigee radius, 1e-12/1e-5 m."""
    evidence = deepcopy(mission["evidence"])
    radius, speed = EARTH_RADIUS+100000., 1000.
    semimajor = -EARTH_MU/(speed**2-2*EARTH_MU/radius)
    evidence["orbit.perigee_altitude_m"]["value"] = -EARTH_RADIUS
    evidence["orbit.apogee_altitude_m"]["value"] = 2*semimajor-EARTH_RADIUS
    evidence["orbit.eccentricity"]["value"] = 1.
    report = check_launch_requirements(mission["requirements"], evidence)
    for name in ("target_perigee_altitude", "target_apogee_altitude", "target_eccentricity"):
        assert report["requirement_checks"][name]["status"] == "FAIL"
    assert report["status"] == "FAIL"
    del evidence["constraints.max_heating_rate_w_m2"]
    assert check_launch_requirements(mission["requirements"], evidence)["requirement_checks"]["max_heating_rate"]["status"] == "NOT RUN"


def test_initial_wind_load_stops_before_release():
    """R1: 5/2 m/s wind dynamic pressure exceeds a declared 10 Pa limit, 1e-3 Pa."""
    requirements = _requirements(28.46)
    requirements["max_dynamic_pressure"]["value"] = 10.
    report = run_reference(requirements=requirements)
    expected = .5*101325./(287.05*280.)*(5.**2+2.**2)
    np.testing.assert_allclose(report["requirement_checks"]["max_dynamic_pressure"]["value"], expected,
                               atol=1e-3, rtol=0)
    assert report["termination"] == "limit:pad:max_dynamic_pressure"
    assert report["status"] == "FAIL"
    assert not report["orbit_insertion_reached"]
    assert report["requirement_checks"]["maximum_navigation_position_error"]["status"] == "NOT RUN"
