"""End-to-end maneuver mode coverage.

Benchmark provenance:
* Hohmann and bi-elliptic checks use the closed-form two-body equations in
  Bate, Mueller, and White, *Fundamentals of Astrodynamics*, Ch. 3, and Curtis,
  *Orbital Mechanics for Engineering Students*, Sec. 6.3--6.4.
* Fixed-time transfer checks compare against the standard Lambert
  two-point boundary-value solution described in
  Vallado, *Fundamentals of Astrodynamics and Applications*, Ch. 7.
* Continuous-burn checks use constant-acceleration identities, delta-v = a t,
  with inclination/velocity direction conventions verified through the canonical
  SSATK transfer result schema.
* Engine sizing and propellant checks are covered in
  ``test_orbital_maneuver_reference_cases.py`` against the Tsiolkovsky rocket
  equation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.misc import bi_elliptic_transfer_delta_v, hohmann_transfer_delta_v
from ssatk.orbital_mechanics.transfer_bielliptic import transfer_bielliptic
from ssatk.orbital_mechanics.transfer_hohmann import transfer_hohmann
from ssatk.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssatk.orbital_mechanics.transfer_ssapy_function import transfer_ssapy
from ssatk.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous


def _state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _assert_standard_transfer(result, method, min_burns=1):
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["method"] == method
    assert result["units"]["delta_v"] == "m/s"
    assert len(result["burns"]) >= min_burns
    assert result["delta_v_total"] == pytest.approx(sum(burn["delta_v_mag"] for burn in result["burns"]))
    assert result["initial"]["r"].shape == (3,)
    assert result["target"]["v"].shape == (3,)


@pytest.mark.parametrize("radius1,radius2", [(7000e3, 9000e3), (9000e3, 7000e3)])
def test_hohmann_outward_and_inward_cases_match_closed_form(radius1, radius2):
    result = transfer_hohmann(radius1, radius2, samples=10, burn_accel=0.25)
    expected = hohmann_transfer_delta_v(radius1, radius2, EARTH_MU)

    _assert_standard_transfer(result, "transfer_hohmann", min_burns=2)
    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:2], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[-1], rel=1e-12)
    assert result["tof"] > 0.0
    assert result["trajectory"]["r"].shape == (10, 3)
    assert all(burn["duration"] == pytest.approx(burn["delta_v_mag"] / 0.25) for burn in result["burns"])


@pytest.mark.parametrize("radius1,radius2", [(7000e3, 9000e3), (9000e3, 7000e3)])
def test_bielliptic_outward_and_inward_cases_match_closed_form(radius1, radius2):
    intermediate_radius = 20_000e3
    result = transfer_bielliptic(radius1, radius2, intermediate_radius=intermediate_radius, samples_per_arc=6)
    expected = bi_elliptic_transfer_delta_v(radius1, radius2, intermediate_radius, EARTH_MU)

    _assert_standard_transfer(result, "transfer_bielliptic", min_burns=3)
    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:3], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[-1], rel=1e-12)
    assert result["trajectory"]["r"].shape == (11, 3)


def test_transfer_ssapy_fixed_time_matches_reference_solution():
    departure = _state(theta=0.0, t=0.0)
    arrival = _state(theta=0.2, t=1000.0)
    result = transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0)

    _assert_standard_transfer(result, "transfer_ssapy", min_burns=2)
    assert result["delta_v_total"] == pytest.approx(13624.643379536796, rel=1e-9)


@pytest.mark.parametrize("target_delta_v", [10.0, -5.0])
def test_velocity_continuous_positive_and_negative_delta_v_cases(target_delta_v):
    r0, v0, _ = _state()
    result = transfer_velocity_continuous(r0, v0, v_target=target_delta_v, a_thrust=1.0, max_time=30.0)

    _assert_standard_transfer(result, "transfer_velocity_continuous")
    assert result["delta_v_total"] == pytest.approx(abs(target_delta_v), rel=1e-12)
    assert result["burns"][0]["duration"] == pytest.approx(abs(target_delta_v), rel=1e-12)


@pytest.mark.parametrize("delta_v", [5.0, -2.0])
def test_inclination_continuous_positive_and_negative_plane_change_cases(delta_v):
    r0, v0, _ = _state()
    result = transfer_inclination_continuous(r0, v0, delta_v=delta_v, a_thrust=1.0, max_time=30.0)

    _assert_standard_transfer(result, "transfer_inclination_continuous")
    assert result["delta_v_total"] == pytest.approx(abs(delta_v), rel=1e-12)
    assert result["trajectory"]["r"].shape[1] == 3
