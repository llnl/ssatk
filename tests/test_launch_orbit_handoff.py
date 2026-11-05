"""Analytic mechanics, native SSAPy overlap and independent Astropy ground tracks."""

import numpy as np
import pytest
import ssapy
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
from astropy.time import Time
from astropy.utils import iers

from demos.launch.demo_launch_orbit_handoff import run_reference
from ssatk.data import data_path
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody


@pytest.fixture(scope="module", params=[False, True], ids=["two-body", "perturbed"])
def handoff(request):
    return run_reference(perturbed=request.param)


@pytest.mark.parametrize("mu", [3.986004418e14, 4.9048695e12])
def test_circular_coast_against_closed_form_with_explicit_mu(mu):
    """R1/R4: inclined circular motion, 1 mm/1 um/s, 1e-8 relative invariants.

    Two specified gravitational parameters detect a hidden Earth-mu default;
    circular elements are assessed without singular periapsis/node angles.
    """
    radius, inclination, epoch = 7e6, .4, 1.4e9
    speed, rate = np.sqrt(mu/radius), np.sqrt(mu/radius**3)
    body = SpacecraftBody(name="circular reference", mass=80., inertia=20.*np.eye(3))
    craft = Spacecraft(r=[radius, 0., 0.], v=speed*np.array([0., np.cos(inclination), np.sin(inclination)]),
                       t=epoch, body=body, area=2., cd=2.1, cr=1.2)
    orbit = craft.to_orbit(mu=mu)
    elapsed = np.linspace(0, 600, 21)
    angle = rate*elapsed
    expected_r = radius*np.column_stack((np.cos(angle), np.sin(angle)*np.cos(inclination), np.sin(angle)*np.sin(inclination)))
    expected_v = speed*np.column_stack((-np.sin(angle), np.cos(angle)*np.cos(inclination), np.cos(angle)*np.sin(inclination)))
    native_r, native_v = ssapy.rv(orbit, epoch+elapsed)
    trajectory = craft.propagate(times=epoch+elapsed, mu=mu, rtol=1e-11, atol=1e-9, max_step=5.)
    for r, v in ((native_r, native_v), (trajectory.r, trajectory.v)):
        np.testing.assert_allclose(r, expected_r, rtol=0, atol=1e-3)
        np.testing.assert_allclose(v, expected_v, rtol=0, atol=1e-6)
    assert orbit.a == pytest.approx(radius, abs=1e-7)
    assert orbit.e == pytest.approx(0, abs=1e-14)
    assert orbit.i == pytest.approx(inclination, abs=1e-14)
    assert orbit.energy == pytest.approx(-mu/(2*radius), rel=1e-8, abs=0)
    np.testing.assert_allclose(orbit.angularMomentum, radius*speed*np.array([0., -np.sin(inclination), np.cos(inclination)]),
                               rtol=1e-8, atol=0)


def test_ascent_handoff_against_mass_balance_and_native_coast(handoff):
    """R1/R2/R4: F/(Isp*g0) inventory, native overlap and state invariants.

    Budgets: 1e-10 kg inventory, 1 mm/1 um/s overlap, 1e-12 m/s^2 assembly,
    1e-8 relative energy/momentum. Cartesian continuity is exact in SI.
    """
    report, inputs, history = handoff["insertion"], handoff["inputs"], handoff["history"]
    mass = 200.-inputs["thrust_n"]*inputs["burn_duration_s"]/(inputs["isp_s"]*9.80665)
    assert report["mass_kg"] == pytest.approx(mass, abs=1e-10)
    assert report["tanks"][0]["propellant_kg"] == pytest.approx(mass-150., abs=1e-10)
    assert report["time_gps_s"] == inputs["epoch_gps_s"]+inputs["burn_duration_s"]
    index, = np.flatnonzero(history["time_gps_s"] == report["time_gps_s"])
    assert np.all(np.diff(history["time_gps_s"]) > 0)
    np.testing.assert_array_equal(history["position_gcrf_m"][index], report["position_gcrf_m"])
    np.testing.assert_array_equal(history["velocity_gcrf_m_s"][index], report["velocity_gcrf_m_s"])
    np.testing.assert_allclose(history["mass_kg"][index:], mass, rtol=0, atol=1e-10)
    comparison = handoff["comparison"]
    assert comparison["overlap_position_m"] < 1e-3
    assert comparison["overlap_velocity_m_s"] < 1e-6
    assert comparison["acceleration_assembly_m_s2"] < 1e-12
    mu, r, v = report["mu_m3_s2"], report["position_gcrf_m"], report["velocity_gcrf_m_s"]
    angular_momentum = np.cross(r, v)
    energy = np.dot(v, v)/2-mu/np.linalg.norm(r)
    eccentricity = np.linalg.norm(np.cross(v, angular_momentum)/mu-r/np.linalg.norm(r))
    assert report["specific_energy_m2_s2"] == pytest.approx(energy, rel=1e-8, abs=0)
    np.testing.assert_allclose(report["specific_angular_momentum_m2_s"], angular_momentum, rtol=1e-8, atol=0)
    assert report["semimajor_axis_m"] == pytest.approx(-mu/(2*energy), abs=1e-7)
    assert report["eccentricity"] == pytest.approx(eccentricity, abs=1e-14)
    if comparison["kepler_reference"] is not None:
        coast_r, coast_v = history["position_gcrf_m"][index:], history["velocity_gcrf_m_s"][index:]
        np.testing.assert_allclose(np.sum(coast_v**2, axis=1)/2-mu/np.linalg.norm(coast_r, axis=1), energy, rtol=1e-8, atol=0)
        np.testing.assert_allclose(np.cross(coast_r, coast_v), np.broadcast_to(angular_momentum, coast_r.shape), rtol=1e-8, atol=0)


def test_ascent_and_coast_ground_track_against_astropy(handoff):
    """R2: Astropy GCRS->ITRS and EarthLocation, shared packaged IERS table.

    3e-7 deg longitude/latitude and 0.02 m height allow the same dX/dY
    difference as test_earth_rotation_route; Astropy omits that correction.
    """
    history = handoff["history"]
    epochs = Time(history["time_gps_s"], format="gps")
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        itrs = GCRS(CartesianRepresentation(history["position_gcrf_m"].T*u.m), obstime=epochs).transform_to(ITRS(obstime=epochs))
        location = EarthLocation.from_geocentric(*itrs.cartesian.xyz)
    longitude_error = (history["longitude_deg"]-location.lon.deg+180)%360-180
    np.testing.assert_allclose(longitude_error, 0, rtol=0, atol=3e-7)
    np.testing.assert_allclose(history["latitude_deg"], location.lat.deg, rtol=0, atol=3e-7)
    np.testing.assert_allclose(history["height_wgs84_m"], location.height.to_value(u.m), rtol=0, atol=.02)
