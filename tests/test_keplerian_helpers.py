import numpy as np
import pytest

from ssatk.constants import EARTH_MU, RGEO
from ssatk.orbital_mechanics import keplerian


def test_hkoe_period_longitude_and_anomaly_helpers(capsys):
    np.testing.assert_allclose(keplerian.hkoe([RGEO, 0.1, 30.0, 40.0, 50.0, 60.0]), [RGEO, 0.1, np.pi / 6, np.deg2rad(40), np.deg2rad(50), np.pi / 3])
    np.testing.assert_allclose(keplerian.hkoe(RGEO, 0.1, 30.0, 40.0, 50.0, 60.0), [RGEO, 0.1, np.pi / 6, np.deg2rad(40), np.deg2rad(50), np.pi / 3])

    with pytest.raises(ValueError, match="6-element"):
        keplerian.hkoe([1, 2, 3])
    with pytest.raises(ValueError, match="Must provide"):
        keplerian.hkoe(RGEO, 0.1)

    assert np.isclose(keplerian.period(RGEO), 2 * np.pi * np.sqrt(RGEO**3 / EARTH_MU))
    assert keplerian.mean_longitude(1.0, 2.0, 3.0) == 6.0
    assert np.isclose(keplerian.true_anomaly(eccentricity=0.0, eccentric_anomaly=0.5), 0.5)
    assert np.isfinite(keplerian.true_anomaly(eccentricity=0.1, mean_anomaly=0.5))
    assert np.isclose(keplerian.true_anomaly(true_longitude=3.0, longitude_of_ascending_node=1.0, argument_of_periapsis=0.5), 1.5)
    assert keplerian.true_anomaly() is None
    assert "Not enough information" in capsys.readouterr().out


def test_kepler_state_conversions_for_circular_equatorial_case():
    r, v = keplerian.kepler_to_state(a=RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, nu=0.0)
    np.testing.assert_allclose(r, [RGEO, 0.0, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(v, [0.0, np.sqrt(EARTH_MU / RGEO), 0.0], rtol=1e-12)

    r_loop, v_loop = keplerian.kepler_to_state_loop(a=RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, nu=0.0)
    np.testing.assert_allclose(r_loop, r)
    np.testing.assert_allclose(v_loop, v)

    r_many, v_many = keplerian.kepler_to_state_loop(
        a=np.array([RGEO, RGEO * 1.1]),
        e=np.array([0.0, 0.1]),
        i=np.array([0.0, 0.1]),
        pa=np.array([0.0, 0.2]),
        raan=np.array([0.0, 0.3]),
        nu=np.array([0.0, 0.4]),
    )
    assert r_many.shape == v_many.shape == (2, 3)

    with pytest.raises(ValueError, match="Semi-major"):
        keplerian.kepler_to_state_loop(a=-1.0)
    with pytest.raises(ValueError, match="Eccentricity"):
        keplerian.kepler_to_state_loop(a=RGEO, e=1.0)


def test_apsis_and_velocity_formula_helpers():
    rp = 7_000_000.0
    ra = 42_000_000.0
    a = (rp + ra) / 2.0
    e = (ra - rp) / (ra + rp)

    assert keplerian.a_from_periap(rp, ra) == a
    assert keplerian.e_from_periap(rp, ra) == e
    assert keplerian.ae_from_periap(rp, ra) == (a, e)
    assert keplerian.periapsis(a, e) == rp
    assert keplerian.apoapsis(a, e) == ra
    assert keplerian.peri_apo_from_rv(rp, ra) == {"a": a, "e": e}
    assert keplerian.apapsis_from_a_rp(a, rp) == ra
    assert np.isclose(keplerian.vcircular(rp, mu_=EARTH_MU), np.sqrt(EARTH_MU / rp))
    assert np.isclose(keplerian.vis_viva(a=a, r=rp, mu=EARTH_MU), np.sqrt(EARTH_MU * (2.0 / rp - 1.0 / a)))
    assert np.isclose(keplerian.v_periapsis(a, rp, EARTH_MU), keplerian.vis_viva(a=a, r=rp, mu=EARTH_MU))


@pytest.mark.parametrize("converter", ["kepler_to_state", "kepler_to_state_loop"])
@pytest.mark.parametrize("eccentricity", [0.0, 0.12, 0.7])
def test_kepler_to_state_matches_ssapy_and_conic_invariants(converter, eccentricity):
    # R2: SSAPy's Orbit.fromKeplerianElements; R1: vis-viva and h = sqrt(mu p).
    import ssapy
    from ssatk.constants import EARTH_MU
    from ssatk.orbital_mechanics import keplerian

    a, i, pa, raan, nu = 8000e3, np.radians(51.6), np.radians(30.0), np.radians(40.0), np.radians(70.0)
    r, v = (np.ravel(x) for x in getattr(keplerian, converter)(a, eccentricity, i, pa, raan, nu))
    reference = ssapy.Orbit.fromKeplerianElements(a, eccentricity, i, pa, raan, nu, t=0.0)

    np.testing.assert_allclose(r, reference.r, atol=1e-6)        # m
    np.testing.assert_allclose(v, reference.v, atol=1e-9)        # m/s
    radius = np.linalg.norm(r)
    assert np.linalg.norm(v) == pytest.approx(np.sqrt(EARTH_MU * (2.0 / radius - 1.0 / a)), rel=1e-12)
    semi_latus_rectum = a * (1.0 - eccentricity**2)
    assert np.linalg.norm(np.cross(r, v)) == pytest.approx(np.sqrt(EARTH_MU * semi_latus_rectum), rel=1e-12)


@pytest.mark.parametrize("eccentricity", [0.1, 0.3, 0.7, 0.95])
def test_true_anomaly_from_mean_anomaly_inverts_keplers_equation(eccentricity):
    # R1: nu -> E -> M in closed form, then true_anomaly(e, M) must return nu.
    from ssatk.orbital_mechanics import keplerian

    nu = np.radians([1.0, 45.0, 120.0, 179.0, 181.0, 300.0])
    ecc_anom = 2 * np.arctan2(np.sqrt(1 - eccentricity) * np.sin(nu / 2),
                              np.sqrt(1 + eccentricity) * np.cos(nu / 2))
    mean = np.mod(ecc_anom - eccentricity * np.sin(ecc_anom), 2 * np.pi)
    result = keplerian.true_anomaly(eccentricity=eccentricity, mean_anomaly=mean)
    np.testing.assert_allclose(np.angle(np.exp(1j * (result - nu))), 0.0, atol=1e-12)
