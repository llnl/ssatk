# Exact cardinal angles: 1e-14 rad; 3-4-5 range-rate Jacobian: 1e-14 per SI derivative component.
import numpy as np
import pytest

from ssatk.navigation import GroundStation


@pytest.fixture
def station():
    return GroundStation(0.0, 0.0, fast=True)


def test_range_and_range_rate_use_relative_si_geometry(station, monkeypatch):
    monkeypatch.setattr(
        GroundStation, "_station_rv",
        lambda self, time: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    )
    state = np.array([4.0, 4.0, 0.0, 2.0, 1.0, 0.0])
    assert station.predict(state, 0.0, "range").value == pytest.approx(5.0)
    prediction = station.predict(state, 0.0, "range_rate")
    assert prediction.value == pytest.approx(1.2)
    np.testing.assert_allclose(prediction.jacobian, [[0.256, -0.192, 0, 0.6, 0.8, 0]], rtol=0, atol=1e-14)


def test_az_el_and_ra_dec_wrap_and_visibility(station, monkeypatch):
    monkeypatch.setattr(
        GroundStation, "_station_rv",
        lambda self, time: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    )
    monkeypatch.setattr(
        GroundStation,
        "_local_basis",
        lambda self, time: (
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
    )
    state = np.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    az_el = station.predict(state, 0.0, "az_el")
    ra_dec = station.predict(state, 0.0, "ra_dec")
    np.testing.assert_allclose(az_el.value, [7 * np.pi / 4, 0.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ra_dec.value, [3 * np.pi / 2, np.pi / 4], rtol=0, atol=1e-14)
    assert az_el.visible
    assert np.all(az_el.jacobian[:, 3:] == 0.0)


def test_analytic_jacobians_match_finite_difference(station, monkeypatch):
    monkeypatch.setattr(
        GroundStation, "_station_rv",
        lambda self, time: (np.array([1.0, 2.0, 3.0]), np.array([-0.4, 0.8, 0.1])),
    )
    state = np.array([8.0, -4.0, 10.0, 1.0, 2.0, -0.5])
    step = 1e-5
    for measurement in ("range", "range_rate", "az_el", "ra_dec"):
        prediction = station.predict(state, 0.0, measurement)
        finite = []
        for index in range(6):
            plus = state.copy(); plus[index] += step
            minus = state.copy(); minus[index] -= step
            delta = station.predict(plus, 0.0, measurement).value - station.predict(minus, 0.0, measurement).value
            if measurement in {"az_el", "ra_dec"} and index < 3:
                delta[0] = (delta[0] + np.pi) % (2 * np.pi) - np.pi
            finite.append(np.atleast_1d(delta / (2 * step)))
        np.testing.assert_allclose(prediction.jacobian, np.asarray(finite).T, rtol=2e-5, atol=2e-7)

    assert prediction.kind == "ra_dec"


def test_station_rv_and_local_basis_share_one_earth_rotation_model():
    """R4: the station's own radial direction and its ENU "up" axis must agree
    to near machine precision now that both come from SSATK's single GCRF<->
    ITRF route. They used to come from two different models (``ssapy``'s
    ``EarthObserver``, IAU 1976/1980, for the station position; SSATK's core,
    IAU 2006/2000A, for the ENU basis), which left a ~6e-9 inconsistency here
    at the stricter tolerance below.
    """
    station = GroundStation(0.0, 0.0)
    east, north, up = station._local_basis(0.0)
    basis = np.array([east, north, up])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=2e-12, rtol=0)

    station_r, _station_v = station._station_rv(0.0)
    target = station_r + 1_000_000.0 * up + 1_000.0 * east
    prediction = station.predict(np.r_[target, [0.0, 0.0, 0.0]], 0.0, "az_el")
    assert prediction.visible
    np.testing.assert_allclose(prediction.value, [np.pi / 2, np.arctan2(1e6, 1e3)], atol=1e-10, rtol=0)
    # Geodetic up at lon=lat=0 is the station's own radial vector.
    np.testing.assert_allclose(up, station_r / np.linalg.norm(station_r), atol=1e-12, rtol=0)
