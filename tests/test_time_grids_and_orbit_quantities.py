import numpy as np
import pytest
import ssapy
from astropy.time import Time

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics import all_orbital_quantities
from ssatk.time_functions import get_times


def test_time_grid_spans_the_2016_leap_second_in_si_seconds():
    # R3: a leap second (23:59:60) was inserted at the end of 2016-12-31, so 21
    # samples one SI second apart starting at 23:59:50 UTC end at 00:00:09 UTC,
    # not 00:00:10. Steps are 1 s to 1 microsecond in TAI.
    t = get_times(20, (1, "s"), t0="2016-12-31T23:59:50")
    assert len(t) == 21
    assert abs((t[-1] - Time("2017-01-01T00:00:09", scale="utc")).sec) < 1e-6
    np.testing.assert_allclose(np.diff((t.tai - t.tai[0]).sec), 1.0, atol=1e-6)


def test_time_grid_anchors_on_the_middle_and_final_epochs():
    # R1: a 2-day span at 6 h steps centred on tm has 9 samples with tm in the
    # middle; a 1-day span at 10 min steps ending at tf has 145 samples. 1 microsecond.
    tm = Time("2026-10-08T00:00:00", scale="utc")
    centred = get_times((2, "day"), (6, "hour"), tm=tm)
    assert len(centred) == 9 and abs((centred[4] - tm).sec) < 1e-6
    ending = get_times((1, "day"), (10, "min"), tf=tm)
    assert len(ending) == 145 and abs((ending[-1] - tm).sec) < 1e-6
    assert abs((ending[-1] - ending[0]).sec - 86400.0) < 1e-6


def test_all_orbital_quantities_agree_with_ssapy():
    # R2: from (a, e, i, pa, raan, M) the returned true anomaly reproduces M
    # through SSAPy (1e-12 rad); R1: periapsis/apoapsis input gives
    # a = (rp + ra)/2 and e = (ra - rp)/(ra + rp) exactly, and vis-viva holds.
    q = all_orbital_quantities(a=9000e3, e=0.3, i=0.5, pa=1.0, raan=2.0, ma=1.2, t=1.4e9)
    orbit = ssapy.Orbit.fromKeplerianElements(9000e3, 0.3, 0.5, 1.0, 2.0, q["trueAnomaly"], t=1.4e9)
    assert orbit.meanAnomaly == pytest.approx(1.2, abs=1e-12)
    assert q["rp"] == pytest.approx(9000e3 * 0.7, rel=1e-12)

    q2 = all_orbital_quantities(periapsis=7000e3, apoapsis=9000e3, t=1.4e9)
    assert (q2["a"], q2["e"]) == (8000e3, 0.125)
    speed2 = np.dot(q2["v"], q2["v"])
    assert speed2 == pytest.approx(EARTH_MU * (2 / np.linalg.norm(q2["r"]) - 1 / 8000e3), rel=1e-12)
