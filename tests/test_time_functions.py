import numpy as np
import pytest
from astropy.time import Time

from ssatk.time_functions import (
    dd_to_dms,
    dd_to_hms,
    dms_to_dd,
    hms_to_dd,
    to_gps,
)
from ssatk.time_functions.convert_gps_to_TT import _gpsToTT


def test_dms_and_hms_decimal_conversions_round_trip_common_values():
    assert dms_to_dd("12:30:0") == pytest.approx(12.5)
    assert dms_to_dd("-12:30:0") == pytest.approx(-12.5)
    np.testing.assert_allclose(
        dms_to_dd(["0:0:0", "1:30:0"]),
        [0.0, 1.5],
    )

    assert hms_to_dd("1:0:0") == pytest.approx(15.0)
    np.testing.assert_allclose(
        hms_to_dd(["0:0:0", "2:30:0"]),
        [0.0, 37.5],
    )

    assert dd_to_dms(12.5) == "12:30:0"
    assert dd_to_hms(15.0) == "1:0:0"
    assert dd_to_hms("15:0:0") == "1:0:0"


@pytest.mark.parametrize("degrees", [-15.0, -0.25, 0.0, 15.0, 187.5, 345.0, 359.99999999, 720.0 + 37.5])
def test_dd_to_hms_wraps_into_one_day_like_astropy(degrees):
    # R2: astropy Angle.wrap_at(360 deg).hms. Hours lie in [0, 24): -15 deg is
    # 23h and 359.99999999 deg rounds to 0h, not 24h. Agreement to 1e-4 s of time.
    import astropy.units as u
    from astropy.coordinates import Angle

    hours, minutes, seconds = (float(part) for part in dd_to_hms(degrees).split(":"))
    assert 0 <= hours < 24
    expected = Angle(degrees * u.deg).wrap_at(360 * u.deg).hour * 3600.0
    actual = hours * 3600.0 + minutes * 60.0 + seconds
    assert min(abs(actual - expected), 86400.0 - abs(actual - expected)) < 1e-4


def test_gps_seconds_and_tt_match_published_offsets():
    # R3: GPS time starts 1980-01-06T00:00:00 UTC and runs 18 s ahead of UTC
    # since 2017-01-01, so 2026-10-08T00:00:00 UTC is 1,475,452,818 GPS
    # seconds. TT - GPS = 19 s + 32.184 s, so GPS 0 is MJD 44244 + 51.184 s in
    # TT. Exact to 1e-6 s.
    utc = "2026-10-08T00:00:00"
    expected = 1475452818.0
    assert to_gps(Time(utc, scale="utc")) == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(
        to_gps([Time(utc, scale="utc"), Time("2026-10-08T00:01:00", scale="utc")]),
        [expected, expected + 60.0],
        rtol=0,
        atol=1e-6,
    )
    assert _gpsToTT(0.0) == pytest.approx(44244.0 + 51.184 / 86400.0, abs=1e-6 / 86400.0)


def test_julian_date_matches_meeus_and_astropy():
    # R3: Meeus, Astronomical Algorithms (2nd ed.), Example 7.a:
    # 1957 Oct 4.81 = JD 2436116.31. R2: astropy Time.jd (proleptic
    # Gregorian, like ERFA cal2jd) for dates on both sides of 1582-10-15.
    # Both to 1e-6 s.
    from datetime import datetime, timedelta

    from ssatk.time_functions import julian_date

    sputnik = datetime(1957, 10, 4) + timedelta(days=0.81)
    assert julian_date(sputnik) == pytest.approx(2436116.31, abs=1e-6 / 86400.0)
    for value in (
        datetime(2026, 10, 8, 12, 34, 56, 789000),
        datetime(1900, 2, 28, 23, 59, 59),
        datetime(1582, 10, 15),
        datetime(1582, 10, 4),
        datetime(1000, 6, 1, 6),
    ):
        expected = Time(value, scale="tt").jd  # calendar-to-JD only; no leap-second table
        assert julian_date(value) == pytest.approx(expected, abs=1e-6 / 86400.0), value


def test_negative_sub_degree_dms_angles_keep_their_sign():
    # R3 (sexagesimal definition): the sign applies to the whole angle, so a
    # declination of -00:30:00 is -0.5 deg and must not come back as +0.5 deg.
    from ssatk.time_functions.convert_dd_and_dms import dd_to_dms, dms_to_dd

    cases = {"-00:30:00": -0.5, "-0:00:36": -0.01, "-10:30:36": -10.51, "10:30:36": 10.51, "+00:30:00": 0.5}
    for text, degrees in cases.items():
        assert dms_to_dd(text) == pytest.approx(degrees, abs=1e-12), text
    assert dd_to_dms(-0.5) == "-0:30:0"
    assert dd_to_dms(-10.51) == "-10:30:36"
    for degrees in (-0.5, -0.01, -0.999999, -10.51, -89.9999, 0.0, 45.25):
        assert dms_to_dd(dd_to_dms(degrees)) == pytest.approx(degrees, abs=1e-6), degrees
