from datetime import datetime

import pytest
from astropy.time import Time

from ssatk.plots.starfield import _to_datetime


@pytest.mark.parametrize("utc", ["2008-09-20T12:25:40.104", "2026-10-08T00:00:00"])
def test_star_field_epochs_follow_the_leap_second_table(utc):
    # R3/R2: GPS - UTC was 14 s in 2008 and 18 s since 2017; astropy's UTC for
    # GPS seconds and for a TT Time must come back to the same UTC instant to
    # 1 ms. A fixed 18 s offset put 2008 star fields 4 s (60 arcsec) late.
    expected = Time(utc, scale="utc").to_datetime()
    for epoch in (Time(utc, scale="utc").gps, Time(utc, scale="utc").tt):
        got = _to_datetime(epoch)
        assert abs((got - expected).total_seconds()) < 1e-3
    assert _to_datetime(datetime(2026, 1, 1)) == datetime(2026, 1, 1)
