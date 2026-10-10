"""Packaged IERS EOP (``ssa-data-core``) against astropy's bundled IERS-B.

Both tests load the snapshot through the default data-package search, so they
also fail if the toolkit stops finding ``environment/eop/finals2000A.all`` in
the split ``ssa-data-*`` distributions (the 1.0.7 regression coverage, where the
toolkit still looked for the retired ``ssapy_data`` package).
"""

import numpy as np
import pytest
from astropy.time import Time

from ssatk.environment_eop import load_packaged_eop

iers = pytest.importorskip("astropy.utils.iers")

# IERS finals2000A (Bulletin A/B) and IERS EOP C04 are separately published
# UT1-UTC series. Over 1990-2026 their daily values differ by at most 0.245 ms
# (median 0.012 ms), so 0.5 ms separates "same series" from "wrong day/step".
UT1_TOLERANCE_S = 0.5e-3


def _iers_b():
    # Ships with astropy-iers-data; opening it needs no network.
    return iers.IERS_B.open()


def _final_mjd_span(eop, iers_b):
    last_final = max(r.mjd_utc for r in eop.records if not r.predicted)
    return 48_000.0, min(last_final, float(iers_b["MJD"][-1].value)) - 1.0


def test_packaged_ut1_minus_utc_matches_iers_c04_at_nodes_and_midday():
    # R2: packaged finals2000A UT1-UTC vs astropy IERS-B (EOP C04), weekly
    # from 1990-04-19 to the last final record, at 00:00 and 12:00 UTC.
    eop = load_packaged_eop()
    assert eop.source.startswith("ssa_data_core:")
    iers_b = _iers_b()
    start, stop = _final_mjd_span(eop, iers_b)

    mjd = np.concatenate([np.arange(start, stop, 7.0), np.arange(start, stop, 7.0) + 0.5])
    times = Time(mjd, format="mjd", scale="utc")
    ours = np.array([eop.at(float(t)).ut1_minus_utc_s for t in times.gps])
    reference = iers_b.ut1_utc(times).to_value("s")

    np.testing.assert_allclose(ours, reference, rtol=0.0, atol=UT1_TOLERANCE_S)


@pytest.mark.parametrize(
    "utc",
    [
        "2016-12-31T00:00:00",
        "2016-12-31T12:00:00",
        "2016-12-31T23:59:00",
        "2017-01-01T00:00:00",
        "2015-06-30T18:00:00",
        "2012-06-30T23:00:00",
    ],
)
def test_packaged_ut1_minus_utc_is_continuous_through_leap_seconds(utc):
    # R6 + R2: UT1 - UTC steps by +1 s at the 2012-06-30, 2015-06-30 and
    # 2016-12-31 leap seconds. Linear interpolation across the step used to
    # spread it over the preceding day: +999.3 ms at 2016-12-31T23:59 UTC.
    # Reference: astropy IERS-B (EOP C04); tolerance 0.5 ms.
    eop = load_packaged_eop()
    t = Time(utc, scale="utc")
    ours = eop.at(float(t.gps)).ut1_minus_utc_s
    reference = float(_iers_b().ut1_utc(t).to_value("s"))
    assert abs(ours - reference) < UT1_TOLERANCE_S
