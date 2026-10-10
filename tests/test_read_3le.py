import numpy as np
import pytest
from astropy.time import Time
from sgp4.api import Satrec

from ssatk.io.read_3le import read_3le

# R3: the ISS (ZARYA) element set from the TLE format documentation.
L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


def test_read_3le_matches_sgp4_twoline2rv(tmp_path):
    # R2: sgp4's Satrec.twoline2rv parses the same lines. Angles to 1e-12 rad,
    # eccentricity and B* exactly, mean motion to 1e-12 relative, and the epoch
    # to 1 microsecond (sgp4 gives JD in UTC; 2008 has a leap second on Dec 31).
    path = tmp_path / "iss.txt"
    path.write_text(f"0 ISS (ZARYA)\n{L1}\n{L2}\n")
    row = read_3le(path).iloc[0]
    sat = Satrec.twoline2rv(L1, L2)
    assert row["inc_rad"] == pytest.approx(sat.inclo, abs=1e-12)
    assert row["raan_rad"] == pytest.approx(sat.nodeo, abs=1e-12)
    assert row["argp_rad"] == pytest.approx(sat.argpo, abs=1e-12)
    assert row["ecc"] == sat.ecco
    assert row["drag"] == pytest.approx(sat.bstar, rel=1e-12, abs=0)
    assert row["n_rad_s"] == pytest.approx(sat.no_kozai / 60.0, rel=1e-12)
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format="jd", scale="utc").gps
    assert row["epoch_gps"] == pytest.approx(epoch, abs=1e-6)
    assert row["mean_anomaly_deg"] == pytest.approx(np.degrees(sat.mo), abs=1e-10)
