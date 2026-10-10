import pytest
import importlib


read_3le_by_bit = importlib.import_module("ssatk.io.read_3le_by_bit")


TLE1 = "1 25544U 98067A   20029.54791435  .00000742  00000-0  20455-4 0  9993"
TLE2 = "2 25544  51.6436  23.4361 0007417  71.2720  40.4325 15.49147159210616"


def test_read_3le_by_bit_schema(tmp_path):
    path = tmp_path / "tle.txt"
    path.write_text(TLE1 + "\n" + TLE2 + "\n", encoding="utf-8")
    df = read_3le_by_bit.read_3le_by_bit(path)
    assert df.shape[0] == 1
    assert df.loc[0, "satnum"] == 25544
    assert df.loc[0, "classification"] == "U"
    assert df.loc[0, "eccentricity"] == pytest.approx(0.0007417)
    assert df.loc[0, "mean_motion"] == pytest.approx(15.49147159)
