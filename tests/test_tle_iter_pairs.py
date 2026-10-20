import pytest

from ssatk.io.tle_iter_pairs import tle_iter_pairs

# R3: the ISS (ZARYA) element set from the TLE format documentation
# (Space-Track / Wikipedia example, epoch 2008-264.51782528), checksums 7 and 7.
L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.mark.parametrize(
    "name_line, expected",
    [("0 ISS (ZARYA)", "ISS (ZARYA)"), ("ISS (ZARYA)", "ISS (ZARYA)"), ("2020 SO", "2020 SO"), (None, None)],
    ids=["spacetrack-3le", "celestrak-3le", "name-starting-with-a-digit", "bare-2le"],
)
def test_tle_pairs_keep_names_in_every_file_style(tmp_path, name_line, expected):
    # R3: both lines pass the published mod-10 checksum, and the name follows
    # the file's convention.
    path = tmp_path / "tle.txt"
    lines = ([name_line] if name_line else []) + [L1, L2]
    path.write_text("\n".join(lines) + "\n")
    assert list(tle_iter_pairs(path, validate_checksum=True)) == [(expected, L1, L2)]
