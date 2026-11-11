import json

import pytest
from astropy.time import Time

from ssatk.io.catalog_ids import decode_tle_catalog_id, encode_tle_catalog_id
from ssatk.io.tle_iter_pairs import tle_iter_pairs
from ssatk.io.tle_prop_to_time import tle_prop_to_time
from ssatk.io.tle_updater import (
    _norad_from_tle,
    _tle_checksum,
    fetch_active_catalog,
    fetch_omm_celestrak,
    verify_satellites,
)

L1 = "1 25544U 98067A   25001.00000000  .00016717  00000+0  10270-3 0  9000"
L2 = "2 25544  51.6400 120.0000 0007000  90.0000  10.0000 15.50000000 00001"


def _with_catalog_id(line, token):
    body = line[:2] + token + line[7:68]
    return body + str(_tle_checksum(body))


@pytest.mark.parametrize(
    ("token", "catalog_id"),
    [
        ("00000", 0),
        ("25544", 25544),
        ("99999", 99999),
        ("A0000", 100000),
        ("A9999", 109999),
        ("H9999", 179999),
        ("J0000", 180000),
        ("Z9999", 339999),
    ],
)
def test_alpha5_matches_defined_catalog_mapping(token, catalog_id):
    assert decode_tle_catalog_id(token) == catalog_id
    assert encode_tle_catalog_id(catalog_id) == token


@pytest.mark.parametrize("token", ["A00X0", "I0000", "O0000", "Z10000", "100000"])
def test_invalid_alpha5_fields_are_rejected(token):
    with pytest.raises(ValueError):
        decode_tle_catalog_id(token)


@pytest.mark.parametrize("catalog_id", [-1, 340000, 1_000_000_000])
def test_ids_outside_tle_alpha5_range_cannot_be_encoded(catalog_id):
    with pytest.raises(ValueError):
        encode_tle_catalog_id(catalog_id)


def test_alpha5_ids_survive_pairing_verification_and_sgp4(tmp_path):
    line1 = _with_catalog_id(L1, "A0000")
    line2 = _with_catalog_id(L2, "A0000")
    assert _norad_from_tle(line1) == 100000

    report = verify_satellites(
        [{"name": "SARAMAGO", "type": "tle", "line1": line1, "line2": line2}],
        verbose=False,
    )
    assert report["unique_norads"] == 1
    assert report["no_norad"] == 0
    assert report["id_mismatch"] == []
    assert report["bad_checksum"] == []

    path = tmp_path / "alpha5.tle"
    path.write_text(f"0 SARAMAGO\n{line1}\n{line2}\n", encoding="utf-8")
    pairs = list(tle_iter_pairs(path, validate_checksum=True))
    assert len(pairs) == 1
    orbits = tle_prop_to_time(Time("2025-01-01T00:01:00", scale="utc"), path)
    assert len(orbits) == 1
    assert orbits[0]._sat.satnum == 100000


def test_bulk_catalog_query_includes_alpha5_ids(monkeypatch):
    line1 = _with_catalog_id(L1, "A0000")
    line2 = _with_catalog_id(L2, "A0000")
    requested_urls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps([{
                "OBJECT_NAME": "SARAMAGO",
                "TLE_LINE1": line1,
                "TLE_LINE2": line2,
            }]).encode()

    monkeypatch.setattr("ssatk.io.tle_updater._credentials", lambda: ("user", "password"))
    monkeypatch.setattr("ssatk.io.tle_updater._spacetrack_login", lambda cookie: "session")

    def open_url(request, **kwargs):
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr("ssatk.io.tle_updater.urllib.request.urlopen", open_url)
    satellites, cookie = fetch_active_catalog(max_satellites=1, verbose=False)

    assert cookie == "session"
    assert "%3C100000" not in requested_urls[0]
    assert len(satellites) == 1
    assert _norad_from_tle(satellites[0]["line1"]) == 100000


def test_celestrak_fetch_requests_omm_json_for_large_catalog_id(monkeypatch):
    payload = [{
        "OBJECT_NAME": "SARAMAGO",
        "OBJECT_ID": "2026-067CY",
        "EPOCH": "2026-11-11T14:46:10.954848",
        "MEAN_MOTION": 15.2230114,
        "ECCENTRICITY": 0.00065434,
        "INCLINATION": 97.4792,
        "RA_OF_ASC_NODE": 272.8406,
        "ARG_OF_PERICENTER": 195.8082,
        "MEAN_ANOMALY": 164.2951,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 100000,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 3411,
        "BSTAR": 0.0004760888,
        "MEAN_MOTION_DOT": 0.00010909,
        "MEAN_MOTION_DDOT": 0,
    }]
    requested_urls = []
    cache = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def open_url(request, **kwargs):
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr("ssatk.io.tle_updater.urllib.request.urlopen", open_url)
    monkeypatch.setattr("ssatk.io.tle_updater._load_cache", lambda: cache)
    monkeypatch.setattr("ssatk.io.tle_updater._save_cache", lambda value: None)
    record = fetch_omm_celestrak(100000)
    cached_record = fetch_omm_celestrak(100000)

    assert record.object_name == "SARAMAGO"
    assert record.norad_cat_id == 100000
    assert cached_record.norad_cat_id == 100000
    assert "CATNR=100000" in requested_urls[0]
    assert "FORMAT=JSON" in requested_urls[0]
    assert len(requested_urls) == 1
    assert cache["100000"]["source"] == "celestrak-omm"
    # Independent native SGP4 API retains the numeric catalog identity.
    assert record.to_satrec().satnum == 100000
