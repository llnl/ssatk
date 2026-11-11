import json
from dataclasses import replace

import numpy as np
import pytest
from astropy.time import Time
from sgp4 import omm
from sgp4.api import Satrec
from ssapy.compute import rv
from ssapy.propagator import SGP4Propagator
from ssapy.utils import teme_to_gcrf

from ssatk.io.ccsds_omm import format_omm_xml, read_omm, read_omm_json, read_omm_xml

TERRA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ndm>
<omm id="CCSDS_OMM_VERS" version="2.0">
<header><CREATION_DATE>2026-08-27T19:05:28.000000</CREATION_DATE><ORIGINATOR>CELESTRAK</ORIGINATOR><COMMENT>first</COMMENT><COMMENT>second</COMMENT><HEADER_EXTRA>kept</HEADER_EXTRA></header>
<body><segment><metadata><OBJECT_NAME>TERRA</OBJECT_NAME><OBJECT_ID>1999-068A</OBJECT_ID><CENTER_NAME>EARTH</CENTER_NAME><REF_FRAME>TEME</REF_FRAME><TIME_SYSTEM>UTC</TIME_SYSTEM><MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY><USER_DEFINED>kept</USER_DEFINED></metadata><data><meanElements><EPOCH>2026-08-27T19:05:27.769056</EPOCH><MEAN_MOTION>14.61150647</MEAN_MOTION><ECCENTRICITY>.0002704</ECCENTRICITY><INCLINATION>97.9399</INCLINATION><RA_OF_ASC_NODE>286.4820</RA_OF_ASC_NODE><ARG_OF_PERICENTER>35.8635</ARG_OF_PERICENTER><MEAN_ANOMALY>75.8669</MEAN_ANOMALY></meanElements><tleParameters><EPHEMERIS_TYPE>0</EPHEMERIS_TYPE><CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE><NORAD_CAT_ID>25994</NORAD_CAT_ID><ELEMENT_SET_NO>999</ELEMENT_SET_NO><REV_AT_EPOCH>42006</REV_AT_EPOCH><BSTAR>.77763E-4</BSTAR><MEAN_MOTION_DOT>.338E-5</MEAN_MOTION_DOT><MEAN_MOTION_DDOT>0</MEAN_MOTION_DDOT></tleParameters></data></segment></body></omm>
</ndm>
"""

# Frozen flat-record response from CelesTrak's public
# /NORAD/elements/gp.php?CATNR=100000&FORMAT=json endpoint.
SARAMAGO_JSON = """[{"OBJECT_NAME":"SARAMAGO","OBJECT_ID":"2026-067CY","EPOCH":"2026-11-11T14:46:10.954848","MEAN_MOTION":15.2230114,"ECCENTRICITY":0.00065434,"INCLINATION":97.4792,"RA_OF_ASC_NODE":272.8406,"ARG_OF_PERICENTER":195.8082,"MEAN_ANOMALY":164.2951,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":100000,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":3411,"BSTAR":0.0004760888,"MEAN_MOTION_DOT":0.00010909,"MEAN_MOTION_DDOT":0}]"""


def test_celestrak_shaped_read_round_trip_and_comments():
    record = read_omm_xml(TERRA_XML)
    assert read_omm(TERRA_XML).object_id == record.object_id
    assert record.object_name == "TERRA"
    assert record.comments == ("first", "second")
    assert record.extra_fields["HEADER_EXTRA"] == "kept"
    assert record.extra_fields["USER_DEFINED"] == "kept"
    assert record.mean_motion_rad_s == pytest.approx(record.mean_motion)
    assert record.mean_motion_dot_rad_s2 == pytest.approx(record.mean_motion_dot)
    assert record.mean_motion_ddot_rad_s3 == pytest.approx(record.mean_motion_ddot)
    assert record.argument_of_pericenter == pytest.approx(record.argument_of_perigee)
    assert record.inclination_rad == pytest.approx(record.inclination)
    assert record.raan_rad == pytest.approx(record.raan)
    # Synthetic CelesTrak-shaped fixture: literal input values, converted to
    # SI. These are parser oracles, not a claim of an observed TERRA ephemeris.
    assert record.mean_motion_rad_s == pytest.approx(14.61150647*2*np.pi/86400, rel=1e-12, abs=0)
    assert record.mean_motion_dot_rad_s2 == pytest.approx(.338e-5*2*np.pi/86400**2, rel=1e-12, abs=0)
    assert record.inclination_rad == pytest.approx(np.deg2rad(97.9399), abs=1e-12)
    assert record.raan_rad == pytest.approx(np.deg2rad(286.4820), abs=1e-12)
    text = format_omm_xml(record)
    assert "<omm " in text and "<omm><omm" not in text
    restored = read_omm_xml(text)
    assert restored.object_id == record.object_id
    assert restored.comments == record.comments
    assert restored.to_satrec().bstar == pytest.approx(record.to_satrec().bstar)
    constructed = replace(record, extra_fields={**record.extra_fields, "MANUAL_EXTRA": "kept"})
    assert read_omm_xml(format_omm_xml(constructed)).extra_fields["MANUAL_EXTRA"] == "kept"


def test_official_celestrak_gp_json_record_uses_gp_units_and_metadata_defaults():
    payload = json.loads(SARAMAGO_JSON)
    assert len(payload) == 1 and len(payload[0]) == 17
    record = read_omm(SARAMAGO_JSON)
    assert read_omm_json(SARAMAGO_JSON).norad_cat_id == 100000
    assert record.object_name == "SARAMAGO"
    assert record.object_id == "2026-067CY"
    assert record.norad_cat_id == 100000
    assert record.center_name == "EARTH"
    assert record.reference_frame == "TEME"
    assert record.time_system == "UTC"
    assert record.mean_element_theory == "SGP4"
    assert record.creation_date is None
    assert record.originator == ""
    # Reference: the captured CelesTrak GP JSON values; JSON uses rev/day,
    # degrees, and rev/day**2, converted here to rad/s and rad/s**2.
    assert record.mean_motion == pytest.approx(15.2230114 * 2 * np.pi / 86400, rel=1e-12, abs=0)
    assert record.mean_motion_dot == pytest.approx(0.00010909 * 2 * np.pi / 86400**2, rel=1e-12, abs=0)
    assert record.mean_motion_ddot == 0.0
    assert record.inclination == pytest.approx(np.deg2rad(97.4792), abs=1e-12)
    assert record.raan == pytest.approx(np.deg2rad(272.8406), abs=1e-12)
    assert record.eccentricity == pytest.approx(0.00065434, rel=0, abs=0)
    assert record.to_satrec().satnum == 100000


def test_json_single_object_and_catalog_list():
    record = json.loads(SARAMAGO_JSON)[0]
    assert read_omm(json.dumps(record)).norad_cat_id == 100000

    second = dict(record, OBJECT_NAME="SARAMAGO TEST", NORAD_CAT_ID=100001)
    catalog = read_omm(json.dumps([record, second]))
    assert tuple(value.norad_cat_id for value in catalog) == (100000, 100001)


def test_nine_digit_catalog_id_builds_and_propagates_native_ssapy_orbit():
    record = replace(read_omm(SARAMAGO_JSON), norad_cat_id=999999999)
    sat = record.to_satrec()
    orbit = record.to_ssapy_orbit()
    assert sat.satnum == orbit._sat.satnum == 999999999

    # Use the accelerated implementation as the independent reference. The
    # SGP4 catalog number is metadata, so a legal Alpha-5 surrogate leaves the
    # propagated state unchanged.
    reference_fields = dict(record._sgp4_fields, NORAD_CAT_ID="339999")
    reference = Satrec()
    omm.initialize(reference, reference_fields)
    time = orbit.t + 600.0
    actual_r, actual_v = rv(orbit, np.array([time]), propagator=SGP4Propagator())
    error, r, v = reference.sgp4_tsince(10.0)
    assert error == 0
    rotation = teme_to_gcrf(time)
    # Independent native SGP4 Satrec reference; state differences are bounded
    # by 1e-8 m and 1e-11 m/s, respectively.
    assert np.allclose(actual_r[0], rotation @ (np.asarray(r) * 1.0e3), rtol=0.0, atol=1e-8)
    assert np.allclose(actual_v[0], rotation @ (np.asarray(v) * 1.0e3), rtol=0.0, atol=1e-11)


def test_submicrosecond_epoch_and_empty_celestrak_header_are_preserved():
    text = TERRA_XML.replace(
        "2026-08-27T19:05:28.000000", ""
    ).replace(
        "CELESTRAK", ""
    ).replace(
        "2026-08-27T19:05:27.769056", "2026-08-27T19:05:27.769056123"
    )
    record = read_omm_xml(text)
    rendered = format_omm_xml(record)
    assert "<CREATION_DATE />" in rendered
    assert "<ORIGINATOR />" in rendered
    assert "2026-08-27T19:05:27.769056123" in rendered
    orbit = record.to_ssapy_orbit()
    sat = orbit._sat
    assert orbit.t == pytest.approx(Time(sat.jdsatepoch, sat.jdsatepochF, format="jd").gps, abs=1e-6)

    ordinal = read_omm_xml(TERRA_XML.replace("2026-08-27T19:05:27.769056", "2026-239T19:05:27.769056"))
    assert ordinal.epoch.isot == "2026-08-27T19:05:27.769056000"


def test_ssapy_sgp4_matches_direct_native_satrec():
    record = read_omm_xml(TERRA_XML)
    direct = Satrec()
    omm.initialize(direct, dict(record._sgp4_fields))
    orbit = record.to_ssapy_orbit()
    time = orbit.t + 600.0
    actual_r, actual_v = rv(orbit, np.array([time]), propagator=SGP4Propagator())
    error, r, v = direct.sgp4_tsince(10.0)
    assert error == 0
    rotation = teme_to_gcrf(time)
    assert np.allclose(actual_r[0], rotation @ (np.asarray(r) * 1.0e3), rtol=0.0, atol=1e-8)
    assert np.allclose(actual_v[0], rotation @ (np.asarray(v) * 1.0e3), rtol=0.0, atol=1e-11)
