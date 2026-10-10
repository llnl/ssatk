from dataclasses import replace

import numpy as np
import pytest
from astropy.time import Time
from sgp4 import omm
from sgp4.api import Satrec
from ssapy.compute import rv
from ssapy.propagator import SGP4Propagator
from ssapy.utils import teme_to_gcrf

from ssatk.io.ccsds_omm import format_omm_xml, read_omm_xml

TERRA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ndm>
<omm id="CCSDS_OMM_VERS" version="2.0">
<header><CREATION_DATE>2026-08-27T19:05:28.000000</CREATION_DATE><ORIGINATOR>CELESTRAK</ORIGINATOR><COMMENT>first</COMMENT><COMMENT>second</COMMENT><HEADER_EXTRA>kept</HEADER_EXTRA></header>
<body><segment><metadata><OBJECT_NAME>TERRA</OBJECT_NAME><OBJECT_ID>1999-068A</OBJECT_ID><CENTER_NAME>EARTH</CENTER_NAME><REF_FRAME>TEME</REF_FRAME><TIME_SYSTEM>UTC</TIME_SYSTEM><MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY><USER_DEFINED>kept</USER_DEFINED></metadata><data><meanElements><EPOCH>2026-08-27T19:05:27.769056</EPOCH><MEAN_MOTION>14.61150647</MEAN_MOTION><ECCENTRICITY>.0002704</ECCENTRICITY><INCLINATION>97.9399</INCLINATION><RA_OF_ASC_NODE>286.4820</RA_OF_ASC_NODE><ARG_OF_PERICENTER>35.8635</ARG_OF_PERICENTER><MEAN_ANOMALY>75.8669</MEAN_ANOMALY></meanElements><tleParameters><EPHEMERIS_TYPE>0</EPHEMERIS_TYPE><CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE><NORAD_CAT_ID>25994</NORAD_CAT_ID><ELEMENT_SET_NO>999</ELEMENT_SET_NO><REV_AT_EPOCH>42006</REV_AT_EPOCH><BSTAR>.77763E-4</BSTAR><MEAN_MOTION_DOT>.338E-5</MEAN_MOTION_DOT><MEAN_MOTION_DDOT>0</MEAN_MOTION_DDOT></tleParameters></data></segment></body></omm>
</ndm>
"""


def test_celestrak_shaped_read_round_trip_and_comments():
    record = read_omm_xml(TERRA_XML)
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
    text = format_omm_xml(record)
    assert "<omm " in text and "<omm><omm" not in text
    restored = read_omm_xml(text)
    assert restored.object_id == record.object_id
    assert restored.comments == record.comments
    assert restored.to_satrec().bstar == pytest.approx(record.to_satrec().bstar)
    constructed = replace(record, extra_fields={**record.extra_fields, "MANUAL_EXTRA": "kept"})
    assert read_omm_xml(format_omm_xml(constructed)).extra_fields["MANUAL_EXTRA"] == "kept"


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
    assert orbit.t == pytest.approx(Time(sat.jdsatepoch, sat.jdsatepochF, format="jd").gps)

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
