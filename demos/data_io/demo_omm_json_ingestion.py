"""Read a captured CelesTrak GP JSON record with a post-99,999 catalog ID.

The record is a data-format example from CelesTrak's GP endpoint, not a
current ephemeris. The demo runs offline and makes no accuracy claim.
"""

import json

from ssatk.io.catalog_ids import decode_tle_catalog_id
from ssatk.io.ccsds_omm import read_omm_json

# CelesTrak /NORAD/elements/gp.php?CATNR=100000&FORMAT=JSON,
# captured 2026-11-11. Metadata omitted by this endpoint uses OMM defaults.
SARAMAGO_GP = {
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
}

GALLERY_INCLUDE = False


def main():
    record = read_omm_json(json.dumps(SARAMAGO_GP))
    orbit = record.to_ssapy_orbit()
    assert decode_tle_catalog_id("A0000") == record.norad_cat_id
    print(f"{record.object_name}: NORAD {record.norad_cat_id}, epoch {record.epoch.isot}")
    print(f"GCRF position [m]: {orbit.r}")
    print(f"GCRF velocity [m/s]: {orbit.v}")
    return record, orbit


if __name__ == "__main__":
    main()
