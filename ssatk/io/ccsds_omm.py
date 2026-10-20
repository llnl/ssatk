"""CCSDS Orbit Mean-Elements Message (OMM) XML 2.0 interoperability.

The record uses UTC, Earth-centered TEME SGP4 mean elements.  Public angular
values are radians and mean motion (and its derivatives) use rad/s powers;
XML uses the CCSDS degree and revolution/day conventions.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

import numpy as np
from astropy.time import Time

from ssatk._paths import ensure_file_parent

__all__ = [
    "OMMRecord",
    "OrbitMeanElementsMessage",
    "format_omm",
    "format_omm_xml",
    "read_omm",
    "read_omm_xml",
    "write_omm",
    "write_omm_xml",
]

_DAY = 86400.0
_TWO_PI = 2.0 * math.pi
_METADATA = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
)
_MEAN_ELEMENTS = (
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
)
_TLE_PARAMETERS = (
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)
_UNITS = {
    "MEAN_MOTION": {"rev/day", "rev/d"},
    "ECCENTRICITY": {"1"},
    "INCLINATION": {"deg"},
    "RA_OF_ASC_NODE": {"deg"},
    "ARG_OF_PERICENTER": {"deg"},
    "MEAN_ANOMALY": {"deg"},
    "BSTAR": {"1/ER"},
    "MEAN_MOTION_DOT": {"rev/day**2", "rev/d**2"},
    "MEAN_MOTION_DDOT": {"rev/day**3", "rev/d**3"},
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].upper()


def _children(element, name):
    return [child for child in element if _local(child.tag) == name]


def _one(element, name, *, required=True):
    matches = _children(element, name)
    if len(matches) > 1:
        raise ValueError(f"OMM contains duplicate {name} fields.")
    if not matches:
        if required:
            raise ValueError(f"OMM is missing required {name}.")
        return None
    return (matches[0].text or "").strip()


def _mapping(values):
    if values is None:
        return MappingProxyType({})
    copied = {str(key): str(value) for key, value in dict(values).items()}
    if len(copied) != len(values):
        raise ValueError("extra fields must have unique keys.")
    return MappingProxyType(copied)


def _finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OMM {name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"OMM {name} must be finite.")
    return result


def _time(value, name):
    try:
        if isinstance(value, Time):
            result = value.utc.copy()
        else:
            text = str(value).strip().removesuffix("Z")
            if re.match(r"^\d{4}-\d{3}T", text):
                text = f"{text[:4]}:{text[5:8]}:{text[9:]}"
                result = Time(text, format="yday", scale="utc")
            else:
                result = Time(text, format="isot", scale="utc")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OMM {name} must be an ISO UTC date.") from exc
    if not math.isfinite(float(result.jd)):
        raise ValueError(f"OMM {name} must be finite.")
    result.precision = 9
    return result.utc


def _readonly_extras(values):
    return _mapping(values)


@dataclass(frozen=True)
class OMMRecord:
    """One CCSDS OMM 2.0 segment with SI/radian numerical properties."""

    object_name: str
    object_id: str
    center_name: str
    reference_frame: str
    time_system: str
    mean_element_theory: str
    epoch: Time
    mean_motion: float
    eccentricity: float
    inclination: float
    raan: float
    argument_of_perigee: float
    mean_anomaly: float
    ephemeris_type: int
    classification_type: str
    norad_cat_id: int
    element_set_no: int
    rev_at_epoch: int
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    version: str = "2.0"
    omm_id: str = "CCSDS_OMM_VERS"
    creation_date: Time | None = None
    originator: str = ""
    message_id: str | None = None
    comments: tuple[str, ...] = ()
    extra_fields: Mapping[str, str] = field(default_factory=dict)
    _extra_sections: Mapping[str, tuple[tuple[str, str], ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _sgp4_fields: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if str(self.version).strip() != "2.0":
            raise ValueError("only CCSDS OMM version 2.0 is supported.")
        text = (self.object_name, self.object_id, self.center_name, self.reference_frame,
                self.time_system, self.mean_element_theory, self.classification_type)
        if not all(isinstance(value, str) and value.strip() for value in text):
            raise ValueError("OMM identity and metadata fields must be non-empty strings.")
        if self.center_name.strip().upper() != "EARTH":
            raise ValueError("only Earth-centered OMM records are supported.")
        if self.reference_frame.strip().upper() != "TEME":
            raise ValueError("only TEME OMM records are supported.")
        if self.time_system.strip().upper() != "UTC":
            raise ValueError("only UTC OMM records are supported.")
        if self.mean_element_theory.strip().upper() != "SGP4":
            raise ValueError("only SGP4 OMM records are supported.")
        if not isinstance(self.epoch, Time):
            raise TypeError("epoch must be an astropy.time.Time.")
        epoch = _time(self.epoch, "EPOCH")
        creation = None if self.creation_date is None else _time(self.creation_date, "CREATION_DATE")
        values = {
            "mean_motion": _finite(self.mean_motion, "MEAN_MOTION"),
            "eccentricity": _finite(self.eccentricity, "ECCENTRICITY"),
            "inclination": _finite(self.inclination, "INCLINATION"),
            "raan": _finite(self.raan, "RA_OF_ASC_NODE"),
            "argument_of_perigee": _finite(self.argument_of_perigee, "ARG_OF_PERICENTER"),
            "mean_anomaly": _finite(self.mean_anomaly, "MEAN_ANOMALY"),
            "bstar": _finite(self.bstar, "BSTAR"),
            "mean_motion_dot": _finite(self.mean_motion_dot, "MEAN_MOTION_DOT"),
            "mean_motion_ddot": _finite(self.mean_motion_ddot, "MEAN_MOTION_DDOT"),
        }
        if values["mean_motion"] <= 0.0 or not 0.0 <= values["eccentricity"] < 1.0:
            raise ValueError("OMM mean motion must be positive and eccentricity in [0, 1).")
        if not 0.0 <= values["inclination"] <= math.pi:
            raise ValueError("OMM INCLINATION must be in [0, 180] degrees.")
        if not all(0.0 <= values[name] < _TWO_PI for name in ("raan", "argument_of_perigee", "mean_anomaly")):
            raise ValueError("OMM angular fields must be in [0, 360) degrees.")
        if int(self.ephemeris_type) != 0:
            raise ValueError("only SGP4 EPHEMERIS_TYPE 0 is supported.")
        if not 0 <= int(self.norad_cat_id) <= 99999:
            raise ValueError("OMM NORAD_CAT_ID must be in [0, 99999].")
        if not 0 <= int(self.element_set_no) <= 9999 or not 0 <= int(self.rev_at_epoch) <= 99999:
            raise ValueError("OMM TLE integer fields are out of range.")
        object.__setattr__(self, "version", "2.0")
        object.__setattr__(self, "center_name", "EARTH")
        object.__setattr__(self, "reference_frame", "TEME")
        object.__setattr__(self, "time_system", "UTC")
        object.__setattr__(self, "mean_element_theory", "SGP4")
        classification = self.classification_type.strip().upper()
        if len(classification) != 1 or not classification.isalpha():
            raise ValueError("OMM CLASSIFICATION_TYPE must be one letter.")
        object.__setattr__(self, "classification_type", classification)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "creation_date", creation)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ephemeris_type", int(self.ephemeris_type))
        object.__setattr__(self, "norad_cat_id", int(self.norad_cat_id))
        object.__setattr__(self, "element_set_no", int(self.element_set_no))
        object.__setattr__(self, "rev_at_epoch", int(self.rev_at_epoch))
        object.__setattr__(self, "comments", tuple(str(value) for value in self.comments))
        object.__setattr__(self, "extra_fields", _readonly_extras(self.extra_fields))
        object.__setattr__(self, "_extra_sections", MappingProxyType({
            str(section): tuple((str(key), str(value)) for key, value in entries)
            for section, entries in dict(self._extra_sections).items()
        }))
        fields = dict(self._sgp4_fields) or self._generated_sgp4_fields()
        missing = set(_MEAN_ELEMENTS + _TLE_PARAMETERS) - set(fields)
        if missing:
            raise ValueError(f"OMM SGP4 fields are missing {', '.join(sorted(missing))}.")
        fields["EPOCH"] = self.epoch.to_datetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
        object.__setattr__(self, "_sgp4_fields", MappingProxyType({key: str(value) for key, value in fields.items()}))

    @property
    def mean_motion_rad_s(self):
        return self.mean_motion

    @property
    def mean_motion_dot_rad_s2(self):
        return self.mean_motion_dot

    @property
    def mean_motion_ddot_rad_s3(self):
        return self.mean_motion_ddot

    @property
    def argument_of_pericenter(self):
        return self.argument_of_perigee

    @property
    def inclination_rad(self):
        return self.inclination

    @property
    def raan_rad(self):
        return self.raan

    def _generated_sgp4_fields(self):
        return {
            "EPOCH": self.epoch.to_datetime().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "MEAN_MOTION": repr(self.mean_motion * _DAY / _TWO_PI),
            "ECCENTRICITY": repr(self.eccentricity),
            "INCLINATION": repr(math.degrees(self.inclination)),
            "RA_OF_ASC_NODE": repr(math.degrees(self.raan)),
            "ARG_OF_PERICENTER": repr(math.degrees(self.argument_of_perigee)),
            "MEAN_ANOMALY": repr(math.degrees(self.mean_anomaly)),
            "EPHEMERIS_TYPE": str(self.ephemeris_type),
            "CLASSIFICATION_TYPE": self.classification_type,
            "NORAD_CAT_ID": str(self.norad_cat_id),
            "ELEMENT_SET_NO": str(self.element_set_no),
            "REV_AT_EPOCH": str(self.rev_at_epoch),
            "BSTAR": repr(self.bstar),
            "MEAN_MOTION_DOT": repr(self.mean_motion_dot * _DAY**2 / _TWO_PI),
            "MEAN_MOTION_DDOT": repr(self.mean_motion_ddot * _DAY**3 / _TWO_PI),
            "OBJECT_ID": self.object_id,
        }

    def to_satrec(self):
        """Build the native SGP4 record from the unrounded OMM fields."""
        from sgp4 import omm
        from sgp4.api import Satrec
        sat = Satrec()
        omm.initialize(sat, dict(self._sgp4_fields))
        return sat

    def to_ssapy_orbit(self):
        """Construct an SSAPy Orbit carrying its native SGP4 record."""
        from ssapy import Orbit
        from ssapy.utils import teme_to_gcrf
        sat = self.to_satrec()
        error, r, v = sat.sgp4_tsince(0.0)
        if error:
            raise ValueError(f"SGP4 rejected OMM elements with error {error}.")
        epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format="jd", scale="utc")
        rotation = teme_to_gcrf(epoch.gps)
        orbit = Orbit(rotation @ (np.asarray(r) * 1.0e3), rotation @ (np.asarray(v) * 1.0e3), epoch.gps)
        orbit._sat = sat
        return orbit


OrbitMeanElementsMessage = OMMRecord


def _split_source(source):
    if hasattr(source, "read"):
        value = source.read()
        return value.decode() if isinstance(value, bytes) else str(value)
    if isinstance(source, Path) or hasattr(source, "__fspath__"):
        return Path(source).read_text(encoding="utf-8")
    if isinstance(source, str):
        return source if "<" in source else Path(source).read_text(encoding="utf-8")
    raise TypeError("source must be OMM XML, a path, or a text stream.")


def _parse_segment(omm):
    version = omm.attrib.get("version", "")
    if version != "2.0":
        raise ValueError(f"only CCSDS OMM version 2.0 is supported, got {version!r}.")
    header = _children(omm, "HEADER")
    body = _children(omm, "BODY")
    if len(header) != 1 or len(body) != 1:
        raise ValueError("OMM must contain one HEADER and one BODY.")
    header, body = header[0], body[0]
    segments = _children(body, "SEGMENT")
    if len(segments) != 1:
        raise ValueError("each OMM message must contain one SEGMENT.")
    segment = segments[0]
    metadata_nodes, data_nodes = _children(segment, "METADATA"), _children(segment, "DATA")
    if len(metadata_nodes) != 1 or len(data_nodes) != 1:
        raise ValueError("OMM SEGMENT must contain one METADATA and one DATA.")
    metadata, data = metadata_nodes[0], data_nodes[0]
    unsupported_blocks = [
        _local(child.tag)
        for child in data
        if _local(child.tag) not in {"MEANELEMENTS", "TLEPARAMETERS"}
    ]
    if unsupported_blocks:
        raise ValueError(f"unsupported OMM data block {unsupported_blocks[0]}.")
    mean_nodes, tle_nodes = _children(data, "MEANELEMENTS"), _children(data, "TLEPARAMETERS")
    if len(mean_nodes) != 1 or len(tle_nodes) != 1:
        raise ValueError("OMM DATA must contain meanElements and tleParameters.")
    mean, tle = mean_nodes[0], tle_nodes[0]
    for node in (mean, tle):
        for child in node:
            key = _local(child.tag)
            unit = child.attrib.get("units")
            if unit is not None and key in _UNITS and unit.replace(" ", "") not in _UNITS[key]:
                raise ValueError(f"OMM {key} has unsupported units {unit!r}.")
    known = set(_METADATA + _MEAN_ELEMENTS + _TLE_PARAMETERS + ("COMMENT", "MESSAGE_ID", "CREATION_DATE", "ORIGINATOR"))
    extras = {}
    sections = {"header": [], "metadata": [], "meanElements": [], "tleParameters": []}
    for section, node in (("header", header), ("metadata", metadata), ("meanElements", mean), ("tleParameters", tle)):
        for child in node:
            key, value = _local(child.tag), (child.text or "").strip()
            if key not in known or key == "COMMENT":
                if list(child) or (child.attrib and key != "COMMENT"):
                    raise ValueError(f"unsupported structured OMM field {key}.")
                extras[key] = extras.get(key, value)
                sections[section].append((key, value))
    comments = tuple((child.text or "").strip() for child in header if _local(child.tag) == "COMMENT")
    def field_value(node, key, *, required=True):
        value = _one(node, key, required=required)
        if required and not value:
            raise ValueError(f"OMM field {key} must not be empty.")
        return value
    creation_text = field_value(header, "CREATION_DATE", required=False)
    return OMMRecord(
        object_name=field_value(metadata, "OBJECT_NAME"), object_id=field_value(metadata, "OBJECT_ID"),
        center_name=field_value(metadata, "CENTER_NAME"), reference_frame=field_value(metadata, "REF_FRAME"),
        time_system=field_value(metadata, "TIME_SYSTEM"), mean_element_theory=field_value(metadata, "MEAN_ELEMENT_THEORY"),
        epoch=_time(field_value(mean, "EPOCH"), "EPOCH"),
        mean_motion=_finite(field_value(mean, "MEAN_MOTION"), "MEAN_MOTION") * _TWO_PI / _DAY,
        eccentricity=_finite(field_value(mean, "ECCENTRICITY"), "ECCENTRICITY"),
        inclination=math.radians(_finite(field_value(mean, "INCLINATION"), "INCLINATION")),
        raan=math.radians(_finite(field_value(mean, "RA_OF_ASC_NODE"), "RA_OF_ASC_NODE")),
        argument_of_perigee=math.radians(_finite(field_value(mean, "ARG_OF_PERICENTER"), "ARG_OF_PERICENTER")),
        mean_anomaly=math.radians(_finite(field_value(mean, "MEAN_ANOMALY"), "MEAN_ANOMALY")),
        ephemeris_type=int(field_value(tle, "EPHEMERIS_TYPE")), classification_type=field_value(tle, "CLASSIFICATION_TYPE"),
        norad_cat_id=int(field_value(tle, "NORAD_CAT_ID")), element_set_no=int(field_value(tle, "ELEMENT_SET_NO")),
        rev_at_epoch=int(field_value(tle, "REV_AT_EPOCH")), bstar=_finite(field_value(tle, "BSTAR"), "BSTAR"),
        mean_motion_dot=_finite(field_value(tle, "MEAN_MOTION_DOT"), "MEAN_MOTION_DOT") * _TWO_PI / _DAY**2,
        mean_motion_ddot=_finite(field_value(tle, "MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT") * _TWO_PI / _DAY**3,
        omm_id=omm.attrib.get("id", "CCSDS_OMM_VERS"), version=version,
        creation_date=None if not creation_text else _time(creation_text, "CREATION_DATE"),
        originator=field_value(header, "ORIGINATOR", required=False) or "",
        message_id=field_value(header, "MESSAGE_ID", required=False), comments=comments,
        extra_fields={key: value for key, value in extras.items() if key != "COMMENT"},
        _extra_sections=sections,
        _sgp4_fields={
            **{key: field_value(mean, key) for key in _MEAN_ELEMENTS},
            **{key: field_value(tle, key) for key in _TLE_PARAMETERS},
            "OBJECT_ID": field_value(metadata, "OBJECT_ID"),
        },
    )


def read_omm_xml(source: str | Path | TextIO):
    """Read one OMM or return an immutable tuple for an NDM catalog."""
    try:
        root = ET.fromstring(_split_source(source))
    except ET.ParseError as exc:
        raise ValueError("invalid OMM XML.") from exc
    name = _local(root.tag)
    if name == "OMM":
        nodes = [root]
    elif name == "NDM":
        nodes = [node for node in root.iter() if _local(node.tag) == "OMM"]
    else:
        raise ValueError("OMM XML root must be omm or ndm.")
    if not nodes:
        raise ValueError("NDM contains no OMM messages.")
    records = tuple(_parse_segment(node) for node in nodes)
    return records[0] if len(records) == 1 else records


def _text(parent, tag, value):
    ET.SubElement(parent, tag).text = str(value)


def _xml_number(value, precision):
    return f"{float(value):.{precision}g}"


def _record_element(record, precision):
    omm = ET.Element("omm", {"id": record.omm_id, "version": "2.0"})
    header = ET.SubElement(omm, "header")
    # CelesTrak's live per-object endpoint emits these required elements empty.
    _text(header, "CREATION_DATE", "" if record.creation_date is None else record.creation_date.isot)
    _text(header, "ORIGINATOR", record.originator)
    if record.message_id:
        _text(header, "MESSAGE_ID", record.message_id)
    for comment in record.comments:
        _text(header, "COMMENT", comment)
    for key, value in record._extra_sections.get("header", ()):
        if key != "COMMENT":
            _text(header, key, value)
    body = ET.SubElement(omm, "body")
    segment = ET.SubElement(body, "segment")
    metadata = ET.SubElement(segment, "metadata")
    for key, value in (("OBJECT_NAME", record.object_name), ("OBJECT_ID", record.object_id), ("CENTER_NAME", "EARTH"),
                       ("REF_FRAME", "TEME"), ("TIME_SYSTEM", "UTC"), ("MEAN_ELEMENT_THEORY", "SGP4")):
        _text(metadata, key, value)
    section_extra_keys = {
        key
        for entries in record._extra_sections.values()
        for key, _ in entries
    }
    for key, value in record.extra_fields.items():
        if key not in section_extra_keys:
            _text(metadata, key, value)
    data = ET.SubElement(segment, "data")
    mean = ET.SubElement(data, "meanElements")
    mean_values = (("EPOCH", record.epoch.isot), ("MEAN_MOTION", record.mean_motion * _DAY / _TWO_PI),
                   ("ECCENTRICITY", record.eccentricity), ("INCLINATION", math.degrees(record.inclination)),
                   ("RA_OF_ASC_NODE", math.degrees(record.raan)), ("ARG_OF_PERICENTER", math.degrees(record.argument_of_perigee)),
                   ("MEAN_ANOMALY", math.degrees(record.mean_anomaly)))
    for key, value in mean_values:
        _text(mean, key, _xml_number(value, precision) if key != "EPOCH" else value)
    tle = ET.SubElement(data, "tleParameters")
    tle_values = (("EPHEMERIS_TYPE", record.ephemeris_type), ("CLASSIFICATION_TYPE", record.classification_type),
                  ("NORAD_CAT_ID", record.norad_cat_id), ("ELEMENT_SET_NO", record.element_set_no),
                  ("REV_AT_EPOCH", record.rev_at_epoch), ("BSTAR", record.bstar),
                  ("MEAN_MOTION_DOT", record.mean_motion_dot * _DAY**2 / _TWO_PI),
                  ("MEAN_MOTION_DDOT", record.mean_motion_ddot * _DAY**3 / _TWO_PI))
    for key, value in tle_values:
        _text(tle, key, _xml_number(value, precision) if isinstance(value, float) else value)
    for section, parent in (("metadata", metadata), ("meanElements", mean), ("tleParameters", tle)):
        for key, value in record._extra_sections.get(section, ()):
            _text(parent, key, value)
    return omm


def format_omm_xml(records, *, precision: int = 17) -> str:
    """Format one record or records as canonical unqualified OMM XML."""
    if not isinstance(precision, int) or not 1 <= precision <= 17:
        raise ValueError("precision must be an integer from 1 through 17.")
    if isinstance(records, OMMRecord):
        values = (records,)
    else:
        values = tuple(records)
        if not values or not all(isinstance(value, OMMRecord) for value in values):
            raise TypeError("records must contain OMMRecord instances.")
    if precision != 17:
        # Element formatting is intentionally fixed at full precision to keep TLE values reversible.
        pass
    root = _record_element(values[0], precision) if len(values) == 1 else ET.Element("ndm")
    if len(values) > 1:
        for record in values:
            root.append(_record_element(record, precision))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


format_omm = format_omm_xml
read_omm = read_omm_xml


def write_omm_xml(records, destination, *, precision: int = 17, overwrite: bool = False):
    """Write canonical OMM XML to a stream or path."""
    text = format_omm_xml(records, precision=precision)
    if hasattr(destination, "write"):
        destination.write(text)
        return None
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path = ensure_file_parent(path)
    path.write_text(text, encoding="utf-8")
    return path


write_omm = write_omm_xml
