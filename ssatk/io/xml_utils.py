#!/usr/bin/env python3
# xml_readers.py
#
# Stand-alone XML loaders:
#   - read_xml(filename, keep_root=False, decode_special=True)
#   - load_xml(filename, keep_root=False, decode_special=True)  # alias of read_xml
#
# Behavior
# - Parses ANY XML file into Python dict/list/scalars.
# - Attributes -> stored under "@attrs"; element text -> under "#text".
# - Repeated child tags become Python lists.
# - If decode_special=True, restores data encoded with markers:
#     * @attrs.type="ndarray"  with dtype + shape + <item> -> numpy.ndarray
#     * @attrs.type="datetime" with #text ISO string       -> datetime.datetime
#     * @attrs.type="astropy_time" with 'scale' + #text    -> astropy.time.Time (if available)
#     * @attrs.type="set"/"tuple" with <item> children     -> set / tuple
#
# Notes
# - Uses numpy (no math, no typing).
# - No CLI / __main__ block; import and call read_xml/load_xml directly.

import numpy as np
from datetime import datetime
from ssatk._paths import ensure_file_parent
try:
    from astropy.time import Time  # optional
except ImportError:
    Time = None  # gracefully degrade if astropy isn't installed
import xml.etree.ElementTree as ET


def _element_to_struct(element):
    """
    Convert an Element into a nested, JSON-like structure:
      - attributes under "@attrs"
      - text under "#text"
      - children grouped by tag; repeated tags become lists
      - pure-text nodes become just the text scalar
    """
    node = {}
    if element.attrib:
        node["@attrs"] = dict(element.attrib)

    # Group children by tag to detect repeats
    children_by_tag = {}
    for child in element:
        child_struct = _element_to_struct(child)
        tag = child.tag
        children_by_tag.setdefault(tag, []).append(child_struct)

    # Attach grouped children (singletons vs lists)
    for tag, items in children_by_tag.items():
        node[tag] = items if len(items) > 1 else items[0]

    # Attach text content
    text = (element.text or "").strip()
    if text:
        if node:  # already has attrs or children
            node["#text"] = text
        else:
            return text  # pure text node becomes scalar

    return node


def _decode_special_struct(struct):
    """
    Decode special types when marked via @attrs['type'] on a dict payload.
    Returns either a decoded Python object or the original struct.
    """
    if not isinstance(struct, dict):
        return struct

    attrs = struct.get("@attrs")
    if not isinstance(attrs, dict):
        return struct

    encoded_type = attrs.get("type")
    if not encoded_type:
        return struct

    if encoded_type == "ndarray":
        dtype = attrs.get("dtype") or "float64"
        shape_txt = attrs.get("shape") or ""
        shape = tuple(int(s) for s in shape_txt.split(",")) if shape_txt else None
        items = struct.get("item", [])
        arr = np.array(items, dtype=dtype)
        if shape:
            try:
                arr = arr.reshape(shape)
            except ValueError:
                # If reshape fails, return flat array
                pass
        return arr

    if encoded_type == "datetime":
        iso = struct.get("#text", "")
        try:
            return datetime.fromisoformat(iso)
        except ValueError:
            return iso  # leave as string if parse fails

    if encoded_type == "astropy_time":
        isot = struct.get("#text", "")
        scale = attrs.get("scale", "utc")
        if Time is not None:
            try:
                return Time(isot, scale=scale)
            except ValueError:
                return isot
        return isot  # astropy not available: return text

    if encoded_type == "set":
        return set(struct.get("item", []))

    if encoded_type == "tuple":
        return tuple(struct.get("item", []))

    return struct


def _struct_to_python(obj, decode_special):
    """Recursively convert the XML-structure to plain Python, applying special decoding."""
    if isinstance(obj, list):
        return [_struct_to_python(v, decode_special) for v in obj]

    if isinstance(obj, dict):
        # Recurse into children first
        recursed = {k: _struct_to_python(v, decode_special) for k, v in obj.items()}
        # Then decode marked special payloads, if requested
        return _decode_special_struct(recursed) if decode_special else recursed

    return obj  # scalars pass through unchanged


def read_xml(filename, keep_root=False, decode_special=True):
    """
    Parse an XML file into Python data.

    Parameters
    ----------
    filename : path-like or str
        Path to an XML file.
    keep_root : bool, optional
        If false (default), return the root content; if true, return
        ``{root_tag: content}``.
    decode_special : bool, optional
        If true (default), restore marked ndarray, datetime, Time, set, and
        tuple values.

    Returns
    -------
    object
        A nested Python representation of the XML content.
    """
    tree = ET.parse(str(filename))
    root = tree.getroot()
    struct = _element_to_struct(root)
    data = _struct_to_python(struct, decode_special=decode_special)
    return {root.tag: data} if keep_root else data


def load_xml(filename, keep_root=False, decode_special=True):
    """
    Alias of read_xml(), provided with the requested name.
    """
    return read_xml(filename, keep_root=keep_root, decode_special=decode_special)


def save_xml(filename, data, root_tag="root", pretty=True, xml_declaration=True, encoding="utf-8"):
    """
    Serialize Python data to XML and write to 'filename'.

    - Dict/List/Scalar supported.
    - Special encodings:
        * numpy.ndarray -> @attrs: type="ndarray", dtype, shape + <item> values
        * numpy scalars  -> converted to native Python numbers
        * datetime       -> @attrs: type="datetime", #text=ISO8601
        * astropy Time   -> @attrs: type="astropy_time", scale + #text=ISOT
        * set/tuple      -> @attrs: type="set"/"tuple" + <item> children
    """

    def _indent_in_place(element, level=0):
        indent_space = "  "
        i = "\n" + level * indent_space
        if len(element):
            if not element.text or not element.text.strip():
                element.text = i + indent_space
            for child in element:
                _indent_in_place(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i

    def _serialize_special(obj):
        if isinstance(obj, np.ndarray):
            flat = obj.ravel().tolist()
            return {
                "@attrs": {"type": "ndarray", "dtype": str(obj.dtype), "shape": ",".join(str(x) for x in obj.shape)},
                "item": [x.item() if isinstance(x, np.generic) else x for x in flat],
            }
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, datetime):
            return {"@attrs": {"type": "datetime"}, "#text": obj.isoformat()}
        if Time is not None and isinstance(obj, Time):
            return {"@attrs": {"type": "astropy_time", "scale": obj.scale}, "#text": obj.isot}
        if isinstance(obj, set):
            return {"@attrs": {"type": "set"}, "item": list(obj)}
        if isinstance(obj, tuple):
            return {"@attrs": {"type": "tuple"}, "item": list(obj)}
        return None

    def _python_to_struct(obj):
        special = _serialize_special(obj)
        if special is not None:
            if isinstance(special, dict):
                out = {}
                for k, v in special.items():
                    if k in ("@attrs", "#text"):
                        out[k] = v
                    else:
                        out[k] = _python_to_struct(v)
                return out
            return special
        if isinstance(obj, dict):
            return {k: _python_to_struct(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_python_to_struct(v) for v in obj]
        if isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, bytes):
            return obj.decode("utf-8", "replace")
        return str(obj)

    def _struct_to_element(tag, payload):
        elem = ET.Element(tag)
        if not isinstance(payload, (dict, list)):
            elem.text = str(payload)
            return elem
        if isinstance(payload, list):
            for item in payload:
                elem.append(_struct_to_element("item", item))
            return elem
        attrs = payload.get("@attrs")
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                elem.set(k, str(v))
        text = payload.get("#text")
        if isinstance(text, (str, int, float, bool)):
            elem.text = str(text)
        for key, value in payload.items():
            if key in ("@attrs", "#text"):
                continue
            if isinstance(value, list):
                for item in value:
                    elem.append(_struct_to_element(key, item))
            else:
                elem.append(_struct_to_element(key, value))
        return elem

    structure = _python_to_struct(data)
    if isinstance(structure, dict) and len(structure) == 1 and next(iter(structure)) not in ("@attrs", "#text"):
        only_key = next(iter(structure))
        root_element = _struct_to_element(only_key, structure[only_key])
    else:
        root_element = _struct_to_element(root_tag, structure)

    if pretty:
        _indent_in_place(root_element)

    path = ensure_file_parent(filename)
    ET.ElementTree(root_element).write(path, encoding=encoding, xml_declaration=xml_declaration)
