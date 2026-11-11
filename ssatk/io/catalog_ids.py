"""Catalog identifiers used in TLE and OMM products."""

from numbers import Integral

_ALPHA5_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def decode_tle_catalog_id(value: str) -> int:
    """Decode a five-character numeric or Alpha-5 TLE catalog field."""
    token = str(value).strip().upper()
    if token.isdigit() and len(token) <= 5:
        return int(token)
    if (
        len(token) == 5
        and token[0] in _ALPHA5_ALPHABET
        and token[1:].isdigit()
    ):
        prefix = _ALPHA5_ALPHABET.index(token[0]) + 10
        return prefix * 10_000 + int(token[1:])
    raise ValueError(f"invalid TLE catalog ID {value!r}")


def encode_tle_catalog_id(norad_cat_id: int) -> str:
    """Encode a numeric catalog ID in the five-character TLE field."""
    if isinstance(norad_cat_id, bool) or not isinstance(norad_cat_id, Integral):
        raise TypeError("NORAD catalog ID must be an integer.")
    norad_cat_id = int(norad_cat_id)
    if 0 <= norad_cat_id <= 99_999:
        return f"{norad_cat_id:05d}"
    prefix, suffix = divmod(norad_cat_id, 10_000)
    index = prefix - 10
    if not 0 <= index < len(_ALPHA5_ALPHABET) or norad_cat_id > 339_999:
        raise ValueError("TLE Alpha-5 can encode catalog IDs only through 339999.")
    return f"{_ALPHA5_ALPHABET[index]}{suffix:04d}"
