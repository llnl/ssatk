"""Fetch optional demo datasets into the local SSATK output cache."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ssatk._paths import DEFAULT_OUTPUT_DIR_NAME

from .datapath import datapath

_USER_AGENT = "ssatk-demo-data/1.0"
_CELESTRAK_ACTIVE_3LE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=3le"
_CELESTRAK_STATIONS_3LE_URL = "https://celestrak.org/NORAD/elements/stations.txt"
_CELESTRAK_ACTIVE_XML_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=xml"


class DemoDataUnavailableWarning(UserWarning):
    """Warning emitted when optional demo data cannot be found or fetched."""


def ensure_demo_data_file(
    filename,
    *,
    allow_download=True,
    timeout=30,
    warn=True,
    local_dirs=None,
):
    """Return a local optional-demo data file, fetching it when available.

    Lookup order: ``datapath(filename)`` (normally ``~/ssatk_output`` or
    ``SSATK_OUTPUT_DIR``), nearby ``ssatk_output`` folders including the
    current directory and its parent, then a known public source when
    ``allow_download`` is true.

    Missing data or download failures return ``None`` and emit a warning by
    default so demos can skip gracefully when offline.
    """
    target = Path(datapath(filename)).expanduser()
    if target.exists():
        return target

    for candidate in _local_data_candidates(filename, local_dirs=local_dirs):
        if candidate.exists():
            return candidate

    if not allow_download:
        _warn_missing(filename, target, "auto-download disabled", warn=warn)
        return None

    fetcher = _DEMO_DATA_FETCHERS.get(Path(filename).name)
    if fetcher is None:
        _warn_missing(filename, target, "no fetcher is configured", warn=warn)
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetcher(target, timeout=timeout)
    except (OSError, HTTPError, URLError, TimeoutError, ValueError) as exc:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        _warn_missing(filename, target, f"download failed: {exc}", warn=warn)
        return None

    if not target.exists() or target.stat().st_size == 0:
        _warn_missing(filename, target, "download produced no file", warn=warn)
        return None
    return target


def _local_data_candidates(filename, *, local_dirs=None):
    relative = Path(filename)
    bases = []
    if local_dirs is not None:
        bases.extend(Path(path).expanduser() for path in local_dirs)

    cwd = Path.cwd()
    bases.extend(
        [
            cwd / DEFAULT_OUTPUT_DIR_NAME,
            cwd.parent / DEFAULT_OUTPUT_DIR_NAME,
        ]
    )

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        bases.append(parent / DEFAULT_OUTPUT_DIR_NAME)
        if parent.name == "workdir":
            break

    seen = set()
    for base in bases:
        candidate = (base / relative).expanduser()
        key = candidate.resolve(strict=False)
        if key == Path(datapath(filename)).expanduser().resolve(strict=False):
            continue
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _warn_missing(filename, target, reason, *, warn=True):
    message = (
        f"Optional SSATK demo data file {filename!r} was not found at {target} "
        f"and could not be fetched ({reason}). The demo will be skipped."
    )
    if warn:
        warnings.warn(message, DemoDataUnavailableWarning, stacklevel=3)


def _download_text(url, *, timeout):
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _atomic_write_text(target, text):
    target = Path(target)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(target.parent)) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, target)


def _fetch_full_catalog_3le(target, *, timeout):
    errors = []
    for url in (_CELESTRAK_ACTIVE_3LE_URL, _CELESTRAK_STATIONS_3LE_URL):
        try:
            text = _download_text(url, timeout=timeout)
            if "\n1 " not in text or "\n2 " not in text:
                raise ValueError(f"response from {url} did not look like 3LE/TLE text")
            _atomic_write_text(target, text)
            return
        except (OSError, HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise URLError("; ".join(errors))


def _fetch_full_catalog_xml(target, *, timeout):
    text = _download_text(_CELESTRAK_ACTIVE_XML_URL, timeout=timeout)
    if "<omm" not in text and "<ndm" not in text:
        raise ValueError("response did not look like OMM XML")
    _atomic_write_text(target, text)


def _fetch_artemis2_orion_state_vectors(target, *, timeout):
    params = {
        "format": "text",
        "COMMAND": "-1024",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "500@399",
        "START_TIME": "'2026-Apr-02 02:00'",
        "STOP_TIME": "'2026-Apr-10 23:00'",
        "STEP_SIZE": "'1 h'",
        "VEC_TABLE": "2",
        "CSV_FORMAT": "YES",
        "REF_PLANE": "FRAME",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urlencode(params)
    text = _download_text(url, timeout=timeout)
    rows = _horizons_vector_rows(text)
    if not rows:
        raise ValueError("JPL Horizons response did not contain vector rows")

    header = "\n".join(
        [
            "# Artemis II (Orion, spacecraft ID -1024) State Vectors",
            "# Source: JPL Horizons API (https://ssd.jpl.nasa.gov/horizons/)",
            "# Reference frame: ICRF | Center body: Earth (geocentric)",
            "# Units: km (position), km/s (velocity) | Time: TDB",
            "# Coverage: 2026-Apr-02 02:00 to 2026-Apr-10 23:00 UTC | Step: 1 hour",
            "# Mission: Artemis II crewed lunar flyby, launched 2026-Apr-01 22:35 UTC",
            "#",
            "JDTDB,Calendar_Date_TDB,X_km,Y_km,Z_km,VX_km_s,VY_km_s,VZ_km_s",
        ]
    )
    _atomic_write_text(target, header + "\n" + "\n".join(rows) + "\n")


def _horizons_vector_rows(text):
    start = text.find("$$SOE")
    stop = text.find("$$EOE")
    if start < 0 or stop < 0 or stop <= start:
        return []

    rows = []
    for raw_line in text[start + len("$$SOE"): stop].splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        rows.append(",".join(parts[:8]))
    return rows


_DEMO_DATA_FETCHERS = {
    "full_catalog_3le.txt": _fetch_full_catalog_3le,
    "full_catalog.xml": _fetch_full_catalog_xml,
    "artemis2_orion_state_vectors.csv": _fetch_artemis2_orion_state_vectors,
}


__all__ = ["DemoDataUnavailableWarning", "ensure_demo_data_file"]
