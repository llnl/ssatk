"""Reproducible acquisition records for external terrain and lunar data.

Scientific terrain and topography are intentionally not embedded in the wheel.
This module downloads a user-selected public product atomically, verifies an
optional SHA-256 digest, and writes a sidecar provenance record.  Rendered
products can then cite the exact file, URL, checksum, access time, licence, and
source description instead of silently substituting decorative geometry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import os
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AcquisitionRecord:
    source_name: str
    url: str
    output_path: str
    sha256: str
    bytes: int
    accessed_utc: str
    citation: str = ""
    licence: str = ""
    metadata: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["$schema"] = "ssapy-toolkit.eclipse.data-acquisition/2.0"
        payload["metadata"] = dict(self.metadata or {})
        return payload


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_file(
    url: str,
    output: str | Path,
    *,
    source_name: str,
    expected_sha256: str | None = None,
    citation: str = "",
    licence: str = "",
    metadata: Mapping[str, object] | None = None,
    timeout_s: float = 120.0,
    overwrite: bool = False,
) -> AcquisitionRecord:
    """Download or copy one scientific data product and seal its provenance.

    ``file://`` URLs are supported so acquisition workflows can be tested
    without network access.  The final path is replaced only after the digest
    check succeeds.  A ``.provenance.json`` sidecar is written beside the data.
    """
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(str(url))
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        if parsed.scheme in {"", "file"}:
            source = Path(parsed.path if parsed.scheme == "file" else str(url)).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, temporary)
        elif parsed.scheme in {"http", "https"}:
            request = Request(str(url), headers={"User-Agent": "ssapy-eclipse-validated/2.2"})
            with urlopen(request, timeout=float(timeout_s)) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
        else:
            raise ValueError("data URL must be file, http, or https")
        digest = file_sha256(temporary)
        if expected_sha256 and digest.lower() != str(expected_sha256).lower():
            raise ValueError(
                f"SHA-256 mismatch for {source_name}: expected {expected_sha256}, received {digest}"
            )
        size = temporary.stat().st_size
        temporary.replace(destination)
        record = AcquisitionRecord(
            source_name=str(source_name), url=str(url), output_path=str(destination),
            sha256=digest, bytes=int(size),
            accessed_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            citation=str(citation), licence=str(licence), metadata=dict(metadata or {}),
        )
        sidecar = destination.with_name(destination.name + ".provenance.json")
        sidecar.write_text(json.dumps(record.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        return record
    finally:
        if temporary.exists():
            temporary.unlink()


def known_source_templates() -> dict[str, dict[str, str]]:
    """Return documented URL templates without downloading any dataset."""
    return {
        "aws-terrarium": {
            "url_template": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
            "source_name": "Terrain Tiles on AWS / Terrarium",
            "citation": "Terrain Tiles accessed from the Registry of Open Data on AWS",
            "licence": "Dataset-specific attribution; see Tilezen/joerd attribution documentation",
        },
        "nasa-srtmgl1-v003": {
            "url_template": "user-selected NASA Earthdata SRTMGL1 V003 product URL",
            "source_name": "NASA SRTMGL1 V003",
            "citation": "NASA Shuttle Radar Topography Mission Global 1 arc second V003",
            "licence": "NASA Earthdata data-use policy",
        },
        "nasa-lola-ldem": {
            "url_template": "user-selected NASA PDS/PGDA LOLA LDEM product URL",
            "source_name": "NASA LRO LOLA LDEM",
            "citation": "Lunar Reconnaissance Orbiter LOLA digital elevation model",
            "licence": "NASA/PDS or PGDA data-use policy for the selected product",
        },
    }
