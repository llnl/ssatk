"""NASA/GSFC eclipse regression corpus for V19.3.

The corpus is intentionally data-only.  It preserves the terminology and
published quantities from the NASA/GSFC Five Millennium Catalogs while
providing deterministic conversion from the catalog's dynamical greatest
instant to an approximate UTC/UT instant by subtracting the published
``Delta T`` value.

It is used for independent regression of the discovery layer.  It is not used
to generate the nominal geometry of a discovered event and therefore cannot
silently pull the solver toward the reference values it is meant to test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping
import json

from ssapy_toolkit.compute.eclipse_reference_events import datetime_to_jd

SOLAR_CATALOG_URL = "https://eclipse.gsfc.nasa.gov/SEcat5/SE2001-2100.html"
LUNAR_CATALOG_URL = "https://eclipse.gsfc.nasa.gov/LEcat5/LE2001-2100.html"


@dataclass(frozen=True)
class ReferenceEclipseRecord:
    """One published eclipse-catalog row normalized for regression."""

    key: str
    mode: str
    eclipse_type: str
    date: str
    greatest_td: str
    delta_t_s: float
    magnitude: float
    catalog_code: str
    source_url: str
    penumbral_magnitude: float | None = None
    umbral_magnitude: float | None = None
    penumbral_duration_min: float | None = None
    partial_duration_min: float | None = None
    total_duration_min: float | None = None
    path_width_km: float | None = None
    central_duration_s: float | None = None
    greatest_lat_deg: float | None = None
    greatest_lon_east_deg: float | None = None
    notes: str = ""

    @property
    def greatest_td_datetime(self) -> datetime:
        return datetime.fromisoformat(f"{self.date}T{self.greatest_td}+00:00")

    @property
    def greatest_utc_datetime(self) -> datetime:
        # NASA's catalog publishes the dynamical instant and Delta T.
        # UT1 ~= TD - DeltaT; UTC differs from UT1 by less than one second
        # over this corpus, so this is the correct catalog-level comparison
        # target without pretending that the row contains a leap-second table.
        return self.greatest_td_datetime - timedelta(seconds=float(self.delta_t_s))

    @property
    def greatest_jd_utc(self) -> float:
        return datetime_to_jd(self.greatest_utc_datetime)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["greatest_utc_approx"] = self.greatest_utc_datetime.isoformat().replace("+00:00", "Z")
        payload["greatest_jd_utc_approx"] = float(self.greatest_jd_utc)
        payload["time_conversion"] = "catalog TD minus published Delta T; UTC/UT1 distinction retained as sub-second qualification"
        return payload


def _solar(date: str, td: str, dt: float, code: str, magnitude: float,
           width: float | None = None, duration: float | None = None,
           lat: float | None = None, lon: float | None = None) -> ReferenceEclipseRecord:
    type_map = {"P": "partial", "A": "annular", "T": "total", "H": "hybrid"}
    key = f"solar_{date.replace('-', '_')}"
    return ReferenceEclipseRecord(
        key=key, mode="solar", eclipse_type=type_map[code], date=date,
        greatest_td=td, delta_t_s=dt, magnitude=magnitude, catalog_code=code,
        source_url=SOLAR_CATALOG_URL, path_width_km=width,
        central_duration_s=duration, greatest_lat_deg=lat,
        greatest_lon_east_deg=lon,
    )


def _lunar(date: str, td: str, dt: float, code: str, pen_mag: float,
           umb_mag: float, pen_dur: float, partial_dur: float | None = None,
           total_dur: float | None = None) -> ReferenceEclipseRecord:
    type_map = {"N": "penumbral", "Ne": "penumbral", "P": "partial-umbral", "T": "total"}
    key = f"lunar_{date.replace('-', '_')}"
    return ReferenceEclipseRecord(
        key=key, mode="lunar", eclipse_type=type_map[code], date=date,
        greatest_td=td, delta_t_s=dt, magnitude=umb_mag,
        catalog_code=code, source_url=LUNAR_CATALOG_URL,
        penumbral_magnitude=pen_mag, umbral_magnitude=umb_mag,
        penumbral_duration_min=pen_dur, partial_duration_min=partial_dur,
        total_duration_min=total_dur,
        notes="Ne is retained as a near-edge penumbral catalog subtype but normalized to penumbral for solver classification.",
    )


# NASA/GSFC Five Millennium Catalog rows, 2021-2030.  Central solar widths
# and durations are omitted for partial eclipses, matching the catalog.
SOLAR_REFERENCE_2021_2030: tuple[ReferenceEclipseRecord, ...] = (
    _solar("2021-06-10", "10:43:07", 72, "A", 0.9435, 527, 231),
    _solar("2021-12-04", "07:34:38", 73, "T", 1.0367, 419, 114),
    _solar("2022-04-30", "20:42:36", 73, "P", 0.6396),
    _solar("2022-10-25", "11:01:20", 73, "P", 0.8619),
    _solar("2023-04-20", "04:17:56", 73, "H", 1.0132, 49, 76),
    _solar("2023-10-14", "18:00:41", 74, "A", 0.9520, 187, 317),
    _solar("2024-04-08", "18:18:29", 74, "T", 1.0566, 198, 268),
    _solar("2024-10-02", "18:46:13", 74, "A", 0.9326, 266, 445),
    _solar("2025-03-29", "10:48:36", 75, "P", 0.9376),
    _solar("2025-09-21", "19:43:04", 75, "P", 0.8550),
    _solar("2026-02-17", "12:13:06", 75, "A", 0.9630, 616, 140),
    _solar("2026-08-12", "17:47:06", 75, "T", 1.0386, 294, 138),
    _solar("2027-02-06", "16:00:48", 76, "A", 0.9281, 282, 471),
    _solar("2027-08-02", "10:07:50", 76, "T", 1.0790, 258, 383),
    _solar("2028-01-26", "15:08:59", 76, "A", 0.9208, 323, 627),
    _solar("2028-07-22", "02:56:40", 77, "T", 1.0560, 230, 310),
    _solar("2029-01-14", "17:13:48", 77, "P", 0.8714),
    _solar("2029-06-12", "04:06:13", 77, "P", 0.4576),
    _solar("2029-07-11", "15:37:19", 77, "P", 0.2303),
    _solar("2029-12-05", "15:03:58", 77, "P", 0.8911),
    _solar("2030-06-01", "06:29:13", 78, "A", 0.9443, 250, 321),
    _solar("2030-11-25", "06:51:37", 78, "T", 1.0468, 169, 224),
)

LUNAR_REFERENCE_2021_2030: tuple[ReferenceEclipseRecord, ...] = (
    _lunar("2021-05-26", "11:19:53", 72, "T", 1.9540, 1.0095, 302.0, 187.4, 14.5),
    _lunar("2021-11-19", "09:04:06", 73, "P", 2.0720, 0.9742, 361.5, 208.4),
    _lunar("2022-05-16", "04:12:42", 73, "T", 2.3726, 1.4137, 318.7, 207.2, 84.9),
    _lunar("2022-11-08", "11:00:22", 73, "T", 2.4143, 1.3589, 353.9, 219.8, 85.0),
    _lunar("2023-05-05", "17:24:05", 73, "N", 0.9636, -0.0457, 257.5),
    _lunar("2023-10-28", "20:15:18", 74, "P", 1.1181, 0.1220, 264.6, 77.4),
    _lunar("2024-03-25", "07:13:59", 74, "N", 0.9557, -0.1325, 279.1),
    _lunar("2024-09-18", "02:45:25", 74, "P", 1.0372, 0.0848, 246.3, 62.8),
    _lunar("2025-03-14", "06:59:56", 75, "T", 2.2595, 1.1784, 362.6, 218.3, 65.4),
    _lunar("2025-09-07", "18:12:58", 75, "T", 2.3440, 1.3619, 326.7, 209.4, 82.1),
    _lunar("2026-03-03", "11:34:52", 75, "T", 2.1838, 1.1507, 338.6, 207.2, 58.3),
    _lunar("2026-08-28", "04:14:04", 75, "P", 1.9645, 0.9299, 337.8, 198.1),
    _lunar("2027-02-20", "23:14:06", 76, "N", 0.9266, -0.0569, 241.0),
    _lunar("2027-07-18", "16:04:09", 76, "Ne", 0.0014, -1.0680, 11.8),
    _lunar("2027-08-17", "07:14:59", 76, "N", 0.5456, -0.5254, 218.6),
    _lunar("2028-01-12", "04:14:13", 76, "P", 1.0468, 0.0662, 250.7, 56.0),
    _lunar("2028-07-06", "18:20:57", 77, "P", 1.4266, 0.3892, 310.6, 141.5),
    _lunar("2028-12-31", "16:53:15", 77, "T", 2.2742, 1.2463, 336.2, 208.8, 71.3),
    _lunar("2029-06-26", "03:23:22", 77, "T", 2.8266, 1.8436, 335.1, 219.5, 101.9),
    _lunar("2029-12-20", "22:43:12", 78, "T", 2.2008, 1.1174, 358.0, 213.3, 53.7),
    _lunar("2030-06-15", "18:34:34", 78, "P", 1.4480, 0.5025, 278.2, 144.4),
    _lunar("2030-12-09", "22:28:51", 78, "N", 0.9416, -0.1628, 279.2),
)

REFERENCE_CORPUS_2021_2030 = SOLAR_REFERENCE_2021_2030 + LUNAR_REFERENCE_2021_2030


def records(mode: str = "all") -> tuple[ReferenceEclipseRecord, ...]:
    key = str(mode).lower()
    if key == "all":
        return REFERENCE_CORPUS_2021_2030
    if key == "solar":
        return SOLAR_REFERENCE_2021_2030
    if key == "lunar":
        return LUNAR_REFERENCE_2021_2030
    raise ValueError("mode must be 'solar', 'lunar', or 'all'")


def export_reference_corpus(path: str | Path, mode: str = "all") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = records(mode)
    payload = {
        "$schema": "ssapy-toolkit.eclipse.reference-corpus/1.9.3",
        "title": "NASA/GSFC eclipse regression corpus, 2021-2030",
        "scope": {"start": "2021-01-01", "end": "2030-12-31", "mode": mode},
        "sources": {"solar": SOLAR_CATALOG_URL, "lunar": LUNAR_CATALOG_URL},
        "record_count": len(selected),
        "records": [item.to_dict() for item in selected],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def index_by_key(mode: str = "all") -> dict[str, ReferenceEclipseRecord]:
    return {item.key: item for item in records(mode)}


__all__ = [
    "ReferenceEclipseRecord", "SOLAR_REFERENCE_2021_2030",
    "LUNAR_REFERENCE_2021_2030", "REFERENCE_CORPUS_2021_2030",
    "SOLAR_CATALOG_URL", "LUNAR_CATALOG_URL", "records",
    "index_by_key", "export_reference_corpus",
]
