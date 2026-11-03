"""Reproducible solar and geomagnetic inputs for atmosphere models."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.time import Time

from .data import data_path, resource_package

__all__ = [
    "SpaceWeatherRecord",
    "SpaceWeatherTable",
    "load_packaged_space_weather",
    "read_space_weather",
]


@dataclass(frozen=True)
class SpaceWeatherRecord:
    """One daily CelesTrak space-weather record."""

    date_utc: date
    gps_seconds: float
    f107_observed_sfu: float
    f107_adjusted_sfu: float
    f107_observed_81_sfu: float
    f107_adjusted_81_sfu: float
    ap_daily: float
    ap_3h: tuple[float, ...]
    data_type: str

    @property
    def predicted(self) -> bool:
        """Whether this record is forecast rather than observed/interpolated."""

        return self.data_type in {"PRD", "PRM"}


@dataclass(frozen=True)
class SpaceWeatherTable:
    """Daily space-weather records with explicit prediction policy."""

    records: tuple[SpaceWeatherRecord, ...]
    source: str = "unknown"

    def __post_init__(self):
        records = tuple(self.records)
        if not records:
            raise ValueError("Space-weather table must contain at least one record.")
        object.__setattr__(self, "records", records)
        gps = np.asarray([record.gps_seconds for record in records])
        if not np.all(np.diff(gps) > 0.0):
            raise ValueError("Space-weather records must be strictly chronological.")
        if any(right.date_utc <= left.date_utc for left, right in zip(records, records[1:])):
            raise ValueError("Space-weather record dates must be strictly chronological.")
        # Match EarthOrientationTable: keep the immutable index out of value equality.
        gps.setflags(write=False)
        object.__setattr__(self, "_source_gps_seconds", gps)
        end_gps = _gps_for_dates([records[-1].date_utc + timedelta(days=1)])[0]
        object.__setattr__(self, "_end_gps", float(end_gps))

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def start_gps(self) -> float:
        return self.records[0].gps_seconds

    @property
    def end_gps(self) -> float:
        return self._end_gps

    @property
    def coverage(self) -> dict:
        """Coverage in GPS seconds; the weather end is exclusive."""
        observed = max(
            (record for record in self.records if not record.predicted),
            key=lambda record: record.gps_seconds,
            default=None,
        )
        observed_end = None
        if observed is not None:
            observed_end = float(
                _gps_for_dates([observed.date_utc + timedelta(days=1)])[0]
            )
        return {
            "source": self.source,
            "start_gps": self.start_gps,
            "end_gps": self.end_gps,
            "observed_end_gps": observed_end,
        }

    def at(self, time, *, allow_predicted: bool = False) -> SpaceWeatherRecord:
        """Return the daily record containing a GPS-second or UTC time."""

        gps = _gps_seconds(time)
        if gps < self.start_gps or gps >= self.end_gps:
            raise ValueError(
                f"Space-weather time {gps} s is outside [{self.start_gps}, {self.end_gps}) s."
            )
        source_times = self._source_gps_seconds
        index = int(np.searchsorted(source_times, gps, side="right") - 1)
        record = self.records[index]
        if index + 1 < len(self.records):
            next_record = self.records[index + 1]
            if (next_record.date_utc - record.date_utc).days > 1:
                expected_next_day = _gps_for_dates([record.date_utc + timedelta(days=1)])[0]
                if gps >= expected_next_day:
                    raise ValueError(
                        f"Space-weather coverage has a missing day after {record.date_utc}."
                    )
        if record.predicted and not allow_predicted:
            raise ValueError(
                "Space-weather query intersects predicted records; pass "
                "allow_predicted=True to opt in explicitly."
            )
        return record

    def msis_inputs(self, time, *, allow_predicted: bool = False):
        """Return ``(F10.7, F10.7a, Ap[0:7])`` inputs for ``pymsis``.

        ``pymsis`` expects the prior day's observed daily F10.7 and the
        observed 81-day centered average, both at the actual Sun-Earth
        distance (not adjusted to 1 au), plus the current and preceding
        3-hour Ap intervals. Missing
        subdaily values fall back to the daily Ap value.
        """

        gps = _gps_seconds(time)
        current = self.at(gps, allow_predicted=allow_predicted)
        try:
            previous = self.at(gps - 86_400.0, allow_predicted=allow_predicted)
        except ValueError:
            # Prior day out of the table range (e.g. the table's first day):
            # fall back to the current day's F10.7 rather than fail, but this
            # is less accurate than the prior-day value pymsis expects.
            warnings.warn(
                "SpaceWeatherTable.msis_inputs: no record for the prior day; "
                "using the current day's F10.7 as a fallback.",
                stacklevel=2,
            )
            previous = current

        ap_values = []
        for offset in range(20):
            sample_gps = gps - 10_800.0 * offset
            record = self.at(sample_gps, allow_predicted=allow_predicted)
            slot = _utc_slot(sample_gps)
            ap_values.append(record.ap_3h[slot] if record.ap_3h[slot] >= 0.0 else record.ap_daily)

        ap = np.asarray(
            [
                current.ap_daily,
                ap_values[0],
                ap_values[1],
                ap_values[2],
                ap_values[3],
                np.mean(ap_values[4:12]),
                np.mean(ap_values[12:20]),
            ],
            dtype=float,
        )
        return (
            float(previous.f107_observed_sfu),
            float(current.f107_observed_81_sfu),
            ap,
        )

    __call__ = at


def read_space_weather(path: str | Path) -> SpaceWeatherTable:
    """Read a CelesTrak ``SW-All.csv`` file."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        table = _parse_rows(csv.DictReader(handle))
    return SpaceWeatherTable(table.records, source=str(Path(path)))


@lru_cache(maxsize=1)
def load_packaged_space_weather() -> SpaceWeatherTable:
    """Load the frozen CelesTrak space-weather snapshot from ``ssa-data-core``."""

    with data_path("environment/space_weather/SW-All.csv") as path:
        table = read_space_weather(path)
    return SpaceWeatherTable(
        table.records,
        source=resource_package("environment/space_weather/SW-All.csv") + ":environment/space_weather/SW-All.csv",
    )


def _parse_rows(rows) -> SpaceWeatherTable:
    parsed = []
    for row in rows:
        try:
            day = date.fromisoformat(row["DATE"])
            values = tuple(float(row[f"AP{i}"]) for i in range(1, 9))
            f107_observed = float(row["F10.7_OBS"])
            f107_adjusted = float(row["F10.7_ADJ"])
            f107_observed_81 = float(row["F10.7_OBS_CENTER81"])
            f107_adjusted_81 = float(row["F10.7_ADJ_CENTER81"])
            ap_daily = float(row["AP_AVG"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite((
            f107_observed,
            f107_adjusted,
            f107_observed_81,
            f107_adjusted_81,
            ap_daily,
            *values,
        ))):
            continue
        parsed.append(
            (
                day,
                f107_observed,
                f107_adjusted,
                f107_observed_81,
                f107_adjusted_81,
                ap_daily,
                values,
                (row.get("F10.7_DATA_TYPE") or "OBS").strip().upper(),
            )
        )
    if not parsed:
        return SpaceWeatherTable(())
    gps = _gps_for_dates([item[0] for item in parsed])
    records = [
        SpaceWeatherRecord(
            date_utc=item[0],
            gps_seconds=float(gps[index]),
            f107_observed_sfu=item[1],
            f107_adjusted_sfu=item[2],
            f107_observed_81_sfu=item[3],
            f107_adjusted_81_sfu=item[4],
            ap_daily=item[5],
            ap_3h=item[6],
            data_type=item[7],
        )
        for index, item in enumerate(parsed)
    ]
    return SpaceWeatherTable(tuple(records))


def _gps_seconds(time) -> float:
    if isinstance(time, Time):
        return float(time.gps)
    if isinstance(time, (str, bytes)):
        return float(Time(time, scale="utc").gps)
    try:
        return float(time)
    except (TypeError, ValueError):
        return float(Time(time, scale="utc").gps)


def _gps_for_dates(days: list[date]) -> np.ndarray:
    """Convert daily UTC dates without pre-1960 ERFA dubious-year warnings."""

    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    gps = np.empty(len(days), dtype=float)
    late = []
    for index, day in enumerate(days):
        moment = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        if day.year < 1960:
            gps[index] = (moment - gps_epoch).total_seconds()
        else:
            late.append((index, day.isoformat()))
    if late:
        values = np.asarray(Time([item[1] for item in late], scale="utc").gps)
        for (index, _), value in zip(late, values):
            gps[index] = value
    return gps


def _utc_slot(gps: float) -> int:
    moment = Time(gps, format="gps", scale="utc").to_datetime(timezone=timezone.utc)
    return min(7, (moment.hour * 60 + moment.minute) // 180)
