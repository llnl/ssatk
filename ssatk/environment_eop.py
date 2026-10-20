"""Reproducible Earth orientation data for frame and time transformations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.time import Time

from .data import data_path, resource_package

__all__ = [
    "EarthOrientationRecord",
    "EarthOrientationTable",
    "load_packaged_eop",
    "read_eop",
]


@dataclass(frozen=True)
class EarthOrientationRecord:
    """One interpolated or source EOP record in explicit SI-compatible units."""

    mjd_utc: float
    gps_seconds: float
    polar_motion_x_arcsec: float
    polar_motion_y_arcsec: float
    ut1_minus_utc_s: float
    polar_motion_flag: str
    ut1_flag: str
    nutation_flag: str
    polar_motion_x_uncertainty_arcsec: float | None = None
    polar_motion_y_uncertainty_arcsec: float | None = None
    ut1_minus_utc_uncertainty_s: float | None = None
    dx_mas: float | None = None
    dy_mas: float | None = None
    interpolated: bool = False

    @property
    def predicted(self) -> bool:
        """Whether any selected EOP component is predicted rather than observed."""

        return "P" in {
            self.polar_motion_flag,
            self.ut1_flag,
            self.nutation_flag,
        }


@dataclass(frozen=True)
class EarthOrientationTable:
    """Daily EOP records with explicit interpolation and prediction policy."""

    records: tuple[EarthOrientationRecord, ...]
    source: str = "unknown"

    def __post_init__(self):
        if not self.records:
            raise ValueError("Earth orientation table must contain at least one record.")
        gps = np.asarray([record.gps_seconds for record in self.records], dtype=float)
        if not np.all(np.diff(gps) > 0.0):
            raise ValueError("Earth orientation records must be strictly chronological.")

    @property
    def start_gps(self) -> float:
        return self.records[0].gps_seconds

    @property
    def end_gps(self) -> float:
        return self.records[-1].gps_seconds

    def at(self, time, *, allow_predicted: bool = False) -> EarthOrientationRecord:
        """Interpolate EOP at a GPS-second, ``astropy`` time, or UTC string.

        Predicted source records are rejected by default. This keeps a caller
        from silently changing a reproducible force/frame model when a mission
        crosses the observed-to-predicted boundary.
        """

        gps = _gps_seconds(time)
        if gps < self.start_gps or gps > self.end_gps:
            raise ValueError(
                f"EOP time {gps} s is outside [{self.start_gps}, {self.end_gps}] s."
            )

        source_times = np.asarray([record.gps_seconds for record in self.records])
        right = int(np.searchsorted(source_times, gps, side="left"))
        if right == 0:
            left_record = right_record = self.records[0]
        elif right == len(self.records):
            left_record = right_record = self.records[-1]
        elif np.isclose(source_times[right], gps, rtol=0.0, atol=1e-6):
            left_record = right_record = self.records[right]
        else:
            left_record = self.records[right - 1]
            right_record = self.records[right]

        if not allow_predicted and (left_record.predicted or right_record.predicted):
            raise ValueError(
                "EOP query intersects predicted records; pass allow_predicted=True "
                "to opt in explicitly."
            )
        if left_record is right_record:
            return left_record

        fraction = (gps - left_record.gps_seconds) / (
            right_record.gps_seconds - left_record.gps_seconds
        )
        # UT1 is continuous but UT1 - UTC steps by an integer number of seconds
        # at a leap second, which falls at the end of the left UTC day (just
        # before the right node). Interpolating across that step spread a 1 s
        # jump over the whole preceding day (999 ms error at 23:59 UTC on
        # 2016-12-31). Remove the step from the right node; every query inside
        # the interval is still on the left day's side of the leap second.
        right_ut1_minus_utc = right_record.ut1_minus_utc_s
        step_s = right_ut1_minus_utc - left_record.ut1_minus_utc_s
        if abs(step_s) > 0.5:
            right_ut1_minus_utc -= round(step_s)
        return EarthOrientationRecord(
            mjd_utc=float(_lerp(left_record.mjd_utc, right_record.mjd_utc, fraction)),
            gps_seconds=gps,
            polar_motion_x_arcsec=float(
                _lerp(left_record.polar_motion_x_arcsec, right_record.polar_motion_x_arcsec, fraction)
            ),
            polar_motion_y_arcsec=float(
                _lerp(left_record.polar_motion_y_arcsec, right_record.polar_motion_y_arcsec, fraction)
            ),
            ut1_minus_utc_s=float(
                _lerp(left_record.ut1_minus_utc_s, right_ut1_minus_utc, fraction)
            ),
            polar_motion_flag=_combined_flag(
                left_record.polar_motion_flag, right_record.polar_motion_flag
            ),
            ut1_flag=_combined_flag(left_record.ut1_flag, right_record.ut1_flag),
            nutation_flag=_combined_flag(left_record.nutation_flag, right_record.nutation_flag),
            polar_motion_x_uncertainty_arcsec=_optional_lerp(
                left_record.polar_motion_x_uncertainty_arcsec,
                right_record.polar_motion_x_uncertainty_arcsec,
                fraction,
            ),
            polar_motion_y_uncertainty_arcsec=_optional_lerp(
                left_record.polar_motion_y_uncertainty_arcsec,
                right_record.polar_motion_y_uncertainty_arcsec,
                fraction,
            ),
            ut1_minus_utc_uncertainty_s=_optional_lerp(
                left_record.ut1_minus_utc_uncertainty_s,
                right_record.ut1_minus_utc_uncertainty_s,
                fraction,
            ),
            dx_mas=_optional_lerp(left_record.dx_mas, right_record.dx_mas, fraction),
            dy_mas=_optional_lerp(left_record.dy_mas, right_record.dy_mas, fraction),
            interpolated=True,
        )

    __call__ = at


def read_eop(path: str | Path) -> EarthOrientationTable:
    """Read an IERS ``finals2000A.all`` file from a filesystem path."""

    with Path(path).open("r", encoding="ascii") as handle:
        return _parse_eop_lines(handle)


@lru_cache(maxsize=1)
def load_packaged_eop() -> EarthOrientationTable:
    """Load the frozen IERS EOP snapshot from ``ssa-data-core``."""

    with data_path("environment/eop/finals2000A.all") as path:
        table = read_eop(path)
    return EarthOrientationTable(table.records, source=resource_package("environment/eop/finals2000A.all") + ":environment/eop/finals2000A.all")


def _parse_eop_lines(lines) -> EarthOrientationTable:
    records = []
    for line in lines:
        if len(line.rstrip("\n")) < 15:
            continue
        mjd = _float_field(line, 8, 15)
        x = _float_field(line, 19, 27)
        y = _float_field(line, 38, 46)
        ut1 = _float_field(line, 59, 68)
        if mjd is None or x is None or y is None or ut1 is None:
            continue
        records.append(
            EarthOrientationRecord(
                mjd_utc=mjd,
                gps_seconds=float(Time(mjd, format="mjd", scale="utc").gps),
                polar_motion_x_arcsec=x,
                polar_motion_y_arcsec=y,
                ut1_minus_utc_s=ut1,
                polar_motion_flag=_flag_field(line, 17),
                ut1_flag=_flag_field(line, 58),
                nutation_flag=_flag_field(line, 96),
                polar_motion_x_uncertainty_arcsec=_float_field(line, 28, 36),
                polar_motion_y_uncertainty_arcsec=_float_field(line, 47, 55),
                ut1_minus_utc_uncertainty_s=_float_field(line, 69, 78),
                dx_mas=_float_field(line, 98, 106),
                dy_mas=_float_field(line, 117, 125),
            )
        )
    return EarthOrientationTable(tuple(records))


def _float_field(line: str, first: int, last: int) -> float | None:
    value = line[first - 1 : last].strip()
    if not value or value in {"-", "-999"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid EOP numeric field at bytes {first}-{last}: {value!r}") from exc


def _flag_field(line: str, position: int) -> str:
    return line[position - 1 : position].strip() or "?"


def _gps_seconds(time) -> float:
    if isinstance(time, Time):
        return float(time.gps)
    if isinstance(time, (str, bytes)):
        return float(Time(time, scale="utc").gps)
    try:
        return float(time)
    except (TypeError, ValueError):
        return float(Time(time, scale="utc").gps)


def _lerp(left: float, right: float, fraction: float) -> float:
    return left + fraction * (right - left)


def _optional_lerp(left, right, fraction):
    if left is None or right is None:
        return None
    return float(_lerp(left, right, fraction))


def _combined_flag(left: str, right: str) -> str:
    return "P" if "P" in {left, right} else left
