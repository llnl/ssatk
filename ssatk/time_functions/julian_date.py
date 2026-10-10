"""Calendar datetime to Julian Date."""

from __future__ import annotations

from datetime import datetime

__all__ = ["julian_date", "_julian_date"]


def julian_date(d: datetime) -> float:
    """Julian Date for a (UTC) datetime.

    A continuous day count from 4713 BC, which is what makes time differences
    across month and year boundaries trivial instead of a calendar problem.

    Applies the Gregorian correction term B to every date, so dates before
    1582-10-15 are read on the proleptic Gregorian calendar, as in ERFA
    ``cal2jd`` and astropy, not on the Julian calendar. Sub-second components are carried
    through deliberately: each second discarded costs ~15 arcsec of sidereal
    angle downstream, which is visible in a rendered star field.

    Previously defined in ssatk/plots/starfield.py. It moved here
    because it is a calendar conversion with no rendering content, and leaving
    it inside the plotting package forced the physics layer to import that
    package to reach it -- which created a circular import.
    """
    y, m = d.year, d.month
    day = d.day + (d.hour + d.minute / 60 + (d.second + d.microsecond / 1e6) / 3600) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + B - 1524.5


# starfield and magnetosphere_core import this under its old private name.
_julian_date = julian_date
