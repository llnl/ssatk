"""Greenwich Mean Sidereal Time."""

from __future__ import annotations

from datetime import datetime

import numpy as np

try:
    from .julian_date import julian_date
except ImportError:  # script mode, no package context
    from ssatk.time_functions.julian_date import julian_date

__all__ = ["gmst_rad", "_gmst_rad"]


def gmst_rad(d: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982 series).

    Sidereal time is measured against the stars rather than the Sun, so a
    sidereal day is ~23h56m04s -- about four minutes short of a solar day,
    because Earth must turn slightly more than 360 degrees to bring the Sun
    back overhead.

    This is the rotation angle between an inertial frame (where star
    catalogues live) and an Earth-fixed one, so it is what any GCRF-to-ECEF
    conversion needs: star fields, ground tracks, and the subsolar point all
    depend on it.

    The cubic series is good to well under an arcsecond for any epoch this
    toolkit handles; verified against astropy to 0.01 s of sidereal time in
    tests/test_magfield_physics.py.

    Previously defined in ssatk/plots/starfield.py -- see julian_date
    for why it moved.
    """
    T = (julian_date(d) - 2451545.0) / 36525.0
    g = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T
         + 0.093104 * T * T - 6.2e-6 * T * T * T)
    return np.radians((g % 86400.0) / 240.0)


# starfield and magnetosphere_core import this under its old private name.
_gmst_rad = gmst_rad
