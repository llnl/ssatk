"""Coordinate-frame quicklook: GCRF, ITRF, lon/lat, and NTW consistency."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.constants import RGEO
from ssatk.coordinates.earth_fixed import gcrf_to_itrf, itrf_to_gcrf
from ssatk.coordinates.geodetic import gcrf_to_lonlat
from ssatk.coordinates.satellite_frames import gcrf_to_ntw, ntw_to_gcrf
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")
    orbit = Orbit.fromKeplerianElements(
        a=RGEO,
        e=0.02,
        i=np.radians(35.0),
        pa=0.0,
        raan=np.radians(15.0),
        trueAnomaly=0.0,
        t=t0,
    )
    times = Time(t0.gps + np.linspace(0.0, (3 if fast else 12) * 3600.0, 24 if fast else 96), format="gps")
    r_gcrf, v_gcrf = rv(orbit, times)
    r_gcrf = np.asarray(r_gcrf, dtype=float).reshape((-1, 3))
    v_gcrf = np.asarray(v_gcrf, dtype=float).reshape((-1, 3))
    t_gps = times.gps

    r_itrf = gcrf_to_itrf(r_gcrf, t_gps)
    r_roundtrip = itrf_to_gcrf(r_itrf, t_gps)
    lon, lat, height = gcrf_to_lonlat(r_gcrf, t_gps)

    dv_gcrf = np.array([1.0, 2.0, 3.0])
    dv_ntw = gcrf_to_ntw(dv_gcrf, r_gcrf[0], v_gcrf[0])
    dv_roundtrip = ntw_to_gcrf(dv_ntw, r_gcrf[0], v_gcrf[0])
    roundtrip_error_m = np.linalg.norm(r_roundtrip - r_gcrf, axis=1)
    ntw_error = float(np.linalg.norm(dv_roundtrip - dv_gcrf))

    if make_figures:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(r_gcrf[:, 0] / 1e3, r_gcrf[:, 1] / 1e3)
        axes[0, 0].set_title("GCRF XY")
        axes[0, 0].set_xlabel("x [km]")
        axes[0, 0].set_ylabel("y [km]")
        axes[0, 0].axis("equal")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(r_itrf[:, 0] / 1e3, r_itrf[:, 1] / 1e3)
        axes[0, 1].set_title("ITRF XY")
        axes[0, 1].set_xlabel("x [km]")
        axes[0, 1].set_ylabel("y [km]")
        axes[0, 1].axis("equal")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(lon, lat, ".-")
        axes[1, 0].set_title("Ground track")
        axes[1, 0].set_xlabel("longitude [deg]")
        axes[1, 0].set_ylabel("latitude [deg]")
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot((t_gps - t_gps[0]) / 3600.0, roundtrip_error_m)
        axes[1, 1].set_title(f"GCRF↔ITRF round trip; NTW error {ntw_error:.2e} m/s")
        axes[1, 1].set_xlabel("time [hr]")
        axes[1, 1].set_ylabel("position error [m]")
        axes[1, 1].grid(alpha=0.3)

        fig.tight_layout()
        figsave(fig, f"{FIGDIR}/coordinate_frame_overview.jpg")

    return {
        "r_gcrf": r_gcrf,
        "v_gcrf": v_gcrf,
        "r_itrf": r_itrf,
        "lon": lon,
        "lat": lat,
        "height": height,
        "roundtrip_error_m": roundtrip_error_m,
        "ntw_error": ntw_error,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
