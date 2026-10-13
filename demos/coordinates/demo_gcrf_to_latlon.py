"""Compare SSATK/SSAPy geodetic ground tracks with Astropy GCRS/ITRS/WGS84.

The circular inclined trajectory is prescribed, not flight tracking data.
The two Earth-orientation models differ: SSAPy uses IAU 1976/1980, while
Astropy uses IAU 2006/2000A. The 0.5 arcsec angular budget follows the
independent frame comparisons in tests/test_plot_earth_rotation.py.
"""
from pathlib import Path
import json

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from ssatk.constants import EARTH_MU
from ssatk.coordinates.sky import gcrf_to_lat_lon
from ssatk.plots.figpath import figpath

TITLE = "Ground track: SSATK versus Astropy"
DESCRIPTION = "Prescribed 7000 km circular orbit; independent GCRS/ITRS/WGS84 comparison, not flight data."


def main(make_figures=True, fast=True):
    radius = 7e6
    period = 2 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    seconds = np.linspace(0, period, 91 if fast else 721)
    epoch = Time("2019-03-01T06:00:00", scale="utc")
    times = epoch + seconds * u.s
    angle = 2 * np.pi * seconds / period
    inclination = np.deg2rad(53)
    r = radius * np.column_stack((np.cos(angle), np.sin(angle)*np.cos(inclination),
                                 np.sin(angle)*np.sin(inclination)))
    lon, lat, height = gcrf_to_lat_lon(r, times.gps)
    itrs = GCRS(CartesianRepresentation(*(r.T*u.m)), obstime=times).transform_to(ITRS(obstime=times))
    reference = EarthLocation.from_geocentric(*itrs.cartesian.xyz).to_geodetic("WGS84")
    dlon = (np.asarray(lon) - reference.lon.to_value(u.rad) + np.pi) % (2*np.pi) - np.pi
    dlat = np.asarray(lat) - reference.lat.to_value(u.rad)
    errors = np.degrees(np.column_stack((dlon, dlat))) * 3600
    dh = np.asarray(height) - reference.height.to_value(u.m)
    metrics = {"reference": "Astropy GCRS/ITRS/WGS84", "trajectory": "prescribed circular orbit, not flight data",
               "samples": len(times), "max_lon_arcsec": float(np.max(np.abs(errors[:,0]))),
               "max_lat_arcsec": float(np.max(np.abs(errors[:,1]))), "max_height_m": float(np.max(np.abs(dh))),
               "angular_budget_arcsec": .5, "height_budget_m": .1}
    # 0.1 m height allows ellipsoid-height differences from the orientation
    # model at this radius (rotation preserves geocentric radius).
    if np.max(np.abs(errors)) > .5 or np.max(np.abs(dh)) > .1:
        raise AssertionError(f"Independent ground-track reference exceeded its budget: {metrics}")
    print(json.dumps(metrics, indent=2))
    if make_figures:
        # gallery-call: *.png
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(seconds/60, errors[:,0], label="Longitude")
        axes[0].plot(seconds/60, errors[:,1], label="Latitude")
        axes[0].set(ylabel="SSATK − Astropy [arcsec]", title=TITLE)
        axes[0].legend()
        axes[1].plot(seconds/60, dh)
        axes[1].set(xlabel="Elapsed time [min]", ylabel="Height difference [m]")
        figure.suptitle("Prescribed orbit; independent model comparison")
        figure.tight_layout()
        figure.savefig(figpath("gcrf_to_lonlat_groundtrack_vs_benchmark.png"), dpi=160)
        # gallery-end
        plt.close(figure)
        Path(figpath("gcrf_to_lonlat_reference.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"title": TITLE, "description": DESCRIPTION}


if __name__ == "__main__":
    main()
