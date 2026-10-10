"""Applied photometry demo: topocentric attenuation and visible/IR band contrast."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from ssatk.compute.lambertian_magnitude import lambertian_reflection
from ssatk.constants import EARTH_RADIUS
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    time = Time("2025-01-01T08:00:00", scale="utc")
    ranges_km = np.linspace(800.0, 36_000.0, 40 if fast else 120)
    positions = np.column_stack(((EARTH_RADIUS + ranges_km * 1e3), np.zeros_like(ranges_km), np.zeros_like(ranges_km)))
    bands = ["V", "SWIR", "LWIR"]

    topocentric = {}
    exoatmospheric = {}
    geocentric_proxy = {}
    for band in bands:
        topo = lambertian_reflection(
            positions,
            time=time,
            lon=0.0,
            lat=0.0,
            elevation=0.0,
            band=band,
            radius_m=1.0,
            albedo=0.25,
        )
        geo = lambertian_reflection(
            positions,
            time=time,
            observer=np.array([1.0, 0.0, 0.0]),
            band=band,
            radius_m=1.0,
            albedo=0.25,
        )
        topocentric[band] = np.asarray(topo["ab_mag_observed"], dtype=float)
        exoatmospheric[band] = np.asarray(topo["ab_mag_exoatmospheric"], dtype=float)
        geocentric_proxy[band] = np.asarray(geo["ab_mag_exoatmospheric"], dtype=float)

    if make_figures:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for band in bands:
            axes[0].plot(ranges_km, topocentric[band], label=f"{band} observed")
            axes[0].plot(ranges_km, exoatmospheric[band], ls="--", label=f"{band} exo")
        axes[0].invert_yaxis()
        axes[0].set_xlabel("range above equator [km]")
        axes[0].set_ylabel("AB magnitude")
        axes[0].set_title("Topocentric observed vs exoatmospheric magnitude")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)

        for band in bands:
            axes[1].plot(ranges_km, topocentric[band] - geocentric_proxy[band], label=band)
        axes[1].set_xlabel("range above equator [km]")
        axes[1].set_ylabel("topocentric observed - geocentric proxy [mag]")
        axes[1].set_title("Observer/atmosphere effect by band")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        figsave(fig, f"{FIGDIR}/photometry_topocentric_geocentric_bands.jpg")

    return {
        "ranges_km": ranges_km,
        "topocentric": topocentric,
        "exoatmospheric": exoatmospheric,
        "geocentric_proxy": geocentric_proxy,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
