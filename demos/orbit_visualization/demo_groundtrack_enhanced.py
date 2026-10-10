"""
Demo script for ssatk groundtrack_enhanced.

This is the instrumented ground track: unlike the plain groundtrack_plot demo,
which draws tracks and nothing else, this one answers an operational question --
*when can a specific site actually see this satellite?*

It adds, on top of the track:

  * a ground site at a real latitude/longitude, with its visibility circle
    computed for a minimum elevation angle (10 deg here, the usual mask for
    horizon clutter and atmospheric refraction);
  * eclipse shading along the track, from real Sun geometry, so you can see
    which passes are in sunlight and which are in Earth's shadow;
  * the subsolar point and day/night terminator at the epoch;
  * a returned site_visibility_pct, the fraction of the propagated arc the
    site can see -- a number, not just a picture.

Change site_lat/site_lon to move the ground station, or swap in your own
ephemeris. To drive it from live TLEs instead of the synthetic orbit below,
see ssatk/plots/tle_updater.py; that path needs Space-Track
credentials and network access, so this demo stays self-contained.

Pytest-safe mode:
- shorter propagation and fewer points
- does not save figures by default
"""

GALLERY_CATEGORY = "orbit_visualization"

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

from ssatk.plots.groundtrack_enhanced import plot_enhanced_groundtrack
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Livermore, CA -- LLNL. Any lat/lon works; this one makes the visibility
# circle land somewhere recognisable on the map.
SITE_LAT = 37.6819
SITE_LON = -121.7068
SITE_NAME = "LLNL, Livermore CA"

# ISS-like: 420 km circular, 51.6 deg inclination. Chosen because its ground
# track is familiar and it genuinely does pass over northern California, so the
# visibility circle is not decorative.
ALT_KM = 420.0
INC_DEG = 51.6

# 10 deg is the conventional mask: below it a pass is usually unusable because
# of terrain, buildings and atmospheric refraction.
MIN_ELEV_DEG = 10.0

EPOCH = "2025-07-02T00:00:00"


def make_circular_orbit(alt_km=ALT_KM, inc_deg=INC_DEG, npts=1200, n_orbits=3.0):
    """Synthetic circular orbit in GCRF, returned as (r_km, Time).

    Kept deliberately simple and dependency-free so the demo runs anywhere.
    Substitute a real propagated ephemeris (ssapy.rv, or an OrbitalState) for
    anything quantitative.
    """
    mu = 398600.4418          # km^3/s^2
    re = 6378.137             # km
    a = re + alt_km
    inc = np.radians(inc_deg)
    n = np.sqrt(mu / a ** 3)  # rad/s

    period_s = 2.0 * np.pi / n
    dt = np.linspace(0.0, n_orbits * period_s, int(npts))
    nu = n * dt

    x_pf = a * np.cos(nu)
    y_pf = a * np.sin(nu)
    r_km = np.column_stack((x_pf, y_pf * np.cos(inc), y_pf * np.sin(inc)))

    t = Time(EPOCH, scale="utc") + dt / 86400.0
    return r_km, t


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # Under pytest, one orbit at coarse sampling is enough to exercise every
    # code path (visibility, eclipse, terminator) without the cost.
    npts = 300 if fast else 1500
    n_orbits = 1.0 if fast else 3.0

    r_km, t = make_circular_orbit(npts=npts, n_orbits=n_orbits)

    fig, ax, stats = plot_enhanced_groundtrack(
        r_eci_km=r_km,
        t=t,
        site_lat=SITE_LAT,
        site_lon=SITE_LON,
        site_name=SITE_NAME,
        sat_name=f"LEO {INC_DEG:.1f} deg, {ALT_KM:.0f} km",
        min_elev_deg=MIN_ELEV_DEG,
    )

    vis = stats.get("site_visibility_pct")
    if vis is not None:
        print(f"  {SITE_NAME} sees the satellite for {vis:.1f}% of the "
              f"{n_orbits:.0f}-orbit arc (>= {MIN_ELEV_DEG:.0f} deg elevation)")

    if make_figures:
        figsave(fig, "demo_gallery/figures/demo_groundtrack_enhanced")
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "site": (SITE_LAT, SITE_LON, SITE_NAME),
        "min_elev_deg": MIN_ELEV_DEG,
        "site_visibility_pct": vis,
        "track": (r_km, t),
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
