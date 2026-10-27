"""Coordinate-frame quicklook: GCRF, ITRF, lon/lat, and NTW consistency."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.constants import EARTH_RADIUS, RGEO
from ssatk.coordinates.earth_fixed import gcrf_to_itrf, itrf_to_gcrf
from ssatk.coordinates.geodetic import gcrf_to_lonlat
from ssatk.coordinates.satellite_frames import frame_to_gcrf_matrix, gcrf_to_ntw, ntw_to_gcrf
from ssatk.coordinates.lunar import gcrf_to_lunar_fixed
from ssatk.plots._body_surfaces import body_disc, body_rotation
from ssatk.plots.plotutils import figsave, load_earth_file

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


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

    # A local Earth-fixed direction requires an epoch before using it in GCRF.
    enu_to_gcrf = frame_to_gcrf_matrix("enu", lat=0.0, lon=0.0, t=t0)
    greenwich_up_gcrf = enu_to_gcrf @ [0.0, 0.0, 1.0]
    # These are instantaneous rotating-frame velocities in m/s, even when
    # output epochs are spaced hours apart.
    r_lunar, v_lunar = gcrf_to_lunar_fixed(r_gcrf, times, v=v_gcrf)

    if make_figures:
        # gallery-call: *.jpg
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        elapsed_hr = (t_gps - t_gps[0]) / 3600.0
        fig.suptitle("One orbit, two reference frames", fontsize=19)
        fig.text(0.5, 0.935,
                 f"Inclined geosynchronous orbit · {elapsed_hr[-1]:g} hours from {t0.isot} UTC",
                 ha="center", fontsize=11)
        # Shared limits include Earth's origin and preserve the physical scale.
        limit_km = 1.12 * max(np.abs(r_gcrf[:, :2]).max(),
                              np.abs(r_itrf[:, :2]).max(), EARTH_RADIUS) / 1e3
        for ax, positions, frame, title in (
            (axes[0, 0], r_gcrf, "gcrf", "GCRF · inertial XY projection"),
            (axes[0, 1], r_itrf, "itrf", "ITRF · Earth-fixed XY projection"),
        ):
            body_disc(ax, "earth", EARTH_RADIUS / 1e3,
                      rotation=body_rotation("earth", t0, frame=frame))
            ax.plot(positions[:, 0] / 1e3, positions[:, 1] / 1e3,
                    color="#d35d00", linewidth=2, zorder=3)
            for index, marker, color, label in (
                (0, "o", "#20a463", "Start"), (-1, "s", "#c02d80", "End"),
            ):
                ax.scatter(positions[index, 0] / 1e3, positions[index, 1] / 1e3,
                           marker=marker, c=color, edgecolors="white", s=55,
                           zorder=4, label=label)
            ax.set(title=title, xlabel="x [km]", ylabel="y [km]",
                   xlim=(-limit_km, limit_km), ylim=(-limit_km, limit_km))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.text(0.98, 0.98, "Earth to scale\nMap at initial epoch",
                    transform=ax.transAxes, fontsize=9, ha="right", va="top")
        axes[0, 0].legend(loc="upper left", framealpha=0.9)

        ax = axes[1, 0]
        with load_earth_file() as earth_map:
            ax.imshow(earth_map, extent=(-180, 180, -90, 90),
                      origin="upper", interpolation="bilinear", aspect="auto")
        ax.plot(lon, lat, ".-", color="#ffdc58", linewidth=2, markersize=4)
        ax.scatter(lon[0], lat[0], c="#20a463", edgecolors="white", s=55, zorder=4)
        ax.scatter(lon[-1], lat[-1], marker="s", c="#c02d80",
                   edgecolors="white", s=55, zorder=4)
        ax.set(title="Ground track · subsatellite longitude / latitude",
               xlabel="longitude [deg east]", ylabel="geodetic latitude [deg north]",
               xlim=(max(-180, lon.min()-15), min(180, lon.max()+15)),
               ylim=(max(-90, lat.min()-12), min(90, lat.max()+12)))
        ax.grid(alpha=0.3, color="white")

        ax = axes[1, 1]
        error_mm = roundtrip_error_m * 1e3
        ax.plot(elapsed_hr, error_mm, color="#3465a4", linewidth=2)
        ax.set(title="GCRF → ITRF → GCRF · numerical consistency",
               xlabel="elapsed time [hr]", ylabel="position residual [mm]",
               ylim=(0, max(0.15, 1.5*error_mm.max())))
        ax.text(0.04, 0.08,
                f"Maximum: {error_mm.max():.4f} mm\n"
                f"NTW vector round trip: {ntw_error:.2e} m/s\n"
                "These are conversion residuals, not orbit accuracy.",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
        ax.grid(alpha=0.3)

        fig.text(0.5, 0.025,
                 "GCRF: Geocentric Celestial Reference Frame · ITRF: International Terrestrial Reference Frame\n"
                 "NTW: velocity-aligned orbital frame · ground track uses geodetic coordinates; XY projections do not show the z coordinate.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0.015, 0.065, 0.985, 0.925), h_pad=3, w_pad=3)
        figsave(fig, "coordinate_frame_overview.jpg")
        # gallery-end

    return {
        "r_gcrf": r_gcrf,
        "v_gcrf": v_gcrf,
        "r_itrf": r_itrf,
        "lon": lon,
        "lat": lat,
        "height": height,
        "roundtrip_error_m": roundtrip_error_m,
        "ntw_error": ntw_error,
        "greenwich_up_gcrf": greenwich_up_gcrf,
        "r_lunar": r_lunar,
        "v_lunar": v_lunar,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
