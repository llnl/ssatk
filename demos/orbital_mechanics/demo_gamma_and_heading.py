"""Earth-relative geocentric flight-path angle and compass heading.

One complete analytic Kepler orbit uses SSAPy position AND velocity. The
reference transforms the state with Astropy, then derives angles from
spherical-coordinate rates. Gamma is positive radially outward; heading is
clockwise from north. This geocentric horizon differs from the WGS84 normal.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from astropy.time import Time
from ssapy import Orbit, compute
from ssapy.propagator import KeplerianPropagator

from ssatk.constants import EARTH_RADIUS, EARTH_MU
from ssatk.orbital_mechanics.gamma_and_heading import (
    calc_gamma_and_heading,
    calc_gamma_and_heading_itrf,
)
from ssatk.plots.figpath import figpath
from ssatk.plots.groundtrack_dashboard_gamma_heading import (
    groundtrack_dashboard_gamma_heading,
)

TITLE = "Gamma and Heading — Earth-relative motion"
DESCRIPTION = (
    "One complete eccentric, inclined Kepler orbit. Geocentric gamma and "
    "compass heading use Earth-relative velocity; speed is GCRF. "
    "Checked against Astropy and analytic two-body invariants."
)
GALLERY_CATEGORY = "orbital_mechanics"
UNDER_PYTEST = (
    "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
)

A_M = EARTH_RADIUS + 7000e3
ECCENTRICITY = 0.2
INCLINATION_DEG = 45.0
EPOCH_UTC = "2025-01-01T00:00:00"


def reference_angles(r, v, t):
    """Independent Astropy state transform and spherical-coordinate rates.

    The production path uses SSAPy's IAU 1976/80 Earth rotation, while Astropy
    uses IAU 2006/2000A. This compares conventions; it is not tracking truth.
    """
    import astropy.units as u
    from astropy.coordinates import (
        GCRS,
        ITRS,
        CartesianRepresentation,
        CartesianDifferential,
    )

    rep = CartesianRepresentation(
        r.T * u.m, differentials=CartesianDifferential(v.T * u.m / u.s)
    )
    state = GCRS(rep, obstime=t).transform_to(ITRS(obstime=t)).cartesian
    x, y, z = state.xyz.to_value(u.m)
    vx, vy, vz = state.differentials["s"].d_xyz.to_value(u.m / u.s)
    rho = np.hypot(x, y)
    radius = np.sqrt(x * x + y * y + z * z)
    radius_dot = (x * vx + y * vy + z * vz) / radius
    rho_dot = (x * vx + y * vy) / rho
    latitude_dot = (rho * vz - z * rho_dot) / radius**2
    longitude_dot = (x * vy - y * vx) / rho**2
    east_speed = rho * longitude_dot
    north_speed = radius * latitude_dot
    return (
        np.degrees(np.arctan2(radius_dot, np.hypot(east_speed, north_speed))),
        np.degrees(np.arctan2(east_speed, north_speed)) % 360,
    )


def heading_gamma_panel():
    """Known local velocities evaluated by the production ITRF angle API.

    Poles are excluded because geographic heading is undefined there. Dot
    and cross markers denote outward/inward radial motion, without inventing
    a horizontal heading. Horizontal arrows are direction symbols, not speed.
    """
    lon, lat = np.meshgrid(np.arange(-150.0, 151.0, 60.0), np.arange(-60.0, 61.0, 30.0))
    lam, phi = np.radians(lon.ravel()), np.radians(lat.ravel())
    up = np.column_stack(
        (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
    )
    east = np.column_stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)))
    north = np.column_stack(
        (-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi))
    )
    positions = (EARTH_RADIUS + 700e3) * up
    directions = [
        ("Up", up, 90, None),
        ("Down", -up, -90, None),
        ("East", east, 0, 90),
        ("West", -east, 0, 270),
        ("North", north, 0, 0),
        ("South", -north, 0, 180),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    norm = Normalize(-90, 90)
    for ax, (name, direction, expected_gamma, expected_heading) in zip(
        axes.flat, directions
    ):
        gamma, heading = calc_gamma_and_heading_itrf(
            positions, np.arange(len(phi)), v=7500 * direction
        )
        np.testing.assert_allclose(gamma, expected_gamma, atol=1e-10, rtol=0)
        if expected_heading is None:
            assert np.isnan(heading).all()
            artist = ax.scatter(
                lon,
                lat,
                c=gamma,
                cmap="coolwarm",
                norm=norm,
                marker=r"$\odot$" if name == "Up" else r"$\otimes$",
                s=180,
            )
            subtitle = "No horizontal heading"
        else:
            error = (heading - expected_heading + 180) % 360 - 180
            np.testing.assert_allclose(error, 0, atol=1e-10, rtol=0)
            east_plot = np.sin(np.radians(heading)) / np.cos(phi)
            north_plot = np.cos(np.radians(heading))
            length = np.hypot(east_plot, north_plot)
            artist = ax.quiver(
                lon.ravel(),
                lat.ravel(),
                18 * east_plot / length,
                18 * north_plot / length,
                gamma,
                cmap="coolwarm",
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=1,
                edgecolor="#26374a",
                linewidth=0.7,
                width=0.007,
            )
            subtitle = f"Heading {expected_heading}° clockwise from north"
        ax.set(
            title=f"{name}: gamma {expected_gamma:+g}°\n{subtitle}",
            xlim=(-180, 180),
            ylim=(-85, 85),
            xlabel="Geocentric longitude [deg]",
            ylabel="Geocentric latitude [deg]",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.colorbar(artist, ax=axes, label="Geocentric gamma [deg]", shrink=0.8)
    fig.suptitle(
        "ITRF local directions | ⊙ radial outward, ⊗ radial inward\n"
        "Geocentric horizon; no heading at the poles or for radial/stationary motion",
        fontsize=16,
    )
    return fig


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    orbit = Orbit.fromKeplerianElements(
        a=A_M,
        e=ECCENTRICITY,
        i=np.radians(INCLINATION_DEG),
        pa=0.0,
        raan=0.0,
        trueAnomaly=0.0,
        t=Time(EPOCH_UTC).gps,
    )
    period = 2 * np.pi * np.sqrt(A_M**3 / EARTH_MU)
    elapsed = np.linspace(0, period, 721 if fast else 1441)
    t = Time(orbit.t + elapsed, format="gps")
    r, v = compute.rv(orbit, t, propagator=KeplerianPropagator())
    gamma, heading = calc_gamma_and_heading(r, t, v=v)
    ref_gamma, ref_heading = reference_angles(r, v, t)
    gamma_error = float(np.max(np.abs(gamma - ref_gamma)))
    heading_error = float(np.max(np.abs((heading - ref_heading + 180) % 360 - 180)))
    speed = np.linalg.norm(v, axis=1)
    vis_viva = np.sqrt(EARTH_MU * (2 / np.linalg.norm(r, axis=1) - 1 / A_M))
    np.testing.assert_allclose(speed, vis_viva, atol=1e-8, rtol=0)
    assert gamma_error < 5e-4 and heading_error < 5e-4, (gamma_error, heading_error)
    summary = dict(
        epoch_utc=EPOCH_UTC,
        propagator="KeplerianPropagator",
        semimajor_axis_m=A_M,
        eccentricity=ECCENTRICITY,
        inclination_deg=INCLINATION_DEG,
        duration_s=period,
        samples=len(t),
        step_s=float(elapsed[1]),
        gamma_max_error_deg=gamma_error,
        heading_max_error_deg=heading_error,
        vis_viva_max_error_m_s=float(np.max(np.abs(speed - vis_viva))),
        convention="Earth-relative geocentric gamma and compass heading; GCRF speed; WGS84 altitude",
        reference="Astropy GCRS-to-ITRS with velocity differentials, spherical coordinate rates, and analytic vis-viva",
    )
    print(f"Validated {len(t)} states over one {period / 3600:.6f}-hour Kepler period.")
    print(
        f"Maximum Astropy angle differences: gamma {gamma_error:.8g}°, heading {heading_error:.8g}°."
    )
    print(f"GCRF speed: {speed.min() / 1000:.6f}–{speed.max() / 1000:.6f} km/s.")
    if make_figures:
        fig = groundtrack_dashboard_gamma_heading(r, t, v=v, fontsize=13)
        fig.suptitle(
            f"Earth-relative gamma and heading | a={A_M / 1000:.3f} km, e={ECCENTRICITY}, i={INCLINATION_DEG:g}°\n"
            f"One Kepler period from {EPOCH_UTC} UTC | {len(t)} states | supplied state velocities",
            fontsize=18,
        )
        fig.subplots_adjust(top=0.89, wspace=0.38, hspace=0.32)
        path = Path(figpath("dashboard_gamma_heading.png"))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        fig = heading_gamma_panel()
        fig.savefig(figpath("heading_gamma_panel_ITRF.png"), dpi=150)
        plt.close(fig)
        np.savetxt(
            figpath("gamma_heading_series.csv"),
            np.column_stack(
                (t.gps, elapsed, gamma, heading, speed, ref_gamma, ref_heading)
            ),
            delimiter=",",
            header="gps_s,elapsed_s,gamma_deg,heading_deg,gcrf_speed_m_s,astropy_gamma_deg,astropy_heading_deg",
            comments="",
        )
        Path(figpath("gamma_heading_validation.json")).write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return dict(
        r=r, v=v, t=t, gamma=gamma, heading=heading, validation=summary, ok=True
    )


if __name__ == "__main__":
    main(make_figures=True, fast=False)
