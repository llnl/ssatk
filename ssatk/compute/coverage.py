"""Finite-window geometric access statistics, with interpolated threshold events.

Inputs are geometric elevations and ITRF positions, without refraction,
terrain, weather, link budgets or optical detectability. Sample spacing is a
resolution limit: an entire contact between two samples can still be missed.
"""

import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u


def access_statistics(elevation_deg, times_s, min_elevation_deg=0.0):
    """Metrics for contacts above a threshold within the supplied time window.

    Crossings are linearly interpolated in elevation. Both endpoints are
    included, but no time beyond them is counted. Contacts already in progress
    at either boundary are clipped to the window and explicitly counted as
    such. Rates count contacts intersecting this finite window, not an
    asymptotic revisit rate. Max gap includes the two window-edge gaps.
    """
    elevation = np.asarray(elevation_deg, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or elevation.shape != times.shape
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(elevation))
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(min_elevation_deg)
    ):
        raise ValueError(
            "Finite elevations and at least two increasing epochs are required"
        )
    t = times - times[0]
    margin = elevation - min_elevation_deg
    visible = margin > 0
    indices = np.flatnonzero(visible[1:] != visible[:-1])
    crossings = t[indices] - margin[indices] * np.diff(t)[indices] / (
        margin[indices + 1] - margin[indices]
    )
    starts = crossings[~visible[indices]]
    ends = crossings[visible[indices]]
    if visible[0]:
        starts = np.r_[0.0, starts]
    if visible[-1]:
        ends = np.r_[ends, t[-1]]
    lengths = ends - starts
    positive = lengths > 0
    starts, ends, lengths = starts[positive], ends[positive], lengths[positive]
    count = len(lengths)
    duration = t[-1]
    gaps = np.diff(np.r_[0.0, np.column_stack((starts, ends)).ravel(), duration])[::2]
    contact = float(lengths.sum())
    return {
        "mean_pass_min": float(lengths.mean() / 60) if count else 0.0,
        "max_pass_min": float(lengths.max() / 60) if count else 0.0,
        "max_gap_min": float(gaps.max() / 60),
        "n_passes": count,
        "passes_per_day": count * 86400 / duration,
        "cum_contact_min_per_day": contact / duration * 1440,
        "time_to_first_min": float(starts[0] / 60) if count else np.inf,
        "fraction": contact / duration,
        "contact_seconds": contact,
        "duration_seconds": float(duration),
        "start_clipped": bool(visible[0]),
        "end_clipped": bool(visible[-1]),
        "intervals_seconds": np.column_stack((starts, ends)),
    }


def latitude_band_weights(latitude_edges_deg):
    """Exact relative surface areas of full-longitude WGS84 latitude bands."""
    # dA is proportional to cos(phi)/(1-e^2 sin(phi)^2)^2 dphi dlambda.
    f = 1 / 298.257223563
    e = np.sqrt(f * (2 - f))
    s = np.sin(np.radians(latitude_edges_deg))
    primitive = s / (2 * (1 - e**2 * s**2)) + np.arctanh(e * s) / (2 * e)
    return np.diff(primitive)


def coverage_grid(r_itrf_m, times_s, *, min_elevation_deg=5.0, n_lat=90, n_lon=180):
    """Cell-center WGS84 sea-level access map, with unmodified cell metrics.

    Grid edges cover [-90,90] latitude and [-180,180] longitude exactly;
    there is no duplicated antimeridian or repeated polar point. The global
    mean is weighted by each cell's ellipsoid surface area.
    """
    positions = np.asarray(r_itrf_m, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if positions.shape != (len(times), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("Positions must be a finite (N,3) ITRF array in metres")
    if not 0 <= min_elevation_deg < 90:
        raise ValueError("Coverage elevation mask must lie in [0,90) degrees")
    if n_lat < 2 or n_lon < 2:
        raise ValueError("Coverage grid needs at least two cells per axis")
    # Validate epochs once even for a map with no access.
    access_statistics(np.zeros(len(times)), times)
    lat_edges = np.linspace(-90, 90, n_lat + 1)
    lon_edges = np.linspace(-180, 180, n_lon + 1)
    lats = (lat_edges[:-1] + lat_edges[1:]) / 2
    lons = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon, lat = np.meshgrid(lons, lats)
    sites = EarthLocation.from_geodetic(
        lon.ravel() * u.deg, lat.ravel() * u.deg, 0 * u.m, ellipsoid="WGS84"
    )
    xyz = np.array([x.to_value(u.m) for x in sites.to_geocentric()]).T
    phi, lam = np.radians(lat.ravel()), np.radians(lon.ravel())
    up = np.column_stack(
        (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
    )
    names = [
        "fraction",
        "mean_pass_min",
        "max_pass_min",
        "max_gap_min",
        "passes_per_day",
        "cum_contact_min_per_day",
        "n_passes",
    ]
    grids = {name: np.zeros(lon.size) for name in names}
    # Per-site vectorization bounds memory even for long time histories.
    for i, (site, normal) in enumerate(zip(xyz, up)):
        rho = positions - site
        vertical = np.einsum("ij,j->i", rho, normal)
        horizontal2 = np.maximum(np.einsum("ij,ij->i", rho, rho) - vertical**2, 0)
        elevation = np.degrees(np.arctan2(vertical, np.sqrt(horizontal2)))
        stats = access_statistics(elevation, times, min_elevation_deg)
        for name in names:
            grids[name][i] = stats[name]
    grids = {name: values.reshape(n_lat, n_lon) for name, values in grids.items()}
    weights = latitude_band_weights(lat_edges)
    return dict(
        grids,
        lat_grid=lats,
        lon_grid=lons,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
        global_mean=float(
            np.sum(grids["fraction"] * weights[:, None]) / (weights.sum() * n_lon)
        ),
    )
