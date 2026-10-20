"""Conditional eclipse uncertainty propagation for V19.3.

This module intentionally separates *nominal truth* from *uncertainty
assumptions*.  The nominal state is produced by the selected eclipse backend.
The Monte Carlo engine perturbs that immutable state according to an explicit,
serializable budget.  Its confidence intervals are therefore conditional on
that budget; they are not presented as a universal covariance for NASA, JPL,
SSAPy, or any other provider.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Mapping
import json
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_brightness import shadow_axis_surface_point, shadow_cross_section
from ssapy_toolkit.compute.eclipse_state import event_gcrf_to_itrf_km, event_positions_gcrf
from ssapy_toolkit.coordinates.eclipse_lunar_geometry import LunarLimbProfile
from ssapy_toolkit.coordinates.eclipse_observer_geometry import ObserverConfig, observer_state
from ssapy_toolkit.compute.eclipse_reference_events import (
    EARTH_AXES_KM, LUNAR_2025, RE_KM, R_MOON_MEAN_KM,
    SOLAR_2024, SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG, SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    SOLAR_PENUMBRA_OPTICAL_RADIUS_KM, build_reference_event,
    geodetic_from_itrf, lunar_reference_state, lunar_track_x_deg, solar_local_contacts, solar_central_line_wgs84,
)
from ssapy_toolkit.compute.eclipse_raytrace import solar_cross_track_width_km

SIDEREAL_DAY_S = 86164.0905


@dataclass(frozen=True)
class UncertaintyBudget:
    """Independent one-sigma inputs used by the conditional Monte Carlo.

    Values are deliberately configuration, not hidden package constants.
    Defaults form a moderate engineering stress-test budget suitable for
    demonstrating propagation.  A release that has provider covariance data
    should replace them and record the source in ``label``.
    """

    label: str = "V19.3 configurable engineering stress-test budget"
    ephemeris_time_sigma_s: float = 0.25
    sun_position_axis_sigma_km: float = 0.5
    moon_position_axis_sigma_km: float = 0.05
    ut1_sigma_s: float = 0.02
    lunar_limb_sigma_km: float = 0.25
    observer_horizontal_sigma_m: float = 3.0
    observer_elevation_sigma_m: float = 2.0
    lunar_shadow_axis_sigma_km: float = 0.20
    earth_shadow_radius_sigma_km: float = 0.30
    nasa_lunar_limb_path_systematic_km: float = 3.0
    nasa_lunar_limb_contact_systematic_s: float = 3.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "label":
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def zero(cls, *, label: str = "zero-width regression budget") -> "UncertaintyBudget":
        values = {name: 0.0 for name in cls.__dataclass_fields__ if name != "label"}
        return cls(label=label, **values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "UncertaintyBudget":
        allowed = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in allowed if key in value})

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IntervalSummary:
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    p2_5: float
    p16: float
    median: float
    p84: float
    p97_5: float
    maximum: float

    @classmethod
    def from_values(cls, values) -> "IntervalSummary":
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if not len(data):
            nan = float("nan")
            return cls(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
        mean = float(np.mean(data))
        span = float(np.max(data) - np.min(data))
        # Identical deterministic samples can differ by a few ULPs after
        # repeated coordinate and root transformations.  Treat only those
        # machine-scale differences as a collapsed interval so a zero-width
        # budget serializes as exactly zero spread rather than ~1e-14.
        tolerance = 1.0e-12 * max(1.0, abs(mean))
        if span <= tolerance:
            return cls(
                count=int(len(data)), mean=mean, standard_deviation=0.0,
                minimum=mean, p2_5=mean, p16=mean, median=mean,
                p84=mean, p97_5=mean, maximum=mean,
            )
        q = np.percentile(data, [2.5, 16, 50, 84, 97.5])
        return cls(
            count=int(len(data)), mean=mean,
            standard_deviation=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            minimum=float(np.min(data)), p2_5=float(q[0]), p16=float(q[1]),
            median=float(q[2]), p84=float(q[3]), p97_5=float(q[4]),
            maximum=float(np.max(data)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rng_vectors(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return rng.normal(0.0, float(sigma), size=(n, 3)) if sigma else np.zeros((n, 3))


def _observer_perturbation(rng: np.random.Generator, n: int, budget: UncertaintyBudget):
    east_m = rng.normal(0.0, budget.observer_horizontal_sigma_m, n) if budget.observer_horizontal_sigma_m else np.zeros(n)
    north_m = rng.normal(0.0, budget.observer_horizontal_sigma_m, n) if budget.observer_horizontal_sigma_m else np.zeros(n)
    elev_m = rng.normal(0.0, budget.observer_elevation_sigma_m, n) if budget.observer_elevation_sigma_m else np.zeros(n)
    lat0 = math.radians(SOLAR_GREATEST_SITE_LAT_DEG)
    lat_deg = SOLAR_GREATEST_SITE_LAT_DEG + north_m / (RE_KM * 1000.0) * 180.0 / math.pi
    lon_deg = SOLAR_GREATEST_SITE_LON_EAST_DEG + east_m / (RE_KM * 1000.0 * max(math.cos(lat0), 1e-9)) * 180.0 / math.pi
    return lat_deg, lon_deg, elev_m


def _solar_geometry(event, jd: float, *, sun_delta, moon_delta, time_bias_s: float,
                    ut1_bias_s: float, observer: ObserverConfig,
                    limb_radius_km: float) -> tuple[float, float, float, object]:
    state_jd = float(jd) + float(time_bias_s) / 86400.0
    sun, moon = event_positions_gcrf(event, state_jd)
    sun = np.asarray(sun, dtype=float) + np.asarray(sun_delta, dtype=float)
    moon = np.asarray(moon, dtype=float) + np.asarray(moon_delta, dtype=float)
    # A UT1 perturbation is first-order equivalent to an east-longitude
    # perturbation of the fixed terrestrial observer.
    config = replace(
        observer,
        longitude_east_deg=float(observer.longitude_east_deg) + 360.0 * float(ut1_bias_s) / SIDEREAL_DAY_S,
    )
    obs_state = observer_state(
        event, float(jd), config, sun_gcrf_km=sun, moon_gcrf_km=moon,
        limb_profile=LunarLimbProfile.circular(limb_radius_km),
    )
    to_sun = sun - obs_state.observer_gcrf_km
    to_moon = moon - obs_state.observer_gcrf_km
    ds = float(np.linalg.norm(to_sun)); dm = float(np.linalg.norm(to_moon))
    sep = math.acos(np.clip(np.dot(to_sun / ds, to_moon / dm), -1.0, 1.0))
    sun_ang = math.asin(np.clip(695700.0 / ds, 0.0, 1.0))
    moon_ang = math.asin(np.clip(limb_radius_km / dm, 0.0, 1.0))
    return sep, sun_ang, moon_ang, obs_state


def _root_near(function, reference_jd: float, initial_half_width_s: float = 90.0) -> float:
    from scipy.optimize import brentq
    half = float(initial_half_width_s)
    for _ in range(5):
        lo, hi = reference_jd - half / 86400.0, reference_jd + half / 86400.0
        grid = np.linspace(lo, hi, 25)
        vals = np.asarray([function(float(x)) for x in grid], dtype=float)
        pairs = []
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if fa == 0.0:
                return float(a)
            if fa * fb < 0.0:
                pairs.append((float(a), float(b)))
        if pairs:
            a, b = min(pairs, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - reference_jd))
            return float(brentq(function, a, b, xtol=1e-12))
        half *= 2.0
    return float("nan")


def propagate_solar_uncertainty(
    *, samples: int = 512, seed: int = 20240408,
    budget: UncertaintyBudget | None = None,
) -> dict[str, object]:
    """Propagate a configured budget through the 2024 total solar eclipse."""
    n = max(1, int(samples)); assumptions = budget or UncertaintyBudget()
    rng = np.random.default_rng(int(seed))
    event = build_reference_event("solar", n_frames=181)
    reference_contacts = solar_local_contacts()
    observer_lat, observer_lon, observer_elev = _observer_perturbation(rng, n, assumptions)
    sun_delta = _rng_vectors(rng, n, assumptions.sun_position_axis_sigma_km)
    moon_delta = _rng_vectors(rng, n, assumptions.moon_position_axis_sigma_km)
    time_bias = rng.normal(0.0, assumptions.ephemeris_time_sigma_s, n) if assumptions.ephemeris_time_sigma_s else np.zeros(n)
    ut1_bias = rng.normal(0.0, assumptions.ut1_sigma_s, n) if assumptions.ut1_sigma_s else np.zeros(n)
    limb_delta = rng.normal(0.0, assumptions.lunar_limb_sigma_km, n) if assumptions.lunar_limb_sigma_km else np.zeros(n)

    contact_names = ("C1", "C2", "C3", "C4")
    contact_offsets = {name: np.full(n, np.nan) for name in contact_names}
    totality_s = np.full(n, np.nan)
    max_visibility = np.full(n, np.nan)
    axis_east_km = np.full(n, np.nan); axis_north_km = np.full(n, np.nan)
    axis_along_km = np.full(n, np.nan); axis_cross_km = np.full(n, np.nan)
    optical_diameter_km = np.full(n, np.nan); corridor_width_km = np.full(n, np.nan)

    nominal_jd = float(SOLAR_2024.greatest_jd)
    nominal_width = float(solar_cross_track_width_km(nominal_jd, n_azimuth=720))
    nominal_sun, nominal_moon = event_positions_gcrf(event, nominal_jd)
    nominal_cs = shadow_cross_section(-nominal_moon, nominal_sun - nominal_moon, SOLAR_UMBRA_OPTICAL_RADIUS_KM)
    nominal_optical_diameter = 2.0 * max(float(nominal_cs.signed_umbra_radius_km), 1e-12)
    width_scale = nominal_width / nominal_optical_diameter

    # Nominal axis point and tangent-plane basis for spatial residuals.
    points_nom = event_gcrf_to_itrf_km(event, np.asarray([nominal_sun, nominal_moon]), np.asarray([nominal_jd, nominal_jd]))
    nominal_hit = shadow_axis_surface_point(points_nom[1], points_nom[0], target_axes_km=EARTH_AXES_KM)
    normal = nominal_hit / np.linalg.norm(nominal_hit)
    east = np.array([-normal[1], normal[0], 0.0]); east /= np.linalg.norm(east)
    north = np.cross(normal, east); north /= np.linalg.norm(north)
    before = solar_central_line_wgs84(nominal_jd - 1.0 / 86400.0)
    after = solar_central_line_wgs84(nominal_jd + 1.0 / 86400.0)
    dv = after[2] - before[2]
    velocity_en = np.array([np.dot(dv, east), np.dot(dv, north)], dtype=float)
    velocity_en /= np.linalg.norm(velocity_en)
    cross_en = np.array([-velocity_en[1], velocity_en[0]], dtype=float)

    from scipy.optimize import minimize_scalar
    for i in range(n):
        config = ObserverConfig(
            latitude_deg=float(observer_lat[i]), longitude_east_deg=float(observer_lon[i]),
            elevation_m=float(observer_elev[i]), apply_refraction=False,
            name="uncertainty-sampled greatest-eclipse observer",
        )
        limb_mean = R_MOON_MEAN_KM + float(limb_delta[i])
        for name in contact_names:
            internal = name in {"C2", "C3"}
            limb = (SOLAR_UMBRA_OPTICAL_RADIUS_KM if internal else SOLAR_PENUMBRA_OPTICAL_RADIUS_KM) + float(limb_delta[i])
            reference = float(reference_contacts[name])
            def equation(jd, internal=internal):
                sep, sun_ang, moon_ang, _ = _solar_geometry(
                    event, jd, sun_delta=sun_delta[i], moon_delta=moon_delta[i],
                    time_bias_s=time_bias[i], ut1_bias_s=ut1_bias[i],
                    observer=config, limb_radius_km=limb,
                )
                return sep - (abs(moon_ang - sun_ang) if internal else moon_ang + sun_ang)
            root = _root_near(equation, reference)
            contact_offsets[name][i] = (root - reference) * 86400.0 if math.isfinite(root) else np.nan
        if np.isfinite(contact_offsets["C2"][i]) and np.isfinite(contact_offsets["C3"][i]):
            c2 = reference_contacts["C2"] + contact_offsets["C2"][i] / 86400.0
            c3 = reference_contacts["C3"] + contact_offsets["C3"][i] / 86400.0
            totality_s[i] = (c3 - c2) * 86400.0
        center_jd = float(reference_contacts.get("MAX", 0.5 * (reference_contacts["C2"] + reference_contacts["C3"])))
        lo_s = (reference_contacts["C2"] - center_jd) * 86400.0
        hi_s = (reference_contacts["C3"] - center_jd) * 86400.0
        result = minimize_scalar(
            lambda offset_s: _solar_geometry(
                event, center_jd + float(offset_s) / 86400.0, sun_delta=sun_delta[i], moon_delta=moon_delta[i],
                time_bias_s=time_bias[i], ut1_bias_s=ut1_bias[i], observer=config,
                limb_radius_km=limb_mean,
            )[0], bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1.0e-4},
        )
        peak_jd = center_jd + float(result.x) / 86400.0
        sep, sun_ang, moon_ang, obs_state = _solar_geometry(
            event, peak_jd, sun_delta=sun_delta[i], moon_delta=moon_delta[i],
            time_bias_s=time_bias[i], ut1_bias_s=ut1_bias[i], observer=config,
            limb_radius_km=limb_mean,
        )
        max_visibility[i] = float(obs_state.photosphere_visible)

        # Global axis state at the nominal greatest instant.  A fixed-time
        # spatial envelope is more reproducible than mixing peak-time and
        # Earth-rotation uncertainty into one undocumented quantity.
        state_jd = nominal_jd + float(time_bias[i]) / 86400.0
        sun_i, moon_i = event_positions_gcrf(event, state_jd)
        sun_i = np.asarray(sun_i) + sun_delta[i]
        moon_i = np.asarray(moon_i) + moon_delta[i]
        transform_jd = nominal_jd + float(ut1_bias[i]) / 86400.0
        itrf = event_gcrf_to_itrf_km(event, np.asarray([sun_i, moon_i]), np.asarray([transform_jd, transform_jd]))
        hit = shadow_axis_surface_point(itrf[1], itrf[0], target_axes_km=EARTH_AXES_KM)
        if hit is not None:
            delta = hit - nominal_hit
            axis_east_km[i] = float(np.dot(delta, east))
            axis_north_km[i] = float(np.dot(delta, north))
            en = np.array([axis_east_km[i], axis_north_km[i]])
            axis_along_km[i] = float(np.dot(en, velocity_en))
            axis_cross_km[i] = float(np.dot(en, cross_en))
        cs = shadow_cross_section(-moon_i, sun_i - moon_i, SOLAR_UMBRA_OPTICAL_RADIUS_KM + float(limb_delta[i]))
        diameter = 2.0 * float(cs.signed_umbra_radius_km)
        optical_diameter_km[i] = diameter
        corridor_width_km[i] = width_scale * diameter

    outputs = {
        "contact_offset_s": {name: IntervalSummary.from_values(values).to_dict() for name, values in contact_offsets.items()},
        "totality_duration_s": IntervalSummary.from_values(totality_s).to_dict(),
        "maximum_photosphere_visible": IntervalSummary.from_values(max_visibility).to_dict(),
        "shadow_axis_east_offset_km": IntervalSummary.from_values(axis_east_km).to_dict(),
        "shadow_axis_north_offset_km": IntervalSummary.from_values(axis_north_km).to_dict(),
        "shadow_axis_along_track_offset_km": IntervalSummary.from_values(axis_along_km).to_dict(),
        "shadow_axis_cross_track_offset_km": IntervalSummary.from_values(axis_cross_km).to_dict(),
        "umbra_optical_diameter_at_earth_center_plane_km": IntervalSummary.from_values(optical_diameter_km).to_dict(),
        "calibrated_wgs84_corridor_width_km": IntervalSummary.from_values(corridor_width_km).to_dict(),
    }
    return {
        "$schema": "ssapy-toolkit.eclipse.uncertainty/1.9.3",
        "kind": "solar",
        "event": SOLAR_2024.key,
        "samples": n,
        "seed": int(seed),
        "budget": assumptions.to_dict(),
        "interpretation": "Conditional Monte Carlo intervals under the recorded independent Gaussian budget; not a universal ephemeris covariance.",
        "nominal": {
            "greatest_jd_utc": nominal_jd,
            "corridor_width_km": nominal_width,
            "umbra_optical_diameter_at_earth_center_plane_km": nominal_optical_diameter,
            "local_contacts_jd": {k: float(v) for k, v in reference_contacts.items()},
            "local_totality_duration_s": (reference_contacts["C3"] - reference_contacts["C2"]) * 86400.0,
        },
        "systematic_envelope_not_randomized": {
            "lunar_limb_path_edge_km": assumptions.nasa_lunar_limb_path_systematic_km,
            "local_contact_s": assumptions.nasa_lunar_limb_contact_systematic_s,
            "note": "Displayed separately so a bounded lunar-limb model discrepancy is not misrepresented as Gaussian provider covariance.",
        },
        "outputs": outputs,
        "raw": {
            "axis_east_km": axis_east_km.tolist(), "axis_north_km": axis_north_km.tolist(),
            "axis_along_km": axis_along_km.tolist(), "axis_cross_km": axis_cross_km.tolist(),
            "corridor_width_km": corridor_width_km.tolist(), "totality_duration_s": totality_s.tolist(),
            **{f"{name.lower()}_offset_s": values.tolist() for name, values in contact_offsets.items()},
        },
    }



def _lunar_track_x_extended(jd: float, event) -> float:
    """Continue the fitted lunar shadow-plane track just beyond P1/P4.

    The public reference track is intentionally clipped to the published event
    interval. Uncertainty perturbations can move an external penumbral contact
    a fraction of a second outside that interval, so root finding needs the
    endpoint tangent rather than a flat clipped value.
    """
    value = float(jd)
    p1 = float(event.definition.contacts_jd["P1"]); p4 = float(event.definition.contacts_jd["P4"])
    step = 30.0 / 86400.0
    if value < p1:
        x0 = float(lunar_track_x_deg(p1)); x1 = float(lunar_track_x_deg(p1 + step))
        return x0 + (value - p1) * (x1 - x0) / step
    if value > p4:
        x0 = float(lunar_track_x_deg(p4 - step)); x1 = float(lunar_track_x_deg(p4))
        return x1 + (value - p4) * (x1 - x0) / step
    return float(lunar_track_x_deg(value))

def _lunar_contact_equation(jd: float, *, contact: str, time_bias_s: float,
                            axis_delta_deg: np.ndarray, shadow_delta_deg: float,
                            moon_delta_deg: float, event) -> float:
    evaluation_jd = float(jd) + float(time_bias_s) / 86400.0
    state = lunar_reference_state(evaluation_jd)
    impact = np.asarray(state.impact_vector_deg, dtype=float)
    impact_hat = impact / np.linalg.norm(impact)
    track_hat = np.array([impact_hat[1], -impact_hat[0]])
    offset = impact + _lunar_track_x_extended(evaluation_jd, event) * track_hat + np.asarray(axis_delta_deg, dtype=float)
    rho = float(np.linalg.norm(offset))
    umbra = float(event.metadata["umbra_radius_deg"]) + float(shadow_delta_deg)
    penumbra = float(event.metadata["penumbra_radius_deg"]) + float(shadow_delta_deg)
    moon = float(event.metadata["moon_semidiameter_deg"]) + float(moon_delta_deg)
    if contact in {"P1", "P4"}:
        boundary = penumbra + moon
    elif contact in {"U1", "U4"}:
        boundary = umbra + moon
    elif contact in {"U2", "U3"}:
        boundary = umbra - moon
    else:
        raise ValueError(contact)
    return rho - boundary


def propagate_lunar_uncertainty(
    *, samples: int = 512, seed: int = 20250314,
    budget: UncertaintyBudget | None = None,
) -> dict[str, object]:
    """Propagate a configured budget through the 2025 total lunar eclipse."""
    n = max(1, int(samples)); assumptions = budget or UncertaintyBudget()
    rng = np.random.default_rng(int(seed))
    event = build_reference_event("lunar", n_frames=181)
    contacts = event.definition.contacts_jd
    moon_distance = float(event.metadata["moon_distance_km"])
    km_to_deg = 180.0 / math.pi / moon_distance
    time_bias = rng.normal(0.0, assumptions.ephemeris_time_sigma_s, n) if assumptions.ephemeris_time_sigma_s else np.zeros(n)
    axis_delta_deg = rng.normal(0.0, assumptions.lunar_shadow_axis_sigma_km * km_to_deg, size=(n, 2)) if assumptions.lunar_shadow_axis_sigma_km else np.zeros((n, 2))
    shadow_delta_deg = rng.normal(0.0, assumptions.earth_shadow_radius_sigma_km * km_to_deg, n) if assumptions.earth_shadow_radius_sigma_km else np.zeros(n)
    moon_delta_deg = rng.normal(0.0, assumptions.lunar_limb_sigma_km * km_to_deg, n) if assumptions.lunar_limb_sigma_km else np.zeros(n)

    names = ("P1", "U1", "U2", "U3", "U4", "P4")
    offsets = {name: np.full(n, np.nan) for name in names}
    totality = np.full(n, np.nan); partial = np.full(n, np.nan); penumbral = np.full(n, np.nan)
    umbral_mag = np.full(n, np.nan); penumbral_mag = np.full(n, np.nan)
    max_offset = np.full(n, np.nan)
    from scipy.optimize import minimize_scalar
    for i in range(n):
        for name in names:
            ref = float(contacts[name])
            fn = lambda jd, name=name: _lunar_contact_equation(
                jd, contact=name, time_bias_s=time_bias[i],
                axis_delta_deg=axis_delta_deg[i], shadow_delta_deg=shadow_delta_deg[i],
                moon_delta_deg=moon_delta_deg[i], event=event,
            )
            root = _root_near(fn, ref, initial_half_width_s=120.0)
            offsets[name][i] = (root - ref) * 86400.0 if math.isfinite(root) else np.nan
        solved = {name: contacts[name] + offsets[name][i] / 86400.0 for name in names if math.isfinite(offsets[name][i])}
        if all(k in solved for k in ("U2", "U3")):
            totality[i] = (solved["U3"] - solved["U2"]) * 86400.0
        if all(k in solved for k in ("U1", "U4")):
            partial[i] = (solved["U4"] - solved["U1"]) * 86400.0
        if all(k in solved for k in ("P1", "P4")):
            penumbral[i] = (solved["P4"] - solved["P1"]) * 86400.0

        center_jd = float(event.greatest_jd)
        lo_s = (contacts["U2"] - center_jd) * 86400.0
        hi_s = (contacts["U3"] - center_jd) * 86400.0
        def rho_offset(offset_s):
            evaluation_jd = center_jd + float(offset_s) / 86400.0 + float(time_bias[i]) / 86400.0
            state = lunar_reference_state(evaluation_jd)
            impact = np.asarray(state.impact_vector_deg); ih = impact / np.linalg.norm(impact)
            track_hat = np.array([ih[1], -ih[0]])
            return float(np.linalg.norm(impact + _lunar_track_x_extended(evaluation_jd, event) * track_hat + axis_delta_deg[i]))
        opt = minimize_scalar(rho_offset, bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1.0e-4})
        r = float(opt.fun); max_offset[i] = r
        ru = float(event.metadata["umbra_radius_deg"]) + shadow_delta_deg[i]
        rp = float(event.metadata["penumbra_radius_deg"]) + shadow_delta_deg[i]
        rm = float(event.metadata["moon_semidiameter_deg"]) + moon_delta_deg[i]
        umbral_mag[i] = (ru + rm - r) / (2.0 * rm)
        penumbral_mag[i] = (rp + rm - r) / (2.0 * rm)

    return {
        "$schema": "ssapy-toolkit.eclipse.uncertainty/1.9.3",
        "kind": "lunar", "event": LUNAR_2025.key, "samples": n, "seed": int(seed),
        "budget": assumptions.to_dict(),
        "interpretation": "Conditional Monte Carlo intervals under the recorded independent Gaussian budget; atmospheric shadow-model discrepancy should be added separately.",
        "nominal": {
            "greatest_jd_utc": float(event.greatest_jd),
            "contacts_jd": {k: float(v) for k, v in contacts.items()},
            "umbral_magnitude": float(event.metadata["umbral_magnitude"]),
            "penumbral_magnitude": float(event.metadata["penumbral_magnitude"]),
        },
        "outputs": {
            "contact_offset_s": {name: IntervalSummary.from_values(values).to_dict() for name, values in offsets.items()},
            "totality_duration_s": IntervalSummary.from_values(totality).to_dict(),
            "partial_umbral_duration_s": IntervalSummary.from_values(partial).to_dict(),
            "penumbral_duration_s": IntervalSummary.from_values(penumbral).to_dict(),
            "umbral_magnitude": IntervalSummary.from_values(umbral_mag).to_dict(),
            "penumbral_magnitude": IntervalSummary.from_values(penumbral_mag).to_dict(),
            "shadow_axis_offset_deg": IntervalSummary.from_values(max_offset).to_dict(),
        },
        "raw": {
            "totality_duration_s": totality.tolist(), "partial_umbral_duration_s": partial.tolist(),
            "penumbral_duration_s": penumbral.tolist(), "umbral_magnitude": umbral_mag.tolist(),
            "penumbral_magnitude": penumbral_mag.tolist(),
            **{f"{name.lower()}_offset_s": values.tolist() for name, values in offsets.items()},
        },
    }


def propagate_uncertainty(kind: str, **kwargs) -> dict[str, object]:
    key = str(kind).lower()
    if key == "solar":
        return propagate_solar_uncertainty(**kwargs)
    if key == "lunar":
        return propagate_lunar_uncertainty(**kwargs)
    raise ValueError("kind must be 'solar' or 'lunar'")


def write_uncertainty_report(path: str | Path, kind: str, **kwargs) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = propagate_uncertainty(kind, **kwargs)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "UncertaintyBudget", "IntervalSummary", "propagate_solar_uncertainty",
    "propagate_lunar_uncertainty", "propagate_uncertainty",
    "write_uncertainty_report",
]
