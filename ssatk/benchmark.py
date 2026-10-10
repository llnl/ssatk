"""Benchmark SSATK functions and build an HTML timing dashboard.

The benchmark registry intentionally uses representative, copyable calls rather
than trying to call every public function with guessed arguments. Add new
``BenchmarkCase`` entries in ``build_benchmark_cases`` when new public workflows
need timing coverage.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import html
import io
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ssatk import __version__
from ssatk.plots.figpath import figpath

CallFactory = Callable[["BenchmarkContext"], Callable[[], Any]]
ValidationScalar = int | float | str | bool
ValidationMetric = dict[str, ValidationScalar]
Validation = dict[str, ValidationMetric]
Validator = Callable[[Any], Validation]


@dataclass(frozen=True)
class BenchmarkContext:
    """Shared immutable inputs available to benchmark factories."""

    output_dir: Path
    rng_seed: int = 12345

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed + int(offset))


@dataclass(frozen=True)
class BenchmarkCase:
    """A single timed SSATK function scenario."""

    name: str
    group: str
    description: str
    factory: CallFactory
    tags: tuple[str, ...] = field(default_factory=tuple)
    default_repeats: int | None = None
    default_min_sample_time: float | None = None
    validator: Validator | None = None
    validation_settings: dict[str, ValidationScalar] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Serializable timing result for one benchmark case."""

    name: str
    group: str
    description: str
    tags: tuple[str, ...]
    success: bool
    repeats: int
    warmups: int
    loops_per_repeat: int
    total_sample_time_s: float
    mean_s: float | None = None
    median_s: float | None = None
    stdev_s: float | None = None
    min_s: float | None = None
    max_s: float | None = None
    p05_s: float | None = None
    p25_s: float | None = None
    p75_s: float | None = None
    p95_s: float | None = None
    p99_s: float | None = None
    iqr_s: float | None = None
    cv: float | None = None
    hz: float | None = None
    peak_memory_bytes: int | None = None
    error: str | None = None
    traceback: str | None = None
    validation: Validation | None = None
    validation_pass: bool | None = None
    validation_settings: dict[str, ValidationScalar] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["tags"] = list(self.tags)
        return _json_safe(data)


# ---------------------------------------------------------------------------
# Benchmark case factories


def _vector_inputs(rows: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(10)
    return rng.normal(size=(rows, 3))


def _circular_state(radius: float, theta: float = 0.0, inclination: float = 0.0, t: float = 0.0):
    from ssatk.constants import EARTH_MU

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array(
        [-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc]
    )
    return r, v, t


def _optimal_transfer_states(tof: float = 3600.0):
    from ssatk.constants import EARTH_MU

    radius1 = 7000e3
    radius2 = 9000e3
    inclination = np.deg2rad(8.0)
    target_initial_theta = 0.35
    target_theta = target_initial_theta + np.sqrt(EARTH_MU / radius2**3) * tof
    r1, v1, _ = _circular_state(radius1)
    r2, v2, _ = _circular_state(radius2, theta=target_theta, inclination=inclination, t=tof)
    return r1, v1, r2, v2, tof


def _close_matplotlib_figures() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def build_benchmark_cases(
    *,
    include_io: bool = True,
    include_plots: bool = True,
    include_slow: bool = False,
) -> list[BenchmarkCase]:
    """Return the default representative SSATK benchmark registry."""

    array = _vector_inputs()
    angles = np.linspace(-10.0, 10.0, 5000)
    positions = np.column_stack(
        [
            7000e3 * np.cos(np.linspace(0.0, 2.0 * np.pi, 120)),
            7000e3 * np.sin(np.linspace(0.0, 2.0 * np.pi, 120)),
            np.zeros(120),
        ]
    )
    r1, v1, r2, v2, tof = _optimal_transfer_states()
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    times = np.linspace(0.0, 30.0, 32)

    cases: list[BenchmarkCase] = [
        BenchmarkCase(
            name="vectors.norm_matrix",
            group="vectors",
            description="Vector norm over a 5000x3 array.",
            factory=lambda _ctx: _call(lambda: _vectors_norm(array)),
            tags=("array", "core"),
        ),
        BenchmarkCase(
            name="vectors.normed_matrix",
            group="vectors",
            description="Row normalization over a 5000x3 array.",
            factory=lambda _ctx: _call(lambda: _vectors_normed(array)),
            tags=("array", "core"),
        ),
        BenchmarkCase(
            name="vectors.einsum_norm_matrix",
            group="vectors",
            description="Einsum norm over a 5000x3 array.",
            factory=lambda _ctx: _call(lambda: _vectors_einsum_norm(array)),
            tags=("array", "core"),
        ),
        BenchmarkCase(
            name="vectors.rotate_points_3d",
            group="vectors",
            description="Rotate 5000 3-D points about the default axis.",
            factory=lambda _ctx: _call(lambda: _vectors_rotate_points_3d(array)),
            tags=("array", "geometry"),
        ),
        BenchmarkCase(
            name="coordinates.rad0to2pi_array",
            group="coordinates",
            description="Wrap 5000 radian values into [0, 2pi).",
            factory=lambda _ctx: _call(lambda: _coordinates_rad0to2pi(angles)),
            tags=("array", "core"),
        ),
        BenchmarkCase(
            name="coordinates.deg90to90_array",
            group="coordinates",
            description="Wrap 5000 degree values into [-90, 90].",
            factory=lambda _ctx: _call(lambda: _coordinates_deg90to90array(np.rad2deg(angles))),
            tags=("array", "core"),
        ),
        BenchmarkCase(
            name="coordinates.cart2sph_deg_scalar",
            group="coordinates",
            description="Single Cartesian-to-spherical coordinate conversion.",
            factory=lambda _ctx: _call(lambda: _coordinates_cart2sph_deg(1.0, 2.0, 3.0)),
            tags=("scalar", "geometry"),
        ),
        BenchmarkCase(
            name="coordinates.cart_to_cyl_scalar",
            group="coordinates",
            description="Single Cartesian-to-cylindrical coordinate conversion.",
            factory=lambda _ctx: _call(lambda: _coordinates_cart_to_cyl(1.0, 2.0, 3.0)),
            tags=("scalar", "geometry"),
        ),
        BenchmarkCase(
            name="time.dd_to_hms_scalar",
            group="time",
            description="Convert decimal degrees to an HMS string.",
            factory=lambda _ctx: _call(lambda: _time_dd_to_hms(123.456)),
            tags=("scalar", "formatting"),
        ),
        BenchmarkCase(
            name="time.hms_to_dd_scalar",
            group="time",
            description="Convert an HMS string to decimal degrees.",
            factory=lambda _ctx: _call(lambda: _time_hms_to_dd("12:30:00")),
            tags=("scalar", "formatting"),
        ),
        BenchmarkCase(
            name="compute.generate_sphere_vectors_1000",
            group="compute",
            description="Generate 1000 uniformly distributed 3-D vectors.",
            factory=lambda _ctx: _call(lambda: _compute_generate_sphere_vectors(1000, 1.0, seed=1)),
            tags=("random", "array"),
        ),
        BenchmarkCase(
            name="compute.lambert_sphere_phase_array",
            group="compute",
            description="Evaluate Lambertian phase over 5000 phase angles.",
            factory=lambda _ctx: _call(lambda: _compute_lambert_sphere_phase(np.linspace(0.0, np.pi, 5000))),
            tags=("photometry", "array"),
        ),
        BenchmarkCase(
            name="compute.airmass_kasten_young_array",
            group="compute",
            description="Evaluate Kasten-Young airmass over 5000 zenith angles.",
            factory=lambda _ctx: _call(lambda: _compute_airmass_kasten_young(np.linspace(0.0, 80.0, 5000))),
            tags=("photometry", "array"),
        ),
        BenchmarkCase(
            name="accelerations_orbit.accel_point_earth",
            group="accelerations_orbit",
            description="Point-Earth gravity acceleration for one position vector.",
            factory=lambda _ctx: _call(lambda: _accel_point_earth(r0)),
            tags=("propagators_6dof", "scalar"),
        ),
        BenchmarkCase(
            name="propagators_orbit.leapfrog_32_steps",
            group="propagators_orbit",
            description="Leapfrog propagation over 32 short time steps.",
            factory=lambda _ctx: _call(lambda: _propagator_leapfrog(r0, v0, times)),
            tags=("propagators_6dof", "propagation"),
            default_min_sample_time=0.01,
        ),
        BenchmarkCase(
            name="propagators_orbit.dop853_32_steps",
            group="propagators_orbit",
            description="Adaptive DOP853 translational propagation over 32 short time steps.",
            factory=lambda _ctx: _call(lambda: _propagator_dop853(r0, v0, times)),
            validator=_validate_orbit_trajectory,
            tags=("propagators_orbit", "propagation", "dop853"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_orbit.dop853_stm_32_steps",
            group="propagators_orbit",
            description="Adaptive DOP853 translational propagation with a 6x6 STM.",
            factory=lambda _ctx: _call(lambda: _propagator_dop853_stm(r0, v0, times)),
            validator=_validate_orbit_stm,
            tags=("propagators_orbit", "propagation", "stm", "dop853"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_6dof.point_mass_32_steps",
            group="propagators_6dof",
            description="6-DoF high-accuracy point-mass propagation over 32 short time steps.",
            factory=lambda _ctx: _call(lambda: _propagator_6dof_point_mass(r0, v0, times)),
            validator=_validate_6dof_point_mass,
            tags=("propagators_6dof", "propagation", "6dof"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_6dof.thruster_mass_32_steps",
            group="propagators_6dof",
            description="6-DoF spacecraft propagation with body thruster torque and mass depletion.",
            factory=lambda _ctx: _call(lambda: _propagator_6dof_thruster_mass(times)),
            validator=_validate_6dof_trajectory,
            tags=("propagators_6dof", "propulsion", "6dof"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_6dof.reaction_wheel_32_steps",
            group="propagators_6dof",
            description="6-DoF attitude propagation with reaction-wheel momentum states.",
            factory=lambda _ctx: _call(lambda: _propagator_6dof_reaction_wheel(times)),
            validator=_validate_6dof_trajectory,
            tags=("propagators_6dof", "attitude", "6dof"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_6dof.environment_facet_32_steps",
            group="propagators_6dof",
            description="6-DoF spacecraft propagation with environment-backed facet SRP and drag.",
            factory=lambda _ctx: _call(lambda: _propagator_6dof_environment(times)),
            validator=_validate_6dof_trajectory,
            tags=("propagators_6dof", "environment", "6dof"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="propagators_6dof.articulated_facet_32_steps",
            group="propagators_6dof",
            description="6-DoF spacecraft propagation with prescribed rotating SRP facets.",
            factory=lambda _ctx: _call(lambda: _propagator_6dof_articulated_facet(times)),
            validator=_validate_6dof_trajectory,
            tags=("propagators_6dof", "environment", "articulated", "6dof"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="orbital.kepler_to_state",
            group="orbital_mechanics",
            description="Convert Keplerian elements to Cartesian state.",
            factory=lambda _ctx: _call(
                lambda: _orbital_kepler_to_state(a=7000e3, e=0.001, i=0.1, pa=0.2, raan=0.3, nu=0.4)
            ),
            tags=("keplerian", "core"),
        ),
        BenchmarkCase(
            name="orbital.state_to_kepler",
            group="orbital_mechanics",
            description="Convert Cartesian state to Keplerian elements.",
            factory=lambda _ctx: _state_to_kepler_case(),
            tags=("keplerian", "core"),
        ),
        BenchmarkCase(
            name="orbital.transfer_hohmann",
            group="orbital_mechanics",
            description="Analytic Hohmann transfer between circular radii.",
            factory=lambda _ctx: _call(lambda: _orbital_transfer_hohmann(7000e3, 9000e3, samples=60)),
            tags=("transfer", "analytic"),
            default_min_sample_time=0.01,
        ),
        BenchmarkCase(
            name="orbital.transfer_bielliptic",
            group="orbital_mechanics",
            description="Analytic bi-elliptic transfer with three sampled arcs.",
            factory=lambda _ctx: _call(
                lambda: _orbital_transfer_bielliptic(
                    radius1=7000e3,
                    radius2=9000e3,
                    rb=12000e3,
                    samples_per_arc=30,
                    plot=False,
                )
            ),
            tags=("transfer", "analytic"),
            default_min_sample_time=0.01,
        ),
        BenchmarkCase(
            name="orbital.transfer_optimal_direct",
            group="orbital_mechanics",
            description="One-cell direct transfer_optimal search with fixed boundary states.",
            factory=lambda _ctx: _call(
                lambda: _orbital_transfer_optimal(
                    r1,
                    v1,
                    r2,
                    v2,
                    t2=tof,
                    departure_mode="now",
                    tof_range=(tof, tof),
                    n_grid=(1, 1),
                    polish=False,
                    propagate=False,
                    refine=False,
                    burn_duration=1.0,
                )
            ),
            tags=("transfer", "search"),
            default_repeats=3,
            default_min_sample_time=0.0,
        ),
        BenchmarkCase(
            name="validation.two_body_invariants",
            group="validation",
            description="Analytical specific-energy and angular-momentum invariants for one two-body orbit.",
            factory=lambda _ctx: _call(_propagator_two_body_invariants),
            validator=_validate_two_body_invariants,
            tags=("validation", "analytical", "orbit"),
            default_repeats=1,
            default_min_sample_time=0.0,
            validation_settings={"method": "DOP853", "rtol": 1.0e-10, "atol": 1.0e-8, "max_step": 60.0},
        ),
        BenchmarkCase(
            name="validation.j2_raan_drift",
            group="validation",
            description="First-order secular J2 nodal-precession check over 20 near-circular-orbit periods.",
            factory=lambda _ctx: _call(_propagator_j2_raan_drift),
            validator=_validate_j2_raan_drift,
            tags=("validation", "analytical", "J2"),
            default_repeats=1,
            default_min_sample_time=0.0,
            validation_settings={
                "method": "DOP853", "rtol": 1.0e-10, "atol": 1.0e-9, "max_step": 120.0,
            },
        ),
        BenchmarkCase(
            name="validation.torque_free_invariants",
            group="validation",
            description="Analytical angular-momentum and energy invariants for torque-free rigid-body motion.",
            factory=lambda _ctx: _call(_propagator_torque_free_invariants),
            validator=_validate_torque_free_invariants,
            tags=("validation", "analytical", "6dof"),
            default_repeats=1,
            default_min_sample_time=0.0,
            validation_settings={
                "method": "DOP853", "rtol": 1.0e-10, "atol": 1.0e-9,
                "max_step": 1.0, "mu": 0.0,
            },
        ),
        BenchmarkCase(
            name="validation.finite_burn_mass_delta_v_impulse",
            group="validation",
            description="Constant finite-burn mass, delta-v, and impulse checks against analytical equations.",
            factory=lambda _ctx: _call(_propagator_finite_burn_analytic),
            validator=_validate_finite_burn_analytic,
            tags=("validation", "analytical", "finite-burn"),
            default_repeats=1,
            default_min_sample_time=0.0,
            validation_settings={
                "method": "DOP853", "rtol": 1.0e-10, "atol": 1.0e-9, "mu": 0.0,
                "impulse_samples": 1001,
            },
        ),
    ]

    if include_io:
        data = {
            "time_s": np.linspace(0.0, 1.0, 256).tolist(),
            "position_m": _vector_inputs(256).tolist(),
        }
        cases.extend(
            [
                BenchmarkCase(
                    name="io.ssatk_save_json",
                    group="io",
                    description="Write a small nested dict to JSON with ssatk_save.",
                    factory=lambda ctx: _call(lambda: _io_ssatk_save_json(data, ctx.output_dir)),
                    tags=("io", "json"),
                    default_min_sample_time=0.0,
                ),
                BenchmarkCase(
                    name="io.ssatk_read_json",
                    group="io",
                    description="Read a small nested dict from JSON with ssatk_read.",
                    factory=lambda ctx: _io_load_json_case(data, ctx.output_dir),
                    tags=("io", "json"),
                    default_min_sample_time=0.0,
                ),
            ]
        )

    if include_plots:
        cases.append(
            BenchmarkCase(
                name="plots.orbit_plot_view_xy",
                group="plots",
                description="Render a 2-D orbit_plot view without saving the figure.",
                factory=lambda _ctx: _call(lambda: _plot_orbit_xy(positions)),
                tags=("plotting", "matplotlib"),
                default_repeats=3,
                default_min_sample_time=0.0,
            )
        )

    if include_slow:
        cases.extend(
            [
                BenchmarkCase(
                    name="propagators_orbit.rk4_32_steps",
                    group="propagators_orbit",
                    description="RK4 propagation over 32 short time steps including third-body terms.",
                    factory=lambda _ctx: _call(lambda: _propagator_rk4(r0, v0, times)),
                    tags=("propagators_6dof", "propagation", "slow"),
                    default_repeats=3,
                    default_min_sample_time=0.0,
                ),
                BenchmarkCase(
                    name="orbital.transfer_optimal_grid_3x3",
                    group="orbital_mechanics",
                    description="Direct transfer_optimal search over a 3x3 grid.",
                    factory=lambda _ctx: _call(
                        lambda: _orbital_transfer_optimal(
                            r1,
                            v1,
                            r2,
                            v2,
                            t2=tof,
                            departure_mode="now",
                            tof_range=(2400.0, tof),
                            n_grid=(1, 3),
                            polish=False,
                            propagate=False,
                            refine=False,
                            burn_duration=1.0,
                        )
                    ),
                    tags=("transfer", "search", "slow"),
                    default_repeats=3,
                    default_min_sample_time=0.0,
                ),
            ]
        )

    return cases


# Import targets lazily through wrappers so importing this module stays cheap.


def _call(func: Callable[[], Any]) -> Callable[[], Any]:
    return func


def _vectors_norm(array):
    from ssatk.vectors import norm

    return norm(array)


def _vectors_normed(array):
    from ssatk.vectors import normed

    return normed(array)


def _vectors_einsum_norm(array):
    from ssatk.vectors import einsum_norm

    return einsum_norm(array, "ij,ij->i")


def _vectors_rotate_points_3d(array):
    from ssatk.vectors import rotate_points_3d

    return rotate_points_3d(array)


def _coordinates_rad0to2pi(angles):
    from ssatk.coordinates.angle_units import rad0to2pi

    return rad0to2pi(angles)


def _coordinates_deg90to90array(angles):
    from ssatk.coordinates.angle_units import deg90to90array

    return deg90to90array(angles)


def _coordinates_cart2sph_deg(x, y, z):
    from ssatk.coordinates.cartesian import cart2sph_deg

    return cart2sph_deg(x, y, z)


def _coordinates_cart_to_cyl(x, y, z):
    from ssatk.coordinates.cartesian import cart_to_cyl

    return cart_to_cyl(x, y, z)


def _time_dd_to_hms(value):
    from ssatk.time_functions.convert_dd_and_hms import dd_to_hms

    return dd_to_hms(value)


def _time_hms_to_dd(value):
    from ssatk.time_functions.convert_dd_and_hms import hms_to_dd

    return hms_to_dd(value)


def _compute_generate_sphere_vectors(n, magnitude, seed):
    from ssatk.compute.generate_sphere_of_vectors import generate_sphere_vectors

    return generate_sphere_vectors(n, magnitude, seed=seed)


def _compute_lambert_sphere_phase(alpha):
    from ssatk.compute.lambertian_magnitude import lambert_sphere_phase

    return lambert_sphere_phase(alpha)


def _compute_airmass_kasten_young(zenith_deg):
    from ssatk.compute.lambertian_magnitude import airmass_kasten_young

    return airmass_kasten_young(zenith_deg)


def _accel_point_earth(r):
    from ssatk.accelerations_orbit.accel_point_earth import accel_point_earth

    return accel_point_earth(r)


def _propagator_leapfrog(r0, v0, t):
    from ssatk.propagators_orbit.leap_frog import leapfrog

    return leapfrog(r0, v0, t)


def _propagator_dop853(r0, v0, t):
    from ssatk.propagators_orbit import propagate_orbit_state

    return propagate_orbit_state(r0=r0, v0=v0, times=t)


def _propagator_dop853_stm(r0, v0, t):
    from ssatk.propagators_orbit import propagate_orbit_state_with_stm

    return propagate_orbit_state_with_stm(r0=r0, v0=v0, times=t)


def _validation_metric(
    value: ValidationScalar,
    unit: str,
    tolerance: ValidationScalar,
    reference: str,
    *,
    passed: bool | None = None,
) -> ValidationMetric:
    if passed is None:
        passed = isinstance(value, (int, float)) and abs(float(value)) <= float(tolerance)
    return {
        "value": value,
        "unit": unit,
        "tolerance": tolerance,
        "pass": bool(passed),
        "reference": reference,
    }


def _validation_passed(validation: Validation | None) -> bool | None:
    if validation is None:
        return None
    return all(bool(metric["pass"]) for metric in validation.values())


def _validate_orbit_trajectory(trajectory) -> Validation:
    state = np.concatenate((trajectory.r.ravel(), trajectory.v.ravel()))
    return {
        "nfev": _validation_metric(
            int(trajectory.nfev), "count", 0, "DOP853 reports a positive evaluation count.",
            passed=trajectory.nfev > 0,
        ),
        "finite_state_residual": _validation_metric(
            1.0 if np.any(~np.isfinite(state)) else 0.0,
            "1", 0.0, "finite propagated position and velocity",
        ),
    }


def _validate_orbit_stm(trajectory) -> Validation:
    metrics = _validate_orbit_trajectory(trajectory)
    metrics["finite_stm_residual"] = _validation_metric(
        1.0 if np.any(~np.isfinite(trajectory.stm)) else 0.0,
        "1", 0.0, "finite state-transition matrix",
    )
    metrics["initial_stm_residual"] = _validation_metric(
        float(np.max(np.abs(trajectory.stm[0] - np.eye(6)))),
        "1", 1.0e-12, "initial STM equals the 6x6 identity",
    )
    return metrics


def _propagator_rk4(r0, v0, t):
    from ssatk.propagators_orbit.rk4 import rk4

    return rk4(r0, v0, t)


def _propagator_6dof_point_mass(r0, v0, times):
    from ssatk.propagators_6dof import propagate_6dof_high_accuracy

    return propagate_6dof_high_accuracy(
        r0=r0,
        v0=v0,
        times=times,
        inertia=np.diag([10.0, 12.0, 8.0]),
        omega0=[0.0, 0.0, 1e-3],
    )


def _propagator_6dof_thruster_mass(times):
    from ssatk.accelerations_6dof import SpacecraftThrusterAccel
    from ssatk.propagators_6dof import Spacecraft
    from ssatk.propagators_6dof import propagate_spacecraft_high_accuracy
    from ssatk.satellites import SpacecraftBody, Thruster

    body = SpacecraftBody.box(name="benchmark_bus", mass=25.0, size=(1.0, 1.0, 1.0)).with_thrusters(
        Thruster(thrust=0.5, direction_body=[1.0, 0.0, 0.0], position_body=[0.0, 0.2, 0.0], isp=220.0),
        append=False,
    )
    spacecraft = Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], body=body)
    return propagate_spacecraft_high_accuracy(
        spacecraft,
        times=times,
        models=[SpacecraftThrusterAccel()],
        mu=0.0,
    )


def _propagator_6dof_reaction_wheel(times):
    from ssatk.accelerations_6dof import SpacecraftReactionWheelTorque
    from ssatk.propagators_6dof import Spacecraft
    from ssatk.propagators_6dof import propagate_spacecraft_high_accuracy
    from ssatk.satellites import SpacecraftBody, reaction_wheel_triplet

    body = SpacecraftBody(
        name="benchmark_wheel_bus",
        mass=20.0,
        inertia=np.diag([4.0, 5.0, 6.0]),
    ).with_reaction_wheels(
        *reaction_wheel_triplet(max_torque=0.02, momentum_capacity=0.2, wheel_inertia=0.01)
    )
    spacecraft = Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], body=body)
    return propagate_spacecraft_high_accuracy(
        spacecraft,
        times=times,
        models=[SpacecraftReactionWheelTorque([0.0, 0.0, 0.01])],
        mu=0.0,
    )


def _propagator_6dof_environment(times):
    from ssatk.constants import AU, EARTH_RADIUS
    from ssatk.propagators_6dof import Spacecraft
    from ssatk.environment import SpaceEnvironment
    from ssatk.propagators_6dof import propagate_spacecraft_high_accuracy
    from ssatk.satellites import SpacecraftBody

    body = SpacecraftBody.box(name="benchmark_plate", mass=50.0, size=(1.0, 1.0, 1.0))
    spacecraft = Spacecraft(
        r=[EARTH_RADIUS + 400_000.0, 0.0, 0.0],
        v=[0.0, 7_700.0, 0.0],
        body=body,
    )
    environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        atmosphere_density_model=1e-12,
        eclipse_model=None,
    )
    return propagate_spacecraft_high_accuracy(
        spacecraft,
        times=times,
        environment=environment,
        environment_models={"drag": True, "solar_radiation": True},
        mu=0.0,
    )


def _propagator_6dof_articulated_facet(times):
    from ssatk.accelerations_6dof import SpacecraftFacetSolRad
    from ssatk.constants import AU, EARTH_RADIUS
    from ssatk.propagators_6dof import Spacecraft
    from ssatk.propagators_6dof import propagate_spacecraft_high_accuracy
    from ssatk.satellites import SpacecraftBody, rotate_facets

    body = SpacecraftBody.box_wing(
        name="benchmark_articulated",
        mass=60.0,
        bus_size=(1.0, 1.0, 1.0),
        solar_array_area=4.0,
        solar_array_axis="y",
        solar_array_mass=5.0,
    )

    def track_panel(facets, *, t, **_kwargs):
        return rotate_facets(facets, axis_body=(0.0, 0.0, 1.0), angle_rad=0.02 * t)

    spacecraft = Spacecraft(
        r=[EARTH_RADIUS + 700_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        body=body,
    )
    return propagate_spacecraft_high_accuracy(
        spacecraft,
        times=times,
        models=[
            SpacecraftFacetSolRad(
                [AU, 0.0, 0.0],
                body=body,
                facet_transform=track_panel,
            )
        ],
        mu=0.0,
    )


def _validate_6dof_trajectory(trajectory) -> Validation:
    """Return stable physical checks for machine-readable 6-DoF results."""
    state = np.concatenate(
        (trajectory.r.ravel(), trajectory.v.ravel(), trajectory.q.ravel(), trajectory.omega.ravel())
    )
    quaternion_residual = float(np.max(np.abs(np.linalg.norm(trajectory.q, axis=1) - 1.0)))
    mass = getattr(trajectory, "mass", None)
    mass_residual = 0.0 if mass is None else float(max(0.0, -np.min(mass)))
    return {
        "nfev": _validation_metric(
            int(trajectory.nfev), "count", 0, "DOP853 reports a positive evaluation count.",
            passed=trajectory.nfev > 0,
        ),
        "finite_state_residual": _validation_metric(
            1.0 if np.any(~np.isfinite(state)) else 0.0,
            "1", 0.0, "finite propagated 6-DoF state",
        ),
        "quaternion_norm_residual": _validation_metric(
            quaternion_residual, "1", 1.0e-10, "unit quaternion norm",
        ),
        "negative_mass_residual": _validation_metric(
            mass_residual, "kg", 0.0, "mass remains non-negative",
        ),
    }


def _validate_6dof_point_mass(trajectory) -> Validation:
    """Add an independent two-body residual to the point-mass benchmark."""
    import rebound

    from ssatk.constants import EARTH_MU

    simulation = rebound.Simulation()
    simulation.G = EARTH_MU
    simulation.add(m=1.0)
    simulation.add(
        x=float(trajectory.r[0, 0]),
        y=float(trajectory.r[0, 1]),
        z=float(trajectory.r[0, 2]),
        vx=float(trajectory.v[0, 0]),
        vy=float(trajectory.v[0, 1]),
        vz=float(trajectory.v[0, 2]),
    )
    simulation.integrator = "ias15"
    for time in trajectory.t:
        simulation.integrate(float(time - trajectory.t[0]), exact_finish_time=1)
    particle = simulation.particles[1]
    reference_r = np.array([particle.x, particle.y, particle.z])
    reference_v = np.array([particle.vx, particle.vy, particle.vz])
    metrics = _validate_6dof_trajectory(trajectory)
    metrics["final_position_residual_m"] = _validation_metric(
        float(np.linalg.norm(trajectory.r[-1] - reference_r)),
        "m", 1.0e-3, "REBOUND IAS15 two-body propagation",
    )
    metrics["final_velocity_residual_mps"] = _validation_metric(
        float(np.linalg.norm(trajectory.v[-1] - reference_v)),
        "m/s", 1.0e-6, "REBOUND IAS15 two-body propagation",
    )
    return metrics


def _propagator_two_body_invariants():
    from ssatk.constants import EARTH_MU
    from ssatk.propagators_orbit import propagate_orbit_state

    radius = 7_000e3
    speed = np.sqrt(EARTH_MU / radius)
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    return propagate_orbit_state(
        r0=[radius, 0.0, 0.0],
        v0=[0.0, speed, 0.0],
        times=np.linspace(0.0, 10.0 * period, 201),
        rtol=1.0e-10,
        atol=1.0e-8,
        max_step=60.0,
    )


def _validate_two_body_invariants(trajectory) -> Validation:
    from ssatk.constants import EARTH_MU

    specific_energy = 0.5 * np.sum(trajectory.v**2, axis=1) - EARTH_MU / np.linalg.norm(trajectory.r, axis=1)
    angular_momentum = np.cross(trajectory.r, trajectory.v)
    angular_momentum_scale = np.linalg.norm(angular_momentum[0])
    return {
        "specific_energy_residual": _validation_metric(
            float(np.max(np.abs(specific_energy - specific_energy[0]))),
            "J/kg", 2.0e-6, "analytic two-body specific energy",
        ),
        "angular_momentum_residual": _validation_metric(
            float(np.max(np.linalg.norm(angular_momentum - angular_momentum[0], axis=1)) / angular_momentum_scale),
            "1", 1.0e-12, "analytic two-body angular momentum magnitude",
        ),
    }


def _propagator_j2_raan_drift():
    from ssatk.accelerations_6dof import SpacecraftAccelJ2
    from ssatk.constants import EARTH_MU
    from ssatk.orbital_mechanics import kepler_to_state
    from ssatk.propagators_6dof import propagate_6dof_high_accuracy

    semimajor_axis = 7_000e3
    eccentricity = 0.01
    inclination = np.deg2rad(63.0)
    r0, v0 = kepler_to_state(semimajor_axis, eccentricity, inclination, 0.2, 0.3, 0.1)
    period = 2.0 * np.pi * np.sqrt(semimajor_axis**3 / EARTH_MU)
    return propagate_6dof_high_accuracy(
        r0=r0,
        v0=v0,
        times=np.linspace(0.0, 20.0 * period, 401),
        inertia=np.eye(3),
        acceleration=SpacecraftAccelJ2(),
        max_step=120.0,
    )


def _raan(trajectory):
    h = np.cross(trajectory.r, trajectory.v)
    node = np.column_stack((-h[:, 1], h[:, 0]))
    return np.unwrap(np.arctan2(node[:, 1], node[:, 0]))


def _validate_j2_raan_drift(trajectory) -> Validation:
    from ssatk.constants import EARTH_MU, EARTH_RADIUS, J2_wgs

    semimajor_axis = 7_000e3
    eccentricity = 0.01
    inclination = np.deg2rad(63.0)
    rate = -1.5 * J2_wgs * np.sqrt(EARTH_MU / semimajor_axis**3)
    rate *= (EARTH_RADIUS / (semimajor_axis * (1.0 - eccentricity**2))) ** 2 * np.cos(inclination)
    actual_drift = _raan(trajectory)[-1] - _raan(trajectory)[0]
    expected_drift = rate * (trajectory.t[-1] - trajectory.t[0])
    return {
        "raan_drift_residual": _validation_metric(
            float(abs(actual_drift - expected_drift)), "rad", 3.0e-4,
            "first-order secular J2 nodal-precession drift over 20 periods",
        ),
    }


def _propagator_torque_free_invariants():
    from ssatk.propagators_6dof import propagate_6dof_high_accuracy

    return propagate_6dof_high_accuracy(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=np.linspace(0.0, 100.0, 101),
        inertia=np.diag([2.0, 3.0, 4.0]),
        q0=[1.0, 0.0, 0.0, 0.0],
        omega0=[0.1, 0.2, 0.3],
        mu=0.0,
        max_step=1.0,
    )


def _validate_torque_free_invariants(trajectory) -> Validation:
    from ssatk.coordinates.attitude import rotate_vector

    inertia = np.diag([2.0, 3.0, 4.0])
    angular_momentum = np.array([
        rotate_vector(q, inertia @ omega)
        for q, omega in zip(trajectory.q, trajectory.omega)
    ])
    kinetic_energy = 0.5 * np.einsum("ij,jk,ik->i", trajectory.omega, inertia, trajectory.omega)
    angular_momentum_scale = np.linalg.norm(angular_momentum[0])
    return {
        "angular_momentum_residual": _validation_metric(
            float(np.max(np.linalg.norm(angular_momentum - angular_momentum[0], axis=1)) / angular_momentum_scale),
            "1", 1.0e-9, "torque-free inertial angular-momentum invariant",
        ),
        "rotational_energy_residual": _validation_metric(
            float(np.max(np.abs(kinetic_energy - kinetic_energy[0]))),
            "J", 1.0e-10, "torque-free rotational-energy invariant",
        ),
    }


def _propagator_finite_burn_analytic():
    from ssatk.accelerations_6dof import (
        SpacecraftManeuverAccel,
        integrated_thrust_impulse,
        thrust_profile_constant,
    )
    from ssatk.propagators_6dof import Spacecraft
    from ssatk.propagators_6dof import propagate_spacecraft_high_accuracy

    profile = thrust_profile_constant(2.0, start=0.0, stop=10.0)
    model = SpacecraftManeuverAccel(
        profile, direction=[1.0, 0.0, 0.0], frame="inertial", isp=200.0,
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], inertia=np.eye(3), mass=10.0,
    )
    trajectory = propagate_spacecraft_high_accuracy(
        spacecraft,
        times=np.linspace(0.0, 10.0, 101), models=[model], mu=0.0,
    )
    return trajectory, integrated_thrust_impulse(profile, 0.0, 10.0)


def _validate_finite_burn_analytic(trajectory) -> Validation:
    from ssatk.constants import STANDARD_GRAVITY

    trajectory, measured_impulse = trajectory
    thrust = 2.0
    burn_time = 10.0
    initial_mass = 10.0
    isp = 200.0
    mass_flow = thrust / (isp * STANDARD_GRAVITY)
    expected_mass = initial_mass - mass_flow * burn_time
    expected_impulse = thrust * burn_time
    expected_delta_v = isp * STANDARD_GRAVITY * np.log(initial_mass / expected_mass)
    actual_delta_v = float(np.linalg.norm(trajectory.v[-1] - trajectory.v[0]))
    return {
        "mass_residual": _validation_metric(
            float(abs(trajectory.mass[-1] - expected_mass)), "kg", 1.0e-8,
            "constant thrust / Isp mass-flow equation",
        ),
        "delta_v_residual": _validation_metric(
            float(abs(actual_delta_v - expected_delta_v)), "m/s", 1.0e-8,
            "Tsiolkovsky rocket equation",
        ),
        "impulse_residual": _validation_metric(
            float(abs(measured_impulse - expected_impulse)),
            "N s", 1.0e-8, "integrated_thrust_impulse(thrust_profile_constant)",
        ),
    }


def _orbital_kepler_to_state(**kwargs):
    from ssatk.orbital_mechanics.keplerian import kepler_to_state

    return kepler_to_state(**kwargs)


def _state_to_kepler_case():
    r, v = _orbital_kepler_to_state(a=7000e3, e=0.001, i=0.1, pa=0.2, raan=0.3, nu=0.4)

    def run():
        from ssatk.orbital_mechanics.keplerian import state_to_kepler

        return state_to_kepler(r, v)

    return run


def _orbital_transfer_hohmann(*args, **kwargs):
    from ssatk.orbital_mechanics.transfer_hohmann import transfer_hohmann

    return transfer_hohmann(*args, **kwargs)


def _orbital_transfer_bielliptic(*args, **kwargs):
    from ssatk.orbital_mechanics.transfer_bielliptic import transfer_bielliptic

    return transfer_bielliptic(*args, **kwargs)


def _orbital_transfer_optimal(*args, **kwargs):
    from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal

    return transfer_optimal(*args, **kwargs)


def _io_ssatk_save_json(data, output_dir: Path):
    from ssatk.io.ssatk_save import ssatk_save

    path = output_dir / "benchmark_io" / "ssatk_save_json.json"
    return ssatk_save(data, path, overwrite=True, root="cwd")


def _io_load_json_case(data, output_dir: Path):
    from ssatk.io.ssatk_save import ssatk_read

    path = _io_ssatk_save_json(data, output_dir)
    return lambda: ssatk_read(path, root="cwd")


def _plot_orbit_xy(positions):
    import matplotlib

    matplotlib.use("Agg", force=True)
    from ssatk.plots.orbit_plot import orbit_plot

    result = orbit_plot(positions, view="xy", show=False, save=False, title="benchmark")
    _close_matplotlib_figures()
    return result


# ---------------------------------------------------------------------------
# Timing, statistics, and output helpers


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_once(call: Callable[[], Any], *, quiet: bool = True) -> Any:
    if not quiet:
        return call()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return call()


def _time_loop(call: Callable[[], Any], loops: int, *, quiet: bool = True, disable_gc: bool = True) -> float:
    was_enabled = gc.isenabled()
    if disable_gc and was_enabled:
        gc.disable()
    try:
        start = time.perf_counter()
        if quiet:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                for _ in range(loops):
                    call()
        else:
            for _ in range(loops):
                call()
        return time.perf_counter() - start
    finally:
        if disable_gc and was_enabled:
            gc.enable()


def _calibrate_loops(
    call: Callable[[], Any],
    *,
    min_sample_time: float,
    max_loops: int,
    quiet: bool,
    disable_gc: bool,
) -> int:
    if min_sample_time <= 0.0:
        return 1
    loops = 1
    while True:
        elapsed = _time_loop(call, loops, quiet=quiet, disable_gc=disable_gc)
        if elapsed >= min_sample_time or loops >= max_loops:
            return max(1, min(loops, max_loops))
        if elapsed <= 0.0:
            loops *= 10
        else:
            scale = max(2, int(math.ceil(min_sample_time / elapsed * 1.25)))
            loops *= scale
        loops = min(loops, max_loops)


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=float), q))


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    p25 = _percentile(samples, 25)
    p75 = _percentile(samples, 75)
    return {
        "mean_s": mean,
        "median_s": median,
        "stdev_s": stdev,
        "min_s": min(samples),
        "max_s": max(samples),
        "p05_s": _percentile(samples, 5),
        "p25_s": p25,
        "p75_s": p75,
        "p95_s": _percentile(samples, 95),
        "p99_s": _percentile(samples, 99),
        "iqr_s": p75 - p25,
        "cv": stdev / mean if mean > 0.0 else 0.0,
        "hz": 1.0 / median if median > 0.0 else 0.0,
    }


def run_benchmark_case(
    case: BenchmarkCase,
    context: BenchmarkContext,
    *,
    repeats: int,
    warmups: int,
    min_sample_time: float,
    max_loops: int,
    quiet: bool = True,
    disable_gc: bool = True,
    trace_memory: bool = False,
) -> BenchmarkResult:
    """Time one benchmark case and return robust per-call statistics."""

    case_repeats = case.default_repeats or repeats
    case_min_sample_time = case.default_min_sample_time
    if case_min_sample_time is None:
        case_min_sample_time = min_sample_time

    try:
        call = case.factory(context)
        for _ in range(warmups):
            _run_once(call, quiet=quiet)
        loops = _calibrate_loops(
            call,
            min_sample_time=case_min_sample_time,
            max_loops=max_loops,
            quiet=quiet,
            disable_gc=disable_gc,
        )
        samples = [
            _time_loop(call, loops, quiet=quiet, disable_gc=disable_gc) / loops
            for _ in range(case_repeats)
        ]
        peak_memory = _measure_peak_memory(call, quiet=quiet) if trace_memory else None
        values = _stats(samples)
        validation = None
        validation_pass = None
        if case.validator is not None:
            validation = case.validator(_run_once(call, quiet=quiet))
            validation_pass = _validation_passed(validation)
        return BenchmarkResult(
            name=case.name,
            group=case.group,
            description=case.description,
            tags=case.tags,
            success=True,
            repeats=case_repeats,
            warmups=warmups,
            loops_per_repeat=loops,
            total_sample_time_s=float(sum(samples) * loops),
            peak_memory_bytes=peak_memory,
            validation=validation,
            validation_pass=validation_pass,
            validation_settings=case.validation_settings or None,
            **values,
        )
    except Exception as exc:
        return BenchmarkResult(
            name=case.name,
            group=case.group,
            description=case.description,
            tags=case.tags,
            success=False,
            repeats=case_repeats,
            warmups=warmups,
            loops_per_repeat=0,
            total_sample_time_s=0.0,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            validation_settings=case.validation_settings or None,
        )


def _measure_peak_memory(call: Callable[[], Any], *, quiet: bool) -> int:
    import tracemalloc

    tracemalloc.start()
    try:
        _run_once(call, quiet=quiet)
        _current, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def run_benchmarks(
    cases: Iterable[BenchmarkCase],
    context: BenchmarkContext,
    *,
    repeats: int = 7,
    warmups: int = 2,
    min_sample_time: float = 0.02,
    max_loops: int = 100_000,
    quiet: bool = True,
    disable_gc: bool = True,
    trace_memory: bool = False,
) -> list[BenchmarkResult]:
    """Run benchmark cases in order."""

    results = []
    for case in cases:
        result = run_benchmark_case(
            case,
            context,
            repeats=repeats,
            warmups=warmups,
            min_sample_time=min_sample_time,
            max_loops=max_loops,
            quiet=quiet,
            disable_gc=disable_gc,
            trace_memory=trace_memory,
        )
        results.append(result)
    return results


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = Path(figpath(f"benchmarks/{timestamp}/benchmark_dashboard.html"))
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    return dashboard_path.parent


def environment_metadata() -> dict[str, Any]:
    import scipy
    import ssapy

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "node": platform.node(),
        "pid": os.getpid(),
        "ssatk_version": __version__,
        "ssapy_version": getattr(ssapy, "__version__", None),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
    }


def write_csv(results: list[BenchmarkResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_dict() for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(results: list[BenchmarkResult], metadata: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": [result.as_dict() for result in results],
    }
    path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _successful(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    return [result for result in results if result.success and result.median_s is not None]


def write_charts(results: list[BenchmarkResult], output_dir: Path) -> list[Path]:
    ok = sorted(_successful(results), key=lambda item: item.median_s or 0.0, reverse=True)
    if not ok:
        return []

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    top = ok[:30]
    labels = [result.name for result in top]
    medians_ms = np.array([(result.median_s or 0.0) * 1e3 for result in top])
    p05_ms = np.array([(result.p05_s or result.median_s or 0.0) * 1e3 for result in top])
    p95_ms = np.array([(result.p95_s or result.median_s or 0.0) * 1e3 for result in top])
    xerr = np.vstack([np.maximum(0.0, medians_ms - p05_ms), np.maximum(0.0, p95_ms - medians_ms)])

    fig_height = max(6.0, 0.36 * len(top) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(top))
    ax.barh(y, medians_ms, xerr=xerr, color="#4c78a8", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("median time per call [ms]; error bars = p05-p95")
    ax.set_title("SSATK Benchmark Timing Summary")
    if np.all(medians_ms > 0.0) and medians_ms.max() / medians_ms.min() > 100.0:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    timing_path = output_dir / "benchmark_timing_summary.png"
    fig.savefig(timing_path, dpi=180)
    plt.close(fig)
    paths.append(timing_path)

    cv_results = sorted(ok, key=lambda item: item.cv or 0.0, reverse=True)[:30]
    labels = [result.name for result in cv_results]
    cv_pct = np.array([(result.cv or 0.0) * 100.0 for result in cv_results])
    fig_height = max(6.0, 0.36 * len(cv_results) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(cv_results))
    ax.barh(y, cv_pct, color="#f58518", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient of variation [%]")
    ax.set_title("Run-to-Run Variability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    variability_path = output_dir / "benchmark_variability.png"
    fig.savefig(variability_path, dpi=180)
    plt.close(fig)
    paths.append(variability_path)
    return paths


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds * 1e3:.4g}"


def _format_hz(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}"


def _format_memory(value: int | None) -> str:
    if value is None:
        return "—"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.3g} {unit}"


def write_dashboard(
    results: list[BenchmarkResult],
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    csv_path: Path,
    json_path: Path,
    chart_paths: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "benchmark_dashboard.html"
    ok = _successful(results)
    failed = [result for result in results if not result.success]
    fastest = min(ok, key=lambda item: item.median_s or float("inf"), default=None)
    slowest = max(ok, key=lambda item: item.median_s or 0.0, default=None)
    total_sample_time = sum(result.total_sample_time_s for result in results)

    rows = []
    for result in sorted(results, key=lambda item: (not item.success, item.group, item.name)):
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.group)}</td>"
            f"<td><code>{html.escape(result.name)}</code></td>"
            f"<td>{html.escape(result.description)}</td>"
            f"<td>{'ok' if result.success else 'failed'}</td>"
            f"<td>{result.repeats}</td>"
            f"<td>{result.loops_per_repeat}</td>"
            f"<td>{_format_ms(result.median_s)}</td>"
            f"<td>{_format_ms(result.mean_s)}</td>"
            f"<td>{_format_ms(result.min_s)}</td>"
            f"<td>{_format_ms(result.p95_s)}</td>"
            f"<td>{_format_ms(result.stdev_s)}</td>"
            f"<td>{'' if result.cv is None else f'{result.cv * 100.0:.3g}'}</td>"
            f"<td>{_format_hz(result.hz)}</td>"
            f"<td>{_format_memory(result.peak_memory_bytes)}</td>"
            f"<td>{html.escape(result.error or '')}</td>"
            "</tr>"
        )

    chart_html = "\n".join(
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></figure>'
        for path in chart_paths
    )
    failed_html = ""
    if failed:
        failed_html = "<h2>Failures</h2>" + "\n".join(
            f"<details><summary><code>{html.escape(result.name)}</code>: {html.escape(result.error or '')}</summary>"
            f"<pre>{html.escape(result.traceback or '')}</pre></details>"
            for result in failed
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SSATK Benchmark Dashboard</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #1f2933; }}
h1 {{ margin-bottom: 0.25rem; }}
.meta {{ color: #52616b; margin-bottom: 1.5rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
.card {{ border: 1px solid #d9e2ec; border-radius: 0.75rem; padding: 1rem; background: #f8fafc; }}
.card strong {{ display: block; font-size: 1.4rem; margin-top: 0.25rem; }}
figure {{ margin: 1.5rem 0; }}
img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 0.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border-bottom: 1px solid #d9e2ec; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #eef2f7; z-index: 1; }}
tr:hover {{ background: #f8fafc; }}
code {{ background: #eef2f7; border-radius: 0.25rem; padding: 0.05rem 0.2rem; }}
pre {{ white-space: pre-wrap; background: #102a43; color: #f0f4f8; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }}
a {{ color: #2563eb; }}
</style>
</head>
<body>
<h1>SSATK Benchmark Dashboard</h1>
<div class="meta">Generated {html.escape(metadata['generated_utc'])} with ssatk {html.escape(metadata['ssatk_version'])} on {html.escape(metadata['platform'])}.</div>
<div class="cards">
  <div class="card">Cases<strong>{len(results)}</strong></div>
  <div class="card">Successful<strong>{len(ok)}</strong></div>
  <div class="card">Failed<strong>{len(failed)}</strong></div>
  <div class="card">Timed Sample Time<strong>{total_sample_time:.3g} s</strong></div>
  <div class="card">Fastest Median<strong>{html.escape(fastest.name if fastest else '—')}<br>{_format_ms(fastest.median_s if fastest else None)} ms</strong></div>
  <div class="card">Slowest Median<strong>{html.escape(slowest.name if slowest else '—')}<br>{_format_ms(slowest.median_s if slowest else None)} ms</strong></div>
</div>
<p>Raw outputs: <a href="{html.escape(csv_path.name)}">CSV</a> and <a href="{html.escape(json_path.name)}">JSON</a>.</p>
{chart_html}
<h2>Timing Table</h2>
<table>
<thead><tr>
<th>Group</th><th>Function / Scenario</th><th>Description</th><th>Status</th><th>Repeats</th><th>Loops</th>
<th>Median [ms]</th><th>Mean [ms]</th><th>Min [ms]</th><th>P95 [ms]</th><th>Std [ms]</th><th>CV [%]</th><th>Hz</th><th>Peak Memory</th><th>Error</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
{failed_html}
<h2>Environment</h2>
<pre>{html.escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre>
</body>
</html>
"""
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def filter_cases(
    cases: list[BenchmarkCase],
    *,
    groups: set[str] | None = None,
    pattern: str | None = None,
) -> list[BenchmarkCase]:
    if groups:
        cases = [case for case in cases if case.group in groups]
    if pattern:
        pattern_lower = pattern.lower()
        cases = [
            case
            for case in cases
            if pattern_lower in case.name.lower()
            or pattern_lower in case.group.lower()
            or pattern_lower in " ".join(case.tags).lower()
        ]
    return cases


def _profile_flags(profile: str) -> tuple[bool, bool, bool]:
    if profile == "validation":
        return False, False, False
    if profile == "core":
        return False, False, False
    if profile == "standard":
        return True, True, False
    if profile == "full":
        return True, True, True
    raise ValueError(f"Unknown profile: {profile}")


def list_cases(cases: list[BenchmarkCase]) -> str:
    lines = ["Available SSATK benchmark cases:"]
    for case in cases:
        tags = f" [{' '.join(case.tags)}]" if case.tags else ""
        lines.append(f"  {case.group:18s} {case.name:38s}{tags} - {case.description}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark representative SSATK functions and build an HTML dashboard."
    )
    parser.add_argument(
        "--profile",
        choices=("core", "standard", "full", "validation"),
        default="standard",
        help="Benchmark set: core excludes IO/plots; standard includes IO/plots; full adds slow cases; validation runs four analytical cases.",
    )
    parser.add_argument("--include-io", action="store_true", help="Force-enable IO benchmarks.")
    parser.add_argument("--no-io", action="store_true", help="Disable IO benchmarks.")
    parser.add_argument("--include-plots", action="store_true", help="Force-enable plotting benchmarks.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting benchmarks.")
    parser.add_argument("--include-slow", action="store_true", help="Force-enable slow benchmarks.")
    parser.add_argument("--groups", help="Comma-separated group filter, e.g. vectors,orbital_mechanics.")
    parser.add_argument("--pattern", help="Substring filter matched against case name, group, or tags.")
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per case before per-case overrides.")
    parser.add_argument("--warmups", type=int, default=2, help="Warmup calls per case before timing.")
    parser.add_argument(
        "--min-sample-time",
        type=float,
        default=0.02,
        help="Target seconds per repeat; cheap functions get inner loops calibrated to this duration.",
    )
    parser.add_argument("--max-loops", type=int, default=100_000, help="Maximum calibrated inner loops per repeat.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to ~/ssatk_output/figures/benchmarks/<timestamp>.")
    parser.add_argument("--list", action="store_true", help="List selected benchmark cases and exit.")
    parser.add_argument("--quiet", action="store_true", default=True, help="Suppress benchmark function stdout/stderr during timing.")
    parser.add_argument("--no-quiet", action="store_false", dest="quiet", help="Allow benchmark function stdout/stderr.")
    parser.add_argument("--keep-gc", action="store_true", help="Do not disable Python garbage collection while timing loops run.")
    parser.add_argument("--trace-memory", action="store_true", help="Run each case once with tracemalloc and report peak allocations.")
    parser.add_argument("--no-dashboard", action="store_true", help="Write CSV/JSON only; skip chart and HTML dashboard generation.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return a nonzero exit code if any benchmark case fails.")
    parser.add_argument("--fail-on-validation", action="store_true", help="Return a nonzero exit code if any validation metric fails.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut for smoke tests: repeats=2, warmups=0, min_sample_time=0, max_loops=1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    include_io, include_plots, include_slow = _profile_flags(args.profile)
    if args.include_io:
        include_io = True
    if args.no_io:
        include_io = False
    if args.include_plots:
        include_plots = True
    if args.no_plots:
        include_plots = False
    if args.include_slow:
        include_slow = True

    repeats = args.repeats
    warmups = args.warmups
    min_sample_time = args.min_sample_time
    max_loops = args.max_loops
    if args.quick:
        repeats = 2
        warmups = 0
        min_sample_time = 0.0
        max_loops = 1

    groups = {group.strip() for group in args.groups.split(",")} if args.groups else None
    cases = build_benchmark_cases(
        include_io=include_io,
        include_plots=include_plots,
        include_slow=include_slow,
    )
    if args.profile == "validation":
        cases = [case for case in cases if case.group == "validation"]
    cases = filter_cases(cases, groups=groups, pattern=args.pattern)
    if args.list:
        print(list_cases(cases))
        return 0
    if not cases:
        print("No benchmark cases selected.", file=sys.stderr)
        return 2

    output_dir = args.output_dir or default_output_dir()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    context = BenchmarkContext(output_dir=output_dir)

    print(f"Running {len(cases)} SSATK benchmark cases...")
    results = run_benchmarks(
        cases,
        context,
        repeats=repeats,
        warmups=warmups,
        min_sample_time=min_sample_time,
        max_loops=max_loops,
        quiet=args.quiet,
        disable_gc=not args.keep_gc,
        trace_memory=args.trace_memory,
    )
    metadata = environment_metadata()
    metadata.update(
        benchmark_profile=args.profile,
        repeats=repeats,
        warmups=warmups,
        min_sample_time=min_sample_time,
        max_loops=max_loops,
        quiet=args.quiet,
        gc_disabled_during_timing=not args.keep_gc,
        trace_memory=args.trace_memory,
        validation_gate=args.profile == "validation" or args.fail_on_validation,
    )

    csv_path = write_csv(results, output_dir / "benchmark_results.csv")
    json_path = write_json(results, metadata, output_dir / "benchmark_results.json")
    chart_paths: list[Path] = []
    dashboard_path: Path | None = None
    if not args.no_dashboard:
        chart_paths = write_charts(results, output_dir)
        dashboard_path = write_dashboard(
            results,
            metadata,
            output_dir,
            csv_path=csv_path,
            json_path=json_path,
            chart_paths=chart_paths,
        )

    ok = _successful(results)
    failed = [result for result in results if not result.success]
    validation_failed = any(result.validation_pass is False for result in results)
    print(f"Completed {len(ok)}/{len(results)} benchmark cases successfully.")
    if ok:
        slowest = max(ok, key=lambda item: item.median_s or 0.0)
        fastest = min(ok, key=lambda item: item.median_s or float("inf"))
        print(f"Fastest median: {fastest.name} ({_format_ms(fastest.median_s)} ms)")
        print(f"Slowest median: {slowest.name} ({_format_ms(slowest.median_s)} ms)")
    if failed:
        print("Failed cases:", file=sys.stderr)
        for result in failed:
            print(f"  {result.name}: {result.error}", file=sys.stderr)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if dashboard_path is not None:
        print(f"Dashboard: {dashboard_path}")
    return 1 if (failed and (args.fail_on_error or args.profile == "validation")) or (
        validation_failed and (args.fail_on_validation or args.profile == "validation")
    ) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
