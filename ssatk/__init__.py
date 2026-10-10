"""Space Situational Awareness Toolkit (SSATK).

Utilities for orbital mechanics, plotting, coordinate transforms,
6-DoF propagators, io helpers, and SSAPy-related workflows.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    # Single source of truth: the version in pyproject.toml, as installed.
    __version__ = _distribution_version("ssatk")
except _PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

_CONSTANT_NAMES = None
_SSAPY_ALIAS_NAMES = frozenset(
    {
        "Accel",
        "AccelConstNTW",
        "AccelDrag",
        "AccelEarthRad",
        "AccelHarmonic",
        "AccelKepler",
        "AccelSolRad",
        "AccelSum",
        "AccelThirdBody",
        "BinarySelectorParams",
        "Body",
        "DanchickTwoPosOrbitSolver",
        "DirectInitializer",
        "DistanceProjectionInitializer",
        "EarthObserver",
        "EarthOrientation",
        "Ellipsoid",
        "EmceeSampler",
        "GEOProjectionInitializer",
        "GaussianRVInitializer",
        "GaussTwoPosOrbitSolver",
        "HarmonicCoefficients",
        "KeplerianPropagator",
        "LMOptimizer",
        "Leapfrog4Propagator",
        "LeapfrogPropagator",
        "Linker",
        "MHSampler",
        "MVNormalProposal",
        "ModelSelectorParams",
        "MoonOrientation",
        "MoonPosition",
        "Orbit",
        "OrbitalObserver",
        "Particles",
        "RK4Propagator",
        "RK78Propagator",
        "RK8Propagator",
        "RVProbability",
        "RVSigmaProposal",
        "SGP4Propagator",
        "SciPyPropagator",
        "SeriesPropagator",
        "SheferTwoPosOrbitSolver",
        "ThreeAngleOrbitSolver",
        "TwoPosOrbitSolver",
        "altaz",
        "circular_guess",
        "datadir",
        "dircos",
        "get_body",
        "groundTrack",
        "quickAltAz",
        "radec",
        "radecRate",
        "rv",
    }
)
_TOOLKIT_SUBMODULE_NAMES = frozenset(
    {
        "accelerations_6dof",
        "accelerations_orbit",
        "asteroids",
        "compute",
        "constants",
        "coordinates",
        "data",
        "demo_gallery",
        "engines",
        "environment",
        "environment_eop",
        "environment_space_weather",
        "hpc",
        "io",
        "launch",
        "navigation",
        "operations",
        "orbit_initializer",
        "orbital_mechanics",
        "plots",
        "propagators_orbit",
        "propagators_6dof",
        "run_all_demos",
        "satellites",
        "ssa",
        "ssapy_wrappers",
        "time_functions",
        "utils",
        "vectors",
        "yastropy",
    }
)
_TOOLKIT_DUPLICATE_ALIASES = {
    "apoapsis": ".orbital_mechanics.keplerian",
    "cart2sph_deg": ".coordinates.cartesian",
    "cart_to_cyl": ".coordinates.cartesian",
    "dd_to_dms": ".time_functions.convert_dd_and_dms",
    "dd_to_hms": ".time_functions.convert_dd_and_hms",
    "deg0to360": ".coordinates.angle_units",
    "deg0to360array": ".coordinates.angle_units",
    "deg90to90": ".coordinates.angle_units",
    "deg90to90array": ".coordinates.angle_units",
    "dms_to_dd": ".time_functions.convert_dd_and_dms",
    "dms_to_deg": ".coordinates.angle_units",
    "dms_to_rad": ".coordinates.angle_units",
    "draw_earth": ".plots.plotutils",
    "draw_moon": ".plots.plotutils",
    "ecliptic_to_equatorial": ".coordinates.equatorial_ecliptic",
    "ecliptic_xyz_to_equatorial": ".coordinates.equatorial_ecliptic",
    "ecliptic_xyz_to_equatorial_xyz": ".coordinates.equatorial_ecliptic",
    "einsum_norm": ".vectors",
    "equatorial_to_ecliptic": ".coordinates.equatorial_ecliptic",
    "equatorial_to_horizontal": ".coordinates.local_equatorial",
    "equatorial_xyz_to_ecliptic_xyz": ".coordinates.equatorial_ecliptic",
    "hms_to_dd": ".time_functions.convert_dd_and_hms",
    "horizontal_to_equatorial": ".coordinates.local_equatorial",
    "inert2rot": ".coordinates.rotating_frames",
    "load_earth_file": ".plots.plotutils",
    "load_moon_file": ".plots.plotutils",
    "lonlat_distance": ".coordinates.geodetic",
    "norm": ".vectors",
    "normSq": ".vectors",
    "normed": ".vectors",
    "periapsis": ".orbital_mechanics.keplerian",
    "propagate_orbit_state": ".propagators_orbit",
    "propagate_orbit_state_with_stm": ".propagators_orbit",
    "propagate_6dof": ".propagators_6dof",
    "propagate_6dof_high_accuracy": ".propagators_6dof",
    "propagate_spacecraft_high_accuracy": ".propagators_6dof",
    "propagate_spacecraft_segments": ".propagators_6dof",
    "ImpulseManeuver": ".propagators_6dof",
    "SixDOFTargetResult": ".propagators_6dof",
    "SixDOFMultiSegmentTargetResult": ".propagators_6dof",
    "SixDOFMultipleShootingTargetResult": ".propagators_6dof",
    "solve_6dof_target": ".propagators_6dof",
    "solve_6dof_multi_segment_target": ".propagators_6dof",
    "solve_6dof_multiple_shooting_target": ".propagators_6dof",
    "SixDOFVariationalTrajectory": ".propagators_6dof",
    "propagate_6dof_covariance": ".propagators_6dof",
    "propagate_6dof_variational": ".propagators_6dof",
    "Extended6DOFTrajectory": ".propagators_6dof",
    "FlexibleMode": ".propagators_6dof",
    "HingedAppendage": ".propagators_6dof",
    "SloshMode": ".propagators_6dof",
    "propagate_6dof_extended": ".propagators_6dof",
    "propagate_6dof_extended_variational": ".propagators_6dof",
    "ReferenceCase": ".io",
    "compare_reference_case": ".io",
    "read_reference_case": ".io",
    "altitude_crossing_event": ".propagators_6dof",
    "attitude_quaternion_from_frame": ".coordinates.attitude",
    "constant_body_thrust": ".accelerations_6dof",
    "constant_body_torque": ".accelerations_6dof",
    "constant_inertial_thrust": ".accelerations_6dof",
    "constant_ntw_thrust": ".accelerations_6dof",
    "attitude_error_quaternion": ".accelerations_6dof",
    "Component": ".satellites",
    "ALL_THIRD_BODY_NAMES": ".environment",
    "DEFAULT_THIRD_BODY_NAMES": ".environment",
    "FORCE_MODEL_PRESETS": ".environment",
    "PLANET_THIRD_BODY_NAMES": ".environment",
    "Spacecraft": ".propagators_6dof",
    "SpaceEnvironment": ".environment",
    "EarthOrientationRecord": ".environment_eop",
    "EarthOrientationTable": ".environment_eop",
    "SpaceWeatherRecord": ".environment_space_weather",
    "SpaceWeatherTable": ".environment_space_weather",
    "SpacecraftAccelConstBody": ".accelerations_6dof",
    "SpacecraftAccelConstInertial": ".accelerations_6dof",
    "SpacecraftAccelConstNTW": ".accelerations_6dof",
    "SpacecraftAccelDrag": ".accelerations_6dof",
    "SpacecraftAccelJ2": ".accelerations_6dof",
    "SpacecraftAccelKepler": ".accelerations_6dof",
    "SpacecraftAccelSolRad": ".accelerations_6dof",
    "SpacecraftAccelSum": ".accelerations_6dof",
    "SpacecraftAccelSSAPy": ".accelerations_6dof",
    "SpacecraftAccelThirdBody": ".accelerations_6dof",
    "SpacecraftManeuverAccel": ".accelerations_6dof",
    "SpacecraftFacetDrag": ".accelerations_6dof",
    "SpacecraftFacetSolRad": ".accelerations_6dof",
    "SpacecraftFlatPlateDrag": ".accelerations_6dof",
    "SpacecraftFlatPlateSolRad": ".accelerations_6dof",
    "SpacecraftGravityGradientTorque": ".accelerations_6dof",
    "SpacecraftAttitudePD": ".accelerations_6dof",
    "SpacecraftMagneticTorque": ".accelerations_6dof",
    "SpacecraftReactionWheelTorque": ".accelerations_6dof",
    "SpacecraftThrusterAccel": ".accelerations_6dof",
    "SpacecraftTorqueSum": ".accelerations_6dof",
    "Facet": ".satellites",
    "MagneticDipole": ".satellites",
    "ReactionWheel": ".satellites",
    "SpacecraftBody": ".satellites",
    "Tank": ".satellites",
    "Thruster": ".satellites",
    "available_satellite_designs": ".satellites",
    "cislunar_probe": ".satellites",
    "cylindrical_eclipse_fraction": ".environment",
    "body_mu": ".environment",
    "body_radius": ".environment",
    "debris_panel": ".satellites",
    "earth_observation_sat": ".satellites",
    "earth_dipole_magnetic_field": ".environment",
    "gcrf_to_itrf_eop": ".coordinates.earth_fixed",
    "exponential_atmosphere": ".environment",
    "gnss_sat": ".satellites",
    "igrf_magnetic_field": ".environment",
    "itrf_to_gcrf_eop": ".coordinates.earth_fixed",
    "magnetic_dipole_torque": ".accelerations_6dof",
    "mass_floor_event": ".propagators_6dof",
    "normalize_quaternion": ".coordinates.attitude",
    "propellant_empty_event": ".propagators_6dof",
    "make_attitude_pd": ".accelerations_6dof",
    "make_finite_burn_acceleration": ".accelerations_6dof",
    "make_gravity_gradient_torque": ".accelerations_6dof",
    "make_magnetic_torque": ".accelerations_6dof",
    "make_maneuver_acceleration": ".accelerations_6dof",
    "make_reaction_wheel_torque": ".accelerations_6dof",
    "make_space_environment": ".environment",
    "load_packaged_eop": ".environment_eop",
    "load_packaged_space_weather": ".environment_space_weather",
    "read_eop": ".environment_eop",
    "read_space_weather": ".environment_space_weather",
    "make_ssapy_drag": ".accelerations_6dof",
    "make_ssapy_earth_harmonics": ".accelerations_6dof",
    "make_ssapy_earth_radiation": ".accelerations_6dof",
    "make_ssapy_perturbation_acceleration": ".accelerations_6dof",
    "make_ssapy_solar_radiation": ".accelerations_6dof",
    "make_ssapy_third_body": ".accelerations_6dof",
    "point_mass_inertia": ".satellites",
    "quaternion_conjugate": ".coordinates.attitude",
    "quaternion_from_matrix": ".coordinates.attitude",
    "quaternion_multiply": ".coordinates.attitude",
    "radius_crossing_event": ".propagators_6dof",
    "reaction_wheel_torque": ".accelerations_6dof",
    "reaction_wheel_torque_commands": ".accelerations_6dof",
    "reaction_wheel_triplet": ".satellites",
    "rotate_facets": ".satellites",
    "satellite_design": ".satellites",
    "sum_accelerations": ".accelerations_6dof",
    "sum_torques": ".accelerations_6dof",
    "period": ".orbital_mechanics.keplerian",
    "ra_dec": ".coordinates.sky",
    "rad0to2pi": ".coordinates.angle_units",
    "rightascension2hourangle": ".coordinates.local_equatorial",
    "sim_lonlatrad": ".coordinates.rotating_frames",
    "ssatk_load_cache": ".io.ssatk_cache",
    "ssatk_save_cache": ".io.ssatk_cache",
    "ssatk_read": ".io.ssatk_save",
    "ssatk_save": ".io.ssatk_save",
    "supported_save_formats": ".io.ssatk_save",
    "sun_ra_dec": ".coordinates.sky",
    "solar_disk_visible_fraction": ".environment",
    "solar_occultation_fraction": ".environment",
    "ThrustCurve": ".accelerations_6dof",
    "integrated_thrust_impulse": ".accelerations_6dof",
    "load_digitized_thrust_curve": ".accelerations_6dof",
    "load_packaged_thrust_curve": ".accelerations_6dof",
    "load_packaged_thrust_curve_metadata": ".accelerations_6dof",
    "load_thrust_curve_data": ".accelerations_6dof",
    "load_thrust_curve_csv": ".accelerations_6dof",
    "packaged_thrust_curve_index": ".accelerations_6dof",
    "load_obj_facets": ".satellites",
    "mesh_facets": ".satellites",
    "thrust_profile_constant": ".accelerations_6dof",
    "thrust_profile_exponential": ".accelerations_6dof",
    "thrust_profile_pulsed": ".accelerations_6dof",
    "thrust_profile_smoothstep": ".accelerations_6dof",
    "thrust_profile_trapezoid": ".accelerations_6dof",
    "thruster_mass_flow_rate": ".accelerations_6dof",
    "ThrusterSpec": ".engines",
    "available_thruster_families": ".engines",
    "available_thruster_scales": ".engines",
    "available_thruster_specs": ".engines",
    "available_throttle_maps": ".engines",
    "build_thruster": ".engines",
    "load_throttle_map": ".engines",
    "make_thruster_profile": ".engines",
    "make_thruster_acceleration": ".engines",
    "propellant_mass_for_delta_v": ".engines",
    "thruster_catalog_dict": ".engines",
    "thruster_spec": ".engines",
    "unit_vector": ".vectors",
    "wrap_ssapy_acceleration": ".accelerations_6dof",
    "xyz_to_ecliptic": ".coordinates.equatorial_ecliptic",
    "xyz_to_equatorial": ".coordinates.equatorial_ecliptic",
}

__all__ = []


def _constant_names():
    global _CONSTANT_NAMES
    if _CONSTANT_NAMES is None:
        _CONSTANT_NAMES = set(import_module(".constants", __name__).__all__)
    return _CONSTANT_NAMES


def _ssapy_module():
    return import_module("ssapy")


def __getattr__(name):
    if name == "ssapy":
        module = _ssapy_module()
        globals()[name] = module
        return module
    if name in _TOOLKIT_SUBMODULE_NAMES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _constant_names():
        value = getattr(import_module(".constants", __name__), name)
        globals()[name] = value
        return value
    if name in _TOOLKIT_DUPLICATE_ALIASES:
        module = import_module(_TOOLKIT_DUPLICATE_ALIASES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SSAPY_ALIAS_NAMES:
        value = getattr(_ssapy_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | {"ssapy"}
        | _TOOLKIT_SUBMODULE_NAMES
        | _constant_names()
        | set(_TOOLKIT_DUPLICATE_ALIASES)
        | _SSAPY_ALIAS_NAMES
    )

try:
    from astropy.utils import iers

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 365
except ImportError:
    pass
