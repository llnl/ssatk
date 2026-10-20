"""Presentation order for the gallery; source folders and demo execution stay intact.

Known examples progress from introductory to specialized within each topic.
New examples inherit their folder's topic and follow the curated examples in
title order. Explicit custom categories remain separate, after the main topics.
"""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class GalleryGroup:
    key: str
    title: str
    description: str
    demos: tuple[str, ...] = ()


GALLERY_GROUPS = (
    GalleryGroup("start", "Getting started",
                 "Begin with a complete workflow, then try plotting and packaged data.", (
        "getting_started/demo_first_user_workflow",
        "getting_started/demo_plotting_quickstart",
        "getting_started/demo_data_package_access",
    )),
    GalleryGroup("scenes", "Earth, Moon and Solar System views",
                 "Explore trajectories and interactive scenes, from Earth orbit to the Solar System. "
                 "Each demo describes its own scale and rendering conventions.", (
        "orbit_visualization/demo_orbit_plot",
        "orbit_visualization/demo_globe_plot",
        "orbit_visualization/demo_camera_modes",
        "orbit_visualization/demo_globe_orbit_daynight",
        "sensor_coverage/demo_satellite_viewer",
        "orbit_visualization/demo_groundtrack_plot",
        "orbit_visualization/demo_groundtrack_enhanced",
        "orbit_visualization/demo_moon_plot",
        "orbit_visualization/demo_moon_webgl",
        "space_environment/demo_earth_sun_plot",
        "space_environment/demo_solar_view_plot",
    )),
    GalleryGroup("orbits", "Orbit geometry and propagation",
                 "Build and sample orbits, inspect their geometry, and compare propagation models.", (
        "orbital_mechanics/demo_ssapy_orbits",
        "orbital_mechanics/demo_ellipses",
        "orbital_mechanics/demo_ellipse_from_rv",
        "orbital_mechanics/demo_ellipse_ae_for_arrival_rv",
        "orbital_mechanics/demo_equally_spaced_ta",
        "orbital_mechanics/demo_gamma_and_heading",
        "orbital_mechanics/demo_leapfrog",
        "orbital_mechanics/demo_compare_models",
        "orbital_mechanics/demo_station_keeping",
    )),
    GalleryGroup("maneuvers", "Maneuvers and transfers",
                 "Fit transfer paths, choose maneuver controls, and explore station keeping.", (
        "orbital_mechanics/demo_orbital_maneuvers",
        "orbital_mechanics/demo_ellipse_fit",
        "orbital_mechanics/demo_structured_transfer_api",
    )),
    GalleryGroup("attitude", "Spacecraft attitude and dynamics",
                 "Choose a spacecraft mesh, view attitude vectors, then explore coupled translation "
                 "and rotation with six degrees of freedom (6-DoF).", (
        "six_dof/demo_spacecraft_mesh_catalog",
        "six_dof/demo_spacecraft_view",
        "six_dof/demo_6dof_gravity_gradient",
        "six_dof/demo_6dof_attitude_control",
        "six_dof/demo_6dof_srp_facets",
        "six_dof/demo_6dof_finite_burn",
    )),
    GalleryGroup("observing", "Coverage and navigation",
                 "Inspect sensor fields of view, map observation opportunities, and estimate "
                 "attitude and position from simulated angular measurements.", (
        "sensor_coverage/demo_sensor_fov_plot",
        "sensor_coverage/demo_coverage_analysis",
        "navigation/demo_star_finder",
    )),
    GalleryGroup("light", "Illumination and brightness",
                 "Explore day/night shading and eclipses before reflected-light models "
                 "and spacecraft light curves.", (
        "space_environment/demo_sun_view",
        "eclipse/demo_eclipse_views",
        "photometry/demo_lambertian_brightness",
        "photometry/demo_ssapy_ground_lambertian_reflectance",
        "photometry/demo_photometry_application",
        "photometry/demo_faceted_light_curves",
    )),
    GalleryGroup("environment", "Space environment",
                 "Visualize Earth's magnetic field and radiation belts.", (
        "space_environment/demo_magfield_plot",
        "space_environment/demo_van_allen",
    )),
    GalleryGroup("analysis", "Dashboards and uncertainty",
                 "Build analysis panels, summarize orbit populations, and examine how "
                 "initial-state uncertainty evolves.", (
        "analysis_dashboards/demo_build_dashboard",
        "analysis_dashboards/demo_orbital_stats_dashboard",
        "analysis_dashboards/demo_divergence_plot",
    )),
    GalleryGroup("coordinates", "Coordinates and sampling",
                 "Work with reference frames, ground coordinates, and reusable sampling tools.", (
        "coordinates/demo_coordinate_frames",
        "coordinates/demo_gcrf_to_latlon",
        "data_io/demo_sampling",
        "data_io/demo_sphere_generation",
    )),
    GalleryGroup("references", "Reference comparisons",
                 "Compare SSATK propagation methods with external propagators or reference "
                 "trajectories. A successful run alone does not establish physical accuracy; "
                 "inspect each comparison's assumptions and residuals.", (
        "orbital_mechanics/demo_ellipse_fit_against_ssapy",
        "orbital_mechanics/demo_groundtrack_accuracy",
        "benchmarks/demo_orekit_benchmark",
        "benchmarks/demo_long_term_propagation_benchmark",
        "benchmarks/demo_nbody_propagation_benchmark",
        "benchmarks/demo_artemis_benchmark",
    )),
    GalleryGroup("animation", "Animation export",
                 "Create GIFs and videos, including a 60-second movie covering one lunar month.", (
        "video_animation/demo_gifify",
        "video_animation/demo_fancy_video",
    )),
)

_GROUPS = {group.key: group for group in GALLERY_GROUPS}
_DEMO_GROUPS = {name: group for group in GALLERY_GROUPS for name in group.demos}
_FOLDER_GROUPS = dict(
    getting_started="start", orbit_visualization="scenes", orbital_mechanics="orbits",
    six_dof="attitude", sensor_coverage="observing", navigation="observing",
    photometry="light", eclipse="light", space_environment="environment",
    analysis_dashboards="analysis", coordinates="coordinates", data_io="coordinates",
    benchmarks="references", video_animation="animation",
)


def gallery_group(name: str, category: str) -> GalleryGroup:
    """Respect custom categories while organizing the built-in folder categories."""
    if name in _DEMO_GROUPS and category == name.rpartition("/")[0]:
        return _DEMO_GROUPS[name]
    if category in _FOLDER_GROUPS:
        return _GROUPS[_FOLDER_GROUPS[category]]
    if not category:
        return GalleryGroup("other", "Other examples", "Examples without a topic category.")
    key = "custom-" + sha256(category.encode("utf-8")).hexdigest()[:16]
    return GalleryGroup(key, category.replace("_", " "), "Additional examples in this category.")


def gallery_title(name: str, title: str) -> str:
    """Improve generated filename titles without replacing authored demo titles."""
    stem = name.rsplit("/", 1)[-1]
    if title != stem.replace("_", " ").title():
        return title
    overrides = {
        "demo_ellipse_ae_for_arrival_rv": "Ellipse parameters from a position and velocity",
        "demo_ellipse_from_rv": "Ellipse from a position and velocity",
        "demo_equally_spaced_ta": "Equal spacing in true anomaly",
        "demo_magfield_plot": "Earth's magnetic field",
        "demo_sensor_fov_plot": "Sensor field of view",
        "demo_gifify": "Export an animated GIF",
        "demo_sun_view": "Sun direction and day/night shading",
        "demo_globe_orbit_daynight": "Earth orbit with day/night shading",
        "demo_groundtrack_accuracy": "SSATK transfer and Kepler propagation",
        "demo_ellipse_fit_against_ssapy": "SSATK ellipse-fit and propagation comparison",
        "demo_compare_models": "Acceleration model comparisons",
        "demo_ssapy_orbits": "Orbit creation and sampling with SSATK",
        "demo_build_dashboard": "Build a dashboard from synthetic data",
        "demo_ellipse_fit": "Transfer ellipse fitting and checks",
        "demo_station_keeping": "Perturbed orbit versus Kepler reference",
    }
    if stem in overrides:
        return overrides[stem]
    words = stem.removeprefix("demo_").split("_")
    acronyms = {"ssapy": "SSAPy", "ssatk": "SSATK", "gcrf": "GCRF", "ntw": "NTW", "itrf": "ITRF",
                "hdf5": "HDF5", "api": "API", "nbody": "N-body", "rv": "state vectors",
                "orekit": "Orekit", "artemis": "ARTEMIS", "lambertian": "Lambertian",
                "moon": "Moon", "van": "Van", "allen": "Allen"}
    return " ".join(
        acronyms.get(word, word.capitalize() if i == 0 else word) for i, word in enumerate(words)
    )


def gallery_sort_key(result):
    """Sort topics and their curated examples before deterministic fallbacks."""
    group = gallery_group(result.name, result.category)
    group_index = next((i for i, item in enumerate(GALLERY_GROUPS) if item.key == group.key), len(GALLERY_GROUPS))
    demo_index = group.demos.index(result.name) if result.name in group.demos else len(group.demos)
    return (group_index, group.title.casefold(), group.key, demo_index,
            result.title.casefold(), result.name.casefold(), result.name)
