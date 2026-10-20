"""Finite-source solar and lunar eclipse workflows for SSATK.

The implementation uses only existing SSATK packages. Numerical and
state modules live in ``compute``; frame/observer modules in ``coordinates``;
resource readers in ``io``; and rendering modules in ``plots``.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "2.2.2"

_EXPORTS = {
    # canonical API
    "SCHEMA_ID": ("ssatk.eclipse_api_impl", "SCHEMA_ID"),
    "EventRequest": ("ssatk.eclipse_api_impl", "EventRequest"),
    "RenderRequest": ("ssatk.eclipse_api_impl", "RenderRequest"),
    "ValidationResult": ("ssatk.eclipse_api_impl", "ValidationResult"),
    "build_event": ("ssatk.eclipse_api_impl", "build_event"),
    "validate_event": ("ssatk.eclipse_api_impl", "validate_event"),
    "render_product": ("ssatk.eclipse_api_impl", "render_product"),
    "event_to_dict": ("ssatk.eclipse_api_impl", "event_to_dict"),
    "event_from_dict": ("ssatk.eclipse_api_impl", "event_from_dict"),
    "export_event_state": ("ssatk.eclipse_api_impl", "export_event_state"),
    "load_event_state": ("ssatk.eclipse_api_impl", "load_event_state"),
    # GUI job
    "EclipseRenderJob": ("ssatk.eclipse_gui", "EclipseRenderJob"),
    "RenderCancelled": ("ssatk.eclipse_gui", "RenderCancelled"),
    "RenderJobResult": ("ssatk.eclipse_gui", "RenderJobResult"),
    "RenderProgress": ("ssatk.eclipse_gui", "RenderProgress"),
    # observer
    "ObserverConfig": ("ssatk.coordinates.eclipse_observer_geometry", "ObserverConfig"),
    "ObserverState": ("ssatk.coordinates.eclipse_observer_geometry", "ObserverState"),
    "HorizonProfile": ("ssatk.coordinates.eclipse_observer_geometry", "HorizonProfile"),
    # core finite-source functions
    "apparent_angular_radius": ("ssatk.compute.eclipse_core", "apparent_angular_radius"),
    "apparent_disk_geometry": ("ssatk.compute.eclipse_core", "apparent_disk_geometry"),
    "circle_overlap_visible_fraction": ("ssatk.compute.eclipse_core", "circle_overlap_visible_fraction"),
    "finite_source_visibility": ("ssatk.compute.eclipse_core", "finite_source_visibility"),
    "finite_source_irradiance": ("ssatk.compute.eclipse_core", "finite_source_irradiance"),
    "ray_sphere_intersections": ("ssatk.compute.eclipse_core", "ray_sphere_intersections"),
    "ray_ellipsoid_intersections": ("ssatk.compute.eclipse_core", "ray_ellipsoid_intersections"),
    "first_positive_intersection": ("ssatk.compute.eclipse_core", "first_positive_intersection"),
    "shadow_cone": ("ssatk.compute.eclipse_core", "shadow_cone"),
    "shadow_cross_section": ("ssatk.compute.eclipse_core", "shadow_cross_section"),
    "sample_spherical_photosphere": ("ssatk.compute.eclipse_core", "sample_spherical_photosphere"),
    # products
    "generate_solar_observer_products": ("ssatk.plots.eclipse_observer_products", "generate_solar_observer_products"),
    "generate_event_scientific_summary": ("ssatk.plots.eclipse_scientific_suite", "generate_event_scientific_summary"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
