"""Finite-source solar and lunar eclipse workflows for SSAPy-Toolkit.

The implementation uses only existing SSAPy-Toolkit packages. Numerical and
state modules live in ``compute``; frame/observer modules in ``coordinates``;
resource readers in ``io``; and rendering modules in ``plots``.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "2.2.2"

_EXPORTS = {
    # canonical API
    "SCHEMA_ID": ("ssapy_toolkit.eclipse_api_impl", "SCHEMA_ID"),
    "EventRequest": ("ssapy_toolkit.eclipse_api_impl", "EventRequest"),
    "RenderRequest": ("ssapy_toolkit.eclipse_api_impl", "RenderRequest"),
    "ValidationResult": ("ssapy_toolkit.eclipse_api_impl", "ValidationResult"),
    "build_event": ("ssapy_toolkit.eclipse_api_impl", "build_event"),
    "validate_event": ("ssapy_toolkit.eclipse_api_impl", "validate_event"),
    "render_product": ("ssapy_toolkit.eclipse_api_impl", "render_product"),
    "event_to_dict": ("ssapy_toolkit.eclipse_api_impl", "event_to_dict"),
    "event_from_dict": ("ssapy_toolkit.eclipse_api_impl", "event_from_dict"),
    "export_event_state": ("ssapy_toolkit.eclipse_api_impl", "export_event_state"),
    "load_event_state": ("ssapy_toolkit.eclipse_api_impl", "load_event_state"),
    # GUI job
    "EclipseRenderJob": ("ssapy_toolkit.eclipse_gui", "EclipseRenderJob"),
    "RenderCancelled": ("ssapy_toolkit.eclipse_gui", "RenderCancelled"),
    "RenderJobResult": ("ssapy_toolkit.eclipse_gui", "RenderJobResult"),
    "RenderProgress": ("ssapy_toolkit.eclipse_gui", "RenderProgress"),
    # observer
    "ObserverConfig": ("ssapy_toolkit.coordinates.eclipse_observer_geometry", "ObserverConfig"),
    "ObserverState": ("ssapy_toolkit.coordinates.eclipse_observer_geometry", "ObserverState"),
    "HorizonProfile": ("ssapy_toolkit.coordinates.eclipse_observer_geometry", "HorizonProfile"),
    # core finite-source functions
    "apparent_angular_radius": ("ssapy_toolkit.compute.eclipse_core", "apparent_angular_radius"),
    "apparent_disk_geometry": ("ssapy_toolkit.compute.eclipse_core", "apparent_disk_geometry"),
    "circle_overlap_visible_fraction": ("ssapy_toolkit.compute.eclipse_core", "circle_overlap_visible_fraction"),
    "finite_source_visibility": ("ssapy_toolkit.compute.eclipse_core", "finite_source_visibility"),
    "finite_source_irradiance": ("ssapy_toolkit.compute.eclipse_core", "finite_source_irradiance"),
    "ray_sphere_intersections": ("ssapy_toolkit.compute.eclipse_core", "ray_sphere_intersections"),
    "ray_ellipsoid_intersections": ("ssapy_toolkit.compute.eclipse_core", "ray_ellipsoid_intersections"),
    "first_positive_intersection": ("ssapy_toolkit.compute.eclipse_core", "first_positive_intersection"),
    "shadow_cone": ("ssapy_toolkit.compute.eclipse_core", "shadow_cone"),
    "shadow_cross_section": ("ssapy_toolkit.compute.eclipse_core", "shadow_cross_section"),
    "sample_spherical_photosphere": ("ssapy_toolkit.compute.eclipse_core", "sample_spherical_photosphere"),
    # products
    "generate_solar_observer_products": ("ssapy_toolkit.plots.eclipse_observer_products", "generate_solar_observer_products"),
    "generate_event_scientific_summary": ("ssapy_toolkit.plots.eclipse_scientific_suite", "generate_event_scientific_summary"),
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
