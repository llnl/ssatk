"""Finite-Sun solar and lunar eclipse events, validation, and products.

Event states and physics live in ``ssatk.compute.eclipse_*``, observer and
lunar geometry in ``ssatk.coordinates.eclipse_*``, and renderers in
``ssatk.plots.eclipse_*``. Names here are imported on first use so that
``import ssatk.eclipse`` stays cheap.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "SCHEMA_ID": "ssatk.eclipse_api_impl",
    "EventRequest": "ssatk.eclipse_api_impl",
    "RenderRequest": "ssatk.eclipse_api_impl",
    "ValidationResult": "ssatk.eclipse_api_impl",
    "build_event": "ssatk.eclipse_api_impl",
    "validate_event": "ssatk.eclipse_api_impl",
    "render_product": "ssatk.eclipse_api_impl",
    "event_to_dict": "ssatk.eclipse_api_impl",
    "event_from_dict": "ssatk.eclipse_api_impl",
    "export_event_state": "ssatk.eclipse_api_impl",
    "load_event_state": "ssatk.eclipse_api_impl",
    "discover_eclipses": "ssatk.compute.eclipse_event_search",
    "build_discovered_event": "ssatk.compute.eclipse_event_search",
    "ObserverConfig": "ssatk.coordinates.eclipse_observer_geometry",
    "ObserverState": "ssatk.coordinates.eclipse_observer_geometry",
    "HorizonProfile": "ssatk.coordinates.eclipse_observer_geometry",
    "apparent_angular_radius": "ssatk.compute.eclipse_core",
    "apparent_disk_geometry": "ssatk.compute.eclipse_core",
    "circle_overlap_visible_fraction": "ssatk.compute.eclipse_core",
    "finite_source_visibility": "ssatk.compute.eclipse_core",
    "finite_source_irradiance": "ssatk.compute.eclipse_core",
    "ray_sphere_intersections": "ssatk.compute.eclipse_core",
    "ray_ellipsoid_intersections": "ssatk.compute.eclipse_core",
    "first_positive_intersection": "ssatk.compute.eclipse_core",
    "shadow_cone": "ssatk.compute.eclipse_core",
    "shadow_cross_section": "ssatk.compute.eclipse_core",
    "sample_spherical_photosphere": "ssatk.compute.eclipse_core",
    "generate_solar_observer_products": "ssatk.plots.eclipse_observer_products",
    "generate_event_scientific_summary": "ssatk.plots.eclipse_scientific_suite",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
