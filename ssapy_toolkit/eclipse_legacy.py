"""Quarantined compatibility imports for pre-V19 eclipse renderers.
Nothing in this module is part of the canonical public API.  Importing a name
emits ``DeprecationWarning`` and forwards to the maintained implementation so
old notebooks can be migrated deliberately without polluting
the ``ssapy_toolkit.eclipse`` root namespace.
"""
from __future__ import annotations
from importlib import import_module
_MODULES = {
    "validated_animation": "ssapy_toolkit.plots.eclipse_animation",
    "interactive_dashboard": "ssapy_toolkit.plots.eclipse_interactive_dashboard",
    "cinematic_light_3d": "ssapy_toolkit.plots.eclipse_cinematic_light_3d",
    # Registered so the entries below resolve. Without them the lookup raises
    # KeyError rather than the documented DeprecationWarning, which affected
    # 14 of the 19 names in _LEGACY. The target modules all existed already.
    "system_raytrace_3d": "ssapy_toolkit.plots.eclipse_system_raytrace_3d",
    "ultra_public_3d": "ssapy_toolkit.plots.eclipse_ultra_public_3d",
    "ultra_public_animation": "ssapy_toolkit.plots.eclipse_ultra_public_animation",
    "local_finite_geometry": "ssapy_toolkit.compute.eclipse_local_finite_geometry",
    "eclipse_state": "ssapy_toolkit.compute.eclipse_state",
    "validated_eclipses": "ssapy_toolkit.compute.eclipse_reference_events",
    "validated_raytrace": "ssapy_toolkit.compute.eclipse_raytrace",
}
import warnings
_LEGACY = {
    "generate_all_validated_outputs": ("validated_animation", "generate_all_validated_outputs"),
    "generate_interactive_dashboard": ("interactive_dashboard", "generate_interactive_dashboard"),
    "generate_system_animation": ("system_raytrace_3d", "generate_system_animation"),
    "generate_system_peak_view": ("system_raytrace_3d", "generate_system_peak_view"),
    "generate_public_ultra_view": ("ultra_public_3d", "generate_public_ultra_view"),
    "generate_public_ultra_animation": ("ultra_public_animation", "generate_public_ultra_animation"),
    "generate_interactive_geometry": ("local_finite_geometry", "generate_interactive_geometry"),
    "generate_validation_figure": ("local_finite_geometry", "generate_validation_figure"),
    "generate_cinematic_view": ("cinematic_light_3d", "generate_cinematic_view"),
    "generate_cinematic_suite": ("cinematic_light_3d", "generate_cinematic_suite"),
    "build_cinematic_payload": ("cinematic_light_3d", "build_cinematic_payload"),
    "event_positions_gcrf": ("eclipse_state", "event_positions_gcrf"),
    "build_reference_event": ("validated_eclipses", "build_reference_event"),
    "build_solar_local_event": ("validated_eclipses", "build_solar_local_event"),
    "solar_besselian_state": ("validated_eclipses", "solar_besselian_state"),
    "solar_central_line_wgs84": ("validated_eclipses", "solar_central_line_wgs84"),
    "trace_reference_rays": ("validated_raytrace", "trace_reference_rays"),
    "trace_gcrf_rays": ("validated_raytrace", "trace_gcrf_rays"),
    "trace_event_rays": ("validated_raytrace", "trace_event_rays"),
}


def __getattr__(name: str):
    if name not in _LEGACY:
        raise AttributeError(name)
    module_name, attribute = _LEGACY[name]
    target = _MODULES.get(module_name)
    if target is None:
        # A mapping exists but names no module: report that rather than
        # letting a bare KeyError surface from inside an attribute lookup.
        raise AttributeError(
            f"{name} maps to the unregistered module key {module_name!r}; "
            "add it to _MODULES"
        )
    warnings.warn(
        f"ssapy_toolkit.eclipse_legacy.{name} is deprecated; use the canonical "
        "build_event/search_eclipses/render_product API",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(target), attribute)


__all__ = []
