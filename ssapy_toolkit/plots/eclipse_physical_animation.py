"""Compatibility exports for the corrected physical eclipse animations."""
from ssapy_toolkit.plots.eclipse_animation import (
    FrameRenderer,
    generate_all_validated_outputs,
    generate_event_3d_animation,
    generate_event_animation,
    generate_peak_3d,
    generate_validation_report,
)
from ssapy_toolkit.compute.eclipse_reference_events import (
    build_reference_event,
    build_solar_local_event,
)
from ssapy_toolkit.compute.eclipse_raytrace import (
    bundle_penetrations,
    tangent_residuals,
    trace_reference_rays,
)

__all__ = [
    "FrameRenderer",
    "build_reference_event",
    "build_solar_local_event",
    "trace_reference_rays",
    "bundle_penetrations",
    "tangent_residuals",
    "generate_event_animation",
    "generate_event_3d_animation",
    "generate_peak_3d",
    "generate_validation_report",
    "generate_all_validated_outputs",
]
