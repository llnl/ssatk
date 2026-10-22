"""Contact-to-contact scientific Plotly eclipse animation.

Every animation state is rebuilt from the validated event geometry.  The
Earth orientation, Moon centre and synchronous attitude, finite-Sun common
tangents, first-surface ray stops, optical volumes, and target-surface shadow
are therefore functions of UTC rather than decorative motion applied to a
frozen greatest-eclipse scene.

The kilometre-scale eclipse scene, the one-AU true-scale scene, and the exact
impact-plane cross-section are separate Plotly figures with a shared HTML
transport.  This preserves WebGL precision while keeping all three views on
the same contact-aware timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ssatk.compute.eclipse_state import build_event, resample_event
from ssatk.plots.eclipse_system_raytrace_3d import _build_scene_geometry, _build_inertial_state
from ssatk.plots.eclipse_ultra_public_3d import (
    PublicPeakState,
    _impact_basis_gcrf,
    build_public_frame_state,
)
from ssatk.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    SOLAR_2024,
    ReferenceDefinition,
    ReferenceEvent,
    jd_to_datetime,
)


@dataclass(frozen=True)
class PublicAnimation:
    event: ReferenceEvent
    states: tuple[PublicPeakState, ...]
    greatest_state: PublicPeakState
    bounds_km: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    frame_names: tuple[str, ...]
    frame_labels: tuple[str, ...]
    phase_labels: tuple[str, ...]


def _event_for_animation(
    kind: str | ReferenceDefinition | ReferenceEvent,
    *,
    n_frames: int,
    solar_scope: str = "local",
    backend: str = "ssapy",
    observer: object | None = None,
    observer_contacts: dict[str, float] | None = None,
) -> ReferenceEvent:
    if isinstance(kind, ReferenceEvent):
        event = kind
        definition = event.definition
        requested = max(25, int(n_frames))
        if requested != len(event.jd):
            contacts = sorted(float(v) for v in event.contacts_jd.values())
            start, stop = contacts[0], contacts[-1]
            u = np.linspace(0.0, 1.0, requested)
            base = start + (stop-start)*(0.5-0.5*np.cos(np.pi*u))
            offsets = event.greatest_jd + np.array([-900,-600,-300,-120,-60,-20,0,20,60,120,300,600,900])/86400.0
            exact = np.concatenate([np.asarray(contacts), offsets])
            exact = exact[(exact >= start) & (exact <= stop)]
            keep = np.ones(base.shape, dtype=bool)
            for value in exact:
                keep &= np.abs(base-value) > 0.05/86400.0
            event = resample_event(event, np.sort(np.unique(np.concatenate([base[keep], exact]))))
    else:
        if isinstance(kind, ReferenceDefinition):
            definition = kind
        else:
            definition = SOLAR_2024 if str(kind).lower().startswith("solar") else LUNAR_2025
        event = build_event(
            definition,
            backend=backend,
            n_frames=max(25, int(n_frames)),
            solar_scope=solar_scope,
            observer=observer,
        )
    if definition.mode == "solar" and str(solar_scope).lower() == "local" and observer_contacts:
        from ssatk.coordinates.eclipse_observer_geometry import ObserverConfig, contact_aware_observer_jd
        site = ObserverConfig.from_value(observer or event.metadata.get("observer_model"))
        if site is None:
            raise ValueError("observer_contacts require a canonical ObserverConfig")
        times = contact_aware_observer_jd(
            event, site, n_frames=max(25, int(n_frames)), contacts=observer_contacts,
        )
        event = build_event(
            definition, backend=backend, jd=times, solar_scope="local", observer=site,
        )
        metadata = dict(event.metadata)
        metadata.update({
            "observer_contacts_jd": dict(observer_contacts),
            "local_max_jd": float(observer_contacts.get("MAX", event.greatest_jd)),
            "observer_contact_backend": "selected ephemeris/frame backend + optional topographic lunar limb",
        })
        event = replace(event, metadata=metadata)
    return event


def _phase_label(event: ReferenceEvent, jd: float) -> str:
    """Contact-aware label supporting total, annular, partial, and penumbral events."""
    value = float(jd)
    tol = 0.75 / 86400.0
    contacts = dict(event.contacts_jd)
    for name, contact in sorted(contacts.items(), key=lambda item: float(item[1])):
        if abs(value - float(contact)) <= tol:
            return str(name)
    if abs(value - float(event.greatest_jd)) <= tol:
        return "MAX"

    if event.mode == "solar":
        local = str(event.metadata.get("solar_scope", "global")) == "local"
        first_name, last_name = (("C1", "C4") if local else ("P1", "P4"))
        first = float(contacts.get(first_name, min(contacts.values())))
        last = float(contacts.get(last_name, max(contacts.values())))
        ingress_central = contacts.get("C2" if local else "U1")
        egress_central = contacts.get("C3" if local else "U4")
        if value < first:
            return f"before {first_name}"
        if ingress_central is not None and egress_central is not None:
            if value < float(ingress_central):
                return "partial ingress"
            if value < event.greatest_jd:
                return "central phase — ingress"
            if value < float(egress_central):
                return "central phase — egress"
            if value < last:
                return "partial egress"
        else:
            if value < event.greatest_jd:
                return "partial ingress"
            if value < last:
                return "partial egress"
        return f"after {last_name}"

    p1 = float(contacts.get("P1", min(contacts.values())))
    p4 = float(contacts.get("P4", max(contacts.values())))
    u1, u4 = contacts.get("U1"), contacts.get("U4")
    u2, u3 = contacts.get("U2"), contacts.get("U3")
    if value < p1:
        return "before P1"
    if u1 is None or u4 is None:
        return "penumbral ingress" if value < event.greatest_jd else (
            "penumbral egress" if value < p4 else "after P4"
        )
    if value < float(u1):
        return "penumbral ingress"
    if u2 is None or u3 is None:
        if value < event.greatest_jd:
            return "partial umbral ingress"
        if value < float(u4):
            return "partial umbral egress"
        if value < p4:
            return "penumbral egress"
        return "after P4"
    if value < float(u2):
        return "partial umbral ingress"
    if value < event.greatest_jd:
        return "totality — ingress"
    if value < float(u3):
        return "totality — egress"
    if value < float(u4):
        return "partial umbral egress"
    if value < p4:
        return "penumbral egress"
    return "after P4"


def _animation_bounds(states: Sequence[PublicPeakState]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    centers = np.asarray([state.moon_local_km for state in states], dtype=float)
    x_min = min(0.0, float(np.min(centers[:, 0])))-36_000.0
    x_max = max(0.0, float(np.max(centers[:, 0])))+36_000.0
    transverse = max(
        4.6*float(EARTH_AXES_KM[0]),
        float(np.max(np.abs(centers[:, 1])))+22_000.0,
        float(np.max(np.abs(centers[:, 2])))+22_000.0,
    )
    return (x_min, x_max), (-transverse, transverse), (-transverse, transverse)


def build_public_animation(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    *,
    n_frames: int = 39,
    solar_scope: str = "local",
    ray_azimuth: int = 48,
    backend: str = "ssapy",
    observer: object | None = None,
    observer_contacts: dict[str, float] | None = None,
) -> PublicAnimation:
    event = _event_for_animation(
        kind, n_frames=n_frames, solar_scope=solar_scope, backend=backend,
        observer=observer, observer_contacts=observer_contacts,
    )
    peak_event = (
        resample_event(event, [event.greatest_jd])
        if bool(event.metadata.get("dynamic_event", False))
        else build_event(
            event.definition,
            backend=str(event.metadata.get("state_source", "reference")),
            jd=[event.greatest_jd],
            solar_scope=solar_scope,
            observer=observer or event.metadata.get("observer_model"),
        )
    )
    geometry = _build_scene_geometry(peak_event)
    peak_raw = _build_inertial_state(
        peak_event,
        peak_event.greatest_jd,
        geometry.reference_cross_axis_gcrf,
        n_azimuth=max(24, int(ray_azimuth)),
    )
    basis, _ = _impact_basis_gcrf(peak_event, peak_raw.bundle_native)
    states = tuple(
        build_public_frame_state(
            event,
            float(jd),
            basis_world_from_scene=basis,
            reference_cross_axis_gcrf=geometry.reference_cross_axis_gcrf,
            ray_azimuth=max(24, int(ray_azimuth)),
            run_ssapy_audit=False,
        )
        for jd in event.jd
    )
    peak_index = int(np.argmin(np.abs(event.jd-event.greatest_jd)))
    greatest = states[peak_index]
    names = tuple(f"utc-{index:03d}" for index in range(len(states)))
    labels = tuple(jd_to_datetime(state.jd_utc).strftime("%H:%M:%S UTC") for state in states)
    phases = tuple(_phase_label(event, state.jd_utc) for state in states)
    return PublicAnimation(
        event=event,
        states=states,
        greatest_state=greatest,
        bounds_km=_animation_bounds(states),
        frame_names=names,
        frame_labels=labels,
        phase_labels=phases,
    )
