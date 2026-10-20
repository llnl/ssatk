"""Finite-Sun ray tracing for the NASA-reference eclipse events.

Direct light is never drawn behind an opaque body.  The central ray ends at
its first solid-surface hit.  Boundary rays are common tangents from the real
solar photosphere to the optical limb of the occluder and are clipped at the
first target-surface hit.  A boundary ray that misses the target ends at the
target-centre plane; it is never extended through the target for appearance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ssatk.compute.eclipse_brightness import gcrf_to_itrf_km, itrf_to_gcrf_km
from ssatk.compute.eclipse_core import ray_sphere_intersections as _core_ray_sphere_intersections
from ssatk.compute.eclipse_state import event_gcrf_to_itrf_km, event_itrf_to_gcrf_km, event_positions_gcrf

from ssatk.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_PENUMBRA_OPTICAL_RADIUS_KM,
    SOLAR_UMBRA_OPTICAL_RADIUS_KM,
    ReferenceDefinition,
    SolarBesselianState,
    lunar_reference_state,
    solar_besselian_state,
    _LUNAR_MOON_DISTANCE_KM,
    _LUNAR_MOON_SD_DEG,
    _LUNAR_P_RADIUS_DEG,
    _LUNAR_SUN_DISTANCE_KM,
    _LUNAR_U_RADIUS_DEG,
    _ray_ellipsoid_roots,
    geodetic_from_itrf,
    _unit,
)


def _ray_sphere_roots(origin, direction, center, radius) -> np.ndarray:
    """Stable sphere roots from the canonical :mod:`ssatk.compute.eclipse_core` kernel."""
    roots = np.asarray(
        _core_ray_sphere_intersections(origin, direction, radius, center=center),
        dtype=float,
    ).reshape(-1)
    return np.sort(roots[np.isfinite(roots)])


def _basis(axis) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis, name="axis")
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(ref, axis), name="transverse basis")
    v = _unit(np.cross(axis, u), name="transverse basis")
    return u, v


def _target_plane_endpoint(origin, direction, target_center, axis) -> np.ndarray:
    origin = np.asarray(origin, dtype=float)
    direction = _unit(direction, name="ray direction")
    target = np.asarray(target_center, dtype=float)
    axis = _unit(axis, name="axis")
    denom = float(np.dot(direction, axis))
    if abs(denom) < 1.0e-14:
        distance = max(float(np.dot(target-origin, direction)), 0.0)
    else:
        distance = max(float(np.dot(target-origin, axis))/denom, 0.0)
    return origin+distance*direction


def _lunar_effective_earth_radius_km() -> float:
    """Effective Danjon shadow radius matching both NASA U/P radii."""
    Dm = _LUNAR_MOON_DISTANCE_KM
    Ds = _LUNAR_SUN_DISTANCE_KM
    Ru = Dm*math.sin(math.radians(_LUNAR_U_RADIUS_DEG))
    Rp = Dm*math.sin(math.radians(_LUNAR_P_RADIUS_DEG))
    from_umbra = (Ru+Dm*R_SUN_KM/Ds)/(1.0+Dm/Ds)
    from_penumbra = (Rp-Dm*R_SUN_KM/Ds)/(1.0+Dm/Ds)
    return 0.5*(from_umbra+from_penumbra)


LUNAR_DANJON_EARTH_RADIUS_KM = _lunar_effective_earth_radius_km()
LUNAR_OPTICAL_MOON_RADIUS_KM = _LUNAR_MOON_DISTANCE_KM*math.sin(
    math.radians(_LUNAR_MOON_SD_DEG)
)


@dataclass(frozen=True)
class RayPath:
    kind: str
    points_km: np.ndarray
    target_hit: bool
    azimuth_rad: float | None = None

    @property
    def source_km(self) -> np.ndarray:
        return self.points_km[0]

    @property
    def endpoint_km(self) -> np.ndarray:
        return self.points_km[-1]


@dataclass(frozen=True)
class ReferenceRayBundle:
    mode: str
    jd_utc: float
    frame: str
    sun_center_km: np.ndarray
    moon_center_km: np.ndarray
    earth_center_km: np.ndarray
    axis_hat: np.ndarray
    occluder_name: str
    target_name: str
    solid_occluder_radius_km: float
    umbra_optical_radius_km: float
    penumbra_optical_radius_km: float
    target_radius_km: float
    central: RayPath
    umbra: tuple[RayPath, ...]
    penumbra: tuple[RayPath, ...]
    umbra_tangent_points_km: np.ndarray
    penumbra_tangent_points_km: np.ndarray
    metadata: dict[str, float | str]
    gcrf_to_itrf_rotation: np.ndarray | None = None

    @property
    def all_paths(self) -> tuple[RayPath, ...]:
        return (self.central,)+self.umbra+self.penumbra


def _first_sphere_hit(origin, direction, center, radius,
                      *, min_distance: float = 1.0e-7) -> tuple[float, np.ndarray] | None:
    roots = _ray_sphere_roots(origin, direction, center, radius)
    roots = roots[roots > min_distance]
    if not roots.size:
        return None
    distance = float(roots[0])
    origin = np.ravel(np.asarray(origin, dtype=float))
    return distance, origin+distance*_unit(direction)


def _first_earth_hit(origin, direction, *, min_distance: float = 1.0e-7) -> tuple[float, np.ndarray] | None:
    roots = _ray_ellipsoid_roots(origin, direction, EARTH_AXES_KM)
    roots = roots[roots > min_distance]
    if not roots.size:
        return None
    distance = float(roots[0])
    origin = np.ravel(np.asarray(origin, dtype=float))
    return distance, origin+distance*_unit(direction)


def _first_earth_hit_gcrf(
    origin,
    direction,
    jd_utc: float,
    *,
    min_distance: float = 1.0e-7,
    prefer_toolkit: bool = True,
    event=None,
) -> tuple[float, np.ndarray] | None:
    """First WGS-84 Earth hit for a GCRF ray at one fixed instant.

    The ray is rotated into ITRF before solving the ellipsoid intersection,
    then the hit point is returned to GCRF.  A unit step is transformed with
    the origin so the direction follows exactly the same frame rotation.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = _unit(direction, name="GCRF ray direction")
    # Transform origin and direction independently.  A one-kilometre helper
    # step next to a ~1-AU origin suffers cancellation after rotation and can
    # perturb the surface intersection.  GCRF/ITRF share the geocentre, so the
    # same rotation applies directly to the direction vector.
    if event is not None:
        origin_itrf = np.asarray(
            event_gcrf_to_itrf_km(event, origin, float(jd_utc)), dtype=float,
        )
        direction_itrf = _unit(
            event_gcrf_to_itrf_km(event, direction, float(jd_utc)),
            name="ITRF ray direction",
        )
    else:
        origin_itrf = np.asarray(
            gcrf_to_itrf_km(origin, float(jd_utc), prefer_toolkit=prefer_toolkit),
            dtype=float,
        )
        direction_itrf = _unit(
            gcrf_to_itrf_km(direction, float(jd_utc), prefer_toolkit=prefer_toolkit),
            name="ITRF ray direction",
        )
    # gcrf_to_itrf_km normalises a single point to (1, 3); the geometry
    # below works in plain vectors, so flatten at the boundary.
    origin_itrf = np.ravel(origin_itrf)
    hit = _first_earth_hit(
        origin_itrf, direction_itrf, min_distance=min_distance,
    )
    if hit is None:
        return None
    distance, endpoint_itrf = hit
    if event is not None:
        endpoint_gcrf = np.asarray(
            event_itrf_to_gcrf_km(event, endpoint_itrf, float(jd_utc)),
            dtype=float,
        )
    else:
        endpoint_gcrf = np.asarray(
            itrf_to_gcrf_km(
                endpoint_itrf,
                float(jd_utc),
                prefer_toolkit=prefer_toolkit,
                prefer_astropy=True,
            ),
            dtype=float,
        )
    return float(distance), np.ravel(endpoint_gcrf)


def _common_tangent_family(
    sun_center,
    occ_center,
    axis,
    occ_radius,
    target_center,
    *,
    family: str,
    n_azimuth: int,
    target_intersector,
) -> tuple[tuple[RayPath, ...], np.ndarray]:
    """Build exact common tangents for one spherical optical limb."""
    sun = np.asarray(sun_center, dtype=float)
    occ = np.asarray(occ_center, dtype=float)
    axis = _unit(axis, name="Sun-to-occluder axis")
    distance = float(np.linalg.norm(occ-sun))
    if family == "umbra":
        ratio = (R_SUN_KM-float(occ_radius))/distance
        source_sign = +1.0
        radial_sign = -1.0
        tangent_axis_sign = +1.0
    elif family == "penumbra":
        ratio = (R_SUN_KM+float(occ_radius))/distance
        source_sign = -1.0
        radial_sign = +1.0
        tangent_axis_sign = -1.0
    else:
        raise ValueError("family must be 'umbra' or 'penumbra'")
    half_angle = math.asin(np.clip(ratio, -1.0, 1.0))
    sin_a, cos_a = math.sin(half_angle), math.cos(half_angle)
    u, v = _basis(axis)
    paths: list[RayPath] = []
    tangent_points: list[np.ndarray] = []
    for phi in np.linspace(0.0, 2.0*math.pi, max(4, int(n_azimuth)), endpoint=False):
        radial = math.cos(phi)*u+math.sin(phi)*v
        normal = tangent_axis_sign*sin_a*axis+cos_a*radial
        direction = cos_a*axis+radial_sign*sin_a*radial
        direction = _unit(direction)
        source = sun+source_sign*R_SUN_KM*normal
        tangent = occ+float(occ_radius)*normal
        # Probe just downstream of tangency so the zero-distance tangent root
        # cannot be mistaken for a penetration by finite precision.
        probe = tangent+direction*1.0e-6
        hit = target_intersector(probe, direction)
        if hit is None:
            endpoint = _target_plane_endpoint(probe, direction, target_center, axis)
            target_hit = False
        else:
            _, endpoint = hit
            target_hit = True
        paths.append(RayPath(
            kind=family,
            points_km=np.vstack([source, tangent, endpoint]),
            target_hit=target_hit,
            azimuth_rad=float(phi),
        ))
        tangent_points.append(tangent)
    return tuple(paths), np.asarray(tangent_points)


def tangent_cross_section_paths(
    bundle: ReferenceRayBundle,
    family: str,
    cross_axis_world,
) -> tuple[RayPath, RayPath, np.ndarray]:
    """Return an exact, continuous upper/lower tangent pair in one plane.

    Interactive plots previously selected the nearest two rays from a sampled
    azimuth ring.  As the eclipse advanced, the selected sample index could
    switch by one bin, making the displayed lines jump or appear to cross.
    This routine evaluates the common-tangent construction directly in the
    caller's fixed optical cross-section, so the same physical plane is used
    in every animation frame with no azimuth quantisation.

    Parameters
    ----------
    bundle
        Validated ray bundle for one UTC instant.
    family
        ``"umbra"`` or ``"penumbra"``.
    cross_axis_world
        World-frame vector that defines screen-up in the desired optical
        plane.  Its component along the shadow axis is removed exactly.
    """
    key = str(family).lower()
    if key not in {"umbra", "penumbra"}:
        raise ValueError("family must be 'umbra' or 'penumbra'")

    axis = _unit(bundle.axis_hat, name="shadow axis")
    radial = np.asarray(cross_axis_world, dtype=float).reshape(3)
    radial = radial-axis*float(np.dot(radial, axis))
    if np.linalg.norm(radial) < 1.0e-12:
        _, radial = _basis(axis)
    radial = _unit(radial, name="cross-section radial")

    sun = np.asarray(bundle.sun_center_km, dtype=float)
    occ = (np.asarray(bundle.moon_center_km, dtype=float)
           if bundle.mode == "solar" else np.asarray(bundle.earth_center_km, dtype=float))
    target = (np.asarray(bundle.earth_center_km, dtype=float)
              if bundle.mode == "solar" else np.asarray(bundle.moon_center_km, dtype=float))
    occ_radius = (bundle.umbra_optical_radius_km
                  if key == "umbra" else bundle.penumbra_optical_radius_km)
    distance = float(np.linalg.norm(occ-sun))

    if key == "umbra":
        ratio = (R_SUN_KM-float(occ_radius))/distance
        source_sign = +1.0
        radial_sign = -1.0
        tangent_axis_sign = +1.0
    else:
        ratio = (R_SUN_KM+float(occ_radius))/distance
        source_sign = -1.0
        radial_sign = +1.0
        tangent_axis_sign = -1.0
    half_angle = math.asin(np.clip(ratio, -1.0, 1.0))
    sin_a, cos_a = math.sin(half_angle), math.cos(half_angle)

    if bundle.mode == "solar":
        if bundle.gcrf_to_itrf_rotation is None:
            target_intersector = _first_earth_hit
        else:
            rotation = bundle.gcrf_to_itrf_rotation

            def target_intersector(origin, direction):
                hit = _first_earth_hit(origin @ rotation, direction @ rotation)
                return None if hit is None else (hit[0], hit[1] @ rotation.T)
    else:
        target_intersector = lambda origin, direction: _first_sphere_hit(
            origin, direction, target, bundle.target_radius_km
        )

    paths: list[RayPath] = []
    tangents: list[np.ndarray] = []
    for sign in (+1.0, -1.0):
        radial_signed = sign*radial
        normal = tangent_axis_sign*sin_a*axis+cos_a*radial_signed
        direction = _unit(cos_a*axis+radial_sign*sin_a*radial_signed)
        source = sun+source_sign*R_SUN_KM*normal
        tangent = occ+float(occ_radius)*normal
        probe = tangent+direction*1.0e-6
        hit = target_intersector(probe, direction)
        if hit is None:
            endpoint = _target_plane_endpoint(probe, direction, target, axis)
            target_hit = False
        else:
            _, endpoint = hit
            target_hit = True
        paths.append(RayPath(
            kind=key,
            points_km=np.vstack([source, tangent, endpoint]),
            target_hit=target_hit,
            azimuth_rad=None,
        ))
        tangents.append(tangent)
    return paths[0], paths[1], np.asarray(tangents)


def trace_reference_rays(kind: str | ReferenceDefinition, jd_utc: float,
                         *, n_azimuth: int = 12) -> ReferenceRayBundle:
    """Trace a validated solar or lunar eclipse at one UTC Julian Date."""
    if isinstance(kind, ReferenceDefinition):
        definition = kind
    else:
        key = str(kind).lower()
        if key in ("solar", "solar_2024", SOLAR_2024.key):
            definition = SOLAR_2024
        elif key in ("lunar", "lunar_2025", LUNAR_2025.key):
            definition = LUNAR_2025
        else:
            raise ValueError("kind must identify the validated 2024 solar or 2025 lunar eclipse")
    earth = np.zeros(3)
    if definition.mode == "solar":
        state: SolarBesselianState = solar_besselian_state(jd_utc)
        sun, moon = state.sun_itrf_km, state.moon_itrf_km
        occ_center, target_center = moon, earth
        axis = _unit(moon-sun)
        # The axial sunlight ray hits the lunar surface, so use the mean
        # solid-body radius.  NASA k1/k2 remain confined to the two
        # eclipse-limb tangent families; k2 is not a subsolar surface radius.
        solid_radius = R_MOON_MEAN_KM
        umbra_radius = SOLAR_UMBRA_OPTICAL_RADIUS_KM
        penumbra_radius = SOLAR_PENUMBRA_OPTICAL_RADIUS_KM
        target_radius = RE_KM
        target_intersector = _first_earth_hit
        occ_name, target_name = "Moon", "Earth"
        frame = "WGS84 Earth-fixed"
        metadata = {
            "moon_optical_radius_umbra_km": umbra_radius,
            "moon_optical_radius_penumbra_km": penumbra_radius,
            "sun_earth_distance_km": float(np.linalg.norm(sun)),
            "sun_moon_distance_km": float(np.linalg.norm(sun-moon)),
            "source": "NASA/GSFC Besselian elements",
        }
        # The central axis is constructed through the lunar center.  Place
        # the first hit analytically on the Sun-facing mean solid surface
        # instead of solving a 1-AU quadratic and losing sub-metre precision.
        central_endpoint = moon-axis*solid_radius
        central_hit = (float(np.linalg.norm(central_endpoint-(sun+axis*R_SUN_KM))),
                       central_endpoint)
    else:
        state = lunar_reference_state(jd_utc)
        sun, moon = state.sun_gcrf_km, state.moon_gcrf_km
        occ_center, target_center = earth, moon
        axis = _unit(earth-sun)
        solid_radius = RE_KM
        umbra_radius = LUNAR_DANJON_EARTH_RADIUS_KM
        penumbra_radius = LUNAR_DANJON_EARTH_RADIUS_KM
        target_radius = LUNAR_OPTICAL_MOON_RADIUS_KM
        target_intersector = lambda origin, direction: _first_sphere_hit(
            origin, direction, moon, target_radius
        )
        occ_name, target_name = "Earth", "Moon"
        frame = "GCRF-like apparent equatorial"
        metadata = {
            "solid_earth_radius_km": RE_KM,
            "danjon_effective_earth_radius_km": LUNAR_DANJON_EARTH_RADIUS_KM,
            "danjon_atmospheric_extension_km": LUNAR_DANJON_EARTH_RADIUS_KM-RE_KM,
            "sun_earth_distance_km": float(np.linalg.norm(sun)),
            "moon_earth_distance_km": float(np.linalg.norm(moon)),
            "source": "NASA/GSFC Danjon-rule lunar shadow",
        }
        central_hit = _first_earth_hit(sun+axis*R_SUN_KM, axis, min_distance=0.0)
    if central_hit is None:
        raise RuntimeError(f"Central sunlight did not hit the {occ_name}")
    _, central_endpoint = central_hit
    central = RayPath(
        kind=("direct sunlight (terminates at mean lunar surface)"
              if definition.mode == "solar"
              else "direct sunlight (terminates at WGS-84 Earth)"),
        points_km=np.vstack([sun+axis*R_SUN_KM, central_endpoint]),
        target_hit=True,
        azimuth_rad=None,
    )
    umbra, tangent_u = _common_tangent_family(
        sun, occ_center, axis, umbra_radius, target_center,
        family="umbra", n_azimuth=n_azimuth,
        target_intersector=target_intersector,
    )
    penumbra, tangent_p = _common_tangent_family(
        sun, occ_center, axis, penumbra_radius, target_center,
        family="penumbra", n_azimuth=n_azimuth,
        target_intersector=target_intersector,
    )
    return ReferenceRayBundle(
        mode=definition.mode,
        jd_utc=float(jd_utc),
        frame=frame,
        sun_center_km=np.asarray(sun),
        moon_center_km=np.asarray(moon),
        earth_center_km=earth,
        axis_hat=axis,
        occluder_name=occ_name,
        target_name=target_name,
        solid_occluder_radius_km=float(solid_radius),
        umbra_optical_radius_km=float(umbra_radius),
        penumbra_optical_radius_km=float(penumbra_radius),
        target_radius_km=float(target_radius),
        central=central,
        umbra=umbra,
        penumbra=penumbra,
        umbra_tangent_points_km=tangent_u,
        penumbra_tangent_points_km=tangent_p,
        metadata=metadata,
    )


def trace_gcrf_rays(
    kind: str | ReferenceDefinition,
    jd_utc: float,
    sun_gcrf_km,
    moon_gcrf_km,
    *,
    n_azimuth: int = 12,
    source_label: str = "LLNL SSAPy DE430",
    prefer_toolkit: bool = True,
    event=None,
) -> ReferenceRayBundle:
    """Trace finite-Sun eclipse rays from explicit GCRF body centres."""
    if isinstance(kind, ReferenceDefinition):
        definition = kind
    else:
        key = str(kind).lower()
        if key in ("solar", "solar_2024", SOLAR_2024.key):
            definition = SOLAR_2024
        elif key in ("lunar", "lunar_2025", LUNAR_2025.key):
            definition = LUNAR_2025
        else:
            raise ValueError("kind must identify the validated 2024 solar or 2025 lunar eclipse")

    earth = np.zeros(3, dtype=float)
    sun = np.asarray(sun_gcrf_km, dtype=float).reshape(3)
    moon = np.asarray(moon_gcrf_km, dtype=float).reshape(3)
    if event is not None:
        gcrf_to_itrf_rotation = np.asarray(
            event_gcrf_to_itrf_km(event, np.eye(3), float(jd_utc)), dtype=float,
        ).reshape(3, 3)
    else:
        gcrf_to_itrf_rotation = np.asarray(
            gcrf_to_itrf_km(
                np.eye(3), float(jd_utc), prefer_toolkit=prefer_toolkit,
            ), dtype=float,
        ).reshape(3, 3)

    if definition.mode == "solar":
        occ_center, target_center = moon, earth
        axis = _unit(moon-sun)
        solid_radius = R_MOON_MEAN_KM
        umbra_radius = SOLAR_UMBRA_OPTICAL_RADIUS_KM
        penumbra_radius = SOLAR_PENUMBRA_OPTICAL_RADIUS_KM
        target_radius = RE_KM
        target_intersector = lambda origin, direction: _first_earth_hit_gcrf(
            origin, direction, float(jd_utc), prefer_toolkit=prefer_toolkit, event=event,
        )
        occ_name, target_name = "Moon", "Earth"
        central_endpoint = moon-axis*solid_radius
        central_hit = (
            float(np.linalg.norm(central_endpoint-(sun+axis*R_SUN_KM))),
            central_endpoint,
        )
        metadata = {
            "moon_optical_radius_umbra_km": umbra_radius,
            "moon_optical_radius_penumbra_km": penumbra_radius,
            "sun_earth_distance_km": float(np.linalg.norm(sun)),
            "sun_moon_distance_km": float(np.linalg.norm(sun-moon)),
            "source": source_label,
            "frame_transform": (
                str(event.metadata.get("frame_backend")) if event is not None
                else ("ssapy-toolkit" if prefer_toolkit else "Astropy")
            ),
        }
    else:
        occ_center, target_center = earth, moon
        axis = _unit(earth-sun)
        solid_radius = RE_KM
        umbra_radius = LUNAR_DANJON_EARTH_RADIUS_KM
        penumbra_radius = LUNAR_DANJON_EARTH_RADIUS_KM
        target_radius = LUNAR_OPTICAL_MOON_RADIUS_KM
        target_intersector = lambda origin, direction: _first_sphere_hit(
            origin, direction, moon, target_radius,
        )
        occ_name, target_name = "Earth", "Moon"
        central_hit = _first_earth_hit_gcrf(
            sun+axis*R_SUN_KM,
            axis,
            float(jd_utc),
            min_distance=0.0,
            prefer_toolkit=prefer_toolkit,
            event=event,
        )
        metadata = {
            "solid_earth_radius_km": RE_KM,
            "danjon_effective_earth_radius_km": LUNAR_DANJON_EARTH_RADIUS_KM,
            "danjon_atmospheric_extension_km": LUNAR_DANJON_EARTH_RADIUS_KM-RE_KM,
            "sun_earth_distance_km": float(np.linalg.norm(sun)),
            "moon_earth_distance_km": float(np.linalg.norm(moon)),
            "source": source_label,
            "frame_transform": (
                str(event.metadata.get("frame_backend")) if event is not None
                else ("ssapy-toolkit" if prefer_toolkit else "Astropy")
            ),
        }

    if central_hit is None:
        raise RuntimeError(f"Central sunlight did not hit the {occ_name}")
    _, central_endpoint = central_hit
    central = RayPath(
        kind=(
            "direct sunlight (terminates at mean lunar surface)"
            if definition.mode == "solar"
            else "direct sunlight (terminates at WGS-84 Earth)"
        ),
        points_km=np.vstack([sun+axis*R_SUN_KM, central_endpoint]),
        target_hit=True,
        azimuth_rad=None,
    )
    umbra, tangent_u = _common_tangent_family(
        sun, occ_center, axis, umbra_radius, target_center,
        family="umbra", n_azimuth=n_azimuth,
        target_intersector=target_intersector,
    )
    penumbra, tangent_p = _common_tangent_family(
        sun, occ_center, axis, penumbra_radius, target_center,
        family="penumbra", n_azimuth=n_azimuth,
        target_intersector=target_intersector,
    )
    return ReferenceRayBundle(
        mode=definition.mode,
        jd_utc=float(jd_utc),
        frame="GCRF",
        sun_center_km=sun,
        moon_center_km=moon,
        earth_center_km=earth,
        axis_hat=axis,
        occluder_name=occ_name,
        target_name=target_name,
        solid_occluder_radius_km=float(solid_radius),
        umbra_optical_radius_km=float(umbra_radius),
        penumbra_optical_radius_km=float(penumbra_radius),
        target_radius_km=float(target_radius),
        central=central,
        umbra=umbra,
        penumbra=penumbra,
        umbra_tangent_points_km=tangent_u,
        penumbra_tangent_points_km=tangent_p,
        metadata=metadata,
        gcrf_to_itrf_rotation=gcrf_to_itrf_rotation,
    )


def trace_event_rays(
    event,
    jd_utc: float,
    *,
    n_azimuth: int = 12,
) -> ReferenceRayBundle:
    """Trace rays with the same backend that built ``event``."""
    source = str(event.metadata.get("state_source", "reference"))
    if source == "reference":
        return trace_reference_rays(event.definition, jd_utc, n_azimuth=n_azimuth)
    sun, moon = event_positions_gcrf(event, float(jd_utc))
    return trace_gcrf_rays(
        event.definition,
        float(jd_utc),
        sun,
        moon,
        n_azimuth=n_azimuth,
        source_label=str(event.backend),
        prefer_toolkit=source == "ssapy",
        event=event,
    )


def _quadratic_open_interval_penetrates(aa: float, bb: float, cc: float,
                                         *, tolerance: float = 2.0e-10) -> bool:
    """Whether q(t)<0 for any t strictly inside a unit parameter segment."""
    if aa <= 0.0:
        return cc < -float(tolerance)
    disc = bb*bb-4.0*aa*cc
    # Zero discriminant is a tangent contact, not an interior crossing.
    scale = max(bb*bb, abs(4.0*aa*cc), 1.0)
    if disc <= float(tolerance)*scale:
        return False
    root = math.sqrt(max(disc, 0.0))
    t1, t2 = sorted(((-bb-root)/(2.0*aa), (-bb+root)/(2.0*aa)))
    # Ignore numerical slivers at a tangent or at the clipped endpoint.
    # A 1e-8 parameter tolerance is sub-metre even for the longest local
    # segments and vastly smaller than any real body crossing.
    eps = 1.0e-8
    return max(t1, eps) < min(t2, 1.0-eps)


def segment_sphere_penetrates(a, b, center, radius, *, tolerance: float = 2.0e-10) -> bool:
    """Analytic open-segment penetration test; tangent/end contacts are safe."""
    a = np.asarray(a, dtype=float)-np.asarray(center, dtype=float)
    delta = np.asarray(b, dtype=float)-np.asarray(center, dtype=float)-a
    aa = float(np.dot(delta, delta))
    bb = float(2.0*np.dot(a, delta))
    cc = float(np.dot(a, a)-float(radius)**2)
    return _quadratic_open_interval_penetrates(aa, bb, cc, tolerance=tolerance)


def segment_ellipsoid_penetrates(a, b, axes_km=EARTH_AXES_KM,
                                  *, tolerance: float = 2.0e-10) -> bool:
    """Analytic open-segment test after scaling WGS-84 to a unit sphere."""
    axes = np.asarray(axes_km, dtype=float)
    qa = np.asarray(a, dtype=float)/axes
    qb = np.asarray(b, dtype=float)/axes
    delta = qb-qa
    aa = float(np.dot(delta, delta))
    bb = float(2.0*np.dot(qa, delta))
    cc = float(np.dot(qa, qa)-1.0)
    return _quadratic_open_interval_penetrates(aa, bb, cc, tolerance=tolerance)


def bundle_penetrations(bundle: ReferenceRayBundle) -> dict[str, int]:
    """Return exact segment counts entering the solid Earth or Moon."""
    counts = {"earth": 0, "moon": 0}
    for path in bundle.all_paths:
        if bundle.mode == "solar":
            # Each ray family is audited against the radius that defines that
            # physical/model boundary: mean solid radius for the axial ray,
            # NASA k2 for umbral tangents, and NASA k1 for penumbral tangents.
            if path.kind == "umbra":
                moon_radius = bundle.umbra_optical_radius_km
            elif path.kind == "penumbra":
                moon_radius = bundle.penumbra_optical_radius_km
            else:
                moon_radius = bundle.solid_occluder_radius_km
        else:
            moon_radius = LUNAR_OPTICAL_MOON_RADIUS_KM
        for a, b in zip(path.points_km[:-1], path.points_km[1:]):
            earth_a, earth_b = a, b
            if bundle.gcrf_to_itrf_rotation is not None:
                earth_a = earth_a @ bundle.gcrf_to_itrf_rotation
                earth_b = earth_b @ bundle.gcrf_to_itrf_rotation
            if segment_ellipsoid_penetrates(earth_a, earth_b):
                counts["earth"] += 1
            if segment_sphere_penetrates(a, b, bundle.moon_center_km, moon_radius):
                counts["moon"] += 1
    return counts


def tangent_residuals(bundle: ReferenceRayBundle) -> dict[str, float]:
    """Maximum optical-radius and orthogonality errors at tangent points."""
    result: dict[str, float] = {}
    for name, paths, points, radius in (
        ("umbra", bundle.umbra, bundle.umbra_tangent_points_km,
         bundle.umbra_optical_radius_km),
        ("penumbra", bundle.penumbra, bundle.penumbra_tangent_points_km,
         bundle.penumbra_optical_radius_km),
    ):
        rel = points-(bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km)
        dirs = np.asarray([_unit(path.points_km[-1]-path.points_km[1]) for path in paths])
        result[f"{name}_radius_error_km"] = float(np.max(np.abs(np.linalg.norm(rel, axis=1)-radius)))
        result[f"{name}_orthogonality_km"] = float(np.max(np.abs(np.sum(rel*dirs, axis=1))))
        sun_r = np.asarray([np.linalg.norm(path.source_km-bundle.sun_center_km) for path in paths])
        result[f"{name}_solar_radius_error_km"] = float(np.max(np.abs(sun_r-R_SUN_KM)))
    return result


def solar_footprint_segments(jd_utc: float, *, family: str = "umbra",
                             n_azimuth: int = 720) -> tuple[np.ndarray, ...]:
    """Return contiguous WGS-84 footprint arcs without false chord joins.

    Two kinds of discontinuity are preserved:

    * tangent azimuths whose rays miss Earth entirely;
    * grazing-intersection jumps where adjacent sampled rays hit widely
      separated parts of the limb.

    The second case matters near first/last penumbral contact.  A simple run of
    ``target_hit=True`` values can still contain a 40--60 degree jump when the
    near-surface intersection changes branch.  Plotting that run as one line
    draws a non-physical chord across the map or globe.
    """
    key = str(family).lower()
    if key not in {"umbra", "penumbra"}:
        raise ValueError("family must be 'umbra' or 'penumbra'")
    bundle = trace_reference_rays("solar", jd_utc, n_azimuth=n_azimuth)
    paths = bundle.umbra if key == "umbra" else bundle.penumbra
    hit = np.asarray([path.target_hit for path in paths], dtype=bool)
    n = len(paths)
    if n == 0 or not np.any(hit):
        return ()
    points = np.asarray([path.endpoint_km for path in paths], dtype=float)

    def split_spatial(run: np.ndarray) -> list[np.ndarray]:
        run = np.asarray(run, dtype=float)
        if len(run) <= 1:
            return [np.vstack([run[0], run[0]])] if len(run) else []

        # Preserve physical branch continuity in 3-D.
        unit = run/np.linalg.norm(run, axis=1, keepdims=True)
        dot = np.clip(np.sum(unit[:-1]*unit[1:], axis=1), -1.0, 1.0)
        step_deg = np.degrees(np.arccos(dot))
        max_step_deg = max(7.0, 720.0/max(float(n_azimuth), 1.0))

        # Also preserve continuity in the equirectangular map projection.
        # Near a pole, physically adjacent points can jump tens of longitude
        # degrees.  Joining them is a visually false chord even when the 3-D
        # angular separation is modest.
        lat_lon = [geodetic_from_itrf(point)[:2] for point in run]
        lat = np.asarray([value[0] for value in lat_lon], dtype=float)
        lon = np.asarray([value[1] for value in lat_lon], dtype=float)
        dlon = np.abs(np.diff(lon))
        dlon = np.minimum(dlon, 360.0-dlon)
        dlat = np.abs(np.diff(lat))
        breaks = np.flatnonzero(
            (step_deg > max_step_deg) | (dlon > 10.0) | (dlat > 3.5)
        )

        start_index = 0
        pieces: list[np.ndarray] = []
        for index in breaks:
            stop = int(index)+1
            piece = run[start_index:stop]
            if len(piece) >= 2:
                pieces.append(piece)
            elif len(piece) == 1:
                pieces.append(np.vstack([piece[0], piece[0]]))
            start_index = stop
        piece = run[start_index:]
        if len(piece) >= 2:
            pieces.append(piece)
        elif len(piece) == 1:
            pieces.append(np.vstack([piece[0], piece[0]]))
        return pieces

    raw_runs: list[np.ndarray] = []
    if np.all(hit):
        raw_runs = [points]
    else:
        starts = [index for index in range(n) if hit[index] and not hit[index-1]]
        for start in starts:
            indices = []
            index = start
            while hit[index]:
                indices.append(index)
                index = (index+1) % n
                if index == start:
                    break
            if indices:
                raw_runs.append(points[np.asarray(indices, dtype=int)])

    segments: list[np.ndarray] = []
    for run in raw_runs:
        segments.extend(split_spatial(run))

    # A fully intersecting family is a closed boundary only when no spatial
    # branch split was required.
    if np.all(hit) and len(segments) == 1 and len(segments[0]):
        segments[0] = np.vstack([segments[0], segments[0][0]])
    return tuple(segments)


def solar_footprint_points(jd_utc: float, *, family: str = "umbra",
                           n_azimuth: int = 720) -> np.ndarray:
    """Physical WGS-84 boundary points reached by solar tangent rays.

    This compatibility wrapper concatenates the contiguous arcs returned by
    :func:`solar_footprint_segments`.  New plotting code should use the
    segmented form so gaps remain explicit.
    """
    segments = solar_footprint_segments(jd_utc, family=family,
                                        n_azimuth=n_azimuth)
    if not segments:
        return np.empty((0, 3), dtype=float)
    return np.vstack(segments)


def solar_cross_track_width_km(jd_utc: float, *, step_seconds: float = 1.0,
                               n_azimuth: int = 1440) -> float:
    """Width of the umbral footprint perpendicular to central-line velocity."""
    from ssatk.compute.eclipse_reference_events import solar_central_line_wgs84
    center = solar_central_line_wgs84(jd_utc)
    if center is None:
        return float("nan")
    p0 = center[2]
    before = solar_central_line_wgs84(jd_utc-step_seconds/86400.0)
    after = solar_central_line_wgs84(jd_utc+step_seconds/86400.0)
    if before is None or after is None:
        return float("nan")
    normal = _unit(p0)
    east = _unit(np.array([-p0[1], p0[0], 0.0]))
    north = _unit(np.cross(normal, east))
    dv = after[2]-before[2]
    velocity = _unit(np.array([np.dot(dv, east), np.dot(dv, north)]))
    cross = np.array([-velocity[1], velocity[0]])
    boundary = solar_footprint_points(jd_utc, family="umbra", n_azimuth=n_azimuth)
    if len(boundary) < 2:
        return float("nan")
    local = np.column_stack([
        (boundary-p0)@east,
        (boundary-p0)@north,
    ])
    projection = local@cross
    return float(np.max(projection)-np.min(projection))
