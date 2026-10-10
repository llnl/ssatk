"""Multi-band light curves of faceted bodies with known attitude.

The sphere model in :mod:`ssatk.compute.lambertian_magnitude` carries
all of its geometry in one term, ``albedo * (R/d)**2 * p(alpha)``. This module
replaces that term with a sum over body-fixed facets,

.. math::

    S(\\hat{s}, \\hat{o}) = \\sum_i \\frac{\\rho_i A_i
        \\max(0, \\hat{n}_i \\cdot \\hat{s})
        \\max(0, \\hat{n}_i \\cdot \\hat{o})}{\\pi},

an effective scattering area in square metres that depends on attitude, and
reuses that module's band table, Planck band fractions, umbra/penumbra
visibility, airmass extinction, and AB magnitude zero point unchanged. Direct
sunshine, earthshine, and moonshine are each evaluated with their own source
direction, so a nadir-facing radiator picks up earthshine that a zenith-facing
one does not.

Facet reflectivity may vary per component and per band: ``diffuse_reflectivity``
is either a scalar or a mapping keyed by band name, which is what separates a
solar array from multi-layer insulation in a colour light curve.

Reflection is Lambertian. A facet's ``specular_reflectivity`` is not used, so
specular glints are outside this model.

Observers follow the same dispatch as the sphere module: an
``ssapy.EarthObserver``, an astropy ``EarthLocation``, or a bare GCRF position
vector. The last case is what makes space-based observing work — pass
``ssapy.OrbitalObserver(orbit).getRV(time)[0]`` to observe from orbit.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..accelerations_6dof.spacecraft import _facet_is_shadowed
from ..propagators_6dof import normalize_quaternion, quaternion_conjugate, rotate_vector
from ._visibility import line_of_sight_blocked
from .lambertian_magnitude import (
    ALBEDO_EARTH,
    ALBEDO_MOON,
    ATMOSPHERE_TOP_M,
    AU_M,
    DEFAULT_TIME,
    F_NU_AB_ZERO,
    OLR_EARTH,
    R_EARTH,
    R_MOON,
    R_SUN,
    SOLAR_CONST,
    T_EARTH_LW,
    T_SUN,
    _angle_between,
    _package,
    _planck_band_fraction,
    _resolve_band,
    _setup,
    lambert_sphere_phase,
    sun_visibility_factor,
)

__all__ = [
    "faceted_light_curve",
    "faceted_reflection",
    "facet_scattering_area",
    "line_of_sight_blocked",
]


def _unit(vector):
    vector = np.asarray(vector, dtype=float).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction vectors must have nonzero length.")
    return vector / norm


def _facet_reflectivity(facet, band_name):
    """Return a facet's diffuse reflectivity, resolving band-keyed mappings."""

    value = getattr(facet, "diffuse_reflectivity", None)
    if value is None:
        raise ValueError("each facet must define diffuse_reflectivity for photometry.")
    if isinstance(value, Mapping):
        if band_name not in value:
            raise ValueError(
                f"facet.diffuse_reflectivity has no entry for band {band_name!r}; "
                f"available bands are {sorted(value)}."
            )
        value = value[band_name]
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("facet.diffuse_reflectivity must be finite and within [0, 1].")
    return value


def _facet_area(facet):
    area = float(facet.area)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("facet.area must be finite and positive.")
    return area


def facet_scattering_area(
    facets,
    quaternion,
    source_unit_inertial,
    observer_unit_inertial,
    *,
    band_name: str = "V",
    self_shadowing: bool = False,
    shadow_epsilon: float = 1.0e-9,
) -> float:
    """Return the effective Lambertian scattering area in square metres.

    ``quaternion`` is body-to-inertial ``[w, x, y, z]``. The two direction
    arguments are inertial unit vectors pointing from the object toward the
    illumination source and toward the observer. Facets that are unlit, turned
    away from the observer, or occluded along either path contribute nothing.
    """

    quaternion = normalize_quaternion(quaternion)
    conjugate = quaternion_conjugate(quaternion)
    source_body = rotate_vector(conjugate, _unit(source_unit_inertial))
    observer_body = rotate_vector(conjugate, _unit(observer_unit_inertial))

    facets = tuple(facets)
    if not facets:
        raise ValueError("facets must not be empty.")

    total = 0.0
    for index, facet in enumerate(facets):
        normal = _unit(facet.normal_body)
        cosine_source = float(np.dot(normal, source_body))
        cosine_observer = float(np.dot(normal, observer_body))
        if cosine_source <= 0.0 or cosine_observer <= 0.0:
            continue
        if self_shadowing and (
            _facet_is_shadowed(index, facet, facets, source_body, shadow_epsilon)
            or _facet_is_shadowed(index, facet, facets, observer_body, shadow_epsilon)
        ):
            continue
        total += (
            _facet_reflectivity(facet, band_name)
            * _facet_area(facet)
            * cosine_source
            * cosine_observer
            / np.pi
        )
    return total


def faceted_reflection(
    obj_pos_gcrs_m,
    quaternion,
    facets,
    observer=None,
    time=DEFAULT_TIME,
    band="V",
    k_extinction=0.16,
    include_sun=True,
    include_earthshine=True,
    include_moonshine=True,
    lon=None,
    lat=None,
    elevation=0.0,
    self_shadowing=False,
    shadow_epsilon=1.0e-9,
    check_line_of_sight=True,
    solar_const=SOLAR_CONST,
    t_sun=T_SUN,
    r_sun_radius=R_SUN,
    r_earth=R_EARTH,
    albedo_earth=ALBEDO_EARTH,
    olr_earth=OLR_EARTH,
    t_earth_lw=T_EARTH_LW,
    r_moon=R_MOON,
    albedo_moon=ALBEDO_MOON,
    atmosphere_top_m=ATMOSPHERE_TOP_M,
    f_nu_ab_zero=F_NU_AB_ZERO,
    _geo=None,
):
    """Light reflected by a faceted body: sunshine, earthshine, and moonshine.

    Returns the same dictionary shape as
    :func:`~ssatk.compute.lambertian_magnitude.lambertian_reflection`,
    with an added ``scattering_area_m2`` entry per component so a curve can be
    read as geometry rather than only as magnitude.

    With ``check_line_of_sight`` set, an object the Earth occults reports zero
    irradiance and infinite magnitude instead of the value it would have had
    with a clear sightline. This covers both a topocentric observer whose
    target has set and a space-based observer looking through the Earth.
    """

    g = _geo or _setup(
        obj_pos_gcrs_m, observer, time, band, k_extinction,
        lon, lat, elevation, r_earth, atmosphere_top_m,
        check_line_of_sight=False,
    )
    r_obj, r_obs = g["r_obj"], g["r_obs"]
    r_sun, r_moon_v = g["r_sun"], g["r_moon"]
    lam_lo, lam_hi = g["lam_lo"], g["lam_hi"]
    band_name = g["band_name"]

    observer_unit = _unit(r_obs - r_obj)
    inverse_range_squared = 1.0 / g["d_obs"] ** 2
    d_sun_obj = np.linalg.norm(r_sun - r_obj)
    F_sun_at_obj = solar_const * (AU_M / d_sun_obj) ** 2

    frac_solar = _planck_band_fraction(t_sun, lam_lo, lam_hi)
    frac_olr = _planck_band_fraction(t_earth_lw, lam_lo, lam_hi)

    def scattering(source_unit):
        return facet_scattering_area(
            facets,
            quaternion,
            source_unit,
            observer_unit,
            band_name=band_name,
            self_shadowing=self_shadowing,
            shadow_epsilon=shadow_epsilon,
        )

    comp_bolo, comp_band, angles, areas = {}, {}, {}, {}

    occulted = bool(g["below_horizon"]) or (
        check_line_of_sight and line_of_sight_blocked(r_obj, r_obs, r_earth)
    )
    if occulted:
        result = _package(g, {"sun": 0.0}, {"sun": 0.0}, {}, time, f_nu_ab_zero)
        result["scattering_area_m2"] = {}
        result["occulted"] = True
        return result

    if include_sun:
        alpha = _angle_between(r_sun - r_obj, r_obs - r_obj)
        vis = sun_visibility_factor(r_obj, r_sun, r_earth, r_sun_radius)
        area = scattering(r_sun - r_obj)
        F = vis * F_sun_at_obj * area * inverse_range_squared
        comp_bolo["sun"] = F
        comp_band["sun"] = F * frac_solar
        angles["phase_sun_obj_obs_deg"] = np.degrees(alpha)
        angles["sun_visibility"] = vis
        areas["sun"] = area

    if include_earthshine:
        d_earth_obj = np.linalg.norm(r_obj)
        beta = _angle_between(r_sun, r_obj)
        E_es_sw = (
            F_sun_at_obj * albedo_earth * (r_earth / d_earth_obj) ** 2
            * lambert_sphere_phase(beta)
        )
        E_es_lw = olr_earth * (r_earth / d_earth_obj) ** 2
        area = scattering(-r_obj)
        gamma = _angle_between(-r_obj, r_obs - r_obj)
        F_sw = E_es_sw * area * inverse_range_squared
        F_lw = E_es_lw * area * inverse_range_squared
        comp_bolo["earthshine"] = F_sw + F_lw
        comp_band["earthshine"] = F_sw * frac_solar + F_lw * frac_olr
        angles["phase_sun_earth_obj_deg"] = np.degrees(beta)
        angles["phase_earth_obj_obs_deg"] = np.degrees(gamma)
        areas["earthshine"] = area

    if include_moonshine:
        d_moon_obj = np.linalg.norm(r_moon_v - r_obj)
        F_sun_at_moon = solar_const * (AU_M / np.linalg.norm(r_sun - r_moon_v)) ** 2
        delta = _angle_between(r_sun - r_moon_v, r_obj - r_moon_v)
        E_ms = (
            F_sun_at_moon * albedo_moon * (r_moon / d_moon_obj) ** 2
            * lambert_sphere_phase(delta)
        )
        area = scattering(r_moon_v - r_obj)
        eps = _angle_between(r_moon_v - r_obj, r_obs - r_obj)
        F = E_ms * area * inverse_range_squared
        comp_bolo["moonshine"] = F
        comp_band["moonshine"] = F * frac_solar
        angles["phase_sun_moon_obj_deg"] = np.degrees(delta)
        angles["phase_moon_obj_obs_deg"] = np.degrees(eps)
        areas["moonshine"] = area

    result = _package(g, comp_bolo, comp_band, angles, time, f_nu_ab_zero)
    result["scattering_area_m2"] = areas
    result["occulted"] = False
    return result


def faceted_light_curve(
    positions_gcrs_m,
    quaternions,
    facets,
    times,
    bands=("V",),
    observer=None,
    k_extinction=0.16,
    lon=None,
    lat=None,
    elevation=0.0,
    r_earth=R_EARTH,
    atmosphere_top_m=ATMOSPHERE_TOP_M,
    **kwargs,
):
    """Return per-band magnitude arrays over a sampled trajectory and attitude.

    ``positions_gcrs_m`` is ``(N, 3)``, ``quaternions`` is ``(N, 4)`` body-to-
    inertial, and ``times`` is a length-``N`` astropy ``Time``.

    ``observer`` takes the same forms the sphere module accepts, and one more:
    an ``(N, 3)`` array of GCRF positions, one per epoch, which is how a
    space-based platform is observed from its own propagated ephemeris rather
    than from a fixed point.

    Geometry and ephemerides are resolved once per epoch and reused across
    bands, since only the wavelength limits differ between them. That matters
    for a topocentric observer, where the horizontal-coordinate transform
    dominates the cost.

    The returned mapping holds one entry per band with ``ab_mag_observed``,
    ``ab_mag_exoatmospheric``, ``range_m``, and a boolean ``occulted`` array.
    Epochs with no illuminated, visible facet yield ``inf`` rather than
    raising, so eclipse, occultation, and edge-on attitude read as gaps.
    """

    positions = np.atleast_2d(np.asarray(positions_gcrs_m, dtype=float))
    attitudes = np.atleast_2d(np.asarray(quaternions, dtype=float))
    if positions.shape[1] != 3:
        raise ValueError("positions_gcrs_m must have shape (N, 3).")
    if attitudes.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4).")
    if positions.shape[0] != attitudes.shape[0]:
        raise ValueError("positions_gcrs_m and quaternions must have equal length.")
    samples = positions.shape[0]
    if len(times) != samples:
        raise ValueError("times must have the same length as positions_gcrs_m.")

    moving_observer = (
        isinstance(observer, np.ndarray) and observer.ndim == 2 and observer.shape[0] == samples
    )
    if moving_observer and observer.shape[1] != 3:
        raise ValueError("a per-epoch observer must have shape (N, 3).")

    bands = tuple(bands)
    names = [band if isinstance(band, str) else "custom" for band in bands]
    curves = {
        name: {
            "ab_mag_observed": np.empty(samples),
            "ab_mag_exoatmospheric": np.empty(samples),
            "range_m": np.empty(samples),
            "occulted": np.zeros(samples, dtype=bool),
        }
        for name in names
    }

    for index in range(samples):
        site = observer[index] if moving_observer else observer
        geometry = _setup(
            positions[index], site, times[index], bands[0], k_extinction,
            lon, lat, elevation, r_earth, atmosphere_top_m,
        )
        for band, name in zip(bands, names):
            per_band = dict(geometry)
            per_band["lam_lo"], per_band["lam_hi"] = _resolve_band(band)
            per_band["band_name"] = name
            sample = faceted_reflection(
                positions[index],
                attitudes[index],
                facets,
                time=times[index],
                band=band,
                r_earth=r_earth,
                _geo=per_band,
                **kwargs,
            )
            curves[name]["ab_mag_observed"][index] = sample["ab_mag_observed"]
            curves[name]["ab_mag_exoatmospheric"][index] = sample["ab_mag_exoatmospheric"]
            curves[name]["range_m"][index] = sample["range_m"]
            curves[name]["occulted"][index] = sample["occulted"]
    return curves
