"""Adapters for using SSAPy translational accelerations in 6-DoF models."""

from __future__ import annotations

import numpy as np

from .spacecraft import SpacecraftAccel, SpacecraftAccelSum

ArrayLike = np.ndarray | list[float] | tuple[float, ...]

__all__ = [
    "SpacecraftAccelSSAPy",
    "make_ssapy_drag",
    "make_ssapy_earth_harmonics",
    "make_ssapy_earth_radiation",
    "make_ssapy_perturbation_acceleration",
    "make_ssapy_solar_radiation",
    "make_ssapy_third_body",
    "wrap_ssapy_acceleration",
]


_DEFAULT_SPACECRAFT_KWARG_MAP = {
    "mass": "mass",
    "area": "area",
    "cd": "CD",
    "cr": "CR",
}


class SpacecraftAccelSSAPy(SpacecraftAccel):
    """Wrap an SSAPy acceleration object for SSATK 6-DoF propagation.

    SSAPy accelerations use ``accel(r, v, t, **kwargs)``. SSATK 6-DoF
    propagation uses ``accel(t, r, v, q, omega)`` and may carry physical
    properties on a ``Spacecraft`` object. This adapter handles that translation
    without changing the underlying SSAPy force model.

    The wrapped model is an additional acceleration term. If it already includes
    central gravity, call ``propagate_6dof(..., mu=0.0, acceleration=model)`` to
    avoid double-counting the built-in point-mass term.
    """

    def __init__(
        self,
        accel,
        *,
        kwargs: dict | None = None,
        spacecraft_kwargs: bool = True,
        kwarg_map: dict[str, str] | None = None,
    ):
        self.accel = accel
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self.spacecraft_kwargs = bool(spacecraft_kwargs)
        self.kwarg_map = dict(_DEFAULT_SPACECRAFT_KWARG_MAP)
        if kwarg_map is not None:
            self.kwarg_map.update(kwarg_map)
        if hasattr(accel, "time_breakpoints"):
            self.time_breakpoints = np.asarray(accel.time_breakpoints, dtype=float).copy()

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> np.ndarray:
        kwargs = dict(self.kwargs)
        if self.spacecraft_kwargs:
            kwargs.update(_spacecraft_kwargs(spacecraft, self.kwarg_map))
        return _vector3(self.accel(r, v, t, **kwargs), "ssapy acceleration")


def wrap_ssapy_acceleration(accel, **kwargs) -> SpacecraftAccelSSAPy:
    """Return a 6-DoF-compatible wrapper around an SSAPy acceleration."""

    return SpacecraftAccelSSAPy(accel, **kwargs)


def make_ssapy_earth_harmonics(
    *,
    model: str = "EGM2008",
    degree: int = 70,
    order: int | None = None,
) -> SpacecraftAccelSSAPy:
    """Return SSAPy Earth harmonic gravity as a 6-DoF perturbing acceleration."""

    from ssapy.body import get_body
    from ssapy.gravity import AccelHarmonic

    order = degree if order is None else order
    return wrap_ssapy_acceleration(AccelHarmonic(get_body("Earth", model=model), degree, order))


def make_ssapy_third_body(body: str = "moon") -> SpacecraftAccelSSAPy:
    """Return an SSAPy point-mass third-body perturbation."""

    from ssapy.body import get_body
    from ssapy.gravity import AccelThirdBody

    return wrap_ssapy_acceleration(AccelThirdBody(get_body(body)))


def make_ssapy_drag(**kwargs) -> SpacecraftAccelSSAPy:
    """Return SSAPy Harris-Priester drag as a 6-DoF acceleration."""

    from ssapy.accel import AccelDrag

    return wrap_ssapy_acceleration(AccelDrag(**kwargs))


def make_ssapy_solar_radiation(**kwargs) -> SpacecraftAccelSSAPy:
    """Return SSAPy cannonball solar-radiation pressure."""

    from ssapy.accel import AccelSolRad

    return wrap_ssapy_acceleration(AccelSolRad(**kwargs))


def make_ssapy_earth_radiation(**kwargs) -> SpacecraftAccelSSAPy:
    """Return SSAPy cannonball Earth-radiation pressure."""

    from ssapy.accel import AccelEarthRad

    return wrap_ssapy_acceleration(AccelEarthRad(**kwargs))


def make_ssapy_perturbation_acceleration(
    *,
    earth_gravity_model: str = "EGM2008",
    earth_degree: int = 70,
    earth_order: int | None = None,
    include_moon: bool = True,
    include_sun: bool = True,
    include_planets: bool = False,
    include_solar_radiation: bool = True,
    include_earth_radiation: bool = True,
    include_drag: bool = False,
    spacecraft_kwargs: dict | None = None,
) -> SpacecraftAccelSum:
    """Build a practical SSAPy perturbation stack for 6-DoF propagation.

    This stack intentionally omits central ``AccelKepler`` because
    ``propagate_6dof`` already includes point-mass gravity through ``mu``.
    """

    from ssapy.accel import AccelDrag, AccelEarthRad, AccelSolRad
    from ssapy.body import get_body
    from ssapy.gravity import AccelHarmonic, AccelThirdBody

    kwargs = {} if spacecraft_kwargs is None else dict(spacecraft_kwargs)
    order = earth_degree if earth_order is None else earth_order
    earth = get_body("Earth", model=earth_gravity_model)
    models = [wrap_ssapy_acceleration(AccelHarmonic(earth, earth_degree, order))]

    if include_moon:
        moon = get_body("moon")
        models.append(wrap_ssapy_acceleration(AccelThirdBody(moon)))
    if include_sun:
        models.append(wrap_ssapy_acceleration(AccelThirdBody(get_body("Sun"))))
    if include_planets:
        for name in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"):
            models.append(wrap_ssapy_acceleration(AccelThirdBody(get_body(name))))
    if include_solar_radiation:
        models.append(wrap_ssapy_acceleration(AccelSolRad(**kwargs)))
    if include_earth_radiation:
        models.append(wrap_ssapy_acceleration(AccelEarthRad(**kwargs)))
    if include_drag:
        models.append(wrap_ssapy_acceleration(AccelDrag(**kwargs)))

    return SpacecraftAccelSum(models)


def _spacecraft_kwargs(spacecraft, kwarg_map: dict[str, str]) -> dict:
    if spacecraft is None:
        return {}
    kwargs = {}
    for attr, key in kwarg_map.items():
        value = getattr(spacecraft, attr, None)
        if value is not None:
            kwargs[key] = float(value)
    return kwargs


def _vector3(value: ArrayLike, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return value
