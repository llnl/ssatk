"""Shared environment providers for SSATK spacecraft dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ._callables import call_with_variants
from .constants import (
    EARTH_DIPOLE_EQUATOR_FIELD,
    EARTH_GEOMAGNETIC_REFERENCE_RADIUS,
    EARTH_RADIUS,
    MOON_RADIUS,
    SOLAR_FLUX_1_AU,
    SUN_RADIUS,
    WGS84_EARTH_OMEGA,
)

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
PLANET_THIRD_BODY_NAMES = (
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
)
DEFAULT_THIRD_BODY_NAMES = ("moon", "sun")
ALL_THIRD_BODY_NAMES = DEFAULT_THIRD_BODY_NAMES + PLANET_THIRD_BODY_NAMES
FORCE_MODEL_PRESETS = {
    "none": {},
    "earth_orbit": {
        "solar_radiation": True,
        "third_bodies": True,
        "gravity_gradient": True,
    },
    "leo": {
        "drag": True,
        "solar_radiation": True,
        "magnetic": True,
        "third_bodies": True,
        "gravity_gradient": True,
    },
    "cislunar": {
        "solar_radiation": True,
        "third_bodies": "all",
        "gravity_gradient": "all",
    },
    "all": {
        "drag": True,
        "solar_radiation": True,
        "magnetic": True,
        "third_bodies": "all",
        "gravity_gradient": "all",
    },
}

__all__ = [
    "ALL_THIRD_BODY_NAMES",
    "DEFAULT_THIRD_BODY_NAMES",
    "FORCE_MODEL_PRESETS",
    "PLANET_THIRD_BODY_NAMES",
    "SpaceEnvironment",
    "body_mu",
    "body_radius",
    "cylindrical_eclipse_fraction",
    "earth_dipole_magnetic_field",
    "exponential_atmosphere",
    "igrf_magnetic_field",
    "make_space_environment",
    "solar_disk_visible_fraction",
    "solar_occultation_fraction",
]


@dataclass(frozen=True)
class SpaceEnvironment:
    """Environment functions shared by 6-DoF force and torque models.

    Positions are Earth-centered GCRF meters. Density is kg/m³. Magnetic field
    is tesla in GCRF. The default Sun and Moon providers reuse SSAPy's
    ephemeris approximations.
    """

    epoch: object | None = None
    earth_orientation_model: object | None = None
    earth_orientation_allow_predicted: bool = False
    space_weather_model: object | None = None
    space_weather_allow_predicted: bool = False
    nrlmsise_version: float | str = 2.1
    sun_position_model: Callable | ArrayLike | None = None
    moon_position_model: Callable | ArrayLike | None = None
    atmosphere_density_model: Callable | float | str | None = 0.0
    atmosphere_velocity_model: Callable | ArrayLike | None = None
    magnetic_field_model: Callable | ArrayLike | str | None = "earth_dipole"
    eclipse_model: Callable | str | None = "conical"
    solar_occulting_bodies: bool | str | tuple[str, ...] | list[str] | None = ("earth", "moon")
    earth_radius: float = EARTH_RADIUS
    moon_radius: float = MOON_RADIUS
    sun_radius: float = SUN_RADIUS
    earth_rotation_rate: float = WGS84_EARTH_OMEGA
    solar_flux_1au: float = SOLAR_FLUX_1_AU
    earth_magnetic_reference_radius: float = EARTH_GEOMAGNETIC_REFERENCE_RADIUS
    earth_dipole_equatorial_field: float = EARTH_DIPOLE_EQUATOR_FIELD
    earth_dipole_axis: ArrayLike = (0.0, 0.0, 1.0)

    def absolute_time(self, time):
        """Return GPS seconds for ``time``, applying ``epoch`` as an offset when set."""

        return _absolute_time(time, self.epoch)

    def earth_orientation(self, time, *, allow_predicted: bool = False):
        """Return reproducible EOP for ``time`` from the configured data table."""

        model = self.earth_orientation_model
        if model is None:
            from .environment_eop import load_packaged_eop

            model = load_packaged_eop()
        model_time = self.absolute_time(time)
        if hasattr(model, "at"):
            return model.at(model_time, allow_predicted=allow_predicted)
        return model(model_time)

    def space_weather(self, time, *, allow_predicted: bool | None = None):
        """Return the configured daily space-weather record for ``time``."""

        model = self.space_weather_model
        if model is None:
            from .environment_space_weather import load_packaged_space_weather

            model = load_packaged_space_weather()
        model_time = self.absolute_time(time)
        allow = self.space_weather_allow_predicted if allow_predicted is None else allow_predicted
        if hasattr(model, "at"):
            return model.at(model_time, allow_predicted=allow)
        return model(model_time)

    def sun_position(self, time, r_inertial=None, v_inertial=None, q_body_to_inertial=None, omega_body=None, spacecraft=None):
        """Return Earth-to-Sun GCRF position in meters."""

        model_time = self.absolute_time(time)
        if self.sun_position_model is None:
            from ssapy.utils import sunPos

            return _vector3(sunPos(model_time), "sun_position")
        return _vector_or_model(
            self.sun_position_model,
            model_time,
            r_inertial,
            v_inertial,
            q_body_to_inertial,
            omega_body,
            spacecraft,
            "sun_position",
        )

    def moon_position(self, time, r_inertial=None, v_inertial=None, q_body_to_inertial=None, omega_body=None, spacecraft=None):
        """Return Earth-to-Moon GCRF position in meters."""

        model_time = self.absolute_time(time)
        if self.moon_position_model is None:
            from ssapy.utils import moonPos

            return _vector3(moonPos(model_time), "moon_position")
        return _vector_or_model(
            self.moon_position_model,
            model_time,
            r_inertial,
            v_inertial,
            q_body_to_inertial,
            omega_body,
            spacecraft,
            "moon_position",
        )

    def density(self, altitude_m, *args):
        """Return atmospheric density at altitude in kg/m³."""

        model = self.atmosphere_density_model
        if model is None:
            return 0.0
        if isinstance(model, str):
            if _body_key(model) not in {"nrlmsise", "nrlmsise00", "msis"}:
                raise ValueError("atmosphere_density_model string must be 'nrlmsise00'.")
            if len(args) < 2:
                raise ValueError("NRLMSISE-00 density requires time and inertial position inputs.")
            return self.nrlmsise00_density(args[0], args[1])
        if callable(model):
            return float(call_with_variants(model, (((altitude_m, *args), {}), ((altitude_m,), {}))))
        return float(model)

    def nrlmsise00_density(
        self,
        time,
        r_inertial,
        *,
        allow_predicted: bool | None = None,
        version: float | str | None = None,
    ) -> float:
        """Return NRLMSISE-00 total mass density at a GCRF position.

        This method requires the optional ``pymsis`` dependency. Geographic
        coordinates use the packaged EOP table and WGS84 geodetic conversion;
        solar and geomagnetic drivers come from ``SpaceWeatherTable``.
        """

        try:
            from pymsis import msis
        except ImportError as exc:
            raise ImportError(
                "NRLMSISE-00 support requires the optional 'atmosphere' dependency; "
                "install with 'pip install ssatk[atmosphere]'."
            ) from exc

        model_time = self.absolute_time(time)
        from .coordinates.earth_fixed import gcrf_to_itrf_eop

        eop = self.earth_orientation_model
        if eop is None:
            from .environment_eop import load_packaged_eop

            eop = load_packaged_eop()
        r_itrf = gcrf_to_itrf_eop(
            np.asarray(r_inertial, dtype=float).reshape(1, 3),
            [model_time],
            eop=eop,
            allow_predicted=self.earth_orientation_allow_predicted,
        )[0]
        longitude, latitude, altitude = _itrf_to_geodetic(r_itrf)

        weather = self.space_weather_model
        if weather is None:
            from .environment_space_weather import load_packaged_space_weather

            weather = load_packaged_space_weather()
        allow = self.space_weather_allow_predicted if allow_predicted is None else allow_predicted
        if hasattr(weather, "msis_inputs"):
            f107, f107a, aps = weather.msis_inputs(model_time, allow_predicted=allow)
        else:
            values = weather(model_time)
            if len(values) != 3:
                raise TypeError("space_weather_model must provide msis_inputs(time) or return (f107, f107a, ap).")
            f107, f107a, aps = values

        from astropy.time import Time

        date = Time(model_time, format="gps", scale="utc").to_datetime()
        output = msis.run(
            [date],
            [longitude],
            [latitude],
            [altitude / 1000.0],
            [f107],
            [f107a],
            np.asarray([aps], dtype=float),
            version=self.nrlmsise_version if version is None else version,
        )
        density = float(np.asarray(output).reshape(-1, 11)[0, 0])
        if not np.isfinite(density) or density < 0.0:
            raise RuntimeError("NRLMSISE-00 returned a non-finite or negative density.")
        return density

    def atmosphere_velocity(
        self,
        time,
        r_inertial,
        v_inertial=None,
        q_body_to_inertial=None,
        omega_body=None,
        spacecraft=None,
    ):
        """Return atmosphere velocity in GCRF m/s at the spacecraft position."""

        model_time = self.absolute_time(time)
        if self.atmosphere_velocity_model is None:
            from .accelerations_6dof import co_rotating_atmosphere_velocity

            return co_rotating_atmosphere_velocity(
                r_inertial,
                earth_rotation_rate=self.earth_rotation_rate,
            )
        return _vector_or_model(
            self.atmosphere_velocity_model,
            model_time,
            r_inertial,
            v_inertial,
            q_body_to_inertial,
            omega_body,
            spacecraft,
            "atmosphere_velocity",
        )

    def magnetic_field(self, time, r_inertial=None, v_inertial=None, q_body_to_inertial=None, omega_body=None, spacecraft=None):
        """Return GCRF magnetic field in tesla."""

        model = self.magnetic_field_model
        if model is None:
            return np.zeros(3)
        model_time = self.absolute_time(time)
        if isinstance(model, str):
            key = _body_key(model)
            if key in {"zero", "none", "off"}:
                return np.zeros(3)
            if key in {"dipole", "earth_dipole", "geomagnetic_dipole"}:
                if r_inertial is None:
                    raise ValueError("r_inertial is required for earth_dipole magnetic_field_model.")
                return earth_dipole_magnetic_field(
                    r_inertial,
                    reference_radius=self.earth_magnetic_reference_radius,
                    equatorial_field=self.earth_dipole_equatorial_field,
                    dipole_axis=self.earth_dipole_axis,
                )
            if key in {"igrf", "ppigrf"}:
                if r_inertial is None:
                    raise ValueError("r_inertial is required for IGRF magnetic_field_model.")
                return igrf_magnetic_field(model_time, r_inertial)
            raise ValueError(
                "magnetic_field_model string must be 'earth_dipole', 'igrf', "
                "'zero', or 'none'."
            )
        return _vector_or_model(
            model,
            model_time,
            r_inertial,
            v_inertial,
            q_body_to_inertial,
            omega_body,
            spacecraft,
            "magnetic_field",
        )

    def eclipse_fraction(self, time, r_inertial, v_inertial=None, q_body_to_inertial=None, omega_body=None, spacecraft=None):
        """Return visible solar-disk fraction at the spacecraft."""

        if self.eclipse_model is None:
            return 1.0
        model_time = self.absolute_time(time)
        if callable(self.eclipse_model):
            return float(
                self.eclipse_model(
                    model_time,
                    r_inertial,
                    v_inertial,
                    q_body_to_inertial,
                    omega_body,
                    spacecraft,
                )
            )
        sun_position = self.sun_position(
            time,
            r_inertial,
            v_inertial,
            q_body_to_inertial,
            omega_body,
            spacecraft,
        )
        mode = str(self.eclipse_model).strip().lower()
        if mode in {"cylindrical", "cylinder", "umbra"}:
            return cylindrical_eclipse_fraction(r_inertial, sun_position, earth_radius=self.earth_radius)
        if mode in {"conical", "disk", "penumbra", "partial"}:
            visible_fraction = 1.0
            for body_name in _solar_occulting_body_names(self.solar_occulting_bodies):
                body_position, body_radius_value = _occulting_body_position_radius(
                    self,
                    body_name,
                    time,
                    r_inertial,
                    v_inertial,
                    q_body_to_inertial,
                    omega_body,
                    spacecraft,
                )
                visible_fraction = min(
                    visible_fraction,
                    solar_occultation_fraction(
                        r_inertial,
                        sun_position,
                        body_position,
                        body_radius_value,
                        sun_radius=self.sun_radius,
                    ),
                )
            return float(visible_fraction)
        raise ValueError("eclipse_model must be None, callable, 'cylindrical', or 'conical'.")

    def force_models(
        self,
        *,
        preset: str | None = None,
        body=None,
        drag: bool | None = None,
        solar_radiation: bool | None = None,
        magnetic: bool | None = None,
        third_bodies=None,
        gravity_gradient=None,
        facet: bool = True,
    ) -> list:
        """Build SSATK force/torque models backed by this environment."""

        from .accelerations_6dof import (
            SpacecraftAccelDrag,
            SpacecraftAccelSolRad,
            SpacecraftAccelThirdBody,
            SpacecraftFacetDrag,
            SpacecraftFacetSolRad,
            SpacecraftGravityGradientTorque,
            SpacecraftMagneticTorque,
        )

        options = _force_model_preset_options(preset)
        drag = options.get("drag", False) if drag is None else drag
        solar_radiation = options.get("solar_radiation", False) if solar_radiation is None else solar_radiation
        magnetic = options.get("magnetic", False) if magnetic is None else magnetic
        third_bodies = options.get("third_bodies", False) if third_bodies is None else third_bodies
        gravity_gradient = options.get("gravity_gradient", False) if gravity_gradient is None else gravity_gradient

        models = []
        density = lambda t, r, v, q, omega, spacecraft: self.density(
            np.linalg.norm(r) - self.earth_radius,
            t,
            r,
            v,
            q,
            omega,
            spacecraft,
        )
        atmosphere_velocity = (
            lambda t, r, v, q, omega, spacecraft: self.atmosphere_velocity(
                t,
                r,
                v,
                q,
                omega,
                spacecraft,
            )
        )
        if drag:
            drag_class = SpacecraftFacetDrag if facet else SpacecraftAccelDrag
            models.append(
                drag_class(
                    density=density,
                    body=body,
                    atmosphere_velocity=atmosphere_velocity,
                    earth_radius=self.earth_radius,
                    earth_rotation_rate=self.earth_rotation_rate,
                )
                if facet
                else drag_class(
                    density=density,
                    atmosphere_velocity=atmosphere_velocity,
                    earth_radius=self.earth_radius,
                    earth_rotation_rate=self.earth_rotation_rate,
                )
            )
        if solar_radiation:
            srp_class = SpacecraftFacetSolRad if facet else SpacecraftAccelSolRad
            models.append(
                srp_class(
                    self.sun_position,
                    body=body,
                    solar_flux_1au=self.solar_flux_1au,
                    eclipse=self.eclipse_fraction,
                )
                if facet
                else srp_class(
                    self.sun_position,
                    solar_flux_1au=self.solar_flux_1au,
                    eclipse=self.eclipse_fraction,
                )
            )
        for name in _third_body_names(third_bodies):
            models.append(SpacecraftAccelThirdBody(self.body_position_model(name), body_mu(name)))
        for name in _gravity_gradient_body_names(gravity_gradient):
            source_position = (0.0, 0.0, 0.0) if _body_key(name) == "earth" else self.body_position_model(name)
            models.append(SpacecraftGravityGradientTorque(source_position, mu=body_mu(name)))
        if magnetic:
            models.append(SpacecraftMagneticTorque(self.magnetic_field, body=body))
        return models

    def body_position_model(self, name: str):
        """Return a GCRF position provider for a named Solar-System body."""

        key = _body_key(name)
        if key == "sun":
            return self.sun_position
        if key == "moon":
            return self.moon_position

        from ssapy.body import get_body

        # One Body, held by the closure: SSAPy closes a Body's ephemeris when the
        # Body is collected, so get_body(name).position(t) on a temporary raises
        # TypeError for every planet.
        body = get_body(name)

        def position(time, *_args):
            return _vector3(body.position(self.absolute_time(time)), f"{name}_position")

        return position


def make_space_environment(**kwargs) -> SpaceEnvironment:
    """Return a :class:`SpaceEnvironment` with explicit override keywords."""

    return SpaceEnvironment(**kwargs)


def body_mu(name: str) -> float:
    """Return a named Solar-System body's gravitational parameter in m³/s²."""

    from . import constants

    key = _body_key(name)
    constant_name = f"{key.upper()}_MU"
    if hasattr(constants, constant_name):
        return float(getattr(constants, constant_name))

    from ssapy.body import get_body

    return float(get_body(name).mu)


def body_radius(name: str) -> float:
    """Return a named Solar-System body's reference radius in meters.

    This is the ``<NAME>_RADIUS`` constant (equatorial for Earth, 6378.137 km;
    SSAPy's value otherwise), not a mean radius.
    """

    from . import constants

    key = _body_key(name)
    constant_name = f"{key.upper()}_RADIUS"
    if hasattr(constants, constant_name):
        return float(getattr(constants, constant_name))

    from ssapy.body import get_body

    body = get_body(name)
    if hasattr(body, "radius"):
        return float(body.radius)
    raise ValueError(f"No radius is known for body {name!r}.")


def earth_dipole_magnetic_field(
    r_inertial: ArrayLike,
    *,
    reference_radius: float = EARTH_GEOMAGNETIC_REFERENCE_RADIUS,
    equatorial_field: float = EARTH_DIPOLE_EQUATOR_FIELD,
    dipole_axis: ArrayLike = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return an aligned centered-dipole Earth magnetic field in tesla."""

    r = _vector3(r_inertial, "r_inertial")
    radius = np.linalg.norm(r)
    if radius == 0.0:
        return np.zeros(3)
    axis = _unit_vector(dipole_axis, "dipole_axis")
    radius_hat = r / radius
    scale = float(equatorial_field) * (float(reference_radius) / radius) ** 3
    return scale * (3.0 * radius_hat * np.dot(axis, radius_hat) - axis)


def igrf_magnetic_field(time, r_inertial: ArrayLike) -> np.ndarray:
    """Return IGRF magnetic field in GCRF tesla at ``time`` and ``r_inertial``.

    ``time`` may be GPS seconds, an ``astropy.time.Time``, a ``datetime``, or a
    parseable time string. ``r_inertial`` is an Earth-centered GCRF position in
    meters. The underlying IGRF synthesis comes from optional ``ppigrf`` through
    :mod:`ssatk.geomagnetics`.
    """

    from .geomagnetics import _HAS_PPIGRF, _bfield_batch

    if not _HAS_PPIGRF:
        raise ImportError(
            "IGRF magnetic-field support requires the optional 'ppigrf' dependency."
        )

    time_astropy = _astropy_time(time)
    from .coordinates.earth_fixed import gcrf_to_itrf_astropy

    r_itrf_m = gcrf_to_itrf_astropy(
        _vector3(r_inertial, "r_inertial").reshape(1, 3),
        time_astropy,
    )[0]
    # Internal field only: the module-level external model (e.g. T89, set by
    # the magnetosphere plots) would otherwise leak into an "IGRF" result.
    b_itrf_nt = _bfield_batch(
        (r_itrf_m / 1000.0).reshape(1, 3),
        _datetime_from_astropy(time_astropy),
        include_external=False,
    )[0]
    return _itrf_vector_to_gcrf(b_itrf_nt, time_astropy) * 1e-9


def exponential_atmosphere(
    *,
    reference_density: float,
    reference_altitude: float,
    scale_height: float,
) -> Callable[[float], float]:
    """Return a simple exponential atmosphere density provider."""

    reference_density = _positive(reference_density, "reference_density")
    scale_height = _positive(scale_height, "scale_height")
    reference_altitude = float(reference_altitude)

    def density(altitude_m: float) -> float:
        return float(reference_density * np.exp(-(float(altitude_m) - reference_altitude) / scale_height))

    return density


def cylindrical_eclipse_fraction(
    r_inertial: ArrayLike,
    sun_position: ArrayLike,
    *,
    earth_radius: float = EARTH_RADIUS,
) -> float:
    """Return 0 inside a cylindrical Earth shadow and 1 otherwise."""

    spacecraft_position = _vector3(r_inertial, "r_inertial")
    sun_vector = _vector3(sun_position, "sun_position")
    sun_distance = np.linalg.norm(sun_vector)
    if sun_distance == 0.0:
        return 1.0
    sun_hat = sun_vector / sun_distance
    if np.dot(spacecraft_position, sun_hat) >= 0.0:
        return 1.0
    perpendicular = spacecraft_position - np.dot(spacecraft_position, sun_hat) * sun_hat
    return 0.0 if np.linalg.norm(perpendicular) < float(earth_radius) else 1.0


def solar_disk_visible_fraction(
    r_inertial: ArrayLike,
    sun_position: ArrayLike,
    *,
    earth_radius: float = EARTH_RADIUS,
    sun_radius: float = SUN_RADIUS,
) -> float:
    """Return Sun disk fraction visible after Earth occultation."""

    return solar_occultation_fraction(
        r_inertial,
        sun_position,
        (0.0, 0.0, 0.0),
        earth_radius,
        sun_radius=sun_radius,
    )


def solar_occultation_fraction(
    r_inertial: ArrayLike,
    sun_position: ArrayLike,
    occulting_position: ArrayLike,
    occulting_radius: float,
    *,
    sun_radius: float = SUN_RADIUS,
) -> float:
    """Return Sun disk fraction visible after occultation by a spherical body."""

    spacecraft_position = _vector3(r_inertial, "r_inertial")
    sun_vector = _vector3(sun_position, "sun_position")
    occulting_vector = _vector3(occulting_position, "occulting_position")
    occulting_from_spacecraft = occulting_vector - spacecraft_position
    sun_from_spacecraft = sun_vector - spacecraft_position
    occulting_distance = np.linalg.norm(occulting_from_spacecraft)
    sun_distance = np.linalg.norm(sun_from_spacecraft)
    occulting_radius = float(occulting_radius)
    if occulting_distance <= 0.0 or sun_distance <= 0.0 or occulting_radius <= 0.0:
        return 1.0

    occulting_angular_radius = np.arcsin(np.clip(occulting_radius / occulting_distance, 0.0, 1.0))
    sun_angular_radius = np.arcsin(np.clip(float(sun_radius) / sun_distance, 0.0, 1.0))
    if occulting_angular_radius == 0.0 or sun_angular_radius == 0.0:
        return 1.0

    separation = _angle_between(occulting_from_spacecraft, sun_from_spacecraft)
    occulted = _circle_overlap_area(sun_angular_radius, occulting_angular_radius, separation)
    return float(np.clip(1.0 - occulted / (np.pi * sun_angular_radius**2), 0.0, 1.0))


def _solar_occulting_body_names(bodies) -> tuple[str, ...]:
    if bodies is None or bodies is False:
        return ()
    if bodies is True:
        return ("earth", "moon")
    if isinstance(bodies, str):
        return (_body_key(bodies),)
    return tuple(_body_key(body) for body in bodies)


def _occulting_body_position_radius(environment, name, time, r, v, q, omega, spacecraft):
    if name == "earth":
        return np.zeros(3), environment.earth_radius
    if name == "moon":
        return environment.moon_position(time, r, v, q, omega, spacecraft), environment.moon_radius
    if name == "sun":
        raise ValueError("The Sun cannot occult itself.")
    return environment.body_position_model(name)(time, r, v, q, omega, spacecraft), body_radius(name)


def _vector_or_model(model, time, r_inertial, v_inertial, q_body_to_inertial, omega_body, spacecraft, name: str) -> np.ndarray:
    if callable(model):
        args = (time, r_inertial, v_inertial, q_body_to_inertial, omega_body, spacecraft)
        value = call_with_variants(model, ((args, {}), ((time,), {})))
    else:
        value = model
    return _vector3(value, name)


def _absolute_time(time, epoch):
    if _looks_absolute_time(time):
        return float(_astropy_time(time).gps)
    if epoch is None:
        return time
    return float(_astropy_time(epoch).gps) + float(time)


def _looks_absolute_time(value) -> bool:
    return isinstance(value, (datetime, str)) or hasattr(value, "gps")


def _astropy_time(value):
    from astropy.time import Time

    if isinstance(value, Time):
        return value
    if isinstance(value, datetime):
        return Time(value, scale="utc")
    if isinstance(value, str):
        return Time(value, scale="utc")
    return Time(float(value), format="gps", scale="utc")


def _datetime_from_astropy(value):
    try:
        return value.to_datetime(timezone=None)
    except TypeError:
        return value.to_datetime()


def _itrf_vector_to_gcrf(vector_itrf, time):
    import astropy.units as unit
    from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord

    vector = _vector3(vector_itrf, "vector_itrf")
    coord = SkyCoord(
        CartesianRepresentation(*(vector * unit.m)),
        frame=ITRS(obstime=time),
    )
    gcrs = coord.transform_to(GCRS(obstime=time))
    return np.array(
        [
            gcrs.cartesian.x.to_value(unit.m),
            gcrs.cartesian.y.to_value(unit.m),
            gcrs.cartesian.z.to_value(unit.m),
        ]
    )


def _third_body_names(third_bodies) -> tuple[str, ...]:
    if not third_bodies:
        return ()
    if third_bodies is True:
        return DEFAULT_THIRD_BODY_NAMES
    if isinstance(third_bodies, str):
        key = _body_key(third_bodies)
        if key in {"default", "nearby"}:
            return DEFAULT_THIRD_BODY_NAMES
        if key in {"planet", "planets"}:
            return PLANET_THIRD_BODY_NAMES
        if key in {"all", "solar_system", "solarsystem"}:
            return ALL_THIRD_BODY_NAMES
        return (key,)
    return tuple(_body_key(body) for body in third_bodies)


def _force_model_preset_options(preset: str | None) -> dict:
    if preset is None:
        return {}
    key = _body_key(preset)
    aliases = {
        "near_earth": "leo",
        "low_earth_orbit": "leo",
        "full": "all",
        "high_fidelity": "all",
        "solar_system": "cislunar",
    }
    key = aliases.get(key, key)
    if key not in FORCE_MODEL_PRESETS:
        choices = ", ".join(sorted(FORCE_MODEL_PRESETS))
        raise ValueError(f"Unknown force model preset {preset!r}. Available: {choices}.")
    return dict(FORCE_MODEL_PRESETS[key])


def _gravity_gradient_body_names(gravity_gradient) -> tuple[str, ...]:
    if not gravity_gradient:
        return ()
    if gravity_gradient is True:
        return ("earth",)
    if isinstance(gravity_gradient, str):
        key = _body_key(gravity_gradient)
        return ("earth", "moon", "sun") if key == "all" else (key,)
    return tuple(_body_key(body) for body in gravity_gradient)


def _body_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _circle_overlap_area(radius_a: float, radius_b: float, separation: float) -> float:
    if separation >= radius_a + radius_b:
        return 0.0
    if separation <= abs(radius_b - radius_a):
        return np.pi * min(radius_a, radius_b) ** 2
    part_a = radius_a**2 * np.arccos((separation**2 + radius_a**2 - radius_b**2) / (2.0 * separation * radius_a))
    part_b = radius_b**2 * np.arccos((separation**2 + radius_b**2 - radius_a**2) / (2.0 * separation * radius_b))
    radicand = (-separation + radius_a + radius_b) * (separation + radius_a - radius_b) * (separation - radius_a + radius_b) * (separation + radius_a + radius_b)
    return float(part_a + part_b - 0.5 * np.sqrt(max(0.0, radicand)))


def _angle_between(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if norm_product == 0.0:
        return 0.0
    return float(np.arccos(np.clip(np.dot(vector_a, vector_b) / norm_product, -1.0, 1.0)))


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _itrf_to_geodetic(r_itrf: ArrayLike) -> tuple[float, float, float]:
    """Convert one ITRF Cartesian position to WGS84 longitude/latitude/height."""

    x, y, z = _vector3(r_itrf, "r_itrf")
    semi_major = 6_378_137.0
    flattening = 1.0 / 298.257223563
    semi_minor = semi_major * (1.0 - flattening)
    first_eccentricity_squared = flattening * (2.0 - flattening)
    second_eccentricity_squared = (semi_major**2 - semi_minor**2) / semi_minor**2
    p = float(np.hypot(x, y))
    if p == 0.0:
        latitude = np.copysign(np.pi / 2.0, z)
        return 0.0, float(np.degrees(latitude)), abs(float(z)) - semi_minor

    theta = np.arctan2(semi_major * z, semi_minor * p)
    latitude = np.arctan2(
        z + second_eccentricity_squared * semi_minor * np.sin(theta) ** 3,
        p - first_eccentricity_squared * semi_major * np.cos(theta) ** 3,
    )
    longitude = np.arctan2(y, x)
    radius_of_curvature = semi_major / np.sqrt(
        1.0 - first_eccentricity_squared * np.sin(latitude) ** 2
    )
    height = p / np.cos(latitude) - radius_of_curvature
    return float(np.degrees(longitude)), float(np.degrees(latitude)), float(height)


def _unit_vector(value: ArrayLike, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _vector3(value: ArrayLike, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) and value.size == 3:
        value = value.reshape(3)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return value
