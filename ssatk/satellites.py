"""Simple spacecraft body and satellite design presets for 6-DoF workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import G0

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class Component:
    """Generic body-frame mass component for inertia and center-of-mass estimates."""

    mass: float
    position_body: ArrayLike = (0.0, 0.0, 0.0)
    inertia: ArrayLike | None = None
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mass", _positive(self.mass, "mass"))
        object.__setattr__(self, "position_body", _vector3(self.position_body, "position_body"))
        if self.inertia is not None:
            object.__setattr__(self, "inertia", _inertia(self.inertia))

    def inertia_about(self, origin: ArrayLike = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Return inertia about ``origin`` using the parallel-axis theorem."""

        inertia = np.zeros((3, 3)) if self.inertia is None else self.inertia
        return inertia + point_mass_inertia(self.mass, self.position_body - _vector3(origin, "origin"))

    def with_updates(self, **kwargs) -> Component:
        """Return a modified copy."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class MagneticDipole:
    """Body-frame magnetic dipole moment in A m²."""

    moment_body: ArrayLike
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "moment_body", _vector3(self.moment_body, "moment_body"))

    def torque_body(self, magnetic_field_body: ArrayLike) -> np.ndarray:
        """Return body-frame magnetic torque for a body-frame field in tesla."""

        return np.cross(self.moment_body, _vector3(magnetic_field_body, "magnetic_field_body"))

    def with_updates(self, **kwargs) -> MagneticDipole:
        """Return a modified copy."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class ReactionWheel:
    """Single-axis reaction wheel actuator.

    ``axis_body`` is the body-frame torque axis. ``max_torque`` is the wheel
    torque limit in N m. ``wheel_inertia`` and ``speed`` initialize optional
    propagated wheel angular-momentum states in the 6-DoF propagator.
    """

    axis_body: ArrayLike
    max_torque: float
    momentum_capacity: float | None = None
    wheel_inertia: float | None = None
    speed: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_body", _unit(self.axis_body, "axis_body"))
        object.__setattr__(self, "max_torque", _positive(self.max_torque, "max_torque"))
        if self.momentum_capacity is not None:
            object.__setattr__(self, "momentum_capacity", _positive(self.momentum_capacity, "momentum_capacity"))
        if self.wheel_inertia is not None:
            object.__setattr__(self, "wheel_inertia", _positive(self.wheel_inertia, "wheel_inertia"))
        object.__setattr__(self, "speed", float(self.speed))

    def torque_body(self, command: float) -> np.ndarray:
        """Return saturated body-frame wheel torque in N m."""

        command = float(np.clip(command, -self.max_torque, self.max_torque))
        return self.axis_body * command

    def with_updates(self, **kwargs) -> ReactionWheel:
        """Return a modified copy."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class Facet:
    """Fixed body-frame surface used by facet drag and SRP models.

    ``cd`` and ``cl`` are the aerodynamic drag and lift coefficients. ``cr`` is
    the effective cannonball-style solar radiation pressure coefficient. Set
    ``specular_reflectivity`` or ``diffuse_reflectivity`` to use the optical
    flat-plate SRP model instead. ``vertices_body`` enables mesh-derived facets
    and optional self-shadowing in the SRP force model.
    """

    area: float
    normal_body: ArrayLike
    center_of_pressure: ArrayLike = (0.0, 0.0, 0.0)
    cd: float = 2.2
    cr: float = 1.3
    specular_reflectivity: float | None = None
    diffuse_reflectivity: float | None = None
    thermal_reemission: float = 0.0
    vertices_body: ArrayLike | None = None
    name: str = ""
    cl: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "area", _positive(self.area, "area"))
        object.__setattr__(self, "normal_body", _unit(self.normal_body, "normal_body"))
        object.__setattr__(self, "center_of_pressure", _vector3(self.center_of_pressure, "center_of_pressure"))
        object.__setattr__(self, "cd", _positive(self.cd, "cd"))
        object.__setattr__(self, "cl", _finite(self.cl, "cl"))
        object.__setattr__(self, "cr", _positive(self.cr, "cr"))
        if self.specular_reflectivity is not None:
            object.__setattr__(self, "specular_reflectivity", _unit_interval(self.specular_reflectivity, "specular_reflectivity"))
        if self.diffuse_reflectivity is not None:
            object.__setattr__(self, "diffuse_reflectivity", _unit_interval(self.diffuse_reflectivity, "diffuse_reflectivity"))
        object.__setattr__(self, "thermal_reemission", _unit_interval(self.thermal_reemission, "thermal_reemission"))
        specular = 0.0 if self.specular_reflectivity is None else self.specular_reflectivity
        diffuse = 0.0 if self.diffuse_reflectivity is None else self.diffuse_reflectivity
        if specular + diffuse > 1.0:
            raise ValueError("specular_reflectivity + diffuse_reflectivity must be <= 1.")
        if self.vertices_body is not None:
            vertices = np.asarray(self.vertices_body, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
                raise ValueError("vertices_body must be an array with shape (N, 3), N >= 3.")
            object.__setattr__(self, "vertices_body", tuple(tuple(row) for row in vertices))

    def with_updates(self, **kwargs) -> Facet:
        """Return a modified copy."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class Thruster:
    """Body-mounted thruster with body-frame direction and position."""

    thrust: float
    direction_body: ArrayLike
    position_body: ArrayLike = (0.0, 0.0, 0.0)
    isp: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "thrust", _positive(self.thrust, "thrust"))
        object.__setattr__(self, "direction_body", _unit(self.direction_body, "direction_body"))
        object.__setattr__(self, "position_body", _vector3(self.position_body, "position_body"))
        if self.isp is not None:
            object.__setattr__(self, "isp", _positive(self.isp, "isp"))

    def force_body(self, throttle: float = 1.0) -> np.ndarray:
        """Return body-frame force in newtons."""

        return self.direction_body * self.thrust * _throttle(throttle)

    def torque_body(self, throttle: float = 1.0) -> np.ndarray:
        """Return body-frame torque about the body origin."""

        return np.cross(self.position_body, self.force_body(throttle))

    def mass_flow_rate(self, throttle: float = 1.0) -> float:
        """Return positive propellant mass flow in kg/s."""

        if self.isp is None:
            return 0.0
        return self.thrust * _throttle(throttle) / (self.isp * G0)

    def with_updates(self, **kwargs) -> Thruster:
        """Return a modified copy."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class Tank:
    """Simple propellant tank mass component."""

    propellant_mass: float
    dry_mass: float = 0.0
    position_body: ArrayLike = (0.0, 0.0, 0.0)
    inertia: ArrayLike | None = None
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "propellant_mass", _nonnegative(self.propellant_mass, "propellant_mass"))
        object.__setattr__(self, "dry_mass", _nonnegative(self.dry_mass, "dry_mass"))
        object.__setattr__(self, "position_body", _vector3(self.position_body, "position_body"))
        if self.inertia is not None:
            object.__setattr__(self, "inertia", _inertia(self.inertia))

    @property
    def mass(self) -> float:
        """Current tank mass in kg."""

        return self.dry_mass + self.propellant_mass

    def inertia_about(self, origin: ArrayLike = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Return tank inertia about ``origin`` using the parallel-axis theorem."""

        inertia = np.zeros((3, 3)) if self.inertia is None else self.inertia
        return inertia + point_mass_inertia(self.mass, self.position_body - _vector3(origin, "origin"))

    def with_updates(self, **kwargs) -> Tank:
        """Return a modified copy."""

        return replace(self, **kwargs)

    def with_propellant_mass(self, propellant_mass: float) -> Tank:
        """Return a copy with updated propellant mass."""

        return self.with_updates(propellant_mass=_nonnegative(propellant_mass, "propellant_mass"))


@dataclass(frozen=True)
class SpacecraftBody:
    """Rigid spacecraft body definition for SSATK 6-DoF models.

    ``mass`` is the dry bus mass. ``current_mass`` adds components, tank dry
    mass, and tank propellant mass. Facet centers, thruster positions, and
    component positions are body-frame vectors measured from the same origin as
    ``center_of_mass``.
    """

    name: str
    mass: float
    inertia: ArrayLike
    center_of_mass: ArrayLike = (0.0, 0.0, 0.0)
    facets: tuple[Facet, ...] = ()
    thrusters: tuple[Thruster, ...] = ()
    magnetic_dipoles: tuple[MagneticDipole, ...] = ()
    reaction_wheels: tuple[ReactionWheel, ...] = ()
    tanks: tuple[Tank, ...] = ()
    components: tuple[Component, ...] = ()
    reference_area: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mass", _positive(self.mass, "mass"))
        object.__setattr__(self, "inertia", _inertia(self.inertia))
        object.__setattr__(self, "center_of_mass", _vector3(self.center_of_mass, "center_of_mass"))
        object.__setattr__(self, "facets", tuple(self.facets))
        object.__setattr__(self, "thrusters", tuple(self.thrusters))
        object.__setattr__(self, "magnetic_dipoles", tuple(self.magnetic_dipoles))
        object.__setattr__(self, "reaction_wheels", tuple(self.reaction_wheels))
        object.__setattr__(self, "tanks", tuple(self.tanks))
        object.__setattr__(self, "components", tuple(self.components))
        if self.reference_area is not None:
            object.__setattr__(self, "reference_area", _positive(self.reference_area, "reference_area"))

    @classmethod
    def box(
        cls,
        *,
        name: str = "box",
        mass: float,
        size: ArrayLike,
        cd: float = 2.2,
        cr: float = 1.3,
        inertia: ArrayLike | None = None,
    ) -> SpacecraftBody:
        """Return a rectangular bus with six flat facets."""

        size = _vector3(size, "size")
        if np.any(size <= 0.0):
            raise ValueError("size components must be positive.")
        inertia = rectangular_prism_inertia(mass, size) if inertia is None else inertia
        return cls(
            name=name,
            mass=mass,
            inertia=inertia,
            facets=box_facets(size, cd=cd, cr=cr),
            reference_area=float(np.max([size[1] * size[2], size[0] * size[2], size[0] * size[1]])),
        )

    @classmethod
    def box_wing(
        cls,
        *,
        name: str = "box_wing",
        mass: float,
        bus_size: ArrayLike = (1.0, 1.0, 1.0),
        solar_array_area: float = 4.0,
        solar_array_axis: str = "y",
        solar_array_mass: float = 0.0,
        cd: float = 2.2,
        cr: float = 1.3,
        inertia: ArrayLike | None = None,
    ) -> SpacecraftBody:
        """Return a box bus with two fixed solar-array facets."""

        bus_size = _vector3(bus_size, "bus_size")
        if np.any(bus_size <= 0.0):
            raise ValueError("bus_size components must be positive.")
        solar_array_area = _positive(solar_array_area, "solar_array_area")
        solar_array_mass = _nonnegative(solar_array_mass, "solar_array_mass")
        body = cls.box(name=name, mass=mass, size=bus_size, cd=cd, cr=cr, inertia=inertia)
        axis = _axis_index(solar_array_axis)
        normals = [np.eye(3)[axis], -np.eye(3)[axis]]
        centers = [normal * (bus_size[axis] / 2.0) for normal in normals]
        wing_facets = tuple(
            Facet(
                name=f"solar_array_{sign}",
                area=solar_array_area / 2.0,
                normal_body=normal,
                center_of_pressure=center,
                cd=cd,
                cr=cr,
                vertices_body=_square_vertices(center, normal, solar_array_area / 2.0),
            )
            for sign, normal, center in zip(("plus", "minus"), normals, centers)
        )
        wing_components = ()
        if solar_array_mass > 0.0:
            wing_components = tuple(
                Component(name=f"solar_array_{sign}", mass=solar_array_mass / 2.0, position_body=center)
                for sign, center in zip(("plus", "minus"), centers)
            )
        return body.with_updates(
            facets=body.facets + wing_facets,
            components=body.components + wing_components,
            reference_area=max(body.area, solar_array_area),
        )

    @classmethod
    def cubesat(
        cls,
        units: int = 3,
        *,
        name: str | None = None,
        mass: float | None = None,
        cd: float = 2.2,
        cr: float = 1.3,
    ) -> SpacecraftBody:
        """Return a simple N-U CubeSat bus."""

        if units <= 0:
            raise ValueError("units must be positive.")
        mass = 1.33 * units if mass is None else mass
        return cls.box(
            name=name or f"{units}u_cubesat",
            mass=mass,
            size=(0.10, 0.10, 0.10 * units),
            cd=cd,
            cr=cr,
        )

    @property
    def current_mass(self) -> float:
        """Dry bus mass plus attached components and tank mass."""

        return self.mass + sum(component.mass for component in self.components) + sum(tank.mass for tank in self.tanks)

    @property
    def dry_mass_total(self) -> float:
        """Bus, component, and tank dry mass without propellant."""

        return self.mass + sum(component.mass for component in self.components) + sum(tank.dry_mass for tank in self.tanks)

    @property
    def propellant_mass(self) -> float:
        """Total propellant mass currently assigned to tanks."""

        return sum(tank.propellant_mass for tank in self.tanks)

    @property
    def current_center_of_mass(self) -> np.ndarray:
        """Mass-weighted body-frame center of mass."""

        weighted = self.mass * self.center_of_mass
        weighted += sum((component.mass * component.position_body for component in self.components), start=np.zeros(3))
        weighted += sum((tank.mass * tank.position_body for tank in self.tanks), start=np.zeros(3))
        return weighted / self.current_mass

    @property
    def current_inertia(self) -> np.ndarray:
        """Body inertia about the current center of mass."""

        center = self.current_center_of_mass
        inertia = self.inertia + point_mass_inertia(self.mass, self.center_of_mass - center)
        inertia += sum((component.inertia_about(center) for component in self.components), start=np.zeros((3, 3)))
        inertia += sum((tank.inertia_about(center) for tank in self.tanks), start=np.zeros((3, 3)))
        return inertia

    @property
    def area(self) -> float:
        """Reference area used for cannonball fallbacks."""

        if self.reference_area is not None:
            return self.reference_area
        return max((facet.area for facet in self.facets), default=1.0)

    def with_updates(self, **kwargs) -> SpacecraftBody:
        """Return a modified copy."""

        return replace(self, **kwargs)

    def with_propellant_mass(self, propellant_mass: float) -> SpacecraftBody:
        """Return a body with total tank propellant set proportionally."""

        propellant_mass = _nonnegative(propellant_mass, "propellant_mass")
        current_propellant = self.propellant_mass
        if not self.tanks:
            if propellant_mass == 0.0:
                return self
            raise ValueError("cannot assign propellant mass without tanks.")
        if current_propellant == 0.0:
            if propellant_mass == 0.0:
                return self
            raise ValueError("cannot distribute propellant across empty tanks.")
        scale = propellant_mass / current_propellant
        return self.with_tanks(
            *(tank.with_propellant_mass(tank.propellant_mass * scale) for tank in self.tanks),
            append=False,
        )

    def with_tank_propellant_mass(self, name: str, propellant_mass: float) -> SpacecraftBody:
        """Return a copy with only the named tank's propellant changed."""

        propellant_mass = _nonnegative(propellant_mass, "propellant_mass")
        matches = [tank for tank in self.tanks if tank.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one tank named {name!r}.")
        return self.with_tanks(
            *(tank.with_propellant_mass(propellant_mass) if tank.name == name else tank for tank in self.tanks),
            append=False,
        )

    def with_current_mass(self, mass: float) -> SpacecraftBody:
        """Return a body whose tanks match the requested total current mass."""

        mass = _positive(mass, "mass")
        propellant_mass = mass - self.dry_mass_total
        if propellant_mass < -1e-9:
            raise ValueError("mass is below dry mass.")
        if propellant_mass > self.propellant_mass + 1e-9:
            raise ValueError("mass exceeds current tank propellant capacity.")
        return self.with_propellant_mass(max(0.0, propellant_mass))

    def with_facets(self, *facets: Facet, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced facets."""

        return self.with_updates(facets=(self.facets if append else ()) + tuple(facets))

    def with_thrusters(self, *thrusters: Thruster, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced thrusters."""

        return self.with_updates(thrusters=(self.thrusters if append else ()) + tuple(thrusters))

    def with_magnetic_dipoles(self, *dipoles: MagneticDipole, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced magnetic dipoles."""

        return self.with_updates(magnetic_dipoles=(self.magnetic_dipoles if append else ()) + tuple(dipoles))

    def with_reaction_wheels(self, *wheels: ReactionWheel, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced reaction wheels."""

        return self.with_updates(reaction_wheels=(self.reaction_wheels if append else ()) + tuple(wheels))

    def with_tanks(self, *tanks: Tank, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced tanks."""

        return self.with_updates(tanks=(self.tanks if append else ()) + tuple(tanks))

    def with_components(self, *components: Component, append: bool = True) -> SpacecraftBody:
        """Return a copy with added or replaced mass components."""

        return self.with_updates(components=(self.components if append else ()) + tuple(components))

    @classmethod
    def from_mesh(
        cls,
        *,
        name: str = "mesh_body",
        mass: float,
        vertices: ArrayLike,
        faces,
        inertia: ArrayLike | None = None,
        scale: float = 1.0,
        **facet_kwargs,
    ) -> SpacecraftBody:
        """Return a spacecraft body from a triangular or polygon mesh."""

        facets = mesh_facets(vertices, faces, scale=scale, **facet_kwargs)
        if inertia is None:
            verts = np.asarray(vertices, dtype=float) * float(scale)
            size = np.ptp(verts, axis=0)
            size = np.maximum(size, 1e-6)
            inertia = rectangular_prism_inertia(mass, size)
        return cls(
            name=name,
            mass=mass,
            inertia=inertia,
            facets=facets,
            reference_area=max((facet.area for facet in facets), default=1.0),
        )

    @classmethod
    def from_obj(
        cls,
        path,
        *,
        name: str | None = None,
        mass: float,
        inertia: ArrayLike | None = None,
        scale: float = 1.0,
        **facet_kwargs,
    ) -> SpacecraftBody:
        """Return a spacecraft body from a Wavefront OBJ mesh."""

        facets = load_obj_facets(path, scale=scale, **facet_kwargs)
        if inertia is None:
            vertices = np.vstack([np.asarray(facet.vertices_body, dtype=float) for facet in facets])
            size = np.maximum(np.ptp(vertices, axis=0), 1e-6)
            inertia = rectangular_prism_inertia(mass, size)
        return cls(
            name=name or str(path),
            mass=mass,
            inertia=inertia,
            facets=facets,
            reference_area=max((facet.area for facet in facets), default=1.0),
        )


def rectangular_prism_inertia(mass: float, size: ArrayLike) -> np.ndarray:
    """Return diagonal inertia for a uniform rectangular prism."""

    mass = _positive(mass, "mass")
    x, y, z = _vector3(size, "size")
    if min(x, y, z) <= 0.0:
        raise ValueError("size components must be positive.")
    return np.diag(
        [
            mass * (y**2 + z**2) / 12.0,
            mass * (x**2 + z**2) / 12.0,
            mass * (x**2 + y**2) / 12.0,
        ]
    )


def point_mass_inertia(mass: float, position: ArrayLike) -> np.ndarray:
    """Return point-mass inertia; zero mass gives zero inertia."""

    mass = _nonnegative(mass, "mass")
    offset = _vector3(position, "position")
    return mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))


def box_facets(size: ArrayLike, *, cd: float = 2.2, cr: float = 1.3) -> tuple[Facet, ...]:
    """Return six body-frame facets for a rectangular box."""

    x, y, z = _vector3(size, "size")
    if min(x, y, z) <= 0.0:
        raise ValueError("size components must be positive.")
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    specs = [
        ("+x", y * z, (1, 0, 0), (hx, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ("-x", y * z, (-1, 0, 0), (-hx, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ("+y", x * z, (0, 1, 0), (0, hy, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ("-y", x * z, (0, -1, 0), (0, -hy, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ("+z", x * y, (0, 0, 1), (0, 0, hz), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ("-z", x * y, (0, 0, -1), (0, 0, -hz), [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)]),
    ]
    return tuple(
        Facet(name=name, area=area, normal_body=normal, center_of_pressure=center, cd=cd, cr=cr, vertices_body=vertices)
        for name, area, normal, center, vertices in specs
    )


def rotate_facets(
    facets,
    *,
    axis_body: ArrayLike,
    angle_rad: float,
    origin_body: ArrayLike = (0.0, 0.0, 0.0),
) -> tuple[Facet, ...]:
    """Return facets rotated about a body-frame axis and origin."""

    axis = _unit(axis_body, "axis_body")
    origin = _vector3(origin_body, "origin_body")
    angle = float(angle_rad)
    return tuple(_rotate_facet(facet, axis, angle, origin) for facet in facets)


def mesh_facets(
    vertices: ArrayLike,
    faces,
    *,
    scale: float = 1.0,
    cd: float = 2.2,
    cr: float = 1.3,
    specular_reflectivity: float | None = None,
    diffuse_reflectivity: float | None = None,
    thermal_reemission: float = 0.0,
    name_prefix: str = "mesh",
) -> tuple[Facet, ...]:
    """Convert mesh vertices and faces into body-frame SRP/drag facets."""

    vertices = np.asarray(vertices, dtype=float) * float(scale)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")

    facets = []
    for index, face in enumerate(faces):
        face = tuple(int(i) for i in face)
        if len(face) < 3:
            raise ValueError("each face must contain at least three vertex indexes.")
        points = vertices[list(face)]
        area, normal, center = _polygon_area_normal_center(points)
        if area == 0.0:
            continue
        facets.append(
            Facet(
                name=f"{name_prefix}_{index}",
                area=area,
                normal_body=normal,
                center_of_pressure=center,
                cd=cd,
                cr=cr,
                specular_reflectivity=specular_reflectivity,
                diffuse_reflectivity=diffuse_reflectivity,
                thermal_reemission=thermal_reemission,
                vertices_body=points,
            )
        )
    return tuple(facets)


def load_obj_facets(path, **kwargs) -> tuple[Facet, ...]:
    """Load a Wavefront OBJ mesh as body-frame facets."""

    vertices = []
    faces = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                face = []
                for token in fields[1:]:
                    raw = int(token.split("/")[0])
                    face.append(raw - 1 if raw > 0 else len(vertices) + raw)
                faces.append(face)
    return mesh_facets(vertices, faces, **kwargs)


def reaction_wheel_triplet(
    *,
    max_torque: float,
    momentum_capacity: float | None = None,
    wheel_inertia: float | None = None,
    name_prefix: str = "rw",
) -> tuple[ReactionWheel, ReactionWheel, ReactionWheel]:
    """Return three orthogonal body-axis reaction wheels."""

    return (
        ReactionWheel(
            axis_body=(1.0, 0.0, 0.0),
            max_torque=max_torque,
            momentum_capacity=momentum_capacity,
            wheel_inertia=wheel_inertia,
            name=f"{name_prefix}_x",
        ),
        ReactionWheel(
            axis_body=(0.0, 1.0, 0.0),
            max_torque=max_torque,
            momentum_capacity=momentum_capacity,
            wheel_inertia=wheel_inertia,
            name=f"{name_prefix}_y",
        ),
        ReactionWheel(
            axis_body=(0.0, 0.0, 1.0),
            max_torque=max_torque,
            momentum_capacity=momentum_capacity,
            wheel_inertia=wheel_inertia,
            name=f"{name_prefix}_z",
        ),
    )


def cubesat_1u(**kwargs) -> SpacecraftBody:
    """Return a simple 1U CubeSat body."""

    return SpacecraftBody.cubesat(1, **kwargs)


def cubesat_3u(**kwargs) -> SpacecraftBody:
    """Return a simple 3U CubeSat body."""

    return SpacecraftBody.cubesat(3, **kwargs)


def cubesat_6u(**kwargs) -> SpacecraftBody:
    """Return a simple 6U CubeSat body."""

    return SpacecraftBody.cubesat(6, **kwargs)


def smallsat(**kwargs) -> SpacecraftBody:
    """Return a generic ESPA-class small satellite body."""

    defaults = {"name": "smallsat", "mass": 180.0, "bus_size": (0.8, 0.8, 1.0), "solar_array_area": 6.0}
    defaults.update(kwargs)
    return SpacecraftBody.box_wing(**defaults)


def earth_observation_sat(**kwargs) -> SpacecraftBody:
    """Return a generic agile LEO Earth-observation spacecraft body."""

    defaults = {
        "name": "earth_observation_sat",
        "mass": 650.0,
        "bus_size": (1.4, 1.4, 2.0),
        "solar_array_area": 12.0,
        "solar_array_mass": 35.0,
    }
    defaults.update(kwargs)
    return SpacecraftBody.box_wing(**defaults)


def gnss_sat(**kwargs) -> SpacecraftBody:
    """Return a generic MEO navigation satellite body."""

    defaults = {
        "name": "gnss_sat",
        "mass": 1_000.0,
        "bus_size": (1.7, 1.7, 2.2),
        "solar_array_area": 22.0,
        "solar_array_mass": 90.0,
    }
    defaults.update(kwargs)
    return SpacecraftBody.box_wing(**defaults)


def geo_bus(**kwargs) -> SpacecraftBody:
    """Return a generic GEO communications satellite body."""

    defaults = {"name": "geo_bus", "mass": 2_000.0, "bus_size": (2.5, 2.0, 3.0), "solar_array_area": 35.0}
    defaults.update(kwargs)
    return SpacecraftBody.box_wing(**defaults)


def cislunar_probe(**kwargs) -> SpacecraftBody:
    """Return a generic deep-space or cislunar probe body."""

    defaults = {
        "name": "cislunar_probe",
        "mass": 350.0,
        "bus_size": (1.1, 1.1, 1.4),
        "solar_array_area": 8.0,
        "solar_array_mass": 25.0,
    }
    defaults.update(kwargs)
    return SpacecraftBody.box_wing(**defaults)


def debris_panel(**kwargs) -> SpacecraftBody:
    """Return a simple non-controlled rectangular debris object."""

    defaults = {"name": "debris_panel", "mass": 30.0, "size": (2.0, 0.08, 1.0)}
    defaults.update(kwargs)
    return SpacecraftBody.box(**defaults)


def satellite_design(name: str, **overrides) -> SpacecraftBody:
    """Return a named preset body, optionally modified by keyword overrides."""

    key = name.lower().replace("-", "_").replace(" ", "_")
    presets = {
        "1u": cubesat_1u,
        "cubesat_1u": cubesat_1u,
        "3u": cubesat_3u,
        "cubesat_3u": cubesat_3u,
        "6u": cubesat_6u,
        "cubesat_6u": cubesat_6u,
        "smallsat": smallsat,
        "small_sat": smallsat,
        "earth_observation": earth_observation_sat,
        "earth_observation_sat": earth_observation_sat,
        "eo": earth_observation_sat,
        "eosat": earth_observation_sat,
        "gnss": gnss_sat,
        "gnss_sat": gnss_sat,
        "navigation": gnss_sat,
        "nav": gnss_sat,
        "geo": geo_bus,
        "geo_bus": geo_bus,
        "cislunar": cislunar_probe,
        "cislunar_probe": cislunar_probe,
        "debris": debris_panel,
        "debris_panel": debris_panel,
    }
    if key not in presets:
        raise ValueError(f"unknown satellite design {name!r}; choose one of {sorted(presets)}")
    return presets[key](**overrides)


def available_satellite_designs() -> tuple[str, ...]:
    """Return supported preset design names."""

    return (
        "cubesat_1u",
        "cubesat_3u",
        "cubesat_6u",
        "smallsat",
        "earth_observation_sat",
        "gnss_sat",
        "geo_bus",
        "cislunar_probe",
        "debris_panel",
    )


def _vector3(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return vector


def _unit(value: ArrayLike, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _throttle(value: float) -> float:
    return _nonnegative(value, "throttle")


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _polygon_area_normal_center(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    anchor = points[0]
    area_vector = np.zeros(3)
    weighted_center = np.zeros(3)
    total_area = 0.0
    for i in range(1, len(points) - 1):
        tri = (anchor, points[i], points[i + 1])
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = 0.5 * np.linalg.norm(cross)
        if area == 0.0:
            continue
        area_vector += 0.5 * cross
        weighted_center += area * (tri[0] + tri[1] + tri[2]) / 3.0
        total_area += area
    if total_area == 0.0:
        return 0.0, np.zeros(3), np.mean(points, axis=0)
    return total_area, area_vector / np.linalg.norm(area_vector), weighted_center / total_area


def _square_vertices(center: ArrayLike, normal: ArrayLike, area: float) -> tuple[tuple[float, float, float], ...]:
    center = _vector3(center, "center")
    normal = _unit(normal, "normal")
    side = np.sqrt(_positive(area, "area"))
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    axis1 = _unit(np.cross(normal, reference), "axis1")
    axis2 = np.cross(normal, axis1)
    half1 = 0.5 * side * axis1
    half2 = 0.5 * side * axis2
    vertices = (center - half1 - half2, center + half1 - half2, center + half1 + half2, center - half1 + half2)
    return tuple(tuple(vertex) for vertex in vertices)


def _rotate_facet(facet: Facet, axis: np.ndarray, angle: float, origin: np.ndarray) -> Facet:
    vertices = getattr(facet, "vertices_body", None)
    return facet.with_updates(
        normal_body=_rotate_axis_angle(facet.normal_body, axis, angle),
        center_of_pressure=origin + _rotate_axis_angle(facet.center_of_pressure - origin, axis, angle),
        vertices_body=None if vertices is None else tuple(
            tuple(origin + _rotate_axis_angle(np.asarray(vertex, dtype=float) - origin, axis, angle))
            for vertex in vertices
        ),
    )


def _rotate_axis_angle(vector: ArrayLike, axis: np.ndarray, angle: float) -> np.ndarray:
    vector = _vector3(vector, "vector")
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _inertia(value: ArrayLike) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("inertia must be symmetric.")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("inertia must be positive definite.")
    return matrix


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    key = axis.lower()
    if key not in axes:
        raise ValueError("solar_array_axis must be 'x', 'y', or 'z'.")
    return axes[key]


__all__ = [
    "G0",
    "Component",
    "Facet",
    "MagneticDipole",
    "ReactionWheel",
    "SpacecraftBody",
    "Tank",
    "Thruster",
    "available_satellite_designs",
    "box_facets",
    "cislunar_probe",
    "cubesat_1u",
    "cubesat_3u",
    "cubesat_6u",
    "debris_panel",
    "earth_observation_sat",
    "geo_bus",
    "gnss_sat",
    "load_obj_facets",
    "mesh_facets",
    "point_mass_inertia",
    "reaction_wheel_triplet",
    "rectangular_prism_inertia",
    "rotate_facets",
    "satellite_design",
    "smallsat",
]
