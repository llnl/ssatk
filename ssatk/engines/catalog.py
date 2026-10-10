"""Representative propulsion presets for SSATK spacecraft and rocket workflows.

The catalog gives engineering-scale defaults for simulation setup.  Values are
representative ranges, not procurement data.  Use mission/vendor thrust curves
from ``ssatk-data-propulsion`` with :class:`ssatk.accelerations_6dof.ThrustCurve` when
flight-specific fidelity is required.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from io import StringIO

import numpy as np

from ..accelerations_6dof import (
    SpacecraftManeuverAccel,
    thrust_profile_constant,
    thrust_profile_smoothstep,
    thrust_profile_trapezoid,
)
from ..constants import G0
from ..data import read_data_text
from ..satellites import Thruster

Range = tuple[float, float]

PACKAGED_THROTTLE_MAPS = {
    "aeps_etu2": "propulsion/throttle_maps/electric/aeps_etu2_throttle_map.csv",
    "spt140": "propulsion/throttle_maps/electric/spt140_performance_map.csv",
    "next_tt10": "propulsion/throttle_maps/electric/next_tt10_thrust_comparison.csv",
    "hermes_tdu3": "propulsion/throttle_maps/electric/hermes_tdu3_throttle_map.csv",
}


@dataclass(frozen=True)
class ThrusterSpec:
    """A reusable propulsion preset with thrust, Isp, power, and mass metadata."""

    name: str
    family: str
    scale: str
    propellant: str
    nominal_thrust_n: float
    nominal_isp_s: float
    thrust_range_n: Range
    isp_range_s: Range
    burn_style: str
    throttleable: bool = True
    restartable: bool = True
    power_range_w: Range | None = None
    dry_mass_range_kg: Range | None = None
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "family", _key(self.family))
        object.__setattr__(self, "scale", _key(self.scale))
        object.__setattr__(self, "propellant", str(self.propellant))
        object.__setattr__(self, "nominal_thrust_n", _positive(self.nominal_thrust_n, "nominal_thrust_n"))
        object.__setattr__(self, "nominal_isp_s", _positive(self.nominal_isp_s, "nominal_isp_s"))
        object.__setattr__(self, "thrust_range_n", _range(self.thrust_range_n, "thrust_range_n"))
        object.__setattr__(self, "isp_range_s", _range(self.isp_range_s, "isp_range_s"))
        if self.power_range_w is not None:
            object.__setattr__(self, "power_range_w", _range(self.power_range_w, "power_range_w"))
        if self.dry_mass_range_kg is not None:
            object.__setattr__(self, "dry_mass_range_kg", _range(self.dry_mass_range_kg, "dry_mass_range_kg"))
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases))
        if not self.thrust_range_n[0] <= self.nominal_thrust_n <= self.thrust_range_n[1]:
            raise ValueError(f"{self.name}: nominal_thrust_n must fall inside thrust_range_n.")
        if not self.isp_range_s[0] <= self.nominal_isp_s <= self.isp_range_s[1]:
            raise ValueError(f"{self.name}: nominal_isp_s must fall inside isp_range_s.")

    @property
    def nominal_power_w(self) -> float | None:
        """Midpoint power draw in watts when applicable."""

        if self.power_range_w is None:
            return None
        return 0.5 * (self.power_range_w[0] + self.power_range_w[1])

    @property
    def nominal_dry_mass_kg(self) -> float | None:
        """Midpoint dry thruster mass in kilograms when known."""

        if self.dry_mass_range_kg is None:
            return None
        return 0.5 * (self.dry_mass_range_kg[0] + self.dry_mass_range_kg[1])

    @property
    def exhaust_velocity_mps(self) -> float:
        """Effective exhaust velocity from nominal specific impulse."""

        return self.nominal_isp_s * G0

    def mass_flow_rate(self, thrust_n: float | None = None) -> float:
        """Return propellant mass flow in kg/s for the selected thrust."""

        return mass_flow_rate(self.nominal_thrust_n if thrust_n is None else thrust_n, self.nominal_isp_s)

    def acceleration_for_mass(self, spacecraft_mass_kg: float, thrust_n: float | None = None) -> float:
        """Return ideal acceleration in m/s² for a spacecraft mass."""

        return _nonnegative(self.nominal_thrust_n if thrust_n is None else thrust_n, "thrust_n") / _positive(spacecraft_mass_kg, "spacecraft_mass_kg")

    def with_updates(self, **kwargs) -> ThrusterSpec:
        """Return a modified copy for mission-specific tuning."""

        return replace(self, **kwargs)

    def to_thruster(
        self,
        *,
        direction_body=(1.0, 0.0, 0.0),
        position_body=(0.0, 0.0, 0.0),
        thrust_n: float | None = None,
        isp_s: float | None = None,
        name: str | None = None,
    ) -> Thruster:
        """Build a body-mounted :class:`ssatk.satellites.Thruster`."""

        return Thruster(
            name=self.name if name is None else name,
            thrust=self.nominal_thrust_n if thrust_n is None else thrust_n,
            isp=self.nominal_isp_s if isp_s is None else isp_s,
            direction_body=direction_body,
            position_body=position_body,
        )

    def thrust_profile(
        self,
        *,
        start: float = 0.0,
        stop: float | None = None,
        burn_time: float | None = None,
        thrust_n: float | None = None,
        throttle: float = 1.0,
        shape: str | None = None,
        rise_time: float = 0.0,
        fall_time: float | None = None,
    ):
        """Return a thrust-vs-time callable in newtons."""

        if not self.throttleable and not np.isclose(float(throttle), 1.0):
            raise ValueError(f"{self.name} is not throttleable; use throttle=1.0.")
        thrust = _nonnegative(self.nominal_thrust_n if thrust_n is None else thrust_n, "thrust_n") * float(throttle)
        if burn_time is None and stop is not None:
            burn_time = float(stop) - float(start)
        if not self.restartable and burn_time is None:
            raise ValueError(f"{self.name} requires a finite burn_time or stop time.")
        if burn_time is None:
            return thrust_profile_constant(thrust, start=start, stop=np.inf if stop is None else stop)

        profile_shape = _key(shape or ("trapezoid" if self.burn_style in {"solid", "liquid", "chemical"} else "smoothstep"))
        if profile_shape in {"constant", "steady"}:
            return thrust_profile_constant(thrust, start=start, stop=float(start) + float(burn_time))
        if profile_shape in {"smooth", "smoothstep"}:
            return thrust_profile_smoothstep(thrust, start=start, burn_time=burn_time, rise_time=rise_time, fall_time=fall_time)
        if profile_shape in {"trapezoid", "trapezoidal", "linear"}:
            return thrust_profile_trapezoid(thrust, start=start, burn_time=burn_time, rise_time=rise_time, fall_time=fall_time)
        raise ValueError("shape must be 'constant', 'trapezoid', or 'smoothstep'.")

    def maneuver_acceleration(
        self,
        *,
        direction=None,
        frame: str = "rtn",
        mass: float | None = None,
        start: float = 0.0,
        stop: float | None = None,
        burn_time: float | None = None,
        thrust_n: float | None = None,
        throttle: float = 1.0,
        shape: str | None = None,
        rise_time: float = 0.0,
        fall_time: float | None = None,
    ) -> SpacecraftManeuverAccel:
        """Build a finite-burn acceleration model from this preset."""

        return SpacecraftManeuverAccel(
            self.thrust_profile(
                start=start,
                stop=stop,
                burn_time=burn_time,
                thrust_n=thrust_n,
                throttle=throttle,
                shape=shape,
                rise_time=rise_time,
                fall_time=fall_time,
            ),
            direction=direction,
            frame=frame,
            mass=mass,
            isp=self.nominal_isp_s,
        )

def available_thruster_specs(family: str | None = None, scale: str | None = None) -> tuple[str, ...]:
    """Return catalog names, optionally filtered by family and scale."""

    family_key = None if family is None else _key(family)
    scale_key = None if scale is None else _key(scale)
    names = []
    for name, spec in _CATALOG.items():
        if family_key is not None and spec.family != family_key:
            continue
        if scale_key is not None and spec.scale != scale_key:
            continue
        names.append(name)
    return tuple(sorted(names))


def available_thruster_families() -> tuple[str, ...]:
    """Return supported propulsion families."""

    return tuple(sorted({spec.family for spec in _CATALOG.values()}))


def available_thruster_scales(family: str | None = None) -> tuple[str, ...]:
    """Return supported scale labels, optionally filtered by family."""

    family_key = None if family is None else _key(family)
    return tuple(sorted({spec.scale for spec in _CATALOG.values() if family_key is None or spec.family == family_key}))


def thruster_spec(name: str, *, scale: str | None = None) -> ThrusterSpec:
    """Return a propulsion preset by name, alias, or ``family`` + ``scale``."""

    if scale is not None:
        scale_key = _key(scale)
        matches = [spec for spec in _CATALOG.values() if spec.family == _key(name) and spec.scale == scale_key]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"No thruster spec for family={name!r}, scale={scale!r}.")
        raise KeyError(f"Ambiguous thruster spec for family={name!r}, scale={scale!r}.")

    key = _ALIASES.get(_key(name), _key(name))
    if key not in _CATALOG:
        raise KeyError(f"Unknown thruster spec {name!r}. Available: {available_thruster_specs()}")
    return _CATALOG[key]


def build_thruster(name: str, *, scale: str | None = None, **kwargs) -> Thruster:
    """Build a body-mounted thruster from a catalog preset."""

    return thruster_spec(name, scale=scale).to_thruster(**kwargs)


def make_thruster_profile(name: str, *, scale: str | None = None, **kwargs):
    """Build a thrust profile callable from a catalog preset."""

    return thruster_spec(name, scale=scale).thrust_profile(**kwargs)


def make_thruster_acceleration(name: str, *, scale: str | None = None, **kwargs) -> SpacecraftManeuverAccel:
    """Build a finite-burn maneuver acceleration from a catalog preset."""

    return thruster_spec(name, scale=scale).maneuver_acceleration(**kwargs)


def thruster_catalog_dict() -> dict[str, ThrusterSpec]:
    """Return a copy of the propulsion catalog."""

    return dict(_CATALOG)


def available_throttle_maps() -> tuple[str, ...]:
    """Return packaged electric-propulsion throttle-map names."""

    return tuple(sorted(PACKAGED_THROTTLE_MAPS))


def load_throttle_map(name: str) -> tuple[dict[str, object], ...]:
    """Load a packaged electric-propulsion throttle/performance map.

    Values are returned as dictionaries with numeric fields converted to
    ``float`` where possible. The source CSV remains in ``ssatk-data-propulsion``.
    """

    key = _key(name)
    try:
        relative_path = PACKAGED_THROTTLE_MAPS[key]
    except KeyError as exc:
        choices = ", ".join(available_throttle_maps())
        raise KeyError(f"Unknown throttle map {name!r}. Available: {choices}.") from exc
    rows = csv.DictReader(StringIO(read_data_text(relative_path)))
    return tuple(_typed_row(row) for row in rows)


def mass_flow_rate(thrust_n: float, isp_s: float) -> float:
    """Return ideal rocket mass flow rate in kg/s."""

    return _nonnegative(thrust_n, "thrust_n") / (_positive(isp_s, "isp_s") * G0)


def propellant_mass_for_delta_v(delta_v_mps: float, wet_mass_kg: float, isp_s: float) -> float:
    """Return ideal propellant mass for a target delta-v using Tsiolkovsky."""

    delta_v_mps = _nonnegative(delta_v_mps, "delta_v_mps")
    wet_mass_kg = _positive(wet_mass_kg, "wet_mass_kg")
    mass_ratio = np.exp(delta_v_mps / (_positive(isp_s, "isp_s") * G0))
    return float(wet_mass_kg * (1.0 - 1.0 / mass_ratio))


def _typed_row(row: dict[str, str]) -> dict[str, object]:
    typed: dict[str, object] = {}
    for key, value in row.items():
        if key is None:
            continue
        if value in {"", None}:
            typed[key] = None
            continue
        try:
            typed[key] = float(value)
        except ValueError:
            typed[key] = value
    return typed


def _spec(
    name: str,
    family: str,
    scale: str,
    propellant: str,
    nominal_thrust_n: float,
    nominal_isp_s: float,
    thrust_range_n: Range,
    isp_range_s: Range,
    burn_style: str,
    **kwargs,
) -> ThrusterSpec:
    return ThrusterSpec(
        name=name,
        family=family,
        scale=scale,
        propellant=propellant,
        nominal_thrust_n=nominal_thrust_n,
        nominal_isp_s=nominal_isp_s,
        thrust_range_n=thrust_range_n,
        isp_range_s=isp_range_s,
        burn_style=burn_style,
        **kwargs,
    )


def _catalog() -> dict[str, ThrusterSpec]:
    specs = [
        _spec("cold_gas_micro", "cold_gas", "micro", "nitrogen/R134a", 0.005, 55.0, (0.0001, 0.05), (35.0, 80.0), "cold_gas", aliases=("vacco_mips_cold_gas_thruster", "mips")),
        _spec("cold_gas_cubesat", "cold_gas", "cubesat", "nitrogen/R134a", 0.02, 60.0, (0.001, 1.0), (40.0, 80.0), "cold_gas"),
        _spec("monoprop_1n", "monopropellant", "small", "hydrazine/green monopropellant", 1.0, 225.0, (0.1, 5.0), (200.0, 260.0), "chemical", aliases=("mr_103g", "mr-103g")),
        _spec("monoprop_22n", "monopropellant", "medium", "hydrazine/green monopropellant", 22.0, 230.0, (5.0, 100.0), (210.0, 260.0), "chemical"),
        _spec("monoprop_500n", "monopropellant", "large", "hydrazine", 527.0, 219.0, (100.0, 700.0), (205.0, 235.0), "chemical", aliases=("mr_104j", "mr-104j"), dry_mass_range_kg=(4.0, 10.0)),
        _spec("biprop_10n", "bipropellant", "small", "MMH/NTO", 10.0, 292.0, (5.0, 25.0), (280.0, 325.0), "chemical", aliases=("10n_bipropellant_thruster",)),
        _spec("biprop_400n", "bipropellant", "medium", "MMH/NTO", 445.0, 312.0, (100.0, 600.0), (300.0, 330.0), "chemical", aliases=("r_4d", "r-4d")),
        _spec("liquid_apogee_engine", "liquid", "upper_stage", "MMH/NTO or LOX/LH2", 5_000.0, 330.0, (400.0, 110_000.0), (300.0, 465.0), "liquid"),
        _spec("solid_kick_motor_small", "solid", "small", "solid composite", 2_000.0, 285.0, (100.0, 20_000.0), (220.0, 310.0), "solid", throttleable=False, restartable=False),
        _spec("solid_kick_motor_large", "solid", "large", "solid composite", 100_000.0, 290.0, (20_000.0, 1_000_000.0), (240.0, 315.0), "solid", throttleable=False, restartable=False),
        _spec("resistojet_micro", "resistojet", "micro", "water/ammonia/hydrazine decomposition gas", 0.005, 150.0, (0.001, 0.05), (100.0, 350.0), "electric", power_range_w=(5.0, 100.0), aliases=("busek_micro_resistojet",)),
        _spec("resistojet_1n", "resistojet", "small", "hydrazine decomposition gas", 0.8, 299.0, (0.05, 2.0), (200.0, 350.0), "electric", power_range_w=(100.0, 1_000.0), aliases=("mr_502", "mr-502")),
        _spec("arcjet_1kw", "arcjet", "small", "hydrazine/ammonia", 0.2, 500.0, (0.02, 2.0), (350.0, 700.0), "electric", power_range_w=(500.0, 5_000.0)),
        _spec("electrospray_micro", "electrospray", "micro", "ionic liquid", 0.0001, 1_200.0, (1.0e-6, 1.0e-3), (500.0, 2_500.0), "electric", power_range_w=(1.0, 100.0)),
        _spec("gridded_ion_small", "gridded_ion", "small", "xenon/krypton", 0.03, 3_000.0, (0.001, 0.25), (1_500.0, 4_500.0), "electric", power_range_w=(100.0, 7_000.0), aliases=("ion", "ion_thruster")),
        _spec("hall_effect_small", "hall_effect", "small", "xenon/krypton", 0.083, 1_604.0, (0.005, 0.3), (1_000.0, 2_000.0), "electric", power_range_w=(200.0, 3_000.0), dry_mass_range_kg=(2.0, 8.0), aliases=("spt_100", "spt-100", "hall")),
        _spec("hall_effect_high_power", "hall_effect", "large", "xenon/krypton", 0.6, 2_800.0, (0.3, 5.0), (1_500.0, 3_500.0), "electric", power_range_w=(5_000.0, 100_000.0), dry_mass_range_kg=(20.0, 100.0), aliases=("aeps",)),
        _spec("dual_mode_chemical", "dual_mode", "chemical", "hydrazine/xenon or mission-specific", 10.0, 230.0, (0.1, 500.0), (200.0, 330.0), "chemical", aliases=("combined_chemical",)),
        _spec("dual_mode_electric", "dual_mode", "electric", "xenon/krypton", 0.05, 1_800.0, (0.001, 1.0), (1_000.0, 3_500.0), "electric", power_range_w=(100.0, 20_000.0), aliases=("combined_electric", "combined_electronic")),
        _spec("mira_transfer_stage", "bipropellant", "transfer_stage", "nitrous oxide/ethane", 208.0, 290.0, (100.0, 300.0), (270.0, 310.0), "chemical", aliases=("mira",)),
    ]
    return {_key(spec.name): spec for spec in specs}


def _aliases(catalog: dict[str, ThrusterSpec]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, spec in catalog.items():
        aliases[key] = key
        aliases[_key(spec.name)] = key
        for alias in spec.aliases:
            aliases[_key(alias)] = key
    return aliases


def _key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _range(value: Range, name: str) -> Range:
    low, high = float(value[0]), float(value[1])
    if low <= 0.0 or high <= 0.0 or high < low:
        raise ValueError(f"{name} must contain positive increasing values.")
    return low, high


_CATALOG = _catalog()
_ALIASES = _aliases(_CATALOG)

__all__ = [
    "ThrusterSpec",
    "available_throttle_maps",
    "available_thruster_families",
    "available_thruster_scales",
    "available_thruster_specs",
    "build_thruster",
    "load_throttle_map",
    "make_thruster_acceleration",
    "make_thruster_profile",
    "mass_flow_rate",
    "propellant_mass_for_delta_v",
    "thruster_catalog_dict",
    "thruster_spec",
]
