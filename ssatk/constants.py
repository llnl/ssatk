"""Physical and astronomical constants exposed through SSATK.

Shared astrodynamics constants are sourced from :mod:`ssapy.constants` so the
Toolkit and base SSAPy cannot drift. Toolkit-specific convenience constants
remain defined here for plotting, demos, and engineering calculations.
"""

import numpy as np
from ssapy import constants as _ssapy_constants


_SSAPY_CONSTANT_NAMES = tuple(
    name for name in dir(_ssapy_constants) if name.isupper()
)

for _name in _SSAPY_CONSTANT_NAMES:
    globals()[_name] = getattr(_ssapy_constants, _name)


# Toolkit-specific material, unit, angle, and time conveniences.
W_rho = 19280  # kg/m^3, density of tungsten

au_to_m = 149597870700  # IAU 2012 Resolution B2, exact
pc_to_au = 648000 / np.pi  # IAU 2015 Resolution B2
pc_to_m = au_to_m * pc_to_au
km_to_m = 1000

deg_to_arcsecond = 3600
rad_to_arcsecond = 180 * 3600 / np.pi
rad_to_deg = 180 / np.pi

day_to_second = 86400
year_to_second = 31557600
year_to_minute = 525960
year_to_hour = 8766
year_to_day = 365.25
year_to_week = 365.25 / 7
year_to_month = 12

kg_to_g = 1000
aupyr_to_mps = au_to_m / year_to_second  # au per Julian year, in m/s
# REBOUND's G = 1 velocity unit in au and Msun: au per yr2pi = 1/k day, with the
# Gaussian gravitational constant k = 0.01720209895 (IAU 1976).
v_rebound_to_si = au_to_m * 0.01720209895 / day_to_second

c = 299792458  # speed of light m/s
G = 6.67430e-11  # gravitational constant m3 kg-1 s-2, CODATA 2018
J2_wgs = 1.08262668e-3
kb = 1.380649e-23  # Boltzmann constant m2 kg s-2 K-1, exact since the 2019 SI
pi = np.pi

STANDARD_GRAVITY = 9.80665  # m/s^2
G0 = STANDARD_GRAVITY
SOLAR_FLUX_1_AU = 1361.0  # W/m^2
SOLAR_CONSTANT = SOLAR_FLUX_1_AU

EARTH_MEAN_RADIUS_KM = 6371.0
EARTH_MEAN_RADIUS = EARTH_MEAN_RADIUS_KM * km_to_m
EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM = 6371.2
EARTH_GEOMAGNETIC_REFERENCE_RADIUS = EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM * km_to_m
EARTH_DIPOLE_EQUATOR_FIELD = 3.12e-5  # tesla, approximate surface equatorial field
EARTH_DIPOLE_POLE_FIELD = 2.0 * EARTH_DIPOLE_EQUATOR_FIELD
WGS84_A_KM = _ssapy_constants.WGS84_EARTH_RADIUS / km_to_m
WGS84_B_KM = _ssapy_constants.WGS84_EARTH_POLAR_RADIUS / km_to_m
MOON_RADIUS_KM = _ssapy_constants.MOON_RADIUS / km_to_m
MOON_MEAN_RADIUS_KM = 1737.4
MOON_MEAN_RADIUS = MOON_MEAN_RADIUS_KM * km_to_m
EARTH_MU_KM3_S2 = _ssapy_constants.EARTH_MU / km_to_m**3
LD_KM = _ssapy_constants.LD / km_to_m
LUNAR_DISTANCE_KM = LD_KM
SIDEREAL_DAY_SECONDS = 86164.0905
EARTH_OBLIQUITY_J2000_DEG = 23.439291
EARTH_OBLIQUITY_J2000_RAD = np.radians(EARTH_OBLIQUITY_J2000_DEG)


# Solar-system convenience values not currently provided by base SSAPy.
MERCURY_a = 0.3871
VENUS_a = 0.7233
EARTH_a = 1.000
MARS_a = 1.5237
JUPITER_a = 5.2028
SATURN_a = 9.5388
URANUS_a = 19.1914
NEPTUNE_a = 30.0611

MERCURY_hill = 0.1753e9
VENUS_hill = 1.0042e9
EARTH_hill = 1.4714e9
MARS_hill = 0.9827e9
JUPITER_hill = 50.5736e9
SATURN_hill = 61.6340e9
URANUS_hill = 66.7831e9
NEPTUNE_hill = 115.0307e9
CERES_hill = 0.2048e9
PLUTO_hill = 5.9921e9
ERIS_hill = 8.1176e9

SUN_RADIUS = 696340000.0
SUN_NOMINAL_RADIUS_KM = 695700.0
SUN_NOMINAL_RADIUS = SUN_NOMINAL_RADIUS_KM * km_to_m
PLUTO_RADIUS = 195000.0

# Uppercase SI aliases for common astronomical units.
AU = float(au_to_m)
AU_M = AU
AU_KM = AU / km_to_m
SUN_RADIUS_KM = SUN_RADIUS / km_to_m
SUN_RADIUS_AU = SUN_RADIUS / AU
SUN_EARTH_AVERAGE_DISTANCE = AU
SUN_EARTH_AVERAGE_DISTANCE_KM = AU_KM
SUN_EARTH_AVERAGE_DISTANCE_AU = 1.0
SUN_ANGULAR_RADIUS_RAD = np.arctan2(SUN_RADIUS, SUN_EARTH_AVERAGE_DISTANCE)
SUN_ANGULAR_DIAMETER_DEG = 2.0 * SUN_ANGULAR_RADIUS_RAD * 180.0 / np.pi

# Solar-system body convenience aliases.  SSAPy supplies the SI radii, masses,
# and gravitational parameters for the eight planets; Toolkit adds uppercase
# semi-major-axis aliases and km/AU forms for plotting and demos.
PLANET_NAMES = (
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
)
SOLAR_SYSTEM_BODY_NAMES = ("Sun",) + PLANET_NAMES

_PLANET_SEMI_MAJOR_AXIS_AU = {
    "MERCURY": MERCURY_a,
    "VENUS": VENUS_a,
    "EARTH": EARTH_a,
    "MARS": MARS_a,
    "JUPITER": JUPITER_a,
    "SATURN": SATURN_a,
    "URANUS": URANUS_a,
    "NEPTUNE": NEPTUNE_a,
}
_SOLAR_SYSTEM_CONSTANT_NAMES = [
    "AU",
    "AU_KM",
    "SUN_RADIUS_KM",
    "SUN_RADIUS_AU",
    "AU_M",
    "SUN_EARTH_AVERAGE_DISTANCE",
    "SUN_EARTH_AVERAGE_DISTANCE_KM",
    "SUN_EARTH_AVERAGE_DISTANCE_AU",
    "SUN_ANGULAR_RADIUS_RAD",
    "SUN_ANGULAR_DIAMETER_DEG",
    "PLANET_NAMES",
    "SOLAR_SYSTEM_BODY_NAMES",
]

for _body, _a_au in _PLANET_SEMI_MAJOR_AXIS_AU.items():
    globals()[f"{_body}_SEMI_MAJOR_AXIS_AU"] = float(_a_au)
    globals()[f"{_body}_SEMI_MAJOR_AXIS"] = float(_a_au) * AU
    globals()[f"{_body}_SEMI_MAJOR_AXIS_KM"] = float(_a_au) * AU_KM
    globals()[f"{_body}_RADIUS_KM"] = globals()[f"{_body}_RADIUS"] / km_to_m
    _SOLAR_SYSTEM_CONSTANT_NAMES.extend([
        f"{_body}_SEMI_MAJOR_AXIS_AU",
        f"{_body}_SEMI_MAJOR_AXIS",
        f"{_body}_SEMI_MAJOR_AXIS_KM",
        f"{_body}_RADIUS_KM",
    ])

PLANET_SEMI_MAJOR_AXIS_AU = {
    name.title(): globals()[f"{name}_SEMI_MAJOR_AXIS_AU"]
    for name in _PLANET_SEMI_MAJOR_AXIS_AU
}
PLANET_SEMI_MAJOR_AXIS = {
    name.title(): globals()[f"{name}_SEMI_MAJOR_AXIS"]
    for name in _PLANET_SEMI_MAJOR_AXIS_AU
}
PLANET_SEMI_MAJOR_AXIS_KM = {
    name.title(): globals()[f"{name}_SEMI_MAJOR_AXIS_KM"]
    for name in _PLANET_SEMI_MAJOR_AXIS_AU
}
PLANET_RADIUS_KM = {
    name.title(): globals()[f"{name}_RADIUS_KM"]
    for name in _PLANET_SEMI_MAJOR_AXIS_AU
}
SOLAR_SYSTEM_RADIUS_KM = {"Sun": SUN_RADIUS_KM, **PLANET_RADIUS_KM}
_SOLAR_SYSTEM_CONSTANT_NAMES.extend([
    "PLANET_SEMI_MAJOR_AXIS_AU",
    "PLANET_SEMI_MAJOR_AXIS",
    "PLANET_SEMI_MAJOR_AXIS_KM",
    "PLANET_RADIUS_KM",
    "SOLAR_SYSTEM_RADIUS_KM",
])


_TOOLKIT_CONSTANT_NAMES = (
    "W_rho",
    "au_to_m",
    "pc_to_au",
    "pc_to_m",
    "km_to_m",
    "deg_to_arcsecond",
    "rad_to_arcsecond",
    "rad_to_deg",
    "day_to_second",
    "year_to_second",
    "year_to_minute",
    "year_to_hour",
    "year_to_day",
    "year_to_week",
    "year_to_month",
    "kg_to_g",
    "v_rebound_to_si",
    "aupyr_to_mps",
    "c",
    "G",
    "J2_wgs",
    "kb",
    "pi",
    "STANDARD_GRAVITY",
    "G0",
    "SOLAR_FLUX_1_AU",
    "SOLAR_CONSTANT",
    "EARTH_MEAN_RADIUS_KM",
    "EARTH_MEAN_RADIUS",
    "EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM",
    "EARTH_GEOMAGNETIC_REFERENCE_RADIUS",
    "EARTH_DIPOLE_EQUATOR_FIELD",
    "EARTH_DIPOLE_POLE_FIELD",
    "WGS84_A_KM",
    "WGS84_B_KM",
    "MOON_RADIUS_KM",
    "MOON_MEAN_RADIUS_KM",
    "MOON_MEAN_RADIUS",
    "EARTH_MU_KM3_S2",
    "LD_KM",
    "LUNAR_DISTANCE_KM",
    "SIDEREAL_DAY_SECONDS",
    "EARTH_OBLIQUITY_J2000_DEG",
    "EARTH_OBLIQUITY_J2000_RAD",
    "MERCURY_a",
    "VENUS_a",
    "EARTH_a",
    "MARS_a",
    "JUPITER_a",
    "SATURN_a",
    "URANUS_a",
    "NEPTUNE_a",
    "MERCURY_hill",
    "VENUS_hill",
    "EARTH_hill",
    "MARS_hill",
    "JUPITER_hill",
    "SATURN_hill",
    "URANUS_hill",
    "NEPTUNE_hill",
    "CERES_hill",
    "PLUTO_hill",
    "ERIS_hill",
    "SUN_RADIUS",
    "SUN_NOMINAL_RADIUS_KM",
    "SUN_NOMINAL_RADIUS",
    "PLUTO_RADIUS",
)

__all__ = sorted(set(_SSAPY_CONSTANT_NAMES + _TOOLKIT_CONSTANT_NAMES + tuple(_SOLAR_SYSTEM_CONSTANT_NAMES)))

del _name
try:
    del _body, _a_au
except NameError:
    pass
