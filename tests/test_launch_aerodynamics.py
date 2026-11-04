"""Independent Step-7 thermophysical, interpolation, frame and dynamics references."""

from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers

from ssatk.coordinates.attitude import quaternion_from_matrix
from ssatk.coordinates.earth_fixed import itrf_to_gcrf_eop
from ssatk.data import data_path
from ssatk.launch import AerodynamicCoefficientTable, LaunchAerodynamics, LaunchAtmosphere, LaunchWeatherProfile
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank


def _profile(heights, pressure, temperature, wind):
    return LaunchWeatherProfile(heights, pressure, temperature, wind,
                                source="Specified synthetic ideal-gas reference",
                                valid_time_gps=(0, 2e9), latitude_range=(-90, 90), longitude_range=(-180, 180),
                                gas_constant=287.05, gamma=1.4, sutherland=(1.716e-5, 273.15, 110.4))


def _uniform_air():
    return SimpleNamespace(sample=lambda t, r: {"density_kg_m3": 1., "sound_speed_m_s": 300.,
                                               "dynamic_viscosity_pa_s": 1.8e-5,
                                               "atmosphere_velocity_gcrf_m_s": np.array([2., 3., 4.])})


def test_weather_knots_and_overlap_match_ideal_gas_and_sutherland():
    """R1: linear primitive profiles and ideal air, 1e-12 relative (SI units)."""
    lower = _profile([0, 2000], [100000, 80000], [300, 280], [[5, 0, 0], [7, 0, 0]])
    upper = _profile([1000, 3000], [90000, 80000], [280, 276], [[10, 0, 0], [10, 0, 0]])
    atmosphere = LaunchAtmosphere(lower, upper=upper, transition=(1000, 2000))
    for height in (0., 500., 1000., 1500., 2000., 2500., 3000.):
        weight = np.clip((height-1000)/1000, 0, 1)
        p = (1-weight)*(100000-10*height) + weight*(95000-5*height)
        temp = (1-weight)*(300-.01*height) + weight*(282-.002*height)
        expected = [p, temp, p/(287.05*temp), np.sqrt(1.4*287.05*temp),
                    1.716e-5*(temp/273.15)**1.5*(273.15+110.4)/(temp+110.4)]
        state = atmosphere.properties(height, 1e9, 0., 0.)
        np.testing.assert_allclose([state[key] for key in ("pressure_pa", "temperature_k", "density_kg_m3",
                                                          "sound_speed_m_s", "dynamic_viscosity_pa_s")],
                                   expected, rtol=1e-12, atol=0)
        np.testing.assert_allclose(state["wind_enu_m_s"], [(1-weight)*(5+.001*height)+weight*10, 0, 0],
                                   rtol=1e-12, atol=0)
    # R5: both sides approach the prescribed endpoint, rather than a discontinuous splice.
    for boundary, p, temp in ((1000., 90000., 290.), (2000., 85000., 278.)):
        for offset in (-1e-6, 0., 1e-6):
            state = atmosphere.properties(boundary+offset, 1e9, 0., 0.)
            np.testing.assert_allclose([state["pressure_pa"], state["temperature_k"]], [p, temp],
                                       atol=3e-5, rtol=0)  # Pa and K, bounded by profile slopes * 1 um.


def test_four_axis_interpolation_matches_multiaffine_closed_form():
    """R1: multilinear polynomials including cross terms, 2e-14 dimensionless."""
    axes = {"mach": [0., 2.], "alpha": [-.3, .3], "beta": [-.2, .2], "deflection": [-.1, .1]}
    m, a, b, d = np.meshgrid(*axes.values(), indexing="ij")
    force = np.stack((-1-m*a, b*(1+m), -a+2*d), axis=-1)
    moment = np.stack((b*d, a*(1+m*b), -b+d), axis=-1)
    table = AerodynamicCoefficientTable(axes, force, moment, source="Specified multiaffine reference")
    for mach, alpha, beta, deflection in ((0., -.3, -.2, -.1), (2., .3, .2, .1), (.7, .13, -.08, .02)):
        result = table(mach=mach, alpha=alpha, beta=beta, deflection=deflection)
        expected = [[-1-mach*alpha, beta*(1+mach), -alpha+2*deflection],
                    [beta*deflection, alpha*(1+mach*beta), -beta+deflection]]
        np.testing.assert_allclose(result, expected, atol=2e-14, rtol=0)


def test_symmetric_vehicle_zero_and_odd_load_limits():
    """R1: symmetric axial drag and odd transverse force/moment, 1e-12 N / N m."""
    alpha = np.array([-.3, 0., .3])
    table = AerodynamicCoefficientTable({"alpha": alpha},
                                        np.column_stack((np.full(3, -.4), np.zeros(3), -2*alpha)),
                                        np.column_stack((np.zeros(3), -alpha, np.zeros(3))), source="Symmetric reference")
    model = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 3, 1],
                               reynolds_length=3, reference_position_body=[0, 0, 0])
    results = []
    for angle in (-.2, 0., .2):
        craft = Spacecraft(r=[7e6, 0, 0], v=np.array([2., 3., 4.])+10*np.array([np.cos(angle), 0, np.sin(angle)]),
                           mass=100, inertia=np.eye(3))
        result = model.loads(t=0, r=craft.r, v=craft.v, q=craft.q, omega=craft.omega, spacecraft=craft)
        np.testing.assert_allclose(result["force_body_n"], [-40, 0, -200*angle], atol=1e-12, rtol=0)
        np.testing.assert_allclose(result["torque_center_body_n_m"], [0, -300*angle, 0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(result["reynolds_number"], 10*3/1.8e-5, rtol=1e-14, atol=0)
        results.append(result)
    np.testing.assert_allclose(results[0]["force_body_n"][[1, 2]], -results[2]["force_body_n"][[1, 2]],
                               atol=1e-12, rtol=0)


def test_wind_and_corotation_match_independent_astropy():
    """R2: Astropy pad motion and ENU axes; 1 mm/s, allowing IERS A/B/dX differences."""
    epoch = Time("2020-05-30T19:22:00", scale="utc")
    longitude, latitude, height = -80.6, 28.6, 500.
    wind = np.array([12., -7., 1.])
    profile = _profile([0, 2000], [100000, 80000], [300, 280], [wind, wind])
    location = EarthLocation.from_geodetic(longitude*u.deg, latitude*u.deg, height*u.m)
    with data_path("environment/eop/finals2000A.all") as path:
        reference_eop = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(reference_eop):
        position, velocity = location.get_gcrs_posvel(epoch)
        axes = ITRS(CartesianRepresentation(np.eye(3)*u.m), obstime=epoch).transform_to(GCRS(obstime=epoch))
    lon, lat = np.radians([longitude, latitude])
    enu = np.array([[-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                    [np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                    [0., np.cos(lat), np.sin(lat)]])
    air = LaunchAtmosphere(profile).sample(epoch, position.xyz.to_value(u.m))
    expected = velocity.xyz.to_value(u.m/u.s) + axes.cartesian.xyz.value @ enu @ wind
    np.testing.assert_allclose(air["atmosphere_velocity_gcrf_m_s"], expected, atol=1e-3, rtol=0)
    np.testing.assert_allclose(air["height_m"], height, atol=.1, rtol=0)  # same model/data comparison budget
    calm = _profile([0, 2000], [100000, 80000], [300, 280], np.zeros((2, 3)))
    calm_air = LaunchAtmosphere(calm).sample(epoch, position.xyz.to_value(u.m))
    np.testing.assert_allclose(calm_air["atmosphere_velocity_gcrf_m_s"], velocity.xyz.to_value(u.m/u.s),
                               atol=1e-3, rtol=0)


def test_reference_point_motion_and_moment_shift_follow_retained_tanks():
    """R1/R4: fixed-point velocity and moment translation with moving C, 1e-11 SI."""
    force_c = np.array([-.4, .2, -.1])
    moment_c = np.array([.03, -.02, .01])
    table = AerodynamicCoefficientTable({"mach": [0, 2]}, [force_c]*2, [moment_c]*2, source="Constant reference")
    # R is a prescribed +90-degree rotation about inertial Z, independent of quaternion helpers.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    q = quaternion_from_matrix(rotation)
    reference = np.array([2., 1., -.5])
    center_rate = np.array([.05, 0., 0.])
    omega = np.array([0., 0., .1])
    model = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 3, 2],
                               reynolds_length=3, reference_position_body=reference,
                               center_of_mass_rate=lambda t, r, v, q, w, craft: center_rate)
    for propellant in (5., 3., 1.):
        body = SpacecraftBody(name="synthetic retained tank", mass=10, inertia=np.eye(3), center_of_mass=[0, 0, 0],
                              tanks=(Tank(name="feed", propellant_mass=propellant, residual_mass=1.,
                                          position_body=[3, 0, 0]),))
        craft = Spacecraft(r=[7e6, 0, 0], v=[2, 103, 4], body=body, q=q, omega=omega)
        c = np.array([3*propellant/(10+propellant), 0, 0])
        arm = reference-c
        expected_air = np.array([100., 0., 0.]) + np.cross(omega, arm)-center_rate
        result = model.loads(t=0, r=craft.r, v=craft.v, q=q, omega=omega, spacecraft=craft)
        np.testing.assert_allclose(result["air_velocity_body_m_s"], expected_air, atol=1e-12, rtol=0)
        pressure = .5*(expected_air @ expected_air)
        expected_force = pressure*2*force_c
        expected_moment = pressure*2*np.array([1, 3, 2])*moment_c
        np.testing.assert_allclose(result["force_gcrf_n"], rotation @ expected_force, atol=1e-11, rtol=0)
        np.testing.assert_allclose(result["torque_center_body_n_m"], expected_moment+np.cross(arm, expected_force),
                                   atol=1e-11, rtol=0)
        np.testing.assert_allclose(result["acceleration_gcrf_m_s2"], rotation @ expected_force/(10+propellant),
                                   atol=1e-12, rtol=0)
    # R1: no relative flow is a known zero-load limit, even below a table's minimum Mach.
    stationary = Spacecraft(r=[7e6, 0, 0], v=[2, 3, 4], mass=100, inertia=np.eye(3))
    zero = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 1, 1],
                              reynolds_length=1, reference_position_body=[0, 0, 0])
    np.testing.assert_allclose(zero(stationary), np.zeros(3), atol=0, rtol=0)
    fixed_mass = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 1, 1],
                                    reynolds_length=1, reference_position_body=[0, 0, 0], mass=100)
    np.testing.assert_allclose(fixed_mass(stationary.r, stationary.v, stationary.t),
                               fixed_mass(stationary), atol=0, rtol=0)


def test_coupled_drag_propagation_matches_quadratic_closed_form():
    """R1: dv/dt=-k v^2, 1e-8 m/s and 1e-7 m, fixed synthetic inertial air."""
    table = AerodynamicCoefficientTable({"mach": [0, 2]}, [[-.4, 0, 0]]*2, [[0, 0, 0]]*2,
                                        source="Constant drag reference")
    air = _uniform_air()
    air.sample = lambda t, r: {"density_kg_m3": 1., "sound_speed_m_s": 300., "dynamic_viscosity_pa_s": 1.8e-5,
                              "atmosphere_velocity_gcrf_m_s": np.zeros(3)}
    model = LaunchAerodynamics(air, table, reference_area=2, reference_lengths=[1, 1, 1],
                               reynolds_length=1, reference_position_body=[0, 0, 0])
    craft = Spacecraft(r=[7e6, 0, 0], v=[100, 0, 0], mass=100, inertia=np.eye(3))
    times = np.linspace(0, 2, 21)
    trajectory = craft.propagate(times=times, models=[model], mu=0, rtol=1e-12, atol=1e-12)
    k = .5*1.*.4*2/100
    np.testing.assert_allclose(trajectory.v[:, 0], 100/(1+k*100*times), atol=1e-8, rtol=0)
    np.testing.assert_allclose(trajectory.r[:, 0], 7e6+np.log1p(k*100*times)/k, atol=1e-7, rtol=0)
    np.testing.assert_allclose(trajectory.omega, np.zeros((21, 3)), atol=0, rtol=0)


def test_coefficient_frame_maps_forward_to_existing_launch_body_z():
    """R1: table X->body Z, Y->Y, Z->-X; 1e-11 N / N m, 1e-14 radians."""
    table = AerodynamicCoefficientTable({"alpha": [-.1, .1], "beta": [-.1, .1]},
                                        np.tile([-.4, .1, .2], (2, 2, 1)),
                                        np.tile([.03, -.02, .01], (2, 2, 1)), source="Synthetic table-frame reference")
    model = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 3, 2],
                               reynolds_length=3, reference_position_body=[1, 0, 0],
                               coefficient_to_body=[[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    craft = Spacecraft(r=[7e6, 0, 0], v=[2, 3, 104], mass=100, inertia=np.eye(3))
    result = model.loads(t=0, r=craft.r, v=craft.v, q=craft.q, omega=craft.omega, spacecraft=craft)
    np.testing.assert_allclose([result["alpha_rad"], result["beta_rad"]], [0, 0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result["air_velocity_coefficient_m_s"], [100, 0, 0], atol=1e-12, rtol=0)
    # q_dyn*A = 10000; rotate coefficient force and dimensioned moment independently.
    np.testing.assert_allclose(result["force_body_n"], [-2000, 1000, -4000], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result["moment_reference_body_n_m"], [-200, -600, 300], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result["torque_center_body_n_m"], [-200, 3400, 1300], atol=1e-11, rtol=0)


def test_coupled_aerodynamic_moment_matches_isotropic_spin_closed_form():
    """R1: constant aerodynamic torque, 1e-11 rad/s and quaternion-component budgets."""
    table = AerodynamicCoefficientTable({"mach": [0, 2]}, [[0, 0, 0]]*2, [[0, 0, .01]]*2,
                                        source="Constant synthetic moment; independent of angles")
    model = LaunchAerodynamics(_uniform_air(), table, reference_area=2, reference_lengths=[1, 1, 3],
                               reynolds_length=1, reference_position_body=[0, 0, 0])
    craft = Spacecraft(r=[7e6, 0, 0], v=[102, 3, 4], mass=100, inertia=20*np.eye(3))
    times = np.linspace(0, .2, 21)
    trajectory = craft.propagate(times=times, models=[model], mu=0, rtol=1e-12, atol=1e-12)
    angular_acceleration = (.5*100**2*2*3*.01)/20
    np.testing.assert_allclose(trajectory.omega[:, 2], angular_acceleration*times, atol=1e-11, rtol=0)
    half_angle = angular_acceleration*times**2/4
    expected_q = np.column_stack((np.cos(half_angle), np.zeros((len(times), 2)), np.sin(half_angle)))
    np.testing.assert_allclose(trajectory.q, expected_q, atol=1e-11, rtol=0)


def test_evidence_demo_matches_prescribed_load_samples():
    """R1: prescribed p/T, speed and alpha, 1e-6 N and N m load budgets."""
    from demos.launch.demo_launch_aerodynamics import run_reference

    result = run_reference()
    for record, pressure, temperature in zip(result["records"], [95000, 86250, 82500], [295, 282, 277]):
        scale = .5*pressure/(287.05*temperature)*100**2*2
        np.testing.assert_allclose(record["loads"]["force_body_n"], scale*np.array([.2, 0, -.3]), atol=1e-6, rtol=0)
        np.testing.assert_allclose(record["loads"]["torque_center_body_n_m"], scale*np.array([0, -.03, 0]),
                                   atol=1e-6, rtol=0)
