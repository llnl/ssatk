"""Falcon 9 catalog example with declared synthetic dynamics and LEO acceptance."""

import argparse
from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.time import Time

from ssatk.accelerations_6dof import SpacecraftAccelJ2, SpacecraftAttitudePD, SpacecraftManeuverAccel
from ssatk.constants import EARTH_MU, EARTH_RADIUS, G0
from ssatk.coordinates.attitude import quaternion_from_matrix, rotate_vector
from ssatk.launch import (
    LaunchActuator, LaunchAerodynamics, LaunchAtmosphere, LaunchControl, LaunchEvent,
    LaunchGuide, LaunchPhase, LaunchWeatherProfile, assemble_stage_stack,
    check_launch_requirements, launch_ground_clearance, launch_initial_state,
    launch_vehicles, propagate_launch, separate_launch_body,
)
from ssatk.propagators_6dof import Spacecraft
from ssatk.propagators_6dof.sixdof import _body_at_state
from ssatk.launch.events import _state_vector
from ssatk.satellites import SpacecraftBody, Tank
from ssatk.launch.preflight import collect_software_baseline, verify_data_resources

GALLERY_INCLUDE = False


class _SphereCoefficients:
    source = "Synthetic sphere: drag coefficient 0.3, no aerodynamic moment"

    def __call__(self, *, mach, alpha, beta, deflection=0.):
        return -.3*np.array([np.cos(alpha)*np.cos(beta), np.sin(beta),
                             np.sin(alpha)*np.cos(beta)]), np.zeros(3)


def _requirements(inclination):
    # These are declared scenario inputs, never inferred from its final trajectory.
    values = {
        "max_dynamic_pressure": (500000., 0., "Pa", "at_most", "constraints.max_dynamic_pressure_pa"),
        "max_heating_rate": (2e7, 0., "W/m^2", "at_most", "constraints.max_heating_rate_w_m2"),
        "max_acceleration": (200., 0., "m/s^2", "at_most", "constraints.max_acceleration_m_s2"),
        "max_angle_of_attack": (180., 0., "deg", "at_most", "constraints.max_angle_of_attack_deg"),
        "minimum_propellant_reserve": (5., 0., "kg", "at_least", "constraints.final_propellant_reserve_kg"),
        "maximum_navigation_position_error": (0., 1e-9, "m", "at_most", "navigation.max_position_error_m"),
        "maximum_mission_duration": (650., 1e-6, "s", "at_most", "events.elapsed_time_s"),
        "frame_position_closure": (0., 1e-6, "m", "within", "initial_state.closure.position_m"),
        "frame_velocity_closure": (0., 1e-6, "m/s", "within", "initial_state.closure.velocity_m_s"),
        "target_perigee_altitude": (200000., 5000., "m", "within", "orbit.perigee_altitude_m"),
        "target_apogee_altitude": (200000., 5000., "m", "within", "orbit.apogee_altitude_m"),
        "target_eccentricity": (0., .001, "1", "within", "orbit.eccentricity"),
        "target_inclination": (inclination, .5, "deg", "within", "orbit.inclination_deg"),
    }
    return {name: dict(value=value, tolerance=tolerance, unit=unit, comparison=comparison,
                       check=field, evidence_field=field,
                       source="Declared synthetic 200 km LEO scenario; not a hardware acceptance budget")
            for name, (value, tolerance, unit, comparison, field) in values.items()}


def run_reference(*, requirements=None, solver=None):
    """One continuous pad/ascent/staging/insertion/coast event graph.

    Uses Falcon 9 Full Thrust catalog masses/engines with synthetic spherical
    drag, inertia, COM point tanks, constant-Isp engines,
    noiseless instantaneous sensors and bounded external attitude torques.
    The exponential ideal-gas atmosphere and wind are synthetic, not weather.
    Heating uses the declared synthetic coefficient k=1.83e-4 SI in
    k*sqrt(rho/nose_radius)*airspeed**3, with nose_radius=1 m. It is a reduced
    convective-heating proxy, not a calibrated thermal model.
    """
    initial = launch_initial_state("Kennedy Space Center LC-39A", "2020-05-30T19:22:00", heading_deg=90.)
    resources = verify_data_resources([{"path":"environment/eop/finals2000A.all"}])
    epoch = initial["epoch_gps_s"]
    rotation = np.asarray(initial["body_to_gcrf"])
    normal = np.cross(initial["gcrf_position_m"], rotation[:, 0])
    normal /= np.linalg.norm(normal)
    defaults = _requirements(np.degrees(np.arccos(normal[2])))
    requirements = defaults if requirements is None else requirements
    check_launch_requirements(requirements, {})
    for name, spec in defaults.items():
        if any(requirements[name][key] != spec[key] for key in ("unit", "evidence_field")):
            raise ValueError(f"{name}: use the documented SI evidence field and unit")
    radius_target = EARTH_RADIUS + (requirements["target_perigee_altitude"]["value"]
                                    + requirements["target_apogee_altitude"]["value"])/2
    vehicle = "Falcon 9 Full Thrust"
    catalog = deepcopy(launch_vehicles[vehicle])
    lower_spec, upper_spec = catalog["stages"]
    lower_engine, upper_engine = lower_spec["engines"][0], upper_spec["engines"][0]
    # Catalog kN are per engine; the mass/inertia assumptions are declared separately.
    thrusts = [1000.*lower_engine["count"]*lower_engine["thrust_SL"],
               1000.*upper_engine["count"]*upper_engine["thrust_vac"]]
    isps = [lower_engine["ISP_SL"], upper_engine["ISP_vac"]]
    payload = SpacecraftBody(name="sample payload", mass=1000., inertia=10000.*np.eye(3))
    upper_stage = SpacecraftBody(name="Falcon 9 catalog upper", mass=upper_spec["mass_empty"],
        inertia=10.*upper_spec["mass_empty"]*np.eye(3),
        tanks=(Tank(name="upper", propellant_mass=upper_spec["mass_propellant"],
                    residual_mass=.001*upper_spec["mass_propellant"]),))
    lower = SpacecraftBody(name="Falcon 9 catalog lower", mass=lower_spec["mass_empty"],
        inertia=10.*lower_spec["mass_empty"]*np.eye(3),
        tanks=(Tank(name="lower", propellant_mass=lower_spec["mass_propellant"],
                    residual_mass=.001*lower_spec["mass_propellant"]),))
    upper = assemble_stage_stack(payload, [upper_stage])
    body = assemble_stage_stack(upper, [lower])
    pitch_duration = lower.tanks[0].usable_propellant_mass*isps[0]*G0/thrusts[0]
    state = Spacecraft(r=initial["gcrf_position_m"], v=initial["gcrf_velocity_m_s"],
                       q=initial["q_body_to_gcrf"], omega=initial["omega_body_rad_s"], t=epoch, body=body)
    heights = np.arange(-2000., 702001., 2000.)
    weather = dict(source="Synthetic isothermal exponential atmosphere with constant ENU wind",
                   valid_time_gps=[epoch-10., epoch+660.], latitude_range=[-90., 90.],
                   longitude_range=[-180., 180.], gas_constant=287.05, gamma=1.4,
                   sutherland=[1.716e-5, 273.15, 110.4])
    profile = LaunchWeatherProfile(heights, 101325.*np.exp(-heights/8500.), np.full(len(heights), 280.),
                                   np.tile([5., 2., 0.], (len(heights), 1)), **weather)
    atmosphere = LaunchAtmosphere(profile)
    aero = LaunchAerodynamics(atmosphere, _SphereCoefficients(), reference_area=10.,
                             reference_lengths=[1., 1., 1.], reynolds_length=1., reference_position_body=[0., 0., 0.])
    torque_scale = np.trace(body.current_inertia)/3/2000.
    actuator = LaunchActuator(initial=[0., 0., 0., 1.], lower=[-500.*torque_scale]*3+[0.],
                              upper=[500.*torque_scale]*3+[1.], rate=[1000.*torque_scale]*3+[1.],
                              lag=[.2, .2, .2, .2], units=["N m", "N m", "N m", "1"],
                              source="Synthetic bounded external torque and engine-throttle actuators")
    pd = SpacecraftAttitudePD(kp=2000., kd=3000.)

    def sensor(truth):
        return dict(r=truth.r, v=truth.v, q=truth.q, omega=truth.omega, mass=truth.mass,
                    inertia_kg_m2=float(np.trace(truth.body.current_inertia)/3),
                    first_stage=any(tank.name == "lower" for tank in truth.body.tanks))

    def guidance(t, estimate):
        r, v = np.asarray(estimate["r"]), np.asarray(estimate["v"])
        radius = np.linalg.norm(r)
        up = r/radius
        horizontal = np.cross(normal, up)
        horizontal /= np.linalg.norm(horizontal)
        radial_speed, tangential_speed = np.dot(v, up), np.dot(v, horizontal)
        throttle = 1.
        if estimate["first_stage"]:
            pitch = np.radians(80.)*np.clip((t-epoch-8.)/pitch_duration, 0., 1.)
            direction = np.cos(pitch)*up + np.sin(pitch)*horizontal
        else:
            radial_target = np.clip((radius_target-radius)/70., -500., 1000.)
            radial = (radial_target-radial_speed)/25. + EARTH_MU/radius**2-tangential_speed**2/radius
            # Account for dv_t/dt = a_t - v_r*v_t/r, including retrograde correction.
            along = (np.sqrt(EARTH_MU/radius)-tangential_speed)/20.+radial_speed*tangential_speed/radius
            # Reserve acceleration for radial control when upper-stage thrust is limited.
            available = thrusts[1]/estimate["mass"]
            radial = np.clip(radial, -available, available)
            along_limit = np.sqrt(max(0., available**2-radial**2))
            along = np.clip(along, -along_limit, along_limit)
            acceleration = radial*up + along*horizontal
            magnitude = np.linalg.norm(acceleration)
            direction = acceleration/magnitude if magnitude else horizontal
            throttle = np.clip(magnitude*estimate["mass"]/thrusts[1], 0., 1.)
        side = normal-direction*np.dot(normal, direction)
        side /= np.linalg.norm(side)
        target = quaternion_from_matrix(np.column_stack((np.cross(side, direction), side, direction)))
        return dict(q=target, throttle=float(throttle))

    def controller(t, estimate, reference):
        pd.q_target = reference["q"]
        pd.kp, pd.kd = estimate["inertia_kg_m2"], 1.5*estimate["inertia_kg_m2"]
        torque = pd.torque(t=t, r=estimate["r"], v=estimate["v"], q=estimate["q"], omega=estimate["omega"])
        return np.r_[torque, reference["throttle"]]

    control = LaunchControl(sensor=sensor, estimator=lambda t, packets, prior:packets[-1]["measurement"],
                            guidance=guidance, controller=controller, actuator=actuator,
                            estimate=sensor(state), sensor_period=.5, control_period=.5)
    first = SpacecraftManeuverAccel(thrusts[0], frame="body", direction=[0., 0., 1.], isp=isps[0],
                                    tank_name="lower", throttle=lambda t,*args:actuator.value(t)[3])
    second = SpacecraftManeuverAccel(thrusts[1], frame="body", direction=[0., 0., 1.], isp=isps[1],
                                     tank_name="upper", throttle=lambda t,*args:actuator.value(t)[3])
    gravity = SpacecraftAccelJ2()
    engines = {"ascent": first, "upper_burn": second}
    last_observation = {}

    def observe(phase, t, y):
        # Several limit roots query the same state; keep only the current evaluation.
        key = (phase, float(t), tuple(y), tuple(actuator.target))
        if last_observation.get("key") == key:
            return last_observation["values"]
        first_stage = phase in ("pad", "ascent", "separation")
        current_body = _body_at_state(body if first_stage else upper, y[13], None,
                                      "lower" if first_stage else "upper")
        craft = SimpleNamespace(mass=float(y[13]), body=current_body)
        loads = aero.loads(t=t, r=y[:3], v=y[3:6], q=y[6:10], omega=y[10:13], spacecraft=craft)
        engine = engines.get(phase)
        # Use the powered phase's one-sided thrust at depletion; the next phase shuts it off.
        thrust = np.zeros(3) if engine is None else rotate_vector(y[6:10],
            engine.thrust_vector(t, y[:3], y[3:6], y[6:10], y[10:13]))/y[13]
        speed = np.linalg.norm(loads["air_velocity_body_m_s"])
        values = {"constraints.max_dynamic_pressure_pa": loads["dynamic_pressure_pa"],
                "constraints.max_heating_rate_w_m2": 1.83e-4*np.sqrt(loads["environment"]["density_kg_m3"])*speed**3,
                "constraints.max_acceleration_m_s2": np.linalg.norm(thrust+loads["acceleration_gcrf_m_s2"]),
                "constraints.max_angle_of_attack_deg": np.degrees(np.arccos(np.clip(
                    loads["air_velocity_body_m_s"][2]/speed, -1., 1.))) if speed else 0.,
                "constraints.final_propellant_reserve_kg":sum(t.propellant_mass for t in current_body.tanks)}
        last_observation.update(key=key, values=values)
        return values

    def split(craft):
        depleted = lower.with_tank_propellant_mass("lower", next(t.propellant_mass for t in craft.body.tanks if t.name == "lower"))
        return separate_launch_body(craft, (upper, depleted))

    def insertion(t, y):
        orbit = Spacecraft(r=y[:3], v=y[3:6], t=t).to_orbit()
        margins = []
        values = {"target_perigee_altitude": orbit.a*(1-orbit.e)-EARTH_RADIUS,
                  "target_apogee_altitude": orbit.a*(1+orbit.e)-EARTH_RADIUS,
                  "target_eccentricity": orbit.e, "target_inclination": np.degrees(orbit.i)}
        for name, value in values.items():
            spec = requirements[name]
            # Leave half the declared orbit budget for J2 osculating variation during coast.
            margins.append((.5*spec["tolerance"]-abs(value-spec["value"]))/max(spec["tolerance"], 1e-12))
        return min(margins)

    def frame(t):
        pad = launch_initial_state(initial["site"], Time(t, format="gps"), heading_deg=90.)
        matrix = np.asarray(pad["body_to_gcrf"])
        omega = np.asarray(pad["omega_body_rad_s"])
        spin = matrix@omega
        r = np.asarray(pad["gcrf_position_m"])
        return r, pad["gcrf_velocity_m_s"], np.cross(spin, np.cross(spin, r)), matrix, omega, np.zeros(3)

    phases = [LaunchPhase("pad", events=(LaunchEvent("release", lambda t,y:t-epoch, "ascent", 1),),
                          constraint=LaunchGuide(frame, length=0.)),
              LaunchPhase("ascent", (gravity, aero, first), depletion_phase="separation"),
              LaunchPhase("separation", events=(LaunchEvent("stage_separation", lambda t,y:0., "upper_burn", reset=split),)),
              LaunchPhase("upper_burn", (gravity, aero, second), (LaunchEvent("orbit_insertion", insertion, "coast", 1),)),
              LaunchPhase("coast", (gravity, aero))]
    guarded = []
    for phase in phases:
        events = list(phase.events)
        for name in ("max_dynamic_pressure", "max_heating_rate", "max_acceleration", "max_angle_of_attack", "minimum_propellant_reserve"):
            spec = requirements[name]
            sign = 1. if spec["comparison"] == "at_most" else -1.
            field, limit = spec["evidence_field"], spec["value"]+sign*spec["tolerance"]
            events.append(LaunchEvent(f"limit:{phase.name}:{name}",
                lambda t,y,p=phase.name,f=field,limit=limit,s=sign:s*(limit-observe(p,t,y)[f]),
                None, -1, -99, 0., trigger_below=True))
        duration = requirements["maximum_mission_duration"]
        events.append(LaunchEvent(f"limit:{phase.name}:maximum_mission_duration",
            lambda t,y:epoch+duration["value"]+duration["tolerance"]-t, None, -1, -99, 0., trigger_below=True))
        guarded.append(replace(phase, events=tuple(events), propagation={
            **phase.propagation, "torque": lambda t,*args:actuator.value(t)[:3]}))
    # Resolve catalog-scale depletion and sampled-control fuel balance.
    settings = {"mu": EARTH_MU, "rtol": 1e-12, "atol": [1e-4]*3+[1e-5]*3+[1e-9]*4+[1e-8]*3+[1e-9], "max_step": .25}
    settings.update(solver or {})
    result = propagate_launch(state, epoch+np.arange(0., 651., 5.), guarded, initial_phase="pad",
                              ground_condition=launch_ground_clearance(0.), condition_tolerance=1e-5,
                              control=control, **settings)
    # Replay accepted commands with the existing actuator, never query its live state backward.
    actuator = copy(actuator)
    actuator._time = None
    actuator._value = actuator.initial.copy()
    actuator.command(epoch, actuator.initial)
    commands = [command for entry in control.history for command in entry["activated"]]
    command_index = 0
    history = []
    # shortcut: retained peaks use accepted boundaries; refine sampling for brief load spikes.
    maximums = {requirements[name]["evidence_field"]:0. for name in
                ("max_dynamic_pressure", "max_heating_rate", "max_acceleration", "max_angle_of_attack")}
    samples = [(phase, t, np.r_[arc.r[index], arc.v[index], arc.q[index], arc.omega[index], arc.mass[index]])
               for phase, arc in result.arcs for index, t in enumerate(arc.t)]
    samples.append((result.phase, result.state.t, _state_vector(result.state)))
    for phase, t, y in samples:
        while command_index < len(commands) and commands[command_index]["time_gps_s"] <= t:
            command = commands[command_index]
            actuator.command(command["time_gps_s"], command["requested"])
            command_index += 1
        values = observe(phase, t, y)
        for field in maximums:
            maximums[field] = max(maximums[field], float(values[field]))
        history.append(dict(phase=phase, time_gps_s=t, position_gcrf_m=y[:3],
                            velocity_gcrf_m_s=y[3:6], q_body_to_gcrf=y[6:10], mass_kg=y[13], loads=values))
    orbit = result.state.to_orbit()
    truth_positions = {entry["time_gps_s"]:entry["position_gcrf_m"] for entry in history}
    navigation_error = max((np.linalg.norm(np.asarray(entry["estimate"]["r"])
                                         - truth_positions[entry["time_gps_s"]])
                           for entry in control.history), default=None)
    values = {**maximums, "events.elapsed_time_s":result.state.t-epoch,
              "constraints.final_propellant_reserve_kg":sum(t.propellant_mass for t in result.state.body.tanks),
              "navigation.max_position_error_m":navigation_error,
              "initial_state.closure.position_m":initial["closure"]["position_m"],
              "initial_state.closure.velocity_m_s":initial["closure"]["velocity_m_s"],
              "orbit.perigee_altitude_m":orbit.a*(1-orbit.e)-EARTH_RADIUS,
              "orbit.apogee_altitude_m":orbit.a*(1+orbit.e)-EARTH_RADIUS,
              "orbit.eccentricity":orbit.e, "orbit.inclination_deg":np.degrees(orbit.i)}
    evidence = {spec["evidence_field"]:dict(value=float(values[spec["evidence_field"]]), unit=spec["unit"])
                for spec in requirements.values() if spec["evidence_field"] in values and values[spec["evidence_field"]] is not None}
    report = check_launch_requirements(requirements, evidence)
    reached_orbit = any(event["event"] == "orbit_insertion" for event in result.events)
    if result.termination != "__duration" or not reached_orbit:
        report["status"] = "FAIL"
    return {**report, "scope":"FALCON 9 CATALOG PAD-TO-LEO EXAMPLE; SYNTHETIC MODEL ASSUMPTIONS",
            "vehicle":vehicle, "catalog_vehicle":catalog,
            "catalog_stage_inventory_mass_kg":sum(s["mass_empty"]+s["mass_propellant"] for s in catalog["stages"]),
            "catalog_unassigned_mass_kg":catalog["total_mass"]-sum(s["mass_empty"]+s["mass_propellant"] for s in catalog["stages"]),
            "requirements":requirements,
            "evidence":evidence, "initial_state":initial, "events":result.events, "history":history,
            "termination":result.termination, "orbit_insertion_reached":reached_orbit,
            "solver":settings, "control":control.history,
            "software":collect_software_baseline(), "resources":resources,
            "inputs":dict(stage_dry_mass_kg=[lower.mass, upper_stage.mass], payload_mass_kg=payload.mass,
                          initial_propellant_kg=[lower.tanks[0].propellant_mass, upper_stage.tanks[0].propellant_mass],
                          residual_propellant_kg=[lower.tanks[0].residual_mass, upper_stage.tanks[0].residual_mass],
                          thrust_n=[first.thrust, second.thrust], isp_s=[first.isp, second.isp],
                          inertia_per_stage_kg_m2=[lower.inertia, upper_stage.inertia],
                          payload_inertia_kg_m2=payload.inertia, reference_area_m2=aero.reference_area,
                          drag_coefficient=.3, weather=weather, weather_heights_m=heights,
                          weather_pressure_pa=101325.*np.exp(-heights/8500.),
                          weather_temperature_k=280., wind_enu_m_s=[5., 2., 0.],
                          heating_coefficient_si=1.83e-4, nose_radius_m=1.,
                          guidance_orbit_budget_fraction=.5,
                          pitch_duration_s=pitch_duration, pitch_max_deg=80., pitch_delay_s=8.,
                          sensor_period_s=control.sensor_period, control_period_s=control.control_period,
                          actuator_rate=actuator.rate, actuator_lag_s=actuator.lag,
                          actuator_lower=actuator.lower, actuator_upper=actuator.upper,
                          kp_per_inertia=1., kd_per_inertia_s=1.5),
            "limits":["Catalog masses/engines have no per-value citations or resolved vehicle version; no measured-flight validation",
                      "Catalog total minus stage inventory is retained as unassigned metadata; sample payload is explicitly 1000 kg",
                      "Lower engines use constant catalog sea-level thrust/Isp; upper uses constant vacuum thrust/Isp",
                      "Synthetic 0.1 percent residual inventory, 10 m2 drag area, I=10 m2 times dry mass, atmosphere and heating proxy",
                      "Fixed COM point propellant, isotropic inertia, spherical drag and ideal external attitude torques",
                      "Flight-limit roots are checked during integration; retained extrema use accepted control/output boundaries",
                      "Altitude targets use osculating two-body elements above EARTH_RADIUS; terrain uses WGS84"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON report, never overwritten")
    args = parser.parse_args(argv)
    report = run_reference()
    encoded = json.dumps(report, indent=2, allow_nan=False, default=lambda value:value.tolist())+"\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
