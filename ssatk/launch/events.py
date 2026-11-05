"""Finite launch phase graphs and conservative instantaneous separation."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ssatk.coordinates.attitude import quaternion_from_matrix, rotate_vector
from ssatk.coordinates._earth_rotation import gcrs_to_itrs_matrix
from ssatk.environment import _itrf_to_geodetic
from ssatk.propagators_6dof import Spacecraft
from ssatk.propagators_6dof.sixdof import _tank_name_for_models, propellant_empty_event

__all__ = ["LaunchEvent", "LaunchPhase", "LaunchReset", "LaunchResult",
           "LaunchGuide", "launch_ground_clearance", "propagate_launch", "separate_launch_body"]


def launch_ground_clearance(terrain_height, *, eop=None, allow_predicted=False):
    """Signed WGS84 clearance (m) using strict packaged EOP and explicit terrain.

    terrain_height is ellipsoidal metres or a callable(latitude_deg,
    longitude_deg). No sea-level or terrain datum is inferred.
    """
    if not callable(terrain_height) and not np.isfinite(terrain_height):
        raise ValueError("terrain height must be finite or a callable")

    def clearance(t, y):
        rotation = gcrs_to_itrs_matrix(t, eop=eop, allow_predicted=allow_predicted)
        longitude, latitude, height = _itrf_to_geodetic(rotation @ y[:3])
        terrain = terrain_height(latitude, longitude) if callable(terrain_height) else terrain_height
        if not np.isfinite(terrain):
            raise ValueError("terrain model returned a nonfinite ellipsoidal height")
        return float(height-terrain)

    return clearance


@dataclass(frozen=True)
class LaunchEvent:
    """One-shot root: condition(t, y), direction and absolute condition tolerance.

    State layout is r, v, q, omega, optional mass, optional wheel momentum.
    Smaller priority fires first; names break ties. A None target terminates.
    Resets receive the accepted Spacecraft, never a solver trial state.
    Conditions must be side-effect free during trial root evaluations.
    Use distinct named phases/events for each restart in a finite mission.
    """

    name: str
    condition: object
    next_phase: str | None
    direction: float = 0.0
    priority: int = 0
    tolerance: float = 1e-9
    reset: object = None

    def __post_init__(self):
        if not self.name or not callable(self.condition):
            raise ValueError("events require a name and callable condition")
        if self.direction not in (-1, 0, 1) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("event direction must be -1, 0 or 1; tolerance finite and nonnegative")
        if not isinstance(self.priority, int) or self.reset is not None and not callable(self.reset):
            raise ValueError("priority must be an integer and reset callable")


@dataclass(frozen=True)
class LaunchPhase:
    """Continuous mode with explicit force models and outgoing transitions.

    Propellant depletion precedes user events (priority -100); ground contact
    precedes depletion (-200). Set depletion_phase to a coast/staging mode;
    None terminates. A guide constrains pad/rail motion, including reactions.
    momentum_transport supplies B in GCRF kg m/s (vector or callable(state)),
    so material momentum is M*v-B. Supply each mode's one-sided value even
    at depletion; geometry alone does not define exhaust/fluid transport.
    """

    name: str
    models: tuple = ()
    events: tuple = ()
    depletion_phase: str | None = None
    constraint: object = None
    propagation: dict = field(default_factory=dict)
    momentum_transport: object = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LaunchReset:
    """Accepted reset with retained state, detached children and external impulses.

    Angular impulse is GCRF about the common pre-event center of mass.
    Internal separation impulses sum to zero and need no external entry.
    child_momentum_transports supplies detached children's B in GCRF kg m/s;
    omitted values mean zero. Retained transport comes from the target phase.
    """

    state: Spacecraft
    children: tuple = ()
    impulse_gcrf: object = (0.0, 0.0, 0.0)
    angular_impulse_gcrf: object = (0.0, 0.0, 0.0)
    child_momentum_transports: tuple = ()


@dataclass(frozen=True)
class LaunchResult:
    """Per-mode trajectories, exact final state and ordered event evidence."""

    state: Spacecraft
    phase: str
    arcs: tuple
    events: tuple
    termination: str


def _state_vector(state):
    values = [state.r, state.v, state.q, state.omega]
    if state.mass is not None:
        values.append([state.mass])
    if state.wheel_momentum is not None:
        values.append(state.wheel_momentum)
    return np.concatenate(values)


def _check_state(state):
    if not isinstance(state, Spacecraft) or not np.isfinite(_state_vector(state)).all():
        raise ValueError("launch states must be finite Spacecraft objects")
    if state.body is None or state.mass is None:
        raise ValueError("launch requires explicit body mass properties")
    if state.inertia is None or not np.allclose(state.inertia, state.body.current_inertia, rtol=0, atol=1e-8):
        raise ValueError("state inertia disagrees with component mass properties")
    if state.wheel_momentum is not None and len(state.wheel_momentum) != len(state.body.reaction_wheels):
        raise ValueError("launch wheel momentum requires matching physical wheel definitions")
    if abs(state.mass - state.body.current_mass) > 1e-8:
        raise ValueError("state mass disagrees with component inventories")
    if any(t.propellant_mass < t.residual_mass for t in state.body.tanks):
        raise ValueError("propellant is below a declared residual")


def _transport(value, state):
    result = np.asarray(value(state) if callable(value) else value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("momentum transport must be a finite GCRF 3-vector in kg m/s")
    return result.copy()


def _snapshot(state, transport):
    return {"time_gps_s": state.t, "position_gcrf_m": state.r.copy(),
            "velocity_gcrf_m_s": state.v.copy(), "q_body_to_gcrf": state.q.copy(),
            "momentum_transport_gcrf_kg_m_s": transport.copy(),
            "material_momentum_gcrf_kg_m_s": state.mass*state.v-transport,
            "omega_body_rad_s": state.omega.copy(), "mass_kg": state.mass,
            "wheel_momentum_kg_m2_s": None if state.wheel_momentum is None else state.wheel_momentum.copy(),
            "configuration": state.body.name,
            "center_of_mass_body_m": state.body.current_center_of_mass.copy(),
            "inertia_center_body_kg_m2": state.body.current_inertia.copy(),
            "tanks": [{"name": tank.name, "propellant_kg": tank.propellant_mass,
                       "residual_kg": tank.residual_mass} for tank in state.body.tanks]}


def _momentum(state, origin, velocity, transport):
    linear = state.mass * state.v - transport
    spin = state.body.current_inertia @ state.omega
    if state.wheel_momentum is not None:
        spin = spin + sum((wheel.axis_body * h for wheel, h in
                           zip(state.body.reaction_wheels, state.wheel_momentum)), np.zeros(3))
    angular = rotate_vector(state.q, spin) + np.cross(state.r - origin, linear - state.mass*velocity)
    return linear, angular


def _conservation(before, reset, tolerances, before_transport=(0., 0., 0.), after_transport=(0., 0., 0.)):
    states = (reset.state,) + tuple(reset.children)
    for state in states:
        _check_state(state)
        if state.t != before.t:
            raise ValueError("reset must preserve the exact event epoch")
    before_tanks = {t.name: (t.propellant_mass, t.residual_mass) for t in before.body.tanks}
    after_tanks = [t for state in states for t in state.body.tanks]
    if (len({t.name for t in after_tanks}) != len(after_tanks)
            or {t.name for t in after_tanks} != set(before_tanks)
            or any(np.max(np.abs(np.array([t.propellant_mass, t.residual_mass])-before_tanks[t.name]))
                   > tolerances["mass_kg"] for t in after_tanks)):
        raise ValueError("reset must partition current propellant and residual inventories")
    impulse = np.asarray(reset.impulse_gcrf, dtype=float)
    angular_impulse = np.asarray(reset.angular_impulse_gcrf, dtype=float)
    if impulse.shape != (3,) or angular_impulse.shape != (3,) or not np.isfinite([impulse, angular_impulse]).all():
        raise ValueError("reset impulses must be finite GCRF 3-vectors")
    child_transports = reset.child_momentum_transports or ((0., 0., 0.),)*len(reset.children)
    if len(child_transports) != len(reset.children):
        raise ValueError("supply one momentum transport per detached child")
    before_transport = _transport(before_transport, before)
    transports = [_transport(value, state) for value, state in
                  zip((after_transport, *child_transports), states)]
    pre_p, pre_h = _momentum(before, before.r, before.v, before_transport)
    totals = [_momentum(state, before.r, before.v, transport) for state, transport in zip(states, transports)]
    errors = {"mass_kg": sum(s.mass for s in states) - before.mass,
              "linear_momentum_kg_m_s": sum((p for p, _ in totals), np.zeros(3)) - pre_p - impulse,
              "angular_momentum_kg_m2_s": sum((h for _, h in totals), np.zeros(3)) - pre_h - angular_impulse}
    if any(np.max(np.abs(value)) > tolerances[key] for key, value in errors.items()):
        raise ValueError(f"event reset violates conservation: {errors}")
    return {"residuals": errors, "tolerances": dict(tolerances), "status": "PASS",
            "external_impulse_gcrf_n_s": impulse, "external_angular_impulse_gcrf_n_m_s": angular_impulse}


def separate_launch_body(state, bodies, *, impulses_body=None, positions_body=None,
                         center_rates_body=None, parent_center_rate_body=(0., 0., 0.),
                         momentum_transport_body=(0., 0., 0.), child_momentum_transports_body=None,
                         tolerance=1e-8):
    """Split a rigid stack into child COM states; first child is retained.

    Bodies share the parent's axes and origin and must partition its current
    mass, first moment and inertia. Each impulse acts at positions_body (m)
    in that common frame; default application is the child's COM. Internal
    impulses must have zero resultant and moment. Velocities include omega
    cross COM offset and explicit centroid rates (m/s), zero after cutoff.
    Supply current depleted child inventories, never catalog initial masses.
    For ongoing flow, also supply parent/child B in common body kg m/s. The
    rate mapping and these transports must conserve material momentum M*v-B.
    """
    _check_state(state)
    bodies = tuple(bodies)
    if len(bodies) < 2 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("separation requires at least two bodies and a positive finite tolerance")
    n = len(bodies)
    masses = np.array([b.current_mass for b in bodies])
    centers = np.array([b.current_center_of_mass for b in bodies])
    c = state.body.current_center_of_mass
    offsets = centers - c
    inertia = sum((b.current_inertia + m * (np.dot(d, d)*np.eye(3) - np.outer(d, d))
                   for b, m, d in zip(bodies, masses, offsets)), np.zeros((3, 3)))
    if (abs(masses.sum() - state.mass) > tolerance
            or np.max(np.abs((masses[:, None]*offsets).sum(axis=0))) > tolerance
            or np.max(np.abs(inertia - state.body.current_inertia)) > tolerance):
        raise ValueError("children do not partition parent mass, first moment and inertia")
    parent_tanks = {t.name: t.propellant_mass for t in state.body.tanks}
    child_tanks = [t for b in bodies for t in b.tanks]
    if (len({t.name for t in child_tanks}) != len(child_tanks)
            or {t.name: t.propellant_mass for t in child_tanks} != parent_tanks):
        raise ValueError("children must partition every current tank exactly once")
    if state.wheel_momentum is not None and np.any(state.wheel_momentum):
        raise ValueError("separation of stored wheel momentum requires an explicit wheel reset")
    impulses = np.zeros((n, 3)) if impulses_body is None else np.asarray(impulses_body, dtype=float)
    points = centers if positions_body is None else np.asarray(positions_body, dtype=float)
    rates = np.zeros((n, 3)) if center_rates_body is None else np.asarray(center_rates_body, dtype=float)
    parent_rate = np.asarray(parent_center_rate_body, dtype=float)
    if (any(x.shape != (n, 3) for x in (impulses, points, rates)) or parent_rate.shape != (3,)
            or not all(np.isfinite(x).all() for x in (impulses, points, rates, parent_rate))):
        raise ValueError("child impulses, points and rates require finite (N, 3) arrays")
    if (np.max(np.abs(impulses.sum(axis=0))) > tolerance
            or np.max(np.abs(np.cross(points-c, impulses).sum(axis=0))) > tolerance):
        raise ValueError("internal impulses must have zero net force impulse and moment")
    children = []
    for body, mass, center, d, impulse, point, rate in zip(bodies, masses, centers, offsets, impulses, points, rates):
        children.append(Spacecraft(
            r=state.r + rotate_vector(state.q, d),
            v=state.v + rotate_vector(state.q, np.cross(state.omega, d) + rate - parent_rate + impulse/mass),
            q=state.q, omega=state.omega + np.linalg.solve(body.current_inertia, np.cross(point-center, impulse)),
            t=state.t, body=body, cd=state.cd, cr=state.cr))
    transports = (np.zeros((n, 3)) if child_momentum_transports_body is None
                  else np.asarray(child_momentum_transports_body, dtype=float))
    if transports.shape != (n, 3) or not np.isfinite(transports).all():
        raise ValueError("child momentum transports require a finite (N, 3) array")
    parent_transport = rotate_vector(state.q, _transport(momentum_transport_body, state))
    transports = tuple(rotate_vector(state.q, value) for value in transports)
    reset = LaunchReset(children[0], tuple(children[1:]), child_momentum_transports=transports[1:])
    _conservation(state, reset, {key: tolerance for key in
                  ("mass_kg", "linear_momentum_kg_m_s", "angular_momentum_kg_m2_s")},
                  parent_transport, transports[0])
    return reset


class LaunchGuide:
    """Ideal hold-down (length=0) or frictionless straight rail (length>0).

    frame(t) returns (origin, velocity, acceleration, body_to_gcrf,
    omega_body, omega_dot_body), in SI. Body +Z follows the rail. The frame
    may rotate; centripetal and Coriolis terms enter the reactions. Supply a
    bounded, verified pad-frame model. This models ideal support, not contact
    compliance, friction, vehicle flexure or a calibrated release mechanism.
    """

    def __init__(self, frame, *, length):
        if not callable(frame) or not np.isfinite(length) or length < 0:
            raise ValueError("guide requires a frame callable and nonnegative finite length")
        self.frame, self.length = frame, float(length)

    def _kinematics(self, t):
        r, v, a, rotation, omega, omega_dot = (np.asarray(x, dtype=float) for x in self.frame(t))
        if (any(x.shape != (3,) for x in (r, v, a, omega, omega_dot)) or rotation.shape != (3, 3)
                or not all(np.isfinite(x).all() for x in (r, v, a, rotation, omega, omega_dot))
                or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-10)
                or abs(np.linalg.det(rotation)-1) > 1e-10):
            raise ValueError("guide frame must supply finite kinematics and a proper rotation")
        axis = rotation[:, 2]
        rate = rotation @ np.cross(omega, [0., 0., 1.])
        second = rotation @ (np.cross(omega_dot, [0., 0., 1.])
                              + np.cross(omega, np.cross(omega, [0., 0., 1.])))
        return r, v, a, rotation, omega, omega_dot, axis, rate, second

    def coordinates(self, t, y):
        r, v, _, _, _, _, axis, rate, _ = self._kinematics(t)
        distance = float(axis @ (y[:3]-r))
        speed = float(axis @ (y[3:6]-v-rate*distance))
        return distance, speed

    def project(self, t, y):
        r, v, _, rotation, omega, _, axis, rate, _ = self._kinematics(t)
        distance, speed = self.coordinates(t, y)
        if not self.length:
            distance, speed = 0., 0.
        result = np.array(y, copy=True)
        result[:3], result[3:6] = r+axis*distance, v+rate*distance+axis*speed
        result[6:10], result[10:13] = quaternion_from_matrix(rotation), omega
        return result

    def derivative(self, t, y, unconstrained):
        _, _, a, _, _, omega_dot, axis, rate, second = self._kinematics(t)
        distance, speed = self.coordinates(t, y)
        kinematic = a + second*distance + 2*rate*speed
        along = float(axis @ (unconstrained[3:6]-kinematic)) if self.length else 0.
        if distance <= 0. and speed == 0.:
            along = max(0., along)
        result = np.array(unconstrained, copy=True)
        result[3:6] = kinematic + axis*along
        # The projected attitude follows the declared rotating guide exactly.
        result[10:13] = omega_dot
        return result

    def reactions(self, t, y, unconstrained, *, mass, inertia):
        """Support force (GCRF N) and torque (body N m) for a supplied free RHS."""
        corrected = self.derivative(t, self.project(t, y), unconstrained)
        return {"force_gcrf_n": mass*(corrected[3:6]-unconstrained[3:6]),
                "torque_body_n_m": np.asarray(inertia)@(corrected[10:13]-unconstrained[10:13])}

    def exit_condition(self, t, y):
        """Geometric rail exit root in metres; use direction +1."""
        if self.length == 0:
            raise ValueError("a hold-down needs an explicit release condition")
        return self.coordinates(t, y)[0] - self.length


def propagate_launch(state, times, phases, *, initial_phase, ground_condition,
                     condition_tolerance=1e-7, conservation_tolerances=None, control=None, **solver):
    """Propagate a finite event graph using Spacecraft.propagate.

    times are output GPS seconds, independent of solver steps. ground_condition
    returns signed clearance in metres (positive above the selected terrain).
    Airborne modes terminate on descending contact. Guides have a bottom stop
    and a terrain guard within condition_tolerance (m), avoiding a repeated
    zero root while resting on the pad. No orbital burnup guard enters here.
    All roots are terminal integration boundaries; simultaneous roots from the
    old mode fire in priority/name order, each reset seeing the previous post
    state. New-mode zero roots are then processed. Events fire at most once.
    Without a custom reset, a transport jump changes centroid velocity by
    (B_post-B_pre)/M, preserving material momentum. Custom maps must supply
    that physics themselves; the ledger checks it. Relative fluid angular
    momentum is assumed zero, as in the Step 5 axial reference.
    Optional LaunchControl inserts recurring accepted scheduling boundaries.
    Physical events (including immediate new-mode roots) precede updates;
    terminal events prevent further updates at their epoch.
    """
    _check_state(state)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] != state.t):
        raise ValueError("output times must increase from the exact initial epoch")
    if not callable(ground_condition) or not np.isfinite(condition_tolerance) or condition_tolerance <= 0:
        raise ValueError("ground condition and finite positive clearance tolerance required")
    phases = tuple(phases)
    modes = {p.name: p for p in phases}
    if len(modes) != len(phases) or initial_phase not in modes or any(not p.name for p in phases):
        raise ValueError("phase names must be unique and include initial_phase")
    names = [e.name for p in phases for e in p.events]
    if len(set(names)) != len(names) or any(n.startswith("__") for n in names):
        raise ValueError("event names must be globally unique and may not start with __")
    for p in phases:
        targets = [e.next_phase for e in p.events] + [p.depletion_phase]
        if any(target is not None and target not in modes for target in targets):
            raise ValueError("all transition targets must name a declared phase")
        if any(e.priority <= -100 for e in p.events):
            raise ValueError("user event priorities must exceed reserved safety priorities")
        if set(p.propagation) & {"times", "events", "dense_output", "stop_at_dry_mass", "constraint", "models", "mass0"}:
            raise ValueError("phase propagation cannot override event-runner arguments")
    if set(solver) & {"events", "dense_output", "stop_at_dry_mass", "constraint", "models", "mass0"}:
        raise ValueError("solver options cannot override event-runner arguments")
    tolerances = {"mass_kg": 1e-8, "linear_momentum_kg_m_s": 1e-6, "angular_momentum_kg_m2_s": 1e-6}
    if conservation_tolerances is not None:
        if set(conservation_tolerances) != set(tolerances):
            raise ValueError("supply all three conservation tolerances with their SI units")
        tolerances.update(conservation_tolerances)
    if any(not np.isfinite(x) or x < 0 for x in tolerances.values()):
        raise ValueError("conservation tolerances must be finite and nonnegative")
    current, fired, arcs, ledger = initial_phase, set(), [], []
    if control is not None:
        from .guidance import LaunchControl
        if not isinstance(control, LaunchControl):
            raise TypeError("control must be a LaunchControl")
        control.start(state.t)
    while True:
        mode = modes[current]
        y = _state_vector(state)
        clearance = float(ground_condition(state.t, y))
        if not np.isfinite(clearance) or clearance < -condition_tolerance:
            raise ValueError("initial/reset state is below ground or has invalid clearance")
        if mode.constraint is not None:
            projected = mode.constraint.project(state.t, y)
            if mode.constraint.length > 0 and mode.constraint.coordinates(state.t, y)[0] < -condition_tolerance:
                raise ValueError("initial/reset state is below the rail base")
            if (np.max(np.abs(projected[:3]-y[:3])) > 1e-6
                    or np.max(np.abs(projected[3:6]-y[3:6])) > 1e-6
                    or min(np.linalg.norm(projected[6:10]-y[6:10]),
                           np.linalg.norm(projected[6:10]+y[6:10])) > 1e-8
                    or np.max(np.abs(projected[10:13]-y[10:13])) > 1e-8):
                raise ValueError("initial/reset state is inconsistent with its pad or rail constraint")
        candidates = [e for e in mode.events if e.name not in fired]
        options = dict(solver)
        options.update(mode.propagation)
        options["mass0"] = state.mass
        force_models = (*mode.models, *(options.get(key) for key in
                        ("acceleration", "torque", "body_acceleration", "ntw_acceleration", "mass_flow_rate")))
        tank_name = _tank_name_for_models(*force_models)
        propulsive = (options.get("mass_flow_rate") is not None
                      or any(callable(getattr(m, "mass_flow_rate", None)) for m in force_models))
        if propulsive and not state.body.tanks:
            raise ValueError("launch mass flow requires explicit tank inventories")
        if propulsive and state.body.tanks:
            candidates.append(LaunchEvent(f"__depletion:{current}", propellant_empty_event(state, tank_name=tank_name),
                                          mode.depletion_phase, -1, -100, 1e-8))
        if mode.constraint is None:
            contact, contact_tolerance = ground_condition, condition_tolerance
        else:
            def contact(t, y):
                return ground_condition(t, y) + condition_tolerance/2
            contact_tolerance = 0.
            if mode.constraint.length > 0:
                guide = mode.constraint
                candidates.append(LaunchEvent("__rail_bottom", lambda t,y,guide=guide:guide.coordinates(t,y)[0],
                                               None, -1, -190, condition_tolerance))
        candidates.append(LaunchEvent("__ground", contact, None, -1, -200, contact_tolerance))
        candidates.append(LaunchEvent("__duration", lambda t, y: t-times[-1], None, 1, 10000,
                                      max(1e-9, 8*np.spacing(abs(times[-1])))))
        candidates = [e for e in candidates if e.name not in fired]
        if control is not None:
            scheduled = control.next_time
            candidates.append(LaunchEvent("__control", lambda t,y:t-scheduled, current, 1, 20000, 0.))

        def evaluate(event, t, y):
            value = float(event.condition(t, y))
            if not np.isfinite(value):
                raise ValueError(f"nonfinite event condition {event.name}")
            return value

        # From rest on the surface, let the solver distinguish liftoff from a fall.
        ready = [e for e in candidates if abs(evaluate(e, state.t, y)) <= e.tolerance
                 and (e.name not in ("__ground", "__rail_bottom") or evaluate(e, state.t, y) < 0.
                      or evaluate(e, state.t+1e-4, np.r_[y[:3]+y[3:6]*1e-4, y[3:]])
                         < evaluate(e, state.t, y))]
        if not ready:
            functions = []
            integration_events = [e for e in candidates if e.name != "__control"]
            for event in integration_events:
                def root(t, y, event=event):
                    return evaluate(event, t, y)
                root.terminal, root.direction = True, event.direction
                functions.append(root)
            end = min(times[-1], control.next_time) if control is not None else times[-1]
            requested = np.r_[state.t, times[(times > state.t) & (times < end)], end]
            trajectory = state.propagate(times=requested, models=mode.models,
                events=functions, stop_at_dry_mass=False, dense_output=True,
                constraint=mode.constraint, **options)
            arcs.append((current, trajectory))
            if trajectory.status == 0:
                # A clock boundary is an exact endpoint, not a numerical root.
                trajectory.t[-1] = end
            state = trajectory.spacecraft(body=state.body, tank_name=tank_name,
                inertia=None if state.body is not None else state.inertia, cd=state.cd, cr=state.cr)
            _check_state(state)
            y = _state_vector(state)
            roots = {e.name for e, values in zip(integration_events, trajectory.t_events or ())
                     if len(values) and abs(float(values[-1])-state.t) <= max(1e-8, 8*np.spacing(abs(state.t)))}
            previous_t = max(float(trajectory.t[0]), state.t-1e-4)
            previous_y = trajectory.solution(previous_t)
            ready = [e for e in candidates if e.name in roots or
                     (abs(evaluate(e, state.t, y)) <= e.tolerance
                      and (e.direction == 0 or e.direction*(evaluate(e, state.t, y)
                           - evaluate(e, previous_t, previous_y)) > 0))]
            if not ready:
                raise RuntimeError("launch integration ended without an accepted event")
        trigger_phase = current
        for event in sorted(ready, key=lambda e: (e.priority, e.name)):
            if event.name == "__control":
                continue
            pre, source = state, current
            value = evaluate(event, pre.t, y)
            target = current if event.next_phase is None else event.next_phase
            before_transport = _transport(modes[source].momentum_transport, pre)
            reset = LaunchReset(deepcopy(pre)) if event.reset is None else event.reset(deepcopy(pre))
            if isinstance(reset, Spacecraft):
                reset = LaunchReset(reset)
            if not isinstance(reset, LaunchReset):
                raise TypeError("event reset must return Spacecraft or LaunchReset")
            after_transport = _transport(modes[target].momentum_transport, reset.state)
            if event.reset is None:
                reset.state.v[:] += (after_transport-before_transport)/pre.mass
                after_transport = _transport(modes[target].momentum_transport, reset.state)
            conservation = _conservation(pre, reset, tolerances, before_transport, after_transport)
            for child in (reset.state,) + tuple(reset.children):
                clearance = float(ground_condition(child.t, _state_vector(child)))
                if not np.isfinite(clearance) or clearance < -condition_tolerance:
                    raise ValueError("event reset places a child below ground or outside terrain coverage")
            state = reset.state
            fired.add(event.name)
            current = target
            ledger.append({"order": len(ledger), "event": event.name, "time_gps_s": pre.t,
                           "condition_value": value, "condition_tolerance": event.tolerance,
                           "direction": event.direction, "priority": event.priority,
                           "trigger_phase": trigger_phase,
                           "pre_phase": source, "post_phase": current,
                           "pre_state": _snapshot(pre, before_transport), "post_state": _snapshot(state, after_transport),
                           "children": [_snapshot(s, _transport(b, s)) for s, b in zip(reset.children,
                               reset.child_momentum_transports or ((0., 0., 0.),)*len(reset.children))],
                           "conservation": conservation})
            if event.next_phase is None:
                return LaunchResult(state, current, tuple(arcs), tuple(ledger), event.name)
        if any(e.name != "__control" for e in ready):
            continue
        if control is not None and state.t == control.next_time:
            before, state = state, deepcopy(state)
            before_transport = _transport(modes[current].momentum_transport, before)
            control.update(deepcopy(state), current)
            after_transport = _transport(modes[current].momentum_transport, state)
            state.v[:] += (after_transport-before_transport)/state.mass
            after_transport = _transport(modes[current].momentum_transport, state)
            conservation = _conservation(before, LaunchReset(state), tolerances, before_transport, after_transport)
            ledger.append({"order": len(ledger), "event": "__control", "time_gps_s": state.t,
                           "condition_value": 0., "condition_tolerance": 0.,
                           "direction": 1, "priority": 20000, "trigger_phase": current,
                           "pre_phase": current, "post_phase": current,
                           "pre_state": _snapshot(before, before_transport),
                           "post_state": _snapshot(state, after_transport), "children": [],
                           "conservation": conservation})
