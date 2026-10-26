"""Curated physics topics for interactive SSATK explorers."""

__all__ = ["TOPICS"]


TOPICS = [
    {
        "id": "integrators",
        "title": "Integrators",
        "kicker": "Choose how the state advances",
        "summary": "Explore adaptive DOP853, fixed-step Runge–Kutta (RK4), and kick–drift–kick leapfrog propagation. Extend the state to coupled attitude dynamics or a state-transition matrix.",
        "chain": [
            "Initial state and time grid",
            "Force and torque callbacks",
            "Numerical integration",
            "Position, velocity, and attitude history",
        ],
        "apis": [
            "ssatk.propagators_orbit.propagate_orbit_state",
            "ssatk.propagators_orbit.propagate_orbit_state_with_stm",
            "ssatk.propagators_orbit.rk4",
            "ssatk.propagators_orbit.leapfrog",
            "ssatk.propagators_6dof.propagate_6dof",
            "ssatk.propagators_6dof.propagate_6dof_high_accuracy",
        ],
        "limits": "Adaptive propagation uses solve_ivp and defaults to DOP853; RK4 and leapfrog expose fixed-step behavior. Leapfrog requires a strictly increasing uniform time grid and its symplectic property applies only to conservative position-only forces.",
        "demos": [
            "orbital_mechanics/demo_leapfrog",
            "orbital_mechanics/demo_compare_models",
            "six_dof/demo_6dof_gravity_gradient",
        ],
    },
    {
        "id": "gravity",
        "title": "Gravity",
        "kicker": "Build the force model",
        "summary": "Start with point-mass gravity, add EGM96 zonal terms J2–J8 and Sun/Moon perturbations, or use SSAPy's EGM2008 spherical-harmonic adapter with selectable degree and order.",
        "chain": [
            "Position and epoch",
            "Central or field gravity",
            "Third-body perturbations",
            "Total acceleration",
        ],
        "apis": [
            "ssatk.accelerations_orbit.accel_point_earth",
            "ssatk.accelerations_orbit.accel_earth_harmonics",
            "ssatk.accelerations_orbit.accel_J2",
            "ssatk.accelerations_orbit.accel_point_moon",
            "ssatk.accelerations_orbit.accel_point_sun",
            "ssatk.accelerations_6dof.make_ssapy_earth_harmonics",
            "ssatk.accelerations_6dof.make_ssapy_third_body",
            "ssatk.accelerations_6dof.make_ssapy_perturbation_acceleration",
        ],
        "limits": "The zonal helpers assume an Earth-rotation-axis z coordinate without precession, nutation or polar motion. accel_earth_harmonics includes central gravity; the SSAPy perturbation stack omits it. Avoid counting central gravity twice when composing callbacks with a propagator's built-in mu term.",
        "demos": [
            "orbital_mechanics/demo_compare_models",
            "orbital_mechanics/demo_groundtrack_accuracy",
            "six_dof/demo_6dof_gravity_gradient",
            "orbital_mechanics/demo_leapfrog",
        ],
    },
    {
        "id": "maneuvers",
        "title": "Maneuvers",
        "kicker": "Design a transfer",
        "summary": "Evaluate analytic, Lambert, continuous-thrust, and constrained transfer designs as states, legs, and burns.",
        "chain": [
            "Boundary states and timing",
            "Transfer or burn model",
            "Delta-v and constraints",
            "Sampled trajectory",
        ],
        "apis": [
            "ssatk.orbital_mechanics.transfer_hohmann",
            "ssatk.orbital_mechanics.transfer_bielliptic",
            "ssatk.orbital_mechanics.transfer_ssapy",
            "ssatk.orbital_mechanics.transfer_optimal",
            "ssatk.orbital_mechanics.transfer_velocity_continuous",
            "ssatk.orbital_mechanics.transfer_inclination_continuous",
            "ssatk.orbital_mechanics.ellipse_inject",
        ],
        "limits": "Hohmann and bi-elliptic transfers are analytic planning cases; transfer_bielliptic is the coplanar circular three-impulse case and does not solve phasing. Transfer planning does not itself perform finite-thrust propagation.",
        "demos": [
            "orbital_mechanics/demo_orbital_maneuvers",
            "orbital_mechanics/demo_structured_transfer_api",
            "orbital_mechanics/demo_station_keeping",
        ],
    },
    {
        "id": "burns",
        "title": "Burns and thrust",
        "kicker": "Apply propulsion over time",
        "summary": "Turn thrust profiles, engine presets, or impulses into acceleration, mass flow, and attitude-coupled spacecraft dynamics.",
        "chain": [
            "Thrust magnitude and direction",
            "Frame conversion",
            "Acceleration and mass flow",
            "Propagated maneuver",
        ],
        "apis": [
            "ssatk.accelerations_6dof.SpacecraftManeuverAccel",
            "ssatk.accelerations_6dof.thrust_profile_constant",
            "ssatk.accelerations_6dof.thrust_profile_trapezoid",
            "ssatk.accelerations_6dof.ThrustCurve",
            "ssatk.accelerations_6dof.make_finite_burn_acceleration",
            "ssatk.accelerations_6dof.integrated_thrust_impulse",
            "ssatk.propagators_6dof.ImpulseManeuver",
            "ssatk.engines.make_thruster_acceleration",
        ],
        "limits": "Finite-burn models accept scalar profiles or vectors in supported maneuver frames and can provide mass flow when Isp is set. ImpulseManeuver applies an instantaneous velocity change at a segment start; it is distinct from finite-thrust integration.",
        "demos": [
            "six_dof/demo_6dof_finite_burn",
            "six_dof/demo_6dof_attitude_control",
            "six_dof/demo_6dof_srp_facets",
        ],
    },
]
