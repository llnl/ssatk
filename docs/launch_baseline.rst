Launch preflight baseline
==========================

``ssatk.launch.preflight`` checks the inputs a launch simulation needs before
an ascent is propagated. It does not propagate a trajectory or claim that a
preflight pass qualifies a mission.

A preflight run evaluates three gates against a JSON mission contract:

* **G0 -- contract completeness.** :func:`~ssatk.launch.preflight.validate_mission_contract`
  checks that the contract has exactly the expected top-level fields, a
  recognized launch site and vehicle, a valid phase/transition sequence, and
  that every requirement carries a numeric value, unit, tolerance, check name,
  evidence field and source. :data:`~ssatk.launch.preflight.LAUNCH_CONVENTIONS`
  fixes the frames, epoch scale and units the contract must declare.
* **G1 -- software and data baseline.** :func:`~ssatk.launch.preflight.collect_software_baseline`
  records installed package versions and, for repositories passed in by the
  caller, their Git revision and worktree cleanliness, without modifying
  either. :func:`~ssatk.launch.preflight.verify_data_resources` checks each
  declared resource's SHA-256 and byte count against the installed wheel's
  ``RECORD`` manifest and rejects an unresolved Git LFS pointer.
* **G2 -- launch-pad state.** :func:`~ssatk.launch.preflight.launch_site_state`
  builds the stationary pad position and the inertial velocity caused by
  Earth's rotation in GCRF. Both the epoch-coverage check and the transform
  use SSATK's packaged Earth-orientation table
  (:func:`ssatk.environment_eop.load_packaged_eop`) through SSATK's IAU
  2006/2000A Earth-rotation route, with no network access. An epoch that falls in the packaged table's predicted
  region raises :class:`~ssatk.launch.preflight.EOPCoverageError` unless the
  caller passes ``allow_predicted_eop=True`` (or the mission contract sets
  ``allow_predicted_eop``); an epoch past the end of the table always raises
  it, since no opt-in can produce data the table does not have. In
  :func:`~ssatk.launch.preflight.run_launch_preflight`, that error BLOCKS
  G2 with a readable reason rather than failing the run -- a preflight for a
  future launch legitimately needs predicted EOP, so the contract's opt-in
  controls whether that is acceptable. The contract's
  ``frame_position_closure`` and ``frame_velocity_closure`` requirements
  bound a self-consistency check: transforming the computed GCRF state back
  to ITRS and comparing it with the starting ITRF state. An internally
  consistent but wrong rotation (for example, a sign error that cancels
  going out and back) still closes, so this closure is not independent
  evidence of accuracy; ``tests/test_launch_preflight.py`` pairs
  ``launch_site_state`` with a closed-form WGS84 reference and with
  Astropy's own ITRS->GCRS implementation fed the same packaged table for
  that.

:func:`~ssatk.launch.preflight.run_launch_preflight` runs all three gates and
returns a JSON-serializable report; it does not raise, so a caller can inspect
every gate even when an earlier one fails.

Mission contract
-----------------

``demos/launch/leo_preflight_template.json`` is an unconfigured LEO template:
every mission-specific value is ``null`` or ``"UNSET"``. Replace those values
with the mission's actual site, epoch, vehicle, payload, targets, limits and
tolerances before using it; loading it unmodified reports gate G0 as
``BLOCKED``. The EOP resource entry is pre-filled because the Earth-orientation
data it names is a fixed dependency of the frame-closure checks, independent
of the mission. The template's ``allow_predicted_eop`` defaults to ``false``;
set it to ``true`` only when the mission epoch is known to need predicted
Earth-orientation data (for example, a launch date beyond the latest IERS
bulletin), and record why in the contract's risk entries.

``ssatk.launch.sites`` and ``ssatk.launch.vehicles`` hold the launch pad and
vehicle catalogs that ``launch_site`` and ``vehicle`` are checked against.

Step 2: mission contract
------------------------

Continue in the existing ``ssatk.launch.preflight`` workflow, using
``demos/launch/leo_preflight_template.json`` as the planning-default LEO contract.
The template is deliberately incomplete: neither the representative vehicle
catalog nor the Demo-2 pad comparison supplies authoritative payload, trajectory
targets or engineering limits. Replace every ``null`` and ``UNSET`` from approved
sources; do not copy the synthetic test fixture into a mission acceptance plan.
The prototype version-1 schema now also requires ``failure_behavior``,
``maximum_mission_duration`` and a ``comparison`` for every requirement.

The contract records site, height, UTC epoch, catalog vehicle and payload mass;
ordered phases and allowed transitions through deployment; software/revision pins;
resource provenance; frame conventions; and open risks. Every numerical
requirement includes ``value``, ``unit``, ``tolerance``, ``comparison``, ``check``,
``evidence_field`` and ``source``. Acceptance is inclusive:

* ``at_most``: measured value <= required value + tolerance (maximum limits).
* ``at_least``: measured value >= required value - tolerance (minimum reserve).
* ``within``: absolute measured-minus-required difference <= tolerance (targets).

``failure_behavior`` must map every termination condition to ``FAIL``, ``ERROR``
or ``BLOCKED``. These outcomes terminate the run and preserve its evidence, not
silently continue or substitute missing inputs. The mission-duration requirement
defines the numerical time limit. This is a contract for later dynamics, not an
implemented ascent or Monte Carlo termination engine.

The canonical preflight record is the JSON returned by ``run_launch_preflight``.
A valid contract is retained under ``contract`` with ``contract_sha256`` computed
over UTF-8 JSON using sorted keys and separators ``(',', ':')``. The evidence
bundle includes the run ID, UTC start/end times, scoped status, gate results,
software/revision/resource checks, initial state and per-requirement check status.
The embedded contract is the requirements table, units/frame definition and
acceptance configuration; retain it with the record rather than separate copies
that can disagree. Incomplete contracts retain the validation errors instead.

Statuses are ``NOT RUN`` (no evaluation), ``BLOCKED`` (missing prerequisites),
``ERROR`` (execution failure), ``FAIL`` (evaluated criteria not met), ``PASS``
(scoped criteria met), and ``WAIVED`` (requires explicit program authority;
preflight never assigns it automatically). ``G0`` checks contract completeness,
``G1`` checks environment/data, and ``G2`` requires schema and dependency/resource
resolution plus initial-state checks. Frame closure alone cannot pass ``G2``
when ``G1`` fails or is blocked. Dynamics requirements remain ``NOT RUN`` even
when the preflight checks pass.

Inspect an approved contract and retain a new evidence file::

   python3 -m demos.launch.demo_launch_preflight --contract path/to/approved_contract.json --output ../SSAPy-Data/.cache/launch_validation/run_record.json

Omit ``--contract`` to inspect the blocked LEO template. The command never
overwrites an existing output and exits with code 2 unless preflight passes.
Supply ``--repository llnl-ssapy=path/to/SSAPy``,
``--repository ssatk=path/to/ssatk`` and
``--repository ssa-data-core=path/to/SSAPy-Data`` to verify repository revisions.
Missing repositories stay blocked; a dirty development checkout cannot silently
substitute for the approved clean baseline. This demo alone does not establish
a clean-install gate.
Step 2 remains incomplete until the selected mission's authoritative inputs,
limits, tolerances and evidence satisfy the document's clean-environment gate.
Pause for review before resource qualification or new propagation.

Readable real-launch preflight check
-------------------------------------

``demos/launch/demo_real_launch_preflight.py`` compares ``launch_site_state``
for SpaceX Demo-2 with the independent SpacePy coordinate backend (the
``validation`` extra). NASA records LC-39A and 3:22 p.m. EDT on May 30, 2020:
https://www.nasa.gov/image-article/demo-2-launch-setting-forth-historic-journey/.
The input epoch is the published launch minute, not subsecond flight
telemetry.

Run the comparison test with its printed output visible::

   python3 -m pytest -q -s tests/test_launch_preflight.py::test_real_demo2_launch_preflight_matches_spacepy

Or generate an HTML report at an explicit output location outside the
repository::

   python3 -m demos.launch.demo_real_launch_preflight --output /tmp/demo2_preflight.html

The check compares independent WGS84 conversions of the same catalog pad
coordinates and the inertial speed from Earth's rotation. The position
tolerance is numerical, not a surveyed pad uncertainty. The speed allowance
comes from the floating-point Julian-day spacing in SpacePy's
central-difference reference. J2000/GCRF direction differences are reported
but not scored: SpacePy omits Earth-orientation-parameter corrections and
assumes UT1 equals UTC. This is not an ascent or telemetry match, and it does
not qualify the mission contract.

SSAPy-Data motor-burn simulation
--------------------------------

Run an actual variable-mass propagation using packaged propulsion inputs::

   python3 -m demos.launch.demo_packaged_motor_burn

This isolated Estes C5 motor example loads a public-domain RASP thrust curve
and its motor mass/provenance sidecar from the installed ``ssa-data-propulsion``
wheel. The demo skips cleanly, printing a message, when that optional
``[propulsion]`` extra is not installed. It pins the expected SHA-256 for the
curve CSV and its sidecar JSON (computed from the installed ``0.0.1`` files)
and verifies them, plus the collection index, against the installed wheel's
manifest before loading anything, using the existing Toolkit data resolver and
strict preflight resource verifier. The index is checked against the manifest
only, without a pinned contract hash, because ``ssa-data-propulsion`` ``0.0.2``
removes curves from it; the curve and sidecar files pinned here are unchanged
between those releases. No runtime downloads or Data source-checkout imports
occur.

An earlier version of this demo used the Estes C6 RASP curve, whose samples
integrate to 10.019 N*s against its catalog ``totImpulseNs`` of 8.82 N*s --
13.6% high, above the C-class 10.00 N*s limit -- and which ``0.0.2`` removes.
The Estes C5 RASP curve used now integrates to 7.791 N*s against a catalog
value of 7.79 N*s, well inside the independent-reference tolerance described
below.

The example reuses ``load_packaged_thrust_curve``, ``SpacecraftManeuverAccel``
and ``Spacecraft.propagate``. It infers a constant effective specific impulse
from the sampled curve's impulse and catalog propellant mass -- the demo
states that this is inferred, not measured -- then propagates mass depletion
and fixed-axis motion. The user-sourced curve's impulse differs from the
catalog value; both and their difference are reported without silently
rescaling the input. The output retains provenance, resource checks, sampled
states, mass balance and a rocket-equation comparison.
``tests/test_packaged_motor_burn.py`` recomputes the impulse independently by
trapezoidal integration over the raw packaged CSV samples (read through
``ssatk.data``, not through the demo's own result) and checks it against the
catalog ``totImpulseNs`` within 1%, alongside the mass-balance and
rocket-equation checks on the demo's variable-mass propagation.

There is no gravity, atmosphere, wind, payload, staging or rotational dynamics.
The identity inertia does not represent calibrated motor geometry. This verifies
a data-fed burn workflow, not the LEO mission contract, ascent or flight
qualification. ``qualification_status`` remains ``NOT RUN``. Retained JSON
output belongs in SSAPy-Data or an explicitly selected external evidence location,
not in the Toolkit source commit.

Step 3: time, frames and launch initialization
----------------------------------------------

:func:`~ssatk.launch.preflight.launch_initial_state` extends the existing
pad-state workflow with the attitude and angular rate needed by the Toolkit's
6-DoF propagator. Supply a catalog site, a scalar UTC string or Astropy ``Time``,
an explicit ``heading_deg`` in [0, 360), and WGS84 ellipsoidal ``height_m``.
The record retains UTC, TAI, TT, UT1 and GPS seconds. UT1 comes from the same
observed packaged EOP as the shared IAU 2006/2000A frame model, including across
UTC leap seconds; no network lookup or fallback Earth orientation is used.

Matrices map vectors by ``matrix @ vector``; their transposes are the inverse
rotations. The output contains ENU-to-ITRF, ITRF-to-GCRF, ENU-to-GCRF,
body-to-ENU and body-to-GCRF rotations. Body +Z is geodetic up, +X is the heading
clockwise from north, and +Y completes the right-handed frame. The active
``q_body_to_gcrf`` quaternion is ``[w, x, y, z]``. ``omega_body_rad_s`` is body
relative to GCRF, expressed in body axes, matching ``Spacecraft`` and
``sixdof_rhs``. An Earth-attached body's inertial angular rate and velocity are
nonzero even though its pad-relative rates vanish.

Pass the state directly to the existing Toolkit spacecraft object::

   from ssatk.propagators_6dof import Spacecraft

   spacecraft = Spacecraft(
       r=state["gcrf_position_m"], v=state["gcrf_velocity_m_s"],
       t=state["epoch_gps_s"], q=state["q_body_to_gcrf"],
       omega=state["omega_body_rad_s"],
   )
   orbit = spacecraft.to_orbit()

``to_orbit`` uses SSAPy's existing ``Orbit`` for the translational state; it
does not carry attitude or establish that the pad state is an achieved orbit.
Actual mass, inertia and release forces must be supplied before propagation.

Position, velocity and body angular rate share the existing Earth-rotation
model. Its derivative now evaluates the full orientation at GPS +/-5 seconds,
including precession, nutation, polar motion and interpolated UT1, and projects
the numerical rate onto the rotation's skew-symmetric tangent. Coverage and
observed-quality checks include both derivative neighbors; unavailable or
predicted neighbors block initialization. This is not a calibrated release or
support-force model. ``geocentric_radius_m`` is distinct from ``height_m``;
``terrain_height_m`` remains unset because no terrain source has been selected.

The existing explicit-EOP transforms accept matching ``(N, 3)`` velocities via
``v=``. Without ``v``, they retain their vector-only behavior. For moving points::

   position_itrf, velocity_itrf = gcrf_to_itrf_eop(position_gcrf, epoch, v=velocity_gcrf)
   position_enu = enu_to_itrf.T @ (position_itrf[0] - pad_position_itrf)
   velocity_enu = enu_to_itrf.T @ velocity_itrf[0]

``itrf_to_gcrf_eop`` is the inverse state transform. Both include the rotation
derivative. Merely rotating inertial velocity, or subtracting the pad's inertial
velocity without the rotation term away from its origin, is incorrect. Use
``body_to_enu.T`` to turn the local ENU displacement and velocity into body axes.

Run the initialization and shared-frame reference tests::

   python3 -m pytest -q tests/test_launch_initialization.py tests/test_launch_preflight.py tests/test_environment_eop.py tests/test_earth_rotation_route.py

These compare against WGS84 closed forms, independent Astropy frame and pad
states, raw packaged IERS-A leap-second data and ERFA, inverse rotations,
moving-point stationarity, and finite-difference 6-DoF quaternion kinematics.
They are initialization checks, not surveyed-pad or flight-telemetry validation.

Retain a three-sample pad-fixed demonstration record::

   python3 -m demos.launch.demo_launch_initialization --heading 90 --output ../SSAPy-Data/.cache/launch_validation/step3-initialization.json

The default site and epoch are the Demo-2 example; heading 90 is illustrative,
not an approved mission heading. The demo verifies the EOP wheel manifest and
payload, records the actual installed software, and never overwrites evidence.
Missing or unobserved EOP is ``BLOCKED``, a resource mismatch is ``FAIL``, and
invalid input is ``ERROR``. ``PASS`` is initialization-scoped;
``qualification_status`` remains ``NOT RUN``. Mission inputs, clean-install
qualification, ascent and Step-4 stage/tank accounting are not waived.

Step 4: stage and tank mass accounting
----------------------------------------------

Use existing ``SpacecraftBody``, ``Component`` and ``Tank`` objects for payload,
stages and detachable structures. :func:`~ssatk.launch.vehicles.assemble_stage_stack`
combines them in one explicit body frame and origin; each dry body's inertia
is about its declared center of mass. Coordinates are metres, masses kg and
inertias kg m squared. Tank names must be unique and nonempty across the stack.
Masses, coordinates and inertia inputs must be finite.
The assembled body's ``reference_area`` is inherited from the payload unchanged;
no composite projected area is inferred, so this helper is for mass-property
accounting rather than a complete aerodynamic vehicle model.
The representative vehicle catalog supplies no calibrated tank geometry or
inertia; supply those from engineering sources for an actual mission.
Multiple-shooting mass bounds and decoded node configurations use the same
dry-plus-residual floor; materially infeasible nodes are rejected.

``Tank.residual_mass`` is unusable inventory, retained in mass and inertia.
Named-tank updates consume only that tank; aggregate updates distribute depletion
in proportion to each tank's usable inventory. Both preserve declared residuals.
The existing thrust adapters and depletion events stop at the usable-inventory
floor, including when another stage still holds fuel. Updates construct new
objects, so rejected trial-state evaluations leave source inventories intact.

Without geometry, the existing tank model has fixed shell inertia and point
propellant at ``position_body``. For a constant-density cylindrical liquid,
supply ``cylinder_radius`` (m), ``propellant_density`` (kg/m cubed), and
``propellant_base_body`` (m), with ``propellant_axis_body`` pointing along the
fill direction. The liquid occupies a cylinder from the base to its current
height, with a flat free surface perpendicular to the axis. Its center moves
as it drains; its full inertia follows the cylinder volume, not an arbitrary
linear interpolation. Shell mass and inertia stay at ``position_body``.
This model excludes slosh, changing density and noncylindrical tanks.

Total mass, first moment and inertia use the existing body properties and the
parallel-axis theorem. ``Tank.inertia_mass_derivative(origin)`` gives the
analytic derivative at a fixed reference point; the shared body inertia-rate
adapter combines it with each tank's consumption rate. Moving-system angular
momentum flux is Step 5: the existing point-propellant flux model rejects
draining-cylinder variable-mass dynamics rather than silently claiming to
represent that physics. Cylinders in unselected tanks, or tanks holding only
unusable residuals during aggregate depletion, do not block point-tank burns.

To account for detachment, retain the current stage inventories, omit the
detached stage/structure from the configuration and rebuild. Compare retained
and detached mass, first moments and inertias about a common origin. Position,
velocity, impulses and event resets belong to Step 8; changing a configuration
does not perform those operations.

Run the independent volume-quadrature, finite-difference, continuity,
positive-inertia, detachment-conservation and residual-burn references::

   python3 -m pytest -q tests/test_launch_mass_accounting.py tests/test_propagators_6dof.py tests/test_integration_inertia_regressions.py tests/test_6dof_elapsed_time_integration.py

Retain a synthetic two-stage depletion and detachment record::

   python3 -m demos.launch.demo_launch_mass_accounting --output ../SSAPy-Data/.cache/launch_validation/step4-mass-accounting.json

The demo never overwrites evidence. Its geometry and inventories are explicitly
synthetic; they establish no engineering-data or mission qualification.

Step 5: changing-mass dynamics
------------------------------

The selected continuous control volume is the retained vehicle, with moments
about its instantaneous center of mass C. ``Spacecraft.r`` is C in GCRF and
``Spacecraft.v`` is its geometric time derivative; it is not automatically
total material momentum divided by mass. Body coordinates retain the fixed
origin used by the tank, component and nozzle geometry. Let R map body to GCRF,
c be C in body coordinates, omega be the body angular rate, and s be signed
mass rate (negative for exhaust). The existing thrust adapters return positive
outflow, and ``sixdof_rhs`` converts it to signed mass rate exactly once.

For a nozzle at body position n, its arm is rho = n - c. If u is exhaust
velocity relative to that body-fixed nozzle, its inertial velocity is

.. math::

   v_e = v_C + R(\omega \times \rho + u - \dot c).

For several ports, sum their signed mass rates s_i. Continuity of the first
mass moment gives the material momentum P, with transport B:

.. math::

   B = R\sum_i s_i\rho_i, \qquad P = M v_C - B,
   \qquad \dot P = F_{\rm external} + \sum_i s_i v_{e,i}.

Consequently, the general centroid acceleration is

.. math::

   M\dot v_C = F_{\rm external} + \sum_i s_i(v_{e,i}-v_C) + \dot B.

It is incorrect to equate M v_C with P when B is nonzero. Nozzle momentum
and pressure must each be counted once: the example's pressure-matched thrust
T already equals s u, so adding another s u would double the same force.
Pressure mismatch, when present, belongs in the declared net thrust convention.

About the moving C, substituting P = M v_C - B into the angular-momentum
transport theorem cancels the origin-velocity term. Net inward angular flux,
expressed in body coordinates, is

.. math::

   J = \sum_i s_i\rho_i \times (\omega\times\rho_i + u_i - \dot c).

If pressure and relative-exhaust thrust moments are already in ``torque``,
their contributions must be removed from J. A specified point nozzle does not
include the additional exit-area spin moment or swirl of a finite nozzle.
Geometry and tank consumption alone do not determine J.

For the chosen no-relative-angular-momentum fluid model, the existing shared
right-hand side integrates

.. math::

   \dot q = \tfrac12 q\otimes[0,\omega], \qquad \dot M = \sum_i s_i,

   I\dot\omega = \tau + J - \dot I\omega
                 - \omega\times(I\omega+h_w).

The reaction-wheel torque adapter includes the reaction to the wheel momentum
rate. If internal fluid motion has relative angular momentum h_f, the full
balance additionally needs -dot(h_f) - omega cross h_f; that physics is not
inferred by the tank model and is outside the example's validity. The documented
RocketPy equations-of-motion v1 are comparison material for these terms, not a
drop-in formula: they use their own reference-point and thrust conventions.
See `RocketPy equations of motion v1
<https://docs.rocketpy.org/en/latest/technical/equations_of_motion.html>`_.

``Tank.mass_first_moment_derivative`` returns d(first moment)/d(propellant mass)
in metres. For the constant-density cylinder this is the current free-surface
position, because removing an infinitesimal layer changes the first moment at
that position. ``SpacecraftBody.center_of_mass_mass_derivative(tank_name)``
computes

.. math::

   \frac{dc}{dM} = \frac{\sum_j f_j\,dQ_j/dm_j-c}{M},

where f_j is one for a selected tank or its fraction of aggregate usable
inventory. Multiplying by signed mass rate supplies dot(c), in body-frame m/s.
It returns zero when the selected usable inventory is exhausted. The existing
body inertia-rate adapter similarly differentiates the full inertia about C.
The moving-C contribution to dI cancels by the zero first moment about C;
holding C fixed while taking each tank derivative is therefore consistent.

The default point-tank adapter models inventory leaving at its local position
while co-rotating: J = dot(I) omega. This cancels the depletion term for that
specific local-shedding assumption, not for general rocket exhaust.
Cylindrical depletion still rejects unspecified exhaust flux. Supply explicit
``angular_momentum_flux`` only after deriving it for the chosen model.

Runnable axial reference
~~~~~~~~~~~~~~~~~~~~~~~~

``demos/launch/demo_launch_changing_mass.py`` reuses ``Tank``, ``SpacecraftBody``,
``SpacecraftManeuverAccel`` and ``Spacecraft.propagate`` for a synthetic steady
burn segment. Its one cylindrical tank and dry-body center lie on body +Z;
rotation is purely about +Z. The liquid has axial plug flow with a flat surface,
constant density and no relative angular momentum. A pressure-matched point
nozzle sits at the tank base on the same axis and has no swirl. The retained
liquid's axial velocity relative to the dry body is s/(density*pi*radius^2),
which supplies an independent mass-weighted material-momentum reference.
Its exhaust angular flux and
thrust moment about C are both zero. Thus I_z omega_z is conserved while I_z
decreases and spin increases; the adapter's local point-shedding cancellation
would give the wrong answer for this nozzle.

With constant s and nozzle position, B = s(n-c) along the unchanged inertial
axis, so dot(B) = -s dot(c). The centroid acceleration is therefore

.. math::

   \dot v_C = \frac{T}{M} - \frac{2s\dot c}{M}.

The demo adds this correction in an acceleration subclass of the existing
maneuver, with explicit zero ``angular_momentum_flux``. Initial centroid
velocity is B(0)/M(0), which initializes zero material momentum even though
the steadily draining geometric centroid moves. All output states are within
the continuous burn, above the unusable residual. Ignition, shutdown and their
velocity/reset consequences are Step 8; this example does not cross them.

For dry mass d, dry axial center z_d, tank base b and k = density*pi*radius^2,
the independent reference is

.. math::

   c(M) = \frac{M}{2k} + b-\frac{d}{k} + \frac{A}{M},
   \qquad A=d(z_d-b)+\frac{d^2}{2k}.

Integrating the rational centroid derivative gives closed-form logarithmic
position and velocity histories; I_z is linear in propellant mass, giving
closed-form spin and quaternion histories. ``tests/test_launch_changing_mass.py``
checks those histories, a sum of dry-body and plug-flow material momentum,
an independent quadrature of expelled linear momentum,
mass/impulse balance, finite-difference first moments and center rates, the
moving thrust lever arm, fixed-mass Euler motion and a nonzero point-nozzle
angular-flux solution. These constrain the stated mathematical models, not
flight accuracy or an unrestricted draining-tank launch model.

Run the reference checks and retain a new evidence record::

   python3 -m pytest -q tests/test_launch_changing_mass.py tests/test_launch_mass_accounting.py tests/test_integration_inertia_regressions.py tests/test_integration_wheel_regressions.py
   python3 -m demos.launch.demo_launch_changing_mass --output ../SSAPy-Data/.cache/launch_validation/step5-changing-mass.json

The JSON retains inputs, installed software, states, center rates, inertia,
exhaust velocity and retained momentum. It never overwrites an existing record.
``PASS`` is scoped to the axial reference; ``qualification_status`` stays
``NOT RUN``. Transverse fluid dynamics, finite-area nozzle transport, swirl,
slosh, propulsion operating maps, atmosphere, staging and mission qualification
require their own models and evidence.

Step 6: propulsion operating conditions
---------------------------------------

``SpacecraftManeuverAccel`` accepts an independent ``propellant_flow_rate`` in
kg/s (scalar or state callable), as an alternative to constant ``isp``. Both
share the maneuver's ``start``/``stop`` window and ``tank_name`` ownership.
Independent flow may continue when thrust is zero, for example during startup;
zero throttle shuts off both. A burn cannot specify both flow and Isp.

Reuse ``load_thrust_curve_csv`` for paired documented histories::

   from ssatk.accelerations_6dof.thrust import load_thrust_curve_csv

   thrust = load_thrust_curve_csv("motor.csv", bounds_error=True)
   flow = load_thrust_curve_csv(
       "motor.csv", thrust_column="mass_flow_kg_s", bounds_error=True)

The default columns are ``time_s`` and ``thrust_n``; choose the flow column
explicitly. The resulting scalar callable supplies kg/s when loaded from a
flow column; its ``total_impulse`` property then integrates kg, not N s.
Interpolation is piecewise linear. ``bounds_error=True`` is also supported by
``ThrustCurve`` and the packaged/digitized curve loaders, and rejects times
outside the table. Without it, the existing ``fill_value`` behavior remains.
Retain the input hash, source, units, test conditions and uncertainty with the
mission evidence. The CSV alone does not establish provenance or reconcile an
inconsistent thrust/flow pair. Use ``verify_data_resources`` and packaged
sidecar metadata for installed data, as in ``demo_packaged_motor_burn``.

``throttle`` is a scalar or state callable. It explicitly selects a linear
scaling law for reference thrust and flow, bounded by ``throttle_range``;
zero always means shutdown. This law requires justification by engine data.
For histories already measured under the commanded throttle, leave throttle
at one to avoid applying the command twice. No operating map or transient
response is inferred from an engine catalog name.

An optional fixed-nozzle correction requires ``reference_pressure`` and
``ambient_pressure`` in Pa, ``nozzle_exit_area`` in square metres, and an
explicit ``pressure_range`` in Pa. The reference pressure must lie in that
range; ambient pressure outside it, or a negative corrected thrust, raises an
error. For an operating engine,

.. math::

   F = F_{\mathrm{ref}} + (p_{\mathrm{ref}}-p_{\mathrm{ambient}}) A_e.

The reference thrust already includes nozzle momentum and the pressure term
at the reference pressure. Apply this correction once only; omit it for an
already ambient-adjusted curve. Flow is unchanged by this pressure correction.
When Isp is used, it refers to the reference-pressure thrust, so pressure
changes do not inadvertently alter tank depletion. Fixed nozzle geometry and
flow are assumptions, not a chamber-pressure or nozzle-separation model.
The maneuver applies force through the instantaneous center of mass; use the
existing body-mounted thruster model for a calibrated thrust moment.

Ignition, cutoff and restart epochs must be integration segment boundaries.
``propagate_spacecraft_segments`` already carries mass and state across those
boundaries. The maneuver's time window alone does not force an adaptive solver
to resolve a short burn. These prescribed boundaries do not implement the
event/reset maps or ignition/shutdown transport consequences of Step 8.

Run the documented synthetic nozzle reference::

   python3 -m demos.launch.demo_launch_propulsion --output path/to/new-evidence.json
   python3 -m pytest -q tests/test_launch_propulsion.py tests/test_thrust_profiles.py tests/test_launch_mass_accounting.py

The demo retains histories, pressure/throttle limits, tank inventory, solver
settings, software versions, impulse and rocket-equation residuals. It never
overwrites an evidence record. Tests independently check constant-flow motion,
the nozzle pressure formula, unchanged flow under pressure changes, triangular
CSV interpolation/integrals, and coast/burn/restart mass balance. These verify
the declared mathematical models; measured-engine calibration, mixture ratios,
pressure-dependent flow maps, ignition delays and mission qualification remain
outside this reference. ``qualification_status`` stays ``NOT RUN``.

Step 7: launch weather and aerodynamic loads
------------------------------------------------

``LaunchWeatherProfile`` accepts bounded arrays of WGS84 ellipsoidal height
(m), pressure (Pa), temperature (K), and ENU wind velocity (m/s). Wind components
are velocities toward east, north and up, not meteorological directions from
which the wind blows. Supply a source, GPS validity interval and latitude and
longitude coverage in degrees. Each profile is horizontally homogeneous only
inside that box; longitude bounds cannot wrap across the date line. Coverage
and increasing finite sample grids are checked before use. Unsupported heights,
times or locations raise an error; there is no endpoint hold or extrapolation.

The declared ideal gas uses specific gas constant R (J/kg/K), heat capacity
ratio gamma, and Sutherland parameters (mu_ref, T_ref, S), in Pa s, K and K:

.. math::

   \rho = \frac{p}{R T}, \qquad a = \sqrt{\gamma R T}, \qquad
   \mu = \mu_{\rm ref}\left(\frac{T}{T_{\rm ref}}\right)^{3/2}
         \frac{T_{\rm ref}+S}{T+S}.

Pressure, temperature and wind interpolate linearly; derived quantities are
computed afterward. The synthetic example declares dry air with R=287.05,
gamma=1.4 and Sutherland parameters (1.716e-5, 273.15, 110.4), using the
Sutherland relation described in the
`NASA Wind-US Sutherland law description
<https://www.grc.nasa.gov/www/winddocs/user/files.html>`_. Those constants are
explicit example inputs, not a globally calibrated composition model. Measured
density must be reconciled with the chosen p/T/composition model before creating
the profile; it is not silently interpolated as an incompatible fourth variable.

``LaunchAtmosphere(lower, upper=upper, transition=(h0, h1))`` provides an explicit
handoff to a profile sampled from the selected upper-atmosphere model. Both
profiles must cover the whole transition interval and use identical gas
parameters. With w=(h-h0)/(h1-h0), the overlap blends p, T and ENU wind as
(1-w)*lower+w*upper; density, sound speed and viscosity are then derived from
those primitives. This ensures value continuity at both endpoints, although
slopes need not be continuous. Unsupported composition changes require another
derived model. Without ``upper``, the lower profile's height limit stays strict.
The existing ``SpaceEnvironment.nrlmsise00_density`` remains available for upper
atmosphere density; that density alone is insufficient to define this handoff's
temperature, pressure, wind and composition.

``atmosphere.sample(time, r_gcrf)`` accepts absolute GPS seconds or Astropy Time.
It reuses the strict packaged-EOP IAU 2006/2000A rotation and its derivative,
plus the existing WGS84 and ENU helpers. If Q maps GCRF to ITRF, the total air
velocity in GCRF is Q.T*(wind_itrf-dot(Q)*r_gcrf). This includes Earth rotation
once; do not add an additional omega_Earth cross r. Predicted EOP needs explicit
``allow_predicted=True``; missing EOP or weather stays an error. Results include
pressure, temperature, density, sound speed, viscosity, wind, source identifiers
and transition weight. There are no weather downloads or catalog defaults.

``AerodynamicCoefficientTable`` reuses SciPy's multilinear regular-grid
interpolation with strict bounds. ``axes`` declares the evidenced dimensions
among ``mach``, ``alpha``, ``beta`` and ``deflection``; angles are radians.
Force and moment arrays have shape ``(*axis_sizes, 3)``, in that axis order,
in the table's aerodynamic axes (+X forward).
Singleton axes enforce their sole value. An omitted axis explicitly declares
independence of that variable for this model, so only omit it when justified
by the selected reference. Provide the dataset's source and body-axis/sign
conventions; no engine or vehicle catalog name establishes an aerodynamic map.
Read packaged inputs through ``ssatk.data`` and retain manifest checks, file
hashes, epoch, spatial coverage, units, conditions and uncertainties alongside
mission evidence, as in the existing preflight/data workflows.

``LaunchAerodynamics`` implements the shared ``SpacecraftAccel`` force/torque
interface, so pass it once in ``Spacecraft.propagate(models=[aero], ...)``.
It expects whole-vehicle aerodynamic-axis coefficients, not separate facet
coefficients or wind-axis drag/lift coefficients. Table +X is forward, and air-relative
velocity components define alpha=atan2(V_z,V_x) and beta=atan2(V_y,hypot(V_x,V_z)).
``coefficient_to_body`` is a fixed proper rotation mapping these table axes
to vehicle body axes, identity by default. For the earlier launch examples with
their rocket axis along body +Z, use ``[[0,0,-1],[0,1,0],[1,0,0]]``: table X
maps to body Z, table Y to body Y, and table Z to negative body X. This preserves
the existing geometry, thrust and pad attitude conventions. Angle queries use
velocity in the table frame, and force and dimensioned right-hand moment are
rotated into body before the reference-point shift.
At exactly zero airspeed, force/moment are zero without a coefficient query;
the otherwise undefined angles are reported as zero by convention.

Let n be the fixed coefficient/flow reference point, c the current retained
center of mass, R the body-to-GCRF quaternion rotation, and omega the body rate.
The adapter samples weather at r_C+R*(n-c), and uses point velocity
v_C+R*(omega cross (n-c)-dot(c)). ``center_of_mass_rate`` supplies dot(c) in
body m/s, either directly or through a state callable. Its default is an
explicit stationary-centroid assumption; draining tanks need their derived
rate (for one selected tank, signed mass rate times
``SpacecraftBody.center_of_mass_mass_derivative(tank_name)``). The adapter cannot
infer individual feed rates from aggregate mass alone.

For dynamic pressure q_dyn=rho*speed**2/2, area A, force coefficients C_F,
and three table-axis roll/pitch/yaw reference lengths L, let H map table to body.
Loads are

.. math::

   F_b = H\,(q_{\rm dyn} A C_F), \qquad
   M_n = H\,[q_{\rm dyn} A\,(L \mathbin{\odot} C_M)], \qquad
   \tau_C = M_n + (n-c)\times F_b.

The returned force and acceleration are in GCRF (N and m/s squared), and
torque is about the instantaneous C in body axes (N m). ``loads`` additionally
returns body force, reference-point moment, center position, air velocity,
Mach, alpha, beta, commanded deflection, dynamic pressure and Reynolds number
rho*speed*reynolds_length/mu. A separately declared ``reynolds_length`` is metres.
The three moment lengths and aerodynamic area remain explicit; assembled stage
geometry does not infer them. ``deflection`` is a prescribed radian scalar or
state callable; Step 9 actuator ``value(t)`` can supply actual deflection.

Run the synthetic evidence workflow and independent references::

   python3 -m demos.launch.demo_launch_aerodynamics --output path/to/new-evidence.json
   python3 -m pytest -q tests/test_launch_aerodynamics.py tests/test_launch_initialization.py tests/test_launch_mass_accounting.py tests/test_launch_changing_mass.py tests/test_launch_propulsion.py

The demo retains all weather/coefficient inputs, gas assumptions, reference
geometry, installed software, prescribed states, forces, moments and residuals;
it never overwrites an evidence file. Tests compare ideal-gas/Sutherland sample
calculations, overlap endpoint continuity, four-axis multiaffine polynomials,
symmetry limits, independent Astropy corotation/wind, retained-tank reference
point shifts, table-to-body axes, quadratic-drag motion and isotropic torque motion. These verify
the declared mathematical models. The demo's states are prescribed, not an
ascent integration or measured launch weather. ``qualification_status`` stays
``NOT RUN``. Rate damping, Reynolds-dependent coefficients, plume corrections,
rarefaction (including mean-free-path inputs), changing gas composition, slosh,
measured vehicle calibration and mission qualification require
their own evidenced models and domains.

Step 8: mission phases and event resets
---------------------------------------

``ssatk.launch.events`` runs a finite graph of :class:`~ssatk.launch.events.LaunchPhase`
modes through the existing ``Spacecraft.propagate`` solver. Each mode supplies
its force models, optional ideal pad/rail constraint and outgoing
:class:`~ssatk.launch.events.LaunchEvent` roots. Conditions use absolute GPS
seconds and the existing state layout: GCRF position and velocity, body-to-GCRF
quaternion, body angular rate, optional total mass and optional wheel momentum.
Targets must name declared phases; a ``None`` target terminates. Configure
distinct phases and globally unique event names for each ignition or restart.
Physical transitions are one-shot; optional Step 9 ``LaunchControl`` supplies
recurring sensor/control boundaries separately.

Roots carry direction, priority and an absolute tolerance in the condition's
units. Initial zero conditions fire immediately, once. During propagation,
direction selects crossings. All roots stop integration; reset callbacks run
only on copied accepted states, preserving the pre-state even if a callback
updates its arrays in place. Simultaneous old-mode roots fire in increasing priority,
then name order, before the new mode's immediate roots. Their conditions and
``trigger_phase`` retain the common pre-root state; each reset receives the
previous reset's post-state. Ground contact has priority -200, rail bottom
contact -190, tank depletion
-100, user events greater than -100, and duration termination 10000. User
terminal events end the run immediately. Output sampling does not select roots
or solver step size; set ``max_step`` small enough to resolve the declared
event functions, since a solver cannot discover multiple unresolved crossings
inside one step.

Mass-flow models require explicit tank inventories. The runner uses existing
named-tank or proportional depletion and residual-aware event helpers.
Depletion is recorded as ``__depletion:<phase>`` and enters the declared
``depletion_phase`` or terminates; it never silently treats unusable fuel as
available. Coast and restart modes replace force configurations and carry
current inventories. The final requested epoch is a recorded duration event.

Each phase declares ``momentum_transport``: a GCRF vector in kg m/s or a
side-effect-free ``callable(state)`` returning B from Step 5. Zero selects
the rigid/no-transport model; it must not be assumed for a draining moving
centroid. Supply the phase's one-sided boundary value, including the incoming
value at depletion, rather than an engine callback that has already switched
off at the residual floor. Continuous acceleration must satisfy the same
material balance; changing B only in the ledger does not correct a force model.

At an instantaneous same-mass transition without a custom reset,
``v_post = v_pre + (B_post-B_pre)/M`` preserves material momentum. Ignition,
cutoff and restart therefore reset geometric centroid velocity even without
a physical external impulse. A custom reset supplies its own consistent
velocity map and physical impulses. A terminal event keeps the current
transport convention; stopping integration alone does not model shutdown.
Fluid relative angular momentum is assumed zero, as in the Step 5 reference.

:class:`~ssatk.launch.events.LaunchGuide` models ideal hold-down (zero length)
or a frictionless straight rail (positive length), with body +Z along the rail.
Its caller-supplied frame includes origin position, velocity and acceleration,
rotation, angular rate and angular acceleration. It may rotate, and includes
centripetal and Coriolis terms. The existing accepted-state projector enforces
rail position, velocity and attitude; the constrained derivative computes
support reactions from the free force/torque equations. ``reactions`` reports
GCRF force in N and body torque in N m. Returning to the rail bottom is a
terminal ``__rail_bottom`` root that records the incoming state, without an
unmodeled velocity reset; contact restitution would need an explicit model.
At rest on the base, ideal support prevents downward acceleration. The
``exit_condition`` is the geometric distance-minus-length root in metres.
Declare an explicit release root for hold-down. Initial states must agree with
their constraints. Contact compliance, friction and calibrated release loads
are outside this ideal model.

``propagate_launch`` requires explicit signed terrain clearance, positive above
ground. :func:`~ssatk.launch.events.launch_ground_clearance` reuses the strict
packaged-EOP rotation and WGS84 conversion; supply ellipsoidal terrain height
in metres or a latitude/longitude terrain callable. No datum is inferred.
Airborne descending contact stops at zero clearance. Constrained terrain contact uses
half the declared clearance tolerance as a guard band to avoid repeated zero
roots while resting on the pad; penetration is bounded by that tolerance.
Initial and reset states below the tolerance or outside terrain coverage fail.
This launch policy is independent of SSAPy's orbital burnup policy: the
6-DoF launch path remains active through 100 km.

:func:`~ssatk.launch.events.separate_launch_body` receives current child bodies
in the common pre-separation body axes and origin. Their masses, first moments,
full inertias and named tank inventories must partition the parent. The first
child is retained; other child states remain in the event ledger. Child COM
positions shift by the rotated parent-to-child COM offset. Velocities include
the common rigid motion, ``omega cross offset``, and explicitly supplied child
minus parent centroid rates. Internal impulses at declared body-frame points
change child translation and spin and must sum to zero force impulse and
moment. Supply zero centroid rates after cutoff, or physically consistent
rates for a moving reference. During flow, also supply
``momentum_transport_body`` and ``child_momentum_transports_body`` in the
common body axes; the rate map must conserve M*v-B, not M*v. Retained-child B
comes from the target phase; detached-child B is carried by
``LaunchReset.child_momentum_transports`` in GCRF. A custom :class:`~ssatk.launch.events.LaunchReset`
may also declare external GCRF linear and angular impulses. Nonzero stored
wheel momentum needs an explicit child wheel reset.

Every accepted reset checks mass (kg), material linear momentum M*v-B (kg m/s) and angular
momentum (kg m squared/s) about the common pre-event COM, after accounting for
external impulses. User-supplied tolerances carry those explicit units. Failed
resets raise rather than continue. The ordered ledger retains condition, epoch,
pre/post modes, full states, B, material momentum, child states, tank accounting and conservation
residuals. Detached states are initial conditions, not propagated child flights.

Run the independent references and retain a synthetic evidence record::

   python3 -m pytest -q tests/test_launch_events.py
   python3 -m demos.launch.demo_launch_events --output ../SSAPy-Data/.cache/launch_validation/step8-events.json

The demo models two synthetic burns with ideal pad/rail constraints, burnout,
conservative staging, coast, restart, cutoff and a 100 km insertion boundary.
It compares velocity with the rocket equation and event times across two output
grids, retaining per-mode truth and the ledger. It reuses the Step 5 axial
acceleration correction for its offset point inventories and declares transport
on both sides of ignition and cutoff. Additional cylindrical references check
ignition, cutoff, restart and depletion against closed-form centroid velocities,
dry-body/plug-flow material momentum, independent exhaust quadrature and axial
spin conservation. Ongoing-flow separation checks centroid rates and material
momentum with nonzero parent and child B. The boundary is not an achieved
orbit. Gravity, atmosphere and Earth rotation are excluded from this closed-form
demo; the separate rotating-rail and independent Astropy/WGS84 references check
those kinematics and ground conversion. ``qualification_status`` stays
``NOT RUN``. Mission acceptance, measured vehicle calibration, solver convergence
qualification and ascent-to-orbit overlap verification remain separate work.

Step 9: sampled guidance, sensors and actuators
------------------------------------------------

Pass ``control=LaunchControl(...)`` to ``propagate_launch``. Integration splits
at exact clock endpoints, independent of output spacing. At a shared epoch all
physical events, including immediate new-mode transitions, settle before
sensing; a terminal event prevents further updates. Four callbacks separate
the physical state from measurements, estimates, references and commands:

* ``sensor(copied_truth)`` returns measurements. Use existing sensor models for
  declared bias, covariance, seeded noise and dropout. Only this callback sees
  truth; sensor location/orientation and validity remain explicit inputs.
* ``estimator(time, delivered_packets, previous_estimate)`` receives packets
  after ``sensor_latency`` seconds, retaining sample/delivery GPS epochs. Supply
  a prior estimate and a policy for invalid measurements; no truth fallback is
  provided. Covariance semantics belong to the estimator.
* ``guidance(time, estimate)`` returns the commanded reference.
* ``controller(time, estimate, reference)`` returns actuator channels. Reuse
  ``SpacecraftAttitudePD`` on the estimate, rather than attaching it directly
  as continuous truth-feedback torque.

Acquisition, delivery/estimation, guidance/control and command activation run in
that order. ``sensor_period``, ``control_period`` and both latencies are seconds.
``command_latency`` delays activation; commands are held between updates.
Clocks use initial GPS epoch plus integer sample index times period. Periods
must exceed eight floating-point spacings at the epoch. Use a fresh control,
actuator and seeded sensor for each run. Callbacks receive copies and return
finite JSON-recordable data; keep hidden truth out of closures/global state.

``LaunchActuator`` requires initial channels, lower/upper limits, positive rates
(channel units/s), nonnegative lags (s), units and a source. Activated commands
are clipped. Between commands its continuous response solves

.. math::

   \dot{x} = \operatorname{clip}((u-x)/\tau,-r,+r).

Positive lag gives a constant-rate ramp until error reaches ``r*tau``, then
exponential convergence; smaller errors are exponential from the start. Zero
lag gives a finite-rate ramp. Pure ``value(t)`` evaluations at/after the last
accepted command do not advance sensors, draw noise or mutate actuator memory.
Record commands and inputs for reconstruction; this is not a historical replay
API. Coupling, backlash and actuator reaction inertia are not inferred.

``SpacecraftManeuverAccel(position_body=...)`` extends the existing thrust,
pressure, throttle and named-tank flow path with body torque
``(position_body-current_COM) cross force_body`` in N m. The fixed application
point is in body metres; vector thrust callables can use actual gimbal values.
Omitted position retains thrust through the instantaneous COM. Off-center thrust
requires body mass properties. Do not add a second copy of the thrust torque.
The existing quaternion/rate equations evolve the plant; commands do not
override attitude. Flow and transport must still satisfy Step 5; scheduled
transport jumps retain the Step 8 material-momentum reset and conservation ledger.

``control.history`` keeps acquisitions, deliveries, estimates, references,
requested commands, delayed activations, bounded targets and actual values.
``LaunchResult`` retains truth arcs and ``__control`` pre/post states alongside
physical event evidence. Exceptions stop the run; completed updates remain in
the caller-owned history.

Run independent references and retain a new synthetic record::

   python3 -m pytest -q tests/test_launch_guidance.py tests/test_launch_events.py tests/test_launch_propulsion.py
   python3 -m demos.launch.demo_launch_guidance --output path/to/new-evidence.json

The demo uses ``CartesianSensor`` for orthogonal direction observations and a
body-rate gyro at the COM, removes an explicitly known bias, fits attitude with
``fit_star_attitude`` and holds the last measurement. Fixed-target guidance and
the existing quaternion PD drive a delayed, rate-limited pitch gimbal on an
off-center force actuator. This synthetic external-wrench reference removes
point inventory at the fixed COM: B=0, dry inertia is constant and relative fluid
angular momentum is zero. It does not infer an off-center nozzle's internal
fluid/transport model. The record keeps truth, measurements,
measurement covariance (not an estimated attitude covariance), estimates,
references, gimbal commands/response, timing, inventory, solver inputs and seed.
It tracks the known nominal pitch/rate equilibrium, obeys stops, matches
constant-Isp inventory and agrees on two output grids. Independent tests constrain
actuator closed forms, delayed angular impulse/quaternion motion, changing-COM
thrust arms, biased GPS position samples and event/transport ordering.
This demonstrates synthetic models and one nominal pitch case, not calibrated
performance, an attitude EKF, three-axis launch guidance or mission qualification.
Gravity, weather, Earth rotation, off-nominal dispersions, slosh/flexibility and
an acceptance mission remain separate work.

Step 10: numerical integration and convergence
------------------------------------------------

``propagate_launch`` reuses the existing coupled 6-DoF solver. DOP853 is the
initial choice for the smooth, nonstiff reference phases; phase roots, force
breakpoints and sampled-control endpoints split integration explicitly.
Output spacing selects retained samples, not solver steps. Fast or stiff
actuator, slosh or flexible-body models require their own solver comparison
and convergence evidence; the references below do not qualify those models.

Supply ``atol`` as a vector in the propagated state order: three positions in
metres, three velocities in m/s, four dimensionless quaternion components,
three body rates in rad/s, optional mass in kg, then optional wheel momenta in
kg m squared/s. A phase may override solver settings through its existing
``propagation`` mapping. The local error scale is ``atol + rtol*abs(y)``;
relative tolerance can dominate absolute tolerance at large GCRF positions.
Neither a local tolerance nor a smaller maximum step alone bounds mission
output error. Choose settings using independent references and refinement of
the actual quantities and event epochs required by the application.

Each standard :class:`~ssatk.propagators_6dof.sixdof.SixDOFTrajectory` retains
``integration_diagnostics``. The tuple preserves separate solve summaries
when depletion/coast or maneuver segments are combined. Summaries report
finite output states, raw quaternion norm error and the magnitude of output
normalization corrections. With ``dense_output=True`` they also report
internal step count, minimum/maximum step in seconds, and raw quaternion norm
errors at accepted endpoints, clipping the last interval to a terminal root.
The launch runner already requests dense output. Without it the internal-mesh
fields are ``None``. The variational and extended propagators do not produce
these summaries. These sampled checks do not bound state health between
samples, and guide projection is a physical constraint, not a normalization
error. A nonfinite output or inspected endpoint fails propagation.

Run the reference study and retain a new evidence file::

   python3 -m pytest -q tests/test_launch_integration.py
   python3 -m demos.launch.demo_launch_integration --output path/to/new-evidence.json

The study reuses the two-burn staging reference and the delayed sampled pitch
reference, with fresh controls and identical noise seeds for every run. It
first tightens state-unit tolerances at fixed maximum step, then halves that
step, and confirms with tighter tolerances and another halving. Each case
also checks coarse and fine output grids through the existing reference.
Comparisons cover common requested history epochs, each event's pre/post and
detached-child states at its root, event topology and event-time residuals.
The JSON retains
every run's settings, truth, control/measurement histories, conservation
ledger, reference residuals, function evaluations, internal step counts,
normalization diagnostics and unit-labelled comparison budgets. Changing
event topology or exceeding a budget fails the study.

Independent isotropic free-spin and linear-motion references constrain
DOP853, RK45 and Radau, including absolute GPS epochs and internal-mesh
independence from output sampling. A binary maximum/first step isolates
position summation roundoff in that closed-form experiment; it is not a
general recommendation for adaptive integration. The rocket equation and
constant-Isp inventory anchor the staging results. Convergence between two
SSATK runs alone does not establish physical accuracy.

The launch runner provides no reset-crossing STM. Linear sensitivities across
events require both the reset derivative and event-time sensitivity (saltation)
terms; a continuous STM must not be concatenated across these resets as if it
contained them. Numerical evidence is scoped to the declared synthetic
staging and nominal pitch cases. ``qualification_status`` remains ``NOT RUN``
for measured vehicle performance, the mission envelope and flight qualification.

Step 11: ascent to orbital analysis
------------------------------------

Use the accepted ``LaunchResult.state`` after cutoff and separation/reset,
not the nearest requested output sample. Its r/v are the current geometric COM
in GCRF metres and m/s; t is the stored GPS epoch. Complete the Step 8
material-momentum transition before handoff: a nonzero flow transport B cannot
be discarded without its centroid-velocity reset. A terminal event alone does
not switch off a force or change that convention.

``state.to_orbit(mu=mu)`` copies r/v/t into SSAPy without rotating,
interpolating or reconstructing from elements. By default ``Orbit.propkw``
carries current mass, area, CD and CR when present. Explicit ``propkw`` replaces
those defaults; include all required coast properties when overriding it.
Conversion requires finite state/epoch and positive finite mu. Keep the
spacecraft alongside the orbit: body, inventories, attitude, rates and wheel
momenta are not represented by Orbit. The launch runner preserves explicit
area and center of pressure through accepted phase advancement.

Select coast forces explicitly. Reuse the native objects wrapped by
``make_ssapy_perturbation_acceleration`` for identical-physics overlap::

   import ssapy

   insertion = launch_result.state
   orbit = insertion.to_orbit(mu=mu)
   # perturbations is the tuple used in the unpowered LaunchPhase.
   native_acceleration = ssapy.AccelSum([
       ssapy.AccelKepler(mu=mu), *(term.accel for term in perturbations)
   ])
   coast = ssapy.SciPyPropagator(native_acceleration, ode_kwargs={
       "method": "DOP853", "rtol": 1e-11, "atol": 1e-9, "max_step": 2.0
   })
   r, v = ssapy.rv(orbit, coast_times, propagator=coast)

Start ``coast_times`` at exactly ``insertion.t``. The 6-DoF runner adds central
gravity through mu; native SciPyPropagator needs the complete force including
exactly one AccelKepler. The perturbation builder and native harmonics omit
central gravity. Do not add the same perturbation directly and through a
preset. Use the same mu for gravity and elements. Match remaining mass, area,
coefficients, ephemerides, frames and forces before interpreting overlap error.
Native orbital propagation retains its low-altitude policy; surface ascent
must remain on the launch path.

This constant-property translational coast requires stopped thrust/mass flow
and force laws independent of evolving attitude/geometry. Retain 6-DoF for
further burns, deployments or facet loads. Orbit does not serialize a force
configuration; retain it explicitly with the insertion report. Native force
laws keep their own environment/frame approximations, which are not upgraded
to the launch atmosphere or Earth-rotation model by this handoff.

Run the two reference cases::

   python3 -m pytest -q tests/test_launch_orbit_handoff.py
   python3 -m demos.launch.demo_launch_orbit_handoff --output path/to/two-body.json
   python3 -m demos.launch.demo_launch_orbit_handoff --perturbed --output path/to/perturbed.json

The reference starts with a specified ascending upper-stage state, burns
fixed-COM point inventory and cuts off at an exact scheduled endpoint. Known
scheduled times should be integration boundaries: at large GPS epochs a
time-only root has a floating-point plateau even with elapsed-time integration.
State-triggered roots retain numerical event-time uncertainty. The reference
joins ascent/coast at their identical boundary sample and compares ten minutes
of native SSAPy coast with the launch runner. Both cases check constant-Isp
inventory and single-count force assembly. The unperturbed case also compares
with analytic Keplerian propagation. The perturbed case enables degree-2 zonal
gravity, Sun and Moon, Harris-Priester drag and cannonball solar/Earth radiation.

The report retains state, epoch, inventory, force settings, specific energy and
angular momentum, osculating semimajor axis, eccentricity, inclination and
equinoctial elements. Cartesian/nonsingular quantities avoid undefined
periapsis/node angles near circular or equatorial insertion. These are
instantaneous osculating elements relative to GCRF and the declared mu, not
mean elements, an SGP4 initialization or a TLE. Independent circular references
check two specified gravitational parameters. Energy/angular momentum
conservation is checked only for two-body coast, not drag or third-body motion.
Ground tracks reuse ``gcrf_to_llh`` with packaged EOP, IAU 2006/2000A and WGS84;
Astropy references share the IERS table and allow its omitted celestial-pole
offsets. The generated JSON retains the continuous history and event ledger.

These cases verify terminal-ascent handoff, not a surface-to-orbit guidance
solution, mission acceptance or measured vehicle/flight qualification.
``qualification_status`` stays ``NOT RUN``. Generated evidence remains outside
version control; persistent reference data belongs in SSAPy-Data.

Step 12: optional coupled slosh increment
-----------------------------------------

The optional-model step in the launch literature-review sequence calls for
one evidenced increment at a time. This increment adds reciprocal slosh
coupling to the existing ``propagate_6dof_extended`` and its variational
wrapper. The rigid launch baseline remains the default. Flexibility and
recovery are not enabled by adding a slosh mode: recovery deployment,
separated-body loads, and structural modes need their own evidence.

Select ``coupled_slosh=True`` with one or more ``SloshMode`` objects. Each is a
point mass on a straight body-fixed guide, with a linear spring and damper.
``natural_frequency`` (rad/s) and ``damping_ratio`` describe a **clamped bus**;
they are not the frequency and damping measured with a freely moving bus.
For modal mass m, stiffness is k=m*w**2 in N/m and damping is c=2*zeta*w*m in
N s/m. Modal displacement/rate are relative metres and metres per second.
``axis_body`` is normalized and ``lever_arm_body`` is the equilibrium point
relative to the rigid bus COM, in body metres.

``bus_mass`` and the constant positive-definite ``inertia`` describe the bus
**excluding** every participating modal mass. Nonparticipating liquid may
remain in the bus properties. Do not put the whole liquid mass in the bus
and then add it again as a slosh mass. Optional ``mass0`` must equal bus plus
modal masses. There is no mass flow in this model. Hinge/flexible modes,
state-dependent inertia and gravity-gradient torque are rejected for this
increment. Acceleration callbacks supply external force on the bus divided
by bus mass; torque is the external body torque about the bus COM. Central
gravity is sampled at the bus and applied uniformly to the assembly, so
common free fall does not excite the relative coordinates. Tidal effects
across the vehicle are excluded.

For body-axis guide direction e_i, equilibrium arm l_i, relative displacement
x_i and body rate omega, define

.. math::

   a_i = l_i + e_i x_i, \qquad
   J_i = [\mathbf{1}, -[a_i]_\times, 0, \ldots, e_i, \ldots],

   b_i = \omega\times(\omega\times a_i)
         + 2\omega\times e_i\dot{x}_i.

The generalized acceleration consists of bus translational acceleration
relative to common free fall (expressed in body axes), body angular
acceleration, and relative modal accelerations. Its symmetric mass matrix is

.. math::

   H = \operatorname{diag}(M\mathbf{1}, I, 0, \ldots)
       + \sum_i m_i J_i^T J_i.

The right-hand side contains external bus force, external bus torque minus
omega cross I*omega, and modal forces -k_i*x_i-c_i*xdot_i, minus
sum(m_i*J_i.T*b_i). Solving this system accounts for axial and transverse
guide reactions on both bus and masses. With no external force/torque and
mu=0, total linear/angular momentum are conserved. Mechanical energy decreases
at sum(c_i*xdot_i**2); energy plus that integrated heat is conserved. These
properties apply to the specified mechanical analogue, not arbitrary fluid
motion.

Every mode must supply a positive ``displacement_limit`` in metres from its
evidenced domain. Initial values and RHS evaluations outside that limit are
rejected; solver trial stages can also trigger this rejection. It is not a
localized impact/deployment event or a proof of continuous-time clearance.
Use the model only while fill level, acceleration regime, viscosity and
geometry support the fitted constant coefficients. A linear guide cannot
predict breaking waves, impacts, arbitrary free-surface geometry, large-angle
pendulum behavior, draining tanks or changing modes. Restarting a segmented
run requires carrying all modal coordinates/rates as well as the bus state;
the rigid launch phase runner does not do that automatically.

The returned ``trajectory.r/v`` describe the **bus COM**, not the whole
vehicle COM. For body-to-inertial rotation R and total mass Mt, recover

.. math::

   r_C = r_b + R\frac{\sum_i m_i a_i}{M_t}, \qquad
   v_C = v_b + R\frac{\sum_i m_i(\omega\times a_i+e_i\dot{x}_i)}{M_t}.

Do not pass the bus state straight to Step 11's orbital handoff. Retain the
extended propagation when internal motion affects the observables or loads;
an Orbit discards these internal states. The old default
``coupled_slosh=False`` keeps the prescribed-oscillator disturbance model for
existing callers. Its modes load the bus without responding to it and must
not be used as evidence of reciprocal slosh mechanics or conservation.

Run the reference and save a fresh evidence record::

   python3 -m demos.launch.demo_launch_slosh --output path/to/new-evidence.json
   python3 -m pytest -q tests/test_launch_slosh.py tests/test_6dof_extended.py

The reference generates synthetic observations from the independent
closed-form two-mass spring/damper solution. It fits modal mass, clamped
frequency and damping to **both bus and relative displacement**, then predicts
a separate initial displacement with a constant external force. Relative
motion alone determines only effective frequency and damping, leaving modal
mass ambiguous; the reference records this deficient sensitivity alongside
the full-observable singular values. This is local identifiability in a
specified synthetic experiment, not global uniqueness or an instrument
uncertainty assessment.

The report includes parameters, observations, held-out predictions, SI
budgets, rigid/coupled residuals, solver refinement, software versions and
domain limits. Numerical tests also compare independent Cartesian momentum
and energy accounting for rotating offset modes, a closed-form circular
free-fall case, and the analytic two-mass sensitivity. Calibration and
held-out flight measurements are **UNAVAILABLE** and ``qualification_status``
remains ``NOT RUN``. Keep the simpler rigid model for a flight application
until measured observables and its error budget justify the extra parameters.

Step 13: verification and independent comparisons
--------------------------------------------------------

Run the launch literature-review benchmark ladder from a source checkout::

   python3 -m demos.launch.demo_launch_verification --output ../SSAPy-Data/.cache/launch-verification/new-run

The output directory must be new. Each executed case retains its inputs,
solver settings, residuals and evidence in a separate JSON file. ``report.json``
records the software baseline, Python source hashes, packaged-resource checks,
evidence hashes and one status per case. Every evaluated metric has a value,
unit, tolerance and reference. Budgets belong to the named synthetic reference;
they do not supply missing mission requirements or measured uncertainties.

The ladder reuses the existing propulsion, changing-mass, atmosphere/load,
staging/control refinement, coupled-slosh and orbital-handoff references.
Basic mechanics adds constant inertial acceleration and spherical torque-free
spin, including tighter integration against the same analytic solution.
The pad frame comparison uses independent Astropy transforms with packaged
EOP. Mission-sequence refinement retains event topology, reset mass/material
momentum accounting and event-time errors separately from state errors.
Attitude errors include relative rotation angles in radians and treat opposite
quaternion signs as the same orientation. Handoff evidence retains GCRF and
radial/transverse/normal residual histories at the common coast epochs;
velocity residuals are vector components, not derivatives in a rotating frame.
The insertion record uses Cartesian and nonsingular/osculating elements.

The optional tool comparisons reuse ``scripts.external_validation_report``:
Orekit and GMAT for point-mass coast, GMAT for a constant inertial finite burn
with chemical propellant depletion, and Basilisk for a rigid body with a
specified wrench. The burn retains state, mass and cutoff-time residuals,
the generated GMAT script, and an independent constant-flow mass reference.
GMAT samples at absolute epoch endpoints to avoid accumulation of rounded
relative stops. No tool is installed automatically. A native Windows GMAT
installation can be selected with ``GMAT_ROOT`` pointing to the directory
containing ``bin/GmatConsole.exe``. Other platforms retain the existing
Podman route. ``GMAT_DE430_BSP`` explicitly selects a kernel when a matched
planetary comparison needs it; there is no machine-specific default. Temporary
Windows startup/output files preserve the installed GMAT configuration.
Orekit needs Java, ``javac``, Maven and its existing jar cache or ``OREKIT_JAR``.

``PASS`` means that case's available numerical criteria passed. ``FAIL`` means
an evaluated criterion failed; ``ERROR`` retains an execution failure and its
traceback. ``UNAVAILABLE`` names missing reference inputs or runtimes. Missing
NASA dropped/tumbling-body and corrected Atmospheric Case 17 histories,
matched atmospheric launch-package cases, external slosh histories,
and independent telemetry are explicit rows. They are never converted into
passes by successful internal tests. An unavailable external case has no
computed residual; its unit/tolerance cannot be inferred from another case.
The command exits with code 2 unless the entire reported ladder is ``PASS``;
the report remains available for review when evidence is missing.

The identical-physics cases isolate implementation/integration differences.
Native SSAPy handoff shares the force implementations with the launch wrapper,
so it does not independently validate those force laws. A **best-supported
model comparison** needs a preregistered common vehicle/case, independently
configured comparator, supported environmental inputs and practical output
budgets. That comparison is recorded separately as unavailable until those
inputs exist; turning on perturbations in the handoff is not a substitute.
Numerical refinement, physical model discrepancy and observational uncertainty
remain distinct. Step 14 measured calibration/held-out validation and flight
qualification remain unclaimed. This report does not establish a complete
pad-to-orbit guidance solution or rank simulation packages.

Step 14: measurement calibration and held-out validation
---------------------------------------------------------

``ssatk.launch.calibrate_launch_measurements`` fits named physical/sensor
parameters on designated calibration cases, freezes them, and evaluates
independent flights or nonoverlapping test segments. It reuses SciPy bounded
least squares; the prediction callback reuses the existing launch models and
propagators. It does not introduce another force model or infer calibrated
hardware from a vehicle catalog.

Prepare the split, sensor operators, uncertainty model and output requirements
before fitting. Supply cases as JSON-compatible mappings with these fields:

* ``id`` (unique across both splits), ``flight_id``, ``evidence`` (``measured``
  or ``synthetic``), ``observable``, and ``unit``. Each case is one scalar
  observable history; angles must already be consistently unwrapped.
* ``time_gps_s`` and ``observed``: matching finite arrays with increasing
  absolute GPS sample tags. ``covariance`` is a positive-definite matrix
  covering all samples, in squared observable units. Preserve instrument
  uncertainty and temporal correlation, including filtering. Propagate
  uncertain measured inputs into this covariance or explicitly estimate their
  identifiable effects; do not count the same uncertainty twice.
* ``sensor``: ``position_body_m`` relative to the model's body origin,
  ``frame``, ``filter``, ``time_alignment``, and ``support_gps_s``. The last
  field bounds the complete physical support, including filter history,
  clock correction and interpolation. Support from the same flight cannot
  overlap between calibration and validation. These descriptions are retained
  evidence; the supplied prediction callback implements them.
* ``inputs``: entries for ``thrust``, ``mass_properties``, ``atmosphere`` and
  ``telemetry``. Each supplied entry contains ``source``, lowercase ``sha256``
  and ``uncertainty`` information. Include test conditions, validity, units
  and alignment in that record. An omitted entry is explicitly unavailable;
  a source/hash alone does not establish that an input is measured. Resolve
  packaged data through ``ssatk.data`` and verify resource hashes with
  ``verify_data_resources`` before building a case. External files are caller
  inputs, never paths embedded in package code.
* ``model_source`` and ``independence``: identify the physical model and justify
  conditional independence between cases. Use an optional ``model`` mapping
  to retain initial conditions, solver settings and other callback inputs.
  Shared unestimated noise or calibration errors violate independence even
  when time spans do not overlap; combine correlated samples into one case
  or use a suitable joint observation model before using this workflow.
* ``requirements``: ``max_abs_residual`` and ``max_abs_bias`` in observable
  units, ``min_coverage`` in [0, 1], and ``source`` for the preregistered
  acceptance budget. Requirements apply to each case; choose them from the
  mission contract and measurement uncertainty, not from the fitted result.

``predict(parameters, case)`` returns predicted sensor observables, in the
declared units/frame/sample order. It receives a read-only parameter mapping
and a copy of the case; it must not derive predictions from ``observed``.
Build fresh stateful control/actuator objects for each run. Account for sensor
position, attitude, acquisition/filtering and clock alignment before returning
the history. For example, a body-mounted velocity sensor sees COM velocity
plus the rotated ``omega cross lever_arm`` contribution. Do not compare its
data directly to COM velocity. Invalid predictions during calibration raise
an error; evaluation errors on held-out cases are retained as ``ERROR`` rows.

With measured cases and the corresponding sensor callback already prepared::

   from ssatk.launch import calibrate_launch_measurements

   report = calibrate_launch_measurements(
       calibration_cases, validation_cases, predict,
       {"thrust_scale": 1.0, "velocity_bias_m_s": 0.0},
       parameter_units=["1", "m/s"], parameter_scale=[1.0, 0.1],
       parameter_bounds=([0.8, -0.5], [1.2, 0.5]),
       coverage_probability=0.95,
   )

Those example bounds/scales are illustrative inputs, not hardware limits.
Parameter names follow insertion order. Bounds are finite physical limits;
scales define the coordinates used for numerical derivatives and SVD rank.
The reported ``rank_rtol`` defaults to ``1e-8`` relative to the largest scaled
singular value. Inspect singular values, parameter correlations and uncertainty
against physical requirements; numerical full rank alone is not practical or
global identifiability. Resolve the prediction change under finite differences
with the integration tolerances, and check sensitivity to numerical refinement.
An unconverged, rank-deficient or active-bound fit has no formal covariance
and cannot produce a validation pass.

The report retains the frozen case inputs and their hashes, parameter guesses,
bounds/scales/units, fitted values, solver termination, rank, active bounds,
software versions and residual histories. Parameter covariance is local inverse
Fisher information and is not scaled down to the observed fit residuals.
For held-out cases, predictive covariance is ``R + J C_parameter J.T``.
Gaussian marginal intervals use ``coverage_probability``; the joint squared
Mahalanobis residual is checked against the corresponding chi-square quantile
with one degree of freedom per held-out sample. This joint check uses temporal
correlation and reports its statistic and limit. Mean signed bias, bias standard
error, RMS, maximum absolute residual and observed interval coverage are retained.
Coverage fraction is descriptive, not an independent-binomial confidence bound
or a demonstrated long-run coverage rate. All uncertainty statements remain
conditional on the local linear/Gaussian model and supplied covariance.

``calibration_status`` describes the fit's convergence/rank/bound checks;
each case separately reports its output-budget checks. ``validation_status``
retains failing and errored cases and is ``UNAVAILABLE`` without usable held-out
evidence. ``measurement_evidence_status`` requires measured cases and all four
input records. Overall ``PASS`` requires all those checks; synthetic cases
cannot turn unavailable flight evidence into a pass. ``qualification_status``
always remains ``NOT RUN``. Assess attitude, lateral motion and loads with
their own measured observables and budgets; an apogee match does not cover them.

Run the numerical demonstration::

   python3 -m demos.launch.demo_launch_calibration --output new-calibration.json
   python3 -m pytest tests/test_launch_calibration.py -q

The example uses the existing changing-mass 6-DoF propagator, a rotating offset
velocity sensor, an explicit clock correction and a two-sample filter. Its
reference observations come independently from the constant-flow rocket
equation and a closed-form rotation. It estimates thrust and sensor bias on
one synthetic flight, then retains a nominal holdout and a deliberately biased
failed holdout. Its JSON output is never overwritten, and exit code 2 reports
the expected failed/incomplete validation. The tests compare fitted parameters,
Fisher/predictive covariance and interval widths against closed-form regression,
and expose an unidentifiable pair of multiplicative thrust coefficients.
This is software verification; measured thrust, mass-property, weather and
synchronized telemetry inputs plus independent flight cases are still required
to complete measurement validation.

Create and customize a sample dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export the complete synthetic observations and fit settings to an editable JSON
file, then run the edited file through the same workflow::

   python3 -m demos.launch.demo_launch_calibration --write-sample my-calibration.json
   python3 -m demos.launch.demo_launch_calibration --dataset my-calibration.json --output my-report.json

The exporter creates parent directories as needed and refuses to overwrite a
file. Exporting returns code 0. The fit command also refuses to overwrite its
report or input dataset. Its exit code remains 2 for failed or unavailable
validation: the default sample deliberately includes a faulty holdout, and all
sample cases are synthetic. Removing that fault can make the numerical checks
pass; it cannot establish measured-flight validation.

The JSON uses ``schema: ssatk.launch_calibration_dataset.v1`` and contains:

* ``calibration_cases`` and ``validation_cases`` with the complete case fields
  described above. Each default case has six time tags, scalar observations,
  a full covariance matrix, sensor/model settings and input provenance. The
  nominal sample has 10 N thrust and 0.04 m/s velocity bias; ``failed_sensor``
  has an additional 0.3 m/s fault. The histories are deterministic closed-form
  values without random noise; the covariance is a specified uncertainty model.
* ``parameters0``, ``parameter_units``, ``parameter_scale``,
  ``parameter_bounds`` and ``coverage_probability``. Optional ``rank_rtol``
  and ``max_nfev`` go directly to the existing fitter. The sample predictor
  estimates ``thrust_n`` and ``velocity_bias_m_s``; preserve those names and
  their N/m/s units when using this predictor.
* A descriptive ``description`` string. This and source/filter descriptions
  record meaning; they do not execute model changes.

Edit ``observed`` to replace measurements and edit ``covariance`` in squared
observable units to describe their uncertainty and correlation. If time tags
or sample counts change, update the observation/covariance dimensions and
``sensor.support_gps_s`` together. Times are absolute GPS seconds. Duplicate
case IDs or overlapping calibration/validation support from the same flight
are rejected. Duplicate an existing case to add another independent run,
giving it appropriate IDs, inputs and provenance.

The sample's ``model`` settings control initial mass, constant propellant flow,
constant initial spin, epoch and integration tolerances. Its sensor settings
control the body-frame lever arm, ``clock_offset_s``,
``first_sample_lead_s`` and ``filter_weights``. Physical sample time is tag
time plus the clock offset. The first extra filter sample is one lead interval
before the first tag; subsequent outputs pair each tag with its predecessor.
The two nonnegative weights sum to one and apply to the previous/current
sample, in that order. Corrected samples must lie after the initial model
epoch and inside the declared support. For example, ``[0.25, 0.75]`` weights
the current sample more strongly than the previous one.

Changing these settings changes predictions; it does not regenerate or replace
the stored observations. This preserves the calibration/held-out split while
trying a different physical or sensor model. Keep the description fields
consistent with the numeric settings. The supplied predictor supports only
``sensor_velocity_x`` in ``m/s`` with frame ``GCRF x velocity``. Different
observables, force laws or instrument filters require a corresponding Python
``predict`` callback passed to ``calibrate_launch_measurements``; JSON labels
alone cannot supply those physics.

Keep ``evidence: synthetic`` and the synthetic source labels for generated or
edited sample histories. When actual data becomes available, replace those
histories, source hashes, uncertainty descriptions and sensor/model metadata
with the measured case information before marking a case ``measured``.
Choose acceptance requirements before fitting and retain failed heldouts.
Generated datasets and reports belong in user-selected data/output locations;
the Toolkit stores the generator and workflow, not generated datasets.

Step 15: reproducible capability statement
------------------------------------------

:doc:`launch_capability` defines the supported domain and the bundle/replay
workflow for these examples. It freezes source revisions and staged changes,
installed versions, data hashes, synthetic inputs, solver settings, seeds,
histories, metrics and explicit missing-evidence statuses. It does not promote
component checks or editable synthetic data to measured-flight validation.

Integrated Falcon 9 catalog pad-to-LEO example
-----------------------------------------------

Run the combined reference through the existing launch event runner::

   python -m demos.launch.demo_integrated_launch --output new-mission.json
   python -m pytest tests/test_integrated_launch.py tests/test_launch_events.py -q

The same propagation starts at the rotating WGS84 pad and covers release,
powered ascent, first-stage depletion, conservative separation, upper-stage
guidance, insertion cutoff and coast. It combines central gravity and the
existing J2 model, a declared atmosphere and ENU wind, aerodynamic loads,
changing propellant inventory, sampled sensors/control and bounded actuators.
The target is a declared synthetic 200 km LEO, with explicit perigee, apogee,
eccentricity, inclination, flight-load, reserve and duration requirements.

The single vehicle sample is ``launch_vehicles["Falcon 9 Full Thrust"]``.
The existing catalog supplies lower/upper dry masses of 25,600/3,900 kg,
propellant inventories of 395,700/92,670 kg and engine counts. The lower stage
uses nine engines at constant sea-level thrust/Isp (845 kN each, 282 s); the
upper uses one engine at constant vacuum thrust/Isp (934 kN, 348 s).
These endpoint values do not model ambient-pressure or throttle performance.
The catalog has no per-value citations or resolved vehicle version. Its
549,000 kg listed total exceeds its stage inventory by 31,130 kg; that
unassigned mass is reported as metadata rather than inferred as payload.

Declared synthetic assumptions are a 1,000 kg payload, residual inventory of
0.1 percent per stage, a 10 m2 aerodynamic area and isotropic inertia equal to
dry mass times 10 m2 (including the payload). Initial wet mass is 518,870 kg.
The bodies use spherical drag, COM point tanks, noiseless instantaneous sensors
and bounded external attitude torques. Upper guidance reserves acceleration
for radial control before requesting tangential acceleration, and sampled PD
gains scale with retained inertia after staging. These are demonstration
controls, not a Falcon flight program. Its atmosphere is isothermal, with
exponentially sampled pressure and constant wind over its declared domain.
The heating observable is a reduced convective proxy with declared coefficient
``1.83e-4`` SI and 1 m nose radius; it does not establish thermal validity.
Angle of attack is the total air-relative incidence to body +Z in this example.
Proper acceleration includes thrust and drag and excludes gravity. Propellant
reserve includes retained residual inventory. The target orbit uses osculating
two-body elements with altitudes above ``EARTH_RADIUS``; ground clearance uses
WGS84 ellipsoidal height. These altitude definitions serve different purposes.

Flight limits are descending event roots and terminate a violation during
propagation, including an already-violated initial/reset condition through
``LaunchEvent(trigger_below=True)``. The final orbit is checked separately:
crossing 100 km alone cannot produce a pass. Failure to insert, upper-stage
propellant exhaustion, ground contact or a mission timeout fails the mission.
Retained load maxima
use accepted control and output boundaries; roots guard crossings between
those samples. Tangential or very narrow violations still require suitable
solver refinement; a numerical pass is conditional on that resolution.
The guidance cutoff uses half the declared orbit tolerance, leaving an
internal margin for J2-driven osculating variation during the remaining coast;
final acceptance still uses the unchanged complete mission budget.

``check_launch_requirements(contract["requirements"], evidence, profile="LEO")``
reuses the contract's comparison semantics. Evidence maps each dotted field to
``{"value": scalar, "unit": string}``; units must match and values must be
finite. Missing evidence remains ``NOT RUN`` and cannot yield a pass. It checks
the supplied values, not their provenance or sampling completeness. The demo's
Python ``run_reference(requirements=...)`` accepts the same requirements mapping
and requires its documented evidence fields/units. Limits and targets are
declared before propagation, never selected from its result.

The JSON retains the catalog snapshot, unassigned catalog mass, synthetic
assumptions, requirements, evidence, pad state, event/conservation ledger,
control history and trajectory. The report is never overwritten; exit code 2
reports failed/incomplete numerical acceptance. Flight qualification remains
``NOT RUN``. This scenario is not a reconstruction of the launch associated
with the pad epoch, and does not replace source-backed vehicle data, measured
weather, thermal/attitude calibration, independent flight validation or
dispersion studies. It is separate from the existing capability bundle's case
inventory.
