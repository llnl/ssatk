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
  Earth's rotation in GCRF. Both the epoch-coverage check and the Astropy
  ITRS<->GCRS transform use SSATK's packaged Earth-orientation snapshot
  (:func:`ssatk.environment_eop.load_packaged_eop` and the matching
  ``finals2000A.all`` loaded as an Astropy ``IERS_A`` table), with network
  access disabled throughout. This avoids depending on whichever IERS table
  ships with the installed Astropy, and avoids Astropy's silent
  UT1-UTC=0/mean-polar-motion fallback for an epoch its own bundled table
  does not cover. An epoch that falls in the packaged table's predicted
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
  ``launch_site_state`` with a closed-form WGS84 reference and with SSATK's
  independent erfa-based ``itrf_to_gcrf_eop`` path for that.

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
``--repository ssatk=path/to/SSAPy-Toolkit`` and
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
