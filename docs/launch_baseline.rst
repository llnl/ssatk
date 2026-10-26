Launch simulation baseline
==========================

Step 1 establishes a reproducible software and data baseline before changing
launch-simulation behavior. This page is a qualification checklist, not evidence
that an installation, preflight check or trajectory has passed.

The planning source is
``SSAPy_Launch_Project_Combined_Literature_Review_and_Execution_Schedule_CURRENT_BASELINE.docx``.
Its reference releases are listed below. They are a planning snapshot, not a
claim about the latest available releases or the current development checkout.

Reference versions
------------------

Repository validation uses SSATK ``main`` revision
``c247817d8a590ca2be1890f8ecc578fe8b74ba12`` (version ``1.1.4``), its declared
``llnl-ssapy>=1.1.12`` floor and the four required ``ssa-data-*==0.0.1`` wheels.
The table below retains the historical document baseline; it is not the current
repository's dependency contract. Record the resolved exact versions, wheel
hashes and import origins in retained validation evidence.

.. list-table:: Document reference baseline
   :header-rows: 1
   :widths: 24 12 64

   * - Package
     - Version
     - Source revision
   * - LLNL SSAPy
     - 1.1.10
     - ``a895a5b58b0d71db69c7abebd256a48737e93efe``
   * - SSAPy Toolkit (historical ``ssapy-toolkit`` distribution)
     - 1.0.5
     - ``f1597212678be5c216e9994cb903e830b8cf7f26``
   * - LLNL SSAPy Data (historical ``llnl-ssapy-data`` distribution)
     - 0.1.5
     - ``5db5d30fb099e2f57f0bfde7fce2dc504a4e8976``

Record both the required baseline and the actual runtime. An installed version
string does not prove that imported source matches its published release.
If development revisions replace these pins, explicitly approve and record the
replacement; an ancestor relationship is not an exact revision match.

Repository and data ownership
-----------------------------

* SSAPy owns the core propagation, force-model and coordinate-transform engine.
* SSAPy Toolkit owns higher-level workflows and analysis-ready utilities.
* SSAPy-Data owns numerical inputs, reference data and retained evidence products.

Use the canonical repositories without replacing them or discarding local work.
Keep launch documentation under ``docs/`` and runnable examples under ``demos/``;
do not create a parallel top-level physics or asset tree. Keep session notes in
the ignored ``AGENTS.local.md`` and generated evidence out of the source commit.
Use repository-relative paths or package resources rather than local user paths.

Existing components to reuse
----------------------------

Inspect these components before introducing a new implementation:

* ``ssatk.launch.sites`` and ``ssatk.launch.vehicles`` provide
  reference catalogs. Their presence does not establish vehicle-owner provenance,
  engineering validity domains or a qualified launch configuration.
* ``ssatk.data.data_resource`` and ``ssatk.data.data_path`` resolve
  packaged resources. Use installed, published ``ssa-data-*`` distributions and
  their wheel RECORD hashes, not a Data source checkout.
* ``ssatk.accelerations_6dof.thrust`` supplies finite-thrust profiles and
  packaged-curve adapters; qualify the selected engineering inputs separately.
* ``ssatk.propagators_6dof.sixdof`` and
  ``ssatk.propagators_orbit.high_accuracy`` provide existing propagation
  interfaces. Importing them is not verification of launch dynamics or handoffs.
* ``ssatk.launch.gravity_turn`` supplies local radial/north/east thrust
  directions. It remains an illustrative helper, not qualified mission guidance.

Evidence required for the baseline gate
---------------------------------------

Retain the following in the evidence-owning data repository or an explicitly
configured external evidence location, not as generated Toolkit source files:

* Exact package and dependency versions, Python version and implementation,
  optional precision dependencies, operating system and machine architecture.
* Exact Git revisions, worktree status, installation mode and actual import
  origins for each required package. An editable install or source shadowing a
  wheel must be visible in the record.
* A tested dependency lock and clean-install log. A list of installed versions
  alone is only an inventory, not proof that the environment can be recreated.
* The baseline configuration and its hash, plus a resource inventory with source,
  license, units, frame or coordinate convention, validity domain and hashes.
* Payload verification against the selected manifest: reject missing files,
  placeholder payloads and Git LFS pointer text. A manifest hash is not evidence
  that the resolved payload was checked.
* An import log for every public module required by the selected mission and a
  requirements/open-issues register that distinguishes unknowns from passed checks.

Qualification runs must use resolved packaged inputs without runtime downloads.
If a check fails, retain the exact command, environment and error; do not silently
substitute a different interpreter, resource or revision to obtain a pass.

Exit condition and next step
----------------------------

Step 1 passes only when a clean environment installs the approved pinned baseline,
imports every required public module and retains the evidence above. Report
unavailable inputs as blocked or not run. Local import success, a small test suite
or cross-code agreement does not establish full-mission or flight qualification.

After baseline review, define the mission contract and separate numerical error,
data uncertainty and model error. LEO is the planning default. Select the site,
epoch, vehicle, payload, final orbit, constraints and numerical tolerances from
authoritative inputs rather than inventing them. A GEO target must include the
transfer, coast and final-orbit maneuvers; a GTO-only endpoint is not GEO completion.
Pause for review after each implementation step.

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

``demos/launch/demo_real_launch_preflight.py`` compares initialization for SpaceX
Demo-2 with the independent SpacePy coordinate backend, supplied by the existing
``validation`` extra. NASA records LC-39A and 3:22 p.m. EDT on May 30, 2020:
https://www.nasa.gov/image-article/demo-2-launch-setting-forth-historic-journey/.
The input epoch is the published launch minute, not subsecond flight telemetry.

Run the single new comparison test with output visible::

   python3 -m pytest -q -s tests/test_launch_preflight.py::test_real_demo2_launch_preflight_matches_spacepy

Create an HTML report at an explicitly selected data/output location::

   python3 -m demos.launch.demo_real_launch_preflight --output ../SSAPy-Data/.cache/launch_validation/demo2.html

The check compares independent WGS84 conversions of the same catalog pad
coordinates and inertial speed from Earth rotation. The position tolerance is
numerical, not a surveyed pad uncertainty. The speed allowance comes from the
floating-point Julian-day spacing in SpacePy's central-difference reference.
J2000/GCRF direction differences are displayed but not scored: SpacePy omits EOP
corrections and assumes UT1 equals UTC. This is not an ascent or telemetry match,
and it is not a GMAT comparison. It does not qualify the mission contract.

SSAPy-Data motor-burn simulation
--------------------------------

Run an actual variable-mass propagation using packaged propulsion inputs::

   python3 -m demos.launch.demo_packaged_motor_burn

This isolated Estes C6 motor example loads a public-domain RASP thrust curve and
its motor mass/provenance sidecar from the installed ``ssa-data-propulsion`` wheel.
It verifies the index, curve and sidecar hashes/sizes against the manifest before
loading them, using the existing Toolkit data resolver and strict preflight
resource verifier. No runtime downloads or Data source-checkout imports occur.

The example reuses ``load_packaged_thrust_curve``, ``SpacecraftManeuverAccel``
and ``Spacecraft.propagate``. It infers a constant effective specific impulse
from the sampled curve's impulse and catalog propellant mass, then propagates
mass depletion and fixed-axis motion. This is a declared demonstration model,
not a measured specific impulse or propellant-depletion history. The user-sourced
curve's impulse differs from the catalog value; both and their difference are
reported without silently rescaling the input. The output retains provenance,
resource checks, sampled states, mass balance and a rocket-equation comparison.

There is no gravity, atmosphere, wind, payload, staging or rotational dynamics.
The identity inertia does not represent calibrated motor geometry. This verifies
a data-fed burn workflow, not the LEO mission contract, ascent or flight
qualification. ``qualification_status`` remains ``NOT RUN``. Retained JSON
output belongs in SSAPy-Data or an explicitly selected external evidence location,
not in the Toolkit source commit.
