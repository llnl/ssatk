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
and it is not a GMAT comparison. Six existing preflight tests remain unchanged.
