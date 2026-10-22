# Working on SSATK

Working rules for changes to this repository. Read this file before changing
anything, and read [skills.md](skills.md) before writing code that uses the
package.

[CONTRIBUTING.md](CONTRIBUTING.md) is authoritative on licensing, layout and
merge requirements, and all of it applies here except where this file sets a
narrower testing rule (see "Tests cover finalized functionality" below). This
file adds what the tree does not show: which rules are machine-enforced, which
are not, and the mistakes this codebase invites. It grants no authority beyond
the permissions of the person doing the work.

## Commits, comments and notes

- **Commit messages and pull request descriptions** state what changed, why,
  and how it was verified. The pull request title is the commit subject.
- **Authorship:** commit with your own Git identity, as the person responsible
  for the change. Authors and co-authors are the people responsible for it.
- **Branch names describe the change:** `fix/...`, `feature/...`, `docs/...`,
  `ci/...`.
- **Comments explain why the code is the way it is**, for its next reader.
  History belongs in the commit message; review notes, change logs and
  commented-out code do not belong in files.
- **Plans, notes and scratch files stay outside the working tree.** If one must
  sit in the checkout, list it in `.git/info/exclude`, not in `.gitignore`.
- **Working guidance lives in `AGENTS.md` and `skills.md`.** Keep it there
  rather than in code, docs or new files.

## Ground rules

- **Work on a topic branch and open a pull request against `main`.** Do not
  push to `main`, force-push a shared branch, rewrite published history, merge,
  tag or release unless a maintainer asks for that specific operation.
- **Keep each change to one concern.** No drive-by reformatting, import
  reordering, mass lint fixes or unrelated renames.
- **Tests cover finalized functionality.** Add a test only for behavior the
  repository relies on and that is settled, not for every change. A bug fix or
  refactor does not need a new test of its own: verify it with the existing
  suite plus a one-off check, and state what you ran in the commit message.
- **Tests compare against a reference.** Each test that is added checks a
  computed value against a closed form, an independent implementation, a
  published value, a conservation law or a finite-difference derivative, with a
  stated tolerance and unit (`docs/testing_policy.md`). Do not add tests that
  only run code, raise coverage, or check imports, rendering or argument
  validation.
- **Never weaken a check to make a change pass.** Do not delete, skip or
  `xfail` a test, loosen a tolerance, relax
  `scripts/check_repository_policy.py`, or add lint suppressions to get green.
  If a check legitimately blocks correct work, say so and stop.
- **Never invent a number.** Residuals, timings, tolerances, test counts and
  benchmark results appear in a commit message, docstring or pull request only
  if you produced them by running something on this change. Mark an unknown
  value as unknown.
- **Report blockers exactly**: the failing command, the environment and the
  error text. Do not work around a failure silently or broaden scope to fix it.
- **Verify, do not inherit.** Old notes, logs and completion claims are
  records, not requirements. Confirm against the tree, the tests and
  the remote before acting on them.

## Requires a maintainer

Do not do these on your own initiative:

- Merging, tagging, releasing, or changing the version. The version lives in
  `pyproject.toml`, `codemeta.json` and `CITATION.cff`, and
  `scripts/check_release_metadata.py` fails CI when they disagree.
- Adding, removing or promoting a dependency. Several packages sit in optional
  extras (`static`, `pdf`, `video`, `browser`, `notebook`, `monitoring`,
  `geomagnetics`, `atmosphere`, `validation`, `propulsion`, `benchmarks`) on
  purpose.
- Adding a tracked top-level path. It also needs an `ALLOWED_TOP_LEVEL` entry
  in `scripts/check_repository_policy.py` and a row in the CONTRIBUTING.md
  layout table.
- Committing data, figures, generated output or binary files. Persistent data
  belongs in the split `ssa-data-*` packages; the policy check rejects media,
  archives, notebooks and files over 1,000,000 bytes.
- Raising the Python floor (`>=3.10`; CI tests 3.10 to 3.14).

## Scope

SSATK is a standalone toolkit built on SSAPy (`llnl-ssapy`, imported as
`ssapy`). Core propagation and force-model work belongs in
[SSAPy](https://github.com/llnl/SSAPy). Before adding a propagator, force
model, frame transform or helper, search SSATK and SSAPy for an existing one:

```bash
rg "def <name>|class <Name>|<keyword>" ssatk demos tests
```

## Environment

```bash
python3 -m pip install -e ".[dev,geomagnetics,atmosphere,propulsion]"
python3 -c "import sys, ssapy, ssatk; print(sys.executable, ssapy.__version__, ssatk.__version__)"
```

Use `python3 -m pip`, never bare `pip`: on machines with several interpreters
they differ, and bare `pip` reports packages as installed in a tree the running
interpreter cannot see. Packaged data comes from the `ssa-data-*` distributions
through `ssatk.data`; do not read data through repository-relative paths.

## Checks

Run the checks that match the change, then the full set before asking for
review:

```bash
python3 scripts/check_repository_policy.py
python3 scripts/check_release_metadata.py
python3 -m pytest tests -q
MPLBACKEND=Agg python3 -m pytest scripts/tests -q
ruff check --select E9,F63,F7,F82 <changed .py files>
git diff --check
node --check ssatk/plots/satellite_viewer_scene.js
python3 -m sphinx -b html -W --keep-going docs /tmp/docs
python3 -m ssatk.run_all_demos --no-open
```

A green run covers less than it appears to:

- The Ruff gate is syntax-level. A full `ruff check` reports many style
  findings; raise specific ones, do not bulk-fix them.
- `check_repository_policy.py` checks the diff only. On a clean checkout it
  inspects nothing.
- External-reference tests (Orekit, Basilisk, SpacePy/IRBEM) skip unless those
  tools are installed. Do not quote their residuals as verified without running
  them.
- Delete `docs/generated/` before a docs build; stale autosummary stubs fail
  the warning-as-error build. Delete `build/` before a release build.

## Demos

Demos live under `demos/<category>/`, write output through
`ssatk.plots.figsave` (relative paths land in `~/ssatk_output/figures`, or
under `$SSATK_OUTPUT_DIR`), and join the gallery unless the module sets
`GALLERY_INCLUDE = False`. Validation-only scripts should opt out and be
covered by `tests/`. Never commit demo output.

## WebGL viewers

- Every WebGL page loads the one vendored three.js, r185, through
  `ssatk/plots/_threejs.py`. `runtime_loader_html()` embeds
  `three.module.min.js`, `three.core.min.js` and the `OrbitControls.js` add-on
  with an import map, or maps them to served URLs. New viewers use it: do not
  vendor another copy or load three.js from a CDN.
- Classic viewer scripts go in `<script type="text/x-ssatk-classic">` blocks
  (`classic_scripts_html`) so their top-level functions stay page globals;
  module scripts can `import * as THREE from "three"`.
- The existing viewers turn off three.js colour management and scale light
  intensities by pi to keep the pre-r152 rendering they were tuned for. New
  pages may use colour management, but set it in their own scripts.
- Upgrade three.js for every viewer in one change: copy the unmodified
  minified builds and the add-on from the npm package after checking its
  integrity hash, keep each file under 1,000,000 bytes, update
  `THREE_REVISION`, `NOTICE` and `three_license.txt`, and compare screenshots
  of each viewer before and after.

## Known traps

- `ssatk.plots` imports its public modules eagerly when it is imported, so a
  missing optional plotting dependency surfaces there, not at the call site.
- Where a name exists in both SSATK and SSAPy, the top-level `ssatk` name is
  the SSATK one (`ssatk.io` is SSATK I/O); reach SSAPy itself as `ssatk.ssapy`.
- Times are GPS seconds or Astropy `Time` objects, and states are SI (m, m/s)
  in memory. CCSDS files use km and km/s on disk; the readers convert.
- Add time offsets with explicit units (`Time(...) + seconds * u.s`); a bare
  float in days emits `TimeDeltaMissingUnitWarning`.
- SSAPy 1.1.5 and earlier rebuild a zero-drag `Satrec` in `SGP4Propagator`, so
  SGP4 comparisons fail there. That is an environment problem, not an SSATK bug.
