# Contributing to Space Situational Awareness Toolkit (SSATK)

All contributions to SSATK must be made under the BSD 3-Clause License.

Contributions are welcome via pull request targeting the `main` branch of the
[SSATK](https://github.com/LLNL/ssatk) repository. Pull requests
are reviewed by the project maintainers and must pass the repository's CI checks
(tests and linting) before they can be merged.

SSATK is a standalone toolkit; changes to its dependencies, including
[SSAPy](https://github.com/llnl/SSAPy) (`llnl-ssapy`), belong in those projects.

By contributing, you agree to abide by the project's
[Code of Conduct](CODE_OF_CONDUCT.md).

## Repository scope and file layout

SSATK is a source-code repository. Keep generated artifacts, analysis
outputs, downloaded data products, figures, screenshots, animations, notebooks
with embedded outputs, and other binary media out of the repository. If a change
requires persistent data, put that data in
the appropriate ``ssatk-data-*`` repository or document an external
download/source instead of committing it here.

Use the existing top-level structure for new work:

| Path | Purpose |
| --- | --- |
| `ssatk/` | Importable package code. |
| `tests/` | Automated regression and behavior tests. |
| `demos/` | Categorized, runnable demonstrations of user-facing workflows. |
| `docs/` | Narrative documentation and API documentation. |
| `scripts/` | Maintainer/development utilities, not importable package code. |
| `.github/` | GitHub Actions, issue/PR templates, and ownership policy. |

Do not introduce new top-level directories or project-level configuration files
unless the pull request explains why the existing layout cannot support the
change and updates the repository policy check if needed.

## Tests and demos

Pull requests that change package behavior must include automated tests under
`tests/`. Each test must compare a computed value with an independent reference
(closed form, independent implementation, published value, conservation law,
or finite-difference derivative) and state its tolerance; see
`docs/testing_policy.md`. Tests that only show code runs or raise coverage are
not accepted. Documentation-only and CI-only changes do not need new tests, but
the pull request should state that explicitly.

New user-facing workflows, plotting utilities, data-ingest utilities, command
line behavior, or analysis recipes must also add or update a runnable demo under
the appropriate `demos/` subfolder. Demos should be small Python or Markdown examples that generate their
outputs locally. Do not commit generated demo outputs; the gallery workflow
builds media artifacts from source during CI.

Before requesting review, run the focused checks that match the change:

```bash
python -m pip install -e .[dev]
node --check ssatk/plots/satellite_viewer_scene.js  # requires Node.js 20+
pytest tests
python -m ssatk.run_all_demos
python scripts/check_repository_policy.py
```

## Development notes

- Supported floors are `llnl-ssapy>=1.1.11` and the published `ssatk-data-*` distributions. Test
  against the split data distributions. SSAPy 1.1.5 and earlier
  rebuild a zero-drag `Satrec` in `SGP4Propagator`, so SGP4 comparisons fail
  there; that is an environment problem, not a Toolkit bug.
- Use `python3 -m pip` rather than bare `pip` so packages install into the
  interpreter that runs the tests.
- Work on the core propagation and force-model engine belongs in
  [SSAPy](https://github.com/llnl/SSAPy). Check SSAPy and the Toolkit for an
  existing function before adding a new propagator, force model, or transform.
- Build the docs with `python3 -m sphinx -b html -W --keep-going docs /tmp/docs`
  after deleting `docs/generated/`; stale autosummary stubs fail the
  warning-as-error build.
- Delete `build/` before a release build so removed demos cannot persist in a
  stale wheel.

## Review and merge requirements

Each pull request must:

- Preserve the repository layout described above.
- Include tests for package-code changes or explain why no behavior changed.
- Include a demo update for new user-facing workflows or explain why no demo is
  needed.
- Avoid committed data, generated outputs, images, binary media, and large local
  artifacts.
- Pass the GitHub Actions checks before merge.

The repository policy workflow blocks pull requests that add disallowed file
types, binary files, large files, or unexpected top-level paths. If a blocked
file is genuinely required, prefer moving it to the appropriate split data repository. If a repository
policy exception is still necessary, make the exception explicit in the pull
request and update `scripts/check_repository_policy.py` in the same change.
