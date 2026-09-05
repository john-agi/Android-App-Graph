# AGENTS.md

Engineering contract for everyone who changes this repository: humans, coding
agents, the git hooks and CI. It follows the AGENTS.md convention
(https://agents.md/): plain Markdown at the repository root. Explicit
instructions in an issue or a chat prompt override this file, but the
definition of done below is enforced by the pre-push hook and by CI as soon as
they exist, and no prompt can override it.

This file states the target contract. The "Tooling" section lists only the
tools adopted so far; a rule that names a tool not yet listed there (type
checker, architecture checker, dependency checker, dead-code checker, tests and
coverage, workflow audit, hooks, CI) applies from the moment that tool is
adopted. Every tooling change updates this file in the same pull request: one
bullet in "Tooling", optionally one section named after the concern, nothing
else.

## Definition of done

```bash
uv run --locked poe check
```

exits 0. That is the whole definition, for a human, an agent, the pre-push hook
and CI alike. Run `uv run --locked poe fix` first if you want Ruff autofixes and
formatting applied, then run `check`. `uv run --locked poe check-fast` is the
quick subset (format check and lint today; type check and architecture check
once those tools are adopted) and every commit must pass it.
`uv run --locked poe --help` lists the public tasks.

Task conventions in `pyproject.toml`. `executor = "simple"` runs commands
directly on PATH: `uv run` has already activated `.venv`, so the default
`auto` executor would nest a second `uv run` per task (it selects the uv
executor whenever `uv.lock` exists). Private leaf tasks are inline strings and
underscore-prefixed: hidden from `poe --help`, not runnable directly,
referenced only from the sequences. Public leaf tasks that carry help text are
tables with `help` and `cmd`. Composite tasks are tables with `help` and
`sequence`. Never write an inline array next to an existing table (TOML
rejects duplicate keys). The canonical final order is:

- `check-fast`: `_format-check`, `_lint`, `_typecheck`, `_architecture`
- `check`: `check-fast`, `_dependencies`, `_environment`, `_dead-code`,
  `_test`, `_actions`, `_build`

A tooling change inserts its task by position relative to the neighbours that
already exist: `_typecheck` right after `_lint`; `_architecture` right after
`_typecheck`; `_dependencies` right after `check-fast`; `_dead-code` right
after `_environment`; `_test` right before `_actions` if present, otherwise
right before `_build`; `_actions` right before `_build`. `_build` is always
last. The help strings are fixed; do not edit them.

## Tooling

One bullet per adopted tool: role, tool, what it checks, how to run it, and the
Poe task it sits in. Later tooling changes append their bullet here in the same
style.

- Project manager: uv (`uv sync --locked`, `uv add --dev <tool>`,
  `uv lock --check`; `uv.lock` is committed and never edited by hand).
- Task runner: Poe the Poet (`uv run --locked poe <task>`; public tasks `fix`,
  `check-fast` and `check`; private components are underscore-prefixed and run
  only inside those sequences; `_build` is always the last component of
  `check`).
- Formatter and linter: Ruff (`uv run --locked ruff format --check .` and
  `uv run --locked ruff check .`; `_format-check` and `_lint` in
  `poe check-fast`; `poe fix` runs `ruff check --fix --exit-zero .` then
  `ruff format .`).
- Environment check: `uv pip check` (`_environment` in `poe check`).
- Package build: `uv build --no-sources` (`_build` in `poe check`).
- GitHub Actions (`.github/workflows/ci.yml`): runs the whole definition of
  done, `uv run --locked poe check`, on every pull request and every push to
  `main`. Poe task: `check`.
- Git hooks: prek (`uv run prek install --hook-type pre-commit --hook-type pre-push`
  once per clone; `.pre-commit-config.yaml` runs `poe fix` then `poe check-fast`
  on `git commit` and `poe check` on `git push`; no Poe task, hooks are not part
  of `check`).
- Dependency hygiene: deptry (`uv run --locked deptry .`; `_dependencies` in
  `poe check`, immediately after `check-fast`). Declared dependencies must
  match the imports in `src/`, `scripts/`, `aitk_files/` and `aw_files/`:
  fix findings with `uv add`, `uv add --dev`, `uv add --optional <extra>` or
  `uv remove`, never with `exclude`, `extend_exclude`, `ignore` or
  `package_module_name_map`. Suppressions are `# deptry: ignore[DEPxxx]  # reason`
  on the import line or a `[tool.deptry.per_rule_ignores]` entry with a TOML
  comment naming the file that needs it; the two pre-approved entries are
  `DEP001 = ["android_world"]` (copy-in Android World adapter) and
  `DEP002 = ["dill"]` (imported by the git-pinned `aitk`, which declares no
  dependencies).
- Type checker: ty (`uv run --locked ty check`; `_typecheck` in
  `poe check-fast`, right after `_lint`; scope is `src/` and `tests/` only, the
  root adapters and scripts are not type-checked).
- Workflow audit: zizmor (`uv run --locked zizmor --offline .`; `_actions` in
  `poe check`, immediately before `_build`; audits every workflow under
  `.github/workflows/` for unpinned actions, broad `GITHUB_TOKEN`
  permissions, persisted credentials, template injection and cache misuse;
  audit reference https://docs.zizmor.sh/audits/).
- Branch protection: GitHub repository ruleset `main`
  (`.github/rulesets/main.json`, applied with the `gh api` commands in
  `.github/rulesets/README.md`; `gh ruleset check main` shows the live rules):
  blocks direct pushes, force pushes and deletion of `main` and requires the
  `Quality` check on an up-to-date pull request. Enforced by GitHub at push and
  merge time; no Poe task.
- Tests: pytest with pytest-cov (branch coverage of `android_app_graph`, floor
  `[tool.coverage.report] fail_under`), pytest-randomly and Hypothesis
  (`_test` in `poe check`; `poe test-unit` for a fast run without coverage).
- Dependency security: uv (`.github/workflows/security.yml` runs
  `UV_MALWARE_CHECK=1 uv sync --locked` for the OSV malware check, then
  `uv audit`; on dependency-changing pull requests, daily, and on demand; no
  Poe task, deliberately outside `poe check`; see "Dependency security").
- Dead code: Vulture (`uv run --locked vulture src tests vulture_allowlist.py
  --min-confidence 90`; `_dead-code` in `poe check`; `poe dead-code-review`
  prints the 60% review queue; see "Dead code" below).
- Mutation testing: mutmut (`uv run --locked --group mutation poe mutate`;
  `mutation` dependency group; not part of `poe check`; runs weekly in
  `.github/workflows/mutation.yml`; see "Mutation testing" below).
- Tach: import direction between the modules of `src/android_app_graph/`, as
  declared in `tach.toml`; `_architecture` (`tach check`), run by `check-fast`
  and therefore by `check`.

## Policies

- Branch and pull request: picking up an issue starts with the one-time
  `gh repo set-default john-agi/Android-App-Graph` (this clone is a fork, and
  every `gh` command that targets the repository relies on that recorded
  default). Branch from `main` as `issue/<n>-<slug>`, where `<n>` is the issue
  number. Open a pull request into `main` whose body contains `Closes #<n>`.
  An issue that explicitly allows several pull requests uses one branch per
  part, `issue/<n>-<slug>-<part>`; every pull request but the last says
  `Part of #<n>` and only the last says `Closes #<n>`. The implementing agent
  never merges the pull request;
  the repository owner does. `git commit --no-verify` is never used. Every
  commit on the branch passes `uv run --locked poe check-fast`, and the pull
  request as a whole passes the definition of done. Read the issue fully,
  including "Out of scope"; do only what it asks; keep mechanical changes
  (autofixes, formatting, lock updates) in their own commits; inspect
  `git diff main...HEAD` before opening the PR and remove anything the issue
  did not ask for. If something in the issue is wrong, impossible or
  ambiguous, stop and report on the issue instead of improvising.
- Suppressions: inline only, each naming the rule and a reason on the same
  line:
  - Ruff: `# noqa: CODE  # reason`
  - ty: `# ty: ignore[rule]  # reason`
  Bare `# noqa`, `# noqa` without a code, and `# type: ignore` are forbidden;
  the type-checker configuration, once adopted, is set so that
  `# type: ignore` is not honoured. `[tool.ruff.lint.per-file-ignores]` was set
  once (`scripts/**`, `aitk_files/**`, `aw_files/**`: `T201`, `T203`,
  `INP001`; `tests/**`: `INP001`, `S101`) and is not extended, except for the
  dead-code allowlist file once the dead-code checker is adopted; no entries
  are added to `[tool.ruff.lint] ignore`.
- Unused parameters: give them a leading underscore instead of a suppression.
  Ruff's ARG rules exempt names matching `dummy-variable-rgx` and Vulture
  ignores names that start with an underscore, so test fakes name unused
  parameters `_request`, `_timeout` and so on.
- Dependencies: add and remove only with `uv add` / `uv remove` (`uv add --dev`
  for tools; a dedicated `--group` for heavy optional tooling). Never edit
  `uv.lock` by hand. Commit `pyproject.toml` and `uv.lock` together, in the
  same commit. `[tool.uv] required-version = ">=0.12.9,<0.13"` is a range that
  matches the `uv_build` bound; bumping it is a deliberate manual change made
  in its own pull request, and Renovate does not update it. A declared version
  floor must be at least 3 days old: `[tool.uv] exclude-newer = "3 days"` hides
  registry releases published inside that window, so a floor pointing at a
  younger release asks for a version uv is not allowed to see and makes
  `uv lock` unsatisfiable. `uv add` always writes `>=<latest>`; when that
  release is younger than 3 days, lower the floor by hand to the newest release
  that predates the cutoff, and let Renovate raise it once the newer release
  ages out. Never remove or shorten `exclude-newer` to make a floor resolve.
- Tests: every behaviour change comes with tests, and every bug fix starts
  with a failing regression test. Tests live in `tests/`, are committed, and
  never need adb, an emulator, network access or API keys. Until the test
  suite exists, say explicitly in the PR that tests are deferred to it.
- Coverage ratchet: the coverage floor is `[tool.coverage.report] fail_under`,
  set from the measured baseline once the test suite exists and only ever
  raised. Never lower `fail_under`.
- No weakening of configuration: never weaken lint, type, architecture, test,
  coverage or security configuration to make a check pass. No new
  `per-file-ignores` or `ignore` entries, no relaxed type-checker rules, no
  lowered coverage floor, no `ignore_fail` on Poe sequences, no skipped or
  disabled hooks or CI steps, no edits to the fixed Poe help strings. If a
  check cannot be satisfied without one of these, change the code; if that is
  impossible, stop and report on the issue or PR instead of guessing.
- Code: annotate everything you touch and do not sprinkle `Any` to silence the
  checker. No `except Exception` unless it sits at a boundary (a CLI entry
  point, an adapter callback, a retry loop) and either logs the traceback or
  re-raises. No global mutable state. No `utils/` or `helpers/` dumping
  grounds: new modules are named for what they do, and `android_app_graph/utils/` does
  not grow further. Delete dead code instead of commenting it out.
- This file: a tooling change appends its bullet at the end of "Tooling" and,
  if it adds a section, appends that section after the last existing `## `
  section of the file, whatever that section is at the time (never inserted
  after "Policies" or "Definition of done"), so the order of the appended
  sections depends on merge order.

## Repository layout

- Distribution `android-app-graph`, import package `android_app_graph` in
  `src/android_app_graph/`, console scripts `app-graph`, `app-graph-audit`,
  `app-graph-plot` and `app-graph-embed`, static version in `pyproject.toml`,
  credential variables prefixed `APP_GRAPH_`. These names are final; the
  former names survive only in `LICENSE`, `NOTICE`, the README attribution
  text and the README's references to the upstream project's own files. This
  is a fork of YuxiangChai/UI-KOBE that diverges freely; upstream
  compatibility is not a constraint.
- Python 3.14 only: `requires-python = ">=3.14"`, `.python-version` is `3.14`.
- `aitk` is a Git dependency pinned to a commit in `[tool.uv.sources]`. There is
  no sibling checkout. `uv run --locked` refuses to run when `uv.lock` is
  stale, which is intended.
- `tests/` holds the test suite and is committed.
- `scripts/`, `aitk_files/`, `aw_files/` and `configs/` at the root are
  operator scripts and copy-in adapters, not part of the package. Ruff lints
  them; the type checker, once adopted, does not check them; the dependency
  checker, once adopted, does not treat their `android_world` imports as
  runtime dependencies. `aitk_files/android_app_graph_v2.py` and
  `aw_files/android_app_graph_aw_agent.py` are copied into other repositories:
  they import `android_app_graph` and their host framework and nothing from
  `scripts/`. `aitk_files/android_app_graph_v2.py` is a shim that re-exports
  `register` and `UIKobeV2Translator` from
  `android_app_graph.adapters.aitk_translator`; the translator itself is package
  code and is type-checked and tested. `scripts/` holds only `run_explore.sh`;
  the `app-graph-*` commands live in `src/android_app_graph/commands/`.
- Generated graphs, logs, outputs and `dist/` are git-ignored; `.gitignore` is
  the authority on what is not tracked.

## Continuous integration

`.github/workflows/ci.yml` defines one GitHub Actions job (job `Quality`) that
checks out the repository, installs uv and Python, runs `uv sync --locked` and
then `uv run --locked poe check`; a red `Quality` check means the change is not
done. The setup-uv step passes `version:` equal to `[tool.uv] required-version`;
when that range is bumped (a manual change, see Policies), update both in the
same commit. The checkout, setup-uv and `uv python install` steps are the
reference copy that every other workflow in `.github/workflows/` reproduces
byte-for-byte.

## Git hooks

After cloning, run `uv run prek install --hook-type pre-commit --hook-type pre-push`. The pre-commit stage runs `poe fix` then `poe check-fast`; the pre-push stage runs `poe check`. Run a stage on demand with `uv run prek run --all-files` (commit stage) or `uv run prek run --all-files --stage pre-push` (push stage). Never bypass the hooks with `git commit --no-verify` or `git push --no-verify` to land a change; fix the cause. If a hook fails with `The lockfile at uv.lock needs to be updated, but --locked was provided`, run `uv lock` and stage `uv.lock`. Hooks are local; CI's `uv run --locked poe check` on every pull request is the gate.

## Type checking

The configuration lives in `[tool.ty]` in `pyproject.toml` and is strict: on
top of ty's defaults it turns `blanket-ignore-comment`,
`dynamic-function-decorator-return`, `missing-override-decorator`,
`missing-type-argument`, `unsound-assignment`, `unsound-return-statement` and
`unsound-yield` into errors, `possibly-unresolved-reference` and
`unsupported-dynamic-base` into warnings (blocking, because
`error-on-warning` is on), and sets `strict-equality-semantics` and
`strict-generic-narrowing`. `division-by-zero`, `possibly-missing-attribute`
and `possibly-missing-import` are deliberately left off: ty's migration guide
says they have a significant number of false positives. Never weaken any of
this to make a check pass.

`# ty: ignore[rule]  # reason` on the offending line is the only suppression
that works, and every one of them is justified in the pull request that adds
it. `# type: ignore` is inert because `respect-type-ignore-comments = false`,
and a suppression that stops matching is reported by `unused-ignore-comment`
and must be deleted.

`src/android_app_graph/py.typed` marks the package as typed, so consumers of the wheel
see these annotations. Untyped payloads are narrowed at the boundary with the
helpers in `src/android_app_graph/payloads.py` rather than with `cast`.

The aitk device surface is typed through the `DeviceController` and
`AvdManager` Protocols in `src/android_app_graph/device.py`. Tests and moved scripts
take those Protocols, never the concrete `ADBController` and `AVDManager`
classes, which is what lets a fake device stand in without adb or an emulator.

## Workflow audit

`uv run --locked poe check` runs zizmor offline (`_actions`) over every workflow
under `.github/workflows/`; every finding at the default `regular` persona is
blocking. Every `uses:` is pinned to a full commit SHA with the release tag in a
trailing `# vX.Y.Z` comment, workflows set `permissions: {}` at the top level and
grant scopes per job, and `actions/checkout` sets `persist-credentials: false`.
A suppression is a last resort, one per verified false positive, listed in the
pull request that adds it: `# zizmor: ignore[audit-id]  # reason` as a YAML
comment inside the finding's span, or a `rules.<audit-id>.ignore` entry with a
`file.yml:line` location plus a YAML comment in `.github/zizmor.yml` (zizmor
does not read `pyproject.toml`). `disable`, `remap`, `--persona`,
`--min-severity`, `--min-confidence` and `--no-exit-codes` are never used.

## Branch policy

- `main` is protected by the repository ruleset `main`. Source of truth:
  `.github/rulesets/main.json`; apply and update commands:
  `.github/rulesets/README.md`.
- Direct pushes to `main` are rejected (`GH013: Repository rule violations
  found`). Every change lands through a pull request whose required status
  check `Quality` (the job in `.github/workflows/ci.yml`) passed on a branch
  that is up to date with `main`.
- Force pushes to `main` and deletion of `main` are blocked. Never run
  `git push --force` against `origin main`.
- If `main` moved after your last CI run, update your branch (merge `main` into
  it, or use "Update branch" on the PR) and wait for `Quality` to pass again
  before it can be merged.
- The repository owner merges pull requests; the implementing agent never does
  (see Policies). The only bypass is the repository admin role, and only when
  merging a pull request. It is not used to merge a red PR; the PR gets fixed
  instead.
- To change the ruleset, edit `.github/rulesets/main.json` in a pull request
  and re-apply it with the `PUT` command in the README after the merge. Never
  weaken it: `enforcement` stays `active` and `Quality` stays required.

## Tests

- Run `uv run --locked poe test-unit` for a fast run and
  `uv run --locked poe check` for the definition of done (includes coverage).
  Free arguments are appended to pytest:
  `uv run --locked poe test-unit -k cli -x`.
- pytest runs in strict mode with `filterwarnings = ["error"]`,
  `--import-mode=importlib` and random ordering (pytest-randomly). Reproduce an
  order-dependent failure with `--randomly-seed=<n>` (the seed is printed at the
  top of every run) and a Hypothesis failure with `--hypothesis-seed=<n>`; the
  two seeds are independent because pytest-randomly does not reseed Hypothesis.
  `-p no:randomly` disables shuffling for one run only; never put it in
  configuration.
- Coverage is branch coverage of `android_app_graph`, measured by `_test`
  (`pytest --cov=android_app_graph --cov-report=term-missing`).
  `[tool.coverage.report] fail_under` is a ratchet: it was set from the measured
  baseline, it is raised when coverage grows, and it is never lowered.
  `--cov-fail-under` never appears in `addopts` or a task. A PR that lowers the
  floor is rejected.
- Tests must never need adb, an emulator, network access or API keys.
  `tests/conftest.py` deletes every `APP_GRAPH_*`, `GEMINI_API_KEY`,
  `GOOGLE_API_KEY` and `OPENAI_API_KEY` variable before each test; do not work
  around it.
- Hypothesis profiles live in `tests/conftest.py`: `default` locally, `ci`
  (500 examples, deterministic) when `CI` is set or `HYPOTHESIS_PROFILE=ci`.
- `tests/` has no `__init__.py`; test modules cannot import each other, so
  shared fixtures go in `tests/conftest.py`. Unused parameters of fakes are
  named with a leading underscore instead of being suppressed.
- Do not add `# pragma: no cover`, `exclude_also` patterns, or warning ignores
  for code in `src/` to make a check pass.

## Dependency security

`.github/workflows/security.yml` runs `uv sync --locked` with uv's OSV
malware check (`UV_MALWARE_CHECK=1`) and then `uv audit` on pull requests
that change `pyproject.toml`, `uv.lock` or the workflow, daily at 06:17
UTC, and on demand (`gh workflow run security.yml`). Both uv features are
preview and may change; see https://docs.astral.sh/uv/concepts/preview/.

- A red run on a dependency-changing pull request blocks that pull request.
  Fix it with `uv lock --upgrade-package <name>` and commit the lockfile
  change on its own, or, when no fixed version exists, add the advisory ID
  to `[tool.uv.audit] ignore-until-fixed` with a comment and a linked issue.
  A plain `ignore` needs a written justification in the pull request.
- A red scheduled or manual run means a new advisory against a locked
  dependency or an OSV outage. Triage it the same day in its own pull
  request; never silence it by disabling the workflow, the job, the check,
  or by setting `UV_MALWARE_CHECK=0`.
- If the only error is "Malware check failed due to an error from OSV"
  (`uv sync` exits 2) or a network error from `uv audit`, rerun the job with
  `gh run rerun <run-id>`.
- uv is pinned to the `[tool.uv] required-version` range, which
  `security.yml` passes to `setup-uv` as `version:` exactly as `ci.yml`
  does; its checkout, setup-uv and `uv python install` steps are
  byte-for-byte copies of `ci.yml`. A patch release inside the range can
  still change these preview features, so a green run proves the malware
  check ran only if the sync step's log contains
  "Malware checks are experimental" (`gh run view <run-id> --log`); if that
  line is gone, treat the run as red and re-read the uv release notes.
- This workflow is not part of `poe check` and is not a required status
  check on `main`: it needs the network and its result changes over time.
- Never hand-edit `uv.lock`; use `uv lock --upgrade-package`.

## Dead code

- `uv run --locked poe check` runs Vulture over `src/`, `tests/` and
  `vulture_allowlist.py` at `--min-confidence 90`. Findings at 90% or above
  (unused imports, unused function arguments, unreachable code) fail the
  check.
- Fix a blocking finding by deleting the code. An unused parameter whose
  signature is imposed by a caller is renamed with a leading underscore
  (`_source`), which Vulture and Ruff's `ARG` rules both ignore; a fixture
  requested only for its side effect becomes `@pytest.mark.usefixtures`. Add
  an entry to `vulture_allowlist.py` only when the name is used outside the
  scanned paths (a parameter passed by keyword, a function or import consumed
  by AITK or Android World through the copy-in adapters in `aitk_files/` and
  `aw_files/`, or by `scripts/`). Every entry has the form `_.name  # reason`
  and names its external caller. Remove entries whose reason stops being true.
- `uv run --locked poe dead-code-review` lists findings at 60% confidence,
  sorted by size ascending (largest last). That list is a review queue: a
  person, or an agent with an explicit task issue naming the item, reads it
  and decides. Findings under 90% are never deleted automatically and never
  fail a check.
- Never lower `min_confidence`, never add `exclude`, `ignore_names` or
  `ignore_decorators` to `[tool.vulture]`, and never add `# noqa` to silence
  Vulture. The allowlist is the only suppression mechanism. The
  `"vulture_allowlist.py" = ["B018"]` entry in
  `[tool.ruff.lint.per-file-ignores]` is the single sanctioned exception to
  the fixed per-file-ignores table and covers only that file.
## Mutation testing

- `uv run --locked --group mutation poe mutate` runs mutmut over `src/android_app_graph` using `tests/`
  (`[tool.mutmut]` in `pyproject.toml`). It is not part of `poe check` and never gates a
  merge. `.github/workflows/mutation.yml` runs it every Monday and on demand
  (`gh workflow run mutation.yml`); the log and the `mutmut-results` artifact list the
  counts (`mutmut export-cicd-stats`) and the surviving mutants.
- Surviving mutants are a review queue. Read them with `mutmut browse` (a TUI that can also
  write a mutant into the real source: only read in it, never use its apply action) and
  decide per mutant: the mutant reveals an unstated behaviour, so add a test that states
  that behaviour; or the mutant is equivalent or touches behaviour we do not promise, so
  leave it (optionally mark the line `# pragma: no mutate` with a reason); or the code is
  dead, so delete it.
- Never write a test merely to kill a mutant if that test would restate the implementation
  (asserting internal call order, exact log text, or re-deriving the formula under test).
  Such a mutant is a signal to leave it surviving or to simplify the code, not to add a test.
- Never run `mutmut apply`; it writes mutations into real source files.
- mutmut needs `fork`: Linux and macOS work; on Windows run inside WSL.
- `mutants/` is a git-ignored local cache; delete it to force a complete rerun.

## Architecture boundaries

- `tach.toml` is the declared import direction for `src/android_app_graph/`;
  `uv run --locked poe check-fast` (and so `poe check`, the pre-commit hook and
  CI) runs `tach check` against it.
- Never run `tach sync` (with or without `--add`) to make `tach check` pass. It
  rewrites `tach.toml` from whatever the code currently imports, which defeats
  the check.
- When `tach check` fails, either fix the import or, if the dependency is
  genuinely intended, edit `depends_on` by hand and state the architectural
  justification in the pull request body. A pull request that touches
  `tach.toml` without that justification is not mergeable.
- `# tach-ignore(reason) name` is the only allowed suppression and must carry a
  reason; `tach check` rejects directives without one.
