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
  in its own pull request, and Renovate does not update it.
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
  grounds: new modules are named for what they do, and `ui_kobe/utils/` does
  not grow further. Delete dead code instead of commenting it out.
- This file: a tooling change appends its bullet at the end of "Tooling" and,
  if it adds a section, appends that section after the last existing `## `
  section of the file, whatever that section is at the time (never inserted
  after "Policies" or "Definition of done"), so the order of the appended
  sections depends on merge order.

## Repository layout

- Distribution `ui-kobe`, import package `ui_kobe` in `src/ui_kobe/`, console
  script `kobe-explore`, static version in `pyproject.toml`. This is a fork of
  YuxiangChai/UI-KOBE that diverges freely; upstream compatibility is not a
  constraint.
- Python 3.14 only: `requires-python = ">=3.14"`, `.python-version` is `3.14`.
- `aitk` is a Git dependency pinned to a commit in `[tool.uv.sources]`. There is
  no sibling checkout. `uv run --locked` refuses to run when `uv.lock` is
  stale, which is intended.
- `tests/` holds the test suite and is committed.
- `scripts/`, `aitk_files/`, `aw_files/` and `configs/` at the root are
  operator scripts and copy-in adapters, not part of the package. Ruff lints
  them; the type checker, once adopted, does not check them; the dependency
  checker, once adopted, does not treat their `android_world` imports as
  runtime dependencies. `aitk_files/ui_kobe_v2.py` and
  `aw_files/ui_kobe_aw_agent.py` are copied into other repositories: keep them
  self-contained, importing `ui_kobe` and their host framework and nothing
  from `scripts/`. `scripts/` import `ui_kobe` as an installed package; no
  `sys.path` edits.
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
