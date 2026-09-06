---
name: adopt-tool
description: Add a new quality check or tool to this repository so it runs inside `uv run --locked poe check`. Use when asked to adopt, add, wire or configure a linter, formatter, type checker, test tool, audit or other gate, or to add a Poe task or a GitHub Actions workflow for one.
---

# Adopting a tool

`uv run --locked poe check` verifies every step below; run it before opening
the pull request.

## 1. Install

- `uv add --dev <tool>`; heavy optional tooling gets its own `--group`, like
  `mutation`.
- `uv add` writes `>=<latest>`, and `exclude-newer = "3 days"` hides releases
  younger than three days. If the latest release is that young, lower the floor
  by hand to the newest older release. Never shorten `exclude-newer`.
- Commit `pyproject.toml` and `uv.lock` together.

## 2. Configure

- Configuration goes under `[tool.<name>]` in `pyproject.toml`, or in the
  tool's own file when it cannot read `pyproject.toml`.
- Comments say why a setting exists, never what it does.
- If the tool has its own suppression syntax, add a row to the Suppressions
  table in `AGENTS.md`.

## 3. Wire the Poe task

`[tool.poe]` uses `executor = "simple"` because `uv run` has already activated
`.venv`; the default `auto` executor would nest a second `uv run` per task.

- A private leaf task is an inline string, underscore-prefixed
  (`_lint = "ruff check ."`), hidden from `poe --help` and referenced only from
  the sequences.
- A public leaf task is a table with `help` and `cmd`; a composite task is a
  table with `help` and `sequence`.
- Never write an inline array next to an existing table: TOML rejects duplicate
  keys.
- `check-fast` holds the fast, read-only checks (format, lint, types,
  architecture). Everything else goes into `check` after `check-fast` and
  before `_actions` and `_build`. `_build` is always last.
- Leave the existing tasks and their `help` strings unchanged.

## 4. Workflow, if the tool needs one

- Copy the checkout, setup-uv and `uv python install` steps from `ci.yml`
  byte-for-byte; setup-uv's `version:` equals `[tool.uv] required-version`.
- `permissions: {}` at the top level and scopes per job,
  `persist-credentials: false` on checkout, and every `uses:` pinned to a full
  commit SHA with the release tag in a `# vX.Y.Z` comment. zizmor (`_actions`)
  rejects deviations.
- A scheduled workflow runs off the top of the hour; runs queued at :00 are
  delayed.

## 5. Documentation

- `AGENTS.md` changes only when the tool adds a rule: a Never bullet or a
  Suppressions row. What the check enforces lives in its configuration.
- A new public Poe task gets a line in the README's Development block.
- A multi-step triage procedure for the tool becomes a skill under
  `.agents/skills/`.
