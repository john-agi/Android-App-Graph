# AGENTS.md

Engineering contract for everyone who changes this repository: humans, coding
agents, the git hooks and CI. Explicit instructions in an issue or a prompt
override this file, except the definition of done, which the pre-push hook and
CI enforce.

## Definition of done

```bash
uv run --locked poe check
```

exits 0. Run `poe fix` first to apply Ruff autofixes and formatting. Every
commit passes `poe check-fast`. `poe --help` lists the public tasks; what each
check enforces lives in its configuration, not in this file. After cloning, run
`uv run prek install --hook-type pre-commit --hook-type pre-push` once so the
hooks run `fix` and `check-fast` on commit and `check` on push.

## Never

- Weaken a check to make it pass: no removed rule, lowered threshold, new
  `per-file-ignores` or `ignore` entry, relaxed type rule, lowered
  `fail_under` or `min_confidence`, `ignore_fail` on a Poe sequence, or
  skipped or disabled hook, CI step or workflow. If the code cannot satisfy a
  check, stop and report.
- `git commit --no-verify` or `git push --no-verify`.
- `tach sync`. It rewrites `tach.toml` from the code and defeats the check.
- `mutmut apply`. It writes mutants into real source files.
- Edit `uv.lock` by hand, or commit it separately from `pyproject.toml`.
- Bare `# noqa`, `# noqa` without a code, or `# type: ignore`. ty is configured
  to ignore `# type: ignore` and report it as unused.
- `except Exception` outside a boundary (CLI entry point, adapter callback,
  retry loop) that logs the traceback or re-raises.
- Global mutable state, or a new module under `android_app_graph/utils/`.
- Merge a pull request, or force-push `main`. The repository owner merges.

## Suppressions

Inline only, naming the rule and a reason on the same line. Every suppression
is justified in the pull request that adds it.

| Tool | Syntax |
|---|---|
| Ruff | `# noqa: CODE  # reason` |
| ty | `# ty: ignore[rule]  # reason` |
| Tach | `# tach-ignore(reason) name` |
| deptry | `# deptry: ignore[DEPxxx]  # reason`, or a `per_rule_ignores` entry with a comment naming the file |
| zizmor | `# zizmor: ignore[audit-id]  # reason` inside the finding's span, or a `.github/zizmor.yml` entry with a comment |
| Vulture | an `_.name  # reason` entry in `vulture_allowlist.py` naming the external caller |
| mutmut | `# pragma: no mutate  # reason`, only for an equivalent mutant |
| coverage | none: no `# pragma: no cover`, no `exclude_also` |

Unused parameters get a leading underscore (`_request`) instead of a
suppression; Ruff's ARG rules and Vulture both ignore them.

## Code

- Comments explain why, never what. A comment that describes what the code does
  is deleted and the code made to say it, by renaming or extracting. A comment
  stays only when it records what the code cannot show: a workaround, an
  invariant, a decision with its link. This applies to configuration files too.
- Annotate everything you touch; no `Any` to silence the checker. Narrow
  untyped payloads with the helpers in `payloads.py`, not with `cast`.
- Take the `DeviceController` and `AvdManager` Protocols from `device.py`,
  never the concrete aitk classes, so a fake device can stand in for tests.
- Delete dead code instead of commenting it out.
- A pull request that edits `depends_on` in `tach.toml` states the
  architectural justification in its body.
- `aitk_files/*.py` and `aw_files/*.py` are copied into other repositories:
  keep them self-contained, importing only `android_app_graph` and their host
  framework.

## Tests

- Every behaviour change comes with tests, and every bug fix starts with a
  failing regression test.
- Tests never need adb, an emulator, network access or API keys.
  `tests/conftest.py` deletes credential variables and blocks sockets; do not
  work around it.
- `tests/` has no `__init__.py`, so shared fixtures go in `conftest.py`.
- Warnings are errors. A third-party warning may be ignored after the `error`
  entry of `filterwarnings`, with a reason; a warning from `src/` is fixed.
- Reproduce an order-dependent failure with `--randomly-seed=<n>` and a
  Hypothesis failure with `--hypothesis-seed=<n>`; the seeds are independent.
  `-p no:randomly` is for one run, never for configuration.
- Coverage is a ratchet: raise `fail_under` when coverage grows.

## Dependencies

- Only `uv add` and `uv remove` (`--dev` for tools, a `--group` for heavy
  optional tooling). Commit `pyproject.toml` and `uv.lock` together. If a hook
  fails because the lockfile is stale, run `uv lock` and stage it.
- `required-version` is bumped by hand in its own pull request, together with
  the `version:` of every setup-uv step under `.github/workflows/`; those
  checkout, setup-uv and Python steps are byte-for-byte copies of `ci.yml`.
- `aitk` is pinned to a commit in `[tool.uv.sources]`; there is no sibling
  checkout.

## Branches and pull requests

- Branch from `main` and open a pull request into `main`. Reference the issue
  with `Closes #<n>` when one exists. The clone is a fork: run
  `gh repo set-default john-agi/Android-App-Graph` once.
- Keep mechanical changes (autofixes, formatting, lock updates) in their own
  commits. Review `git diff main...HEAD` before opening the pull request and
  remove what the task did not ask for.
- If the task is wrong, impossible or ambiguous, stop and report instead of
  improvising.
- `main` is protected by the `main` ruleset in `.github/rulesets/`: a pull
  request with a green `Quality` check on an up-to-date branch. Changing the
  ruleset follows that directory's README; `enforcement` stays `active` and
  `Quality` stays required.

## Facts the tree does not show

- Distribution `android-app-graph`, package `android_app_graph`, console
  scripts `app-graph*`, credential variables `APP_GRAPH_*`. These names are
  final; the former names survive only in `LICENSE`, `NOTICE` and the README
  attribution. This is a fork of YuxiangChai/UI-KOBE that diverges freely.
- Python 3.14 only.
- `scripts/`, `aitk_files/`, `aw_files/` and `configs/` are operator scripts
  and copy-in adapters: Ruff lints them, nothing else checks them, and they are
  not in the wheel.

## Where everything else lives

Each fact has one home. Other places link to it and never restate it.

| Kind | Home |
|---|---|
| Commands, task order, thresholds, rule lists | The configuration that enforces them: `pyproject.toml`, `tach.toml`, `.github/` |
| Rules for agents and contributors | This file |
| Multi-step procedures | One skill each under `.agents/skills/`; Claude Code reads them through the `.claude/skills` symlink |
| Setup and usage for people | `README.md` |
| Why a setting exists | A comment on the setting, only when the reason is not obvious |

Skills: `adopt-tool` (adding a check to `poe check`), `security-triage` (a red
Security run), `mutation-triage` (surviving mutants), `dead-code-review` (the
60% Vulture queue).
