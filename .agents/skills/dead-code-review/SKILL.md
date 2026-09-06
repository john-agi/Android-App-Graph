---
name: dead-code-review
description: Work through Vulture's dead-code findings, both the blocking 90% run inside `poe check` and the 60% review queue from `poe dead-code-review`. Use when the `_dead-code` task fails, when asked to review or remove unused code, or when deciding between deleting a name, renaming it with an underscore, and adding it to `vulture_allowlist.py`.
---

# Reviewing dead-code findings

`uv run --locked poe check` runs
`vulture src tests vulture_allowlist.py --min-confidence 90`; findings at 90%
or above (unused imports, unused arguments, unreachable code) fail the check.
`uv run --locked poe dead-code-review` lists findings at 60%, sorted by size.
That list is a review queue: never deleted automatically, never a failing
check.

For each finding:

1. Delete the code if nothing uses it.
2. Rename a parameter whose signature is imposed by a caller with a leading
   underscore (`_source`); Vulture and Ruff's ARG rules both ignore it. A
   fixture requested only for its side effect becomes
   `@pytest.mark.usefixtures`.
3. Add an entry to `vulture_allowlist.py` only when the name is used outside
   `src/` and `tests/`: a parameter passed by keyword, or a function or import
   consumed by AITK or Android World through `aitk_files/`, `aw_files/` or
   `scripts/`. Every entry has the form `_.name  # reason` and names the
   external caller. Remove entries whose reason stops being true.

Never lower `min_confidence`, add `exclude`, `ignore_names` or
`ignore_decorators` to `[tool.vulture]`, or silence Vulture with `# noqa`. The
allowlist is the only suppression.
