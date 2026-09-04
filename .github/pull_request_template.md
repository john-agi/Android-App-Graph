## Linked issue

Closes #

## Summary

## Definition of done

- [ ] The branch is `issue/<number>-<slug>` (with a `-<part>` suffix only for issues that explicitly allow several PRs) and the body names the issue with `Closes #` or `Part of #`
- [ ] `uv run --locked poe check-fast` passed on every commit on the branch; `git commit --no-verify` was never used
- [ ] `uv run --locked poe check` (the definition of done) passed locally on the final commit
- [ ] No configuration was weakened to make a check pass: no rule removed, no threshold lowered, no `per-file-ignores` or `[tool.ruff.lint] ignore` entry beyond the ones `AGENTS.md` sanctions
- [ ] Every suppression is inline and names the rule and a reason: `# noqa: CODE  # reason` for Ruff, `# ty: ignore[rule]  # reason` for ty; no bare `# noqa`, no `# type: ignore`
- [ ] Tests added or updated, or the summary explains why none were needed
- [ ] `uv.lock` is committed together with `pyproject.toml` if dependencies changed
- [ ] The full diff was reviewed before requesting review

The repository owner merges this pull request; the author never merges it.
