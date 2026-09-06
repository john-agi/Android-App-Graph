---
name: security-triage
description: Triage a red run of the Security workflow in `.github/workflows/security.yml`, which runs uv's OSV malware check and `uv audit` on dependency-changing pull requests, daily, and on demand. Use when that workflow fails, when a new advisory is reported against a locked dependency, or when asked to check dependency security.
---

# Triaging a red Security run

The workflow runs `UV_MALWARE_CHECK=1 uv sync --locked` and then `uv audit`.
It is deliberately outside `poe check` and not a required status check: it
needs the network, and its result changes as advisories are published.

1. Read the failing step's log: `gh run view <run-id> --log`.
2. If the only error is "Malware check failed due to an error from OSV"
   (`uv sync` exits 2) or a network error from `uv audit`, rerun the job with
   `gh run rerun <run-id>`. Never set `UV_MALWARE_CHECK=0`.
3. A real advisory against a locked dependency: run
   `uv lock --upgrade-package <name>` and commit the lockfile change on its
   own. Never hand-edit `uv.lock`.
4. No fixed version exists: add the advisory ID to
   `[tool.uv.audit] ignore-until-fixed` with a comment and a linked issue. A
   plain `ignore` needs a written justification in the pull request.
5. On a dependency-changing pull request a red run blocks that pull request. A
   red scheduled or manual run is triaged the same day in its own pull request.
   Never silence it by disabling the workflow, the job or the check.
6. Both uv features are preview. A green run proves the malware check ran only
   if the sync step's log contains "Malware checks are experimental"; if that
   line is gone, treat the run as red and re-read the uv release notes.
