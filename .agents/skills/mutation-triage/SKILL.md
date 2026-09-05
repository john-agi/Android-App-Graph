---
name: mutation-triage
description: Run mutmut mutation testing over `src/android_app_graph` and review the surviving mutants. Use when the weekly Mutation workflow reports survivors, when asked to run or read mutation testing, or when deciding whether a surviving mutant needs a test, a `# pragma: no mutate`, or a deletion.
---

# Reviewing surviving mutants

`uv run --locked --group mutation poe mutate` runs mutmut with `tests/`
(`[tool.mutmut]` in `pyproject.toml`). It gates nothing.
`.github/workflows/mutation.yml` runs it every Monday and on demand
(`gh workflow run mutation.yml`) and uploads the `mutmut-results` artifact
with the counts and the survivors.

- Read survivors with `mutmut browse`. Only read in it: never use its apply
  action, and never run `mutmut apply`, which writes mutants into real source
  files.
- Decide per mutant:
  - It reveals an unstated behaviour: add a test that states that behaviour.
  - It is equivalent, or touches behaviour the project does not promise: leave
    it, optionally marking the line `# pragma: no mutate  # reason`.
  - The code is dead: delete it.
- Never write a test only to kill a mutant if that test would restate the
  implementation (call order, exact log text, re-deriving the formula under
  test). Such a mutant means leave it, or simplify the code.
- mutmut needs `fork`: Linux and macOS work; on Windows use WSL.
- `mutants/` is a git-ignored cache; delete it to force a complete rerun.
