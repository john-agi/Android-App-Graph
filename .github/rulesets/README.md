# Repository rulesets

`main.json` is the exact payload of the GitHub repository ruleset named `main`
that protects the default branch. GitHub stores the live ruleset outside git,
so this file is the reviewable source of truth: change it here in a pull
request, then re-apply it after the merge.

Create (first time only; the endpoint answers HTTP 201 and prints the new id):

    gh api --method POST repos/{owner}/{repo}/rulesets --input .github/rulesets/main.json

Update (every later change; the id is looked up by name):

    RULESET_ID=$(gh api repos/{owner}/{repo}/rulesets --jq '.[] | select(.name == "main") | .id')
    gh api --method PUT "repos/{owner}/{repo}/rulesets/$RULESET_ID" --input .github/rulesets/main.json

Inspect:

    gh ruleset list
    gh ruleset view "$RULESET_ID"
    gh ruleset check main

Requirements: `gh` authenticated as a repository admin (`gh auth status` must
list the `repo` scope). `{owner}` and `{repo}` are filled in by gh from the
current repository. Rules reference:
https://docs.github.com/en/rest/repos/rules
