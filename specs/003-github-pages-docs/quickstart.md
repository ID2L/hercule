# Quickstart: Publish API Documentation on GitHub Pages

**Feature**: `003-github-pages-docs` | **Date**: 2026-07-28

Everything a maintainer needs to activate, verify, and troubleshoot the documentation publication.

## 1. One-time activation (required, do this before merging)

GitHub Pages is currently **not** enabled on `ID2L/hercule` (verified 2026-07-28: `has_pages: false`). Enable it
with the *GitHub Actions* source. Requires admin rights on the repository — the current maintainer has them.

```bash
gh api -X POST repos/ID2L/hercule/pages -f build_type=workflow
```

UI equivalent: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

Verify:

```bash
gh api repos/ID2L/hercule/pages --jq '{status, html_url, build_type}'
# expected: build_type = "workflow", html_url = "https://id2l.github.io/hercule/"
```

On a fork, substitute your `owner/repo`; the published URL becomes `https://<owner>.github.io/<repo>/`.

## 2. Verify the pull request check (User Story 4)

From the feature branch, with the workflow file committed:

```bash
gh pr create --fill
gh pr checks --watch
```

Expected: the build check runs and passes; **no** deployment appears in
`gh api repos/ID2L/hercule/deployments`. Nothing is published from a pull request.

To exercise the failure path deliberately, temporarily add `import nonexistent_module` at the top of
`src/hercule/__init__.py`, push, and confirm the check turns red — then revert.

## 3. Verify the publication (User Stories 1–2)

After merging to `main`:

```bash
gh run watch                    # follow the run
gh run list --workflow=docs.yml --limit 3
```

Then open <https://id2l.github.io/hercule/> and confirm:

- the index redirects to the package page;
- every model sub-package is listed (`td_models`, `simple_q_learning`, `simple_sarsa`, `deep_q_learning`,
  `dummy`) — SC-005;
- a class page shows its docstrings.

Manual republication without a commit (FR-004):

```bash
gh workflow run docs.yml
```

First run is slow: pdoc imports every module, so torch is installed from scratch. Subsequent runs hit the uv
cache. Budget is 10 minutes end to end (SC-001).

## 4. Verify the clean working tree (User Story 3)

```bash
git ls-files docs          # MUST print nothing — no tracked file is being masked
rm -rf docs                # discard the local build artefacts
uv run gen-doc             # regenerate
git status --short         # MUST print nothing about docs/
```

Local preview must still work (FR-006):

```bash
uv run serve-doc
```

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Deploy job fails with a Pages "not found" error | Pages never activated, or source is not *GitHub Actions* | Run step 1 |
| Build fails on `uv sync --frozen` | `uv.lock` is stale relative to `pyproject.toml` | Run `uv lock` locally and commit the result |
| Build fails during `gen-doc` with an import error | A module raises on import; pdoc imports everything | Fix the import — this is the check working as intended |
| Guard step fails: output directory missing or empty | `[tool.pyscaf.documentation].output_path` changed without updating `DOCS_DIR` in the workflow | Realign the two values |
| Site is live but stale | Deployment superseded or failed | `gh run list --workflow=docs.yml`, inspect the newest run |
| Pages show broken styling under `/hercule/` | An absolute-root link crept into the generated output | Re-check research D-003; a `configure-pages` step would then become necessary |
| Fork pull request shows no deploy | Expected behaviour | Forks never publish (FR-011) |

## 6. Rollback

The feature is fully reversible without touching history:

1. Delete `.github/workflows/docs.yml` — publication stops, the last published site keeps being served.
2. Optionally disable Pages: `gh api -X DELETE repos/ID2L/hercule/pages`.
3. Optionally remove the `docs/` line from `.gitignore` to return to a locally generated directory.

No generated file ever entered git history, so there is nothing to purge.
