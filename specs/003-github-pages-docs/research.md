# Phase 0 Research: Publish API Documentation on GitHub Pages

**Feature**: `003-github-pages-docs` | **Date**: 2026-07-28

All action versions below were resolved against the GitHub API on 2026-07-28, not from memory.

## D-001 — Deployment mechanism: Pages artifact vs `gh-pages` branch

**Decision**: Deploy with the official Pages artifact pipeline
(`actions/upload-pages-artifact` → `actions/deploy-pages`), repository Pages source set to *GitHub Actions*.

**Rationale**:

- Satisfies FR-005 / SC-004 directly: nothing generated is ever committed, not even to an orphan branch, so the
  repository history stays free of build output.
- Satisfies FR-008: a failed build produces no artifact, so `deploy-pages` never runs and the previously
  published site keeps being served. A `gh-pages` branch approach force-pushes and can leave the branch in a
  half-written state.
- Native deployment status, environment URL, and rollback history in the repository's Deployments tab.

**Alternatives considered**:

- **`peaceiris/actions-gh-pages` (commit to `gh-pages`)**: widely used, works without changing repository
  settings, but reintroduces generated HTML into git history — the exact thing this feature removes. Rejected.
- **Committing `docs/` on `main` and serving from `/docs`**: the current de-facto state. Directly contradicts
  the feature request. Rejected.

## D-002 — Action versions (verified 2026-07-28)

**Decision**: pin to the current major tags below.

| Action                          | Version  | Notes                                                              |
|---------------------------------|----------|--------------------------------------------------------------------|
| `actions/checkout`              | `v7.0.1` | latest release                                                     |
| `astral-sh/setup-uv`            | `v9.0.0` | v9 changed `prune-cache` default to `false`; enable caching        |
| `actions/upload-pages-artifact` | `v5.0.0` | built on `upload-artifact` v7                                      |
| `actions/deploy-pages`          | `v5.0.0` | Node 24 runtime                                                    |
| `actions/configure-pages`       | `v6.0.0` | **not used** — see D-003                                           |

**Rationale**: all four used actions are first-party (`actions/*`) or vendor-official (`astral-sh/*`), so major
tags are an acceptable trust boundary; SHA pinning is available later via Dependabot if the project adopts it.

**Alternatives considered**: SHA-pinning every action. Rejected for now — higher maintenance for a public
repository with no supply-chain policy in place; noted as a possible follow-up.

## D-003 — `actions/configure-pages` and `.nojekyll` are unnecessary

**Decision**: omit both.

**Rationale**: verified against the actual generated output:

- `docs/index.html` is a meta-refresh to `./hercule.html` — a **relative** URL. `grep` for `href="/…"` in the
  generated pages returns nothing, so the site works unchanged under the project sub-path
  `https://id2l.github.io/hercule/`. `configure-pages` exists mainly to expose a `base_url`/`base_path` to
  generators that need it (Next.js, Nuxt…); pdoc does not.
- No generated file or directory starts with `_`, and the artifact pipeline serves the upload **as-is** without
  running Jekyll, so a `.nojekyll` marker has no effect.

**Alternatives considered**: adding `configure-pages` "just in case". Rejected — one more action, one more
failure mode, zero benefit for this generator.

## D-004 — Python environment in CI

**Decision**: `astral-sh/setup-uv@v9` with `enable-cache: true` and an explicit `python-version: "3.12"`, then
`uv sync --frozen`.

**Rationale**:

- `--frozen` installs strictly from the committed `uv.lock` and fails if the lock is stale → FR-010, and it
  doubles as a lockfile drift check.
- The dev group (which carries `pdoc`) is installed automatically thanks to `default-groups = "all"` in
  `pyproject.toml`; no extra flag needed.
- An explicit Python version makes builds deterministic without adding a `.python-version` file that would also
  constrain local development. The project supports 3.10–3.14; 3.12 sits comfortably inside that range.

**Cost/risk identified**: `pdoc` imports every module to introspect it, and `hercule.models.deep_q_learning`
imports **torch**. The docs build therefore installs the full torch wheel (hundreds of MB). Mitigation: the
uv cache via `enable-cache: true` keeps warm runs short. First run will be slow; SC-001 allows 10 minutes.

**Alternatives considered**:

- **Mocking torch / `pdoc --no-import`**: pdoc has no such mode for this layout; would silently degrade output.
- **A docs-only dependency group excluding torch**: would make the published documentation differ from the
  local one (FR-003 violation) and hide import errors that Story 4 is meant to catch. Rejected.
- **`uv sync --no-dev`**: would drop `pdoc` itself. Rejected.

## D-005 — Triggers, job split, and fork safety

**Decision**: one workflow, two jobs.

- `build`: runs on `push` to `main`, on `pull_request` targeting `main`, and on `workflow_dispatch`. Generates
  the documentation; uploads the Pages artifact only when the event is not a pull request.
- `deploy`: `needs: build`, guarded by `if: github.event_name != 'pull_request'`, holds the
  `github-pages` environment and the `pages: write` / `id-token: write` permissions.

**Rationale**:

- FR-007: every PR gets the build as a required-capable check, with nothing published.
- FR-011: default workflow permissions are `contents: read`; elevated permissions live only on the deploy job.
  A pull request from a fork receives a read-only token and never reaches the deploy job — enforced by the
  `if` guard, not only by token scope (defence in depth).
- FR-004: `workflow_dispatch` gives manual republication with no commit.

**Alternatives considered**: separate `docs-pr.yml` and `docs-deploy.yml` files. Rejected — duplicates the build
steps, so the validated build could drift from the published one.

## D-006 — Concurrency semantics

**Decision**: `concurrency: { group: "pages", cancel-in-progress: false }` on the deploy job.

**Rationale**: this is GitHub's documented pattern for Pages. An in-flight deployment finishes (never leaving a
half-published site), while runs queued behind it are superseded by the newest one — so the final published
state corresponds to the most recent commit, satisfying FR-009 and the "two rapid merges" edge case.

**Alternatives considered**: `cancel-in-progress: true`. Rejected — cancelling mid-deployment is exactly the
scenario that can leave the live site inconsistent.

## D-007 — Output directory coupling

**Decision**: declare the output directory once as a workflow-level env var (`DOCS_DIR: docs`), matching
`[tool.pyscaf.documentation].output_path`, and add a guard step that fails the build if the directory is
missing or empty after generation.

**Rationale**: keeps the YAML simple and gives a clear, early failure instead of publishing an empty site if
generation silently produces nothing.

**Trade-off accepted**: the directory name is duplicated between `pyproject.toml` and the workflow. The guard
turns that duplication into a loud failure rather than a silent one. FR-003 is about *content* divergence
(same command, same configured package paths), which is preserved since the workflow calls `uv run gen-doc`.

**Alternative recorded** (if the duplication ever bites — the project already ships `tomli` as a dev dependency):

```yaml
- id: docsdir
  run: |
    echo "path=$(uv run python -c "import tomli,pathlib;print(tomli.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['tool']['pyscaf']['documentation']['output_path'])")" >> "$GITHUB_OUTPUT"
```

## D-008 — One-time Pages activation

**Decision**: activate via the API and document both paths in `quickstart.md`.

Verified current repository state on 2026-07-28: `has_pages: false`, `visibility: public`, `default_branch: main`,
and the maintainer holds `admin` — so activation is possible with no permission escalation, and Pages is free
for this public repository.

```bash
gh api -X POST repos/ID2L/hercule/pages -f build_type=workflow
```

UI fallback: *Settings → Pages → Build and deployment → Source: **GitHub Actions***.

**Rationale**: FR-013 / SC-008 — reproducible on a fork in one command. Until activation, `deploy-pages` fails
with an explicit "Pages site not found" style error rather than appearing to succeed, which matches the
"hosting not yet activated" edge case.

**Alternatives considered**: `actions/configure-pages` with `enablement: true` to self-activate on first run.
Rejected — it silently changes repository settings from CI, needs extra token scope, and hides a decision that
belongs to a human once in the repository's lifetime.

## D-009 — Removing generated output from the working tree

**Decision**: add `docs/` to `.gitignore` and delete the local `docs/` directory. No history rewriting.

**Rationale**: the 19 generated HTML files are currently **untracked** (`?? docs/` in status), so they have never
entered history — nothing to purge, and SC-004 ("0 generated files added from this feature onward") is met by
the ignore rule alone. `dist/` and `outputs/` are already ignored the same way, so this follows existing
convention.

**Verification required at implementation time**: confirm `git ls-files docs` is empty before ignoring, so no
tracked file is accidentally masked.

## D-010 — Verification record (added during implementation, 2026-07-28)

**Decision**: record which rows of the failure-semantics table in `contracts/workflow-contract.md` §6 were
exercised for real, and which are reasoned from the mechanism, so the distinction is not lost.

| Contract §6 row | Status | Evidence |
|-----------------|--------|----------|
| `gen-doc` raises (import error) | **Exercised** | Temporary `import nonexistent_module` pushed to the PR branch: run 30315268159 failed at *Generate the documentation*, the guard, upload, and deploy steps were skipped, no deployment was created |
| Pull request builds but never publishes | **Exercised** | Run 30315107496 (PR #1): upload and deploy steps skipped; `deployments` count stayed at 0 |
| Guard fails on empty/missing output | **Reasoned** | Not exercised — would require temporarily desynchronising `output_path`. The step is a plain `[ ! -d ] || [ -z "$(ls -A)" ]` test that runs before any upload |
| Dependency install fails (stale `uv.lock`) | **Reasoned** | Not exercised — `uv lock --check` was green throughout. `--frozen` is documented to fail rather than re-resolve |
| Pages not activated | **Not observed** | Pages was activated (T011) before the first merge, exactly as the quickstart prescribes, so the error path never triggered |
| Deploy fails mid-flight | **Not observed** | All three deployments succeeded |

**No divergence found** between the contract and observed behaviour.

**Measured performance** (supersedes the estimate in D-004):

- Cold cache, full torch stack: build **2 min 19 s**.
- Warm cache: build **72 s**; push-to-live **≈ 1 min 30 s** — well inside the 10-minute SC-001 budget.
- The uv cache absorbs the torch download as predicted; no runner disk or time pressure observed.

## D-011 — `ruff` is not installed (found during implementation)

**Finding**: `uv run ruff check .` fails with `program not found`. Ruff is absent from `[dependency-groups].dev`
in `pyproject.toml`, even though the constitution (principle V) names it the project's single linter/formatter
and both `README.md` and `AGENTS.md` advertised the command.

**First decision (superseded)**: document `uvx ruff check .` instead, running Ruff as an ephemeral tool with no
dependency change, since adding a dependency was outside the scope of a documentation-publication feature.

**Resolution (2026-07-28, at the maintainer's request)**: `ruff>=0.16.0` was added to
`[dependency-groups].dev`, so the linter version is pinned in `uv.lock` and shared by every developer and by CI.
The documented commands are back to `uv run ruff check .` / `uv run ruff format .`, which is now accurate.

**Baseline recorded at the time of adoption** — the existing codebase is not Ruff-clean, so future changes should
be judged against this baseline rather than against zero:

| Rule | Count | Rule | Count |
|------|-------|------|-------|
| `PLC0415` import-outside-top-level | 10 | `B904` raise-without-from-inside-except | 1 |
| `F401` unused-import | 9 | `TC001` typing-only-first-party-import | 1 |
| `E501` line-too-long | 2 | `TC003` typing-only-standard-library-import | 1 |
| `UP035` deprecated-import | 2 | `W293` blank-line-with-whitespace | 1 |

27 errors total (2 auto-fixable), 15 of them in `src/hercule/reports/__init__.py`; `ruff format --check .`
reports 5 files needing reformatting. Cleaning this debt is deliberately left out of feature 003 — it would mix
a formatting sweep into an infrastructure change and touch files this feature has no business editing.

## Open questions

None. No `NEEDS CLARIFICATION` remained after the specification phase, and no new unknown surfaced during
research. Two findings surfaced during implementation and are recorded above as D-010 and D-011.
