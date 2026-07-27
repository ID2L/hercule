# Contract: Documentation Publication Workflow

**Feature**: `003-github-pages-docs` | **Date**: 2026-07-28
**Artifact under contract**: `.github/workflows/docs.yml`

This is the externally observable contract of the feature — what maintainers, contributors, and readers can rely
on. The implementation may change as long as this contract holds.

## 1. Triggers

| Event | Build runs | Artifact uploaded | Deployment |
|-------|-----------|-------------------|------------|
| `push` to `main` | yes | yes | yes |
| `pull_request` targeting `main` | yes | no | **never** |
| `workflow_dispatch` (manual, any branch) | yes | yes | yes |
| `push` to any other branch | no | no | no |

Paths filter: none. The build runs for every qualifying event — docstrings live in source files, and restricting
paths would silently skip legitimate documentation changes.

## 2. Job graph

```text
build  (ubuntu-latest)
  ├─ checkout
  ├─ setup uv (cached)
  ├─ uv sync --frozen
  ├─ uv run gen-doc
  ├─ guard: output directory exists and is non-empty      ← fails the build if not
  └─ upload Pages artifact                                 [skipped on pull_request]

deploy (ubuntu-latest, needs: build, if: event != pull_request)
  └─ deploy-pages → environment "github-pages"
```

## 3. Permissions contract

| Scope | Workflow default | `build` job | `deploy` job |
|-------|------------------|-------------|--------------|
| `contents` | `read` | `read` | `read` |
| `pages` | — | — | `write` |
| `id-token` | — | — | `write` |

- No repository secret is consumed. No `write` access to repository contents at any point.
- A pull request opened from a fork receives a read-only token **and** is excluded by the deploy job's `if`
  guard — two independent barriers (FR-011).

## 4. Concurrency contract

- Group: `pages`. `cancel-in-progress: false`.
- Guarantee: an in-flight deployment always completes; runs queued behind it are superseded so that only the
  newest one deploys. The final published state therefore corresponds to the most recent qualifying commit
  (FR-009).

## 5. Output contract

| Property | Value |
|----------|-------|
| Published URL | `https://id2l.github.io/hercule/` — stable, does not change between publications |
| Content | pdoc API reference of the package paths declared in `[tool.pyscaf.documentation].package_paths` |
| Generation command | `uv run gen-doc` — byte-for-byte the same command a contributor runs locally |
| Sub-path safety | all internal links are relative; the site is valid under the `/hercule/` project path |

## 6. Failure semantics

| Failure | Observable outcome |
|---------|--------------------|
| Dependency install fails (stale `uv.lock`) | Build fails; nothing uploaded; live site unchanged |
| `gen-doc` raises (import error, bad docstring) | Build fails; nothing uploaded; live site unchanged |
| Generation produces an empty/missing directory | Guard step fails; nothing uploaded; live site unchanged |
| Pages not activated on the repository | `deploy` job fails with an explicit Pages error; live site unchanged (there is none yet) |
| Deploy fails mid-flight | Previous deployment remains the served version; rerun is idempotent |

**Invariant**: no partial or empty site is ever published. Every failure path leaves the previously published
content served as-is (FR-008).

## 7. Contributor-facing contract

- A pull request that breaks documentation generation shows a failing check named after the build job, before
  merge (FR-007).
- Generating documentation locally leaves the working tree clean; the output directory is ignored (FR-005).
- `uv run gen-doc` and `uv run serve-doc` keep working locally exactly as before (FR-006).

## 8. Explicit non-contract

The following are **not** guaranteed and must not be relied upon:

- Documentation for tags, releases, or branches other than `main` (single latest version only).
- Any custom domain, analytics, or server-side search.
- Documentation of the vendored `pyscaf` package.
- A specific build duration; only the 10-minute end-to-end budget of SC-001.
