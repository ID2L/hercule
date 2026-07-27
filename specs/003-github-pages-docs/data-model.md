# Phase 1 Data Model: Publish API Documentation on GitHub Pages

**Feature**: `003-github-pages-docs` | **Date**: 2026-07-28

## Scope note

This feature introduces **no application data model**: no Pydantic model, no persisted structure, no change to
`HerculeConfig`, `EpochResult`, or any Root Class Registry entry. The spec's optional *Key Entities* section was
intentionally omitted for that reason.

What follows documents the build-time entities the pipeline manipulates, and the configuration keys it consumes,
so that a future change to either is recognised as affecting this feature.

## Build-time entities

### Documentation Site

The published output as seen by a reader.

| Field | Value / Source | Notes |
|-------|----------------|-------|
| URL | `https://id2l.github.io/hercule/` | Derived from owner + repository name; stable across publications (FR-001) |
| Entry point | `index.html` | Meta-refresh to `hercule.html`; relative, so the project sub-path works (research D-003) |
| Content root | one HTML page per documented module + `search.js` | Produced by pdoc |
| Documented packages | `["src/hercule"]` | From `[tool.pyscaf.documentation].package_paths` |
| Freshness | commit SHA of the last successful `main` publication | Visible in the repository's Deployments tab |

**State transitions**: `absent` → `published` (first successful deploy) → `republished` (each subsequent
successful deploy). A failed build performs **no** transition: the previous state persists (FR-008).

### Documentation Build Output

The local/CI filesystem product of generation.

| Field | Value / Source | Notes |
|-------|----------------|-------|
| Directory | `docs/` | From `[tool.pyscaf.documentation].output_path`, mirrored as `DOCS_DIR` in the workflow (research D-007) |
| Producer | `uv run gen-doc` → `pyscaf.documentation.scripts.parse_doc:gen_doc` | Same command locally and in CI (FR-003) |
| Tracked by git | **No** | `.gitignore` entry; currently 19 untracked files to delete (FR-005) |
| Validity rule | directory MUST exist and be non-empty after generation | Enforced by the guard step; empty output fails the build |

### Pages Artifact

The hand-off between the two jobs.

| Field | Value / Source | Notes |
|-------|----------------|-------|
| Name | `github-pages` | Fixed by `actions/upload-pages-artifact` |
| Payload | tar of the build output directory | Serves as-is; no Jekyll processing |
| Produced when | event is not `pull_request` | Pull requests build but never upload (FR-007) |
| Retention | default workflow artifact retention | Not relied upon; each deployment re-uploads |

### Deployment

| Field | Value / Source | Notes |
|-------|----------------|-------|
| Environment | `github-pages` | Carries the resulting URL as its environment URL |
| Trigger events | `push` to `main`, `workflow_dispatch` | Never `pull_request` (FR-011) |
| Concurrency group | `pages`, `cancel-in-progress: false` | In-flight deploy completes; queued runs superseded by the newest (FR-009) |
| Required permissions | `pages: write`, `id-token: write` (job-scoped) | Workflow default remains `contents: read` |

## Configuration contract consumed

These existing keys are inputs to this feature. Changing one without updating the other side breaks publication.

| Key | File | Consumed by | Breakage if changed alone |
|-----|------|-------------|---------------------------|
| `[tool.pyscaf.documentation].output_path` | `pyproject.toml` | `gen-doc`, workflow `DOCS_DIR`, `.gitignore` | Guard step fails the build (loud, not silent) |
| `[tool.pyscaf.documentation].package_paths` | `pyproject.toml` | `gen-doc` | Published content changes accordingly — intended (FR-003) |
| `[project.scripts].gen-doc` | `pyproject.toml` | workflow build step | Build step fails: command not found |
| `[dependency-groups].dev` (contains `pdoc`, `tomli`) | `pyproject.toml` | `uv sync --frozen` | `gen-doc` unavailable in CI |
| `uv.lock` | repository root | `uv sync --frozen` | Build fails if the lock is stale (intended, FR-010) |

## Validation rules

- **VR-001**: after generation, the output directory exists and contains at least one file — otherwise fail the
  build before any upload.
- **VR-002**: `git ls-files docs` must return nothing before adding the ignore rule, so no tracked file is
  masked (research D-009).
- **VR-003**: the deploy job runs only when `github.event_name != 'pull_request'`.
- **VR-004**: dependency installation uses the committed lock without resolution (`--frozen`).
