# Implementation Plan: Publish API Documentation on GitHub Pages

**Branch**: `003-github-pages-docs` | **Date**: 2026-07-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-github-pages-docs/spec.md`

## Summary

Publish the pdoc-generated API reference of `src/hercule` to GitHub Pages automatically, and stop keeping the
generated HTML in the working tree.

Technical approach: a single GitHub Actions workflow with two jobs. `build` installs the locked environment with
uv, runs the project's existing `uv run gen-doc` command, guards that output was actually produced, and uploads
it as a Pages artifact. `deploy` — skipped for pull requests — publishes that artifact through the official
Pages deployment pipeline. `docs/` is added to `.gitignore`, and Pages is switched to the *GitHub Actions*
source once by a maintainer. No application code is touched.

## Technical Context

**Language/Version**: No new application code. CI definition in GitHub Actions YAML; build runs Python 3.12 (project supports 3.10–3.14)  
**Primary Dependencies**: `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v9.0.0`, `actions/upload-pages-artifact@v5.0.0`, `actions/deploy-pages@v5.0.0`; existing `pdoc` dev dependency via `uv run gen-doc`  
**Storage**: N/A — the published site is a deployment artifact, not persisted state  
**Testing**: Manual/observational (workflow run on a PR, then on `main`), plus the existing `uv run pytest` suite which this feature must not disturb  
**Target Platform**: `ubuntu-latest` runner; published to `https://id2l.github.io/hercule/`  
**Project Type**: Single Python package (`src/hercule`) with CLI — this feature adds repository infrastructure only  
**Performance Goals**: Publication completes within 10 minutes of merge (SC-001); warm-cache runs expected far below that  
**Constraints**: `pdoc` imports every module, so the docs build must install the full runtime including torch — mitigated by the uv cache (research D-004). No secrets required. Fork pull requests must never publish  
**Scale/Scope**: 1 workflow file, 1 `.gitignore` line, 1 README link, 1 one-time repository setting, ~18 documented public modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against `.specify/memory/constitution.md` v1.0.0.

| Principle | Applies? | Verdict |
|-----------|----------|---------|
| I. Generic Algorithm Architecture (NON-NEGOTIABLE) | No | No model class is added, modified, or subclassed. |
| II. Configuration-Driven Design | Indirectly | PASS — the workflow invokes `uv run gen-doc`, which reads `[tool.pyscaf.documentation]` from `pyproject.toml`; package paths are not duplicated in CI (see research D-007 for the one accepted duplication and its guard). |
| III. Gymnasium-First Integration | No | No environment handling involved. |
| IV. Module Separation | Yes | PASS — no new package under `src/hercule/`. New files live in `.github/workflows/`, outside the source tree, so the documented package responsibility table is unchanged. |
| V. Modern Python & Code Quality | Yes | PASS — no Python code is added (YAML only), so Ruff scope, typing rules, and docstring rules are unaffected. Research D-007 deliberately rejected embedding a Python helper in the workflow. |
| VI. Extensibility & Discoverability | Yes | PASS — a newly added algorithm sub-package is picked up by the documentation build with no workflow change, preserving "adding an algorithm modifies no file outside its sub-package". |

**Root Class Registry impact**: none. No entry (`RLModel`, `TDModel`, `BaseConfig`, `HyperParamsBase`,
`HerculeConfig`, `EpochResult`, `Runner`, `Supervisor`) is read or modified.

**Constitution Impact for the PR description**: *No amendment required — this feature adds repository
infrastructure only and touches no root class.*

**Gate result — initial**: ✅ PASS, no violations, Complexity Tracking not required.

## Project Structure

### Documentation (this feature)

```text
specs/003-github-pages-docs/
├── plan.md              # This file
├── spec.md              # Phase -1 output (/speckit.specify)
├── research.md          # Phase 0 output — 9 decisions with rationale
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output — activation + verification + troubleshooting
├── contracts/
│   └── workflow-contract.md   # Phase 1 output — triggers, permissions, artifact contract
├── checklists/
│   └── requirements.md  # Spec quality checklist (16/16 pass)
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
.github/
└── workflows/
    └── docs.yml              # NEW — build + deploy jobs (the only new source file)

.gitignore                    # MODIFIED — ignore the generated docs output
README.md                     # MODIFIED — link to the published documentation
pyproject.toml                # UNCHANGED — [tool.pyscaf.documentation] stays the source of truth
pyscaf/documentation/         # UNCHANGED — gen-doc / serve-doc entry points
src/hercule/                  # UNCHANGED — the documented package
docs/                         # DELETED locally, then ignored (generated artefact)
```

**Structure Decision**: repository-infrastructure feature. It introduces the repository's first
`.github/workflows/` directory and otherwise only edits two root files. The Python source layout defined by
constitution principle IV is untouched, which is why no `src/`-side structure option applies here.

## Phase 0 — Research

Complete. See [research.md](./research.md). Nine decisions recorded (D-001 … D-009), all action versions
verified against the GitHub API on 2026-07-28, no open questions.

Key outcomes feeding the design:

- Pages **artifact** deployment, not a `gh-pages` branch (D-001).
- `configure-pages` and `.nojekyll` proven unnecessary against the real generated output (D-003).
- Full dependency install is mandatory because pdoc imports `torch` transitively; uv cache mitigates (D-004).
- Fork safety enforced by both an `if` guard and token scope (D-005).
- Pages activation is a one-time human decision, documented, not self-applied by CI (D-008).

## Phase 1 — Design & Contracts

Complete. Artifacts:

- [data-model.md](./data-model.md) — no application data model; documents the build/deploy entities, their
  fields, and the configuration keys the pipeline depends on.
- [contracts/workflow-contract.md](./contracts/workflow-contract.md) — the externally observable contract of the
  workflow: triggers, job graph, permissions, artifact shape, published URL, failure semantics.
- [quickstart.md](./quickstart.md) — one-time activation, end-to-end verification against each user story, and
  troubleshooting.

**Agent context update**: `.specify/scripts/powershell/update-agent-context.ps1 -AgentType cursor-agent` run at
the end of Phase 1.

**Gate result — post-design re-check**: ✅ PASS. The design adds one YAML file and two one-line edits; no
principle is engaged more strongly after design than before, and no root class is involved.

## Implementation Order

Derived from the user-story priorities in the spec; each step is independently verifiable.

1. **P1 (Stories 1–2)** — create `.github/workflows/docs.yml` with both jobs; activate Pages; merge to `main`;
   confirm the site is live and updates on the next commit.
2. **P2 (Story 4)** — verified by the same workflow: open a PR and confirm the build check runs and publishes
   nothing. No extra code.
3. **P2 (Story 3)** — verify `git ls-files docs` is empty, add `docs/` to `.gitignore`, delete the local
   directory, confirm a clean status after `uv run gen-doc`.
4. **P3 (Story 5)** — add the documentation link to `README.md`.

Steps 3 and 4 are independent of each other and of the workflow file; only step 1 must precede the live checks
in steps 2–4.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| First CI run installs torch from scratch | Slow initial build, possible runner disk/time pressure | `enable-cache: true` on setup-uv; SC-001 budget is 10 min (D-004) |
| Pages not activated before the first merge | `deploy` job fails on `main` | Activate **before** merging (quickstart step 1); failure is explicit, site unaffected |
| `uv.lock` drifts from `pyproject.toml` | `uv sync --frozen` fails the docs build | Intended — surfaces lockfile drift early rather than publishing from an unpinned resolve |
| `output_path` changed in `pyproject.toml` without updating the workflow | Empty publication | Guard step fails the build when the output directory is missing or empty (D-007) |
| A future module raises on import | pdoc build fails, blocking publication | Caught on the PR by Story 4 before it reaches `main` |

## Complexity Tracking

> Not applicable — Constitution Check passed with no violations.
