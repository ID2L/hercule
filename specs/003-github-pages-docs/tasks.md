---

description: "Task list for feature 003 — Publish API Documentation on GitHub Pages"
---

# Tasks: Publish API Documentation on GitHub Pages

**Status**: ✅ Implemented and verified on 2026-07-28 — all 27 tasks complete, site live at
<https://id2l.github.io/hercule/>. Verification evidence is recorded in `research.md` D-010.

**Input**: Design documents from `/specs/003-github-pages-docs/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/workflow-contract.md, quickstart.md

**Tests**: No automated test tasks. The feature adds no application code; the specification requested none, and
its acceptance criteria are observational (a workflow runs, a site is reachable, a working tree is clean). The
verification tasks below are the executable form of each story's *Independent Test*.

**Organization**: Tasks are grouped by user story so each can be implemented and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1…US5)
- Include exact file paths in descriptions

## Path Conventions

Repository-infrastructure feature. Paths are relative to the repository root `D:\code\hercule`:
`.github/workflows/` (new), `.gitignore`, `README.md`. No file under `src/` or `tests/` is touched.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the starting state matches what the plan assumed, before changing anything.

- [x] T001 Run `git ls-files docs` at the repository root and confirm it prints nothing — proves no tracked file would be masked by the later ignore rule (VR-002, research D-009). If it prints paths, STOP and revise the plan.
- [x] T002 [P] Run `gh api repos/ID2L/hercule/pages` and record the result (expect HTTP 404 "Not Found" — Pages not yet enabled), confirming the target URL will be `https://id2l.github.io/hercule/`.
- [x] T003 [P] Run `uv run gen-doc` locally and confirm it exits successfully and populates `docs/`, establishing the baseline the CI build must reproduce.
- [x] T004 Create the directory `.github/workflows/` at the repository root (first workflow directory in this repository).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared `build` job. Every user story except US3 and US5 depends on it.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

All tasks in this phase edit the same file, so none is parallelizable.

- [x] T005 Create `.github/workflows/docs.yml` with the workflow name, the three triggers (`push` on `main`, `pull_request` targeting `main`, `workflow_dispatch`), workflow-level `permissions: contents: read`, and `env: DOCS_DIR: docs` — matching `contracts/workflow-contract.md` §1 and §3 and `[tool.pyscaf.documentation].output_path` in `pyproject.toml`.
- [x] T006 Add the `build` job (`runs-on: ubuntu-latest`) to `.github/workflows/docs.yml` with steps: `actions/checkout@v7.0.1`, then `astral-sh/setup-uv@v9.0.0` with `enable-cache: true` and `python-version: "3.12"` (research D-002, D-004).
- [x] T007 Add the dependency and generation steps to the `build` job in `.github/workflows/docs.yml`: `uv sync --frozen` (VR-004, FR-010) followed by `uv run gen-doc` (FR-003 — same command as local).
- [x] T008 Add the guard step to the `build` job in `.github/workflows/docs.yml`: fail the build when `$DOCS_DIR` does not exist or contains no file, with an error message naming `[tool.pyscaf.documentation].output_path` (VR-001, research D-007).
- [x] T009 Add the artifact step to the `build` job in `.github/workflows/docs.yml`: `actions/upload-pages-artifact@v5.0.0` with `path: ${{ env.DOCS_DIR }}`, guarded by `if: github.event_name != 'pull_request'` (contract §2).

**Checkpoint**: the build job alone must be able to validate a pull request. US1 can now start.

---

## Phase 3: User Story 1 — Reader consults the API documentation online (Priority: P1) 🎯 MVP

**Goal**: A public, stable URL serves the API reference of `hercule`.

**Independent Test**: open the published URL in a private browsing window and confirm the index lists every
public module and that a class page renders its docstrings.

- [x] T010 [US1] Add the `deploy` job to `.github/workflows/docs.yml`: `needs: build`, `if: github.event_name != 'pull_request'`, `environment: github-pages` with its `url` bound to the deploy step output, job-scoped `permissions: pages: write` + `id-token: write`, `concurrency: { group: pages, cancel-in-progress: false }`, and the single step `actions/deploy-pages@v5.0.0` (contract §2–§4, research D-005, D-006).
- [x] T011 [US1] Activate GitHub Pages with the Actions source: `gh api -X POST repos/ID2L/hercule/pages -f build_type=workflow`, then confirm with `gh api repos/ID2L/hercule/pages --jq '{status, html_url, build_type}'` (quickstart §1, FR-013). Must be done **before** the first merge to `main`.
- [x] T012 [US1] Commit `.github/workflows/docs.yml`, merge to `main`, and follow the run with `gh run watch` until the `deploy` job succeeds.
- [x] T013 [US1] Verify the published site at `https://id2l.github.io/hercule/` in a private browsing window: the index redirects to the package page, all model sub-packages are listed (`td_models`, `simple_q_learning`, `simple_sarsa`, `deep_q_learning`, `dummy`) per SC-005, and a class page renders its docstrings.

**Checkpoint**: the MVP is delivered — the documentation is publicly readable.

---

## Phase 4: User Story 2 — Documentation stays current without manual action (Priority: P1)

**Goal**: Every merge to `main` republishes; a maintainer can also republish on demand.

**Independent Test**: change a docstring on `main`, wait for the publication, reload the page and see the new
text.

No new workflow code — the triggers built in Phase 2 and the deploy job from Phase 3 provide the behaviour.
These tasks verify the contract holds.

- [x] T014 [US2] Push a small docstring change to `main` (e.g. in `src/hercule/models/__init__.py`), then confirm the published page reflects it within 10 minutes of the merge (SC-001, FR-002).
- [x] T015 [US2] Trigger `gh workflow run docs.yml` with no commit and confirm a new successful deployment appears in `gh api repos/ID2L/hercule/deployments` (FR-004).
- [x] T016 [US2] Re-read the `concurrency` block in `.github/workflows/docs.yml` and confirm it is `group: pages` with `cancel-in-progress: false`, so an in-flight deployment completes and queued runs are superseded by the newest (FR-009, contract §4).

**Checkpoint**: publication is autonomous and deterministic.

---

## Phase 5: User Story 3 — Generated documentation disappears from the repository (Priority: P2)

**Goal**: `docs/` stops polluting the working tree and can never be committed by accident.

**Independent Test**: generate locally, then confirm `git status` reports nothing.

Independent of US4 and US5 — these three stories can be done in any order.

- [x] T017 [P] [US3] Add a `docs/` entry to `.gitignore`, grouped with the existing generated-output entries (`outputs/`, `dist/`), with a short comment identifying it as pdoc output (FR-005).
- [x] T018 [US3] Delete the local `docs/` directory (19 untracked generated HTML files — build artefacts, nothing to preserve; research D-009).
- [x] T019 [US3] Verify: run `uv run gen-doc`, then `git status --short` must report nothing about `docs/` (SC-003), and `git add -A && git status --short` must stage no generated file. Confirm `uv run serve-doc` still starts (FR-006).

**Checkpoint**: clean working tree, local preview intact.

---

## Phase 6: User Story 4 — Broken documentation is caught before merge (Priority: P2)

**Goal**: Pull requests validate the docs build and publish nothing.

**Independent Test**: open a PR that breaks generation, see a red check and an unchanged live site.

Behaviour comes from Phase 2 (the `pull_request` trigger) and Phase 3 (the deploy guard); these tasks prove it.

- [x] T020 [P] [US4] Open a pull request targeting `main` (`gh pr create --fill`) and confirm with `gh pr checks --watch` that the build check runs and passes (FR-007, quickstart §2).
- [x] T021 [US4] Confirm the pull request created no deployment: compare `gh api repos/ID2L/hercule/deployments --jq 'length'` before and after the run (FR-007, FR-011).
- [x] T022 [US4] Exercise the failure path: temporarily add `import nonexistent_module` at the top of `src/hercule/__init__.py`, push to the PR branch, confirm the check turns red **and** `https://id2l.github.io/hercule/` still serves the previous content (FR-008, SC-006), then revert the change.

**Checkpoint**: broken documentation cannot reach the published site.

---

## Phase 7: User Story 5 — Reader finds the documentation from the repository (Priority: P3)

**Goal**: One click from the repository home page to the documentation.

**Independent Test**: open the repository home page and follow the link.

- [x] T023 [P] [US5] Add a clearly labelled link to `https://id2l.github.io/hercule/` near the top of `README.md`, in the intro block above *Motivation* (FR-012, SC-007).

**Checkpoint**: the site is discoverable.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [x] T024 [P] Document the one-time Pages activation in the *Development* section of `README.md` (the `gh api -X POST … -f build_type=workflow` command plus the Settings → Pages fallback), so a fork can reproduce it in under 5 minutes (FR-013, SC-008).
- [x] T025 [P] Add a "Constitution Impact: no amendment required — repository infrastructure only, no Root Class Registry entry touched" section to the pull request description, as required by the constitution's governance rules.
- [x] T026 Walk the failure-semantics table in `contracts/workflow-contract.md` §6 and confirm each row matches observed behaviour; record any divergence in `research.md` as a new decision.
- [x] T027 [P] Update `CLAUDE.md` — add `uv run gen-doc` output being git-ignored and the published documentation URL to the commands section, so future sessions do not re-create `docs/` expectations.

---

## Dependencies

```text
Phase 1 (Setup: T001–T004)
      │
      ▼
Phase 2 (Foundational build job: T005–T009)  ── blocks US1, US2, US4
      │
      ├──► Phase 3  US1 (T010–T013)  ── P1, MVP; T011 must precede T012
      │        │
      │        └──► Phase 4  US2 (T014–T016)  ── P1, needs a live site to observe updates
      │
      └──► Phase 6  US4 (T020–T022)  ── P2, needs only the build job + deploy guard

Phase 5  US3 (T017–T019)  ── P2, independent of everything above
Phase 7  US5 (T023)       ── P3, needs the published URL to exist (T011) to be meaningful
Phase 8  Polish (T024–T027) ── last
```

**Story independence**: US3 depends on no workflow code at all and could ship alone. US5 only needs the URL to
be known. US2 and US4 add no code — they verify guarantees produced in Phases 2–3.

## Parallel Execution Opportunities

- **Phase 1**: T002 and T003 run in parallel (different systems: GitHub API vs local build). T001 first.
- **Phase 2**: none — T005–T009 all edit `.github/workflows/docs.yml` sequentially.
- **Across stories once Phase 2 is merged**: T017 (`.gitignore`), T020 (PR check), T023 (`README.md`) touch
  disjoint files and can proceed simultaneously.
- **Phase 8**: T024, T025, T027 are parallel (README / PR description / CLAUDE.md); T026 last.

## Implementation Strategy

**MVP scope**: Phases 1–3 (T001–T013). That alone delivers a publicly readable, automatically built API
reference — User Story 1 in full, and User Story 2 implicitly on the next merge.

**Incremental delivery**:

1. Ship the MVP, confirm the site is live.
2. Add US3 (`.gitignore` cleanup) — the visible payoff for contributors.
3. Confirm US4 on the next real pull request; no code needed.
4. Finish with US5 and polish.

**Rollback**: delete `.github/workflows/docs.yml`; the last published site keeps being served and no generated
file ever entered git history (quickstart §6).

## Task Summary

| Phase | Story | Priority | Tasks | Count |
|-------|-------|----------|-------|-------|
| 1 Setup | — | — | T001–T004 | 4 |
| 2 Foundational | — | — | T005–T009 | 5 |
| 3 | US1 | P1 | T010–T013 | 4 |
| 4 | US2 | P1 | T014–T016 | 3 |
| 5 | US3 | P2 | T017–T019 | 3 |
| 6 | US4 | P2 | T020–T022 | 3 |
| 7 | US5 | P3 | T023 | 1 |
| 8 Polish | — | — | T024–T027 | 4 |
| **Total** | | | | **27** |

**Files touched**: `.github/workflows/docs.yml` (new), `.gitignore`, `README.md`, `CLAUDE.md`, plus the deletion
of the local `docs/` directory. No file under `src/` or `tests/`.
