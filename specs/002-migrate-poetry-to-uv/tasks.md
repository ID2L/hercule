# Tasks: Migrate from Poetry to uv

**Input**: Design documents from `/specs/002-migrate-poetry-to-uv/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not explicitly requested in spec. Verification via quickstart.md commands.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)

---

## Phase 1: Setup

**Purpose**: Prepare migration environment

- [x] T001 Run `uvx migrate-to-uv --dry-run` to preview changes in pyproject.toml
- [x] T002 Run `uvx migrate-to-uv` to convert pyproject.toml and generate uv.lock in D:\code\hercule

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core migration steps that MUST complete before user story verification

- [x] T003 Add `package_paths = ["src/hercule"]` to `[tool.pyscaf.documentation]` in pyproject.toml
- [x] T004 [US1] Update `get_poetry_package_paths` to `get_package_paths` in pyscaf/documentation/scripts/parse_doc.py to read from `tool.pyscaf.documentation.package_paths`
- [x] T005 Remove poetry.toml from D:\code\hercule
- [x] T006 Remove poetry.lock if still present (migrate-to-uv typically removes it)

---

## Phase 3: User Story 1 - Developer installs project dependencies with uv (Priority: P1)

**Goal**: Developer can run `uv sync` and get a working environment with all dependencies.

**Independent Test**: `uv sync` then `uv run pytest` — all tests pass.

- [x] T007 [US1] Run `uv sync` and verify virtual environment created in .venv/
- [x] T008 [US1] Run `uv run pytest` and verify all tests pass

---

## Phase 4: User Story 2 - Developer runs CLI entry points with uv (Priority: P1)

**Goal**: hercule, gen-doc, serve-doc execute correctly via `uv run`.

**Independent Test**: `uv run hercule --help`, `uv run gen-doc`, `uv run serve-doc` succeed.

- [x] T009 [US2] Verify `uv run hercule --help` executes in D:\code\hercule
- [x] T010 [US2] Verify `uv run gen-doc` executes in D:\code\hercule
- [x] T011 [US2] Verify `uv run serve-doc` executes in D:\code\hercule

---

## Phase 5: User Story 3 - Developer reads updated documentation (Priority: P2)

**Goal**: README contains uv instructions; no Poetry references.

**Independent Test**: README search for "poetry" returns zero user-facing hits.

- [x] T012 [US3] Replace Poetry Integration section with uv Integration in README.md
- [x] T013 [US3] Update Documentation scripts section: `poetry run gen-doc` → `uv run gen-doc`, `poetry run serve-doc` → `uv run serve-doc` in README.md
- [x] T014 [US3] Update Development section: Poetry → uv commands in README.md

---

## Phase 6: User Story 4 - Developer adds or removes a dependency (Priority: P2)

**Goal**: README documents `uv add` and `uv remove` workflows.

**Independent Test**: Run `uv add --dev black` then `uv remove black` — pyproject.toml and uv.lock update correctly.

- [x] T015 [US4] Add uv add/remove command examples to README.md
- [x] T016 [US4] Smoke test: `uv add --dev black` then `uv remove black` in D:\code\hercule

---

## Phase 7: User Story 5 - Build system produces installable package (Priority: P3)

**Goal**: `uv build` produces wheel/sdist; package installs and hercule CLI works.

**Independent Test**: `uv build` creates dist/; `pip install dist/*.whl` in fresh venv; `hercule --help` works.

- [x] T017 [US5] Run `uv build` and verify dist/ contains wheel and/or sdist
- [x] T018 [US5] Verify built package installs and hercule CLI is available

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and validation

- [x] T019 [P] Verify .gitignore: uv.lock is NOT ignored (should be committed)
- [ ] T020 Run full quickstart.md validation in D:\code\hercule
- [x] T021 [P] Update .cursor/rules/specify-rules.mdc or AGENTS.md if uv should be in Active Technologies

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies
- **Phase 2 (Foundational)**: Depends on Phase 1 — migrate-to-uv must run first
- **Phase 3 (US1)**: Depends on Phase 2 — parse_doc.py and package_paths must be updated
- **Phase 4 (US2)**: Depends on Phase 3 — uv sync must succeed first
- **Phase 5 (US3)**: Can run after Phase 2 (README update independent of sync)
- **Phase 6 (US4)**: Depends on Phase 3
- **Phase 7 (US5)**: Depends on Phase 3
- **Phase 8 (Polish)**: Depends on Phases 3–7 completion

### Parallel Opportunities

- T009, T010, T011 (US2 verification) can run in sequence but are independent checks
- T012, T013, T014 (US3) can be done in one README edit
- T019, T021 (Polish) marked [P] — different files

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 (uv sync, pytest)
4. Complete Phase 4: US2 (CLI verification)
5. **STOP and VALIDATE**: Developer can install and run the project

### Full Delivery

6. Phase 5: US3 (README)
7. Phase 6: US4 (add/remove docs + smoke test)
8. Phase 7: US5 (build)
9. Phase 8: Polish
