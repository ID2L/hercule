---
description: "Task list for feature 004 — Improved Experiment Report Generation"
---

# Tasks: Improved Experiment Report Generation

**Input**: Design documents from `specs/004-improve-report-generation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/reports-api.md, quickstart.md

**Tests**: Test tasks ARE included. The spec makes verification a success criterion (SC-004,
SC-005, SC-006, SC-009 are only checkable by running reports over the real result sets), and
`reports/` has **zero test coverage** today — `grep -rn -i report tests/` returns nothing.

**Organization**: Tasks are grouped by user story so each can be implemented and validated
independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task serves (US1–US5)
- Exact file paths are given in every task

## Path Conventions

Single project: `src/hercule/`, `tests/` at repository root (per plan.md Structure Decision).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: dependencies and test scaffolding

- [X] T001 Add main dependencies with `uv add "jupytext>=1.16,<2" "nbformat>=5.10,<6" "nbconvert>=7.16,<8"` and verify `pyproject.toml` `[project].dependencies` per research R10
- [X] T002 Move `ipykernel>=7.1,<8` from the `dev` group to main in `pyproject.toml` (`uv remove --group dev ipykernel && uv add "ipykernel>=7.1,<8"`) — `hercule report` executes notebooks, so the kernel is an end-user dependency
- [X] T003 [P] Add `uv add --group dev "pypdf>=5.1,<6"` — SC-006 is only verifiable by reading the PDF back
- [X] T004 [P] Add the optional extra with `uv add --optional pdf "playwright>=1.49,<2"`, and confirm `[tool.uv] default-groups = "all"` does not pull it (it governs groups, not extras)
- [X] T005 [P] Create the test package `tests/reports/__init__.py`
- [X] T006 [P] Add shared fixtures to `tests/reports/conftest.py`: a builder that writes a synthetic run tree (`env/env_sig/model/model_sig/{environment.json,model.json,run_info.json}`) with a parameterisable run count, model families, hyperparameter grid and episode counts
- [X] T007 Confirm `scikit-learn` is NOT added — PCA uses `numpy.linalg.svd` per research R4; record this in `pyproject.toml` review

**Checkpoint**: `uv sync` succeeds; `uv run python -c "import jupytext, nbformat, nbconvert"` works

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: fix what prevents a generated report from executing at all, and widen the return
contract. Nothing downstream — least of all the PDF — can work until this is done.

**⚠️ CRITICAL**: No user story work can begin until this phase completes

- [X] T008 Delete the dead unregistered Click group `src/hercule/reports/cli.py` and the dead script `src/hercule/reports/example_usage.py` (never reachable; see contracts C3)
- [X] T009 Delete `create_learning_plots` from `src/hercule/reports/__init__.py` (lines ~508-608) — never called, duplicates the templates' plots, and carries the removed `boxplot(labels=)` kwarg; remove the now-unused `base64` and `io` imports
- [X] T010 Delete `_format_python_value` and the `env.filters["topython"]` registrations at `src/hercule/reports/__init__.py:325` and `:472` — registered but used by neither template
- [X] T011 Fix `src/hercule/reports/__init__.py:211`: `load_environment(self.experiment_path)` passes a directory to a function that opens a file, so the error is swallowed at `:212` and `env` is always `None`. Pass `experiment_path / environment_file_name`
- [X] T012 [P] Define the result models per data-model.md: `ReportArtifacts`, `ReportBundle`, `SkippedGroup` (Pydantic v2, `X | Y` unions). Placed in `src/hercule/reports/__init__.py` rather than `controller/__init__.py` — see the note below the checkpoint for why.
- [X] T013 Change `generate_experiment_report()` in `src/hercule/controller/__init__.py:172` to return `ReportBundle` and accept `execute`, `render_pdf` and `progress` keyword arguments per contracts C2
- [X] T014 Narrow the `except Exception → ValueError` wrapper at `src/hercule/controller/__init__.py:202-204` so `FileNotFoundError` propagates as the docstring already promises
- [X] T015 Update the `report` command in `src/hercule/cli/main.py:185-234` to consume `ReportBundle`, list every artifact and every skipped group, and add the `--no-pdf` and `--no-execute` options per contracts C1
- [X] T016 Make the `report` command exit non-zero when no notebook could be produced, while keeping exit 0 when only the PDF was skipped (FR-026)
- [X] T017 Establish the cell-tag vocabulary as module constants in `src/hercule/reports/__init__.py` (`remove_cell`, `remove_input`, `remove_output`) and pass them into the Jinja context so templates and `render.py` cannot drift (contracts C4)
- [X] T018 [P] Add `tests/reports/test_controller_contract.py`: a missing path raises `FileNotFoundError` (not `ValueError`); a non-directory raises `ValueError`

**Implementation note on T012's file placement**: data-model.md's "Module placement" table assigns
`ReportArtifacts`/`SkippedGroup`/`ReportBundle` to `src/hercule/reports/__init__.py`, while this task's
text names `controller/__init__.py`. The two documents disagree; `reports/__init__.py` was chosen because
Constitution IV (Module Separation) and plan.md's stated dependency direction are unambiguous:
`controller` imports `reports`, never the reverse, and `generate_report()` (which lives in `reports/`)
must itself return a `ReportBundle`. Defining the models in `controller/` would force `reports/` to
import from its own caller. `controller/__init__.py` imports `ReportBundle` from `hercule.reports` under
`TYPE_CHECKING` (annotation-only use, consistent with the existing `Path` pattern) and uses it as the
return type of `generate_experiment_report()`.

**Checkpoint**: `uv run hercule report --help` shows the new options; `uv run pytest tests/reports` passes; existing 50 tests still pass

---

## Phase 3: User Story 2 - One data-loading loop (Priority: P1) 🎯 MVP

**Goal**: replace the per-run emitted blocks with one directory walk into one table. This is the
technical enabler for US1, US3 and US4, so it is implemented first despite sharing P1.

**Independent Test**: generate a report for `outputs/frozenlake_4x4` (135 runs) and confirm the
generated document contains exactly one loading loop, its length is within 10% of a 2-run
report, and the table has one row per run.

### Tests for User Story 2

- [X] T019 [P] [US2] `tests/reports/test_run_table.py`: `build_run_table` returns one `RunRecord` per run over the synthetic tree, with `model_name` taken from the directory and hyperparameters from `run_info.json["model_hyperparameters"]`
- [X] T020 [P] [US2] `tests/reports/test_run_table.py`: a run whose `model.json` is deliberately corrupt still loads successfully — proves `model.json` is never opened (FR-007, SC-010)
- [X] T021 [P] [US2] `tests/reports/test_run_table.py`: a run with missing or malformed `run_info.json` becomes a `SkippedRun` with a reason, and the remaining runs still load (FR-008)
- [X] T022 [P] [US2] `tests/reports/test_run_table.py`: aggregate derivations — `mean_learning_reward`, `learning_success_rate`, `mean_testing_reward`, `testing_success_rate`, `episode_count`; a run with an empty testing phase yields `None`, not `0.0`
- [X] T023 [P] [US2] `tests/reports/test_run_table.py`: `to_dataframe()` column schema matches data-model.md, with `hp_`/`env_` prefixes and nullable columns for hyperparameters absent from a model family

### Implementation for User Story 2

- [X] T024 [US2] Create `src/hercule/reports/run_table.py` with the `RunRecord` Pydantic model per data-model.md, reading **only** `environment.json` and `run_info.json`
- [X] T025 [US2] Implement `SkippedRun` and `build_run_table(root) -> RunTable` in `src/hercule/reports/run_table.py`, walking via the existing `find_experiment_directories`
- [X] T026 [US2] Implement the derived aggregates as `@computed_field`/`cached_property` on `RunRecord`, typed `float | None` so "scored 0" stays distinct from "no evaluation phase"
- [X] T027 [US2] Implement `RunTable.to_dataframe()` in `src/hercule/reports/run_table.py`, holding scalars only — per-episode lists stay on the records to avoid multi-hundred-MB object columns
- [X] T028 [US2] Export `build_run_table`, `RunRecord`, `RunTable` from `src/hercule/reports/__init__.py` — the generated notebook imports them, so they are a public end-user contract (contracts C3)
- [X] T029 [US2] Add the `_locate_report_dir()` helper to both templates in `src/hercule/reports/templates/`, implementing the verifying candidate search of research R9; `__file__` is undefined in a kernel so bare `Path(__file__)` must go
- [X] T030 [US2] Write `report_manifest.json` beside each comparative report in `src/hercule/reports/__init__.py` as the anchor file the directory search verifies against (contracts C5)
- [X] T031 [US2] Rewrite `src/hercule/reports/templates/comparative_report_template.py.j2`: delete the `{% for exp in experiments %}` loop at lines 48-85 and replace it with a single tagged loading cell calling `build_run_table`, printing "loaded N runs, skipped M"
- [X] T032 [US2] Replace `ExperimentData` usage in `src/hercule/reports/__init__.py:405-470` with `build_run_table`, removing the per-run `load_data()` call at `:442` whose data was discarded
- [X] T033 [US2] Delete the `ExperimentData` class from `src/hercule/reports/__init__.py` (superseded per contracts C3)
- [X] T034 [US2] Fix `generate_report` in `src/hercule/reports/__init__.py` to honour or explicitly reject `output_path` for comparative reports (currently ignored at `:434`) and collapse the identical if/else return at `:501-505`
- [X] T035 [US2] Add `tests/reports/test_report_generation.py`: generated line count for a 2-run group vs a 50-run group differs by less than 10% (SC-001)
- [X] T036 [US2] Run `uv run ruff check . --fix && uv run ruff format .` and confirm clean

**Checkpoint**: `uv run hercule report outputs/simple_games --no-pdf` produces 2 notebooks whose loading cell is a single loop

---

## Phase 4: User Story 1 - Identify the environment at a glance (Priority: P1)

**Goal**: every report names its Gymnasium environment and settings in readable prose.

**Independent Test**: generate a report for any group and confirm the environment identifier and
its settings appear as text within the first screen, without reading a raw config block.

### Tests for User Story 1

- [X] T037 [P] [US1] `tests/reports/test_report_generation.py`: the generated comparative document contains the literal environment id (`FrozenLake-v1`) and each setting as `name=value` text
- [X] T038 [P] [US1] `tests/reports/test_report_generation.py`: a group whose environment has empty `kwargs` produces an explicit "no environment-specific setting was overridden" statement rather than an empty structure (FR-002 scenario 2)

### Implementation for User Story 1

- [X] T039 [US1] Add an environment-summary helper to `src/hercule/reports/run_table.py` that formats `env_id`, `env_kwargs` and `max_episode_steps` as prose, handling the empty-kwargs case
- [X] T040 [US1] Add the title and environment cells to `src/hercule/reports/templates/comparative_report_template.py.j2` so the environment is named in the markdown title and restated in the environment section (FR-001, FR-002)
- [X] T041 [US1] Add the same environment statement to `src/hercule/reports/templates/report_template.py.j2` in identical form (FR-003)
- [X] T042 [US1] Route the notebook's environment introspection through `load_environment(dir / "environment.json")` and `EnvironmentInspector`, never `gym.make` (Constitution III)
- [X] T043 [US1] Run `uv run ruff check . --fix && uv run ruff format .` and confirm clean

**Checkpoint**: every generated report states its environment in prose

---

## Phase 5: User Story 3 - Comparison charts stay readable (Priority: P1)

**Goal**: cap every multi-run chart at 9 ranked series — best 3, median 3, worst 3.

**Independent Test**: generate the comparative report for `outputs/frozenlake_4x4` (135 runs)
and confirm no chart draws more than 9 curves, that the legend names each bucket, and that the
document states how many runs were omitted.

### Tests for User Story 3

- [X] T044 [P] [US3] `tests/reports/test_selection.py`: 135 records yield exactly 9 selected with correct bucket labels and `omitted_count == 126`
- [X] T045 [P] [US3] `tests/reports/test_selection.py`: 5 records yield all 5 and `omitted_count == 0` (FR-011)
- [X] T046 [P] [US3] `tests/reports/test_selection.py`: all-equal metric values produce a stable, repeatable order across repeated calls — the `(-metric, directory_name)` tie-break (FR-014, SC-009)
- [X] T047 [P] [US3] `tests/reports/test_selection.py`: bucket overlap is de-duplicated for counts between 4 and 9, never returning the same record twice
- [X] T048 [P] [US3] `tests/reports/test_selection.py`: ranking on two different metrics selects different subsets (FR-012)

### Implementation for User Story 3

- [X] T049 [US3] Create `src/hercule/reports/selection.py` with `SeriesBucket` (`str, Enum` — `StrEnum` is 3.11+), `SelectedSeries` and `SeriesSelection` per data-model.md
- [X] T050 [US3] Implement `select_series(records, metric, per_bucket=3)` in `src/hercule/reports/selection.py`, sorting on `(-metric_value, directory_name)` for determinism and de-duplicating overlapping buckets
- [X] T051 [US3] Export `select_series` and the selection models from `src/hercule/reports/__init__.py` (public — the notebook imports them)
- [X] T052 [US3] Replace the rewards comparison chart in `src/hercule/reports/templates/comparative_report_template.py.j2` (currently 2N curves at line ~235) with a `select_series`-driven chart ranked on mean learning reward
- [X] T053 [US3] Replace the success-rate comparison chart (currently 2N curves at line ~256) with a `select_series`-driven chart ranked on learning success rate
- [X] T054 [US3] Plot each selected series against its own index range rather than a shared x-array, so runs with different episode counts render correctly (FR-015)
- [X] T055 [US3] Cap the evaluation boxplot and bar charts (lines ~288, ~304, ~314) at the same 9 selections, ranked on the evaluation metric each presents
- [X] T056 [US3] Add the omitted-run count and bucket legend text to every capped chart (FR-013)
- [X] T057 [US3] Fix `boxplot(labels=...)` → `tick_labels=` in `src/hercule/reports/templates/report_template.py.j2:274,281` — renamed in matplotlib 3.9; verified deprecated-but-accepted on the installed 3.10.8 (warning only), dropped in 3.11, and the pin allows `<4.0.0`
- [X] T058 [US3] Restructure `src/hercule/reports/templates/report_template.py.j2:236-273` so `# %%` markers are no longer emitted inside the indented `if testing_rewards:` block; use unconditional cells with in-cell guards
- [X] T059 [US3] Add `tests/reports/test_report_generation.py` assertion: no generated chart call receives more than 9 series (SC-002)
- [X] T060 [US3] Run `uv run ruff check . --fix && uv run ruff format .` and confirm clean

**Checkpoint**: `uv run hercule report outputs/frozenlake_4x4 --no-pdf` yields readable charts

---

## Phase 6: User Story 4 - Hyperparameter PCA (Priority: P2)

**Goal**: a per-model-family principal-component projection of the hyperparameter grid, with
explained variance and loadings.

**Independent Test**: generate the report for `outputs/frozenlake_4x4` and confirm a
two-component projection, an explained-variance figure and per-hyperparameter contributions
appear, with performance encoded on the projection.

### Tests for User Story 4

- [X] T061 [P] [US4] `tests/reports/test_pca.py`: standardisation uses the correlation matrix with `ddof=1`, and loadings equal Pearson correlations between each column and each score to within 1e-10
- [X] T062 [P] [US4] `tests/reports/test_pca.py`: zero-variance columns are dropped before the SVD and reported in `dropped_columns`; no `nan` or `inf` appears in the result
- [X] T063 [P] [US4] `tests/reports/test_pca.py`: sign pinning is deterministic — repeated calls and a row-permuted input give identical loadings and scores (FR-014)
- [X] T064 [P] [US4] `tests/reports/test_pca.py`: `explained_variance_ratio` sums to 1 and is ordered descending
- [X] T065 [P] [US4] `tests/reports/test_pca.py`: degenerate shapes return `PcaUnavailable` with a reason and never raise — `p_kept == 0`, `p_kept == 1`, `n_samples < 3` (FR-021)
- [X] T066 [P] [US4] `tests/reports/test_pca.py`: `bool` hyperparameters are excluded despite `isinstance(True, int)` being `True`, and non-numeric values are excluded with a stated reason (FR-022)
- [X] T067 [P] [US4] `tests/reports/test_pca.py`: components are truncated to `min(n - 1, p_kept)` so no zero-variance trailing component is returned

### Implementation for User Story 4

- [X] T068 [US4] Create `src/hercule/reports/pca.py` with `PcaResult` and `PcaUnavailable` per data-model.md
- [X] T069 [US4] Implement `hyperparameter_pca(records)` in `src/hercule/reports/pca.py` following research R4 exactly: alphabetical column order, numeric-only with `bool` excluded, `ddof=1` standardisation, exact-range drop (`np.ptp(X, axis=0) > 1e-12 * max(abs(X).max(axis=0), 1.0)`, not `sd > 0` — a genuinely constant column's `ddof=1` std can be a tiny nonzero float due to rounding error), `np.linalg.svd`, sign pinned on the largest-magnitude entry of each `Vt` row and applied to both `Vt` and `U`, truncation to `min(n - 1, p_kept)`
- [X] T070 [US4] Compute and return `scores`, `explained_variance_ratio`, `loadings` and `communalities` in `src/hercule/reports/pca.py`
- [X] T071 [US4] Group records by `model_name` so families with different hyperparameter sets are never mixed (FR-020)
- [X] T072 [US4] Export `hyperparameter_pca` and its models from `src/hercule/reports/__init__.py` (public — the notebook imports them)
- [X] T073 [US4] Add the PCA section to `src/hercule/reports/templates/comparative_report_template.py.j2`: scatter of `scores[:, :2]` with `c=performance`, `cmap="viridis"`, `edgecolors="black"`, `aspect="equal"`, axis labels carrying the variance percentage
- [X] T074 [US4] Guard the constant-performance case in the template: when `np.ptp(finite) == 0`, draw a single fixed colour with no colorbar and annotate the shared score — otherwise matplotlib maps every point to the darkest colour and renders fabricated colorbar ticks (research R5)
- [X] T075 [US4] Filter `nan`/`inf` from the performance array before computing `vmin`/`vmax`, so one unfinished run cannot collapse the scale
- [X] T076 [US4] Print the explained-variance ratio prominently and add the caveat that full cartesian grids make the correlation matrix near-identity, so PC1+PC2 typically capture only ~2/p of the variance (research R4)
- [X] T077 [US4] Render the loadings and communalities as a labelled pandas table, and render `PcaUnavailable` as text via `if/elif/else` — never `raise`, since an exception in a middle cell blocks every cell below it
- [X] T078 [US4] Run `uv run ruff check . --fix && uv run ruff format .` and confirm clean

**Checkpoint**: the FrozenLake report shows a PCA per model family; the `dummy` family (seed only) reports "not applicable"

---

## Phase 7: User Story 5 - PDF alongside the notebook (Priority: P2)

**Goal**: execute the report and print it to PDF with mechanical cells stripped, degrading
gracefully when no browser is available.

**Independent Test**: generate a report and confirm a PDF appears beside the notebook containing
every chart and table but no import or loading code.

### Tests for User Story 5

- [X] T079 [P] [US5] `tests/reports/test_render.py`: round-trip a generated template through `jupytext.read(fmt="py:percent")` and assert `# %% tags=["remove_input"]` parses to `{"tags": ["remove_input"]}`
- [X] T080 [P] [US5] `tests/reports/test_render.py`: `TagRemovePreprocessor` drops a `remove_input` cell's source while keeping its output, and drops a `remove_cell` cell entirely
- [X] T081 [P] [US5] `tests/reports/test_render.py`: with browser discovery stubbed to find nothing and `playwright` absent, `RenderResult.pdf is None`, `pdf_skip_reason` is set, and no exception is raised (FR-026)
- [X] T082 [P] [US5] `tests/reports/test_render.py`: a cell-execution failure writes `<name>.failed.ipynb` and returns a sanitised reason with ANSI escapes stripped
- [X] T083 [P] [US5] `tests/reports/test_render.py` marked `slow`: a generated PDF read back with `pypdf` has at least one page, extracted text contains a known table heading, and contains no `import matplotlib` (SC-006)

### Implementation for User Story 5

- [X] T084 [US5] Create `src/hercule/reports/render.py` with `RenderResult` per data-model.md
- [X] T085 [US5] Implement notebook conversion and execution in `src/hercule/reports/render.py`: `jupytext.read(fmt="py:percent")` then `ExecutePreprocessor(timeout=1800, startup_timeout=120, interrupt_on_timeout=True, allow_errors=False, record_timing=False)`, passing `resources={"metadata": {"path": str(report_dir)}}` — the default `timeout=None` is unbounded and would hang forever
- [X] T086 [US5] Wire the `on_cell_start` progress hook in `src/hercule/reports/render.py` — these are traitlets `Callable` traits invoked with keyword arguments, so pass them to the constructor and accept `cell=None, cell_index=None, **_` (SC-008)
- [X] T087 [US5] Catch `CellExecutionError`, `CellTimeoutError` and `DeadKernelError` separately in `src/hercule/reports/render.py` — they are unrelated types, `CellTimeoutError` deriving from `TimeoutError`/`OSError`
- [X] T088 [US5] Write the partial notebook on failure in `src/hercule/reports/render.py` — nbconvert does not save the notebook when it raises, so the traceback is otherwise lost
- [X] T089 [US5] Implement the sanitiser in `src/hercule/reports/render.py`: strip ANSI with `re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", ...)` then coerce to ASCII, since exception text carries arbitrary user characters (research R11)
- [X] T090 [US5] Configure `HTMLExporter` in `src/hercule/reports/render.py` with `c.TagRemovePreprocessor.enabled = True` (**the default is `False`**), the three tag traits, `c.HTMLExporter.preprocessors`, and `exclude_input_prompt`/`exclude_output_prompt`
- [X] T091 [US5] Write the HTML with `encoding="utf-8"` explicitly in `src/hercule/reports/render.py` — the 308 KB output contains non-cp1252 characters and `write_text` defaults to the locale encoding on Windows
- [X] T092 [US5] Inject print CSS (`@page { size: A4; margin: 14mm }`, `.jp-Cell { break-inside: avoid }`, `img { max-width: 100% }`) before printing so charts do not straddle pages
- [X] T093 [US5] Implement system-browser discovery in `src/hercule/reports/render.py` via `shutil.which` for `msedge`/`chrome`/`chromium` plus the standard install paths
- [X] T094 [US5] Implement the print invocation in `src/hercule/reports/render.py` with `--headless=new --disable-gpu --no-sandbox --user-data-dir=<fresh temp> --no-pdf-header-footer --virtual-time-budget=10000 --host-resolver-rules="MAP * ~NOTFOUND" --print-to-pdf=<out>` and `html.as_uri()`
- [X] T095 [US5] Verify PDF success by `pdf.exists() and pdf.stat().st_size > 0`, **not** the return code — a running browser instance makes a bare `--print-to-pdf` exit 0 having written nothing (research R11)
- [X] T096 [US5] Add the `WebPDFExporter` fallback in `src/hercule/reports/render.py`, gated on `importlib.util.find_spec("playwright")`, with `c.WebPDFExporter.preprocessors` set separately since traitlets config is per-class
- [X] T097 [US5] Write intermediates to a short temp dir and move the finished PDF into place, since the output tree approaches MAX_PATH before a filename is appended
- [X] T098 [US5] Suppress the benign Proactor `RuntimeWarning` narrowly with `warnings.catch_warnings()` around `preprocess` — do **not** change the global asyncio event loop policy, which would break the Playwright fallback and side-effect any embedding host
- [X] T099 [US5] Add the `Agg` backend guard to both templates (`if "get_ipython" not in globals(): matplotlib.use("Agg")`) so `python report.py` does not block on `plt.show()`
- [X] T100 [US5] Tag every cell in both templates per contracts C4: `remove_cell` for imports and model reconstruction, `remove_input` for loading and summary cells, untagged for charts and analysis
- [X] T101 [US5] Call `render_report` from `generate_report` in `src/hercule/reports/__init__.py`, honouring the `execute` and `render_pdf` flags, and populate `ReportArtifacts`
- [X] T102 [US5] Run `uv run ruff check . --fix && uv run ruff format .` and confirm clean

**Checkpoint**: `uv run hercule report outputs/simple_games` produces notebook, HTML and PDF per group

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T103 Rewrite `src/hercule/reports/README.md` — it documents the removed `generate_report`-only API and no longer matches the module
- [X] T104 Update the `reports/` architecture section of `CLAUDE.md` (lines ~127-131) to describe the run table, capped charts, PCA and PDF pipeline, and add the new gotchas: `Path(__file__)` is undefined in a kernel, `TagRemovePreprocessor.enabled` defaults to `False`, `--user-data-dir` is mandatory, PDF success is checked by file size not return code
- [X] T105 [P] Update `AGENTS.md` where it describes report generation, keeping it consistent with `CLAUDE.md`
- [X] T106 [P] Add a "Constitution Impact" note to the PR description: the Root Class Registry is untouched and **no amendment is required**, but `controller.generate_experiment_report()` changes its public return type from `Path` to `ReportBundle`
- [X] T107 Verify `uv run gen-doc` still succeeds — a PR that breaks it fails the docs check, so every module must stay importable
- [X] T108 Run the full validation from quickstart.md over `outputs/simple_games` (2 groups × 11 runs) and confirm SC-002, SC-003, SC-006, SC-007, SC-010
- [X] T109 Run the full validation over `outputs/frozenlake_4x4` (135 runs, 2 model families) and confirm SC-001, SC-002 and the PCA acceptance scenarios
- [X] T110 Run the full validation over `outputs/dq_cartpole` (218 runs, 2 groups, ~211 MB) and time it against SC-008's 10-minute bound with progress at least every 30 s
- [X] T111 Confirm SC-004: the three invocations produce **5** comparative reports total, with every skipped group explained
- [X] T112 Confirm SC-009 by regenerating one report twice and diffing the selected series
- [X] T113 Confirm SC-010 by grepping every generated artifact for `q_network_state_dict` and `q_table` — both must be absent
- [X] T114 Run `uv run pytest` in full and confirm the pre-existing 50 tests still pass alongside the new `tests/reports` suite
- [X] T115 Final `uv run ruff check . && uv run ruff format --check .` — the repo must stay Ruff-clean; note ruff also reformats Python snippets inside `.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup — **blocks every user story**
- **US2 (Phase 3)**: depends on Foundational
- **US1 (Phase 4)**, **US3 (Phase 5)**, **US4 (Phase 6)**: depend on US2's run table
- **US5 (Phase 7)**: depends on Foundational; the tag vocabulary (T017) and the templates
  existing in final shape mean it lands cleanest after US1/US3/US4
- **Polish (Phase 8)**: depends on all desired stories

### User Story Dependencies — deliberate deviation from priority order

US1, US2 and US3 are all P1 in the spec. They are sequenced **US2 → US1 → US3** because US1 and
US3 both read from the run table US2 builds; implementing them first would mean writing
throwaway per-run loading code. US2 is therefore the MVP.

- **US2 (P1)**: independent once Foundational is done — the enabler
- **US1 (P1)**: needs `RunRecord.env_id`/`env_kwargs` from US2
- **US3 (P1)**: needs the aggregates from US2 for ranking
- **US4 (P2)**: needs `RunRecord.hyperparameters` and `model_name` from US2
- **US5 (P2)**: needs the tag vocabulary from Foundational; independently testable against a
  hand-written tagged notebook, so it can be developed in parallel with US1/US3/US4

### Within Each User Story

Tests are written first and must fail before implementation. Models before helpers, helpers
before templates, templates before orchestration. Lint at the end of every phase.

### Parallel Opportunities

- T003–T006 in Setup
- T012 and T018 in Foundational
- All test tasks within a story (T019–T023, T037–T038, T044–T048, T061–T067, T079–T083)
- US5's `render.py` work can proceed in parallel with US1/US3/US4 against a fixture notebook
- T105 and T106 in Polish

---

## Parallel Example: User Story 4

```bash
# All PCA tests can be written together — one file, independent cases:
Task: "Correlation-matrix standardisation and loadings equal Pearson correlations"
Task: "Zero-variance columns dropped before the SVD, reported in dropped_columns"
Task: "Sign pinning deterministic under repetition and row permutation"
Task: "explained_variance_ratio sums to 1, ordered descending"
Task: "Degenerate shapes return PcaUnavailable, never raise"
Task: "bool excluded despite isinstance(True, int); non-numeric excluded with a reason"
Task: "Components truncated to min(n - 1, p_kept)"
```

---

## Implementation Strategy

### MVP First (US2 only)

1. Phase 1 Setup
2. Phase 2 Foundational — **critical**, and it alone makes the templates executable for the
   first time
3. Phase 3 US2
4. **STOP and VALIDATE**: `uv run hercule report outputs/simple_games --no-pdf`, confirm one
   loading loop and SC-001
5. This is already shippable: reports become generatable at 135-run scale, which they are not
   today

### Incremental Delivery

1. Setup + Foundational → templates can execute at all
2. US2 → the run table (MVP)
3. US1 → environment named in prose
4. US3 → readable charts
5. US4 → hyperparameter PCA
6. US5 → PDF distribution
7. Polish → docs, full-corpus validation, lint

### Notes

- `[P]` tasks touch different files with no incomplete dependencies
- Lint at the close of every phase, not only at the end (`uv run ruff check . --fix && uv run ruff format .`)
- Commit per task or logical group
- `reports/` has no tests today, so every new test is net-new coverage on a previously unguarded
  module
- Validation against `outputs/` is the acceptance evidence the deliverable calls for; it is not
  optional polish
