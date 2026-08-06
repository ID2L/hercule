# Implementation Plan: Improved Experiment Report Generation

**Branch**: `004-improve-report-generation` | **Date**: 2026-07-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/004-improve-report-generation/spec.md`

## Summary

Restructure `hercule report` so a generated report is a small, data-driven document instead of
a per-run code dump: it discovers its own runs by walking the directory tree, loads them into
one pandas table, states its Gymnasium environment in prose, caps every multi-run chart at 9
ranked series (best/median/worst triples), adds a numpy-SVD PCA of the hyperparameter grid per
model family, and is then executed and printed to PDF with mechanical cells stripped.

Technically: two rewritten Jinja templates whose size no longer scales with run count; new
`reports/` submodules holding the pure, unit-testable table/selection/PCA helpers the templates
call; a new `reports/render.py` owning the jupytext → execute → HTML → browser-print pipeline;
and a widened return contract from `generate_report` through
`controller.generate_experiment_report` to the CLI so every artifact (and any PDF skip reason)
is reported. Five pre-existing defects that prevent a generated report from executing at all
are fixed as part of the work, because a PDF cannot be produced from a report that cannot run.

## Technical Context

**Language/Version**: Python 3.10+ (`X | Y` unions, no `typing.Any`)
**Primary Dependencies**: existing — numpy, pandas, matplotlib, jinja2, click, pydantic; added
— jupytext, nbformat, nbconvert, ipykernel (moved dev → main); dev — pypdf; optional extra
`pdf` — playwright
**Storage**: read-only over the existing `outputs/` JSON tree; no stored format changes
**Testing**: pytest with `--strict-markers`; markers `unit`, `integration`, `slow`
**Target Platform**: Windows 11 primary dev, cross-platform library; no system dependency added
**Project Type**: single Python package (`src/hercule`) with a Click CLI
**Performance Goals**: SC-008 — largest result set (`dq_cartpole`, 218 runs / ~211 MB across 2
groups) generates within 10 minutes with progress output at least every 30 s
**Constraints**: ruff-clean at line-length 120; generated notebooks must execute unattended and
also open cleanly in Jupyter; PDF must degrade gracefully when no browser is available (FR-026)
**Scale/Scope**: 3 result sets, 375 runs, **5 comparative groups**; largest single report group
135 runs (`frozenlake_4x4`); largest invocation 218 runs (`dq_cartpole`, 2 groups)

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Verdict | Evidence |
|---|---|---|
| **I. Generic Algorithm Architecture** | Not engaged | No model class added or modified. `reports/` only reads persisted results. |
| **II. Configuration-Driven Design** | Not engaged | No YAML surface change; `ParameterValue` untouched. Hyperparameters read from `run_info.json`, not re-derived (R2). |
| **III. Gymnasium-First Integration** | Compliant | Environment metadata read from the persisted `environment.json` that `save_environment` wrote from `env.spec`, never duplicated in config. Where a generated report instantiates an env it goes through `load_environment`, never `gym.make`. |
| **IV. Module Separation** | Compliant | New code lands in `reports/` (report generation) plus a widened return type in `controller/`. Dependency direction stays top → bottom: `reports` imports `config`/`environnements`/`models`/`run`/`supervisor`; `controller` imports `reports`; `cli` imports `controller`. `explore` confirms no cycles (R1). New files are submodules of `reports/`, **not** a new top-level package, so no amendment is needed. |
| **V. Modern Python & Code Quality** | Compliant | `X \| Y` unions, no `Any`, Google-style docstrings, ruff-clean at 120. Note ruff also reformats Python snippets inside `.md`. |
| **VI. Extensibility & Discoverability** | Compliant | Report generation stays algorithm-agnostic: model families are discovered from the directory tree and hyperparameters from the persisted mapping, so a new algorithm needs no report change. |

**Root Class Registry**: untouched. `RLModel`, `TDModel`, `BaseConfig`, `HyperParamsBase`,
`HerculeConfig`, `EpochResult`, `Runner`, `Supervisor` are all unmodified — in particular
`Runner.save()` keeps its current output, which R2 confirmed is already sufficient.
**No constitution amendment required. No version bump.**

**Called out for the PR anyway**: `controller.generate_experiment_report()` changes its return
type from `Path` to a structured result. `controller/` is outside the registry so no amendment
is triggered, but this is a public, frontend-agnostic API change and belongs in the PR
description.

### Post-Phase-1 re-evaluation

Re-checked after the design below was fixed: no gate changed verdict. The design adds four
submodules under `reports/`, Pydantic result models, and two rewritten templates. Nothing
crosses a module boundary in the wrong direction and no registry class is touched.

## Project Structure

### Documentation (this feature)

```text
specs/004-improve-report-generation/
├── spec.md                  # /speckit.specify output
├── checklists/
│   └── requirements.md      # Spec quality checklist
├── plan.md                  # This file (/speckit.plan)
├── research.md              # Phase 0 output — 11 verified decisions
├── data-model.md            # Phase 1 output
├── quickstart.md            # Phase 1 output
├── contracts/
│   └── reports-api.md       # Phase 1 output — public function contracts
└── tasks.md                 # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/hercule/
├── reports/
│   ├── __init__.py          # MODIFIED: discovery + generation orchestration; public API
│   ├── run_table.py         # NEW: RunRecord/RunTable — walk a group, load runs, one table
│   ├── selection.py         # NEW: ranked best/median/worst series selection (pure)
│   ├── pca.py               # NEW: numpy-SVD PCA on hyperparameters (pure)
│   ├── render.py            # NEW: jupytext -> execute -> HTML -> PDF, graceful skip
│   ├── cli.py               # DELETED: dead unregistered Click group
│   ├── example_usage.py     # DELETED: dead script with a hardcoded path
│   ├── README.md            # MODIFIED: stale API description
│   └── templates/
│       ├── report_template.py.j2              # REWRITTEN
│       └── comparative_report_template.py.j2  # REWRITTEN
├── controller/__init__.py   # MODIFIED: ReportBundle return type, narrowed exception wrapper
└── cli/main.py              # MODIFIED: report command reports every artifact

tests/
└── reports/                 # NEW package (reports has zero coverage today)
    ├── test_run_table.py         # unit: walk, load, skip unreadable, never read model.json
    ├── test_selection.py         # unit: 9-cap, <=9 passthrough, tie determinism
    ├── test_pca.py               # unit: standardisation, zero-variance drop, sign pinning,
    │                             #       explained variance, degenerate shapes
    ├── test_render.py            # unit: tag filtering, graceful skip when no browser
    ├── test_report_generation.py # integration: generate over a synthetic tree
    └── test_report_cli.py        # integration: `hercule report` end to end

pyproject.toml               # MODIFIED: deps per research R10
CLAUDE.md, AGENTS.md         # MODIFIED: reports description becomes stale
```

**Structure Decision**: single project, existing `src/hercule` layout. New code is added as
**submodules of the existing `reports/` package** rather than a new top-level package, which
keeps Constitution principle IV satisfied without an amendment. The pure logic (table,
selection, PCA) is separated from the emitting templates specifically so it can be unit
tested — today `reports/` has **zero test coverage** (`grep -rn -i report tests/` returns
nothing), and ruff does not lint template bodies.

## Design

### D1. Shape of the generated notebook

Both templates become fixed-size documents. The per-run Jinja `{% for %}` loop that emits ~35
lines per run is deleted; the template receives only the group's root path, the three
filenames, and the tag names. Cell layout for the comparative report:

| # | tag | content |
|---|---|---|
| 1 | — | markdown title; environment named in prose (FR-001/002/003) |
| 2 | `remove_cell` | imports, `Agg` guard, `_locate_report_dir()` (R9) |
| 3 | `remove_input` | walk the tree, build the run table, print "loaded N runs, skipped M" |
| 4 | — | markdown: environment |
| 5 | `remove_input` | print environment id, settings, spaces, reward threshold |
| 6 | — | markdown: hyperparameter grid |
| 7 | `remove_input` | print the grid table (varying parameters only) |
| 8 | — | markdown: learning progress |
| 9 | — | ranked-selection charts (rewards, success rate) |
| 10 | — | markdown: final performance |
| 11 | — | evaluation charts |
| 12 | — | markdown: hyperparameter PCA |
| 13 | — | projection + explained variance + loadings, with degradation branches |
| 14 | — | markdown: ranking and conclusion |
| 15 | `remove_input` | metrics table, winner, runs-omitted counts |

The individual report follows the same shape minus the comparison and PCA sections.

### D2. Run table (`run_table.py`)

`RunRecord` is a Pydantic model — per Conventions, structured data is a model, not a dict —
carrying `directory`, `model_name`, `env_id`, `env_kwargs`, `max_episode_steps`,
`hyperparameters`, `learning_rewards`, `learning_steps`, `testing_rewards`, `testing_steps`,
plus the aggregates used for ranking (`mean_learning_reward`, `learning_success_rate`,
`mean_testing_reward`, `testing_success_rate`, `episode_count`).

`build_run_table(root)` walks with the existing `find_experiment_directories`, then per run
reads **only** `environment.json` and `run_info.json` — never `model.json` (R3, FR-007,
SC-010). `model_name` comes from the run directory's parent name. Unreadable runs become
`SkippedRun(path, reason)` and never abort the walk (FR-008). `to_dataframe()` produces the
wide pandas frame the charts consume.

The same function is called **both** by the generator (to write the manifest and decide whether
a group qualifies) and by the generated notebook at runtime, so there is one implementation.

### D3. Series selection (`selection.py`)

`select_series(records, metric, per_bucket=3)` sorts descending by `metric`, takes the first 3,
the 3 centred on the median index, and the last 3, de-duplicating when buckets overlap. Returns
the chosen records tagged with their bucket plus `omitted_count`.

Determinism (FR-014/SC-009) comes from sorting on `(-metric_value, directory_name)` —
`directory_name` is unique within a group and stable on disk, which resolves the pervasive ties
(many FrozenLake runs score exactly 0). With `len(records) <= 9` every record is returned and
`omitted_count == 0` (FR-011). Runs of differing episode counts are drawn against their own
index range rather than a shared x-array (FR-015).

### D4. PCA (`pca.py`)

`hyperparameter_pca(records) -> PcaResult | PcaUnavailable` implements R4 exactly: alphabetical
column order, numeric-only with `bool` excluded, correlation-matrix standardisation with
`ddof=1`, exact `sd > 0` zero-variance drop, `np.linalg.svd`, sign pinned on the
largest-magnitude entry of each `Vt` row and applied to both `Vt` and `U`, truncation to
`min(n - 1, p_kept)`. Returns scores, `explained_variance_ratio`, loadings, communalities, kept
and dropped column names, and a reason when unavailable.

Grouped per model family (FR-020) by `model_name`. `PcaUnavailable` carries a human-readable
reason for the `p_kept < 2` / `n < 3` cases (FR-021) and the template renders it as text rather
than raising — an exception in a middle cell blocks everything below it.

Explained variance is printed prominently because R4 established that full cartesian grids make
the correlation matrix near-identity, so PC1+PC2 typically capture only ~2/p of the variance.

### D5. Render pipeline (`render.py`)

`render_report(py_path, *, execute_timeout=1800) -> RenderResult` runs R6/R7/R8:
`jupytext.read(fmt="py:percent")` → `ExecutePreprocessor(timeout=1800, startup_timeout=120,
interrupt_on_timeout=True, allow_errors=False, record_timing=False, on_cell_start=<progress>)`
with `resources={"metadata": {"path": str(report_dir)}}` → `HTMLExporter` configured with
`TagRemovePreprocessor` (`enabled = True` — the default is `False`) → write HTML with
`encoding="utf-8"` → print via system browser.

Browser discovery tries `msedge`/`chrome`/`chromium` via `shutil.which` plus standard install
paths, invoking `--headless=new --disable-gpu --no-sandbox --user-data-dir=<fresh temp>
--no-pdf-header-footer --virtual-time-budget=10000 --host-resolver-rules="MAP * ~NOTFOUND"
--print-to-pdf=<out> <html.as_uri()>`. Success is `pdf.exists() and pdf.stat().st_size > 0`,
**not** the return code (R11). Fallback engine is `WebPDFExporter`, gated on
`importlib.util.find_spec("playwright")`.

Every failure path returns `RenderResult` with `pdf=None` and a sanitised reason (ANSI
stripped, ASCII-coerced) — the executed notebook and the HTML are still written and the command
exits 0 (FR-026). Intermediates go to a short temp dir and the finished PDF is moved into place
(MAX_PATH, R11).

### D6. Widened contract

`generate_report()` returns `ReportBundle`: a list of per-group `ReportArtifacts(notebook, html,
pdf, pdf_skip_reason, runs_loaded, runs_skipped)` plus the groups that were skipped and why
(FR-027, FR-030). `controller.generate_experiment_report()` returns the same model;
`cli/main.py` formats it, listing every artifact and any skip reason. The existing
`except Exception → ValueError` wrapper in the controller is narrowed so `FileNotFoundError`
propagates as its docstring already promises.

### D7. Pre-existing defects fixed (in scope per spec Assumptions)

1. `report_template.py.j2:236-273` — `# %%` markers emitted **inside** an indented
   `if testing_rewards:` block, producing a notebook with orphaned indented code.
2. `boxplot(labels=...)` at `report_template.py.j2:274,281` — renamed to `tick_labels` in
   matplotlib 3.9. Verified on the installed 3.10.8: deprecated but still accepted (warning
   only), dropped in 3.11. Not fatal today; fatal on upgrade, and noisy now.
3. `Path(__file__)` in both templates — undefined in a kernel (R9).
4. `reports/__init__.py:211` — `load_environment(self.experiment_path)` passes a *directory* to
   a function that opens a *file*; swallowed at `:212`, so `env` is always `None`.
5. `generate_report` ignores `output_path` in the comparative branch (`:434`) and has an
   identical if/else return (`:501-505`).

Also removed: `create_learning_plots` (`:508-608`, dead, duplicates the template's plots and
carries the same `labels=` bug), the unregistered `reports/cli.py` group, and
`example_usage.py`. `ExperimentData.load_data()` being called per run purely as a validity
check while its data is discarded (`:442`) is replaced by the run table.

## Phase Gates

- **Phase 0 complete**: `research.md` — 11 decisions, no NEEDS CLARIFICATION remaining.
- **Phase 1 complete**: `data-model.md`, `contracts/reports-api.md`, `quickstart.md`, agent
  context updated.
- **Phase 2**: `/speckit.tasks`.

## Complexity Tracking

No constitution violations to justify. Two scope notes recorded for reviewers:

| Item | Why needed | Simpler alternative rejected because |
|---|---|---|
| Fixing 5 pre-existing defects | FR-023 requires a PDF, which requires the report to execute; today neither template can execute under a kernel | Shipping the PDF feature on top of a non-executing report would fail SC-005/SC-006 immediately |
| New `reports/` submodules instead of logic inside templates | `reports/` has zero test coverage today; pure helpers make selection and PCA unit-testable | Logic inside `.j2` bodies is untestable and unlinted by ruff |

## Constitution Impact (post-implementation, Phase 8)

**Root Class Registry**: untouched. `RLModel`, `TDModel`, `BaseConfig`, `HyperParamsBase`,
`HerculeConfig`, `EpochResult`, `Runner`, `Supervisor` were not modified by this feature — confirmed
again at the end of implementation, not just at the Phase 0/1 gates above. **No constitution
amendment is required, and no version bump is needed.**

**Called out for the PR description anyway** (public API changes outside the registry, per the
Constitution Check gate above):

- `controller.generate_experiment_report()` changes its public return type from `Path` to
  `ReportBundle` (`reports: list[ReportArtifacts]`, `skipped_groups: list[SkippedGroup]`). Any
  caller that treated the return value as a single path (e.g. printed it directly, or did
  `Path(result)`) breaks and must be updated to read `bundle.reports[i].notebook`/`.pdf` etc.
- `src/hercule/reports/` lost several previously-public symbols: `ExperimentData`,
  `create_learning_plots`, and the `_format_python_value`/`topython` Jinja filter registrations
  are all removed (dead code, per research R1/contracts C3); `reports/cli.py` and
  `reports/example_usage.py` were deleted outright (unregistered/unreachable). Any external code
  importing these directly (none found within this repository) would break.
- New public symbols were added to `hercule.reports`: `build_run_table`, `RunRecord`, `RunTable`,
  `SkippedRun`, `select_series`, `SeriesBucket`, `SelectedSeries`, `SeriesSelection`,
  `hyperparameter_pca`, `PcaResult`, `PcaUnavailable`, `is_spectrum_degenerate`, `render_report`,
  `RenderResult`, `ReportArtifacts`, `ReportBundle`, `SkippedGroup`, `ReportManifest`. These are
  part of the public contract because the generated `.py` report itself imports
  `build_run_table`/`select_series`/`hyperparameter_pca` at execution time — a previously
  generated report will re-import them by name, so these three signatures in particular should
  not change without a compatibility note in a future PR.

None of the above touches a Root Class Registry entry, so this is reported for reviewer awareness
in the PR body, not as a constitution amendment.
