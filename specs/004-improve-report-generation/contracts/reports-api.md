# Contracts: Report Generation Public Interfaces

**Feature**: `004-improve-report-generation` | **Date**: 2026-07-28

Hercule is a library plus a CLI, so its external interfaces are (1) the `hercule report` command
surface, (2) the frontend-agnostic `controller` API, (3) the `reports` package API, and (4) the
generated documents themselves, which end users open and execute. All four are specified here.

---

## C1. CLI contract — `hercule report`

**Unchanged**: the argument and option surface stays exactly as today, per FR-029.

```
hercule report EXPERIMENT_PATH [-o|--output PATH] [-v|-vv]
```

| Element | Type | Behaviour |
|---|---|---|
| `EXPERIMENT_PATH` | existing path | A single run directory → one individual report. A parent directory → one comparative report per environment-settings group. |
| `--output`, `-o` | path | Output path for the generated report. **Behaviour change**: currently silently ignored for comparative reports (`__init__.py:434`); it must now either be honoured or rejected with a clear message rather than accepted and discarded. |
| `-v` / `-vv` | flag | INFO / DEBUG logging, via the existing `verbose_option`. |

**New options**:

| Option | Default | Behaviour |
|---|---|---|
| `--no-pdf` | off | Generate and execute nothing beyond the notebook; skip execution and PDF entirely. Escape hatch for the large sweeps. |
| `--no-execute` | off | Write the notebook without executing it. Implies `--no-pdf`. Restores today's fast scaffold-only behaviour. |

**Exit codes**: `0` on success **and** when the PDF was skipped for an environmental reason
(FR-026). Non-zero only when no notebook could be produced at all. Note the current command
`return`s on every exception, so a hard failure exits 0 today; that is tightened.

**Output contract**: the command must name every artifact it produced and every group it
skipped (FR-027, FR-030). Illustrative:

```
📊 Generating report for experiment: outputs/simple_games
   ▶ FrozenLake-v1 / is_sli_False__map_nam_4x4__max_epi_ste_200 — 11 runs
   ▶ FrozenLake-v1 / is_sli_True__map_nam_4x4__max_epi_ste_200 — 11 runs
✅ 2 reports generated
   outputs/simple_games/FrozenLake-v1/is_sli_False__.../comparative_report.py
   outputs/simple_games/FrozenLake-v1/is_sli_False__.../comparative_report.pdf
   ...
⚠️  1 group skipped: outputs/.../single_run (only 1 run, nothing to compare)
```

All markers pass through `harden_output_streams()` via the `cli` group callback, so cp1252
stdout stays safe. Any text interpolated from a cell-execution failure must additionally be
ANSI-stripped and ASCII-coerced (research R11).

---

## C2. Controller contract

```python
def generate_experiment_report(
    experiment_path: Path,
    output_path: Path | None = None,
    *,
    execute: bool = True,
    render_pdf: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportBundle:
```

**Change**: return type widens from `Path` to `ReportBundle`. This is the public,
frontend-agnostic API change flagged in the plan's Constitution Check — `controller/` is outside
the Root Class Registry, so no amendment is triggered, but it must appear in the PR description.

| Aspect | Contract |
|---|---|
| Raises `FileNotFoundError` | `experiment_path` does not exist. **Must propagate**, as the existing docstring already promises — today the blanket `except Exception → ValueError` at `:202-204` swallows it. |
| Raises `ValueError` | Path is not a directory, or no qualifying report group was found. |
| Returns | `ReportBundle` — never a partial success expressed as an exception. A group whose PDF was skipped is a successful group with `pdf=None` and a populated `pdf_skip_reason`. |
| `progress` | Optional sink for human-readable progress lines, so the CLI can satisfy SC-008's 30-second cadence without `reports/` importing Click. |
| Purity | No printing. All user-facing text is the caller's concern; this layer returns data. |

---

## C3. `reports` package API

```python
# Discovery — unchanged signatures, retained
MAX_DEPTH: int


def is_valid_experiment_directory(directory: Path) -> bool: ...
def find_experiment_directories(
    root_directory: Path, max_depth: int = MAX_DEPTH, current_depth: int = 0
) -> list[Path]: ...


# Generation
def generate_report(
    experiment_path: Path,
    output_path: Path | None = None,
    *,
    execute: bool = True,
    render_pdf: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportBundle: ...
def generate_individual_report(experiment_path: Path, output_path: Path | None = None, **kwargs) -> ReportArtifacts: ...


# Run table — called by BOTH the generator and the generated notebook
def build_run_table(root: Path) -> RunTable: ...


# Pure helpers — called by the generated notebook
def select_series(records: Sequence[RunRecord], metric: str, per_bucket: int = 3) -> SeriesSelection: ...
def hyperparameter_pca(records: Sequence[RunRecord]) -> PcaResult | PcaUnavailable: ...


# Rendering
def render_report(py_path: Path, *, execute_timeout: int = 1800) -> RenderResult: ...
```

**Removed** (all dead today, per research R1):

| Symbol | Reason |
|---|---|
| `create_learning_plots` | Never called; duplicates the template's plots and carries the same removed `boxplot(labels=)` kwarg. |
| `ExperimentData` | Superseded by `RunRecord`/`RunTable`. Its `load_data()` was called per run purely as a validity check while its data was discarded (`:442`). |
| `_format_python_value` / the `topython` Jinja filter | Registered at `:325` and `:472` but used by neither template — zero occurrences in the `.j2` files. |
| `reports/cli.py` (`generate`, `reports` group) | Never registered on the main CLI; unreachable except via `python -m`. |
| `example_usage.py` | Standalone script with a hardcoded FrozenLake path. |

`build_run_table`, `select_series` and `hyperparameter_pca` are **part of the public API
specifically because the generated notebook imports them**. A user who opens
`comparative_report.py` in their own Jupyter executes `from hercule.reports import ...`, so
these signatures are an end-user contract and cannot change without breaking previously
generated reports. This is the reason the logic lives in the package rather than being inlined
into the template.

---

## C4. Generated-document contract

The generated `.py` is itself an interface — users open, read, execute and re-run it.

| Guarantee | Requirement |
|---|---|
| Format | jupytext percent (`# %%`), with the existing YAML front-matter (`format_name: percent`, kernel `python3`), so it opens as a notebook. |
| Size | Independent of run count (FR-005, SC-001). No `{% for %}` over runs in the template. |
| Executability | Executes end to end with no error in all three contexts: framework-driven execution, a manually opened notebook, and `python report.py` (FR-031, SC-005). |
| Cell markers | Never emitted inside an indented block — that breaks the percent-format cell contract and yields orphaned indented code (current defect at `report_template.py.j2:236-273`). Conditionals live *inside* a cell, never around cell boundaries. |
| Data location | Resolved by the verifying candidate search of research R9, never bare `Path(__file__)` — undefined in a kernel. |
| Idempotence | Re-running over unchanged results selects identical chart series (FR-014, SC-009). Figure *geometry* is not guaranteed bit-identical across platforms when singular values tie (research R4). |
| No weights | Never reads or prints `model.json` payloads (FR-007, SC-010). |
| No magics | No `%matplotlib inline` — the kernel auto-selects the inline backend, and a magic would break `python report.py`. |

### Cell tag vocabulary

The tag names are the contract between the templates and `render.py`. nbconvert ships no
defaults, so these strings must match exactly on both sides.

| Tag | Literal marker in the generated `.py` | Effect in the PDF |
|---|---|---|
| `remove_cell` | `# %% tags=["remove_cell"]` | Cell absent entirely — source and output. For imports and model reconstruction. |
| `remove_input` | `# %% tags=["remove_input"]` | Source dropped, **output kept**. For loading and summary cells whose printed output is informative. |
| `remove_output` | `# %% tags=["remove_output"]` | Output dropped, source kept. Reserved; not used by the initial templates. |
| *(untagged)* | `# %%` | Fully retained. Charts and analysis code. |

Mapping to FR-025: mechanical code (imports, discovery, loading, model reconstruction) is
excluded, while the informative output such code produced is retained. `remove_input_tags`
keeping the output while dropping the source was empirically confirmed (research R7).

---

## C5. Artifact layout

| Report kind | Location | Files |
|---|---|---|
| Individual | the run directory | `report.py`, `report.ipynb` (executed), `report.html`, `report.pdf` |
| Comparative | the environment-settings directory | `comparative_report.py`, `comparative_report.ipynb`, `comparative_report.html`, `comparative_report.pdf`, `report_manifest.json` |

`report_manifest.json` is written beside a comparative report as the anchor file the notebook's
directory search verifies against (research R9) — the environment-settings level has no
naturally occurring file to key on.

Regeneration replaces these in place (FR-028). On failure to execute, `<name>.failed.ipynb` is
written with partial outputs for debugging, since nbconvert does not save the notebook when it
raises (research R8).

---

## C6. Contract tests

Each contract maps to at least one test, in `tests/reports/`:

| Contract | Test |
|---|---|
| C1 exit codes and artifact listing | `test_report_cli.py` — Click runner over a synthetic tree; asserts exit 0 with `--no-pdf`, and that skipped groups are named |
| C2 `FileNotFoundError` propagates | `test_report_generation.py` — missing path must raise `FileNotFoundError`, not `ValueError` |
| C3 `build_run_table` never opens `model.json` | `test_run_table.py` — write an intentionally corrupt `model.json`; loading must still succeed |
| C3 selection determinism | `test_selection.py` — all-equal metric values must yield a stable order across repeated calls |
| C3 PCA degradation | `test_pca.py` — single varying column and `n < 3` must return `PcaUnavailable`, not raise |
| C4 size independence | `test_report_generation.py` — generated line count for 2 runs vs 50 runs within 10% (SC-001) |
| C4 tag vocabulary | `test_render.py` — round-trip a template through jupytext and assert the tags parse to the expected metadata |
| C5 graceful PDF skip | `test_render.py` — with browser discovery stubbed to find nothing, `RenderResult.pdf is None` and a reason is set, no exception |
