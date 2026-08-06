# Phase 1 Data Model: Improved Experiment Report Generation

**Feature**: `004-improve-report-generation` | **Date**: 2026-07-28 | **Plan**: [plan.md](./plan.md)

All structures below are **Pydantic v2 models** (project convention: structured data is a model, not a dict),
Python 3.10+ (`X | Y` unions only — no `typing.Optional`/`Union`/`Any`), `@field_validator` +
`@classmethod`, Google-style docstrings, ruff line-length 120.

Nothing in the Root Class Registry is touched. `EpochResult` keeps its current definition and remains the
authoritative description of one on-disk metric entry.

## Module placement

| Module | Entities |
|---|---|
| `src/hercule/reports/run_table.py` | `RankingMetric`, `RunRecord`, `SkippedRun`, `RunTable`, `ReportManifest` |
| `src/hercule/reports/selection.py` | `SeriesBucket`, `SelectedSeries`, `SeriesSelection` |
| `src/hercule/reports/pca.py` | `PcaResult`, `PcaUnavailable` |
| `src/hercule/reports/render.py` | `RenderResult` |
| `src/hercule/reports/__init__.py` | `ReportArtifacts`, `SkippedGroup`, `ReportBundle` |

## Relationships

```text
ReportBundle
├── reports: list[ReportArtifacts]        # one per report group actually rendered
│     └── built from RenderResult (paths) + RunTable counts (runs_loaded/runs_skipped)
└── skipped_groups: list[SkippedGroup]    # candidate groups not rendered, with a reason

RunTable  (one per report group, built once, consumed by every section — FR-009)
├── records: list[RunRecord]              # one per run leaf directory
├── skipped: list[SkippedRun]             # unreadable runs (FR-008)
├── to_dataframe() -> pandas.DataFrame    # scalar projection, one row per RunRecord
└── by_model_family() -> dict[str, list[RunRecord]]
      └── hyperparameter_pca(family_records) -> PcaResult | PcaUnavailable   # FR-020

select_series(records, metric) -> SeriesSelection   # one per chart, FR-010..FR-015
└── series: list[SelectedSeries] -> (RunRecord, SeriesBucket, metric_value)
```

`RunRecord` is the single unit of truth inside a report. `SeriesSelection` and the PCA entities hold
*references* to `RunRecord` instances or to their names; they never re-read the disk.

---

## 1. `RunRecord`

**Purpose**: one run — one trained model in one environment configuration with one hyperparameter set —
loaded from its leaf directory. Produced only by the run-table loader; consumed by every chart, table,
selection and projection in the report.

### Fields

| Field | Type | Origin | Notes |
|---|---|---|---|
| `directory` | `Path` | walk | Absolute path to the run leaf directory. |
| `model_name` | `str` | `directory.parent.name` | Model family (`simple_sarsa`, `deep_q_learning`, …). **Not** read from `model.json`. |
| `env_id` | `str` | `environment.json["id"]` | Gymnasium identifier, e.g. `CartPole-v1` (FR-001). |
| `env_kwargs` | `dict[str, bool \| int \| float \| str \| None]` | `environment.json["kwargs"]` | Environment settings; `{}` when nothing was overridden (FR-002). |
| `max_episode_steps` | `int \| None` | `environment.json["max_episode_steps"]` | `None` when the env declares no limit. |
| `hyperparameters` | `dict[str, bool \| int \| float \| None]` | `run_info.json["model_hyperparameters"]` | Flat `name → scalar` mapping with full unabbreviated names (R2). |
| `learning_rewards` | `list[float]` | `run_info.json["learning_metrics"][*]["reward"]` | One entry per learning episode. |
| `learning_steps` | `list[int]` | `learning_metrics[*]["steps_number"]` | Same length as `learning_rewards`. |
| `testing_rewards` | `list[float]` | `testing_metrics[*]["reward"]` | Empty when the run has no evaluation phase. |
| `testing_steps` | `list[int]` | `testing_metrics[*]["steps_number"]` | Same length as `testing_rewards`. |

`hyperparameters` value types **as observed on disk**: `float` (`learning_rate: 0.0001`, `weight_decay: 0.0`,
`epsilon_decay: 0.005`) and `int` (`batch_size: 32`, `replay_buffer_size: 10000`, `seed: 42`, `step_modulo: 1`).
`bool` is possible in principle — `ParameterValue` admits it and `expand_variants()` can sweep a boolean flag —
but no current experiment records one. `str` is deliberately **not** admitted here: a non-numeric
hyperparameter would be excluded from the projection with a stated reason (FR-022), and `hercule` writes
`ParameterValue` scalars, so a string value must surface as a validation error rather than be silently mixed
into the PCA matrix. `None` is admitted because `HyperParamsBase.to_dict()` drops `None` values but
`Runner.save()` does not guarantee it.

`bool` is listed **first** in both unions so a stored `true` is never widened to `1` — Pydantic v2 smart-union
prefers an exact type match, and left-to-right fallback would otherwise coerce.

### Derived fields (computed, not stored on disk)

Declared as `@computed_field` over `functools.cached_property`, so they are derived exactly once per record,
cannot drift from the lists they summarise, and still appear in `model_dump()` (which `to_dataframe()` uses).

| Field | Type | Formula | Empty-input result |
|---|---|---|---|
| `run_name` | `str` | `self.directory.name` | — |
| `episode_count` | `int` | `len(self.learning_rewards)` | `0` |
| `testing_episode_count` | `int` | `len(self.testing_rewards)` | `0` |
| `mean_learning_reward` | `float \| None` | `sum(learning_rewards) / len(learning_rewards)` | `None` |
| `learning_success_rate` | `float \| None` | `sum(1 for r in learning_rewards if r > 0) / len(learning_rewards)` | `None` |
| `mean_testing_reward` | `float \| None` | `sum(testing_rewards) / len(testing_rewards)` | `None` |
| `testing_success_rate` | `float \| None` | `sum(1 for r in testing_rewards if r > 0) / len(testing_rewards)` | `None` |
| `performance` | `float \| None` | `mean_testing_reward if mean_testing_reward is not None else mean_learning_reward` | `None` |

**Success rate is the fraction of episodes whose reward is strictly greater than 0.** That is the only
outcome-independent definition available: `final_state` distinguishes `terminated` from `truncated` but not
success from failure (FrozenLake terminates on both the goal and a hole), so reward is the discriminator.

`None` rather than `0.0` on an empty list is load-bearing: a FrozenLake run that genuinely scored a mean
reward of `0.0` must remain distinguishable from a run with no evaluation phase at all, otherwise the
`performance` fallback (spec Assumptions, FR-017) and the "degrade to learning-phase sections" edge case both
break.

`performance` is the value the PCA scatter encodes as colour (FR-017); R5's constant-value and non-finite
guards operate on the collected `performance` values, not on this field.

### Validation rules

| Rule | Mechanism |
|---|---|
| `env_id` non-empty after `strip()` | `@field_validator("env_id") @classmethod` — raises `ValueError` |
| `max_episode_steps` is `> 0` when not `None` | `@field_validator("max_episode_steps") @classmethod` |
| `len(learning_rewards) == len(learning_steps)`, `len(testing_rewards) == len(testing_steps)` | `@model_validator(mode="after")` — a mismatch means a truncated `run_info.json` |
| No `hyperparameters` value is a `list` | `@field_validator("hyperparameters") @classmethod` — mirrors `BaseConfig.get_hyperparameters_signature()`; a list means `expand_variants()` never ran |
| No empty `hyperparameters` key | same validator |

The model performs **no filesystem check** on `directory`: the loader owns disk access, which keeps the model
constructible from literals in unit tests.

### Relationship to `EpochResult`

`EpochResult` describes one entry of `learning_metrics`/`testing_metrics` on disk. `RunRecord` deliberately
does **not** hold `list[EpochResult]`: it keeps only the two fields any report section consumes, as parallel
primitive lists. `final_state` is read and discarded. Rationale: the largest group parses ~211 MB of metric
JSON inside the kernel (SC-008), and 135 runs x thousands of episodes of Pydantic model instantiation is the
dominant avoidable cost.

### `model.json` is never read

The loader opens `environment.json` and `run_info.json` only. `model.json` is **not opened, not parsed, not
referenced** — `model_name` comes from `directory.parent.name`, which the existing code already relies on.

Why: a single report group holds **~55 MB across its `model.json` files** (`q_network_state_dict`), and
`outputs/` holds 110.6 MB of them in total. The
current template parses all of it per run and prints it verbatim, skipping only `['q_table']`. Not reading it
is what satisfies FR-007 (loading routine must not read or retain stored model weights) and SC-010 (no
generated report contains stored model weights), and it removes the dominant I/O cost of a large group. There
is no field on `RunRecord` capable of holding weights, so the guarantee is structural rather than a
convention.

---

## 2. `SkippedRun`

**Purpose**: record a run directory that could not be read, so the walk continues and the report can state
what it excluded (FR-008, User Story 2 scenario 3).

| Field | Type | Notes |
|---|---|---|
| `path` | `Path` | The run directory that was skipped. |
| `reason` | `str` | Human-readable cause, e.g. `"run_info.json is not valid JSON"`, `"environment.json missing 'id'"`. |

**Validation**: `reason` must be non-empty after `strip()`; `@field_validator("reason")` strips ANSI escapes
(`re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)`) and coerces to ASCII with a replacement character, because the
reason is derived from exception text and may be logged or echoed to a cp1252 console (R11).

A skipped run is never a `RunRecord` with missing fields: partial rows would poison aggregates, rankings and
the PCA matrix.

---

## 3. `RunTable`

**Purpose**: the whole report group in one structure — built once, by the same `build_run_table(root)` used
by the generator (to write the manifest and decide whether a group qualifies) and by the generated notebook at
runtime, so there is exactly one implementation (FR-004, FR-005, FR-009).

| Field | Type | Notes |
|---|---|---|
| `root` | `Path` | Group root the walk started from (the `env/env_params` level for a comparative report). |
| `records` | `list[RunRecord]` | Successfully loaded runs. |
| `skipped` | `list[SkippedRun]` | Unreadable runs. |

### Derived members

| Member | Type | Derivation |
|---|---|---|
| `runs_loaded` | `int` | `len(self.records)` |
| `runs_skipped` | `int` | `len(self.skipped)` |
| `env_id` | `str \| None` | The single `env_id` shared by all records; `None` when `records` is empty. Records within a group share it by construction. |
| `env_kwargs` | `dict[...] \| None` | `records[0].env_kwargs`; `None` when empty. Group identity is the env-params directory level. |
| `max_episode_steps` | `int \| None` | `records[0].max_episode_steps` |
| `model_names` | `list[str]` | Sorted unique `record.model_name`. |
| `varying_hyperparameters` | `dict[str, list[...]]` | Per hyperparameter name, the sorted distinct values across records; keys with a single distinct value are excluded. Feeds the grid table (cell 7) and FR-019's exclusion statement. |
| `by_model_family()` | `dict[str, list[RunRecord]]` | Groups `records` by `model_name`, keys in sorted order — the per-family input to the PCA (FR-020). |
| `to_dataframe()` | `pandas.DataFrame` | See schema below. |

**Validation**: `@model_validator(mode="after")` rejects duplicate `record.directory` values (the walk must not
visit a leaf twice). It does **not** require `records` to be non-empty: a group whose every run was unreadable
is a legitimate, reportable outcome. It does **not** require a single `env_id` — the individual-report path
builds a one-record table, and a mixed table must degrade rather than raise inside a notebook cell.

### `to_dataframe()` column schema

Charts, tables, rankings and the PCA matrix all read this frame, so the column names and dtypes are part of
the contract.

| Column | dtype | Source |
|---|---|---|
| `directory` | `object` (`str`) | `str(record.directory)` |
| `run_name` | `object` (`str`) | `record.run_name` — unique within a group; the deterministic tie-break key |
| `model_name` | `object` (`str`) | `record.model_name` |
| `env_id` | `object` (`str`) | `record.env_id` |
| `max_episode_steps` | `Int64` (nullable) | `record.max_episode_steps` |
| `episode_count` | `int64` | derived |
| `testing_episode_count` | `int64` | derived |
| `mean_learning_reward` | `float64` | derived; `NaN` where the record's value is `None` |
| `learning_success_rate` | `float64` | derived; `NaN` where `None` |
| `mean_testing_reward` | `float64` | derived; `NaN` where `None` |
| `testing_success_rate` | `float64` | derived; `NaN` where `None` |
| `performance` | `float64` | derived; `NaN` where `None` |
| `hp_<name>` | `float64`, or `boolean` (nullable) for boolean hyperparameters | one column per hyperparameter name seen anywhere in the table, `<name>` verbatim (unabbreviated) |
| `env_<name>` | `float64` / `boolean` / `object` per value type | one column per `env_kwargs` key seen anywhere in the table |

- **Index**: `RangeIndex`. Rows are sorted by `(model_name, run_name)` so regeneration is byte-stable.
- **Absent is absent, not a value**: `frozenlake_4x4` mixes deep Q-learning with tabular Q-learning, so a
  `hp_<name>` column is `NaN` (or `pd.NA`) for runs whose family has no such hyperparameter. Nullable `Int64`
  and `boolean` dtypes are used rather than `int64`/`bool` precisely so absence is representable.
- **Numeric hyperparameters land in `float64`, not `int64`**, even when every observed value is an integer:
  one absent value would silently upcast the column anyway, and the PCA standardises to float regardless.
- **No per-episode column exists.** `learning_rewards`/`learning_steps`/`testing_rewards`/`testing_steps` stay
  on the `RunRecord` objects and are reached through `SeriesSelection`. Putting them in the frame would create
  object columns holding ~211 MB of Python lists for the largest group, and the frame's consumers (grid table,
  ranking table, PCA matrix) are all scalar-only.
- **Column-name collisions** are possible in principle (a hyperparameter literally named `run_name`); the
  `hp_`/`env_` prefixes exist to make them impossible in practice.

---

## 4. `SeriesBucket`

**Purpose**: name the selection group a drawn curve belongs to, for the legend and for the report's prose
(FR-013).

```python
class SeriesBucket(str, Enum):
    """Which ranked group a selected run belongs to."""

    BEST = "best"
    MEDIAN = "median"
    WORST = "worst"
```

`str, Enum` rather than `enum.StrEnum`: `StrEnum` is 3.11+, and the project floor is 3.10. Mixing in `str`
means the member can be used directly as a matplotlib label and serialises to its own value.

Ordering is `BEST < MEDIAN < WORST` by declaration; deduplication and legend ordering rely on it.

---

## 5. `SeriesSelection`

**Purpose**: the bounded, ranked subset of a report group chosen for **one** chart, plus how many runs it left
out (FR-010 – FR-014). Produced by `select_series(records, metric, per_bucket=3)`.

### `RankingMetric`

```python
RankingMetric = Literal[
    "mean_learning_reward",
    "learning_success_rate",
    "mean_testing_reward",
    "testing_success_rate",
]
```

The name of the `RunRecord` derived field a chart ranks on. Each chart passes the aggregate of the quantity it
draws, over the phase it draws (spec Assumptions), so two charts over the same group may select different runs
(FR-012).

### `SelectedSeries`

| Field | Type | Notes |
|---|---|---|
| `record` | `RunRecord` | The run to draw. Gives the chart its per-episode lists. |
| `bucket` | `SeriesBucket` | Legend group. |
| `metric_value` | `float \| None` | The ranking value used, carried so the legend and the ranking table need not recompute it. `None` when the record has no value for that metric. |

### `SeriesSelection`

| Field | Type | Notes |
|---|---|---|
| `metric` | `RankingMetric` | What the selection ranked on. |
| `series` | `list[SelectedSeries]` | Runs to draw, in `(bucket, rank)` order. |
| `omitted_count` | `int` | Runs in the group not drawn (FR-013). |

Derived: `total_count = len(self.series) + self.omitted_count`; `counts_by_bucket -> dict[SeriesBucket, int]`
for the prose.

**Validation**

| Rule | Mechanism |
|---|---|
| `omitted_count >= 0` | `Field(ge=0)` |
| `len(series) <= 3 * per_bucket` (9 at the default) | `@model_validator(mode="after")` — the hard expression of FR-010/SC-002 |
| No `record.directory` appears twice in `series` | same validator — buckets overlap on small groups and must be deduplicated by the selector |

### Selection and determinism rules

1. **Sort key**: `(-metric_value, directory_name)` — descending on the metric, ascending on
   `record.directory.name` as tie-break. `directory_name` is unique within a group and stable on disk, which
   is what makes regeneration reproducible (FR-014, SC-009) despite pervasive ties: many FrozenLake runs score
   exactly `0`.
2. **Missing metric**: a record whose `metric_value` is `None` sorts as `-inf` (last among values), still
   tie-broken by name. It is never dropped, so a chart over a group with no evaluation phase still selects a
   deterministic subset instead of an empty one.
3. **Buckets**: `BEST` = first `per_bucket` of the sorted list; `WORST` = last `per_bucket`; `MEDIAN` =
   `per_bucket` entries starting at `max(0, min((n - per_bucket) // 2, n - per_bucket))`, i.e. the window
   centred on the median index and clamped inside the list.
4. **Deduplication**: when buckets overlap (small `n`), a run keeps its **first** assignment in
   `BEST → MEDIAN → WORST` order, and `series` holds it once.
5. **Passthrough**: when `len(records) <= 3 * per_bucket`, every record is returned and `omitted_count == 0`
   (FR-011). Buckets are still assigned, so the legend stays uniform across group sizes.
6. **Unequal lengths**: a `SelectedSeries` carries its own record, so each curve is drawn against its own
   `range(len(record.learning_rewards))`. Nothing in this structure implies a shared x-array, which is how
   FR-015 is met without truncating or padding (`simple_games` mixes 200-episode SARSA with 5000-episode
   Q-learning).

---

## 6. `PcaResult`

**Purpose**: the principal-component projection of one model family's hyperparameter grid, everything the
report needs to render the scatter, the explained variance and the per-hyperparameter contributions
(FR-016 – FR-019). Returned by `hyperparameter_pca(records)`.

Let **n** = number of runs in the family, **p_kept** = number of retained hyperparameter columns,
**k** = `min(n - 1, p_kept)` = number of retained components. `PcaResult` is only ever produced when
`k >= 2`, therefore `p_kept >= 2` and `n >= 3`.

| Field | Type | Shape | Notes |
|---|---|---|---|
| `model_name` | `str` | — | The family this projection covers (FR-020). |
| `run_names` | `list[str]` | `n` | `record.run_name`, in the row order of `scores`. Links each point back to its run and to its `performance` colour. |
| `kept_columns` | `list[str]` | `p_kept` | Retained hyperparameter names, **alphabetical** — the fixed column order the sign rule and tie-breaking depend on. |
| `dropped_columns` | `dict[str, str]` | — | Excluded hyperparameter name → reason. Renders FR-019 ("which were excluded and why") and FR-022 ("never silently dropped"). |
| `scores` | `list[list[float]]` | `(n, k)` | `U * S`. Row `i` is run `run_names[i]`; column `j` is component `j`. The scatter plots `scores[:, 0]` vs `scores[:, 1]`. |
| `explained_variance_ratio` | `list[float]` | `k` | `S**2 / sum(S**2)`, descending. Printed prominently (FR-018) and used in the axis labels (`PC1 (30.3% of variance)`). |
| `loadings` | `list[list[float]]` | `(k, p_kept)` | `Vt * (S / sqrt(n - 1))[:, None]`. Row `i` is component `i`; column `j` is `kept_columns[j]`. Equals `corrcoef(Xs[:, j], scores[:, i])`. |
| `communalities` | `list[float]` | `p_kept` | Per retained hyperparameter, `sum(loadings[i][j] ** 2 for i in range(2))` — the fraction of that hyperparameter captured by the 2-D plot. |

Nested lists rather than `numpy.ndarray`: the model stays JSON-serialisable without
`arbitrary_types_allowed`, and `numpy.asarray(result.scores)` restores the matrix where a consumer wants one.
The helper converts with `.tolist()` at the boundary.

**Reason vocabulary for `dropped_columns`**: `"not numeric in every run"` (value is `bool` or non-numeric in
at least one run — `bool` is excluded explicitly because `isinstance(True, int)` is `True`),
`"single value across the N runs of this family (<value>)"` (zero variance, FR-019).

**Validation** (`@model_validator(mode="after")`, all shape invariants asserted because a silently ragged
matrix produces a plausible-looking wrong plot):

| Rule |
|---|
| `len(kept_columns) >= 2` and no duplicate name |
| `len(run_names) == len(scores)` and no duplicate name |
| every row of `scores` has length `len(explained_variance_ratio)` (= `k`), and `k >= 2` |
| `len(loadings) == k`, every row of `loadings` has length `len(kept_columns)` |
| `len(communalities) == len(kept_columns)`, each in `[0.0, 1.0]` (tolerance `1e-9`) |
| every `explained_variance_ratio` entry in `[0.0, 1.0]`, descending, `sum(...) <= 1.0 + 1e-9` |
| `set(dropped_columns) & set(kept_columns) == set()` |

**Derivation rules** (R4, each verified numerically):

1. Matrix built per family, columns alphabetical, numeric-only, `bool` excluded.
2. Standardise on the **correlation** matrix, not covariance: `sd = X.std(axis=0, ddof=1)`,
   `keep = sd > 0` (exact test, no epsilon), zero-variance columns dropped **before** the SVD.
3. `U, S, Vt = np.linalg.svd(Xs, full_matrices=False)`.
4. Sign pinned on the largest-magnitude entry of each `Vt` row, applied to **both** `Vt` and `U`, so the plot
   orientation cannot flip between generations.
5. Truncate everything to `k = min(n - 1, p_kept)`, discarding the noise components `full_matrices=False`
   still returns.

**Documented limitation**: Hercule grids are full cartesian products, so the correlation matrix is
near-identity and PC1+PC2 typically capture only ~2/p of the variance. Tied singular values also make the
eigenvector basis non-unique, so BLAS differences can rotate the plot across platforms. The report states
this; SC-009 is scoped to series selection, not figure geometry.

---

## 7. `PcaUnavailable`

**Purpose**: the alternative return of `hyperparameter_pca()` when a projection cannot be computed, so the
generated cell renders a sentence instead of raising — an exception in a middle cell blocks every cell below
it (FR-021).

| Field | Type | Notes |
|---|---|---|
| `model_name` | `str` | The family with no projection. |
| `reason` | `str` | Human-readable, embeds the counts that produced it. Non-empty after `strip()`. |

`hyperparameter_pca(records) -> PcaResult | PcaUnavailable` — a plain `|` union, discriminated at the call
site with `isinstance`. No exception path, no `None`.

### Exact conditions

| Condition | Reason text (shape) |
|---|---|
| `p_kept == 0` | `"no hyperparameter varies numerically across the N runs of <family>; nothing to project"` |
| `p_kept == 1` | `"only 1 hyperparameter varies (<name>); a two-component projection needs at least 2"` |
| `n_samples < 3` | `"only N run(s) in <family>; a two-component projection needs at least 3"` |

`p_kept == 0` is listed separately from `p_kept < 2` because its reason is different in kind — the family
swept nothing numeric at all (the `dummy` random baseline varies only `seed`, and `seed` is constant within a
family here) rather than swept exactly one thing.

The `n_samples < 3` guard is **stricter than FR-021's "fewer than 2 runs"**, deliberately: `k = min(n - 1,
p_kept)`, so `n == 2` yields `k == 1`, there is no PC2, and `scores[:, 1]` raises `IndexError`. FR-021's
intent — state non-applicability and continue — is satisfied for both cases by the same structure.

---

## 8. `RenderResult`

**Purpose**: the outcome of `render_report(py_path)` — the jupytext → execute → HTML → browser-print pipeline
(plan D5). Purely about artifacts on disk; it carries no run counts.

| Field | Type | Notes |
|---|---|---|
| `notebook` | `Path` | The **executed** notebook (`.ipynb`), written on both the success and the failure branch — nbconvert does not save it when it raises, and the traceback would otherwise be lost. |
| `html` | `Path` | Tag-filtered HTML export, always written (`encoding="utf-8"`; the export is ~308 KB of non-cp1252 UTF-8, so `write_text` without an explicit encoding raises on Windows). Remains a shareable artifact when the PDF is skipped. |
| `pdf` | `Path \| None` | `None` on every failure path (FR-026). |
| `pdf_skip_reason` | `str \| None` | Set exactly when `pdf is None`. |

The jupytext `.py` source is not a separate field: it is the input to `render_report` and sits beside the
outputs with the same stem (`notebook.with_suffix(".py")`).

**Validation**

| Rule | Mechanism |
|---|---|
| Exactly one of `pdf` / `pdf_skip_reason` is set (XOR) | `@model_validator(mode="after")` — "PDF present *and* a skip reason" and "PDF absent *and* no reason" are both bugs |
| `pdf_skip_reason` non-empty after `strip()` when set, ANSI-stripped, ASCII-coerced | `@field_validator("pdf_skip_reason") @classmethod` — the string is built from `CellExecutionError` text, which carries arbitrary user output and raw ANSI escapes; the reproduced failure was `UnicodeEncodeError: 'charmap' codec can't encode character '→'` |

**Success is not the browser's return code.** `pdf` is set only when
`pdf.exists() and pdf.stat().st_size > 0`: a bare `--print-to-pdf` handed off to an already-running Edge or
Chrome exits 0 having written nothing. The finished PDF is moved in from a short temp dir (MAX_PATH — the
output tree already approaches 260 characters before a filename is appended).

**Skip-reason vocabulary** (all end in `pdf=None`, `exit code 0`): no Chromium-family browser found (with the
`uv sync --extra pdf && uv run playwright install chromium` remediation), the print produced no file, the
optional `WebPDFExporter` is unavailable, cell execution failed (`CellExecutionError`), cell execution timed
out (`CellTimeoutError` — **not** a subclass, its MRO goes `TimeoutError → OSError`), or the kernel died
(`DeadKernelError`, a plain `RuntimeError`).

---

## 9. `ReportArtifacts`

**Purpose**: everything produced for **one** report group, so the CLI can report the location of every
artifact and any skip reason (FR-027).

| Field | Type | Notes |
|---|---|---|
| `notebook` | `Path` | From `RenderResult.notebook`. Its parent is the group directory, so no separate group field is needed. |
| `html` | `Path` | From `RenderResult.html`. |
| `pdf` | `Path \| None` | From `RenderResult.pdf`. |
| `pdf_skip_reason` | `str \| None` | From `RenderResult.pdf_skip_reason`. |
| `runs_loaded` | `int` | `RunTable.runs_loaded` — echoes the notebook's own "loaded N runs, skipped M". |
| `runs_skipped` | `int` | `RunTable.runs_skipped`. |

**Validation**: the same `pdf` / `pdf_skip_reason` XOR and the same reason sanitisation as `RenderResult`;
`runs_loaded` and `runs_skipped` are `Field(ge=0)`.

**Relationships**: composed from one `RenderResult` (paths, skip reason) and one `RunTable` (counts). It is a
flat projection rather than a nested `RenderResult` so the CLI formatter, the controller and the tests all
read one shape.

Regeneration **overwrites** these paths in place rather than adding suffixed siblings (FR-028), so the field
set fully describes what exists on disk for that group after a run.

---

## 10. `SkippedGroup`

**Purpose**: a candidate report group that was found but not rendered, with the reason — so a silent absence
of output is never mistaken for success (FR-030).

| Field | Type | Notes |
|---|---|---|
| `path` | `Path` | The candidate group directory (`env/env_params` level). |
| `reason` | `str` | e.g. `"only 1 run, nothing to compare"`, `"all 3 runs were unreadable"`. |

**Validation**: `reason` non-empty after `strip()`, ANSI-stripped and ASCII-coerced by the same
`@field_validator` helper as `SkippedRun.reason`.

`SkippedGroup` and `SkippedRun` stay two types rather than one shared `SkippedPath`: they have different
consumers (`ReportBundle` vs `RunTable`), different granularity, and disjoint reason vocabularies. They share
only the reason-sanitising validator function.

The single-run case is the canonical instance: a comparative report needs at least two runs, and today such a
group is dropped with only a log line.

---

## 11. `ReportBundle`

**Purpose**: the widened return of `generate_report()`, `controller.generate_experiment_report()` and the
input to the CLI's output formatting (plan D6). Replaces the current bare `Path` return.

| Field | Type | Notes |
|---|---|---|
| `reports` | `list[ReportArtifacts]` | One per rendered group. Exactly one entry for the individual-report path. |
| `skipped_groups` | `list[SkippedGroup]` | Candidate groups not rendered (FR-030). |

Derived: `report_count = len(self.reports)`; `pdf_count` = entries whose `pdf is not None`;
`has_skips = bool(self.skipped_groups)`.

**Validation**: `@model_validator(mode="after")` rejects a bundle where both lists are empty — "no valid
experiment directories found" is already a `ValueError` from `generate_report`, so an empty bundle would be an
unreported no-op. Duplicate `ReportArtifacts.notebook` paths are also rejected.

**Contract note for the PR**: `controller.generate_experiment_report()` changes its return type from `Path` to
`ReportBundle`. `controller/` is outside the Root Class Registry so no constitution amendment is triggered,
but it is a public, frontend-agnostic API change and belongs in the PR description. The controller's
`except Exception → ValueError` wrapper is narrowed at the same time so `FileNotFoundError` propagates as its
docstring already promises.

---

## 12. `ReportManifest`

**Purpose**: the small file the generator writes beside a comparative notebook, purely so the generated
notebook can **verify** its own data directory at runtime instead of hoping. `Path(__file__)` is undefined in
a kernel, and the `env/env_params` level has no naturally occurring anchor file (an individual report anchors
on `environment.json`).

| Field | Type | Notes |
|---|---|---|
| `root` | `Path` | Group root, as written at generation time. The notebook's last-resort fallback path. |
| `env_id` | `str` | For the prose statement (FR-001) without re-deriving it. |
| `env_kwargs` | `dict[str, bool \| int \| float \| str \| None]` | For the settings statement; `{}` means "no setting overridden" (FR-002). |
| `max_episode_steps` | `int \| None` | Part of the same statement. |
| `model_names` | `list[str]` | Sorted; lets the generator's group decision be reviewable. |
| `runs_loaded` | `int` | `Field(ge=0)` |
| `runs_skipped` | `int` | `Field(ge=0)` |

Every field is a projection of the `RunTable` the generator already built — the manifest introduces no
information of its own and is never the authority for a report section; the notebook rebuilds the table from
the run directories at execution time. It deliberately carries **no timestamp**: per-generation timestamps
would make byte-identical regeneration impossible (the same reason `record_timing=False` is set on the
executor).

Serialised with `model_dump_json()` to `report_manifest.json` in the group root, written with
`encoding="utf-8"`.

---

## Existing structures read from disk

The on-disk schema below was **verified empirically** against the existing `outputs/` tree. No stored format
changes (spec Out of Scope); the loader is strictly read-only.

### `environment.json`

```json
{"id": "FrozenLake-v1", "max_episode_steps": 200, "disable_env_checker": false,
 "kwargs": {"map_name": "4x4", "is_slippery": true}}
```

| Key | Type | Consumed as |
|---|---|---|
| `id` | `str` | `RunRecord.env_id` |
| `max_episode_steps` | `int` | `RunRecord.max_episode_steps` |
| `disable_env_checker` | `bool` | not consumed — a `gym.make` kwarg, not a reportable setting |
| `kwargs` | `dict` | `RunRecord.env_kwargs` |

Only the keys `gym.make` accepts are stored, which is what makes the file round-trippable by
`load_environment` and consumable by `hercule play`. `kwargs` is `{}` for an environment such as
`CartPole-v1` that overrides nothing — the report must then say so explicitly rather than print an empty
structure (User Story 1 scenario 2).

### `run_info.json`

```json
{"learning_ongoing_epoch": 5000, "testing_ongoing_epoch": 100,
 "learning_metrics": [...], "testing_metrics": [...],
 "model_hyperparameters": {"learning_rate": 0.0001, "batch_size": 32, "seed": 42}}
```

| Key | Type | Consumed as |
|---|---|---|
| `learning_ongoing_epoch` | `int` | not consumed — `episode_count` is derived from `len(learning_metrics)`, which is what the charts actually plot |
| `testing_ongoing_epoch` | `int` | not consumed, same reason |
| `learning_metrics` | `list` | `learning_rewards` + `learning_steps` |
| `testing_metrics` | `list` | `testing_rewards` + `testing_steps`; **may be empty** when a run has no evaluation phase |
| `model_hyperparameters` | `dict` | `RunRecord.hyperparameters` |

`model_hyperparameters` is a flat `name → scalar` mapping with full unabbreviated names, so no decoding of the
3-letter directory-signature abbreviations is needed. A documented hazard was tested and **does not
materialise**: `epsilon` is mutable state written back on each update, yet the stored value is the
*configured* one (`epsilon: 1.0` after 5000 episodes at decay 0.005), matching the directory signature.
Verified across three runs.

### Each metric entry (`EpochResult`)

```json
{"reward": 1.0, "steps_number": 37, "final_state": "terminated"}
```

| Key | Type | Consumed as |
|---|---|---|
| `reward` | `float` | `learning_rewards` / `testing_rewards` |
| `steps_number` | `int` | `learning_steps` / `testing_steps` |
| `final_state` | `"terminated" \| "truncated"` | read and discarded — it does not distinguish success from failure (FrozenLake terminates on both goal and hole), so `success_rate` uses `reward > 0` |

### `model.json`

```json
{"model_name": "deep_q_learning", "q_network_state_dict": {...}}
```

**Never opened.** `model_name` is taken from the run directory's parent name instead. The remaining payload is
algorithm-specific weights — ~55 MB per report group, 110.6 MB across `outputs/` — and FR-007/SC-010 forbid loading or printing
it. `is_valid_experiment_directory()` continues to check that the file *exists*, which is a `Path.exists()`
call and reads nothing.
