# Hercule Reports Module

Generates Jupyter-style reports (and, when possible, a PDF) from the JSON results Hercule
training writes to `outputs/`. `hercule report <path>` is the entry point; this module is what
it calls into.

## What it does

`generate_report(experiment_path)` auto-detects what `experiment_path` is:

- a **run leaf directory** (holds `environment.json`, `model.json`, `run_info.json`) →
  one individual report, written as `report.py` beside those files.
- a **parent directory** → recursively discovers run leaf directories up to `MAX_DEPTH` levels
  deep, groups them by *environment + environment settings* (the `env/env_params` level of the
  output tree), and writes one `comparative_report.py` per group beside its runs. A candidate
  group with fewer than 2 loadable runs is skipped, not silently dropped — the reason is
  reported back to the caller.

Both paths then execute the notebook and render it to HTML and (when a Chromium-family browser
is available) PDF, unless the caller opted out.

## Pipeline

```
directory walk (find_experiment_directories)
        │
        ▼
build_run_table(root) -> RunTable            # reads environment.json + run_info.json only
        │                                     # (never model.json — no weights loaded, ever)
        ▼
Jinja2 template (templates/*.py.j2) -> a fixed-size jupytext "# %%" .py file
        │
        ▼
render_report(py_path) -> RenderResult        # jupytext -> execute -> HTML -> PDF
        │
        ▼
ReportArtifacts (paths + loaded/skipped counts) -> ReportBundle
```

The generated `.py` file **also** calls `build_run_table`, `select_series`, `detect_redundant_metrics`,
`format_series_labels`, `order_varying_hyperparameters_by_importance`, `hyperparameter_grid_cardinality`,
`rank_runs_by_performance`, `select_top_table_metric_columns`, `format_relative_run_path`, and the
sensitivity-analysis functions (`variance_decomposition`, `hyperparameter_main_effects`,
`top_decile_comparison`, `interaction_ranking`, `interaction_grid`, `replication_status`) at
execution time — it re-walks the run directories itself
rather than embedding the loaded data, so it stays small regardless of run count and can be re-run
after further training. This makes those functions an end-user contract: a previously generated
report imports them by name, so their signatures cannot change without breaking it.

## Modules

| Module | Purpose |
|---|---|
| `__init__.py` | Discovery (`find_experiment_directories`, `is_valid_experiment_directory`), orchestration (`generate_report`, `generate_individual_report`), and the result models `ReportArtifacts`, `SkippedGroup`, `ReportBundle`. |
| `run_table.py` | `RunRecord` (one run), `SkippedRun`, `RunTable` (one report group), `ReportManifest`, `build_run_table(root)`, `format_environment_summary(...)`, `detect_redundant_metrics(records)` (bit-identical metric-pair detection), `detect_constant_metrics(records)` (one-metric-constant-in-itself detection), `format_varying_hyperparameters(record, varying_names)` (short legend/tick label for one record), `format_series_labels(records, prefixes, varying_names)` (the same, for a whole chart selection, with a pairwise-distinctness guarantee), `hyperparameter_grid_cardinality(records)` (one model family's grid size — a product-of-levels expression — vs. the runs actually present, complete/partial), `rank_runs_by_performance(records, top_n=3)` (the top-N runs by `performance`, for the summary table), `select_top_table_metric_columns(records)` (which extra metric columns beside `performance` are not a duplicate of it or of each other), `format_relative_run_path(record, report_dir)` (a run's directory relative to the report's own directory, for a short, always-unique, never-truncated path). |
| `selection.py` | `select_series(records, metric, per_bucket=3)` — bounded best/median/worst chart series selection; `SeriesBucket`, `SelectedSeries`, `SeriesSelection`. |
| `sensitivity.py` | Variance-decomposition (ANOVA / eta-squared) sensitivity analysis: `variance_decomposition` (consolidated main-effects + pairwise-interaction table), `hyperparameter_main_effects` (mean AND max per level, rendered as small multiples; `max_performance_is_saturated` detects a ceiling-saturated max), `top_decile_comparison` (full grid vs top-decile main effects), `interaction_ranking`/`interaction_grid`, `replication_status`, `order_varying_hyperparameters_by_importance` (ranks a family's varying hyperparameters by descending eta-squared, for `format_series_labels`' legend ordering). No PCA anywhere — see below for why. |
| `render.py` | `render_report(py_path)` — jupytext → execute → HTML → PDF; `RenderResult`. `check_artifacts_writable(py_path)` — writability preflight for every sibling artifact; `ArtifactWriteError` — raised when one is locked. |
| `templates/report_template.py.j2` | Individual-report template. |
| `templates/comparative_report_template.py.j2` | Comparative-report template (the one with charts and the sensitivity analysis). |

The table/selection/sensitivity logic lives in plain, unit-tested Python modules rather than inside the
`.j2` templates deliberately: ruff does not lint template bodies, and pure functions are what
`tests/reports/` actually exercises.

## The run table (`run_table.py`)

`build_run_table(root)` is the **single** loading routine, called both by the generator (to
decide whether a group qualifies and to write `report_manifest.json`) and by the generated
notebook itself at execution time — one implementation, one loop, regardless of run count.

- Reads only `environment.json` and `run_info.json` per run. **`model.json` is never opened** —
  `model_name` comes from the run directory's parent name. A report group can hold tens of
  megabytes of stored model weights (`q_network_state_dict`, `q_table`); none of it is ever
  loaded or printed.
- An unreadable run (missing file, invalid JSON, missing key) becomes a `SkippedRun(path,
  reason)`; the walk continues past it.
- `RunRecord` exposes both raw fields (`learning_rewards`, `testing_rewards`, `hyperparameters`,
  `env_kwargs`, ...) and derived aggregates as Pydantic `@computed_field`s: `mean_learning_reward`,
  `learning_success_rate`, `mean_testing_reward`, `testing_success_rate`, `performance`. These are
  `float | None`, never `0.0` as a stand-in for "no data" — a run that scored a genuine `0.0`
  must stay distinguishable from a run with no evaluation phase.
- `RunTable.to_dataframe()` produces a wide, scalar-only pandas frame (one row per run,
  `hp_<name>`/`env_<name>` columns) for tables and rankings; per-episode lists stay on the
  `RunRecord` objects rather than becoming multi-hundred-MB object columns.
- `RunTable.by_model_family()` groups records by `model_name` — the sensitivity analysis's input,
  so families with disjoint hyperparameter sets (e.g. `deep_q_learning` vs `simple_q_learning` in
  the same FrozenLake group) are never mixed into one decomposition.
- `detect_redundant_metrics(records)` → `list[MetricRedundancy]` — detects `RunRecord` derived
  metric pairs whose per-run values are exactly, bit-identically equal (never a tolerance). On a
  binary-reward environment (FrozenLake's `{0, 1}`), `mean_testing_reward` IS
  `testing_success_rate` mechanically (measured max abs difference `0.00e+00` on a real
  `deep_q_learning` family), so a chart of one is a y-axis-rescaled duplicate of the other; the
  comparative template renders only one chart per pair it detects as redundant instead of two
  identical ones. Only the two natural per-phase pairs are checked
  (`mean_learning_reward`/`learning_success_rate` and `mean_testing_reward`/`testing_success_rate`);
  on a continuous-reward environment the two genuinely differ and no pair is flagged.
- `detect_constant_metrics(records)` → `list[ConstantMetric]` — detects a `RunRecord` derived
  metric that is constant **in itself** across the whole group (exact `np.ptp`, relative
  tolerance, never `sd > 0`). Distinct from `detect_redundant_metrics`: that compares two
  *different* metrics to *each other*; this checks *one* metric against its own group. CartPole
  awards `+1` reward per step, so `learning_success_rate`/`testing_success_rate` are exactly
  `1.0` for every run there — structurally, not by chance — while the same metrics genuinely vary
  on FrozenLake. Ranking `select_series()` on a constant metric would sort purely on the
  directory-name tie-break and mislabel the result `[best]`/`[median]`/`[worst]`; the comparative
  template checks this before ranking on `learning_success_rate` and skips both the chart and the
  ranking when it fires, printing the constant value instead.
- `format_varying_hyperparameters(record, varying_names, max_length=60)` → `str` — a short,
  deterministic chart legend/tick label (e.g. `"bat_siz=64 lea_rat=0.001"`) restricted to the
  group's *varying* hyperparameters (reusing `RunTable.varying_hyperparameters`, not duplicating
  its logic), each name abbreviated to the same first-3-letters-per-word scheme as the on-disk
  directory signature (`BaseConfig.get_hyperparameters_signature`), with an ellipsis backstop cap.
  Replaces the previous label built from the *full* run-directory signature (~100 characters,
  every hyperparameter including the ones constant across the whole group), which made a chart
  legend wider than the plot it decorated.
- `format_series_labels(records, prefixes, varying_names, max_length=60)` → `list[str]` — labels
  a whole chart's selection at once and **guarantees the result is pairwise distinct**. A label
  capped to only the first few varying names (in whatever order `varying_names` gives, one
  ordered sequence per record) can otherwise collide — two different runs whose differentiating
  hyperparameter happens to be dropped for space render identical text, and a legend with two
  indistinguishable entries cannot do its job. Collisions are resolved in two steps, applied only
  to the colliding entries so most labels stay short: (1) re-render those records' hyperparameter
  text *uncapped* — resolves any collision caused by truncation alone, since two distinct runs in
  one report group differ in at least one varying hyperparameter by construction; (2) if still
  tied (the two records coincide on every name in their own `varying_names`), append each
  record's own run directory name — unique within a report group by construction
  (`RunTable`'s directory-uniqueness invariant), so this step always terminates. The comparative
  template passes each record its own model family's varying names, ordered by
  `sensitivity.order_varying_hyperparameters_by_importance(family_records, varying_names, metric)`
  — descending eta-squared share of `metric` (reusing `hyperparameter_importance()`'s
  already-sorted `entries`, never recomputing eta-squared), so the length cap in step (1)'s
  initial pass drops the *least* informative hyperparameter first rather than whichever sorts
  last alphabetically (measured case: `learning_rate` at eta-squared 40.4% — the dominant factor
  on a `deep_q_learning` family — sorted alphabetically after `batch_size`/`discount_factor`/
  `epsilon_min` and was the one a plain alphabetical cap truncated away).

- `hyperparameter_grid_cardinality(records)` → `HyperparameterGridCardinality` — one model
  family's grid size (a product-of-levels expression, e.g. `"2x3x3x3x2"` → `108` cells) compared
  against `len(records)`, reporting the grid **complete** (every cell has a run) or **partial**
  (with the exact number of missing cells). Computed **per model family**, never over the union
  of every family's varying hyperparameters in a report group: different families vary different
  hyperparameters over different value sets, so a union-based product wildly over-counts
  (measured: a FrozenLake group's 6-hyperparameter union gives `960` against `135` actual runs,
  because its two families declare disjoint hyperparameter sets, while computed per family the
  two grids are `108`/`108` and `27`/`27`, both complete). `runs_present` can never exceed `cells`
  by construction: Hercule's on-disk directory signature is a deterministic function of a run's
  hyperparameters, so two runs sharing every varying value collide on the same directory.
- `rank_runs_by_performance(records, top_n=3)` → `list[RankedRun]` — the top-N runs across every
  model family by descending `performance` (tie-broken by directory name, the same convention
  `select_series` uses), excluding runs with no usable `performance` entirely rather than sorting
  them to the bottom. Replaces a full one-row-per-run performance dump: a reader wants "who won
  and by how much", not every run's figures.
- `select_top_table_metric_columns(records)` → `list[str]` — which extra metric columns belong
  beside `performance` in the top-run table: `performance` is always shown, and this adds only
  the columns not bit-identical to it or to each other (`_metric_values_equal`, the same exact
  equality check `detect_redundant_metrics` uses). `mean_testing_reward` is dropped whenever
  every ranked run has a testing phase (`performance` is *defined* as `mean_testing_reward` then,
  so the two columns are identical by construction); `testing_success_rate` is dropped when it is
  bit-identical to whichever of those two was kept (the binary-reward identity). Replaces always
  rendering all three, which on a binary-reward environment printed the same number three times
  under three different headers.
- `format_top_table_hyperparameter_cells(records, varying_names, max_length=39)` → `list[str]` —
  the top-run table's hyperparameters column, **guaranteed pairwise distinct** among the rows
  shown. A 20-character cap previously truncated away the one hyperparameter that told two ranked
  runs apart (e.g. two `simple_q_learning` rows both rendering `"eps_min=0.05 dis_..."` while
  differing only in `learning_rate`). Shares its collision resolution with
  `format_series_labels` via the private `_disambiguate_labels` (re-render uncapped, then fall
  back to `record.run_name`) rather than a second implementation — the table has no per-row
  prefix (rank and model are their own columns), so this renders bare hyperparameter text. The
  default cap of `39` spends the ~19 characters of spare width measured between the table's
  previous line length (`69`) and a monospace PDF page's printable width (empirically `88`
  characters — confirmed by binary-searching the wrap point of a probe notebook cell). The
  comparative template further degrades when even that is not enough (a collision resolved by
  falling back to the very long uncapped or run-name text can still push a row past `88`):
  it drops `extra_metric_columns` one at a time, least informative first, then — only if every
  ranked run happens to share one model family, so the per-row `model` column repeats a single
  value and carries zero information — drops that column too and states the shared family once
  in prose instead (measured case: a `deep_q_learning` family varying 5 hyperparameters where
  the differentiating one is the least important, so disambiguating two rows needs all 5 spelled
  out — `69`-`70` characters for the hyperparameters cell alone).
- `format_relative_run_path(record, report_dir)` → `str` — a run's directory relative to the
  report's own directory (typically `<model_name>/<signature>`), forward-slashed for a
  platform-independent string. Replaces the removed `truncate_path_for_table`, which
  tail-truncated an *absolute* path to fit a table column; the table no longer has a `path`
  column at all — each ranked run's full, untruncated relative path is printed as a short
  numbered list below the table instead, so it is both compact and exactly findable.

## Bounded chart series (`selection.py`)

`select_series(records, metric, per_bucket=3)` ranks on one metric and returns at most
`3 * per_bucket` (9 by default) records: the best, the ones nearest the median, and the worst,
each tagged with its `SeriesBucket`. Sorting on `(-metric_value, directory_name)` makes the
selection deterministic even under pervasive ties (many FrozenLake runs score exactly `0`
reward) — `directory_name` is unique within a group and stable across regenerations. When a
group has `<= 9` records, all of them are returned and `omitted_count == 0`.

## Sensitivity analysis (`sensitivity.py`)

**No PCA anywhere in this module, on explicit user direction.** An earlier revision ran a PCA on
`[hyperparameters | performance]`, and before that a fully unsupervised PCA of the hyperparameter
grid alone. Both are gone, removed for two independent reasons:

1. PCA of the grid alone is **unsupervised** — it never looks at performance, so it answers "how
   does the grid vary", which on Hercule's full-cartesian sweeps is a known lattice by
   construction (hyperparameter columns are exactly orthogonal, measured max off-diagonal
   correlation `3.16e-17` on a real 108-run `deep_q_learning` family), not a discovery.
2. Even with performance appended as an active SVD variable, the same orthogonality makes the
   eigenspace beyond PC1 exactly tied (measured `evr = [0.2221, 0.1667, 0.1667, 0.1667, 0.1667,
   0.1112]`) — PC2's specific orientation was an arbitrary rotation of LAPACK's choosing, not a
   measurement, reproducible to 6 decimals under row permutation for PC1 but swinging by 0.4-0.9
   for PC2. A tool with only one trustworthy axis is not the right shape for a five-hyperparameter
   report section.

What a reader actually wants is a **variance decomposition** (ANOVA / sensitivity analysis): how
much of the performance variance each hyperparameter — and each pairwise combination — accounts
for. Eta-squared by grouping computes that exactly (no p-value, no approximation) for Hercule's
balanced full-factorial grids, using nothing beyond numpy. `sensitivity.py` provides five views,
all built on the same run table, all taking one model family's records (e.g. one value of
`RunTable.by_model_family()`) and a `metric: MetricName = "performance"` (`"performance" |
"mean_testing_reward" | "mean_learning_reward"`), and all returning a "not applicable" result with
a human-readable reason instead of raising when the family has too few runs, no varying numeric
hyperparameter, or constant performance:

1. **`variance_decomposition(records, metric)`** → `VarianceDecompositionResult |
   VarianceDecompositionUnavailable` — ONE consolidated ANOVA-style table: every main effect
   (from `hyperparameter_importance()`) AND every pure two-way interaction (from
   `interaction_ranking()`), named individually (e.g. `"batch_size:discount_factor"`), sorted by
   descending eta-squared, plus `residual_eta_squared` (three-way-and-higher interaction terms
   *and* run-to-run noise — on Hercule's default saturated design the two are mathematically
   inseparable from this data alone, see `replication_status()` below). No p-value is computed
   anywhere: a saturated design has zero residual degrees of freedom, which makes one meaningless.
2. **`hyperparameter_main_effects(records, metric)`** → `MainEffectsResult |
   MainEffectsUnavailable` — mean **and max** performance per level per hyperparameter. The mean
   alone is misleading for optimisation: measured on a real `deep_q_learning` family,
   `learning_rate`'s means are nearly flat (`0.086 / 0.067 / 0.087`, "mean says irrelevant") while
   its maxima strictly decrease (`0.420 / 0.330 / 0.260`, "max says this sets your ceiling") —
   `batch_size` even inverts (mean prefers one level, max prefers the other).
   `MainEffectsForHyperparameter.mean_and_max_disagree` flags exactly this divergence (best level
   by mean != best level by max) with no invented magnitude threshold. This view also catches a
   non-monotonic effect a linear correlation would miss or understate entirely (measured example:
   `epsilon_min` has `corr = -0.102` with performance but `eta_squared = 3.4%`, because its middle
   level is the worst, not an endpoint) — it is the authority on shape.

   The comparative template renders this as **small multiples**: one subplot per hyperparameter
   (grid sized `ncols = min(3, count)`), x-axis = that hyperparameter's own levels as categorical
   ticks labelled with the real values — never several hyperparameters overlaid on one shared
   "level index" axis. An earlier revision did exactly that (two panels, MEAN and MAX, every
   hyperparameter as a line against a shared integer index) and it was actively misleading: the
   index has no common meaning across hyperparameters, so the lines crossed at points that meant
   nothing, and value-point annotations for five overlaid lines collided into unreadable mush
   (`"5000"` and `"10000"` printed on top of each other). `max_performance_is_saturated(levels,
   relative_tolerance=0.01)` detects when a hyperparameter's max is (near-)constant *relative to
   its own scale* (never an absolute epsilon — the same `0.31` absolute spread is real signal at
   scale `~1` and pure noise at scale `500`): on CartPole every level's max sits within `0.06%` of
   the `500`-step episode cap, and plotting that on its own axis let matplotlib autoscale the
   noise into a dramatic-looking curve. When saturated, the panel omits the max *line* (drawing a
   flat ceiling reference instead, sharing the mean's own y-axis so the flatness is visually
   obvious) and the report states the saturation in prose rather than implying a real effect;
   `MainEffectsForHyperparameter.max_is_saturated` and `MainEffectsResult.all_max_saturated`
   expose the same predicate so `mean_and_max_disagree` never reports a divergence that is really
   just saturation noise (a saturated max nominally "preferring" a different level than the mean
   is an autoscaling artifact on CartPole, not a finding, and is suppressed accordingly).
3. **`top_decile_comparison(records, metric, quantile=0.9, min_top_n=8)`** → `TopDecileComparisonResult |
   TopDecileComparisonUnavailable` — reruns `hyperparameter_importance()` on the subset scoring at
   or above the `quantile`-th percentile, side by side with the full grid. "Important on average"
   is not "important where the good configurations are": measured on the same family
   (`quantile=0.9`, threshold `0.180`, `n=13`), `discount_factor`'s share collapses from 10.6%
   (full grid) to 0.6% (top decile) while `learning_rate` rises from 1.0% to 43.7% — the ranking
   inverts. `rank_shifts` reports every hyperparameter present on both sides, sorted by descending
   `abs(shift)` with no magnitude cutoff deciding which shifts are "material" — the report states
   them and lets the reader judge. Refuses (`TopDecileComparisonUnavailable`) below `min_top_n`
   runs rather than printing a decomposition drawn from a handful of points.
4. **`interaction_ranking(records, metric)`** → `InteractionRankingResult |
   InteractionRankingUnavailable` and **`interaction_grid(records, metric, first=None,
   second=None)`** → `InteractionGridResult | InteractionGridUnavailable` — every pair of
   varying hyperparameters ranked by its *pure* two-way interaction share (the joint-cell
   eta-squared minus both main effects), and a mean-performance heatmap for a pair (defaulting
   to the ranking's top pair — not simply the two individually most important hyperparameters,
   since a pair can each have a large main effect while interacting weakly, or vice versa).

**`order_varying_hyperparameters_by_importance(records, varying_names, metric)`** → `list[str]` —
not one of the five report views itself, but a small reuse of `hyperparameter_importance()` for
the comparative template's chart legends (`run_table.format_series_labels()`): reorders
`varying_names` by descending eta-squared (reading `entries`, already sorted, off the existing
result rather than recomputing it), appending any name it does not score (e.g. boolean-valued)
alphabetically afterwards. Degrades to the alphabetical order of `varying_names` — never raises —
when `records` is empty or importance itself is `ImportanceUnavailable` for this family.

`replication_status(records)` → `ReplicationStatus` (never raises, even on an empty sequence)
detects whether the family's design has any replication at all, by comparing the number of runs
against the number of *distinct hyperparameter configurations* (every hyperparameter except
`seed`). `distinct_configurations == n_samples` means the design is **saturated** — one run per
cell, zero residual degrees of freedom — so interaction effects and run-to-run noise are not
merely confounded, they are mathematically inseparable from that dataset alone; the status
carries a printable warning to that effect (and, when per-episode testing rewards are available,
a `noise_floor_fraction` estimate of how much of the observed across-run variance is plausibly
just evaluation sampling noise). This function only warns — it does not aggregate replicates;
separating interaction from noise for real requires re-running with several seeds (e.g. `seed:
[42, 43, 44]` in the YAML) and averaging performance per configuration. The noise floor stays the
criterion for "distinguishable from noise" everywhere in this module — no magic thresholds.

All five share the exact-range (`np.ptp`) constant-column test — including for **performance
itself**: centring on the mean before squaring accumulates floating-point rounding error, so
bit-identical performance values across every run can still produce a tiny nonzero sum of
squares (measured `6.16e-32`); the constant-performance guard checks `np.ptp` on the raw values,
never `sum((y - mean) ** 2) <= 0`.

## PDF rendering (`render.py`)

`render_report(py_path)`:

1. `jupytext.read(py_path, fmt="py:percent")` — parses the `.py` file's `# %%` cell markers.
2. `ExecutePreprocessor` runs it in a real `python3` kernel (bounded per-cell timeout,
   `interrupt_on_timeout=True`, `record_timing=False` so regeneration is byte-identical).
3. `HTMLExporter` with `TagRemovePreprocessor` strips cells tagged `remove_cell` entirely and
   drops the *source* (keeping the *output*) of cells tagged `remove_input` — that is how the
   PDF omits imports/loading code while keeping "loaded 218 runs, skipped 2". A second
   preprocessor, `_StripStderrPreprocessor`, rides alongside it (registered on both
   `HTMLExporter.preprocessors` and `WebPDFExporter.preprocessors`) and drops only `stream`/
   `stderr` output entries from every cell, leaving stdout, figures and text/plain results
   untouched — a warning raised during a chart cell's execution (e.g. matplotlib) lands on
   stderr as an *output* of that cell, and `remove_input` only drops a cell's *source*, so an
   unfiltered warning would still surface as visible report body text (measured incident: a
   local temp path and a kernel PID printed into the PDF). `remove_all_outputs_tags` was
   considered and rejected for this — it would drop the cell's chart or table along with the
   stray warning sharing its output list, not just the warning.
4. The HTML is printed to PDF with a system Chromium-family browser
   (`--headless=new --print-to-pdf`), found via `PATH` or the standard install locations. An
   optional Playwright-backed `WebPDFExporter` fallback is used when the `pdf` extra is
   installed. The injected print CSS (`_PRINT_CSS`) also sets `.jp-OutputArea-output pre {
   white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }`: a `<pre>` output
   defaults to `white-space: pre`, so a printed line longer than the page's printable width
   (~88 characters, measured) is silently CLIPPED by the printer rather than wrapped — measured
   incident: a 92-character run-path line lost its `__seed_42` tail with no error anywhere.
   Scoped to `.jp-OutputArea-output pre` so it cannot touch a figure (`jp-RenderedImage` output
   has no `<pre>`) or the pre-existing `break-inside: avoid` rule, and a table row that already
   fits the printable width is untouched — wrapping only engages once a line actually overflows.

Every failure path (no browser found, execution error, timeout, dead kernel) still returns the
notebook and HTML, with `pdf=None` and a `pdf_skip_reason` — the command never fails just
because a PDF could not be produced.

### Locked output artifacts (robustness)

A separate failure mode from every one above: the *destination* cannot be written at all, most
commonly because a previously generated `.pdf` is open in another program (a preview tab, an
editor — measured incident: Windows `WinError 32` on a PDF held open by a VSCode preview tab).
This is an *output* problem, not a report-content one, so it is handled distinctly:

- `check_artifacts_writable(py_path)` probes every sibling (`.ipynb`, `.failed.ipynb`, `.html`,
  `.pdf`, and `py_path` itself) by renaming it aside and immediately back — non-destructive, so
  nothing is deleted or truncated by the check itself — and returns a reason naming the first
  locked one, or `None` if all are safe to replace.
- `render_report()` runs this check **first**, before deleting or writing anything, and raises
  `ArtifactWriteError` (an `OSError` subclass) if it fails. `generate_individual_report()` and
  each group of `generate_report()`'s comparative loop run the same check before writing even
  the `.py`/`report_manifest.json` for that group — so a locked PDF never results in a freshly
  regenerated `.py` sitting beside a stale, un-updated `.pdf`: either every artifact for that
  group is from the previous successful run, or (once writable again) all replaced together.
- In `generate_report()`'s multi-group loop, this failure is caught per group and recorded as a
  `SkippedGroup` with a reason naming the locked file — **it does not abort the remaining
  groups**. A sibling group with nothing wrong with it is still generated in the same
  invocation. Only when *every* group fails this way does `generate_report()` raise
  `ArtifactWriteError` (distinct from the `ValueError` raised when every group simply had too
  few runs — see the exception contract in `controller.generate_experiment_report`'s docstring).
- `hercule.controller.generate_experiment_report()` and the `hercule report` CLI both catch
  `OSError` before the generic `ValueError` branch, so a locked output is reported as "Cannot
  write report output: ..." — never mislabelled "Invalid experiment data".

Both templates' mechanical setup cell (tagged `remove_cell`) also carries a narrow
`warnings.filterwarnings("ignore", message=r"Tight layout not applied.*", category=UserWarning)`
— filtered by message rather than blanket-silenced, so an unanticipated warning class still
reaches whoever regenerates the report by hand; `_StripStderrPreprocessor` above is the systemic
safety net for everything this narrow filter does not anticipate. The comparative template's
"Comparative Learning Progress" figure (a legend placed below the axes via
`bbox_to_anchor=(0.5, -0.16)`) used to trigger exactly this warning from
`plt.tight_layout(rect=...)`: the rect-constrained auto-fit algorithm could not always
accommodate both the suptitle and the below-axes legend and gave up with a `UserWarning` instead
of raising. It is replaced with an explicit `fig.subplots_adjust(top=0.88, bottom=0.32,
wspace=0.25)`, which sets the margins directly instead of auto-fitting into a rect, leaving no
feasibility check to fail.

### Cell-tag vocabulary

Shared as constants (`TAG_REMOVE_CELL`, `TAG_REMOVE_INPUT`, `TAG_REMOVE_OUTPUT` in
`reports/__init__.py`) between the templates and `render.py`; nbconvert ships no default tag
names, so these are the project's own fixed strings and must match exactly on both sides.

| Tag | Literal marker | Effect in the PDF |
|---|---|---|
| `remove_cell` | `# %% tags=["remove_cell"]` | Cell entirely absent — source and output. Imports, `Agg`-backend guard, directory-resolution helper. |
| `remove_input` | `# %% tags=["remove_input"]` | Source dropped, **output kept**. Every other code cell — loading/summary cells, charts, tables, the sensitivity analysis — whose printed output or figure is informative but whose Python is not something a report reader needs to see. |
| `remove_output` | `# %% tags=["remove_output"]` | Output dropped, source kept. Reserved; not used by the current templates. |

Every code cell in both templates carries one of `remove_cell` or `remove_input` — there is no
third, untagged case left. Earlier revisions left the chart/analysis cells untagged, so their
full Python source (including the `for` loops driving the sensitivity analysis) printed into the
PDF alongside the figures it produced; a PDF a researcher reads should show prose, tables and
figures, never the code that built them.

## Return contract

```python
ReportBundle
├── reports: list[ReportArtifacts]     # one per report group actually rendered
│     ├── notebook: Path               # executed .ipynb (or the .py itself if execute=False)
│     ├── html: Path
│     ├── pdf: Path | None
│     ├── pdf_skip_reason: str | None  # set iff pdf is None
│     ├── runs_loaded: int
│     └── runs_skipped: int
└── skipped_groups: list[SkippedGroup] # candidate groups NOT rendered, each with a reason
      ├── path: Path
      └── reason: str
```

`generate_report()`/`generate_individual_report()` return this; `controller.generate_experiment_report()`
returns the same `ReportBundle` (this is a public API change from the previous `Path` return —
see the feature's plan.md "Constitution Impact" note). Nothing in this contract raises for a
*partial* success: a rendered group whose PDF could not be produced is a successful
`ReportArtifacts` with `pdf=None`, and a group whose output is locked by another program is
recorded in `skipped_groups` rather than aborting the others (see "Locked output artifacts"
above). `generate_report()` raises `ValueError` when *no* report group could be generated
because every candidate had too few runs, `ArtifactWriteError` (an `OSError`) when *every*
candidate group failed to write its output, and `FileNotFoundError` when `experiment_path` does
not exist.

## Artifact layout

| Report kind | Location | Files |
|---|---|---|
| Individual | the run directory | `report.py`, `report.ipynb`, `report.html`, `report.pdf` |
| Comparative | the environment-settings directory | `comparative_report.py`, `comparative_report.ipynb`, `comparative_report.html`, `comparative_report.pdf`, `report_manifest.json` |

`report_manifest.json` is written only for comparative reports — it is the anchor file the
generated notebook's directory search verifies against at runtime, since the `env/env_params`
level has no naturally occurring file the way a run leaf directory has `environment.json`.
Regenerating a report replaces these files in place; on an execution failure, a
`<name>.failed.ipynb` is written instead (with whatever cells did run) since nbconvert does not
save the notebook when it raises.

## Usage

```python
from pathlib import Path
from hercule.reports import generate_report

bundle = generate_report(Path("outputs/frozenlake_4x4"))
for artifact in bundle.reports:
    print(artifact.notebook, artifact.pdf or artifact.pdf_skip_reason)
for skipped in bundle.skipped_groups:
    print("skipped:", skipped.path, skipped.reason)
```

Or from the CLI:

```bash
uv run hercule report outputs/frozenlake_4x4              # notebook + HTML + PDF
uv run hercule report outputs/frozenlake_4x4 --no-pdf      # notebook + HTML, skip execution's PDF step
uv run hercule report outputs/frozenlake_4x4 --no-execute  # write the .py scaffold only, fastest
```

## Notable gotchas (see `CLAUDE.md` for the full list)

- `Path(__file__)` is undefined inside a Jupyter kernel — the generated templates use a
  verifying candidate search anchored on `environment.json` (individual) or
  `report_manifest.json` (comparative) instead.
- An unconditional `matplotlib.use("Agg")` silently makes every chart disappear (`plt.show()`
  becomes a no-op) with no error — the templates guard it with
  `if "get_ipython" not in globals(): matplotlib.use("Agg")`.
- `TagRemovePreprocessor.enabled` defaults to `False` in nbconvert; `render.py` sets it
  explicitly.
- PDF success is judged by `pdf.exists() and pdf.stat().st_size > 0`, never the browser's
  return code — a bare `--print-to-pdf` against an already-running browser exits 0 having
  written nothing, which is why `--user-data-dir=<fresh temp>` is mandatory.
- A locked output artifact (e.g. a PDF open in a preview tab) is an `OSError`, not a
  `ValueError` — it must never be reported as "Invalid experiment data", and one locked group
  must never abort the other groups in the same `hercule report` invocation. See "Locked output
  artifacts" above.

## Testing

```bash
uv run pytest tests/reports                    # unit + integration
uv run pytest tests/reports -m "not slow"       # skip the PDF-round-trip test
```
