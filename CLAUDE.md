# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hercule is a framework for benchmarking Reinforcement Learning and Deep RL algorithms on
[Gymnasium](https://gymnasium.farama.org/) environments. A YAML file declares *(environments × models ×
hyperparameter grids)*; the framework expands that into a cartesian product of runs, trains each one, saves
metrics/weights into a deterministic directory tree, and can generate Jupyter reports from the results.

Package lives in `src/hercule` (import as `hercule.*`, **never** `src.hercule.*` — banned via ruff `banned-api`).
`pyscaf/` is vendored scaffolding tooling (pdoc wrappers only), not part of the framework.

**Governance — read this before touching a base class.** `AGENTS.md` defines a *Root Class Registry* (`RLModel`,
`TDModel`, `BaseConfig`, `HyperParamsBase`, `HerculeConfig`, `EpochResult`, `Runner`, `Supervisor`). Any *semantic*
change to their public API (abstract method added/removed/renamed, signature change, lifecycle contract, `@final`
methods, `ClassVar` subclasses depend on) **must** trigger a review of `.specify/memory/constitution.md`, a version
bump if amended, and a "Constitution Impact" section in the PR. `AGENTS.md` is the shared human/agent reference and
overlaps this file; the sections below add the operational detail it does not cover.

## Commands

```bash
uv sync                                  # install deps (dev group included: default-groups = "all")

uv run hercule learn experiments/frozenlake_4x4.yaml       # train + test from a config
uv run hercule learn <config.yaml> -o <dir> -vv            # override output dir, DEBUG logging
uv run hercule play <model.json> <environment.json>        # replay a trained model (Ctrl+C to stop)
uv run hercule play <model.json> <environment.json> --no-render
uv run hercule report outputs/frozenlake_4x4              # individual or comparative report (auto-detected)

uv run pytest                            # 50 tests, ~25s
uv run pytest tests/config/test_config_expansion.py::TestConfigExpansion::test_expand_model_variants
uv run pytest -m "not slow"              # markers: slow, integration, unit (--strict-markers is on)

uv run ruff check . --fix && uv run ruff format .   # line-length 120
uv run gen-doc                           # pdoc -> docs/ (git-ignored; do not commit)
uv run serve-doc
```

`docs/` is generated output and is git-ignored. The API reference is published to
<https://id2l.github.io/hercule/> by `.github/workflows/docs.yml` on every push to `main`
(`gh workflow run docs.yml` to republish). Pull requests build it as a check but never publish — a PR that
breaks `gen-doc` fails there, so keep modules importable.

Ready-made configs live in `experiments/`; results land in `outputs/`.

## Architecture

### Execution pipeline

`cli/main.py` (Click, presentation only) → `controller/` (frontend-agnostic business API, also usable from a web
API; provides `CancellationToken`) → `supervisor/` (double loop over environments × models) → `run.Runner` (epoch
loop + persistence) → `models.RLModel.run_epoch()` (the algorithm).

One **epoch = one episode**. `Supervisor.execute_learn_phase()` runs all combinations to `learn_max_epoch`, then
`execute_test_phase()` re-loads each model and runs `test_epoch` episodes with `train_mode=False`.

### Config expansion and output layout (the core mechanic)

`load_config_from_yaml()` always calls `HerculeConfig.expand_variants()`: any hyperparameter whose value is a
**list** is expanded into the cartesian product of variants (`BaseConfig.expand_variants`). So a config with
`learning_rate: [1e-4, 2.5e-4, 1e-3]` and `batch_size: [32, 64]` produces 6 independent runs.

Each run gets a unique directory from `HerculeConfig.get_directory_for()`:

```
{base_output_dir}/{config.name}/{env_name}/{env_signature}/{model_name}/{model_signature}/
    environment.json   # gym.make kwargs, from environnements.save_environment()
    model.json         # weights/Q-table + "model_name" key, from RLModel.save()
    run_info.json      # epoch counters + learning_metrics/testing_metrics, from Runner.save()
```

The signature is `BaseConfig.get_hyperparameters_signature()`: first 3 letters of each word of the param name +
its value, sorted alphabetically (`lea_rat_0.0001__bat_siz_32__…`). It raises if a value is still a list, i.e.
`expand_variants()` must run first.

**Runs are resumable and this is by design**: `Runner.load(directory)` restores epoch counters from
`run_info.json` and `model.load(directory)` restores weights, so re-running the same config continues from where
it stopped (`Runner.learn` iterates `range(learning_ongoing_epoch, max_epoch)`). Increase `learn_max_epoch` to
train further; delete the directory to start clean. `save_every_n_epoch` checkpoints mid-training.

### Models

`models/__init__.py` defines `RLModel(BaseConfig, ABC, Generic[HyperParamsType])` — a Pydantic model, so `env` is
a field and internal state uses `PrivateAttr`. Subclasses must implement `act`, `run_epoch`, `predict`, `_export`,
`_import`; `save`/`load`/`check_environment_or_raise` are `@final` (JSON via `model.json`).

Hyperparameters are dual-representation:
- typed: a `HyperParamsBase` subclass declared as `hyperparams_class: ClassVar`, reachable via
  `self.get_hyperparameters()` (autocompletion, defaults) — **prefer this inside algorithms**;
- generic: `self.hyperparameters: list[HyperParameter]`, kept in sync so it can be serialized/signed.
`configure()` merges provided values over defaults and populates both. Mutable state such as decaying `epsilon`
lives in the typed hyperparameters and is written back to both on each update (see `TDModel.run_epoch`).

Model discovery is **directory-based**: `get_available_models()` scans `src/hercule/models/*/`, imports each
package and registers every `RLModel` subclass under its `model_name` `ClassVar`. `create_model("simple_sarsa")`
resolves through that registry, and YAML `models[].name` is the same key.

Current hierarchy:
- `td_models/` — abstract `TDModel` (Q-table, ε-greedy, epoch loop); subclasses only implement `update()`:
  `simple_q_learning/` (off-policy), `simple_sarsa/` (on-policy). Requires **discrete** action *and* observation
  spaces (`environnements/spaces_checker.py`).
- `deep_q_learning/` — DQN (PyTorch): `QNetwork` picks MLP for 1-D observations / CNN for 3-D, plus
  `ExperienceReplayBuffer`.
- `dummy/` — random baseline, works on any space.

### Adding a model

1. `src/hercule/models/<name>/__init__.py`, subclass `RLModel[MyHyperParams]` (or `TDModel` for tabular TD).
2. Set `model_name: ClassVar[str] = "<name>"` and `hyperparams_class: ClassVar = MyHyperParams`.
3. Implement the abstract methods, plus `load_from_dict(model_data)` if the model must be usable with
   `hercule play` (see gotcha below).
4. Reference `<name>` in a YAML under `models:` — no registration code needed.

### Environments

`environnements/` wraps Gymnasium: `EnvironmentFactory` (validated `gym.make` + cache keyed on name+kwargs),
`EnvironmentRegistry` (static registry queries with similar-name suggestions), `EnvironmentInspector` (metadata
from `env.spec`), and `save_environment`/`load_environment` which round-trip an env through JSON by keeping only
the keys accepted by `gym.make`. That JSON is what `hercule play` consumes to rebuild the env with
`render_mode="human"`.

### Reports

`reports/generate_report()` auto-detects: a directory holding the three JSON files → individual `report.py`; a
parent directory → recursive search (`MAX_DEPTH`), grouped by environment+env-params, producing
`comparative_report.py`. Output is a Jupytext-format `.py` (`# %%` cells) rendered from Jinja2 templates in
`reports/templates/`, meant to be opened as a notebook, then **executed and rendered to HTML/PDF** as part of
report generation itself (`hercule report` is a compute step, not just a scaffold step).

Pipeline: `build_run_table(root)` (`reports/run_table.py`) walks the run directories once and reads only
`environment.json` + `run_info.json` per run — **never `model.json`**, so stored weights (`q_network_state_dict`,
`q_table`, tens of MB per group) are never loaded or printed; `model_name` comes from the run directory's parent
name. The same function is called both by the generator and by the generated notebook at runtime, so there is
exactly one loading loop regardless of run count. `select_series()` (`reports/selection.py`) caps every
multi-run chart at 9 ranked series (3 best/3 median/3 worst on that chart's own metric), sorted on
`(-metric_value, directory_name)` for determinism across regeneration.

`reports/sensitivity.py` (renamed from `pca.py` — no PCA anywhere in it) answers "which hyperparameters drive
success, and are there combinations that work well" with a **variance decomposition** (ANOVA / eta-squared by
grouping), never a PCA of any kind, mixing hyperparameters with an outcome measure or not: on Hercule's
full-cartesian grids the hyperparameter columns are exactly orthogonal by construction, so a PCA there is
unsupervised structure (never looks at performance) or, once performance is appended as an active variable, has
only one trustworthy axis (the rest of the eigenspace is exactly tied) — see Gotchas for the measured detail.
`variance_decomposition()` builds ONE consolidated table: every main effect (`hyperparameter_importance()`'s
eta-squared, exact for a balanced design) AND every pure two-way interaction (`interaction_ranking()`'s cell
eta-squared minus both main effects), named individually (e.g. `"batch_size:discount_factor"`), sorted
descending, plus a residual (three-way-and-higher terms *and* noise — inseparable on a saturated design). No
p-value is ever computed (zero residual degrees of freedom on a saturated design makes one meaningless).
`hyperparameter_main_effects()` reports mean **and max** performance per level — the mean alone is misleading for
optimisation (measured: `learning_rate`'s mean is nearly flat while its max strictly decreases across levels) —
and `MainEffectsForHyperparameter.mean_and_max_disagree` flags the divergence. `top_decile_comparison()` reruns
the main-effects decomposition on the subset scoring at or above the 90th percentile, side by side with the full
grid, since "important on average" is not "important where the good configurations are" (measured: a
hyperparameter's rank can invert entirely between the two, e.g. `discount_factor` 10.6% -> 0.6%, `learning_rate`
1.0% -> 43.7%); it refuses below `min_top_n` runs rather than printing noise from a handful of points.
`interaction_grid()` renders a heatmap defaulting to the top-ranked pair by pure interaction share, not simply the
two individually most important hyperparameters. `replication_status()` detects a **saturated** design (one run
per distinct hyperparameter configuration, zero residual degrees of freedom) and warns that interaction and
run-to-run noise are then mathematically inseparable, not merely confounded — it never aggregates replicates
itself; the estimated noise floor stays the criterion for "distinguishable from noise" everywhere, never a magic
threshold. `run_table.py`'s `detect_redundant_metrics()` detects exactly-equal (never a tolerance) `RunRecord`
metric pairs — on a binary-reward environment `mean_testing_reward` IS `testing_success_rate` mechanically — so
the comparative template renders one chart per distinct metric instead of two identical ones differing only by a
factor of 100 on the y-axis. `detect_constant_metrics()` (same module, same exact-range convention) instead
detects one metric being constant *in itself* across a whole report group (e.g. CartPole's success rate, always
`1.0`) and the template skips both that chart and its `select_series()` ranking rather than plotting/ranking on
zero information; `format_varying_hyperparameters()` renders a chart legend/tick label from only the
hyperparameters that vary within the group (abbreviated to the on-disk directory-signature convention, capped
with an ellipsis backstop), reusing `RunTable.varying_hyperparameters` rather than the full ~100-character run
signature. Two further steps sit on top of it for a **chart legend**, where a label naming only the first few
varying hyperparameters can otherwise collide: `sensitivity.order_varying_hyperparameters_by_importance()` reorders
a family's varying names by descending eta-squared (reusing `hyperparameter_importance()`'s already-sorted
`entries`, never recomputing it) so a length cap drops the *least* informative parameter first instead of
whichever sorts last alphabetically — measured case, a `deep_q_learning` family where `learning_rate` (eta-squared
40.4%, the dominant factor) sorted after `batch_size`/`discount_factor`/`epsilon_min` and was the one truncated
away; `run_table.format_series_labels()` then builds one label per selected run and *guarantees* the whole batch
is pairwise distinct — on a truncation collision it re-renders the colliding labels uncapped, and if that still
ties (two records identical on every hyperparameter passed in) appends each record's own run directory name as a
last, always-unique resort. A legend that renders two different runs with identical text cannot do its job, so
this is enforced, not left to the length cap. `hyperparameter_main_effects()`'s mean-vs-max view is rendered as **small multiples** — one panel per
hyperparameter, x-axis = that hyperparameter's own levels as categorical ticks — never multiple hyperparameters
overlaid on one shared "level index" axis (see Gotchas); `max_performance_is_saturated()` detects a
(near-)constant max relative to its own scale and the panel then omits the max line (a flat ceiling reference
instead) rather than letting it autoscale into a false effect. Every function in `sensitivity.py` returns an
"…Unavailable" result with a reason instead of raising when a family has too few runs, no varying numeric
hyperparameter, or constant performance. `render_report()` (`reports/render.py`) then runs jupytext → `ExecutePreprocessor` → tag-
filtered `HTMLExporter` → a system Chromium-family browser's `--print-to-pdf`, degrading to `pdf=None` +
`pdf_skip_reason` (never an exception) when no browser is available. `generate_report()` /
`generate_individual_report()` / `controller.generate_experiment_report()` all return a `ReportBundle`
(`reports: list[ReportArtifacts]`, `skipped_groups: list[SkippedGroup]`) — **not** a bare `Path` — so every
artifact and every skipped candidate group is reported to the caller.

The comparative template's "Hyperparameter Grid" section reports grid size **per model family**,
never over the union of every family's varying hyperparameters: `run_table.hyperparameter_grid_cardinality()`
computes one family's product-of-levels expression (e.g. `"2x3x3x3x2" = 108`) against
`len(records)`, printing the grid complete or partial (with the number of missing cells) — a
union across families wildly over-counts (measured: FrozenLake's 6-hyperparameter union gives
`960` against `135` actual runs, since `deep_q_learning` and `simple_q_learning` declare disjoint
hyperparameter sets, while computed per family the two grids are `108`/`108` and `27`/`27`, both
complete). The former "Performance Metrics Analysis" + "Winner Declaration" sections (a
one-row-per-run dump plus a duplicated winner computation) are now one "Performance Metrics"
section: `run_table.rank_runs_by_performance()` ranks every run by `performance` once, and the
top 3 are rendered as one compact table (model family, importance-ordered varying
hyperparameters, and only the metric columns `run_table.select_top_table_metric_columns()` finds
genuinely distinct from `performance` and from each other — reusing the same bit-identical
equality check `detect_redundant_metrics` uses, never a second comparison — dropping e.g.
`mean_testing_reward`/`testing_success_rate` whenever they duplicate `performance`). The
hyperparameters column renders via `run_table.format_top_table_hyperparameter_cells()`, not the
plain `format_varying_hyperparameters()` chart legends use: the same length cap could otherwise
render two DIFFERENT ranked runs with an IDENTICAL cell once the differentiating parameter fell
past it (measured: two `simple_q_learning` rows both rendering `"eps_min=0.05 dis_..."` while
differing only in `learning_rate`). It shares its collision resolution with
`run_table.format_series_labels()` via the private `_disambiguate_labels` (re-render uncapped,
then fall back to `record.run_name`) rather than a second implementation, so every displayed
row's cell is guaranteed pairwise distinct. The kept metric columns render under
`run_table.TOP_TABLE_COLUMN_LABELS`' short headers (`test_reward`, `test_success`,
`learn_reward`) rather than their full attribute names — three of those side by side (20
characters each) would push a monospace table past a monospace PDF page's printable width
(empirically `88` characters, confirmed by binary-searching the wrap point of a probe notebook
cell) on their own, before the hyperparameters or model columns even enter into it. There is no
`path` column: `run_table.truncate_path_for_table()` (a tail-truncated absolute path that ate ~50
characters of table width for a string that only repeated the hyperparameters column) is gone;
each ranked run's full, untruncated path — relative to the report's own directory via
`run_table.format_relative_run_path()`, so it stays short (`<model_name>/<signature>`) and
exactly findable — is printed as a short numbered list right below the table instead. When
guaranteeing distinctness would still push a row past `88` characters (a collision resolved by
falling back to the long uncapped or run-name text), the template drops width in a fixed order
instead of letting a row wrap mid-line: `select_top_table_metric_columns()`'s columns one at a
time, least informative first, then — only if every ranked run happens to share one model family,
so the per-row `model` column repeats a single value and carries zero information — that column
too, stating the shared family once in prose instead (measured case: a `deep_q_learning` family
varying 5 hyperparameters where the differentiating one is the least important, so disambiguating
two rows needs all 5 spelled out, ~70 characters for the cell alone). This keeps every table row
one continuous line in the rendered PDF (previously up to 200 characters, wrapping mid-row into
unreadable mush) — never the run directory as a row label, never the full hyperparameter
enumeration. Every code cell in both templates now
carries `TAG_REMOVE_CELL` or `TAG_REMOVE_INPUT`; there is no third, untagged case — a PDF a
researcher reads shows prose, tables and figures, never the Python that produced them (a chart or
sensitivity-analysis cell left untagged previously printed its full source, including the `for`
loops, into the PDF). Internal spec traceability markers (`FR-nnn`, `SC-nnn`, "(spec Assumptions)")
are never printed in generated report prose — they belong in `specs/`, not in a document an
end user reads.

## Conventions

- Code and comments in **English** (some legacy French comments/docstrings remain in `run/` and `supervisor/`).
- Python 3.10+ syntax: `X | Y` unions, including in `isinstance()`; avoid `typing.Union`/`Optional`/`Any`; be
  explicit rather than falling back to `Any`.
- Pydantic v2 with `@field_validator` + `@classmethod` (never `@validator`). Structured data is a Pydantic model,
  not a dict.
- Google-style docstrings on public functions/classes.
- Ruff selects `B, C4, E, F, N, W, I, UP, TID, TC, PLC, PLE, PLW`; isort `known-first-party = ["hercule"]`,
  2 blank lines after imports; relative imports to parents are banned.
- To exercise behaviour manually, drive the **CLI** with a config in `experiments/` — do not add ad-hoc test
  scripts at the repo root.
- Tests use `tests/conftest.py` fixtures (`temp_test_dir`, `change_to_temp_dir`, Click `runner`) and YAML fixtures
  in `tests/fixtures/`.

## Gotchas

- **`.cursor/rules/*.mdc` and `src/hercule/run/README.md` are partly stale.** They document a `TrainingRunner` /
  `ModelExecutor` / `RunManager` API and a `benchmark/` module that no longer exist — the real orchestration is
  `Supervisor` + `Runner`. Their *style* and *design-rationale* sections are still authoritative; their code
  samples are not. Check the source before following them.
- `load_from_dict()` is used by `controller.play_interactive()` but is **not** declared on `RLModel`; it is
  implemented per model (`TDModel`, `DummyModel`, DQN). A new model without it breaks `hercule play` only.
- `TDModel.configure()` returns `False` (instead of raising) when the env spaces are not discrete, and
  `Supervisor` ignores the return value — pairing a tabular model with e.g. `CartPole-v1` fails later with an
  unrelated error. Validate the pairing in the config.
- `EnvironmentManager.load_environment()` still contains debug `print()` calls; `Supervisor` bypasses that class
  and uses `EnvironmentFactory` directly.
- Model persistence is JSON, so `save_every_n_epoch` on a large DQN writes big files; tune it per experiment.
- The repo is **Ruff-clean** as of 2026-07-28 (`check` and `format --check` both pass). Keep it that way: a new
  violation is yours. Note `ruff format` also reformats Python snippets inside `.md` files.
- CLI output is emoji-heavy (`🎯 📊 ✅ …`). `harden_output_streams()` in `cli/main.py` runs from the group
  callback to keep that safe on non-UTF-8 streams — without it every command died with `UnicodeEncodeError` on
  its first message under a cp1252 stdout. If you add a new entry point that prints markers without going
  through the `cli` group, call it there too.
- `EpochResult` in `reports/` and `Path` in `controller/` live in `TYPE_CHECKING` blocks. That is safe *because*
  they appear only in annotations that are never evaluated at runtime (method-body attribute annotations, and
  `controller` has `from __future__ import annotations`). Do not move a name used by a **Pydantic** field
  annotation into `TYPE_CHECKING` — Pydantic resolves those at runtime and it will raise.
- Speckit workflow: feature specs live in `specs/`, commands in `.cursor/commands/speckit.*.md`.
- **`Path(__file__)` is undefined inside a Jupyter kernel.** `sys.argv[0]` resolves to
  `ipykernel_launcher.py`, so a bare `__file__`-derived path resolves into site-packages. Generated report
  templates locate their own data directory with a *verifying* candidate search anchored on a file that must
  exist (`environment.json` for an individual report, `report_manifest.json` for a comparative one — the
  `env/env_params` level has no naturally occurring anchor). nbclient sets the kernel's working directory from
  `resources["metadata"]["path"]` passed to `ExecutePreprocessor.preprocess()`; if you omit it, the notebook
  inherits the shell's cwd instead.
- An **unconditional** `matplotlib.use("Agg")` in a report template silently destroys every chart: `plt.show()`
  becomes a no-op, so the figure is never a `display_data` output — 0 `<img>` tags in the HTML, 0 images in the
  PDF, **no error raised**. The guard `if "get_ipython" not in globals(): matplotlib.use("Agg")` is mandatory,
  not stylistic.
- Asserting `"image/png" in html` is an **invalid** test for "did the chart render" — it false-positives on a
  CodeMirror CSS artifact (`.cm-trailingspace` background-image). Assert on embedded images in the PDF via
  `pypdf` (`sum(len(page.images) for page in reader.pages)`) instead.
- `TagRemovePreprocessor.enabled` defaults to **`False`** in nbconvert — it must be set explicitly or nothing is
  stripped. `remove_input_tags` drops a cell's *source* but **keeps its output** — that is the mechanism by
  which the PDF drops mechanical loading code while keeping "loaded N runs, skipped M".
- PDF printing shells out to a system Chromium-family browser. `shutil.which()` finds **nothing** on a stock
  Windows box even when Edge/Chrome are installed — they must also be probed at their standard install paths
  (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`, etc.). `--user-data-dir=<fresh temp dir>` is
  **mandatory**: if a browser instance is already running (near-certain on a dev workstation), a bare
  `--print-to-pdf` hands off to it and **exits 0 having written nothing**. Verify success by
  `pdf.exists() and pdf.stat().st_size > 0`, never the return code.
- Detecting a constant hyperparameter column with `sd > 0` is **wrong**: `np.std(ddof=1)` on a genuinely
  constant column (e.g. a repeated `0.0005`) can return a tiny nonzero float (`~2.18e-19`) from floating-point
  rounding error, silently keeping a dead column in the decomposition. Use an exact range test
  (`np.ptp(X, axis=0) > 1e-12 * scale`), not the standard deviation. The same bug recurs on the *performance*
  side of `reports/sensitivity.py`'s views: `sum((y - y.mean()) ** 2) <= 0` is **not** a valid "is
  performance constant" test either — centring on the mean before squaring accumulates rounding error, so 20
  bit-identical performance values (e.g. every run scoring exactly `2/5 = 0.4`) can still produce a tiny nonzero
  sum of squares (measured `6.16e-32`), which would silently print a nonsensical "100% eta-squared" instead of
  "not applicable". Check `np.ptp(y)` on the raw values, exactly as for hyperparameter columns.
- **PCA of hyperparameters — with or without performance mixed in — was removed from `reports/sensitivity.py`
  (formerly `pca.py`) on explicit user direction, and should not be reintroduced.** Two independent reasons, both
  measured on a real 108-run `deep_q_learning` family: (1) a PCA of the hyperparameter grid alone is
  **unsupervised** — it never looks at performance, and since Hercule expands configs into full cartesian
  products, the hyperparameter columns are exactly orthogonal by construction (max off-diagonal correlation
  `3.16e-17`), so that projection is mostly a known lattice, not a discovery; (2) even with performance appended
  as an active SVD variable, the same orthogonality forces `p - 1` eigenvalues to exactly 1, so the eigenspace
  beyond PC1 is exactly tied (`evr = [0.2221, 0.1667, 0.1667, 0.1667, 0.1667, 0.1112]`) — PC2's specific
  orientation was an arbitrary LAPACK rotation, not a measurement (stable to 6 decimals under row permutation for
  PC1, swinging by 0.4-0.9 for PC2). What the report needs is a variance decomposition (ANOVA / eta-squared by
  grouping), which needs no projection at all and is exact for a balanced design.
- **"Important on average" is not "important where the good configurations are" — the average-vs-top-decile
  trap.** A hyperparameter's eta-squared share on the full grid can be nearly the opposite of its share among the
  best-scoring runs. Measured on the same family (`top_decile_comparison()`, `quantile=0.9`, threshold `0.180`,
  `n=13`): `discount_factor` collapses from 10.6% (full grid) to 0.6% (top decile) while `learning_rate` rises
  from 1.0% to 43.7% — the ranking inverts entirely. If the goal is to *optimise* rather than describe average
  behaviour, the top-decile column is the one that matters, not the full-grid one. `top_decile_comparison()`
  refuses below `min_top_n` runs (default 8) rather than printing a decomposition drawn from a handful of points.
- **The mean alone is misleading for optimisation — a hyperparameter's mean and its maximum attainable
  performance can disagree on which level is best.** Measured on `learning_rate` in the same family: means are
  nearly flat (`0.086 / 0.067 / 0.087` — "mean says irrelevant") while the maxima strictly decrease (`0.420 /
  0.330 / 0.260` — "max says this sets your ceiling"); `batch_size` even inverts outright (mean prefers one
  level, max prefers the other). `hyperparameter_main_effects()` reports both series and
  `MainEffectsForHyperparameter.mean_and_max_disagree` flags the divergence with no invented magnitude threshold
  — it is a straight "best level by mean != best level by max" comparison.
- **A binary reward makes `mean_testing_reward` mechanically equal `testing_success_rate`** (and likewise for
  the learning-phase pair): `mean(rewards) == mean(rewards > 0)` whenever every reward is in `{0, 1}` (e.g.
  FrozenLake), measured max abs difference `0.00e+00` on a real `deep_q_learning` family. Rendering both as
  separate charts used to produce two visually identical curves differing only by a factor of 100 on the y-axis.
  `run_table.detect_redundant_metrics()` detects exactly-equal (never a tolerance) metric pairs and the
  comparative template renders only one chart per distinct metric; on a continuous-reward environment the two
  genuinely differ and both charts still appear.
- nbconvert does **not** save the notebook when execution raises — the traceback would otherwise be lost
  entirely. Write the partial `<name>.failed.ipynb` yourself in the exception handler.
- Do **not** change the global asyncio event loop policy to silence the benign Proactor `RuntimeWarning` that
  every nbclient execution emits on Windows (`add_reader` not implemented). Doing so would strip asyncio
  subprocess support, which the optional Playwright PDF fallback needs, and is a global side effect on any host
  embedding `hercule.controller`. Suppress narrowly with `warnings.catch_warnings()` around the `preprocess()`
  call instead.
- **Classifying hyperparameter relevance by a magic threshold is wrong, whatever the underlying measure.** A
  previous revision bucketed hyperparameters into "same direction as performance" / "opposite" / "unrelated"
  with a hard-coded `+/-0.5` cutoff on a PCA loading. On a real 108-run `deep_q_learning` family,
  `replay_buffer_size` sat at `+0.498` — 0.002 short of the cutoff — and was printed as "unrelated to
  performance" despite having the family's *second-largest* eta-squared (4.1%, behind only `discount_factor`'s
  10.6%). A loading (or any single continuous score) can sit arbitrarily close to any fixed cutoff, so no
  magnitude threshold on it can reliably decide whether a hyperparameter matters. `sensitivity.py` never buckets
  by a threshold: strength is **eta-squared** everywhere (`hyperparameter_importance()`'s own scale-free
  importance measure, correct for a non-monotonic effect a linear reading would understate); "distinguishable
  from noise" compares that eta-squared against the **estimated noise floor** (`ReplicationStatus.noise_floor_fraction`)
  instead of an invented constant — on a saturated design this is a deliberately harsh test (only
  `discount_factor` at 10.6% clears a measured ~7.8% floor) and must not be softened to make more parameters
  look significant. The same "no magic threshold" rule applies to `top_decile_comparison()`'s rank shifts (sorted
  by magnitude, not filtered by a cutoff) and to `MainEffectsForHyperparameter.mean_and_max_disagree` (an exact
  "best level differs" comparison, not a tolerance).
- **A "max-attainable" view is meaningless once an environment's ceiling is reached almost everywhere, and
  matplotlib's autoscale will hide that fact rather than reveal it.** CartPole caps every episode at 500 steps,
  so a hyperparameter's per-level *maximum* performance sits at `499.69-500.00` for every level — an amplitude
  of `0.31` on a scale of `500`, i.e. `0.06%`. Plotted on its own y-axis, matplotlib autoscales that noise into
  a steep-looking curve indistinguishable from a real effect, and the "best level by max" it implies is an
  artifact of which level happened to round to `500.0` first, not a measurement. Detect this with a **relative**
  range test (`max_performance_is_saturated()` in `sensitivity.py`, `np.ptp(maxima) <= relative_tolerance *
  max(abs(maxima).max(), 1.0)`, default `1%`) — never an absolute epsilon, since the same `0.31` absolute spread
  is a real signal at scale `~1` (FrozenLake) and pure saturation noise at scale `500` (CartPole). When
  saturated, the report omits the max line from that hyperparameter's panel (drawing a flat ceiling reference
  instead) and states the saturation in prose; `MainEffectsForHyperparameter.mean_and_max_disagree` also checks
  this before reporting a mean/max divergence, since a saturated max "preferring" a different level than the
  mean is autoscaling noise, not a finding.
- **A metric can be structurally constant for a given environment, not just by chance in one dataset — and a
  ranked "best/median/worst" selection on it is then meaningless, not merely uninformative.** CartPole awards
  `+1` reward per step, so every episode scores `> 0` and `learning_success_rate`/`testing_success_rate` are
  exactly `1.0` for every run there; the same two metrics are genuinely informative on FrozenLake. Ranking
  `select_series()` on a constant metric sorts entirely on the directory-name tie-break and labels the result
  `[best]`/`[median]`/`[worst]` as if that meant something — actively misleading, not just a wasted chart. This
  is distinct from `detect_redundant_metrics()` (two *different* metrics bit-identical to *each other*):
  `run_table.detect_constant_metrics()` detects *one* metric being constant in itself, via the same exact-range
  (`np.ptp`), relative-tolerance test used throughout this module, never `sd > 0` or an absolute epsilon. The
  comparative template checks it before ranking on `learning_success_rate` and skips both the chart and the
  selection when it fires, printing the constant value and an explicit "no ranking is meaningful" statement
  instead.
- **A printed line longer than the printable page width is CLIPPED, not wrapped, by a PDF export's default
  CSS — silently losing content, not just looking ugly.** A `<pre>` output defaults to `white-space: pre`; a
  line longer than the printable page width (~88 characters, measured) then overflows the page box and the
  printer clips it with no error and no visible sign of the loss. Measured incident: the run-path list under
  the top-3 table extracted from the PDF at exactly 88 characters with its `__seed_42` tail gone, while the
  same line in the notebook's own stdout was the full 92 characters. Fixed systemically in the print CSS
  `render.py` injects (`_PRINT_CSS`), not by trimming any one line: `.jp-OutputArea-output pre { white-space:
  pre-wrap; overflow-wrap: anywhere; word-break: break-word; }`, scoped to output `<pre>` blocks only so it
  cannot touch a figure (`jp-RenderedImage` output has no `<pre>`) or the pre-existing `break-inside: avoid`
  rule, and so a table row that already fits the printable width renders as one line exactly as before —
  wrapping only engages once a line actually overflows.
- **A warning emitted during a report's own execution becomes visible report body text unless filtered — a
  `remove_input`-tagged cell is not enough.** `remove_input` drops only a cell's *source*; it keeps the
  *output* by design (that is how the PDF keeps "loaded N runs, skipped M" while hiding the loading code), so
  a `UserWarning` raised during that same cell's execution lands on stderr as an output too and prints straight
  into the exported HTML/PDF. Measured incident: matplotlib's "Tight layout not applied" warning, printed
  verbatim into a report's PDF body together with a local temp path and a kernel PID. Two independent fixes,
  both required: (1) `render.py`'s `_StripStderrPreprocessor`, registered alongside `TagRemovePreprocessor` on
  both `HTMLExporter.preprocessors` and `WebPDFExporter.preprocessors`, drops only `stream`/`stderr` output
  entries from every cell — `remove_all_outputs_tags` was considered and rejected, since it would delete that
  cell's chart or table along with the stray warning sharing its output list; (2) each template's mechanical
  setup cell (tagged `remove_cell`) also calls `warnings.filterwarnings(...)` narrowly by message, stopping the
  specific known warning at its source rather than blanket-silencing every warning class. The root cause of the
  measured warning was `plt.tight_layout(rect=...)` failing to reconcile a manually reserved rect with a legend
  placed *outside* the axes via `bbox_to_anchor` (its feasibility check gives up with a `UserWarning` instead of
  raising); replacing it with an explicit `fig.subplots_adjust(top=..., bottom=..., wspace=...)` sets the
  margins directly, leaving no feasibility check to fail. `constrained_layout=True` was tried first and
  rejected: verified by rendering the exact figure (9 series, `bbox_to_anchor` legend below two side-by-side
  axes) that constrained_layout does not shrink the axes to make room for a below-axes `bbox_to_anchor` legend
  the way it does for a `fig.legend()` or an in-bounds `ax.legend()` — the two axes collapsed to slivers with a
  large empty gap between them and both legends overlapped into unreadable text.
- **On Windows, a PDF held open by a preview tab (VSCode, a PDF viewer, ...) holds an exclusive lock, and
  artifact replacement around that lock must be all-or-nothing per report group.** Measured incident:
  regenerating `hercule report outputs/dq_cartpole` with `comparative_report.pdf` open in a VSCode preview
  raised `[WinError 32] The process cannot access the file because it is being used by another process` from
  the *old* code's unconditional `stale.unlink(missing_ok=True)` in `render.py`, which ran *before* anything new
  was written. Three compounding defects, all fixed together: (1) the unhandled exception aborted the *entire*
  `hercule report` invocation, so a sibling group with nothing wrong with it (`sut_bar_rew_True`, no lock) was
  never generated at all, even though the CLI/`ReportBundle` contract already models per-group
  success/skip — Defect 1's fix is that a locked group is now caught and recorded as a `SkippedGroup` with a
  reason, and the remaining groups still run; (2) the old code deleted the stale `.ipynb`/`.html`/`.pdf` before
  the new ones existed, so an abort left a *freshly regenerated* `.py` beside a *stale* `.pdf` with no
  `.ipynb`/`.html` at all — worse than a missing file, since nothing signals the PDF is out of date. The fix
  (`reports/render.py`'s `check_artifacts_writable()`) is a non-destructive preflight — rename each sibling
  aside and immediately back — run *before* deleting or writing anything, for every artifact including the
  `.py`/`report_manifest.json` themselves; a locked destination aborts that group with nothing touched, so its
  artifacts are always either all from the previous successful run or (once writable again) all replaced
  together, never a fresh/stale mix; (3) the write failure was an `OSError` from an *output* problem, but the
  previous `except Exception -> ValueError("Failed to generate report: ...")` in
  `controller.generate_experiment_report` collapsed it into the CLI's `except ValueError` branch, printing
  "Invalid experiment data" for data that loaded perfectly fine. `ArtifactWriteError` (an `OSError` subclass,
  `reports/render.py`) is raised instead and caught in its own branch — before the `ValueError` branch, since
  `FileNotFoundError`/`PermissionError` are themselves `OSError` subclasses and must be listed first if they
  need different wording — in both `controller.generate_experiment_report` and the `hercule report` CLI, so a
  locked output is reported as "Cannot write report output: ..." and names the actionable fix (close the
  program holding the file) rather than blaming invalid input.
