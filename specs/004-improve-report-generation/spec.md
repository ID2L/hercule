# Feature Specification: Improved Experiment Report Generation

**Feature Branch**: `004-improve-report-generation`
**Created**: 2026-07-28
**Status**: Draft
**Input**: User description: "Améliorer la génération des rapports : présenter l'environnement Gymnasium dans un texte imprimé (nom du module gym) ; remplacer la lecture individuelle dupliquée de chaque dossier par une itération sur les sous-dossiers stockant les données dans un tableau dynamique ; borner les graphiques à trop de courbes en sélectionnant top 3 / médian 3 / last 3 pour la métrique concernée ; ajouter une représentation ACP sur les hyperparamètres ; générer le PDF du notebook en plus du notebook, en sautant les blocs de code non intéressants mais en gardant ceux informatifs."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Identify the environment at a glance (Priority: P1)

A researcher opens a generated report and immediately wants to know which Gymnasium
environment the results belong to — the environment's identifier ("CartPole-v1",
"FrozenLake-v1") stated as readable text, together with the environment settings that
distinguish this group of runs from another (map size, slipperiness, episode step limit).
Today that information is only reachable by reading a raw configuration dump, and the
comparative report shows it buried in a per-run listing.

**Why this priority**: It is the first thing the reader needs in order to interpret every
other number in the document, and a report that does not say what it is about cannot be
shared or archived. It is also independent of every other story.

**Independent Test**: Generate a report for any run directory in `outputs/` and confirm the
rendered document names the environment identifier and its settings in prose, without the
reader having to inspect a raw configuration block.

**Acceptance Scenarios**:

1. **Given** a report group for `outputs/frozenlake_4x4/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200`,
   **When** the report is generated and rendered, **Then** the document states the
   environment identifier `FrozenLake-v1` and the settings `map_name=4x4`,
   `is_slippery=True`, `max_episode_steps=200` as readable text.
2. **Given** a report group for `outputs/dq_cartpole/CartPole-v1/default` whose environment
   takes no extra settings, **When** the report is rendered, **Then** it states
   `CartPole-v1` and indicates that no environment-specific setting was overridden, rather
   than printing an empty structure.
3. **Given** an individual single-run report, **When** it is rendered, **Then** it names the
   environment identifier in the same readable form as the comparative report.

---

### User Story 2 - One data-loading loop instead of one block per run (Priority: P1)

A maintainer inspecting a generated notebook expects to see a single, readable loop that
walks the run directories below the report and collects every run into one tabular
structure. Today the generator emits a separate, near-identical ~35-line loading block for
every run, with the run index baked into every variable name, all inside a single cell. For
a group of 135 runs that is several thousand lines of duplicated code in one cell, which is
both unreadable and impractical to open or execute.

**Why this priority**: It is the structural precondition for the reader-facing improvements
— ranked curve selection and hyperparameter analysis both need all runs available in one
table — and it is what makes a report over a large hyperparameter grid usable at all.

**Independent Test**: Generate a report for the 135-run FrozenLake group and confirm the
generated document contains exactly one run-loading loop whose length does not grow with the
number of runs, and that the resulting table has one row per run with the run's identity,
settings and metrics addressable as columns.

**Acceptance Scenarios**:

1. **Given** a group containing 135 runs, **When** the report is generated, **Then** the
   generated document contains one loading loop and its size is within a small constant of
   the size generated for a 2-run group.
2. **Given** a generated report, **When** its loading step runs, **Then** a single table
   holds one row per discovered run, carrying at least the run's model name, its
   hyperparameter values, and its learning and evaluation metrics.
3. **Given** a run directory in the group whose results are missing or unreadable, **When**
   the loading step runs, **Then** that run is reported as skipped with its path and the
   remaining runs are still loaded.
4. **Given** a group of runs whose stored model weights are large, **When** the loading step
   runs, **Then** model weight payloads are not loaded into the table and are never printed
   in the document.

---

### User Story 3 - Comparison charts stay readable at any grid size (Priority: P1)

A researcher comparing a hyperparameter sweep wants the learning-progress charts to remain
legible. With hundreds of runs the current comparison charts draw two curves per run, which
produces an unreadable mass of overlapping lines and an oversized legend. Instead, for each
comparison chart the researcher wants the runs ranked on the metric being plotted and only a
representative subset drawn: the 3 best, the 3 closest to the median, and the 3 worst — at
most 9 curves — with the legend saying which group each curve belongs to and the report
stating how many runs were left out.

**Why this priority**: This is the most visible defect: the charts that are supposed to be
the core of a comparative report currently convey nothing on realistic sweeps.

**Independent Test**: Generate the comparative report for the 135-run FrozenLake group and
confirm that every multi-run comparison chart draws at most 9 curves, that the selected runs
are the ranked best/median/worst triples for that chart's metric, and that the document
states the number of runs not shown.

**Acceptance Scenarios**:

1. **Given** a group of 135 runs, **When** a learning-progress comparison chart is rendered,
   **Then** it draws at most 9 curves and the legend identifies each as best, median or
   worst on that chart's metric.
2. **Given** a group of 5 runs, **When** a comparison chart is rendered, **Then** all 5 runs
   are drawn and no run is reported as omitted.
3. **Given** two charts ranking on different metrics, **When** both are rendered, **Then**
   each selects its own subset according to its own metric.
4. **Given** many runs sharing an identical ranking value, **When** the subset is selected,
   **Then** the selection is deterministic across repeated generation of the same report.
5. **Given** a group whose runs were trained for different numbers of episodes, **When** a
   comparison chart is rendered, **Then** curves of differing length are drawn correctly
   without silently truncating or padding data.

---

### User Story 4 - See which hyperparameters drive performance (Priority: P2)

A researcher who has swept several hyperparameters at once wants a compact picture of how
the hyperparameter combinations relate to each other and to the outcome. They want a
principal-component projection of the runs' hyperparameter sets, with each run positioned by
its hyperparameters and visually encoded by its achieved performance, plus a statement of how
much variance each retained component explains and how each hyperparameter contributes to
those components.

**Why this priority**: It is a genuinely new analytical capability rather than a fix, and it
is only meaningful once the run table exists. High value for reading a sweep, but the report
is already useful without it.

**Independent Test**: Generate the report for the deep-Q sweep and confirm the document shows
a two-component projection of the runs, an explained-variance figure, and per-hyperparameter
contributions, with performance visible on the projection.

**Acceptance Scenarios**:

1. **Given** a model family whose runs vary in at least 2 hyperparameters, **When** the
   report is rendered, **Then** it shows a two-component projection with one point per run,
   the explained variance of each component, and each hyperparameter's contribution to them.
2. **Given** hyperparameters that do not vary across the runs (for example a fixed seed),
   **When** the projection is computed, **Then** those hyperparameters are excluded and the
   report states which were excluded and why.
3. **Given** a model family with fewer than 2 runs or fewer than 2 varying hyperparameters
   (for example the random baseline, which varies only its seed), **When** the report is
   rendered, **Then** it states that the projection is not applicable for that family and the
   rest of the report is unaffected.
4. **Given** a group containing several model families with different hyperparameter sets,
   **When** the projection is rendered, **Then** each family is projected separately rather
   than mixing incompatible hyperparameter sets into one projection.
5. **Given** a hyperparameter whose values are not numeric, **When** the projection is
   computed, **Then** it is either represented in a documented numeric form or excluded with
   a stated reason, and never silently dropped.

---

### User Story 5 - Read and share the report without opening Jupyter (Priority: P2)

A researcher wants to circulate results to colleagues who do not run the framework. In
addition to the notebook, they want a PDF of the same report, containing the narrative text,
the tables, the charts and the informative printed output, but not the mechanical code —
library imports, file loading, model reconstruction. Code whose printed output carries
information stays represented by that output.

**Why this priority**: It is additive distribution value on top of a report that already
works; the notebook remains the primary artifact.

**Independent Test**: Generate the report for a group in `outputs/` and confirm a PDF is
produced next to the notebook, that it contains every chart and table present in the
notebook, and that it contains no import or data-loading code listing.

**Acceptance Scenarios**:

1. **Given** a report group, **When** the report is generated, **Then** both a notebook and a
   PDF of that report are produced and the user is told where each is.
2. **Given** the generated PDF, **When** it is inspected, **Then** every chart and every
   table shown in the notebook is present, and the narrative text is preserved.
3. **Given** the generated PDF, **When** it is inspected, **Then** no library-import or
   data-loading code listing appears, while the printed output that carries information
   (summary statistics, comparison tables, conclusions) is present.
4. **Given** an environment where the PDF conversion cannot run, **When** the report is
   generated, **Then** the notebook is still produced and the user is told the PDF was
   skipped and why, and the command does not fail.
5. **Given** a report over the largest available result set, **When** the report is
   generated, **Then** generation completes and reports its progress rather than appearing to
   hang.

---

### Edge Cases

- **Group with a single run**: a comparative report needs at least two runs to compare; such
  a group is skipped today with only a log line. The user must be told which groups were
  skipped and why, so a silent absence of output is never mistaken for success.
- **Runs of unequal length in one group**: charts, rankings and aggregates must not assume a
  common episode count. Measured note: within each *group*, the existing corpus is uniform
  (`simple_games` groups are all 200 episodes, `frozenlake_4x4` all 5000) — the counts differ
  only *between* configs. Because runs are resumable, raising `learn_max_epoch` and re-running
  a subset produces mixed lengths inside one group, so this stays a required behaviour; it must
  be covered by a synthetic fixture rather than by the real datasets.
- **Runs with no evaluation phase**: reports must degrade to the learning-phase sections
  instead of failing when the evaluation metrics list is empty.
- **All runs scoring identically**: many FrozenLake runs score a mean reward of 0; ranked
  selection must still produce a deterministic, non-empty subset.
- **Zero-variance and single-valued hyperparameters**: `seed`, `epsilon`, `step_modulo` and
  `weight_decay` are constant across the CartPole sweep; any normalisation must not divide by
  zero.
- **Disjoint hyperparameter sets in one group**: `frozenlake_4x4` mixes deep Q-learning with
  tabular Q-learning; a comparison table spanning both must show absent hyperparameters as
  absent rather than as a value.
- **Large stored weights**: a single report group holds ~55 MB of network weights across its
  runs; these must be neither aggregated into the table nor printed into the document.
- **Large result volume**: the largest single group holds ~65 MB of episode metrics across 135
  runs, and one command invocation may cover 218 runs / ~211 MB across two groups; reports must
  remain generatable and openable at that size.
- **Non-UTF-8 console**: report progress messages use the same emoji-bearing style as the rest
  of the CLI and must not break on a legacy code page.
- **Report regenerated over an existing report**: regenerating must replace the previous
  notebook and PDF rather than accumulating stale artifacts alongside them.

## Requirements *(mandatory)*

### Functional Requirements

**Environment presentation**

- **FR-001**: Every generated report MUST state, as readable text, the Gymnasium environment
  identifier of the runs it covers (for example `CartPole-v1`, `FrozenLake-v1`).
- **FR-002**: Every generated report MUST state the environment settings that characterise its
  group, and MUST state explicitly when no setting was overridden.
- **FR-003**: The environment statement MUST appear in both the single-run report and the
  multi-run comparative report, in the same readable form.

**Consolidated run table**

- **FR-004**: A generated report MUST discover the runs it covers by walking the directory
  tree below the report location, rather than by containing one pre-generated block per run.
- **FR-005**: A generated report MUST contain exactly one run-loading routine, whose size does
  not grow with the number of runs covered.
- **FR-006**: The loading routine MUST produce a single tabular structure with one row per
  run, exposing at least: the run's directory, its model name, its hyperparameter values, its
  environment identifier and settings, its learning metrics and its evaluation metrics.
- **FR-007**: The loading routine MUST NOT read or retain stored model weights, and no report
  MUST print stored model weights.
- **FR-008**: The loading routine MUST skip and report any run whose results cannot be read,
  and MUST continue with the remaining runs.
- **FR-009**: All subsequent sections of a report — statistics, tables, charts and analyses —
  MUST derive from that single table rather than re-reading run directories.

**Bounded chart series**

- **FR-010**: Any chart that would otherwise draw one series per run MUST limit itself to at
  most 9 runs, chosen as the 3 best, the 3 nearest the median, and the 3 worst on the metric
  that chart presents.
- **FR-011**: When a group holds 9 or fewer runs, all of them MUST be drawn and none reported
  as omitted.
- **FR-012**: Each chart MUST rank on the metric it presents, so charts presenting different
  metrics MAY select different runs.
- **FR-013**: The legend of such a chart MUST identify which selection group each drawn run
  belongs to, and the report MUST state how many runs were not drawn.
- **FR-014**: The selection MUST be deterministic: regenerating a report over unchanged
  results MUST select the same runs, including when ranking values tie.
- **FR-015**: Charts MUST correctly draw runs of differing episode counts on shared axes.

**Hyperparameter analysis**

- **FR-016**: A comparative report MUST present a two-component principal-component projection
  of the covered runs' hyperparameter sets, one point per run.
- **FR-017**: The projection MUST encode each run's achieved performance visually, so that the
  relationship between hyperparameter region and outcome is readable.
- **FR-018**: The report MUST state the proportion of variance explained by each retained
  component, and each hyperparameter's contribution to those components.
- **FR-019**: Hyperparameters that take a single value across the covered runs MUST be excluded
  from the projection, and the report MUST state which were excluded and why.
- **FR-020**: Runs MUST be projected per model family, so families with different
  hyperparameter sets are never mixed into one projection.
- **FR-021**: When a family has fewer than 2 runs or fewer than 2 varying hyperparameters, the
  report MUST state that the projection is not applicable for that family and MUST continue
  rendering the rest of the report.
- **FR-022**: Non-numeric hyperparameter values MUST either be represented in a stated numeric
  form or be excluded with a stated reason.

**PDF output**

- **FR-023**: Generating a report MUST produce a PDF rendering of that report in addition to
  the notebook, stored alongside it.
- **FR-024**: The PDF MUST contain the report's narrative text, all its tables, all its charts,
  and the printed output that carries information.
- **FR-025**: The PDF MUST omit code whose only role is mechanical — imports, file discovery,
  data loading, model reconstruction — while retaining the output such code produced where
  that output is informative.
- **FR-026**: When the PDF cannot be produced, the command MUST still deliver the notebook,
  MUST tell the user the PDF was skipped and why, and MUST NOT report overall failure.
- **FR-027**: The command MUST report the location of every artifact it produced.
- **FR-028**: Regenerating a report MUST replace previously generated artifacts for that report
  rather than leaving stale ones beside them.

**Behaviour preserved**

- **FR-029**: The existing report entry point MUST keep its current interface: a path to a
  single run directory produces a single-run report; a path to a parent directory produces one
  comparative report per environment-settings group.
- **FR-030**: The command MUST tell the user which candidate groups were skipped and why,
  including groups holding too few runs to compare.
- **FR-031**: Generated notebooks MUST be openable and executable in a standard notebook
  environment, and executing a generated report MUST NOT raise.
- **FR-032**: Reports MUST be generatable from the existing result directories without
  re-running any training.

### Key Entities

- **Run**: one trained model in one environment configuration with one hyperparameter set; the
  leaf of the output tree. Carries an environment identifier and settings, a model name,
  hyperparameter values, learning-phase episode metrics and evaluation-phase episode metrics.
- **Report group**: the set of runs sharing an environment and its settings; the unit a
  comparative report covers.
- **Run table**: the tabular collection of all runs in a report group, built once when the
  report is opened and consumed by every section of it.
- **Episode metric**: one episode's outcome within a run — its reward, its number of steps, and
  how it ended.
- **Model family**: the algorithm a run belongs to; determines which hyperparameters exist and
  therefore which runs can be projected together.
- **Chart series selection**: the ranked best/median/worst subset of a report group chosen for
  one chart, derived from that chart's metric.
- **Report artifacts**: the notebook and the PDF produced for a report group.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A report generated over a 135-run group is no more than 10% longer than a report
  generated over a 2-run group of the same kind, measured in lines of the generated document.
  (Today it grows by roughly 35 lines per run.)
- **SC-002**: No chart in any generated report displays more than 9 data series.
- **SC-003**: Every generated report names its Gymnasium environment identifier in readable
  text within the first screen of the document.
- **SC-004**: Reports generate successfully for all three existing result sets — the CartPole
  sweep (2 groups of 109 runs), the FrozenLake sweep (1 group of 135 runs) and the mixed-model
  set (2 groups of 11 runs) — producing 5 comparative reports in total, with no errors and no
  skipped group left unexplained.
- **SC-005**: Every generated report executes end to end without raising, on all three existing
  result sets.
- **SC-006**: A PDF is produced for every generated notebook, contains 100% of that notebook's
  charts and tables, and contains no import or data-loading code listing.
- **SC-007**: For a group where at least 2 hyperparameters vary, the report shows the
  projection, its explained variance and its hyperparameter contributions; for a group where
  they do not, the report states why the projection is absent.
- **SC-008**: Report generation for the largest result set (`dq_cartpole` — 218 runs, ~211 MB of
  metrics, 2 groups) completes within 10 minutes on a developer workstation and emits progress
  output at least every 30 seconds.
- **SC-009**: Regenerating a report twice over unchanged results produces identical chart
  series selections.
- **SC-010**: No generated report contains stored model weights in its output.

## Assumptions

- **"Last 3" is read as the 3 worst-ranking runs** on the chart's metric, not the 3 most
  recently produced; the stated intent was to bracket a large population by best, typical and
  worst.
- **Ranking metric per chart** is the aggregate of the quantity that chart draws — for a reward
  chart, mean reward over the run; for a success-rate chart, the run's success rate — computed
  over the run's learning phase for learning charts and its evaluation phase for evaluation
  charts.
- **The run table is built when the report is opened**, from the result files on disk, rather
  than being embedded into the generated document; this keeps the generated document small and
  lets it be re-run after further training. Embedding ~211 MB of metrics into a generated
  document is not viable.
- **The PDF is produced per notebook**, named after it and placed beside it — one PDF per
  comparative group, matching how notebooks are already placed.
- **Producing a PDF containing rendered charts requires executing the report**, so report
  generation becomes a compute step rather than a pure scaffold step. This is inherent to the
  request for a PDF with charts.
- **Which code is "mechanical" is decided when the report is generated**, not inferred at
  conversion time, so the decision is explicit and reviewable.
- **Hyperparameter values are read from the stored run results**, which already record each
  run's hyperparameters by name; no change to how training records results is needed.
- **Performance encoding on the projection** uses the run's mean evaluation reward, falling back
  to mean learning reward when a run has no evaluation phase.
- **Existing result directories remain readable**: the feature adds no requirement to re-train,
  and no stored file format changes.
- **Pre-existing defects that prevent a generated report from executing are in scope**, because
  a PDF cannot be produced from a report that cannot run. This covers the malformed conditional
  section in the single-run report and the chart calls that fail on the pinned plotting library
  version.

## Out of Scope

- Changing how training results are written to disk, or the directory layout of results.
- Changing the `learn` or `play` commands.
- Adding new metrics to what training records per episode.
- Interactive or web-hosted report viewing.
- Statistical significance testing or automated hyperparameter recommendation beyond the
  descriptive projection described above.
- Producing a single consolidated document spanning multiple environments.

## Dependencies

- Existing result sets under `outputs/` are the validation corpus; no training run is required
  to exercise this feature.
- Hyperparameter values and episode metrics are already recorded per run by the existing
  training pipeline.
- Rendering a PDF from a report requires document-conversion capability in the development
  environment; FR-026 governs behaviour when it is unavailable.
