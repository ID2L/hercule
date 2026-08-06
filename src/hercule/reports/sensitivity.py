"""Variance-decomposition (ANOVA-style) sensitivity analysis per model family (User Story 4).

Earlier revisions of this module instead projected the hyperparameter grid -- alone, and later
with performance folded in as an active variable -- onto principal components. Both projections
are gone: on explicit user direction, this report no longer runs any PCA that mixes hyperparameters
with an outcome measure. A PCA-based view was the wrong tool for "which hyperparameters drive
success" for two independent reasons, not one:

1. It answered the **unsupervised** question, not attribution. Projecting the grid alone never
   looks at performance at all -- it answers "how does the grid vary", a known lattice by
   construction on Hercule's full-cartesian sweeps, not a discovery.
2. Even folding performance in as an active variable inherited a structural defect from the same
   orthogonality: with the hyperparameter block exactly orthogonal by construction, the
   correlation matrix's eigenspace beyond the first component is exactly tied (measured on a real
   108-run `deep_q_learning` family: `evr = [0.2221, 0.1667, 0.1667, 0.1667, 0.1667, 0.1112]`), so
   the second component's specific orientation was an arbitrary rotation, not a measurement --
   reproducible to 6 decimals under row permutation for the first component, swinging by 0.4-0.9
   for the second. A tool that only ever has one trustworthy axis is not the right shape for a
   five-hyperparameter report section.

What a reader actually wants is a **variance decomposition** (ANOVA / sensitivity analysis):
how much of the performance variance does each hyperparameter -- and each pairwise combination --
account for. That is exactly what eta-squared by grouping computes, exactly (no p-value, no
approximation) for Hercule's balanced full-factorial grids, and needs nothing beyond numpy. This
module now provides five views built on that one primitive:

1. `variance_decomposition()` -- ONE consolidated ANOVA-style table: every main effect and every
   pure two-way interaction, named individually, sorted by descending eta-squared, plus a final
   residual row (three-way-and-higher terms plus noise -- the design is saturated, so the two are
   inseparable; see `replication_status()`).
2. `hyperparameter_main_effects()` -- mean **and max** performance per level per hyperparameter.
   The mean alone is misleading for optimisation: a hyperparameter can have a flat mean (mean
   says "irrelevant") while its maximum strictly decreases across levels (max says "this sets
   your ceiling") -- measured on `learning_rate` in a real `deep_q_learning` family. Both series
   are on the model, not just plotted, so this is unit-testable without touching matplotlib.
3. `top_decile_comparison()` -- reruns the main-effects decomposition on the subset scoring at or
   above the 90th percentile, side by side with the full grid. "Important on average" is not
   "important where the good configurations are": on the same family, `discount_factor`'s share
   collapses from 10.6% (full grid) to 0.6% (top decile) while `learning_rate`'s rises from 1.0%
   to 43.7% -- the ranking inverts. If the goal is to optimise rather than describe, the top-decile
   column is the one that matters. Refuses (with a reason) below `min_top_n` runs rather than
   printing noise from a handful of points.
4. `interaction_ranking()` / `interaction_grid()` -- unchanged from the previous revision: every
   pair of hyperparameters ranked by its *pure* two-way interaction share, and a mean-performance
   heatmap for the strongest pair, annotated with cell means and run counts.
5. `replication_status()` -- unchanged: detects whether the design has any replication at all
   (several seeds per hyperparameter configuration); on Hercule's default saturated design (one
   run per cell) interaction and run-to-run noise are not merely confounded but mathematically
   inseparable, and estimates a noise floor so "distinguishable from noise" has a stated,
   non-magic criterion.

Every function returns an "...Unavailable" result with a human-readable reason instead of raising
when a family has too few runs, no varying numeric hyperparameter, or constant performance -- an
exception in a middle notebook cell would otherwise block every cell below it. The exact-range
(`np.ptp`) constant-column test is used throughout, never `sd > 0` or `sum((y - mean) ** 2) <= 0`
(see `_drop_near_constant_columns` and `_prepare_family`'s docstrings for the measured floating-
point failure modes both naive tests have already caused on this exact codebase).
"""

from collections.abc import Sequence
from itertools import combinations
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from hercule.reports.run_table import RunRecord


MetricName = Literal["performance", "mean_testing_reward", "mean_learning_reward"]
"""The name of the `RunRecord` derived aggregate the sensitivity views rank/attribute on."""


# ---------------------------------------------------------------------------------------------
# Shared plumbing: numeric-column extraction, constant-column drop, per-family preparation.
# ---------------------------------------------------------------------------------------------


def _numeric_hyperparameter_columns(records: Sequence[RunRecord]) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Extract per-run numeric (non-bool) hyperparameter columns, aligned with `records` order.

    A hyperparameter is kept only if it is present in **every** record and its value is
    `int | float` (never `bool`, since `isinstance(True, int)` is `True`) in every record.

    Args:
        records: The runs to extract columns from, in the order the returned lists are aligned to.

    Returns:
        `(numeric_columns, dropped_columns)`: `numeric_columns` maps a kept hyperparameter name to
        one `float` per record; `dropped_columns` maps an excluded name to a human-readable reason.
    """
    all_keys = sorted({key for record in records for key in record.hyperparameters})

    numeric_columns: dict[str, list[float]] = {}
    dropped_columns: dict[str, str] = {}

    for key in all_keys:
        values: list[float] = []
        reason: str | None = None
        for record in records:
            if key not in record.hyperparameters:
                reason = "not present in every run"
                break
            value = record.hyperparameters[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                reason = "not numeric in every run"
                break
            values.append(float(value))
        if reason is not None:
            dropped_columns[key] = reason
        else:
            numeric_columns[key] = values

    return numeric_columns, dropped_columns


def _drop_near_constant_columns(
    numeric_columns: dict[str, list[float]], n_samples: int
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Drop near-constant columns using the exact-range test, never `sd > 0`.

    `numpy.std(ddof=1)` on a genuinely constant column (e.g. a repeated `0.0005`) can return a
    tiny nonzero float (`~2.18e-19`) from floating-point rounding error in its sum of squares, so
    a naive `sd > 0` keep test would wrongly retain it. `np.ptp` (max - min) is exact for identical
    values and immune to this failure mode; the relative-tolerance term additionally rejects
    columns whose spread is pure numerical noise relative to the column's own magnitude, rather
    than a real grid axis.

    Args:
        numeric_columns: Per-hyperparameter-name list of `n_samples` floats, from
            `_numeric_hyperparameter_columns`.
        n_samples: Number of rows (runs) each column holds.

    Returns:
        `(kept_columns, x_kept, dropped_columns)`: `kept_columns` sorted alphabetically,
        `x_kept` the `(n_samples, len(kept_columns))` matrix of their raw values, and
        `dropped_columns` mapping each dropped constant column to a reason naming its value.
    """
    numeric_keys = sorted(numeric_columns)
    if not numeric_keys:
        return [], np.zeros((n_samples, 0)), {}

    x_matrix = np.array([numeric_columns[key] for key in numeric_keys], dtype=float).T

    column_range = np.ptp(x_matrix, axis=0)
    column_scale = np.maximum(np.abs(x_matrix).max(axis=0), 1.0)
    keep_mask = column_range > 1e-12 * column_scale

    kept_columns = [key for key, keep in zip(numeric_keys, keep_mask, strict=True) if keep]
    dropped_columns: dict[str, str] = {}
    for key, keep, value in zip(numeric_keys, keep_mask, x_matrix[0], strict=True):
        if not keep:
            dropped_columns[key] = f"single value across the {n_samples} runs of this family ({value:g})"

    return kept_columns, x_matrix[:, keep_mask], dropped_columns


def _finite_metric_records(records: Sequence[RunRecord], metric: MetricName) -> tuple[list[RunRecord], list[float]]:
    """Filter to records whose `metric` is set and finite, preserving order.

    Args:
        records: The runs to filter.
        metric: The `RunRecord` derived aggregate to read (`performance`, `mean_testing_reward`,
            or `mean_learning_reward`); all three are `float | None`.

    Returns:
        `(finite_records, finite_values)`, aligned one-to-one, holding only the records whose
        metric was not `None` and not `nan`/`inf`.
    """
    finite_records: list[RunRecord] = []
    finite_values: list[float] = []
    for record in records:
        value = getattr(record, metric)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        finite_records.append(record)
        finite_values.append(value)
    return finite_records, finite_values


def _one_way_eta_squared(y: np.ndarray, labels: np.ndarray, grand_mean: float, ss_total: float) -> float:
    """Eta-squared of one column against `y`: `SS_between / SS_total`.

    `SS_between = sum over distinct levels v of n_v * (mean_v - grand_mean)^2`. Exact only for a
    balanced design (every level combination equally represented); on Hercule's full-cartesian
    grids that holds by construction.

    Args:
        y: The `(n,)` metric values.
        labels: The `(n,)` raw values of the hyperparameter column to group by.
        grand_mean: `y.mean()`, passed in so callers computing several columns share it.
        ss_total: `sum((y - grand_mean) ** 2)`, passed in for the same reason; must be `> 0`.

    Returns:
        The eta-squared share, in `[0, 1]`.
    """
    ss_between = 0.0
    for level in np.unique(labels):
        mask = labels == level
        n_v = int(mask.sum())
        mean_v = float(y[mask].mean())
        ss_between += n_v * (mean_v - grand_mean) ** 2
    return ss_between / ss_total


def _joint_eta_squared(
    y: np.ndarray, col_a: np.ndarray, col_b: np.ndarray, grand_mean: float, ss_total: float
) -> float:
    """Eta-squared of the *joint* (cartesian) grouping by two columns together.

    Treats every distinct `(col_a, col_b)` pair as one level of a single combined factor -- this
    captures both main effects and their interaction. `_joint_eta_squared(...) -
    _one_way_eta_squared(a) - _one_way_eta_squared(b)` is the pure two-way interaction share.

    Args:
        y: The `(n,)` metric values.
        col_a: The `(n,)` raw values of the first hyperparameter column.
        col_b: The `(n,)` raw values of the second hyperparameter column.
        grand_mean: `y.mean()`.
        ss_total: `sum((y - grand_mean) ** 2)`; must be `> 0`.

    Returns:
        The joint eta-squared share, in `[0, 1]`.
    """
    joint = np.stack([col_a, col_b], axis=1)
    ss_between = 0.0
    for row in np.unique(joint, axis=0):
        mask = np.all(joint == row, axis=1)
        n_v = int(mask.sum())
        mean_v = float(y[mask].mean())
        ss_between += n_v * (mean_v - grand_mean) ** 2
    return ss_between / ss_total


class _FamilyPrep(NamedTuple):
    """Shared preprocessing result for every sensitivity view (avoids repeating the same five
    steps -- filter by finite metric, extract numeric columns, drop constants -- in each
    function)."""

    model_name: str
    finite_records: list[RunRecord]
    y: np.ndarray
    kept_columns: list[str]
    x_kept: np.ndarray
    dropped_columns: dict[str, str]
    grand_mean: float
    ss_total: float
    performance_is_constant: bool


def _prepare_family(records: Sequence[RunRecord], metric: MetricName) -> _FamilyPrep:
    """Run the shared preprocessing pipeline: filter to a finite metric, extract numeric
    hyperparameter columns, drop near-constant ones. Never raises; callers apply their own
    minimum-size/variance guards on the result.

    Args:
        records: The runs of one model family; must be non-empty (checked by the public caller).
        metric: The `RunRecord` derived aggregate to attribute/rank on.

    Returns:
        The `_FamilyPrep` every public function in this module builds its guard checks from.
    """
    model_name = records[0].model_name
    finite_records, finite_metric = _finite_metric_records(records, metric)

    numeric_columns, dropped_columns = _numeric_hyperparameter_columns(finite_records)
    n_samples = len(finite_records)
    kept_columns, x_kept, constant_dropped = _drop_near_constant_columns(numeric_columns, n_samples)
    dropped_columns.update(constant_dropped)

    y = np.array(finite_metric, dtype=float)
    grand_mean = float(y.mean()) if n_samples else 0.0
    ss_total = float(np.sum((y - grand_mean) ** 2)) if n_samples else 0.0

    # Whether performance is constant must be decided on the EXACT range of the raw values
    # (np.ptp), never on `ss_total <= 0` or `sd > 0`: centering on the mean before squaring
    # accumulates floating-point rounding error, so 20 bit-identical performance values (e.g.
    # every run scoring exactly 2/5 = 0.4) can still produce a tiny nonzero `ss_total` (measured
    # `6.16e-32` on real data) purely from that rounding -- the same failure mode already fixed
    # once for hyperparameter columns (see `_drop_near_constant_columns`), reintroduced here if
    # the constancy check is done on `ss_total` instead of the raw values.
    if n_samples:
        performance_range = float(np.ptp(y))
        performance_scale = max(float(np.abs(y).max()), 1.0)
        performance_is_constant = performance_range <= 1e-12 * performance_scale
    else:
        performance_is_constant = True

    return _FamilyPrep(
        model_name,
        finite_records,
        y,
        kept_columns,
        x_kept,
        dropped_columns,
        grand_mean,
        ss_total,
        performance_is_constant,
    )


# ---------------------------------------------------------------------------------------------
# View: hyperparameter_importance -- eta-squared per hyperparameter (main effects only).
#
# Kept as a building block: `variance_decomposition()` folds its entries into the consolidated
# table, and `top_decile_comparison()` calls it directly on both the full grid and the top-decile
# subset. Not rendered on its own by the comparative template anymore -- see
# `variance_decomposition()` for the report's actual [1] view.
# ---------------------------------------------------------------------------------------------


class HyperparameterEtaSquared(BaseModel):
    """One hyperparameter's eta-squared share of the performance variance."""

    name: str
    eta_squared: float

    @field_validator("eta_squared")
    @classmethod
    def _validate_range(cls, value: float) -> float:
        tolerance = 1e-9
        if not (-tolerance <= value <= 1.0 + tolerance):
            raise ValueError(f"eta_squared out of [0, 1] range: {value}")
        return value


class ImportanceResult(BaseModel):
    """Per-hyperparameter eta-squared for one model family, exact for a balanced design.

    `eta_squared` for column `j` is `sum over distinct levels v of n_v * (mean_v -
    grand_mean)^2`, divided by the total sum of squares of the metric -- the one-way-ANOVA
    effect size of that hyperparameter alone, ignoring every other factor. This decomposition is
    **exact only for a balanced design** (every level combination equally represented, which
    holds by construction on Hercule's full-cartesian grids); on a ragged or resumed-subset grid
    the main effects would not sum cleanly and the residual would absorb the imbalance too.
    """

    model_name: str
    metric: MetricName
    n_samples: int = Field(ge=1)
    entries: list[HyperparameterEtaSquared] = Field(default_factory=list)
    main_effects_sum: float
    interaction_residual: float
    dropped_columns: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sorted_and_totals(self) -> "ImportanceResult":
        tolerance = 1e-6
        values = [entry.eta_squared for entry in self.entries]
        if values != sorted(values, reverse=True):
            raise ValueError("entries must be sorted by descending eta_squared")
        if abs(sum(values) - self.main_effects_sum) > tolerance:
            raise ValueError("main_effects_sum must equal the sum of entries' eta_squared")
        if abs((self.main_effects_sum + self.interaction_residual) - 1.0) > tolerance:
            raise ValueError("main_effects_sum + interaction_residual must equal 1.0")
        return self


class ImportanceUnavailable(BaseModel):
    """The alternative return of `hyperparameter_importance()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def hyperparameter_importance(
    records: Sequence[RunRecord], metric: MetricName = "performance"
) -> ImportanceResult | ImportanceUnavailable:
    """Attribute the variance of `metric` to each varying numeric hyperparameter (eta-squared).

    A building block for `variance_decomposition()` (adds pairwise interactions) and
    `top_decile_comparison()` (calls this twice, once per subset) -- see either for the report's
    actual rendered views.

    Args:
        records: The runs of one model family (e.g. one value of `RunTable.by_model_family()`).
        metric: The `RunRecord` derived aggregate to attribute variance of.

    Returns:
        `ImportanceResult` when at least one hyperparameter varies numerically, `metric` is
        finite and non-constant across at least 2 runs; otherwise `ImportanceUnavailable` with a
        human-readable reason. Never raises for a degenerate shape.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("hyperparameter_importance requires at least one record")

    prep = _prepare_family(records, metric)
    n_finite = len(prep.finite_records)

    if n_finite < 2:
        return ImportanceUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(f"only {n_finite} run(s) of {prep.model_name} have a finite {metric}; need at least 2"),
        )
    if not prep.kept_columns:
        return ImportanceUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"no hyperparameter varies numerically across the {n_finite} runs of {prep.model_name} with a "
                f"finite {metric}; nothing to attribute"
            ),
        )
    if prep.performance_is_constant:
        return ImportanceUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"{metric} is constant ({prep.y[0]:g}) across the {n_finite} runs of {prep.model_name} with a "
                f"finite value; there is no variance to attribute"
            ),
        )

    entries = [
        HyperparameterEtaSquared(
            name=name,
            eta_squared=_one_way_eta_squared(prep.y, prep.x_kept[:, j], prep.grand_mean, prep.ss_total),
        )
        for j, name in enumerate(prep.kept_columns)
    ]
    entries.sort(key=lambda entry: (-entry.eta_squared, entry.name))

    main_effects_sum = float(sum(entry.eta_squared for entry in entries))
    interaction_residual = 1.0 - main_effects_sum

    return ImportanceResult(
        model_name=prep.model_name,
        metric=metric,
        n_samples=n_finite,
        entries=entries,
        main_effects_sum=main_effects_sum,
        interaction_residual=interaction_residual,
        dropped_columns=prep.dropped_columns,
    )


def order_varying_hyperparameters_by_importance(
    records: Sequence[RunRecord], varying_names: Sequence[str], metric: MetricName = "performance"
) -> list[str]:
    """Order `varying_names` by descending eta-squared share of `metric` (defect A).

    A chart legend built from `format_varying_hyperparameters`/`format_series_labels` caps its
    label length -- alphabetical order (`RunTable.varying_hyperparameters`' natural order) then
    truncates whichever parameter happens to sort last, which is exactly as likely to be the
    single most informative one as the least (measured: a `deep_q_learning` family where
    `learning_rate` -- eta-squared `40.4%`, the dominant factor -- sorted alphabetically after
    `batch_size`/`discount_factor`/`epsilon_min` and was the one cut). Ordering by importance
    first means a length cap always drops the LEAST informative parameters.

    Reuses `hyperparameter_importance()` -- eta-squared is not recomputed here, only its
    `entries` (already sorted by descending eta-squared) are read off.

    Args:
        records: The runs to attribute variance across -- typically one model family's records
            (`RunTable.by_model_family()`), since different families may declare different
            hyperparameters and an importance ranking is only meaningful within one family.
        varying_names: The names to order; every name is present exactly once in the result.
        metric: The `RunRecord` derived aggregate `hyperparameter_importance` attributes.

    Returns:
        `varying_names` reordered: importance-ranked names first (descending eta-squared, as
        computed by `hyperparameter_importance`), then any remaining name from `varying_names`
        not scored there (e.g. boolean-valued, or absent from some record) in alphabetical
        order. Never raises; degrades to the alphabetical order of `varying_names` when
        `records` is empty or `hyperparameter_importance` returns `ImportanceUnavailable`
        (e.g. fewer than 2 runs, or no numeric hyperparameter varies).
    """
    if not records:
        return sorted(varying_names)

    result = hyperparameter_importance(records, metric)
    if isinstance(result, ImportanceUnavailable):
        return sorted(varying_names)

    varying_name_set = set(varying_names)
    ranked = [entry.name for entry in result.entries if entry.name in varying_name_set]
    remaining = sorted(varying_name_set - set(ranked))
    return ranked + remaining


# ---------------------------------------------------------------------------------------------
# View 4 (interaction primitives, needed by variance_decomposition below): interaction_ranking /
# interaction_grid -- pure two-way interactions.
# ---------------------------------------------------------------------------------------------


class PairwiseInteraction(BaseModel):
    """One pair of hyperparameters' pure two-way interaction share, plus both main effects.

    `interaction_eta_squared = cell_eta_squared - first_eta_squared - second_eta_squared`, where
    `cell_eta_squared` is the eta-squared of the *joint* grouping (every distinct `(first,
    second)` combination as one level). Printing both main effects alongside the interaction
    lets a reader tell whether a pair's cell differences are genuinely a combination effect or
    just the two additive main effects added together.
    """

    first: str
    second: str
    interaction_eta_squared: float
    first_eta_squared: float = Field(ge=0.0)
    second_eta_squared: float = Field(ge=0.0)
    cell_eta_squared: float = Field(ge=0.0)


class InteractionRankingResult(BaseModel):
    """Every pair of varying hyperparameters, ranked by descending pure interaction share."""

    model_name: str
    metric: MetricName
    n_samples: int = Field(ge=1)
    pairs: list[PairwiseInteraction] = Field(default_factory=list)

    @property
    def total_pairwise_interaction(self) -> float:
        """Sum of every pair's pure interaction share -- everything beyond additive main
        effects that is still explained by *two*-way combinations; the remainder (`1 -
        main_effects_sum - total_pairwise_interaction`) sits in three-way-and-higher terms plus
        noise."""
        return sum(pair.interaction_eta_squared for pair in self.pairs)

    @model_validator(mode="after")
    def _validate_sorted(self) -> "InteractionRankingResult":
        values = [pair.interaction_eta_squared for pair in self.pairs]
        if values != sorted(values, reverse=True):
            raise ValueError("pairs must be sorted by descending interaction_eta_squared")
        return self


class InteractionRankingUnavailable(BaseModel):
    """The alternative return of `interaction_ranking()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def interaction_ranking(
    records: Sequence[RunRecord], metric: MetricName = "performance"
) -> InteractionRankingResult | InteractionRankingUnavailable:
    """Rank every pair of varying numeric hyperparameters by their pure two-way interaction share.

    Args:
        records: The runs of one model family.
        metric: The `RunRecord` derived aggregate to attribute variance of.

    Returns:
        `InteractionRankingResult` when at least 2 hyperparameters vary numerically and `metric`
        is finite and non-constant across at least 2 runs; otherwise `InteractionRankingUnavailable`
        with a human-readable reason. Never raises for a degenerate shape.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("interaction_ranking requires at least one record")

    prep = _prepare_family(records, metric)
    n_finite = len(prep.finite_records)

    if n_finite < 2:
        return InteractionRankingUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(f"only {n_finite} run(s) of {prep.model_name} have a finite {metric}; need at least 2"),
        )
    if len(prep.kept_columns) < 2:
        return InteractionRankingUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"only {len(prep.kept_columns)} hyperparameter(s) vary numerically across the {n_finite} runs of "
                f"{prep.model_name}; a pairwise interaction needs at least 2"
            ),
        )
    if prep.performance_is_constant:
        return InteractionRankingUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"{metric} is constant ({prep.y[0]:g}) across the {n_finite} runs of {prep.model_name} with a "
                f"finite value; there is no variance to attribute"
            ),
        )

    single_eta = {
        name: _one_way_eta_squared(prep.y, prep.x_kept[:, j], prep.grand_mean, prep.ss_total)
        for j, name in enumerate(prep.kept_columns)
    }

    pairs = []
    for i, j in combinations(range(len(prep.kept_columns)), 2):
        first_name, second_name = prep.kept_columns[i], prep.kept_columns[j]
        cell_eta = _joint_eta_squared(prep.y, prep.x_kept[:, i], prep.x_kept[:, j], prep.grand_mean, prep.ss_total)
        interaction_eta = cell_eta - single_eta[first_name] - single_eta[second_name]
        pairs.append(
            PairwiseInteraction(
                first=first_name,
                second=second_name,
                interaction_eta_squared=interaction_eta,
                first_eta_squared=single_eta[first_name],
                second_eta_squared=single_eta[second_name],
                cell_eta_squared=cell_eta,
            )
        )

    pairs.sort(key=lambda pair: (-pair.interaction_eta_squared, pair.first, pair.second))

    return InteractionRankingResult(model_name=prep.model_name, metric=metric, n_samples=n_finite, pairs=pairs)


class InteractionCell(BaseModel):
    """One cell of an interaction grid: the mean performance (if any run landed there) of one
    `(first_level, second_level)` combination."""

    first_level: float
    second_level: float
    mean_performance: float | None
    n_runs: int = Field(ge=0)


class InteractionGridResult(BaseModel):
    """Mean performance for every combination of two hyperparameters' levels.

    `interaction_eta_squared`/`first_eta_squared`/`second_eta_squared`/`cell_eta_squared` mirror
    `PairwiseInteraction` for the specific pair shown, so the heatmap's caption can state whether
    the visible cell differences are a genuine combination effect or just the two main effects
    added together.
    """

    model_name: str
    metric: MetricName
    first: str
    second: str
    first_levels: list[float] = Field(default_factory=list)
    second_levels: list[float] = Field(default_factory=list)
    cells: list[InteractionCell] = Field(default_factory=list)
    interaction_eta_squared: float
    first_eta_squared: float = Field(ge=0.0)
    second_eta_squared: float = Field(ge=0.0)
    cell_eta_squared: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "InteractionGridResult":
        if self.first == self.second:
            raise ValueError("first and second must differ")
        if len(self.cells) != len(self.first_levels) * len(self.second_levels):
            raise ValueError("cells must have one entry per (first_level, second_level) combination")
        return self


class InteractionGridUnavailable(BaseModel):
    """The alternative return of `interaction_grid()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def interaction_grid(
    records: Sequence[RunRecord],
    metric: MetricName = "performance",
    first: str | None = None,
    second: str | None = None,
) -> InteractionGridResult | InteractionGridUnavailable:
    """Build a mean-performance heatmap over two hyperparameters' levels.

    Defaults `first`/`second` to the pair with the **strongest pure two-way interaction**
    (`interaction_ranking()`'s top entry) -- not simply the two individually most important
    hyperparameters, since a pair can each have a large main effect while interacting weakly (or
    vice versa): on the measured `deep_q_learning` family, `discount_factor` and
    `replay_buffer_size` are individually the two strongest main effects but interact at only
    0.6%, while `batch_size` x `discount_factor` interacts at 6.5% despite `batch_size` alone
    explaining almost nothing.

    Args:
        records: The runs of one model family.
        metric: The `RunRecord` derived aggregate to average per cell.
        first: Name of the first hyperparameter axis; defaults to the top interaction pair.
        second: Name of the second hyperparameter axis; defaults likewise. Both must be given
            together or both left `None` -- `interaction_ranking()`'s pair is used as a whole
            when either is omitted.

    Returns:
        `InteractionGridResult` when the (defaulted or given) pair both vary numerically and
        `metric` is finite and non-constant across at least 2 runs; otherwise
        `InteractionGridUnavailable` with a human-readable reason. Never raises.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("interaction_grid requires at least one record")

    prep = _prepare_family(records, metric)
    n_finite = len(prep.finite_records)

    ranking = interaction_ranking(prep.finite_records, metric=metric) if n_finite else None

    if first is None or second is None:
        if isinstance(ranking, InteractionRankingUnavailable):
            return InteractionGridUnavailable(model_name=prep.model_name, metric=metric, reason=ranking.reason)
        if ranking is None or not ranking.pairs:
            return InteractionGridUnavailable(
                model_name=prep.model_name,
                metric=metric,
                reason=f"no pair of hyperparameters is available to default to for {prep.model_name}",
            )
        top_pair = ranking.pairs[0]
        first = first or top_pair.first
        second = second or top_pair.second

    if first == second:
        return InteractionGridUnavailable(
            model_name=prep.model_name, metric=metric, reason="first and second hyperparameters must differ"
        )
    missing = [name for name in (first, second) if name not in prep.kept_columns]
    if missing:
        return InteractionGridUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"{', '.join(missing)} do(es) not vary numerically across the {n_finite} runs of {prep.model_name} "
                f"with a finite {metric}"
            ),
        )
    if prep.performance_is_constant:
        return InteractionGridUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"{metric} is constant ({prep.y[0]:g}) across the {n_finite} runs of {prep.model_name} with a "
                f"finite value; every cell would show the same mean"
            ),
        )

    first_idx = prep.kept_columns.index(first)
    second_idx = prep.kept_columns.index(second)
    first_column = prep.x_kept[:, first_idx]
    second_column = prep.x_kept[:, second_idx]

    first_eta = _one_way_eta_squared(prep.y, first_column, prep.grand_mean, prep.ss_total)
    second_eta = _one_way_eta_squared(prep.y, second_column, prep.grand_mean, prep.ss_total)
    cell_eta = _joint_eta_squared(prep.y, first_column, second_column, prep.grand_mean, prep.ss_total)
    interaction_eta = cell_eta - first_eta - second_eta

    first_levels = sorted(np.unique(first_column).tolist())
    second_levels = sorted(np.unique(second_column).tolist())

    cells = []
    for level_a in first_levels:
        for level_b in second_levels:
            mask = (first_column == level_a) & (second_column == level_b)
            n_runs = int(mask.sum())
            mean_performance = float(prep.y[mask].mean()) if n_runs > 0 else None
            cells.append(
                InteractionCell(
                    first_level=level_a, second_level=level_b, mean_performance=mean_performance, n_runs=n_runs
                )
            )

    return InteractionGridResult(
        model_name=prep.model_name,
        metric=metric,
        first=first,
        second=second,
        first_levels=first_levels,
        second_levels=second_levels,
        cells=cells,
        interaction_eta_squared=interaction_eta,
        first_eta_squared=first_eta,
        second_eta_squared=second_eta,
        cell_eta_squared=cell_eta,
    )


# ---------------------------------------------------------------------------------------------
# View [1]: variance_decomposition -- the consolidated ANOVA-style table (main effects AND
# pure 2-way interactions, one sorted list, plus a residual row).
# ---------------------------------------------------------------------------------------------


class VarianceDecompositionEntry(BaseModel):
    """One row of the consolidated decomposition table: a single hyperparameter's main effect,
    or one pair's pure two-way interaction, named `"{first}:{second}"`."""

    name: str
    eta_squared: float
    kind: Literal["main_effect", "interaction"]

    @field_validator("eta_squared")
    @classmethod
    def _validate_range(cls, value: float) -> float:
        tolerance = 1e-9
        if not (-tolerance <= value <= 1.0 + tolerance):
            raise ValueError(f"eta_squared out of [0, 1] range: {value}")
        return value


class VarianceDecompositionResult(BaseModel):
    """The consolidated variance decomposition: every main effect and every pure two-way
    interaction in ONE sorted table, plus a residual.

    Deliberately stops at two-way order for readability -- `residual_eta_squared` is stated
    explicitly to contain three-way-and-higher interaction terms *as well as* run-to-run noise
    (see `replication_status()`: on Hercule's default saturated design the two are mathematically
    inseparable from this data alone, not merely confounded). p-values are meaningless on a
    saturated design (zero residual degrees of freedom) and are never computed here.
    """

    model_name: str
    metric: MetricName
    n_samples: int = Field(ge=1)
    entries: list[VarianceDecompositionEntry] = Field(default_factory=list)
    main_effects_sum: float
    interaction_sum: float
    residual_eta_squared: float
    dropped_columns: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sorted_and_totals(self) -> "VarianceDecompositionResult":
        tolerance = 1e-6
        values = [entry.eta_squared for entry in self.entries]
        if values != sorted(values, reverse=True):
            raise ValueError("entries must be sorted by descending eta_squared")

        main_effects_total = sum(entry.eta_squared for entry in self.entries if entry.kind == "main_effect")
        interaction_total = sum(entry.eta_squared for entry in self.entries if entry.kind == "interaction")
        if abs(main_effects_total - self.main_effects_sum) > tolerance:
            raise ValueError("main_effects_sum must equal the sum of main_effect entries")
        if abs(interaction_total - self.interaction_sum) > tolerance:
            raise ValueError("interaction_sum must equal the sum of interaction entries")
        if abs((self.main_effects_sum + self.interaction_sum + self.residual_eta_squared) - 1.0) > tolerance:
            raise ValueError("main_effects_sum + interaction_sum + residual_eta_squared must equal 1.0")
        return self


class VarianceDecompositionUnavailable(BaseModel):
    """The alternative return of `variance_decomposition()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def variance_decomposition(
    records: Sequence[RunRecord], metric: MetricName = "performance"
) -> VarianceDecompositionResult | VarianceDecompositionUnavailable:
    """Build the consolidated ANOVA-style table: every main effect and pure 2-way interaction.

    Combines `hyperparameter_importance()` (main effects) and `interaction_ranking()` (pure
    two-way interactions) into one table sorted by descending eta-squared, rather than the
    report showing them as two disconnected views -- a reader scanning for "what matters" should
    not have to mentally merge a main-effects bar chart with a separate interaction ranking.

    Args:
        records: The runs of one model family.
        metric: The `RunRecord` derived aggregate to attribute variance of.

    Returns:
        `VarianceDecompositionResult` under the same conditions `hyperparameter_importance()`
        succeeds under (interactions are simply omitted, with `interaction_sum == 0.0`, when
        `interaction_ranking()` itself is unavailable, e.g. fewer than 2 varying
        hyperparameters); otherwise `VarianceDecompositionUnavailable` with a human-readable
        reason. Never raises for a degenerate shape.

    Raises:
        ValueError: `records` is empty.
    """
    importance = hyperparameter_importance(records, metric)
    if isinstance(importance, ImportanceUnavailable):
        return VarianceDecompositionUnavailable(
            model_name=importance.model_name, metric=importance.metric, reason=importance.reason
        )

    entries = [
        VarianceDecompositionEntry(name=entry.name, eta_squared=entry.eta_squared, kind="main_effect")
        for entry in importance.entries
    ]

    ranking = interaction_ranking(records, metric)
    interaction_sum = 0.0
    if isinstance(ranking, InteractionRankingResult):
        for pair in ranking.pairs:
            entries.append(
                VarianceDecompositionEntry(
                    name=f"{pair.first}:{pair.second}", eta_squared=pair.interaction_eta_squared, kind="interaction"
                )
            )
        interaction_sum = ranking.total_pairwise_interaction

    entries.sort(key=lambda entry: (-entry.eta_squared, entry.name))

    residual = 1.0 - importance.main_effects_sum - interaction_sum

    return VarianceDecompositionResult(
        model_name=importance.model_name,
        metric=metric,
        n_samples=importance.n_samples,
        entries=entries,
        main_effects_sum=importance.main_effects_sum,
        interaction_sum=interaction_sum,
        residual_eta_squared=residual,
        dropped_columns=importance.dropped_columns,
    )


# ---------------------------------------------------------------------------------------------
# View [2]: hyperparameter_main_effects -- mean AND max performance per level.
# ---------------------------------------------------------------------------------------------


class MainEffectLevel(BaseModel):
    """One level of one hyperparameter: its mean AND max performance, run count, and spread.

    The mean alone is misleading for optimisation: a hyperparameter can have a perfectly flat
    mean across its levels (mean says "irrelevant") while its *maximum* attainable performance
    strictly decreases (max says "this sets your ceiling") -- measured on `learning_rate` in a
    real `deep_q_learning` family (means `0.086 / 0.067 / 0.087`, essentially flat; maxima
    `0.420 / 0.330 / 0.260`, strictly decreasing).
    """

    level: float
    mean_performance: float
    max_performance: float
    n_runs: int = Field(ge=1)
    std: float = Field(ge=0.0)


def max_performance_is_saturated(levels: Sequence[MainEffectLevel], relative_tolerance: float = 0.01) -> bool:
    """Whether the MAX performance is (near-)constant across a hyperparameter's levels.

    A max series barely moving relative to its own scale (measured example: CartPole's
    per-level maxima range from 499.69 to 500.00 -- an amplitude of 0.31 on a scale of 500,
    i.e. 0.06%) carries no signal about which level is "best": the episode-step cap is reached
    almost everywhere, and matplotlib's autoscale would otherwise magnify that 0.06% spread
    into a dramatic-looking curve -- the same class of misleading autoscaling already fixed
    once for a constant-value colorbar (see `CLAUDE.md`). This is a RELATIVE test against the
    series' own scale, never an absolute epsilon, consistent with the exact-range convention
    used throughout this module (`_drop_near_constant_columns`).

    Args:
        levels: The hyperparameter's ordered levels (as on `MainEffectsForHyperparameter`).
        relative_tolerance: The maximum allowed range as a fraction of the series' own scale
            (`max(abs(values).max(), 1.0)`); the default `0.01` (1%) cleanly separates the
            measured CartPole case (0.06%) from a genuinely varying max (FrozenLake's
            `learning_rate` maxima span 16% of scale).

    Returns:
        `True` when fewer than 2 levels are given (nothing to compare, trivially constant) or
        the maxima's range is within `relative_tolerance` of their own scale; `False` when the
        max series has a real, plottable spread.
    """
    if len(levels) < 2:
        return True
    maxima = np.array([level.max_performance for level in levels], dtype=float)
    value_range = float(np.ptp(maxima))
    scale = max(float(np.abs(maxima).max()), 1.0)
    return value_range <= relative_tolerance * scale


class MainEffectsForHyperparameter(BaseModel):
    """One hyperparameter's ordered-by-level main effect, by mean AND max."""

    name: str
    levels: list[MainEffectLevel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_levels_ordered(self) -> "MainEffectsForHyperparameter":
        values = [level.level for level in self.levels]
        if values != sorted(values):
            raise ValueError("levels must be ordered by ascending level value")
        return self

    @property
    def best_level_by_mean(self) -> float | None:
        """The level with the highest mean performance, or `None` when there are no levels."""
        if not self.levels:
            return None
        return max(self.levels, key=lambda level: level.mean_performance).level

    @property
    def best_level_by_max(self) -> float | None:
        """The level with the highest max (best attainable) performance, or `None` when empty."""
        if not self.levels:
            return None
        return max(self.levels, key=lambda level: level.max_performance).level

    @property
    def max_is_saturated(self) -> bool:
        """Whether the MAX performance is (near-)constant across levels -- see
        `max_performance_is_saturated()`. When `True`, the max view carries no signal for this
        hyperparameter, and `mean_and_max_disagree` never reports a divergence on its account."""
        return max_performance_is_saturated(self.levels)

    @property
    def mean_and_max_disagree(self) -> bool:
        """Whether the best level by mean differs from the best level by max.

        A `True` here is exactly the divergence the report must call out in prose: the mean
        ranking and the max ranking disagree on which level of this hyperparameter is "best",
        so a reader optimising for a single good run (max) would make a different choice than
        one optimising for average behaviour (mean). A disagreement is never reported when the
        max side is merely saturated noise (`max_is_saturated`) -- on CartPole every level's max
        sits within 0.06% of the episode-step ceiling, so which level "wins" by max is an
        autoscaling artifact, not a measurement.
        """
        best_mean = self.best_level_by_mean
        best_max = self.best_level_by_max
        if best_mean is None or best_max is None or best_mean == best_max:
            return False
        return not self.max_is_saturated


class MainEffectsResult(BaseModel):
    """Per-hyperparameter main effects for one model family, sorted alphabetically by name.

    A linear correlation can miss or understate a non-monotonic effect entirely (the canonical
    example on real data: `epsilon_min` has `corr(performance) = -0.102` but `eta_squared =
    3.4%`, because its middle level is the worst, not an endpoint) -- this view is the authority
    on shape. The mean/max pair additionally catches the case where the mean and the maximum
    attainable performance disagree on which level is best (`mean_and_max_disagree`).
    """

    model_name: str
    metric: MetricName
    n_samples: int = Field(ge=1)
    hyperparameters: list[MainEffectsForHyperparameter] = Field(default_factory=list)
    dropped_columns: dict[str, str] = Field(default_factory=dict)

    @property
    def all_max_saturated(self) -> bool:
        """Whether every plottable hyperparameter (>= 2 levels) has a saturated max series.

        Signal for the report to state the ceiling saturation once for the whole family (e.g.
        "capped at ~500 for every level of every hyperparameter on CartPole") instead of
        per-hyperparameter, and to treat the max view as carrying no signal anywhere in this
        family rather than in isolated panels.
        """
        plottable = [hp for hp in self.hyperparameters if len(hp.levels) >= 2]
        return bool(plottable) and all(hp.max_is_saturated for hp in plottable)


class MainEffectsUnavailable(BaseModel):
    """The alternative return of `hyperparameter_main_effects()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def hyperparameter_main_effects(
    records: Sequence[RunRecord], metric: MetricName = "performance"
) -> MainEffectsResult | MainEffectsUnavailable:
    """Compute, per varying numeric hyperparameter, the mean AND max `metric` at each level.

    Args:
        records: The runs of one model family.
        metric: The `RunRecord` derived aggregate to average/max.

    Returns:
        `MainEffectsResult` when at least one hyperparameter varies numerically and `metric` is
        finite and non-constant across at least 2 runs; otherwise `MainEffectsUnavailable` with
        a human-readable reason. Never raises for a degenerate shape.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("hyperparameter_main_effects requires at least one record")

    prep = _prepare_family(records, metric)
    n_finite = len(prep.finite_records)

    if n_finite < 2:
        return MainEffectsUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(f"only {n_finite} run(s) of {prep.model_name} have a finite {metric}; need at least 2"),
        )
    if not prep.kept_columns:
        return MainEffectsUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"no hyperparameter varies numerically across the {n_finite} runs of {prep.model_name} with a "
                f"finite {metric}"
            ),
        )
    if prep.performance_is_constant:
        return MainEffectsUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"{metric} is constant ({prep.y[0]:g}) across the {n_finite} runs of {prep.model_name} with a "
                f"finite value; every level would show the same mean"
            ),
        )

    hyperparameters = []
    for j, name in enumerate(prep.kept_columns):
        column = prep.x_kept[:, j]
        levels = []
        for level in np.unique(column):
            mask = column == level
            values = prep.y[mask]
            n_v = int(mask.sum())
            std_v = float(values.std(ddof=1)) if n_v > 1 else 0.0
            levels.append(
                MainEffectLevel(
                    level=float(level),
                    mean_performance=float(values.mean()),
                    max_performance=float(values.max()),
                    n_runs=n_v,
                    std=std_v,
                )
            )
        hyperparameters.append(MainEffectsForHyperparameter(name=name, levels=levels))

    return MainEffectsResult(
        model_name=prep.model_name,
        metric=metric,
        n_samples=n_finite,
        hyperparameters=hyperparameters,
        dropped_columns=prep.dropped_columns,
    )


# ---------------------------------------------------------------------------------------------
# View [3]: top_decile_comparison -- full grid vs top-decile decomposition, side by side.
# ---------------------------------------------------------------------------------------------


class RankShift(BaseModel):
    """One hyperparameter's importance rank in the full grid vs the top decile.

    `shift = rank_full - rank_top_decile`: positive means the hyperparameter became *more*
    important (moved to a lower/better rank number) in the top decile than it was on average;
    negative means it became less important there. No magnitude threshold decides "material" --
    callers sort by descending `abs(shift)` and read off the biggest movers themselves.
    """

    name: str
    rank_full: int = Field(ge=1)
    rank_top_decile: int = Field(ge=1)
    shift: int


class TopDecileComparisonResult(BaseModel):
    """Main-effects decomposition on the full grid and on its top-decile subset, side by side.

    "Important on average" (the full-grid column) is not "important where the good
    configurations are" (the top-decile column) -- see `top_decile_comparison()`'s docstring for
    the measured inversion this is built to surface. `rank_shifts` is sorted by descending
    `abs(shift)` so the biggest rank movers come first, with no invented magnitude cutoff
    deciding which ones are "material" -- the report states them and lets the reader judge.
    """

    model_name: str
    metric: MetricName
    quantile: float = Field(gt=0.0, lt=1.0)
    threshold: float
    n_full: int = Field(ge=1)
    n_top_decile: int = Field(ge=1)
    full_main_effects: list[HyperparameterEtaSquared] = Field(default_factory=list)
    top_decile_main_effects: list[HyperparameterEtaSquared] = Field(default_factory=list)
    rank_shifts: list[RankShift] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_rank_shifts_sorted(self) -> "TopDecileComparisonResult":
        magnitudes = [abs(shift.shift) for shift in self.rank_shifts]
        if magnitudes != sorted(magnitudes, reverse=True):
            raise ValueError("rank_shifts must be sorted by descending absolute shift")
        if self.n_top_decile > self.n_full:
            raise ValueError("n_top_decile cannot exceed n_full")
        return self


class TopDecileComparisonUnavailable(BaseModel):
    """The alternative return of `top_decile_comparison()` when it cannot be computed."""

    model_name: str
    metric: MetricName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def _rank_shifts(
    full_entries: Sequence[HyperparameterEtaSquared], top_entries: Sequence[HyperparameterEtaSquared]
) -> list[RankShift]:
    """Build the sorted `RankShift` list for `top_decile_comparison()`, from two already-ranked
    (descending eta-squared) `ImportanceResult.entries` lists.

    Args:
        full_entries: `ImportanceResult.entries` for the full grid.
        top_entries: `ImportanceResult.entries` for the top-decile subset.

    Returns:
        One `RankShift` per hyperparameter present in both lists (a hyperparameter dropped from
        one side, e.g. constant within the top-decile subset, is silently omitted rather than
        raising), sorted by descending absolute shift, tie-broken by name.
    """
    full_rank = {entry.name: index + 1 for index, entry in enumerate(full_entries)}
    top_rank = {entry.name: index + 1 for index, entry in enumerate(top_entries)}
    common_names = sorted(set(full_rank) & set(top_rank))

    shifts = [
        RankShift(
            name=name,
            rank_full=full_rank[name],
            rank_top_decile=top_rank[name],
            shift=full_rank[name] - top_rank[name],
        )
        for name in common_names
    ]
    shifts.sort(key=lambda shift: (-abs(shift.shift), shift.name))
    return shifts


def top_decile_comparison(
    records: Sequence[RunRecord],
    metric: MetricName = "performance",
    quantile: float = 0.9,
    min_top_n: int = 8,
) -> TopDecileComparisonResult | TopDecileComparisonUnavailable:
    """Rerun the main-effects decomposition on the subset scoring at or above `quantile`.

    "Important on average" is not "important where the good configurations are". Measured on a
    real 108-run `deep_q_learning` family (`quantile=0.9`, threshold `0.180`, 13 runs at or above
    it): `discount_factor`'s share collapses from 10.6% on the full grid to 0.6% in the top
    decile, while `learning_rate` rises from 1.0% to 43.7% -- the ranking inverts entirely. If the
    goal is to *optimise* rather than describe average behaviour, the top-decile column is the
    relevant one.

    Args:
        records: The runs of one model family.
        metric: The `RunRecord` derived aggregate to threshold and attribute variance of.
        quantile: The percentile threshold defining "top decile"; runs with `metric >=` this
            percentile are kept. `0.9` for the traditional top decile.
        min_top_n: Minimum number of runs the top-decile subset must have; below this the result
            would be noise, not a measurement, so the function refuses with a reason instead of
            printing a decomposition drawn from a handful of points.

    Returns:
        `TopDecileComparisonResult` when both the full grid and a sufficiently large top-decile
        subset admit a main-effects decomposition; otherwise `TopDecileComparisonUnavailable`
        with a human-readable reason. Never raises for a degenerate shape.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("top_decile_comparison requires at least one record")

    full_importance = hyperparameter_importance(records, metric)
    if isinstance(full_importance, ImportanceUnavailable):
        return TopDecileComparisonUnavailable(
            model_name=full_importance.model_name,
            metric=metric,
            reason=f"full grid: {full_importance.reason}",
        )

    prep = _prepare_family(records, metric)
    n_full = len(prep.finite_records)
    threshold = float(np.percentile(prep.y, quantile * 100))
    top_mask = prep.y >= threshold
    top_records = [record for record, keep in zip(prep.finite_records, top_mask.tolist(), strict=True) if keep]
    n_top = len(top_records)

    if n_top < min_top_n:
        return TopDecileComparisonUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=(
                f"the top {1.0 - quantile:.0%} of {n_full} runs of {prep.model_name} by {metric} (threshold "
                f"{threshold:g}) has only {n_top} run(s); refusing to decompose below {min_top_n} runs, since "
                f"the result would be noise rather than a measurement"
            ),
        )

    top_importance = hyperparameter_importance(top_records, metric)
    if isinstance(top_importance, ImportanceUnavailable):
        return TopDecileComparisonUnavailable(
            model_name=prep.model_name,
            metric=metric,
            reason=f"top decile ({n_top} run(s), threshold {threshold:g}): {top_importance.reason}",
        )

    return TopDecileComparisonResult(
        model_name=prep.model_name,
        metric=metric,
        quantile=quantile,
        threshold=threshold,
        n_full=n_full,
        n_top_decile=n_top,
        full_main_effects=full_importance.entries,
        top_decile_main_effects=top_importance.entries,
        rank_shifts=_rank_shifts(full_importance.entries, top_importance.entries),
    )


# ---------------------------------------------------------------------------------------------
# Replication status: is interaction even separable from run-to-run noise?
# ---------------------------------------------------------------------------------------------


class ReplicationStatus(BaseModel):
    """Whether one model family's design has any replication (several seeds per configuration).

    `is_saturated` is the primary signal: it compares the number of runs against the number of
    *distinct hyperparameter configurations* (every hyperparameter except `seed`, so a family
    where `seed` was never swept at all is handled the same way as one where it was swept but is
    now held constant). `distinct_configurations == n_samples` means exactly one run per cell --
    zero residual degrees of freedom -- so interaction and run-to-run noise are not merely
    confounded, they are mathematically inseparable from this dataset alone. `noise_floor_fraction`
    (when computable from per-episode testing rewards) estimates how much of the *observed*
    across-run performance variance is plausibly just evaluation sampling noise, independent of
    the interaction question, to stop a reader over-crediting a large residual share as rich
    exploitable structure.
    """

    model_name: str
    n_samples: int = Field(ge=0)
    distinct_configurations: int = Field(ge=0)
    replicates_per_cell: float = Field(ge=0.0)
    is_saturated: bool
    seed_hyperparameter_present: bool
    distinct_seed_count: int = Field(ge=0)
    has_replication: bool
    noise_floor_fraction: float | None = None
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


def _typical_sampling_noise_variance(records: Sequence[RunRecord]) -> float | None:
    """Average within-run sampling variance of the testing-phase mean reward.

    For a run with `m >= 2` testing episodes, the sampling variance of its own mean reward is
    `var(testing_rewards, ddof=1) / m`; this averages that quantity across every run that has
    one, generalising the Bernoulli-specific noise-floor estimate (`p(1-p)/m`) to any reward
    distribution actually observed, without assuming Bernoulli rewards.

    Args:
        records: The runs to average over.

    Returns:
        The mean per-run sampling variance, or `None` when no run has at least 2 testing
        episodes to estimate it from.
    """
    variances = [
        float(np.var(record.testing_rewards, ddof=1)) / len(record.testing_rewards)
        for record in records
        if len(record.testing_rewards) >= 2
    ]
    if not variances:
        return None
    return float(np.mean(variances))


def replication_status(records: Sequence[RunRecord]) -> ReplicationStatus:
    """Detect whether a model family's design has replication, and how confounded interaction
    and noise are without it.

    Does **not** aggregate replicates -- by design, this function only warns; a report or caller
    that wants interaction effects cleanly separated from noise must re-run with several seeds
    (e.g. `seed: [42, 43, 44]` in the YAML) and average performance per configuration itself.

    Args:
        records: The runs of one model family. An empty sequence is a legitimate input (unlike
            the other functions in this module) since replication is a property of the family
            as a whole, not an analysis that needs a minimum sample size to be defined.

    Returns:
        The `ReplicationStatus`; never raises.
    """
    if not records:
        return ReplicationStatus(
            model_name="",
            n_samples=0,
            distinct_configurations=0,
            replicates_per_cell=0.0,
            is_saturated=False,
            seed_hyperparameter_present=False,
            distinct_seed_count=0,
            has_replication=False,
            reason="no runs to inspect for replication",
        )

    model_name = records[0].model_name
    n_samples = len(records)

    def _config_key(record: RunRecord) -> tuple:
        return tuple(sorted((key, value) for key, value in record.hyperparameters.items() if key != "seed"))

    distinct_configs = len({_config_key(record) for record in records})
    replicates_per_cell = n_samples / distinct_configs if distinct_configs else 0.0
    is_saturated = distinct_configs >= n_samples

    seed_present_everywhere = all("seed" in record.hyperparameters for record in records)
    seed_values = {record.hyperparameters["seed"] for record in records if "seed" in record.hyperparameters}
    distinct_seed_count = len(seed_values)

    has_replication = (not is_saturated) and seed_present_everywhere and distinct_seed_count > 1

    reseed_suggestion = (
        "re-run with several seeds (e.g. seed: [42, 43, 44] in the YAML) and average performance "
        "per configuration to separate them"
    )

    noise_floor_fraction: float | None = None
    finite_performance = [record.performance for record in records if record.performance is not None]
    finite_performance = [value for value in finite_performance if np.isfinite(value)]
    if len(finite_performance) >= 2:
        observed_variance = float(np.var(finite_performance, ddof=1))
        noise_variance = _typical_sampling_noise_variance(records)
        if noise_variance is not None and observed_variance > 0.0:
            noise_floor_fraction = noise_variance / observed_variance

    noise_context = ""
    if noise_floor_fraction is not None:
        noise_context = (
            f" The typical per-run evaluation noise variance is ~{noise_floor_fraction:.1%} of the observed "
            f"across-run performance variance, so part of any residual share is measurement noise rather than "
            f"structure."
        )

    if is_saturated:
        reason = (
            f"the design is SATURATED: {distinct_configs} distinct hyperparameter configuration(s) (excluding "
            f"seed) across {n_samples} run(s) of {model_name} -- exactly {replicates_per_cell:.1f} replicate(s) "
            f"per cell. With zero residual degrees of freedom, interaction effects and run-to-run noise are "
            f"mathematically inseparable, not merely confounded: any residual share is an upper bound on real "
            f"interaction structure, not a measurement of it. " + reseed_suggestion + "." + noise_context
        )
    elif distinct_seed_count <= 1:
        seed_clause = "is not recorded for every run" if not seed_present_everywhere else "is constant"
        reason = (
            f"'seed' {seed_clause} across {model_name}, and only {replicates_per_cell:.1f} replicate(s) per cell "
            f"were found on average -- the residual share is confounded with run-to-run noise and cannot be "
            f"reliably separated. " + reseed_suggestion + "." + noise_context
        )
    else:
        reason = (
            f"{distinct_seed_count} distinct seed value(s) and {replicates_per_cell:.1f} replicate(s) per cell on "
            f"average across {n_samples} run(s) of {model_name}; replication may allow separating interaction "
            f"effects from noise, though this function does not aggregate replicates itself." + noise_context
        )

    return ReplicationStatus(
        model_name=model_name,
        n_samples=n_samples,
        distinct_configurations=distinct_configs,
        replicates_per_cell=replicates_per_cell,
        is_saturated=is_saturated,
        seed_hyperparameter_present=seed_present_everywhere,
        distinct_seed_count=distinct_seed_count,
        has_replication=has_replication,
        noise_floor_fraction=noise_floor_fraction,
        reason=reason,
    )


__all__ = [
    "HyperparameterEtaSquared",
    "ImportanceResult",
    "ImportanceUnavailable",
    "InteractionCell",
    "InteractionGridResult",
    "InteractionGridUnavailable",
    "InteractionRankingResult",
    "InteractionRankingUnavailable",
    "MainEffectLevel",
    "MainEffectsForHyperparameter",
    "MainEffectsResult",
    "MainEffectsUnavailable",
    "MetricName",
    "PairwiseInteraction",
    "RankShift",
    "ReplicationStatus",
    "TopDecileComparisonResult",
    "TopDecileComparisonUnavailable",
    "VarianceDecompositionEntry",
    "VarianceDecompositionResult",
    "VarianceDecompositionUnavailable",
    "hyperparameter_importance",
    "hyperparameter_main_effects",
    "interaction_grid",
    "interaction_ranking",
    "max_performance_is_saturated",
    "order_varying_hyperparameters_by_importance",
    "replication_status",
    "top_decile_comparison",
    "variance_decomposition",
]
