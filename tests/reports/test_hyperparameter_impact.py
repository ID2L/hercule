"""Unit tests for the sensitivity-analysis views (feature 004, reworked again on explicit user
direction away from any PCA that mixes hyperparameters with an outcome measure).

`hercule.reports.sensitivity` now answers "which hyperparameters drive success" purely through
variance decomposition (ANOVA / eta-squared by grouping), never a PCA: `variance_decomposition`
(main effects AND pure two-way interactions in one consolidated table), `hyperparameter_
main_effects` (mean AND max performance per level, since the mean alone can hide a divergent
max), `top_decile_comparison` (the same decomposition rerun on the top-scoring subset, since
"important on average" is not "important where the good configurations are"),
`interaction_ranking`/`interaction_grid` (pure two-way interaction shares), and
`replication_status` (saturated-design detection).

The core interaction fixture (`_interaction_records`) is a small, fully hand-computable 2x2
factorial in hyperparameters `a`/`b`, replicated twice, with a third hyperparameter `c` set to
the (performance-irrelevant) replicate index -- chosen so every eta-squared/interaction share
below has an exact closed-form answer to assert against, not just a re-run of the same code
under test. `_inversion_records` is a larger, still hand-computable 4x4 grid engineered to
reproduce the measured full-grid-vs-top-decile ranking inversion.
"""

from pathlib import Path

import numpy as np
import pytest

from hercule.reports.run_table import RunRecord
from hercule.reports.sensitivity import (
    ImportanceResult,
    ImportanceUnavailable,
    InteractionGridResult,
    InteractionGridUnavailable,
    InteractionRankingResult,
    InteractionRankingUnavailable,
    MainEffectLevel,
    MainEffectsForHyperparameter,
    MainEffectsResult,
    MainEffectsUnavailable,
    ReplicationStatus,
    TopDecileComparisonResult,
    TopDecileComparisonUnavailable,
    VarianceDecompositionResult,
    VarianceDecompositionUnavailable,
    hyperparameter_importance,
    hyperparameter_main_effects,
    interaction_grid,
    interaction_ranking,
    max_performance_is_saturated,
    order_varying_hyperparameters_by_importance,
    replication_status,
    top_decile_comparison,
    variance_decomposition,
)


def _record(
    name: str,
    hyperparameters: dict[str, object],
    performance: float,
    model_name: str = "deep_q_learning",
) -> RunRecord:
    """Build a bare `RunRecord` whose `performance` (via `mean_testing_reward`) is exactly the
    given value: a single testing episode scoring that reward."""
    return RunRecord(
        directory=Path(name),
        model_name=model_name,
        env_id="FrozenLake-v1",
        hyperparameters=hyperparameters,
        testing_rewards=[performance],
        testing_steps=[1],
    )


def _interaction_records() -> list[RunRecord]:
    """A hand-computable 2x2 factorial in `a`, `b` (2 replicates each, `c` = replicate index).

    Performance is 4.0 at (a=1, b=1) and 0.0 everywhere else -- a pure interaction pattern with
    no noise. Grand mean = 1.0, SS_total = 24.0 (8 runs: six at deviation -1, two at deviation
    +3). Main effects: eta(a) = eta(b) = 8/24 = 1/3 (grouping by `a` alone gives means 0 and 2 for
    4 runs each; same for `b` by symmetry). The joint (a, b) cell grouping explains all variance
    (each cell's two replicates are identical, so there is no residual): cell_eta = 24/24 = 1.0,
    hence pure interaction eta(a, b) = 1.0 - 1/3 - 1/3 = 1/3. `c` (the replicate index) has zero
    effect on performance by construction, so eta(c) = 0 and every pair involving `c` has zero
    interaction too -- included so ranking-order tests have something to sort below `(a, b)`.
    """
    perf_by_ab = {(0.0, 0.0): 0.0, (0.0, 1.0): 0.0, (1.0, 0.0): 0.0, (1.0, 1.0): 4.0}
    records = []
    for (a, b), perf in perf_by_ab.items():
        for rep in range(2):
            records.append(
                _record(
                    f"run_a{a:g}_b{b:g}_rep{rep}",
                    {"a": a, "b": b, "c": float(rep)},
                    performance=perf,
                )
            )
    return records


# ---------------------------------------------------------------------------------------------
# hyperparameter_importance
# ---------------------------------------------------------------------------------------------


def test_importance_reproduces_hand_computed_eta_squared() -> None:
    """R4-style closed-form check: eta(a) = eta(b) = 1/3, main_effects_sum = 2/3,
    interaction_residual = 1/3 (all of it is the pure a-b interaction, since c has no effect and
    there is no noise)."""
    result = hyperparameter_importance(_interaction_records())
    assert isinstance(result, ImportanceResult)

    by_name = {entry.name: entry.eta_squared for entry in result.entries}
    assert abs(by_name["a"] - 1 / 3) < 1e-9
    assert abs(by_name["b"] - 1 / 3) < 1e-9
    assert abs(by_name["c"] - 0.0) < 1e-9

    assert abs(result.main_effects_sum - 2 / 3) < 1e-9
    assert abs(result.interaction_residual - 1 / 3) < 1e-9
    assert result.n_samples == 8


def test_importance_entries_sorted_descending() -> None:
    result = hyperparameter_importance(_interaction_records())
    assert isinstance(result, ImportanceResult)
    values = [entry.eta_squared for entry in result.entries]
    assert values == sorted(values, reverse=True)
    assert result.entries[-1].name == "c"  # zero-effect column sorts last


def test_importance_unavailable_when_no_hyperparameter_varies() -> None:
    records = [_record(f"run_{i}", {"seed": 42}, performance=float(i)) for i in range(5)]
    result = hyperparameter_importance(records)
    assert isinstance(result, ImportanceUnavailable)
    assert "nothing to attribute" in result.reason


def test_importance_unavailable_when_performance_constant() -> None:
    records = [_record(f"run_{i}", {"a": float(i)}, performance=0.5) for i in range(5)]
    result = hyperparameter_importance(records)
    assert isinstance(result, ImportanceUnavailable)
    assert "constant" in result.reason


def test_importance_unavailable_when_too_few_finite_metric_runs() -> None:
    records = [_record("run_0", {"a": 1.0}, performance=1.0)]
    result = hyperparameter_importance(records)
    assert isinstance(result, ImportanceUnavailable)
    assert "only 1 run" in result.reason


# ---------------------------------------------------------------------------------------------
# order_varying_hyperparameters_by_importance (defect A: legend labels must be ordered so a
# length cap drops the LEAST informative hyperparameter first, not whichever sorts last
# alphabetically).
# ---------------------------------------------------------------------------------------------


def test_order_varying_hyperparameters_ranks_by_descending_eta_squared() -> None:
    """`a` and `b` tie at eta=1/3, `c` has zero effect -- alphabetical order would already put
    `a`, `b` first here, so this is re-checked against the *values*, not just against the
    alphabetical fixture, via the entries' own reported ordering."""
    records = _interaction_records()
    ordered = order_varying_hyperparameters_by_importance(records, ["c", "b", "a"])
    # c has eta_squared 0.0 and must sort after both a and b, which tie -- the tie is broken
    # alphabetically inside hyperparameter_importance itself (entries sorted by (-eta, name)).
    assert ordered == ["a", "b", "c"]


def test_order_varying_hyperparameters_puts_dominant_factor_first() -> None:
    """The measured real-world shape (defect A): a hyperparameter with a much larger eta-squared
    must be ordered before one with a smaller share, reversing alphabetical order when
    alphabetical order would rank the weak factor first."""
    # "z_weak" has almost no effect (near-constant across its two levels); "a_strong" fully
    # determines performance. Alphabetically "a_strong" already sorts first, so use names that
    # invert that: "weak_a" before "strong_z" alphabetically, but strong_z must come out first.
    records = [
        _record(f"run_{i}", {"weak_a": float(i % 2), "strong_z": float(i // 2)}, performance=float(i // 2) * 10.0)
        for i in range(4)
    ]
    ordered = order_varying_hyperparameters_by_importance(records, ["strong_z", "weak_a"])
    assert ordered[0] == "strong_z"


def test_order_varying_hyperparameters_appends_unranked_names_alphabetically() -> None:
    """A varying name absent from every record `hyperparameter_importance` scored (e.g. it does
    not vary numerically in this specific `records` subset) is still returned exactly once, after
    every ranked name, in alphabetical order."""
    records = _interaction_records()
    ordered = order_varying_hyperparameters_by_importance(records, ["a", "b", "c", "never_varies"])
    assert ordered[-1] == "never_varies"
    assert set(ordered) == {"a", "b", "c", "never_varies"}


def test_order_varying_hyperparameters_degrades_to_alphabetical_when_unavailable() -> None:
    """When `hyperparameter_importance` itself returns `ImportanceUnavailable` (here: no
    hyperparameter varies numerically), the function degrades to alphabetical order rather than
    raising."""
    records = [_record(f"run_{i}", {"seed": 42}, performance=float(i)) for i in range(5)]
    ordered = order_varying_hyperparameters_by_importance(records, ["zeta", "alpha"])
    assert ordered == ["alpha", "zeta"]


def test_order_varying_hyperparameters_empty_records_is_alphabetical() -> None:
    """Empty `records` must not raise -- degrades to the alphabetical order of `varying_names`."""
    assert order_varying_hyperparameters_by_importance([], ["zeta", "alpha"]) == ["alpha", "zeta"]


# ---------------------------------------------------------------------------------------------
# variance_decomposition -- consolidated ANOVA-style table (view [1]).
# ---------------------------------------------------------------------------------------------


def test_variance_decomposition_reproduces_hand_computed_shares() -> None:
    """Main effects AND the pure a-b interaction, folded into ONE sorted table with a residual
    of exactly 0.0 (this fixture has no noise and no higher-order structure beyond a-b)."""
    result = variance_decomposition(_interaction_records())
    assert isinstance(result, VarianceDecompositionResult)

    by_name = {entry.name: entry for entry in result.entries}
    assert abs(by_name["a"].eta_squared - 1 / 3) < 1e-9
    assert by_name["a"].kind == "main_effect"
    assert abs(by_name["b"].eta_squared - 1 / 3) < 1e-9
    assert by_name["b"].kind == "main_effect"
    assert abs(by_name["a:b"].eta_squared - 1 / 3) < 1e-9
    assert by_name["a:b"].kind == "interaction"
    assert abs(by_name["c"].eta_squared - 0.0) < 1e-9
    assert abs(by_name["a:c"].eta_squared - 0.0) < 1e-9
    assert abs(by_name["b:c"].eta_squared - 0.0) < 1e-9

    assert abs(result.main_effects_sum - 2 / 3) < 1e-9
    assert abs(result.interaction_sum - 1 / 3) < 1e-9
    assert abs(result.residual_eta_squared - 0.0) < 1e-9


def test_variance_decomposition_entries_sorted_descending() -> None:
    """`a`, `a:b`, `b` sit at ~1/3 each (floating-point subtraction in the interaction share can
    break an exact three-way tie by a few ULPs, so only descending order is asserted, not a fixed
    tie-break -- `a:c`, `b:c`, `c` sit at exactly 0.0 and sort after all three)."""
    result = variance_decomposition(_interaction_records())
    assert isinstance(result, VarianceDecompositionResult)

    values = [entry.eta_squared for entry in result.entries]
    assert values == sorted(values, reverse=True)

    names = [entry.name for entry in result.entries]
    assert set(names[:3]) == {"a", "a:b", "b"}
    assert set(names[3:]) == {"a:c", "b:c", "c"}


def test_variance_decomposition_no_p_value_field_anywhere() -> None:
    """Explicit guard against reintroducing a meaningless p-value on a saturated design: neither
    the entry model nor the result model exposes anything named like a p-value."""
    result = variance_decomposition(_interaction_records())
    assert isinstance(result, VarianceDecompositionResult)
    dumped = result.model_dump()
    assert not any("p_value" in key.lower() or key.lower() == "p" for key in dumped)
    for entry in result.entries:
        entry_dumped = entry.model_dump()
        assert not any("p_value" in key.lower() for key in entry_dumped)


def test_variance_decomposition_unavailable_when_no_hyperparameter_varies() -> None:
    records = [_record(f"run_{i}", {"seed": 42}, performance=float(i)) for i in range(5)]
    result = variance_decomposition(records)
    assert isinstance(result, VarianceDecompositionUnavailable)


def test_variance_decomposition_with_single_varying_hyperparameter_has_no_interactions() -> None:
    """Fewer than 2 varying hyperparameters: `interaction_ranking` itself is unavailable, so the
    decomposition degrades to main-effects-only with `interaction_sum == 0.0`, not an exception."""
    records = [_record(f"run_{i}", {"a": float(i % 3)}, performance=float(i % 3)) for i in range(6)]
    result = variance_decomposition(records)
    assert isinstance(result, VarianceDecompositionResult)
    assert all(entry.kind == "main_effect" for entry in result.entries)
    assert result.interaction_sum == 0.0


# ---------------------------------------------------------------------------------------------
# hyperparameter_main_effects -- mean AND max per level (view [2]).
# ---------------------------------------------------------------------------------------------


def _non_monotonic_records() -> list[RunRecord]:
    """`hump` in {0, 1, 2} maps to performance {0, 10, 0}: a linear correlation with `hump` is
    exactly zero by symmetry (deviations cancel), yet the level clearly drives performance."""
    perf_by_level = {0.0: 0.0, 1.0: 10.0, 2.0: 0.0}
    records = []
    for level, perf in perf_by_level.items():
        for rep in range(4):
            records.append(_record(f"run_l{level:g}_r{rep}", {"hump": level}, performance=perf))
    return records


def _divergent_mean_max_records() -> list[RunRecord]:
    """Three levels of `lr`, engineered so the MEAN ranking and the MAX ranking disagree on
    which level is best -- the exact real-data pattern measured on `learning_rate` in a
    `deep_q_learning` family (near-flat means; maxima strictly decreasing across levels).

    level 0.0: values [0.420, -0.248] -> mean 0.086, max 0.420 (best by MAX)
    level 1.0: values [0.330, -0.196] -> mean 0.067, max 0.330
    level 2.0: values [0.260, -0.086] -> mean 0.087, max 0.260 (best by MEAN)
    """
    data = {
        0.0: [0.420, -0.248],
        1.0: [0.330, -0.196],
        2.0: [0.260, -0.086],
    }
    records = []
    for level, values in data.items():
        for i, value in enumerate(values):
            records.append(_record(f"run_lr{level:g}_{i}", {"lr": level}, performance=value))
    return records


def test_main_effects_capture_non_monotonic_pattern_correlation_would_miss() -> None:
    records = _non_monotonic_records()
    result = hyperparameter_main_effects(records)
    assert isinstance(result, MainEffectsResult)

    hump = next(h for h in result.hyperparameters if h.name == "hump")
    means = [level.mean_performance for level in hump.levels]
    assert means == [0.0, 10.0, 0.0]  # non-monotonic: rises then falls
    assert [level.n_runs for level in hump.levels] == [4, 4, 4]
    assert [level.level for level in hump.levels] == [0.0, 1.0, 2.0]

    # The corresponding linear correlation is exactly zero, but eta-squared is not: the whole
    # point of this view is to catch what a correlation-only reading would miss.
    hump_values = np.array([record.hyperparameters["hump"] for record in records], dtype=float)
    perf_values = np.array([record.performance for record in records], dtype=float)
    corr = np.corrcoef(hump_values, perf_values)[0, 1]
    assert abs(corr) < 1e-9

    importance = hyperparameter_importance(records)
    assert isinstance(importance, ImportanceResult)
    eta = next(e.eta_squared for e in importance.entries if e.name == "hump")
    assert eta > 0.9  # deterministic, single-cause pattern: almost all variance is explained


def test_main_effects_levels_ordered_ascending() -> None:
    result = hyperparameter_main_effects(_interaction_records())
    assert isinstance(result, MainEffectsResult)
    for hp in result.hyperparameters:
        levels = [level.level for level in hp.levels]
        assert levels == sorted(levels)


def test_main_effects_unavailable_when_performance_constant() -> None:
    records = [_record(f"run_{i}", {"a": float(i)}, performance=1.0) for i in range(4)]
    result = hyperparameter_main_effects(records)
    assert isinstance(result, MainEffectsUnavailable)
    assert "constant" in result.reason


def test_main_effects_unavailable_when_no_varying_hyperparameter() -> None:
    records = [_record(f"run_{i}", {}, performance=float(i)) for i in range(4)]
    result = hyperparameter_main_effects(records)
    assert isinstance(result, MainEffectsUnavailable)


def test_main_effects_reports_max_alongside_mean() -> None:
    result = hyperparameter_main_effects(_divergent_mean_max_records())
    assert isinstance(result, MainEffectsResult)
    lr = next(hp for hp in result.hyperparameters if hp.name == "lr")

    means = {level.level: level.mean_performance for level in lr.levels}
    maxima = {level.level: level.max_performance for level in lr.levels}
    assert means[0.0] == pytest.approx(0.086)
    assert means[1.0] == pytest.approx(0.067)
    assert means[2.0] == pytest.approx(0.087)
    assert maxima[0.0] == pytest.approx(0.420)
    assert maxima[1.0] == pytest.approx(0.330)
    assert maxima[2.0] == pytest.approx(0.260)


def test_main_effects_mean_and_max_disagree_on_best_level() -> None:
    """The whole point of view [2]: the mean says level 2.0 is best, the max says level 0.0 is --
    the report must flag this divergence rather than print only the mean."""
    result = hyperparameter_main_effects(_divergent_mean_max_records())
    assert isinstance(result, MainEffectsResult)
    lr = next(hp for hp in result.hyperparameters if hp.name == "lr")

    assert lr.best_level_by_mean == pytest.approx(2.0)
    assert lr.best_level_by_max == pytest.approx(0.0)
    assert lr.mean_and_max_disagree is True


def test_main_effects_mean_and_max_agree_when_same_best_level() -> None:
    """When the best level is the same by both mean and max, no divergence is flagged."""
    result = hyperparameter_main_effects(_interaction_records())
    assert isinstance(result, MainEffectsResult)
    for hp in result.hyperparameters:
        if hp.name in ("a", "b"):
            assert hp.mean_and_max_disagree is False


def test_main_effects_disagree_is_none_safe_with_fewer_than_two_levels() -> None:
    """A hyperparameter with a single (constant) level would already be dropped upstream, but
    the property itself must not raise on an empty level list."""
    result = hyperparameter_main_effects(_non_monotonic_records())
    assert isinstance(result, MainEffectsResult)
    hump = next(h for h in result.hyperparameters if h.name == "hump")
    hump.levels = []
    assert hump.best_level_by_mean is None
    assert hump.best_level_by_max is None
    assert hump.mean_and_max_disagree is False


# ---------------------------------------------------------------------------------------------
# max_performance_is_saturated -- the ceiling-saturation predicate (defects 3/4: CartPole's
# max is capped at ~500 for every level, so plotting it as its own line lets matplotlib
# autoscale a 0.06%-of-scale spread into a dramatic-looking curve).
# ---------------------------------------------------------------------------------------------


def test_max_performance_is_saturated_true_for_near_constant_relative_spread() -> None:
    """The measured CartPole shape: maxima 499.69 / 500.0 / 500.0 -- a 0.31 amplitude on a scale
    of 500, i.e. 0.06% -- is saturated under the default 1% relative tolerance."""
    levels = [
        MainEffectLevel(level=0.0001, mean_performance=184.4, max_performance=499.69, n_runs=36, std=50.0),
        MainEffectLevel(level=0.00025, mean_performance=341.4, max_performance=500.0, n_runs=36, std=50.0),
        MainEffectLevel(level=0.001, mean_performance=401.9, max_performance=500.0, n_runs=36, std=50.0),
    ]
    assert max_performance_is_saturated(levels) is True


def test_max_performance_is_saturated_false_for_genuinely_varying_max() -> None:
    """The measured FrozenLake shape: maxima 0.420 / 0.330 / 0.260 -- a 0.16 amplitude on a
    scale of ~1, i.e. 16% -- is a real, plottable spread, not saturation."""
    levels = [
        MainEffectLevel(level=0.0, mean_performance=0.086, max_performance=0.420, n_runs=2, std=0.1),
        MainEffectLevel(level=1.0, mean_performance=0.067, max_performance=0.330, n_runs=2, std=0.1),
        MainEffectLevel(level=2.0, mean_performance=0.087, max_performance=0.260, n_runs=2, std=0.1),
    ]
    assert max_performance_is_saturated(levels) is False


def test_max_performance_is_saturated_relative_not_absolute() -> None:
    """The identical absolute spread (0.31) is saturated at scale 500 but NOT at scale ~1 -- a
    relative test against the series' own scale, never an absolute epsilon."""
    large_scale = [
        MainEffectLevel(level=0.0, mean_performance=1.0, max_performance=499.69, n_runs=2, std=0.0),
        MainEffectLevel(level=1.0, mean_performance=1.0, max_performance=500.0, n_runs=2, std=0.0),
    ]
    small_scale = [
        MainEffectLevel(level=0.0, mean_performance=1.0, max_performance=0.10, n_runs=2, std=0.0),
        MainEffectLevel(level=1.0, mean_performance=1.0, max_performance=0.41, n_runs=2, std=0.0),
    ]
    assert max_performance_is_saturated(large_scale) is True
    assert max_performance_is_saturated(small_scale) is False


def test_max_performance_is_saturated_safe_for_fewer_than_two_levels() -> None:
    """Nothing to compare with 0 or 1 level -- trivially reported as saturated, and must not
    raise (e.g. an empty `IndexError`/`ValueError` from `np.ptp` on an empty array)."""
    assert max_performance_is_saturated([]) is True
    single = [MainEffectLevel(level=0.0, mean_performance=1.0, max_performance=1.0, n_runs=1, std=0.0)]
    assert max_performance_is_saturated(single) is True


def test_mean_and_max_disagree_suppressed_when_max_saturated() -> None:
    """Reproduces the measured CartPole false positive: the mean clearly prefers level 0.001,
    the max nominally prefers a different level (0.00025, the first level to hit the tied
    499.69/500.0/500.0 ceiling) -- but that "disagreement" is autoscaling noise, not signal, and
    must NOT be reported."""
    hp = MainEffectsForHyperparameter(
        name="learning_rate",
        levels=[
            MainEffectLevel(level=0.0001, mean_performance=184.4, max_performance=499.69, n_runs=36, std=50.0),
            MainEffectLevel(level=0.00025, mean_performance=341.4, max_performance=500.0, n_runs=36, std=50.0),
            MainEffectLevel(level=0.001, mean_performance=401.9, max_performance=500.0, n_runs=36, std=50.0),
        ],
    )
    assert hp.best_level_by_mean == pytest.approx(0.001)
    assert hp.best_level_by_max == pytest.approx(0.00025)
    assert hp.max_is_saturated is True
    assert hp.mean_and_max_disagree is False


def test_mean_and_max_disagree_still_reported_when_max_genuinely_varies() -> None:
    """Regression guard: the saturation suppression must not swallow a REAL divergence -- the
    FrozenLake-style case already covered by
    `test_main_effects_mean_and_max_disagree_on_best_level` must still report `True`."""
    result = hyperparameter_main_effects(_divergent_mean_max_records())
    assert isinstance(result, MainEffectsResult)
    lr = next(hp for hp in result.hyperparameters if hp.name == "lr")
    assert lr.max_is_saturated is False
    assert lr.mean_and_max_disagree is True


def test_main_effects_result_all_max_saturated_true_when_every_hyperparameter_saturated() -> None:
    result = MainEffectsResult(
        model_name="deep_q_learning",
        metric="performance",
        n_samples=4,
        hyperparameters=[
            MainEffectsForHyperparameter(
                name="a",
                levels=[
                    MainEffectLevel(level=0.0, mean_performance=10.0, max_performance=499.8, n_runs=2, std=0.0),
                    MainEffectLevel(level=1.0, mean_performance=20.0, max_performance=500.0, n_runs=2, std=0.0),
                ],
            ),
            MainEffectsForHyperparameter(
                name="b",
                levels=[
                    MainEffectLevel(level=0.0, mean_performance=5.0, max_performance=500.0, n_runs=2, std=0.0),
                    MainEffectLevel(level=1.0, mean_performance=15.0, max_performance=499.9, n_runs=2, std=0.0),
                ],
            ),
        ],
    )
    assert result.all_max_saturated is True


def test_main_effects_result_all_max_saturated_false_when_one_hyperparameter_varies() -> None:
    result = hyperparameter_main_effects(_divergent_mean_max_records())
    assert isinstance(result, MainEffectsResult)
    assert result.all_max_saturated is False


def test_main_effects_result_all_max_saturated_false_with_no_plottable_hyperparameter() -> None:
    """An empty hyperparameters list must not vacuously report saturation -- `all()` over an
    empty sequence would otherwise be `True`."""
    result = MainEffectsResult(model_name="m", metric="performance", n_samples=2, hyperparameters=[])
    assert result.all_max_saturated is False


# ---------------------------------------------------------------------------------------------
# top_decile_comparison -- full grid vs top decile, side by side (view [3]).
# ---------------------------------------------------------------------------------------------


def _inversion_records() -> list[RunRecord]:
    """An 80-run balanced 4x4 grid (`P`, `Q` in {0, 1, 2, 3}, 5 replicates per cell) engineered
    so `P` dominates the full-grid main effects while, among the top-scoring subset, `P` is
    CONSTANT and `Q` alone explains the remaining spread -- the exact ranking-inversion pattern
    measured on real `deep_q_learning` data (`discount_factor` 10.6% -> 0.6%, `learning_rate`
    1.0% -> 43.7%), reduced to hand-computable integers.

    `performance(p, q) = 10*p + 3*(q == 3)`: with P, Q uniform over {0, 1, 2, 3} and independent
    (a balanced full factorial), `Var(10*P) = 100 * Var(P) = 125` and
    `Var(3 * (Q == 3)) = 9 * Var(indicator) = 9 * 0.1875 = 1.6875`; since the two terms are
    additive with no interaction, `eta(P) = 125 / 126.6875` and `eta(Q) = 1.6875 / 126.6875`.

    The 90th percentile of the resulting 80 values is exactly 30.0 (worked out from the value
    histogram: 15 runs at 30 either side of the boundary plus 5 at 33 land exactly on/above it),
    so the top-decile subset is the 20 runs with `P == 3` -- constant, hence dropped -- split
    5-5-5-5 across all four `Q` levels, where `Q` alone determines whether performance is 30 or
    33: `eta(Q)` in that subset is exactly 1.0.
    """
    records = []
    for p in range(4):
        for q in range(4):
            performance = 10.0 * p + (3.0 if q == 3 else 0.0)
            for rep in range(5):
                records.append(
                    _record(f"run_p{p}_q{q}_r{rep}", {"P": float(p), "Q": float(q)}, performance=performance)
                )
    return records


def test_top_decile_comparison_reproduces_the_full_grid_vs_top_decile_inversion() -> None:
    result = top_decile_comparison(_inversion_records())
    assert isinstance(result, TopDecileComparisonResult)

    assert result.n_full == 80
    assert result.threshold == pytest.approx(30.0)
    assert result.n_top_decile == 20

    full_by_name = {entry.name: entry.eta_squared for entry in result.full_main_effects}
    assert full_by_name["P"] == pytest.approx(125.0 / 126.6875, rel=1e-9)
    assert full_by_name["Q"] == pytest.approx(1.6875 / 126.6875, rel=1e-9)
    assert full_by_name["P"] > full_by_name["Q"]  # "important on average": P dominates

    top_by_name = {entry.name: entry.eta_squared for entry in result.top_decile_main_effects}
    assert "P" not in top_by_name  # constant within the top decile: dropped, not zero
    assert top_by_name["Q"] == pytest.approx(1.0)  # Q alone explains 100% of the top decile

    # Only names present on BOTH sides get a rank shift -- P dropped from the top-decile side
    # entirely, so only Q's inversion is reported: it becomes MORE important (positive shift).
    shift_by_name = {shift.name: shift for shift in result.rank_shifts}
    assert set(shift_by_name) == {"Q"}
    assert shift_by_name["Q"].rank_full == 2
    assert shift_by_name["Q"].rank_top_decile == 1
    assert shift_by_name["Q"].shift == 1


def test_top_decile_comparison_refuses_below_min_top_n() -> None:
    """Guard against printing noise from a handful of points: raising `min_top_n` above the
    actual top-decile size (20 here) must refuse with a reason, not degrade silently."""
    result = top_decile_comparison(_inversion_records(), min_top_n=25)
    assert isinstance(result, TopDecileComparisonUnavailable)
    assert "refusing" in result.reason
    assert "25" in result.reason


def test_top_decile_comparison_propagates_full_grid_unavailable_reason() -> None:
    records = [_record(f"run_{i}", {"seed": 42}, performance=float(i)) for i in range(10)]
    result = top_decile_comparison(records)
    assert isinstance(result, TopDecileComparisonUnavailable)
    assert "full grid" in result.reason


def test_top_decile_comparison_raises_on_empty_records() -> None:
    with pytest.raises(ValueError):
        top_decile_comparison([])


def test_top_decile_comparison_quantile_and_min_top_n_are_configurable() -> None:
    """A looser quantile (e.g. top half) on the same fixture yields a larger, still-decomposable
    subset -- confirms the threshold/quantile plumbing is not hard-coded to 0.9."""
    result = top_decile_comparison(_inversion_records(), quantile=0.5, min_top_n=8)
    assert isinstance(result, TopDecileComparisonResult)
    assert result.quantile == pytest.approx(0.5)
    assert result.n_top_decile > 20  # a looser threshold admits more than the top decile did


# ---------------------------------------------------------------------------------------------
# interaction_ranking / interaction_grid
# ---------------------------------------------------------------------------------------------


def test_interaction_ranking_orders_ab_above_zero_effect_pairs() -> None:
    """The (a, b) pair carries the entire interaction (1/3); every pair involving `c` (which has
    no effect on performance) must rank at exactly zero, below it."""
    result = interaction_ranking(_interaction_records())
    assert isinstance(result, InteractionRankingResult)

    by_pair = {(p.first, p.second): p.interaction_eta_squared for p in result.pairs}
    assert abs(by_pair[("a", "b")] - 1 / 3) < 1e-9
    assert abs(by_pair[("a", "c")] - 0.0) < 1e-9
    assert abs(by_pair[("b", "c")] - 0.0) < 1e-9

    assert result.pairs[0].first == "a"
    assert result.pairs[0].second == "b"
    assert abs(result.total_pairwise_interaction - 1 / 3) < 1e-9


def test_interaction_ranking_reports_both_main_effects_alongside_interaction() -> None:
    result = interaction_ranking(_interaction_records())
    assert isinstance(result, InteractionRankingResult)
    ab_pair = next(p for p in result.pairs if {p.first, p.second} == {"a", "b"})
    assert abs(ab_pair.first_eta_squared - 1 / 3) < 1e-9
    assert abs(ab_pair.second_eta_squared - 1 / 3) < 1e-9
    assert abs(ab_pair.cell_eta_squared - 1.0) < 1e-9


def test_interaction_ranking_unavailable_with_fewer_than_two_varying_hyperparameters() -> None:
    records = [_record(f"run_{i}", {"a": float(i)}, performance=float(i)) for i in range(5)]
    result = interaction_ranking(records)
    assert isinstance(result, InteractionRankingUnavailable)
    assert "at least 2" in result.reason


def test_interaction_grid_defaults_to_the_strongest_interaction_pair() -> None:
    """Default pair selection must follow `interaction_ranking`'s top entry (a, b), not simply
    "top two by main-effect importance" -- here all three columns share equal main-effect size,
    so only the interaction ranking distinguishes them."""
    result = interaction_grid(_interaction_records())
    assert isinstance(result, InteractionGridResult)
    assert {result.first, result.second} == {"a", "b"}
    assert abs(result.interaction_eta_squared - 1 / 3) < 1e-9


def test_interaction_grid_cell_means_and_counts_are_correct() -> None:
    result = interaction_grid(_interaction_records(), first="a", second="b")
    assert isinstance(result, InteractionGridResult)
    assert result.first_levels == [0.0, 1.0]
    assert result.second_levels == [0.0, 1.0]

    by_cell = {(cell.first_level, cell.second_level): cell for cell in result.cells}
    assert len(by_cell) == 4
    for key, cell in by_cell.items():
        assert cell.n_runs == 2
        expected_mean = 4.0 if key == (1.0, 1.0) else 0.0
        assert cell.mean_performance == expected_mean


def test_interaction_grid_rejects_identical_first_and_second() -> None:
    result = interaction_grid(_interaction_records(), first="a", second="a")
    assert isinstance(result, InteractionGridUnavailable)
    assert "must differ" in result.reason


def test_interaction_grid_rejects_unknown_hyperparameter_name() -> None:
    result = interaction_grid(_interaction_records(), first="a", second="does_not_exist")
    assert isinstance(result, InteractionGridUnavailable)


def test_interaction_grid_unavailable_with_fewer_than_two_varying_hyperparameters() -> None:
    records = [_record(f"run_{i}", {"a": float(i)}, performance=float(i)) for i in range(5)]
    result = interaction_grid(records)
    assert isinstance(result, InteractionGridUnavailable)


# ---------------------------------------------------------------------------------------------
# replication_status
# ---------------------------------------------------------------------------------------------


def test_replication_status_detects_saturated_design() -> None:
    """One run per distinct hyperparameter configuration (no seed field at all): zero residual
    degrees of freedom, interaction and noise mathematically inseparable."""
    records = [
        _record(f"run_{i}", {"learning_rate": float(i), "batch_size": float(i % 2)}, performance=float(i))
        for i in range(6)
    ]
    # Make every (learning_rate, batch_size) combination unique so distinct_configs == n_samples.
    status = replication_status(records)

    assert isinstance(status, ReplicationStatus)
    assert status.is_saturated is True
    assert status.distinct_configurations == status.n_samples == 6
    assert abs(status.replicates_per_cell - 1.0) < 1e-9
    assert status.has_replication is False
    assert "SATURATED" in status.reason


def test_replication_status_detects_constant_seed_as_no_replication() -> None:
    """Two distinct configurations, each repeated 3 times, but `seed` is constant: replication
    exists in principle (`replicates_per_cell == 3`) but is not attributable to seed variation."""
    records = []
    for config in range(2):
        for _rep in range(3):
            records.append(
                _record(f"run_{config}_{_rep}", {"learning_rate": float(config), "seed": 42}, performance=1.0)
            )
    status = replication_status(records)

    assert status.is_saturated is False
    assert status.distinct_configurations == 2
    assert abs(status.replicates_per_cell - 3.0) < 1e-9
    assert status.seed_hyperparameter_present is True
    assert status.distinct_seed_count == 1
    assert status.has_replication is False
    assert "constant" in status.reason


def test_replication_status_detects_genuine_seed_replication() -> None:
    """Two distinct configurations, each repeated across 3 distinct seeds: genuine replication."""
    records = []
    for config in range(2):
        for seed in range(3):
            records.append(
                _record(f"run_{config}_{seed}", {"learning_rate": float(config), "seed": seed}, performance=1.0)
            )
    status = replication_status(records)

    assert status.is_saturated is False
    assert status.seed_hyperparameter_present is True
    assert status.distinct_seed_count == 3
    assert status.has_replication is True
    assert "SATURATED" not in status.reason


def test_replication_status_on_empty_records_never_raises() -> None:
    status = replication_status([])
    assert isinstance(status, ReplicationStatus)
    assert status.n_samples == 0
    assert status.has_replication is False


def test_replication_status_noise_floor_fraction_when_computable() -> None:
    """With multiple testing episodes per run, `noise_floor_fraction` must be a finite,
    non-negative estimate rather than `None`."""
    records = []
    rng_rewards = [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
    for i, rewards in enumerate(rng_rewards):
        records.append(
            RunRecord(
                directory=Path(f"run_{i}"),
                model_name="deep_q_learning",
                env_id="FrozenLake-v1",
                hyperparameters={"learning_rate": float(i)},
                testing_rewards=rewards,
                testing_steps=[1] * len(rewards),
            )
        )
    status = replication_status(records)
    assert status.noise_floor_fraction is not None
    assert status.noise_floor_fraction >= 0.0


def test_replication_status_noise_floor_fraction_none_without_enough_episodes() -> None:
    records = [_record(f"run_{i}", {"a": float(i)}, performance=float(i)) for i in range(3)]
    status = replication_status(records)
    assert status.noise_floor_fraction is None
