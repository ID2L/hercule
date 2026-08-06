"""Unit tests for `hercule.reports.selection` (User Story 3).

`select_series` caps every multi-run chart at 9 ranked series (best 3 / median 3 / worst 3)
so a large comparative report stays readable (FR-010..FR-015). `RunRecord` needs no filesystem
backing (data-model.md), so these tests build records directly from literals.
"""

from pathlib import Path

from hercule.reports.run_table import RunRecord
from hercule.reports.selection import SeriesBucket, select_series


def _record(
    name: str, *, mean_learning_reward: float | None = None, mean_testing_reward: float | None = None
) -> RunRecord:
    """Build a `RunRecord` whose derived aggregates equal exactly the given values.

    A single-episode learning/testing phase makes the mean equal the given reward exactly;
    `None` is represented by an empty phase, matching the "no evaluation phase" semantics
    documented on `RunRecord`.
    """
    learning_rewards = [] if mean_learning_reward is None else [mean_learning_reward]
    testing_rewards = [] if mean_testing_reward is None else [mean_testing_reward]
    return RunRecord(
        directory=Path(name),
        model_name="simple_q_learning",
        env_id="FrozenLake-v1",
        learning_rewards=learning_rewards,
        learning_steps=[0] * len(learning_rewards),
        testing_rewards=testing_rewards,
        testing_steps=[0] * len(testing_rewards),
    )


def test_large_group_is_capped_at_nine_with_correct_buckets() -> None:
    """135 records must yield exactly 9 selected, correctly bucketed, 126 omitted."""
    records = [_record(f"run_{i:03d}", mean_learning_reward=float(i)) for i in range(135)]

    selection = select_series(records, "mean_learning_reward")

    assert len(selection.series) == 9
    assert selection.omitted_count == 126
    assert selection.total_count == 135

    counts = selection.counts_by_bucket
    assert counts[SeriesBucket.BEST] == 3
    assert counts[SeriesBucket.MEDIAN] == 3
    assert counts[SeriesBucket.WORST] == 3

    # Higher mean_learning_reward is better: run_134/133/132 are best, run_000/001/002 worst.
    best_names = {s.record.run_name for s in selection.series if s.bucket == SeriesBucket.BEST}
    worst_names = {s.record.run_name for s in selection.series if s.bucket == SeriesBucket.WORST}
    assert best_names == {"run_134", "run_133", "run_132"}
    assert worst_names == {"run_000", "run_001", "run_002"}


def test_small_group_returns_everything_with_no_omission() -> None:
    """FR-011: a group at or below the 9-record cap returns every record, omitted_count == 0."""
    records = [_record(f"run_{i}", mean_learning_reward=float(i)) for i in range(5)]

    selection = select_series(records, "mean_learning_reward")

    assert len(selection.series) == 5
    assert selection.omitted_count == 0
    selected_names = {s.record.run_name for s in selection.series}
    assert selected_names == {r.run_name for r in records}


def test_all_equal_metric_values_are_deterministic_across_repeated_calls() -> None:
    """FR-014/SC-009: pervasive ties (e.g. many FrozenLake runs scoring exactly 0) must still
    produce a stable, repeatable order via the (-metric, directory_name) tie-break."""
    records = [_record(f"run_{i:03d}", mean_learning_reward=0.0) for i in range(20)]

    first = select_series(records, "mean_learning_reward")
    second = select_series(list(reversed(records)), "mean_learning_reward")

    first_order = [(s.bucket, s.record.run_name) for s in first.series]
    second_order = [(s.bucket, s.record.run_name) for s in second.series]
    assert first_order == second_order


def test_bucket_overlap_is_deduplicated_for_small_counts() -> None:
    """Bucket windows overlap for 4 <= n <= 9; no record is ever returned twice."""
    for n in range(4, 10):
        records = [_record(f"run_{i}", mean_learning_reward=float(i)) for i in range(n)]

        selection = select_series(records, "mean_learning_reward")

        run_names = [s.record.run_name for s in selection.series]
        assert len(run_names) == len(set(run_names)), f"duplicate record in selection for n={n}"
        assert len(selection.series) == n
        assert selection.omitted_count == 0


def test_missing_metric_value_sorts_last_but_is_not_dropped() -> None:
    """A record with no value for the ranking metric must still be selectable, tie-broken by
    directory name, rather than dropped from the ranking."""
    records = [_record(f"run_{i}", mean_learning_reward=float(i)) for i in range(3)]
    records.append(_record("run_none", mean_learning_reward=None))

    selection = select_series(records, "mean_learning_reward")

    assert len(selection.series) == 4
    none_entry = next(s for s in selection.series if s.record.run_name == "run_none")
    assert none_entry.metric_value is None
    assert none_entry.bucket == SeriesBucket.WORST


def test_different_metrics_select_different_subsets() -> None:
    """FR-012: ranking on two different metrics over the same group can select different runs."""
    records = [
        _record("run_a", mean_learning_reward=10.0, mean_testing_reward=0.0),
        _record("run_b", mean_learning_reward=0.0, mean_testing_reward=10.0),
        _record("run_c", mean_learning_reward=5.0, mean_testing_reward=5.0),
    ]

    by_learning = select_series(records, "mean_learning_reward")
    by_testing = select_series(records, "mean_testing_reward")

    learning_best = next(s for s in by_learning.series if s.bucket == SeriesBucket.BEST)
    testing_best = next(s for s in by_testing.series if s.bucket == SeriesBucket.BEST)
    assert learning_best.record.run_name == "run_a"
    assert testing_best.record.run_name == "run_b"
