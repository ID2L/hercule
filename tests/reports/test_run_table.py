"""Tests for `hercule.reports.run_table` (User Story 2, contracts C3).

`build_run_table` is the single loading routine used both by the report generator and by the
generated notebook at runtime (FR-004, FR-005, FR-009). These tests pin down: one `RunRecord`
per run leaf directory; `model.json` is never opened (FR-007, SC-010); unreadable runs become
`SkippedRun` without aborting the walk (FR-008); the derived aggregates distinguish "no
evaluation phase" (`None`) from "scored 0" (`0.0`); and `to_dataframe()`'s column schema matches
data-model.md exactly, holding scalars only.
"""

from pathlib import Path

import pandas as pd
import pytest

from hercule.reports.run_table import (
    ConstantMetric,
    HyperparameterGridCardinality,
    MetricRedundancy,
    RankedRun,
    RunRecord,
    RunTable,
    SkippedRun,
    build_run_table,
    detect_constant_metrics,
    detect_redundant_metrics,
    format_relative_run_path,
    format_series_labels,
    format_top_table_hyperparameter_cells,
    format_varying_hyperparameters,
    hyperparameter_grid_cardinality,
    rank_runs_by_performance,
    select_top_table_metric_columns,
)

from .conftest import RunTreeBuilder


def test_build_run_table_returns_one_record_per_run(run_tree_builder: RunTreeBuilder) -> None:
    """One `RunRecord` is produced per run leaf directory, with model_name and hyperparameters."""
    run_tree_builder.add_run(
        model_name="simple_q_learning",
        model_signature="sig_a",
        hyperparameters={"learning_rate": 0.0001, "discount_factor": 0.95},
    )
    run_tree_builder.add_run(
        model_name="simple_sarsa",
        model_signature="sig_b",
        hyperparameters={"learning_rate": 0.001, "epsilon": 1.0},
    )

    table = build_run_table(run_tree_builder.root)

    assert table.runs_loaded == 2
    assert table.runs_skipped == 0
    model_names = sorted(record.model_name for record in table.records)
    assert model_names == ["simple_q_learning", "simple_sarsa"]

    by_model = {record.model_name: record for record in table.records}
    assert by_model["simple_q_learning"].hyperparameters == {"learning_rate": 0.0001, "discount_factor": 0.95}
    assert by_model["simple_sarsa"].hyperparameters == {"learning_rate": 0.001, "epsilon": 1.0}

    # model_name comes from the directory layout, never from model.json.
    for record in table.records:
        assert record.directory.parent.name == record.model_name


def test_corrupt_model_json_does_not_prevent_loading(run_tree_builder: RunTreeBuilder) -> None:
    """A run whose model.json is corrupt still loads — model.json is never opened (SC-010)."""
    run_tree_builder.add_run(model_name="deep_q_learning", model_signature="sig", corrupt_model=True)

    table = build_run_table(run_tree_builder.root)

    assert table.runs_loaded == 1
    assert table.runs_skipped == 0
    assert table.records[0].model_name == "deep_q_learning"


def test_malformed_run_info_becomes_skipped_run(run_tree_builder: RunTreeBuilder) -> None:
    """A run with corrupt run_info.json is skipped with a reason; other runs still load."""
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="good", hyperparameters={"seed": 1})
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="bad", corrupt_run_info=True)

    table = build_run_table(run_tree_builder.root)

    assert table.runs_loaded == 1
    assert table.runs_skipped == 1
    assert isinstance(table.skipped[0], SkippedRun)
    assert table.skipped[0].reason  # non-empty, human readable
    assert table.records[0].hyperparameters == {"seed": 1}


def test_corrupt_environment_json_becomes_skipped_run(run_tree_builder: RunTreeBuilder) -> None:
    """A run whose environment.json is corrupt is skipped, not fatal to the walk.

    Unlike an entirely missing environment.json (which makes the directory invisible to
    `find_experiment_directories`, since it only checks existence), a corrupt-but-present file
    is discovered and must surface as a `SkippedRun`.
    """
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="good")
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="bad_env", corrupt_environment=True)

    table = build_run_table(run_tree_builder.root)

    assert table.runs_loaded == 1
    assert table.runs_skipped == 1


def test_omitted_environment_json_is_invisible_to_the_walk(run_tree_builder: RunTreeBuilder) -> None:
    """A directory entirely missing environment.json fails the existence check and is never
    visited at all — it is neither loaded nor recorded as a SkippedRun."""
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="good")
    run_tree_builder.add_run(model_name="simple_q_learning", model_signature="missing_env", omit_environment=True)

    table = build_run_table(run_tree_builder.root)

    assert table.runs_loaded == 1
    assert table.runs_skipped == 0


def test_aggregate_derivations(run_tree_builder: RunTreeBuilder) -> None:
    """mean/success-rate aggregates are computed correctly; empty testing phase yields None."""
    run_dir = run_tree_builder.add_run(
        model_name="simple_q_learning",
        model_signature="sig",
        learning_episode_count=4,
        learning_reward_fn=lambda i: [1.0, 0.0, 1.0, 0.0][i],
        testing_episode_count=2,
        testing_reward_fn=lambda i: [1.0, 1.0][i],
    )
    no_testing_dir = run_tree_builder.add_run(
        model_name="simple_q_learning",
        model_signature="sig_no_test",
        learning_episode_count=3,
        learning_reward_fn=lambda i: 0.0,
        testing_episode_count=0,
    )

    table = build_run_table(run_tree_builder.root)
    by_dir = {record.directory: record for record in table.records}

    record = by_dir[run_dir]
    assert record.episode_count == 4
    assert record.testing_episode_count == 2
    assert record.mean_learning_reward == pytest.approx(0.5)
    assert record.learning_success_rate == pytest.approx(0.5)
    assert record.mean_testing_reward == pytest.approx(1.0)
    assert record.testing_success_rate == pytest.approx(1.0)
    assert record.performance == pytest.approx(1.0)

    no_test_record = by_dir[no_testing_dir]
    assert no_test_record.testing_episode_count == 0
    assert no_test_record.mean_testing_reward is None
    assert no_test_record.testing_success_rate is None
    # A run that genuinely scored 0 on every learning episode must stay distinguishable
    # from a run with no evaluation phase: mean_learning_reward is 0.0, not None.
    assert no_test_record.mean_learning_reward == pytest.approx(0.0)
    assert no_test_record.learning_success_rate == pytest.approx(0.0)
    # performance falls back to the learning aggregate when there is no testing phase.
    assert no_test_record.performance == pytest.approx(0.0)


def test_to_dataframe_column_schema(run_tree_builder: RunTreeBuilder) -> None:
    """to_dataframe() matches the data-model.md schema: scalars only, hp_/env_ prefixes."""
    run_tree_builder.add_run(
        model_name="simple_q_learning",
        model_signature="sig_a",
        env_kwargs={"map_name": "4x4", "is_slippery": True},
        hyperparameters={"learning_rate": 0.0001, "seed": 42},
        learning_episode_count=3,
        testing_episode_count=2,
    )
    run_tree_builder.add_run(
        model_name="dummy",
        model_signature="sig_b",
        env_kwargs={"map_name": "4x4", "is_slippery": True},
        hyperparameters={"seed": 7},
        learning_episode_count=3,
        testing_episode_count=0,
    )

    table = build_run_table(run_tree_builder.root)
    df = table.to_dataframe()

    expected_scalar_columns = {
        "directory",
        "run_name",
        "model_name",
        "env_id",
        "max_episode_steps",
        "episode_count",
        "testing_episode_count",
        "mean_learning_reward",
        "learning_success_rate",
        "mean_testing_reward",
        "testing_success_rate",
        "performance",
    }
    assert expected_scalar_columns <= set(df.columns)

    # No per-episode list columns: every column must hold scalars only.
    for column in df.columns:
        assert not df[column].apply(lambda v: isinstance(v, list)).any()

    # hp_/env_ prefixed columns exist for every hyperparameter/env-kwarg name seen anywhere.
    assert "hp_learning_rate" in df.columns
    assert "hp_seed" in df.columns
    assert "env_map_name" in df.columns
    assert "env_is_slippery" in df.columns

    # A hyperparameter absent from the "dummy" family (learning_rate) is null there, not 0.
    dummy_row = df[df["model_name"] == "dummy"].iloc[0]
    assert pd.isna(dummy_row["hp_learning_rate"])
    assert dummy_row["hp_seed"] == pytest.approx(7)

    # Row order is (model_name, run_name).
    assert list(df["model_name"]) == sorted(df["model_name"])

    # max_episode_steps is a nullable integer dtype.
    assert str(df["max_episode_steps"].dtype) == "Int64"


def test_run_table_never_requires_non_empty_records(tmp_path: Path) -> None:
    """A group whose every run was unreadable is a legitimate, reportable RunTable."""
    table = RunTable(root=tmp_path, records=[], skipped=[SkippedRun(path=tmp_path / "x", reason="boom")])
    assert table.runs_loaded == 0
    assert table.runs_skipped == 1


def test_run_record_hyperparameters_reject_list_values() -> None:
    """A hyperparameter still holding a list means expand_variants() never ran."""
    with pytest.raises(ValueError, match="list"):
        RunRecord(
            directory=Path("."),
            model_name="simple_q_learning",
            env_id="FrozenLake-v1",
            hyperparameters={"learning_rate": [0.1, 0.2]},
        )


# ---------------------------------------------------------------------------------------------
# detect_redundant_metrics (feature 004 rework: binary-reward metric redundancy detection).
# ---------------------------------------------------------------------------------------------


def _binary_reward_record(name: str, learning_rewards: list[float], testing_rewards: list[float]) -> RunRecord:
    """A record whose learning AND testing rewards are all in {0, 1}: mean reward mechanically
    equals success rate for both phases, exactly the measured real-data FrozenLake shape."""
    return RunRecord(
        directory=Path(name),
        model_name="deep_q_learning",
        env_id="FrozenLake-v1",
        learning_rewards=learning_rewards,
        learning_steps=[1] * len(learning_rewards),
        testing_rewards=testing_rewards,
        testing_steps=[1] * len(testing_rewards),
    )


def test_detect_redundant_metrics_flags_both_binary_reward_pairs() -> None:
    """On binary rewards (FrozenLake's {0, 1}), BOTH the learning and testing pairs are exactly
    redundant -- mean(rewards) == mean(rewards > 0) for every run, by construction."""
    records = [
        _binary_reward_record("run_0", [0.0, 1.0, 1.0, 0.0], [1.0, 1.0]),
        _binary_reward_record("run_1", [1.0, 1.0, 1.0, 1.0], [0.0, 0.0]),
        _binary_reward_record("run_2", [0.0, 0.0, 0.0, 0.0], [1.0, 0.0]),
    ]

    redundancies = detect_redundant_metrics(records)

    pairs = {(r.first, r.second) for r in redundancies}
    assert ("mean_learning_reward", "learning_success_rate") in pairs
    assert ("mean_testing_reward", "testing_success_rate") in pairs
    assert all(isinstance(r, MetricRedundancy) for r in redundancies)
    for redundancy in redundancies:
        assert redundancy.reason  # non-empty, human readable


def test_detect_redundant_metrics_empty_for_continuous_reward() -> None:
    """On a continuous-reward environment (e.g. CartPole), the mean reward and the success rate
    genuinely differ -- no pair is flagged, and both charts must still be drawn."""
    records = [
        RunRecord(
            directory=Path(f"run_{i}"),
            model_name="deep_q_learning",
            env_id="CartPole-v1",
            learning_rewards=[10.0, 25.0, 42.0, 8.0],
            learning_steps=[10, 25, 42, 8],
            testing_rewards=[30.0, 55.0],
            testing_steps=[30, 55],
        )
        for i in range(3)
    ]

    assert detect_redundant_metrics(records) == []


def test_detect_redundant_metrics_exact_not_tolerant() -> None:
    """A near-miss (differs by a tiny amount, not exactly equal) must NOT be flagged -- the
    detection is an exact `==` comparison, never a tolerance."""
    records = [
        RunRecord(
            directory=Path("run_0"),
            model_name="deep_q_learning",
            env_id="FrozenLake-v1",
            learning_rewards=[0.0, 1.0, 1.0, 0.9999999],  # not binary: success rate would be 0.5, mean 0.7499...
            learning_steps=[1, 1, 1, 1],
        )
    ]

    redundancies = detect_redundant_metrics(records)
    assert ("mean_learning_reward", "learning_success_rate") not in {(r.first, r.second) for r in redundancies}


def test_detect_redundant_metrics_empty_on_empty_records() -> None:
    """Empty input must not be misread as a trivial (vacuous) redundancy."""
    assert detect_redundant_metrics([]) == []


def test_detect_redundant_metrics_none_pattern_must_match() -> None:
    """A run with no testing phase (None for both) does not by itself make the pair redundant if
    another run in the set has genuinely differing testing values."""
    records = [
        _binary_reward_record("run_0", [1.0, 0.0], []),  # no testing phase: both None
        RunRecord(
            directory=Path("run_1"),
            model_name="deep_q_learning",
            env_id="FrozenLake-v1",
            learning_rewards=[1.0, 0.0],
            learning_steps=[1, 1],
            testing_rewards=[5.0, 5.0],  # non-binary: mean 5.0, success rate 1.0 -- genuinely differ
            testing_steps=[1, 1],
        ),
    ]

    redundancies = detect_redundant_metrics(records)
    assert ("mean_testing_reward", "testing_success_rate") not in {(r.first, r.second) for r in redundancies}
    # The learning pair is still redundant on its own: both runs are binary-reward there.
    assert ("mean_learning_reward", "learning_success_rate") in {(r.first, r.second) for r in redundancies}


# ---------------------------------------------------------------------------------------------
# detect_constant_metrics (defect A: a metric constant in ITSELF, distinct from
# detect_redundant_metrics, which flags two DIFFERENT metrics equal to EACH OTHER).
# ---------------------------------------------------------------------------------------------


def _cartpole_style_record(name: str, learning_rewards: list[float], testing_rewards: list[float]) -> RunRecord:
    """CartPole awards +1 reward per step, so every episode scores > 0 and the success rate is
    exactly 1.0 for every run -- structurally, not by chance. Mean reward still varies."""
    return RunRecord(
        directory=Path(name),
        model_name="deep_q_learning",
        env_id="CartPole-v1",
        learning_rewards=learning_rewards,
        learning_steps=[int(r) for r in learning_rewards],
        testing_rewards=testing_rewards,
        testing_steps=[int(r) for r in testing_rewards],
    )


def test_detect_constant_metrics_flags_structurally_constant_success_rate() -> None:
    """The measured CartPole shape: learning/testing success rate is exactly 1.0 for every run
    (every step earns positive reward), while the mean reward genuinely varies -- only the
    success-rate metrics are flagged, not the reward ones."""
    records = [
        _cartpole_style_record("run_0", [22.0, 35.0, 405.0], [200.0, 250.0]),
        _cartpole_style_record("run_1", [30.0, 60.0, 120.0], [180.0, 300.0]),
        _cartpole_style_record("run_2", [50.0, 80.0, 90.0], [220.0, 260.0]),
    ]

    constants = detect_constant_metrics(records)

    names = {c.metric for c in constants}
    assert "learning_success_rate" in names
    assert "testing_success_rate" in names
    assert "mean_learning_reward" not in names
    assert "mean_testing_reward" not in names
    assert all(isinstance(c, ConstantMetric) for c in constants)
    for constant in constants:
        assert constant.value == pytest.approx(1.0)
        assert constant.reason  # non-empty, human readable


def test_detect_constant_metrics_empty_for_frozenlake_style_informative_metrics() -> None:
    """On FrozenLake, success rate genuinely varies across runs (not every run reaches the
    goal) -- no metric should be flagged as constant."""
    records = [
        _binary_reward_record("run_0", [0.0, 1.0, 1.0, 0.0], [1.0, 1.0]),
        _binary_reward_record("run_1", [1.0, 1.0, 1.0, 1.0], [0.0, 0.0]),
        _binary_reward_record("run_2", [0.0, 0.0, 0.0, 0.0], [1.0, 0.0]),
    ]

    assert detect_constant_metrics(records) == []


def test_detect_constant_metrics_relative_tolerance_not_absolute() -> None:
    """A genuinely constant float column (e.g. bit-identical 0.4 rewards) must not slip through
    on floating-point rounding noise, and the test must be relative to scale, never an absolute
    epsilon: rewards constant at ~0.4 (small scale) are flagged the same way as rewards constant
    at ~400 (large scale, matching the CartPole ceiling)."""
    small_scale_records = [
        RunRecord(
            directory=Path(f"run_{i}"),
            model_name="deep_q_learning",
            env_id="FrozenLake-v1",
            learning_rewards=[0.4, 0.4],
            learning_steps=[1, 1],
        )
        for i in range(3)
    ]
    assert any(c.metric == "mean_learning_reward" for c in detect_constant_metrics(small_scale_records))

    large_scale_records = [
        RunRecord(
            directory=Path(f"run_{i}"),
            model_name="deep_q_learning",
            env_id="CartPole-v1",
            learning_rewards=[500.0, 500.0],
            learning_steps=[500, 500],
        )
        for i in range(3)
    ]
    assert any(c.metric == "mean_learning_reward" for c in detect_constant_metrics(large_scale_records))


def test_detect_constant_metrics_empty_on_empty_or_insufficient_records() -> None:
    """Empty input, and a metric with fewer than 2 finite values, must not be misread as a
    trivial (vacuous) constancy."""
    assert detect_constant_metrics([]) == []

    single_record = [
        RunRecord(
            directory=Path("run_0"),
            model_name="deep_q_learning",
            env_id="CartPole-v1",
            learning_rewards=[10.0],
            learning_steps=[10],
        )
    ]
    assert detect_constant_metrics(single_record) == []


# ---------------------------------------------------------------------------------------------
# format_varying_hyperparameters (defect C: short, deterministic legend/tick labels).
# ---------------------------------------------------------------------------------------------


def _hp_record(name: str, hyperparameters: dict[str, object]) -> RunRecord:
    return RunRecord(
        directory=Path(name),
        model_name="deep_q_learning",
        env_id="CartPole-v1",
        hyperparameters=hyperparameters,
    )


def test_format_varying_hyperparameters_abbreviates_and_restricts_to_varying_names() -> None:
    """Only the given (varying) names appear, abbreviated to the same first-3-letters-per-word
    scheme as the on-disk directory signature -- constant hyperparameters (e.g. `seed`, not
    passed in `varying_names`) never appear."""
    record = _hp_record(
        "run_0",
        {"batch_size": 64, "learning_rate": 0.001, "seed": 42},
    )
    label = format_varying_hyperparameters(record, ["batch_size", "learning_rate"])
    assert label == "bat_siz=64 lea_rat=0.001"
    assert "seed" not in label


def test_format_varying_hyperparameters_skips_names_absent_from_this_record() -> None:
    """A varying hyperparameter that a DIFFERENT model family in the same report group declares,
    but this record does not, is silently skipped rather than raising."""
    record = _hp_record("run_0", {"batch_size": 32})
    label = format_varying_hyperparameters(record, ["batch_size", "epsilon"])
    assert label == "bat_siz=32"


def test_format_varying_hyperparameters_falls_back_to_run_name_when_nothing_varies() -> None:
    record = _hp_record("run_0", {})
    assert format_varying_hyperparameters(record, []) == "run_0"


def test_format_varying_hyperparameters_truncates_with_backstop_cap() -> None:
    """Many varying hyperparameters at once must still respect `max_length`, truncated with a
    trailing ellipsis rather than left to overflow a legend."""
    record = _hp_record(
        "run_0",
        {
            "batch_size": 64,
            "discount_factor": 0.99,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "replay_buffer_size": 10000,
        },
    )
    label = format_varying_hyperparameters(
        record,
        ["batch_size", "discount_factor", "epsilon_min", "learning_rate", "replay_buffer_size"],
        max_length=20,
    )
    assert len(label) == 20
    assert label.endswith("...")


def test_format_varying_hyperparameters_deterministic_regardless_of_dict_iteration_order() -> None:
    """The label's order follows `varying_names`, not the record's own hyperparameters dict
    insertion order, so regeneration is byte-identical regardless of JSON key order on disk."""
    record = _hp_record("run_0", {"b": 2, "a": 1})
    assert format_varying_hyperparameters(record, ["a", "b"]) == "a=1 b=2"
    assert format_varying_hyperparameters(record, ["b", "a"]) == "b=2 a=1"


# ---------------------------------------------------------------------------------------------
# format_series_labels (defect A: two different runs must never render an identical legend
# label -- a length cap that truncates away the one hyperparameter that differentiates them is
# exactly the measured real bug, e.g. two `simple_q_learning` runs both rendering
# "dis_fac=0.99 eps_dec=0.005 eps_min=0..." with the differentiating parameter cut off).
# ---------------------------------------------------------------------------------------------


def test_format_series_labels_disambiguates_a_collision_caused_by_truncation() -> None:
    """Two records whose low-priority (low-importance) hyperparameter differs, but whose
    high-priority ones are identical, collide under a short length cap -- the differentiating
    parameter must reappear once the collision is resolved, rather than staying hidden."""
    long_name_records = [
        _hp_record("run_0", {"discount_factor": 0.99, "epsilon_min": 0.01, "learning_rate": 0.1}),
        _hp_record("run_1", {"discount_factor": 0.99, "epsilon_min": 0.01, "learning_rate": 0.2}),
    ]
    # Both are the top-3 "best" bucket of the same model family -- an identical prefix, exactly
    # the measured real collision (two different [best] entries of the same model family). A
    # cap short enough that only "discount_factor" fits makes both records collide there.
    labels = format_series_labels(
        long_name_records,
        ["[best] simple_q_learning", "[best] simple_q_learning"],
        [["discount_factor", "epsilon_min", "learning_rate"]] * 2,
        max_length=15,
    )
    assert len(labels) == 2
    assert len(set(labels)) == 2
    # The disambiguated labels must actually surface the differentiating parameter, not just
    # differ by some opaque suffix that carries no information.
    assert "lea_rat=0.1" in labels[0]
    assert "lea_rat=0.2" in labels[1]


def test_format_series_labels_last_resort_disambiguator_is_the_run_name() -> None:
    """Two records genuinely identical on every one of THEIR OWN varying names (e.g. two runs
    whose passed-in varying names happen to coincide in value, even though the full runs differ)
    still get pairwise-distinct labels, via a suffix built from each record's own (unique) run
    name -- this is the guaranteed-terminating last resort, reached because neither the capped
    nor the uncapped hyperparameter text can tell these two apart on their own."""
    records = [
        _hp_record("run_x", {"a": 1}),
        _hp_record("run_y", {"a": 1}),
    ]
    labels = format_series_labels(
        records,
        ["[best] family_a", "[best] family_a"],
        [["a"], ["a"]],
        max_length=60,
    )
    assert len(set(labels)) == 2
    assert "run_x" in labels[0]
    assert "run_y" in labels[1]


def test_format_series_labels_no_collision_leaves_labels_at_the_initial_cap() -> None:
    """When the initial capped labels already differ, nothing is re-rendered -- most labels in
    a chart legend should stay short rather than every label paying the disambiguation cost."""
    records = [
        _hp_record("run_0", {"learning_rate": 0.1}),
        _hp_record("run_1", {"learning_rate": 0.2}),
    ]
    labels = format_series_labels(
        records,
        ["[best] simple_q_learning", "[median] simple_q_learning"],
        [["learning_rate"], ["learning_rate"]],
        max_length=60,
    )
    assert labels == ["[best] simple_q_learning lea_rat=0.1", "[median] simple_q_learning lea_rat=0.2"]


def test_format_series_labels_respects_per_record_varying_names() -> None:
    """Different records may pass different `varying_names` (different model families declaring
    different hyperparameters) -- each record's label is built from its own list."""
    records = [
        _hp_record("run_0", {"batch_size": 64}),
        _hp_record("run_1", {"learning_rate": 0.1}),
    ]
    labels = format_series_labels(
        records,
        ["[best] deep_q_learning", "[median] simple_q_learning"],
        [["batch_size"], ["learning_rate"]],
        max_length=60,
    )
    assert labels == ["[best] deep_q_learning bat_siz=64", "[median] simple_q_learning lea_rat=0.1"]


def test_format_series_labels_rejects_mismatched_lengths() -> None:
    records = [_hp_record("run_0", {"a": 1})]
    with pytest.raises(ValueError, match="same length"):
        format_series_labels(records, ["prefix_1", "prefix_2"], [["a"], ["a"]])


# ---------------------------------------------------------------------------------------------
# format_top_table_hyperparameter_cells (same defect A, applied to the top-run summary table's
# hyperparameters column instead of a chart legend: a length cap must never truncate away the
# one hyperparameter that distinguishes two ranked runs, e.g. two rows both rendering
# "eps_min=0.05 dis_..." while differing only in `learning_rate`).
# ---------------------------------------------------------------------------------------------


def test_format_top_table_hyperparameter_cells_disambiguates_a_collision_caused_by_truncation() -> None:
    """Three records that agree on their highest-importance hyperparameters and differ only on a
    lower-importance one must produce three DISTINCT table cells -- a short cap that only fits
    the shared, high-importance parameters must not be allowed to hide the differentiating one."""
    records = [
        _hp_record("run_0", {"discount_factor": 0.99, "epsilon_min": 0.05, "learning_rate": 0.1}),
        _hp_record("run_1", {"discount_factor": 0.99, "epsilon_min": 0.05, "learning_rate": 0.2}),
        _hp_record("run_2", {"discount_factor": 0.99, "epsilon_min": 0.05, "learning_rate": 0.3}),
    ]
    # Importance-ranked: discount_factor and epsilon_min (identical across all three records)
    # are listed first, learning_rate (the ONLY differentiator) last -- exactly the shape a short
    # cap truncates away.
    varying_names = [["discount_factor", "epsilon_min", "learning_rate"]] * 3
    cells = format_top_table_hyperparameter_cells(records, varying_names, max_length=20)

    assert len(cells) == 3
    assert len(set(cells)) == 3
    assert "lea_rat=0.1" in cells[0]
    assert "lea_rat=0.2" in cells[1]
    assert "lea_rat=0.3" in cells[2]


def test_format_top_table_hyperparameter_cells_last_resort_disambiguator_is_the_run_name() -> None:
    """Two records identical on every one of their own varying names still get pairwise-distinct
    cells, via each record's own (unique) run name as the guaranteed-terminating last resort."""
    records = [
        _hp_record("run_x", {"a": 1}),
        _hp_record("run_y", {"a": 1}),
    ]
    cells = format_top_table_hyperparameter_cells(records, [["a"], ["a"]], max_length=60)

    assert len(set(cells)) == 2
    assert "run_x" in cells[0]
    assert "run_y" in cells[1]


def test_format_top_table_hyperparameter_cells_no_collision_leaves_cells_at_the_initial_cap() -> None:
    """When the initial capped cells already differ, nothing is re-rendered."""
    records = [
        _hp_record("run_0", {"learning_rate": 0.1}),
        _hp_record("run_1", {"learning_rate": 0.2}),
    ]
    cells = format_top_table_hyperparameter_cells(records, [["learning_rate"], ["learning_rate"]], max_length=60)

    assert cells == ["lea_rat=0.1", "lea_rat=0.2"]


def test_format_top_table_hyperparameter_cells_rejects_mismatched_lengths() -> None:
    records = [_hp_record("run_0", {"a": 1})]
    with pytest.raises(ValueError, match="same length"):
        format_top_table_hyperparameter_cells(records, [["a"], ["a"]])


# ---------------------------------------------------------------------------------------------
# hyperparameter_grid_cardinality (feature 004: per-family grid size vs. runs actually present --
# NEVER a union across model families, which massively over-counts when families declare
# disjoint hyperparameter sets, e.g. 960 vs 135 measured on a real FrozenLake group).
# ---------------------------------------------------------------------------------------------


def _grid_record(name: str, model_name: str, hyperparameters: dict[str, object]) -> RunRecord:
    return RunRecord(
        directory=Path(name), model_name=model_name, env_id="FrozenLake-v1", hyperparameters=hyperparameters
    )


def test_hyperparameter_grid_cardinality_complete_grid_matches_measured_frozenlake_shape() -> None:
    """Reproduces the measured deep_q_learning shape: a 2x3x3x3x2 = 108-cell grid, fully
    populated -- runs_present == cells and the grid reports complete with zero missing cells."""
    records = []
    i = 0
    for a in range(2):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(2):
                        records.append(
                            _grid_record(
                                f"run_{i}",
                                "deep_q_learning",
                                {"a": a, "b": b, "c": c, "d": d, "e": e},
                            )
                        )
                        i += 1

    cardinality = hyperparameter_grid_cardinality(records)

    assert isinstance(cardinality, HyperparameterGridCardinality)
    assert cardinality.model_name == "deep_q_learning"
    assert cardinality.cells == 108
    assert cardinality.runs_present == 108
    assert cardinality.dimensions_expression == "2x3x3x3x2"
    assert cardinality.is_complete is True
    assert cardinality.missing_cells == 0


def test_hyperparameter_grid_cardinality_partial_grid_reports_missing_cells() -> None:
    """A deliberately PARTIAL grid: a 2x3 = 6-cell grid with only 5 runs present -- one cell was
    never trained. `is_complete` is False and `missing_cells` states exactly how many."""
    combos = [(a, b) for a in range(2) for b in range(3)]
    combos.pop()  # drop the last combination -- one cell is now missing
    records = [_grid_record(f"run_{i}", "simple_q_learning", {"a": a, "b": b}) for i, (a, b) in enumerate(combos)]

    cardinality = hyperparameter_grid_cardinality(records)

    assert cardinality.cells == 6
    assert cardinality.runs_present == 5
    assert cardinality.dimensions_expression == "2x3"
    assert cardinality.is_complete is False
    assert cardinality.missing_cells == 1


def test_hyperparameter_grid_cardinality_ignores_constant_hyperparameters() -> None:
    """A hyperparameter held constant within this family (e.g. `seed`) contributes no dimension
    and no factor to the product, even though it is present on every record."""
    records = [_grid_record(f"run_{i}", "dummy", {"a": float(i), "seed": 42}) for i in range(3)]

    cardinality = hyperparameter_grid_cardinality(records)

    assert [d.name for d in cardinality.dimensions] == ["a"]
    assert cardinality.cells == 3


def test_hyperparameter_grid_cardinality_single_grid_point_when_nothing_varies() -> None:
    """The measured `dummy` family shape: nothing varies, a single grid point, one run."""
    records = [_grid_record("run_0", "dummy", {"seed": 42})]

    cardinality = hyperparameter_grid_cardinality(records)

    assert cardinality.dimensions == []
    assert cardinality.cells == 1
    assert cardinality.runs_present == 1
    assert cardinality.dimensions_expression == "1"
    assert cardinality.is_complete is True


def test_hyperparameter_grid_cardinality_never_unions_across_families() -> None:
    """Two families with disjoint hyperparameter sets must be computed independently -- a caller
    that (incorrectly) unioned them would over-count; this asserts the per-family API only ever
    sees one family's own records and their own grid size."""
    deep_q_records = [
        _grid_record(f"dq_{i}", "deep_q_learning", {"batch_size": float(bs), "learning_rate": float(lr)})
        for i, (bs, lr) in enumerate((bs, lr) for bs in range(2) for lr in range(3))
    ]
    simple_q_records = [_grid_record(f"sq_{i}", "simple_q_learning", {"epsilon": float(i)}) for i in range(4)]

    deep_q_cardinality = hyperparameter_grid_cardinality(deep_q_records)
    simple_q_cardinality = hyperparameter_grid_cardinality(simple_q_records)

    assert deep_q_cardinality.cells == 6
    assert simple_q_cardinality.cells == 4
    # A union of the two families' varying hyperparameters would give 6 * 4 = 24, not 6 and 4
    # computed independently.


def test_hyperparameter_grid_cardinality_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        hyperparameter_grid_cardinality([])


# ---------------------------------------------------------------------------------------------
# rank_runs_by_performance (feature 004: top-N summary table, replacing a one-row-per-run dump).
# ---------------------------------------------------------------------------------------------


def _perf_record(name: str, model_name: str, performance: float | None) -> RunRecord:
    kwargs = {} if performance is None else {"testing_rewards": [performance], "testing_steps": [1]}
    return RunRecord(directory=Path(name), model_name=model_name, env_id="FrozenLake-v1", **kwargs)


def test_rank_runs_by_performance_orders_descending_and_caps_at_top_n() -> None:
    records = [_perf_record(f"run_{i}", "deep_q_learning", performance=float(i)) for i in range(10)]

    ranked = rank_runs_by_performance(records, top_n=3)

    assert [r.rank for r in ranked] == [1, 2, 3]
    assert [r.record.performance for r in ranked] == [9.0, 8.0, 7.0]


def test_rank_runs_by_performance_excludes_records_with_no_usable_performance() -> None:
    records = [
        _perf_record("run_with_perf", "deep_q_learning", performance=1.0),
        _perf_record("run_without_perf", "deep_q_learning", performance=None),
    ]

    ranked = rank_runs_by_performance(records, top_n=3)

    assert len(ranked) == 1
    assert ranked[0].record.run_name == "run_with_perf"


def test_rank_runs_by_performance_empty_when_nothing_ranks() -> None:
    records = [_perf_record("run_0", "deep_q_learning", performance=None)]
    assert rank_runs_by_performance(records) == []


def test_rank_runs_by_performance_ties_broken_by_directory_name() -> None:
    """Same tie-break as `select_series`: deterministic across regeneration."""
    records = [
        _perf_record("run_b", "deep_q_learning", performance=1.0),
        _perf_record("run_a", "deep_q_learning", performance=1.0),
    ]
    ranked = rank_runs_by_performance(records, top_n=2)
    assert [r.record.run_name for r in ranked] == ["run_a", "run_b"]


def test_rank_runs_by_performance_returns_ranked_run_instances() -> None:
    records = [_perf_record("run_0", "deep_q_learning", performance=1.0)]
    ranked = rank_runs_by_performance(records)
    assert isinstance(ranked[0], RankedRun)


# ---------------------------------------------------------------------------------------------
# select_top_table_metric_columns (feature 004 fix: drop metric columns that duplicate
# `performance` or each other, instead of always rendering all three).
# ---------------------------------------------------------------------------------------------


def _metric_record(
    name: str,
    *,
    testing_rewards: list[float] | None = None,
    learning_rewards: list[float] | None = None,
) -> RunRecord:
    kwargs: dict[str, list[float]] = {}
    if testing_rewards is not None:
        kwargs["testing_rewards"] = testing_rewards
        kwargs["testing_steps"] = [1] * len(testing_rewards)
    if learning_rewards is not None:
        kwargs["learning_rewards"] = learning_rewards
        kwargs["learning_steps"] = [1] * len(learning_rewards)
    return RunRecord(directory=Path(name), model_name="deep_q_learning", env_id="FrozenLake-v1", **kwargs)


def test_select_top_table_metric_columns_drops_everything_redundant_on_binary_reward() -> None:
    """On a binary reward, performance == mean_testing_reward == testing_success_rate exactly;
    only mean_learning_reward (a genuinely different phase) survives."""
    records = [
        _metric_record("r0", testing_rewards=[1.0], learning_rewards=[0.5]),
        _metric_record("r1", testing_rewards=[0.0], learning_rewards=[0.2]),
    ]
    assert select_top_table_metric_columns(records) == ["mean_learning_reward"]


def test_select_top_table_metric_columns_keeps_success_rate_on_continuous_reward() -> None:
    """`mean_testing_reward` still duplicates `performance` by construction (every run has a
    testing phase), but `testing_success_rate` genuinely differs once the reward is continuous."""
    records = [
        _metric_record("r0", testing_rewards=[10.0], learning_rewards=[1.0]),
        _metric_record("r1", testing_rewards=[20.0], learning_rewards=[2.0]),
    ]
    assert select_top_table_metric_columns(records) == ["testing_success_rate", "mean_learning_reward"]


def test_select_top_table_metric_columns_keeps_mean_testing_reward_when_some_runs_lack_it() -> None:
    """`performance` falls back to the learning mean for a run with no testing phase, so it no
    longer duplicates `mean_testing_reward` -- the column must stay."""
    records = [
        _metric_record("r0", testing_rewards=[5.0], learning_rewards=[1.0]),
        _metric_record("r1", testing_rewards=None, learning_rewards=[2.0]),
    ]
    assert select_top_table_metric_columns(records) == [
        "mean_testing_reward",
        "testing_success_rate",
        "mean_learning_reward",
    ]


def test_select_top_table_metric_columns_empty_on_empty_records() -> None:
    assert select_top_table_metric_columns([]) == []


# ---------------------------------------------------------------------------------------------
# format_relative_run_path (feature 004 fix: full, untruncated, findable path -- replaces the
# tail-truncated absolute `path` table column).
# ---------------------------------------------------------------------------------------------


def test_format_relative_run_path_relative_to_report_dir() -> None:
    report_dir = Path("outputs/frozenlake_4x4/FrozenLake-v1/env_sig")
    record = _metric_record(str(report_dir / "simple_q_learning" / "sig_a"), testing_rewards=[1.0])

    assert format_relative_run_path(record, report_dir) == "simple_q_learning/sig_a"


def test_format_relative_run_path_falls_back_to_absolute_when_not_nested() -> None:
    record = _metric_record("other/root/simple_q_learning/sig_a", testing_rewards=[1.0])
    report_dir = Path("outputs/frozenlake_4x4/FrozenLake-v1/env_sig")

    result = format_relative_run_path(record, report_dir)

    assert result == record.directory.as_posix()
