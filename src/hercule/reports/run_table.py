"""The run table: one row per training run, built by walking a report group's directory tree.

`build_run_table(root)` is the single loading routine used both by the report generator (to
write the manifest and decide whether a group qualifies) and by the generated notebook at
runtime, so there is exactly one implementation (FR-004, FR-005, FR-009). It reads **only**
`environment.json` and `run_info.json` per run — `model.json` is never opened, since a single
report group can hold tens of megabytes of stored model weights (FR-007, SC-010); `model_name`
comes from the run directory's parent name instead (research R2, R3).
"""

import functools
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from hercule.run import run_info_file_name
from hercule.supervisor import environment_file_name


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sanitize_reason(value: str) -> str:
    """Strip ANSI escapes and coerce to ASCII text safe for any console encoding.

    Duplicated from `hercule.reports` rather than imported from it: `hercule.reports.__init__`
    imports this module to re-export its public API, so a top-level import in the other
    direction would be circular. The helper is a few lines of pure text handling with no
    state, so duplication is cheaper than restructuring the import graph (research R11).

    Args:
        value: The raw reason text.

    Returns:
        The stripped, ANSI-free, ASCII-safe text.
    """
    stripped = _ANSI_ESCAPE_RE.sub("", value).strip()
    return stripped.encode("ascii", errors="replace").decode("ascii")


class RunRecord(BaseModel):
    """One run: one trained model in one environment configuration with one hyperparameter set.

    Loaded from its leaf directory by `build_run_table`. Deliberately does **not** hold
    `list[EpochResult]`: only the two primitive lists any report section consumes are kept, to
    avoid the cost of instantiating thousands of Pydantic models per run over a 135-run group.
    """

    directory: Path
    model_name: str
    env_id: str
    env_kwargs: dict[str, bool | int | float | str | None] = Field(default_factory=dict)
    max_episode_steps: int | None = None
    hyperparameters: dict[str, bool | int | float | None] = Field(default_factory=dict)
    learning_rewards: list[float] = Field(default_factory=list)
    learning_steps: list[int] = Field(default_factory=list)
    testing_rewards: list[float] = Field(default_factory=list)
    testing_steps: list[int] = Field(default_factory=list)

    @field_validator("env_id")
    @classmethod
    def _validate_env_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("env_id must not be empty")
        return value

    @field_validator("max_episode_steps")
    @classmethod
    def _validate_max_episode_steps(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_episode_steps must be > 0 when set")
        return value

    @field_validator("hyperparameters")
    @classmethod
    def _validate_hyperparameters(
        cls, value: dict[str, bool | int | float | None]
    ) -> dict[str, bool | int | float | None]:
        for key, item in value.items():
            if not key:
                raise ValueError("hyperparameter keys must not be empty")
            if isinstance(item, list):
                raise ValueError(
                    f"hyperparameter {key!r} still holds a list; expand_variants() must run before loading"
                )
        return value

    @model_validator(mode="after")
    def _validate_matching_lengths(self) -> "RunRecord":
        if len(self.learning_rewards) != len(self.learning_steps):
            raise ValueError("learning_rewards and learning_steps must have the same length")
        if len(self.testing_rewards) != len(self.testing_steps):
            raise ValueError("testing_rewards and testing_steps must have the same length")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def run_name(self) -> str:
        """The run leaf directory's own name; unique within a report group."""
        return self.directory.name

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def episode_count(self) -> int:
        """Number of learning-phase episodes recorded for this run."""
        return len(self.learning_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def testing_episode_count(self) -> int:
        """Number of testing-phase episodes recorded for this run; 0 when there was none."""
        return len(self.testing_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def mean_learning_reward(self) -> float | None:
        """Mean reward over the learning phase, or `None` when there were no episodes."""
        if not self.learning_rewards:
            return None
        return sum(self.learning_rewards) / len(self.learning_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def learning_success_rate(self) -> float | None:
        """Fraction of learning episodes whose reward is strictly greater than 0."""
        if not self.learning_rewards:
            return None
        return sum(1 for reward in self.learning_rewards if reward > 0) / len(self.learning_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def mean_testing_reward(self) -> float | None:
        """Mean reward over the testing phase, or `None` when the run has no evaluation phase.

        `None` rather than `0.0` is load-bearing: a run that genuinely scored a mean reward of
        `0.0` on evaluation must stay distinguishable from a run that was never evaluated.
        """
        if not self.testing_rewards:
            return None
        return sum(self.testing_rewards) / len(self.testing_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def testing_success_rate(self) -> float | None:
        """Fraction of testing episodes whose reward is strictly greater than 0, or `None`."""
        if not self.testing_rewards:
            return None
        return sum(1 for reward in self.testing_rewards if reward > 0) / len(self.testing_rewards)

    @computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def performance(self) -> float | None:
        """The testing-phase mean reward, falling back to the learning-phase mean when absent."""
        if self.mean_testing_reward is not None:
            return self.mean_testing_reward
        return self.mean_learning_reward


class SkippedRun(BaseModel):
    """A run leaf directory that could not be read, so the walk continues past it (FR-008)."""

    path: Path
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        sanitized = _sanitize_reason(value)
        if not sanitized:
            raise ValueError("reason must not be empty")
        return sanitized


def format_environment_summary(
    env_id: str,
    env_kwargs: dict[str, bool | int | float | str | None],
    max_episode_steps: int | None,
) -> str:
    """Format a Gymnasium environment id and its settings as one prose sentence (FR-001, FR-002).

    Used both at generation time — so the environment is named in the report's markdown title
    and overview with the values baked in as literal text, appearing before any cell executes —
    and by the generated notebook itself, in identical form, so a re-run states the same
    identity from the runs it just loaded (FR-003).

    Args:
        env_id: The Gymnasium environment identifier, e.g. `"FrozenLake-v1"`.
        env_kwargs: The environment settings; `{}` when nothing was overridden.
        max_episode_steps: The episode step cap, or `None` when the environment declares none.

    Returns:
        A single human-readable sentence naming the environment and its settings. When
        `env_kwargs` is empty, states so explicitly rather than describing an empty structure
        (FR-002 scenario 2).
    """
    if env_kwargs:
        settings = ", ".join(f"{name}={value}" for name, value in sorted(env_kwargs.items()))
        settings_text = f"settings: {settings}"
    else:
        settings_text = "no environment-specific setting was overridden"

    summary = f"Environment: {env_id} ({settings_text})"
    if max_episode_steps is not None:
        summary += f", max episode steps: {max_episode_steps}"
    return summary


def _numeric_or_boolean_array(values: list, *, allow_object: bool = False) -> object:
    """Build a pandas nullable array for one `to_dataframe()` column.

    Boolean-valued columns become the nullable `boolean` dtype so absence is representable
    without upcasting to object; numeric columns become `float64` (never `int64` — one absent
    value would silently upcast the column anyway, and the PCA in a later phase standardises to
    float regardless). When `allow_object` is set (environment-kwarg columns, which may hold
    strings) and the values are neither uniformly boolean nor uniformly numeric, the column
    falls back to `object` so no information is lost.

    Args:
        values: One value per row, in row order; `None` marks an absent value.
        allow_object: Whether a non-numeric, non-boolean column may fall back to `object`.

    Returns:
        A `pandas.array` with the appropriate dtype.
    """
    non_null = [value for value in values if value is not None]

    if non_null and all(isinstance(value, bool) for value in non_null):
        return pd.array(values, dtype="boolean")

    if non_null and all(isinstance(value, int | float) and not isinstance(value, bool) for value in non_null):
        return pd.array([float(value) if value is not None else None for value in values], dtype="float64")

    if allow_object:
        return pd.array(values, dtype="object")

    # A hyperparameter column observed nowhere as non-null (should not happen in practice) still
    # needs a concrete dtype; float64 all-NaN matches the "no hyperparameter is int64" rule.
    return pd.array([None for _ in values], dtype="float64")


class RunTable(BaseModel):
    """One report group's runs, loaded once and consumed by every chart, table and projection.

    Built by `build_run_table(root)`. Does **not** require `records` to be non-empty: a group
    whose every run was unreadable is a legitimate, reportable outcome. Does **not** require a
    single shared `env_id`: an individual-report path builds a one-record table, and a mixed
    table must degrade rather than raise inside a notebook cell.
    """

    root: Path
    records: list[RunRecord] = Field(default_factory=list)
    skipped: list[SkippedRun] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_directories(self) -> "RunTable":
        seen: set[Path] = set()
        for record in self.records:
            if record.directory in seen:
                raise ValueError(f"duplicate run directory in run table: {record.directory}")
            seen.add(record.directory)
        return self

    @property
    def runs_loaded(self) -> int:
        """Number of runs successfully loaded."""
        return len(self.records)

    @property
    def runs_skipped(self) -> int:
        """Number of runs that could not be read."""
        return len(self.skipped)

    @property
    def env_id(self) -> str | None:
        """The Gymnasium environment id shared by all records; `None` when there are none."""
        return self.records[0].env_id if self.records else None

    @property
    def env_kwargs(self) -> dict[str, bool | int | float | str | None] | None:
        """The environment settings shared by all records; `None` when there are none."""
        return self.records[0].env_kwargs if self.records else None

    @property
    def max_episode_steps(self) -> int | None:
        """The episode step cap shared by all records; `None` when there are none or unset."""
        return self.records[0].max_episode_steps if self.records else None

    @property
    def model_names(self) -> list[str]:
        """Sorted, unique model families present in this group."""
        return sorted({record.model_name for record in self.records})

    @property
    def varying_hyperparameters(self) -> dict[str, list[bool | int | float | None]]:
        """Per hyperparameter name, the sorted distinct values across all records.

        A hyperparameter with a single distinct value everywhere is excluded — it did not vary,
        so it carries no information for the grid table.
        """

        def _sort_key(value: bool | int | float | None) -> tuple:
            return (value is None, value) if value is not None else (True, 0)

        keys: set[str] = set()
        for record in self.records:
            keys.update(record.hyperparameters.keys())

        varying: dict[str, list[bool | int | float | None]] = {}
        for key in sorted(keys):
            distinct = {record.hyperparameters[key] for record in self.records if key in record.hyperparameters}
            if len(distinct) > 1:
                varying[key] = sorted(distinct, key=_sort_key)
        return varying

    def by_model_family(self) -> dict[str, list[RunRecord]]:
        """Group `records` by `model_name`, in sorted-key order (FR-020)."""
        families: dict[str, list[RunRecord]] = {}
        for record in sorted(self.records, key=lambda r: (r.model_name, r.run_name)):
            families.setdefault(record.model_name, []).append(record)
        return dict(sorted(families.items()))

    def to_dataframe(self) -> pd.DataFrame:
        """Project the run table to a wide, scalar-only pandas frame.

        Charts, tables, rankings and any later projection all read this frame, so its column
        names and dtypes are part of the contract (data-model.md). No per-episode list is ever
        placed in a column: `learning_rewards`/`learning_steps`/`testing_rewards`/`testing_steps`
        stay on the `RunRecord` objects, since the largest group would otherwise create
        multi-hundred-MB object columns.

        Returns:
            One row per record, sorted by `(model_name, run_name)`, with `hp_<name>` and
            `env_<name>` columns for every hyperparameter/environment-setting name seen anywhere
            in the table.
        """
        sorted_records = sorted(self.records, key=lambda r: (r.model_name, r.run_name))

        hp_keys = sorted({key for record in self.records for key in record.hyperparameters})
        env_keys = sorted({key for record in self.records for key in record.env_kwargs})

        data: dict[str, list] = {
            "directory": [str(record.directory) for record in sorted_records],
            "run_name": [record.run_name for record in sorted_records],
            "model_name": [record.model_name for record in sorted_records],
            "env_id": [record.env_id for record in sorted_records],
            "episode_count": [record.episode_count for record in sorted_records],
            "testing_episode_count": [record.testing_episode_count for record in sorted_records],
            "mean_learning_reward": [record.mean_learning_reward for record in sorted_records],
            "learning_success_rate": [record.learning_success_rate for record in sorted_records],
            "mean_testing_reward": [record.mean_testing_reward for record in sorted_records],
            "testing_success_rate": [record.testing_success_rate for record in sorted_records],
            "performance": [record.performance for record in sorted_records],
        }

        df = pd.DataFrame(data)
        df["max_episode_steps"] = pd.array([record.max_episode_steps for record in sorted_records], dtype="Int64")

        for key in hp_keys:
            values = [record.hyperparameters.get(key) for record in sorted_records]
            df[f"hp_{key}"] = _numeric_or_boolean_array(values)

        for key in env_keys:
            values = [record.env_kwargs.get(key) for record in sorted_records]
            df[f"env_{key}"] = _numeric_or_boolean_array(values, allow_object=True)

        return df.reset_index(drop=True)


class ReportManifest(BaseModel):
    """The small file written beside a comparative report so its notebook can verify its own
    data directory at runtime instead of hoping (research R9).

    `Path(__file__)` is undefined in a kernel, and the environment-settings level has no
    naturally occurring anchor file (an individual report anchors on `environment.json`
    instead). Every field is a projection of the `RunTable` the generator already built; the
    manifest introduces no information of its own and carries no timestamp, so byte-identical
    regeneration stays possible.
    """

    root: Path
    env_id: str
    env_kwargs: dict[str, bool | int | float | str | None] = Field(default_factory=dict)
    max_episode_steps: int | None = None
    model_names: list[str] = Field(default_factory=list)
    runs_loaded: int = Field(ge=0)
    runs_skipped: int = Field(ge=0)


def _load_run_record(run_dir: Path) -> RunRecord:
    """Load one `RunRecord` from its leaf directory, reading only `environment.json` and
    `run_info.json`.

    Args:
        run_dir: The run leaf directory.

    Returns:
        The loaded `RunRecord`.

    Raises:
        ValueError: `environment.json` or `run_info.json` is missing, not valid JSON, or missing
            a required key. Callers turn this into a `SkippedRun` rather than aborting the walk.
    """
    environment_path = run_dir / environment_file_name
    run_info_path = run_dir / run_info_file_name

    try:
        with open(environment_path, encoding="utf-8") as f:
            environment_data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"{environment_file_name} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{environment_file_name} is not valid JSON: {exc}") from exc

    if "id" not in environment_data:
        raise ValueError(f"{environment_file_name} missing 'id'")

    try:
        with open(run_info_path, encoding="utf-8") as f:
            run_info_data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"{run_info_file_name} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{run_info_file_name} is not valid JSON: {exc}") from exc

    learning_metrics = run_info_data.get("learning_metrics", [])
    testing_metrics = run_info_data.get("testing_metrics", [])

    return RunRecord(
        directory=run_dir,
        model_name=run_dir.parent.name,
        env_id=environment_data["id"],
        env_kwargs=environment_data.get("kwargs", {}),
        max_episode_steps=environment_data.get("max_episode_steps"),
        hyperparameters=run_info_data.get("model_hyperparameters", {}),
        learning_rewards=[metric.get("reward") for metric in learning_metrics],
        learning_steps=[metric.get("steps_number") for metric in learning_metrics],
        testing_rewards=[metric.get("reward") for metric in testing_metrics],
        testing_steps=[metric.get("steps_number") for metric in testing_metrics],
    )


class MetricRedundancy(BaseModel):
    """Two `RunRecord` derived metrics whose per-run values are exactly, bit-identically equal.

    On an environment whose reward is binary (e.g. FrozenLake's `{0, 1}`), the mean reward IS
    the success rate mechanically: `mean(rewards) == mean(rewards > 0)` for every run, so
    `mean_testing_reward`/`testing_success_rate` (respectively the learning-phase pair) carry
    the exact same information and a chart of one is a y-axis-rescaled duplicate of a chart of
    the other -- measured max absolute difference `0.00e+00` on a real `deep_q_learning` family.
    On a continuous-reward environment the two genuinely differ and no `MetricRedundancy` is
    produced for that phase.
    """

    first: str
    second: str
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


_REDUNDANCY_CANDIDATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("mean_learning_reward", "learning_success_rate"),
    ("mean_testing_reward", "testing_success_rate"),
)
"""The only pairs checked by `detect_redundant_metrics` -- the natural reward/success-rate pair
per phase. A cross-phase pair (e.g. learning reward vs testing success rate) is never expected to
coincide and is not checked."""


def _metric_values_equal(records: Sequence[RunRecord], first: str, second: str) -> bool:
    """Whether two `RunRecord` derived metrics are exactly, bit-identically equal across `records`.

    The shared primitive behind `detect_redundant_metrics` (report-group-wide chart redundancy)
    and `select_top_table_metric_columns` (top-run table column redundancy), so both use exactly
    the same equality check rather than two subtly different ones. Comparing the whole per-run
    value list at once (rather than element by element) means the `None` pattern -- a run with no
    episodes for that phase -- must also match for the pair to count as equal.

    Args:
        records: The runs to compare across.
        first: The first `RunRecord` attribute name.
        second: The second `RunRecord` attribute name.

    Returns:
        Whether the two attributes' values are `==` across every record, in order.
    """
    first_values = [getattr(record, first) for record in records]
    second_values = [getattr(record, second) for record in records]
    return first_values == second_values


def detect_redundant_metrics(records: Sequence[RunRecord]) -> list[MetricRedundancy]:
    """Detect `RunRecord` derived metric pairs whose raw per-run values are exactly equal.

    Exact (`==`) comparison only, never a tolerance: the point is to catch a mechanical identity
    (binary reward -> mean equals success rate), not a near-coincidence. Comparing the whole
    per-run list at once (rather than element by element) means the `None` pattern -- a run with
    no episodes for that phase -- must also match for the pair to count as redundant.

    Args:
        records: The runs to compare across. Only the two natural per-phase pairs
            (`_REDUNDANCY_CANDIDATE_PAIRS`) are checked.

    Returns:
        One `MetricRedundancy` per pair whose values are exactly equal across every record;
        an empty list when `records` is empty or neither pair is redundant (e.g. a
        continuous-reward environment, where the two metrics genuinely differ).
    """
    if not records:
        return []

    redundancies = []
    for first, second in _REDUNDANCY_CANDIDATE_PAIRS:
        if _metric_values_equal(records, first, second):
            redundancies.append(
                MetricRedundancy(
                    first=first,
                    second=second,
                    reason=(
                        f"{first} and {second} are bit-identical across {len(records)} run(s) -- the reward "
                        f"domain is binary here, so the mean reward mechanically equals the success rate; "
                        f"only one chart is needed for this pair"
                    ),
                )
            )
    return redundancies


class ConstantMetric(BaseModel):
    """One `RunRecord` derived metric that is constant (exact `np.ptp`, relative tolerance)
    across every run in a report group that has a finite value for it.

    A metric that never varies carries zero ranking information: `select_series` sorts on
    `(-metric_value, directory_name)`, so a perfectly tied metric is ordered purely by the
    directory-name tie-break and the result is labelled `[best]`/`[median]`/`[worst]` as if that
    meant something. CartPole awards +1 reward per step, so `learning_success_rate` and
    `testing_success_rate` are exactly `1.0` for every run there -- structurally, not by chance
    -- while the same two metrics are genuinely informative on FrozenLake. This is distinct from
    `MetricRedundancy`: that detects two *different* metrics being bit-identical to *each
    other*; this detects *one* metric being constant in itself.
    """

    metric: str
    value: float
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be empty")
        return stripped


_CONSTANT_METRIC_CANDIDATES: tuple[str, ...] = (
    "mean_learning_reward",
    "learning_success_rate",
    "mean_testing_reward",
    "testing_success_rate",
)
"""The `RunRecord` derived metrics `detect_constant_metrics` checks -- the same four per-phase
aggregates `_REDUNDANCY_CANDIDATE_PAIRS` compares pairwise, checked here individually instead."""


def detect_constant_metrics(records: Sequence[RunRecord]) -> list[ConstantMetric]:
    """Detect `RunRecord` derived metrics whose value is constant across every run that has one.

    Uses the exact-range (`np.ptp`) relative-tolerance test used throughout this project (e.g.
    `sensitivity._drop_near_constant_columns`), never `sd > 0` or an absolute epsilon: a
    genuinely constant float column can still carry a tiny nonzero standard deviation purely
    from floating-point rounding error.

    Args:
        records: The runs to check across -- typically one whole report group's
            `RunTable.records` (every model family together, not one family alone), since a
            chart ranking on one of these metrics draws from the whole group.

    Returns:
        One `ConstantMetric` per candidate metric that is constant across at least 2 runs with
        a finite value for it; an empty list when `records` is empty, every candidate has fewer
        than 2 finite values, or every candidate genuinely varies.
    """
    if not records:
        return []

    constants: list[ConstantMetric] = []
    for name in _CONSTANT_METRIC_CANDIDATES:
        finite_values = [value for value in (getattr(record, name) for record in records) if value is not None]
        if len(finite_values) < 2:
            continue
        array = np.array(finite_values, dtype=float)
        value_range = float(np.ptp(array))
        scale = max(float(np.abs(array).max()), 1.0)
        if value_range <= 1e-9 * scale:
            constants.append(
                ConstantMetric(
                    metric=name,
                    value=float(array[0]),
                    reason=(
                        f"{name} is constant ({array[0]:g}) across all {len(finite_values)} run(s) that report "
                        f"it -- no ranking on this metric would be meaningful, since every run scores identically"
                    ),
                )
            )
    return constants


class HyperparameterGridDimension(BaseModel):
    """One varying hyperparameter's contribution to a model family's grid size: its name and how
    many distinct values it takes within that family."""

    name: str
    level_count: int = Field(ge=1)


class HyperparameterGridCardinality(BaseModel):
    """One model family's hyperparameter grid size, compared against the runs actually on disk.

    Computed **per model family**, never over the union of every family's varying
    hyperparameters in a report group: different model families can vary different
    hyperparameters over different value sets, so a union-based product wildly over-counts
    (measured on a real FrozenLake group: the union of 6 varying hyperparameters across
    `deep_q_learning` and `simple_q_learning` gives a product of 960 against 135 actual runs,
    because the two families declare disjoint hyperparameter sets).

    `runs_present` is simply `len(records)`: Hercule's on-disk directory signature
    (`BaseConfig.get_hyperparameters_signature`) is a deterministic function of a run's full
    hyperparameter set, so two runs sharing every varying hyperparameter's value would collide
    on the same directory and only load once (`RunTable`'s duplicate-directory invariant) --
    `runs_present` can therefore never exceed `cells`, and `missing_cells` is never negative.
    """

    model_name: str
    dimensions: list[HyperparameterGridDimension] = Field(default_factory=list)
    cells: int = Field(ge=1)
    runs_present: int = Field(ge=0)
    missing_cells: int = Field(ge=0)

    @property
    def is_complete(self) -> bool:
        """Whether every grid cell has at least one run (`missing_cells == 0`)."""
        return self.missing_cells == 0

    @property
    def dimensions_expression(self) -> str:
        """The grid size as a product expression, e.g. `"2x3x3x3x2"`, or `"1"` when nothing
        varies within this family (a single grid point)."""
        if not self.dimensions:
            return "1"
        return "x".join(str(dimension.level_count) for dimension in self.dimensions)

    @model_validator(mode="after")
    def _validate_cells_matches_dimensions(self) -> "HyperparameterGridCardinality":
        expected_cells = 1
        for dimension in self.dimensions:
            expected_cells *= dimension.level_count
        if expected_cells != self.cells:
            raise ValueError(
                f"cells ({self.cells}) must equal the product of dimension level counts ({expected_cells})"
            )
        return self


def hyperparameter_grid_cardinality(records: Sequence[RunRecord]) -> HyperparameterGridCardinality:
    """Compute one model family's hyperparameter grid size vs. the runs actually present.

    Only hyperparameters that vary (more than one distinct value) *within this family's own
    records* become grid dimensions -- a hyperparameter constant within this family (even if it
    varies across other families in the same report group) contributes a factor of 1 and is
    omitted from `dimensions`.

    Args:
        records: One model family's runs (e.g. one value of `RunTable.by_model_family()`); must
            be non-empty.

    Returns:
        The `HyperparameterGridCardinality` for this family.

    Raises:
        ValueError: `records` is empty.
    """
    if not records:
        raise ValueError("hyperparameter_grid_cardinality requires at least one record")

    model_name = records[0].model_name
    keys = sorted({key for record in records for key in record.hyperparameters})

    dimensions: list[HyperparameterGridDimension] = []
    for key in keys:
        distinct = {record.hyperparameters[key] for record in records if key in record.hyperparameters}
        if len(distinct) > 1:
            dimensions.append(HyperparameterGridDimension(name=key, level_count=len(distinct)))

    cells = 1
    for dimension in dimensions:
        cells *= dimension.level_count

    runs_present = len(records)
    missing_cells = max(0, cells - runs_present)

    return HyperparameterGridCardinality(
        model_name=model_name,
        dimensions=dimensions,
        cells=cells,
        runs_present=runs_present,
        missing_cells=missing_cells,
    )


def _abbreviate_hyperparameter_name(name: str) -> str:
    """Abbreviate a hyperparameter name to first-3-letters-per-word, split on non-alphanumeric
    characters -- the exact scheme `BaseConfig.get_hyperparameters_signature` already uses for
    on-disk directory names (e.g. `batch_size` -> `bat_siz`), so a reader who has seen a run's
    directory recognises the same shorthand in a chart legend or tick label."""
    words = [word for word in re.split(r"[^a-zA-Z0-9]", name) if word]
    return "_".join(word[:3] for word in words) if words else name


def _format_hyperparameter_value(value: bool | int | float | None) -> str:
    """Format one hyperparameter value compactly for a legend/tick label."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_varying_hyperparameters(record: RunRecord, varying_names: Sequence[str], *, max_length: int = 60) -> str:
    """Format `record`'s hyperparameters restricted to `varying_names`, abbreviated and capped.

    A full run-directory signature is roughly 100 characters (it names every hyperparameter,
    including the ones held constant across the whole report group) -- used as a chart legend
    entry it makes the legend box wider than the plot it decorates. Restricting to the group's
    *varying* hyperparameters (`RunTable.varying_hyperparameters`, already computed once per
    report -- not duplicated here) keeps only what actually distinguishes one series from
    another.

    Args:
        record: The run whose hyperparameters are formatted.
        varying_names: The hyperparameter names to include, in the order they appear in the
            label; callers should pass them alphabetically sorted (as
            `RunTable.varying_hyperparameters` already yields) for a deterministic label across
            regeneration. A name absent from `record.hyperparameters` is silently skipped (a
            different model family in the same group may not declare it at all).
        max_length: Backstop cap on the returned string's length -- a label that would still
            exceed it (many varying hyperparameters at once) is truncated with a trailing
            ellipsis rather than left to overflow the legend.

    Returns:
        A short, deterministic string, e.g. `"bat_siz=64 lea_rat=0.001"`, or the run's own name
        when none of `varying_names` is present on this record.
    """
    parts = [
        f"{_abbreviate_hyperparameter_name(name)}={_format_hyperparameter_value(record.hyperparameters[name])}"
        for name in varying_names
        if name in record.hyperparameters
    ]
    text = " ".join(parts) if parts else record.run_name
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text


# A cap large enough that `format_varying_hyperparameters` never truncates in practice -- used
# by `_disambiguate_labels` to recover the untruncated text for a colliding label without
# duplicating its formatting logic under a different code path.
_UNTRUNCATED_LENGTH = 10_000


def _disambiguate_labels(
    labels: list[str],
    records: Sequence[RunRecord],
    render_uncapped: Callable[[int], str],
) -> list[str]:
    """Resolve collisions in an initial, length-capped set of labels, guaranteeing pairwise
    distinctness (defect A).

    Shared by `format_series_labels` (chart legends/tick labels, one prefixed label per record)
    and `format_top_table_hyperparameter_cells` (the top-run table's hyperparameters column, no
    prefix) so the two-step resolution -- re-render uncapped, then fall back to the run's own
    name -- is implemented exactly once. Mutates and returns `labels` in place.

    Args:
        labels: The initial, possibly-colliding label per record, same order as `records`.
        records: The records the labels describe, in the same order.
        render_uncapped: Given a record's index, returns that record's uncapped rendering (step
            1). Callers close over whatever prefix/varying-names shape their label needs.

    Returns:
        `labels`, with every colliding group resolved to pairwise-distinct text.
    """
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue

        # Step 1: re-render uncapped -- resolves any collision caused by truncation alone.
        uncapped = [render_uncapped(i) for i in indices]
        if len(set(uncapped)) == len(indices):
            for i, label in zip(indices, uncapped, strict=True):
                labels[i] = label
            continue

        # Step 2: last resort -- append the run's own leaf directory name, guaranteed unique
        # within a report group.
        for i, label in zip(indices, uncapped, strict=True):
            labels[i] = f"{label} #{records[i].run_name}"

    return labels


def format_series_labels(
    records: Sequence[RunRecord],
    prefixes: Sequence[str],
    varying_names: Sequence[Sequence[str]],
    *,
    max_length: int = 60,
) -> list[str]:
    """Format one legend/tick label per record, guaranteed pairwise distinct (defect A).

    A length-capped label built from only the first few varying hyperparameters can collide:
    two different runs whose differentiating parameter sorts late (or is dropped for space)
    render identically, and a legend with two indistinguishable entries cannot do its job.
    This builds an initial label per record via `format_varying_hyperparameters`, then resolves
    any collision via `_disambiguate_labels`, applied only to the colliding entries so most
    labels stay short:

    1. Re-render the colliding records' hyperparameter text uncapped (every varying name given
       for that record). This alone resolves a collision caused by truncation, since any two
       distinct runs sharing a report group differ in at least one varying hyperparameter by
       construction (`RunTable.varying_hyperparameters`) — unless the two happen to share the
       same value for every name in `varying_names` while differing only in a hyperparameter
       that does not vary within *this specific* record subset (e.g. two different runs from
       two different model families whose OWN varying hyperparameters happen to coincide).
    2. If still equal, append a short, stable disambiguator: `record.run_name`, which is the
       run's own leaf directory name and therefore encodes its complete hyperparameter
       signature (`BaseConfig.get_hyperparameters_signature`) -- guaranteed unique within a
       report group (`RunTable`'s directory-uniqueness invariant), so this step always
       terminates.

    Args:
        records: The runs to label, e.g. one chart's selected `RunRecord`s, in the order the
            returned labels correspond to.
        prefixes: One prefix per record (e.g. a bucket tag and model family name), the same
            length and order as `records`.
        varying_names: One ordered sequence of hyperparameter names per record — typically that
            record's own model family's varying hyperparameters, ranked by descending
            importance (`hercule.reports.sensitivity.order_varying_hyperparameters_by_importance`)
            so a length cap always drops the least informative parameters first. Different
            records may pass different sequences, since different model families in the same
            report group can declare different hyperparameters.
        max_length: Backstop cap for each record's initial pass, forwarded to
            `format_varying_hyperparameters`. A label disambiguated in step 1 or 2 above may
            exceed it.

    Returns:
        One label per record, in the same order as `records`, guaranteed pairwise distinct.

    Raises:
        ValueError: `records`, `prefixes` and `varying_names` do not all have the same length.
    """
    if not (len(records) == len(prefixes) == len(varying_names)):
        raise ValueError(
            "records, prefixes and varying_names must have the same length: "
            f"{len(records)}, {len(prefixes)}, {len(varying_names)}"
        )

    labels = [
        f"{prefix} {format_varying_hyperparameters(record, names, max_length=max_length)}"
        for record, prefix, names in zip(records, prefixes, varying_names, strict=True)
    ]

    def _render_uncapped(index: int) -> str:
        hyperparameter_text = format_varying_hyperparameters(
            records[index], varying_names[index], max_length=_UNTRUNCATED_LENGTH
        )
        return f"{prefixes[index]} {hyperparameter_text}"

    return _disambiguate_labels(labels, records, _render_uncapped)


def format_top_table_hyperparameter_cells(
    records: Sequence[RunRecord],
    varying_names: Sequence[Sequence[str]],
    *,
    max_length: int = 39,
) -> list[str]:
    """Format the top-run summary table's hyperparameters column, guaranteed pairwise distinct
    among `records` (the same defect A as `format_series_labels`, applied to a table cell rather
    than a chart legend -- a 20-character cap previously truncated away the one hyperparameter
    that told two ranked runs apart, e.g. two `simple_q_learning` rows both rendering
    `"eps_min=0.05 dis_..."` while differing only in `learning_rate`).

    Shares its collision-resolution logic with `format_series_labels` via `_disambiguate_labels`
    rather than reimplementing it. The table has no per-row prefix (rank and model name are
    already their own columns), so this renders bare hyperparameter text instead of going
    through `format_series_labels`'s prefix-plus-hyperparameters label shape.

    Args:
        records: The ranked runs shown in the table, in row order.
        varying_names: One ordered sequence of hyperparameter names per record, importance-ranked
            (as `format_series_labels` expects) so an unavoidable truncation drops the least
            informative parameter first.
        max_length: Backstop cap for each record's initial pass. Defaults to `39`: the previous
            `20`-character cap plus the ~19 characters of spare width measured between a rendered
            table's current line length (69) and a monospace PDF page's printable width (~88).

    Returns:
        One cell per record, in the same order as `records`, guaranteed pairwise distinct.

    Raises:
        ValueError: `records` and `varying_names` do not have the same length.
    """
    if len(records) != len(varying_names):
        raise ValueError(f"records and varying_names must have the same length: {len(records)}, {len(varying_names)}")

    cells = [
        format_varying_hyperparameters(record, names, max_length=max_length)
        for record, names in zip(records, varying_names, strict=True)
    ]

    def _render_uncapped(index: int) -> str:
        return format_varying_hyperparameters(records[index], varying_names[index], max_length=_UNTRUNCATED_LENGTH)

    return _disambiguate_labels(cells, records, _render_uncapped)


class RankedRun(BaseModel):
    """One run, ranked by descending `performance`, for the top-N summary table."""

    rank: int = Field(ge=1)
    record: RunRecord


def rank_runs_by_performance(records: Sequence[RunRecord], top_n: int = 3) -> list[RankedRun]:
    """Rank runs with a usable `performance` value and return the top `top_n`.

    Replaces printing one row per run (`RunTable.to_dataframe()`'s full `performance` column) --
    a reader wants "who won and by how much", not every run's figures. Records with no usable
    `performance` (no learning phase at all, e.g. a corrupt/skipped run never reaches this
    function) are excluded entirely rather than sorted to the bottom, since there is nothing to
    rank them on.

    Args:
        records: The runs to rank -- typically a whole report group's `RunTable.records` (every
            model family together), since the summary table compares across families.
        top_n: Number of top-ranked runs to return.

    Returns:
        Up to `top_n` `RankedRun`s, `rank=1` first (highest `performance`), tie-broken by
        `directory.name` for determinism across regeneration -- the same tie-break
        `select_series()` uses. Empty when no record has a usable `performance`.
    """
    rankable = [record for record in records if record.performance is not None]

    def _sort_key(record: RunRecord) -> tuple[float, str]:
        return (-record.performance, record.directory.name)  # type: ignore[operator]

    ordered = sorted(rankable, key=_sort_key)
    return [RankedRun(rank=index + 1, record=record) for index, record in enumerate(ordered[:top_n])]


_TOP_TABLE_METRIC_CANDIDATES: tuple[str, ...] = (
    "mean_testing_reward",
    "testing_success_rate",
    "mean_learning_reward",
)
"""Extra metric columns considered for the top-run summary table, in the order they are
evaluated and, when kept, rendered -- `performance` itself is always the first column and is
never one of these candidates."""

TOP_TABLE_COLUMN_LABELS: dict[str, str] = {
    "performance": "performance",
    "mean_testing_reward": "test_reward",
    "testing_success_rate": "test_success",
    "mean_learning_reward": "learn_reward",
}
"""Short display headers for the top-run table's columns, keyed by the `RunRecord` attribute
name (`select_top_table_metric_columns`'s output plus `"performance"` itself). The full
attribute names (`mean_testing_reward`, `testing_success_rate`, `mean_learning_reward`) are
20 characters each; three such headers alone already push a monospace table past a typical PDF
page's printable width even after every other column is minimised, so the table renders these
shorter labels while the rest of the report keeps using the full attribute names."""


def select_top_table_metric_columns(records: Sequence[RunRecord]) -> list[str]:
    """Select which extra metric columns belong beside `performance` in the top-run table.

    `performance` (the testing-phase mean reward, falling back to the learning-phase mean) is
    always the table's headline column. This adds only the columns that are not an exact
    duplicate of it or of a column already selected, using the same bit-identical equality check
    `detect_redundant_metrics` uses (`_metric_values_equal`) rather than a second, parallel
    comparison:

    - `mean_testing_reward` is dropped once every record in `records` has a testing phase, since
      `performance` is *defined* as `mean_testing_reward` whenever one exists -- the two columns
      are then identical by construction, not merely by coincidence.
    - `testing_success_rate` is dropped when it is bit-identical to whichever of `performance` /
      `mean_testing_reward` was kept (the mechanical binary-reward identity).
    - `mean_learning_reward` reflects a different phase than the other three columns, so it is
      dropped only if it happens to be bit-identical too.

    Args:
        records: The runs the table will show, typically `RankedRun.record` for each row.

    Returns:
        The ordered extra column names to render after `performance`, a subset of
        `_TOP_TABLE_METRIC_CANDIDATES`; empty when `records` is empty.
    """
    if not records:
        return []

    kept = ["performance"]
    columns: list[str] = []
    for candidate in _TOP_TABLE_METRIC_CANDIDATES:
        if any(_metric_values_equal(records, candidate, other) for other in kept):
            continue
        columns.append(candidate)
        kept.append(candidate)
    return columns


def format_relative_run_path(record: RunRecord, report_dir: Path) -> str:
    """Format a run's directory relative to the report's own directory.

    Replaces a tail-truncated absolute path in the top-run table: a comparative report always
    sits inside the group directory its runs belong to, so the relative form is short (typically
    `<model_name>/<signature>`, e.g. `simple_q_learning/eps_dec_0.005__eps_min_0.05...`) and
    never needs truncating, while still being exactly findable on disk from the report's own
    location.

    Args:
        record: The run whose directory is formatted.
        report_dir: The report's own directory (`RunTable.root` for a comparative report).

    Returns:
        `record.directory` relative to `report_dir`, rendered with forward slashes so the same
        string reads correctly regardless of the host OS the report was generated or read on.
        Falls back to the absolute path (still forward-slashed) when `record.directory` is not
        inside `report_dir` -- defensive only, since every `RunRecord` returned by
        `build_run_table(report_dir)` is always inside it.
    """
    try:
        relative = record.directory.relative_to(report_dir)
    except ValueError:
        relative = record.directory
    return relative.as_posix()


def build_run_table(root: Path) -> RunTable:
    """Walk `root` and load every run leaf directory into one `RunTable`.

    The same function is called both by the report generator (to write the manifest and decide
    whether a group qualifies) and by the generated notebook at execution time, so there is
    exactly one implementation of "what is a run" (FR-004, FR-005, FR-009). `model.json` is
    never opened (FR-007, SC-010); unreadable runs become a `SkippedRun` and never abort the
    walk (FR-008).

    Args:
        root: The report group root — either a single run leaf directory (individual report) or
            the environment-settings directory holding several runs (comparative report).

    Returns:
        The loaded `RunTable`.
    """
    # Local import: `hercule.reports.__init__` imports this module to re-export its public API,
    # so a top-level import here would be circular.
    from hercule.reports import find_experiment_directories  # noqa: PLC0415

    records: list[RunRecord] = []
    skipped: list[SkippedRun] = []

    for run_dir in find_experiment_directories(root):
        try:
            records.append(_load_run_record(run_dir))
        except ValueError as exc:
            skipped.append(SkippedRun(path=run_dir, reason=str(exc)))

    return RunTable(root=root, records=records, skipped=skipped)
