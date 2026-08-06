"""Bounded, ranked series selection for one chart (User Story 3).

`select_series(records, metric, per_bucket=3)` caps every multi-run chart at `3 * per_bucket`
series — best/median/worst — so a 135-run comparative report never asks a reader to
distinguish more curves than that (FR-010..FR-015). Determinism over pervasive ties (many
FrozenLake runs score exactly 0) comes from sorting on `(-metric_value, directory_name)`:
`directory_name` is unique within a report group and stable on disk (FR-014, SC-009).
"""

from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from hercule.reports.run_table import RunRecord


RankingMetric = Literal[
    "mean_learning_reward",
    "learning_success_rate",
    "mean_testing_reward",
    "testing_success_rate",
]
"""The name of the `RunRecord` derived field a chart ranks on."""


class SeriesBucket(str, Enum):
    """Which ranked group a selected run belongs to, for the legend and report prose (FR-013).

    `str, Enum` rather than `enum.StrEnum`: `StrEnum` is 3.11+ and the project floor is 3.10.
    Ordering is `BEST < MEDIAN < WORST` by declaration; deduplication and legend ordering rely
    on this order.
    """

    BEST = "best"
    MEDIAN = "median"
    WORST = "worst"


class SelectedSeries(BaseModel):
    """One run chosen for a chart, tagged with the bucket and value it was ranked on."""

    record: RunRecord
    bucket: SeriesBucket
    metric_value: float | None = None


class SeriesSelection(BaseModel):
    """The bounded, ranked subset of a report group chosen for one chart.

    Produced by `select_series(records, metric, per_bucket=3)`.
    """

    metric: RankingMetric
    series: list[SelectedSeries] = Field(default_factory=list)
    omitted_count: int = Field(ge=0)

    @property
    def total_count(self) -> int:
        """Total number of records the selection was drawn from."""
        return len(self.series) + self.omitted_count

    @property
    def counts_by_bucket(self) -> dict[SeriesBucket, int]:
        """Per-bucket count of selected series, for the report's prose (FR-013)."""
        counts: dict[SeriesBucket, int] = dict.fromkeys(SeriesBucket, 0)
        for selected in self.series:
            counts[selected.bucket] += 1
        return counts

    @model_validator(mode="after")
    def _validate_bounds_and_uniqueness(self) -> "SeriesSelection":
        if len(self.series) > 9:
            raise ValueError(f"series selection exceeds the 9-series cap: {len(self.series)}")
        seen: set = set()
        for selected in self.series:
            if selected.record.directory in seen:
                raise ValueError(f"duplicate run in series selection: {selected.record.directory}")
            seen.add(selected.record.directory)
        return self


def select_series(
    records: Sequence[RunRecord],
    metric: RankingMetric,
    per_bucket: int = 3,
) -> SeriesSelection:
    """Rank `records` on `metric` and select up to `3 * per_bucket` for one chart.

    Sorts descending on `metric` (a record with no value for it sorts last, tie-broken by
    `directory.name`, never dropped — FR-014). Takes the first `per_bucket` as `BEST`, the last
    `per_bucket` as `WORST`, and `per_bucket` entries centred on the median index as `MEDIAN`.
    When buckets overlap (small groups), a run keeps its first assignment in
    `BEST -> MEDIAN -> WORST` order, so `series` never holds the same run twice. When
    `len(records) <= 3 * per_bucket`, every record is returned and `omitted_count == 0`
    (FR-011).

    Args:
        records: The runs to rank; typically one report group's `RunTable.records`.
        metric: The name of the `RunRecord` derived aggregate to rank on. Two charts over the
            same group ranking on different metrics may select different subsets (FR-012).
        per_bucket: Number of runs kept per bucket; the chart draws at most `3 * per_bucket`.

    Returns:
        The bounded, bucketed `SeriesSelection`.
    """

    def _sort_key(record: RunRecord) -> tuple[float, str]:
        value = getattr(record, metric)
        rank = -value if value is not None else float("inf")
        return (rank, record.directory.name)

    ordered = sorted(records, key=_sort_key)
    total = len(ordered)

    if total == 0:
        return SeriesSelection(metric=metric, series=[], omitted_count=0)

    best_range = range(0, min(per_bucket, total))
    worst_start = max(0, total - per_bucket)
    worst_range = range(worst_start, total)
    median_start = max(0, min((total - per_bucket) // 2, total - per_bucket))
    median_range = range(median_start, min(median_start + per_bucket, total))

    series: list[SelectedSeries] = []
    seen_indices: set[int] = set()
    for bucket, index_range in (
        (SeriesBucket.BEST, best_range),
        (SeriesBucket.MEDIAN, median_range),
        (SeriesBucket.WORST, worst_range),
    ):
        for index in index_range:
            if index in seen_indices:
                continue
            seen_indices.add(index)
            record = ordered[index]
            series.append(SelectedSeries(record=record, bucket=bucket, metric_value=getattr(record, metric)))

    return SeriesSelection(metric=metric, series=series, omitted_count=total - len(series))


__all__ = [
    "RankingMetric",
    "SelectedSeries",
    "SeriesBucket",
    "SeriesSelection",
    "select_series",
]
