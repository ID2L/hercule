"""Module for generating experiment reports."""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, field_validator, model_validator

from hercule.models import model_file_name
from hercule.reports.render import ArtifactWriteError, RenderResult, check_artifacts_writable, render_report
from hercule.reports.run_table import (
    TOP_TABLE_COLUMN_LABELS,
    ConstantMetric,
    HyperparameterGridCardinality,
    HyperparameterGridDimension,
    MetricRedundancy,
    RankedRun,
    ReportManifest,
    RunRecord,
    RunTable,
    SkippedRun,
    build_run_table,
    detect_constant_metrics,
    detect_redundant_metrics,
    format_environment_summary,
    format_relative_run_path,
    format_series_labels,
    format_top_table_hyperparameter_cells,
    format_varying_hyperparameters,
    hyperparameter_grid_cardinality,
    rank_runs_by_performance,
    select_top_table_metric_columns,
)
from hercule.reports.selection import (
    RankingMetric,
    SelectedSeries,
    SeriesBucket,
    SeriesSelection,
    select_series,
)
from hercule.reports.sensitivity import (
    HyperparameterEtaSquared,
    ImportanceResult,
    ImportanceUnavailable,
    InteractionCell,
    InteractionGridResult,
    InteractionGridUnavailable,
    InteractionRankingResult,
    InteractionRankingUnavailable,
    MainEffectLevel,
    MainEffectsForHyperparameter,
    MainEffectsResult,
    MainEffectsUnavailable,
    MetricName,
    PairwiseInteraction,
    RankShift,
    ReplicationStatus,
    TopDecileComparisonResult,
    TopDecileComparisonUnavailable,
    VarianceDecompositionEntry,
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
from hercule.run import run_info_file_name
from hercule.supervisor import environment_file_name


logger = logging.getLogger(__name__)

# Maximum depth for recursive search of experiment directories
MAX_DEPTH = 4

# Cell-tag vocabulary shared between the generated templates and the render pipeline
# (contracts C4). nbconvert ships no default tag names, so these are the project's own
# fixed strings; templates and reports/render.py must both use these constants rather
# than literal strings, or they will silently drift apart.
TAG_REMOVE_CELL: Final = "remove_cell"
TAG_REMOVE_INPUT: Final = "remove_input"
TAG_REMOVE_OUTPUT: Final = "remove_output"

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sanitize_reason(value: str) -> str:
    """Strip ANSI escapes and coerce to ASCII text safe for any console encoding.

    Reason strings may be derived from exception text, which can carry arbitrary user
    characters and raw ANSI escape codes (research R11) — printing them unsanitised has
    already been observed to raise `UnicodeEncodeError` on a cp1252 console.

    Args:
        value: The raw reason text.

    Returns:
        The stripped, ANSI-free, ASCII-safe text.
    """
    stripped = _ANSI_ESCAPE_RE.sub("", value).strip()
    return stripped.encode("ascii", errors="replace").decode("ascii")


class SkippedGroup(BaseModel):
    """A candidate report group that was found but not rendered.

    Recorded so a silent absence of output is never mistaken for success (FR-030).
    """

    path: Path
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        sanitized = _sanitize_reason(value)
        if not sanitized:
            raise ValueError("reason must not be empty")
        return sanitized


class ReportArtifacts(BaseModel):
    """Everything produced for one report group.

    Composed from a render pipeline result (paths, PDF skip reason) and a run table
    (loaded/skipped run counts), so the CLI, the controller and tests all read one shape
    (FR-027).

    `source` is the generated jupytext `.py` -- the durable, re-runnable, version-controllable
    artifact; it is always written and always distinct from every other field. `notebook` is
    the *executed* `.ipynb` when `execute=True`, i.e. a derived, disposable product of running
    `source` through a kernel. When `execute=False` (`--no-execute`), no kernel ever runs, so
    there is nothing to derive: `notebook` (and `html`) fall back to `source` itself, and
    callers must compare `notebook == source` before treating them as separate artifacts.
    """

    source: Path
    notebook: Path
    html: Path
    pdf: Path | None = None
    pdf_skip_reason: str | None = None
    runs_loaded: int = Field(ge=0)
    runs_skipped: int = Field(ge=0)

    @field_validator("pdf_skip_reason")
    @classmethod
    def _validate_pdf_skip_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sanitized = _sanitize_reason(value)
        if not sanitized:
            raise ValueError("pdf_skip_reason must not be empty when set")
        return sanitized

    @model_validator(mode="after")
    def _validate_pdf_xor_reason(self) -> "ReportArtifacts":
        if (self.pdf is None) == (self.pdf_skip_reason is None):
            raise ValueError("exactly one of pdf or pdf_skip_reason must be set")
        return self


class ReportBundle(BaseModel):
    """The widened return of `generate_report()`/`generate_experiment_report()`.

    Replaces the previous bare `Path` return so every artifact and every skip reason can
    be reported to the caller (FR-027, FR-030).
    """

    reports: list[ReportArtifacts] = Field(default_factory=list)
    skipped_groups: list[SkippedGroup] = Field(default_factory=list)

    @property
    def report_count(self) -> int:
        """Number of report groups actually rendered."""
        return len(self.reports)

    @property
    def pdf_count(self) -> int:
        """Number of rendered groups whose PDF was produced (not skipped)."""
        return sum(1 for artifact in self.reports if artifact.pdf is not None)

    @property
    def has_skips(self) -> bool:
        """Whether at least one candidate group was skipped."""
        return bool(self.skipped_groups)

    @model_validator(mode="after")
    def _validate_non_empty_and_unique(self) -> "ReportBundle":
        if not self.reports and not self.skipped_groups:
            raise ValueError("a report bundle must contain at least one report or one skipped group")
        seen: set[Path] = set()
        for artifact in self.reports:
            if artifact.notebook in seen:
                raise ValueError(f"duplicate notebook path in report bundle: {artifact.notebook}")
            seen.add(artifact.notebook)
        return self


def is_valid_experiment_directory(directory: Path) -> bool:
    """
    Check if a directory contains all required experiment files.

    Args:
        directory: Path to the directory to check

    Returns:
        True if the directory contains environment.json, model.json, and run_info.json
    """
    if not directory.is_dir():
        return False

    required_files = [
        environment_file_name,
        model_file_name,
        run_info_file_name,
    ]

    return all((directory / filename).exists() for filename in required_files)


def find_experiment_directories(root_directory: Path, max_depth: int = MAX_DEPTH, current_depth: int = 0) -> list[Path]:
    """
    Recursively find all directories that contain a valid experiment structure.

    Searches up to max_depth levels deep starting from root_directory.

    Args:
        root_directory: Root directory to search in
        max_depth: Maximum depth to search (default: MAX_DEPTH)
        current_depth: Current recursion depth (used internally)

    Returns:
        List of paths to directories containing valid experiment structures
    """
    experiment_dirs: list[Path] = []

    if not root_directory.is_dir():
        return experiment_dirs

    # Check if current directory is a valid experiment directory
    if is_valid_experiment_directory(root_directory):
        experiment_dirs.append(root_directory)
        return experiment_dirs

    # If we've reached max depth, stop searching
    if current_depth >= max_depth:
        return experiment_dirs

    # Recursively search subdirectories
    try:
        for item in root_directory.iterdir():
            if item.is_dir():
                subdir_experiments = find_experiment_directories(item, max_depth, current_depth + 1)
                experiment_dirs.extend(subdir_experiments)
    except PermissionError:
        logger.warning(f"Permission denied accessing {root_directory}")

    return experiment_dirs


# Reason recorded when the caller opted out of execution entirely (--no-execute); render.py
# owns the render_pdf=False reason text for the executed-but-not-printed case, since that
# path actually goes through render_report.
_NOT_EXECUTED_REASON: Final = "notebook was not executed (--no-execute)"


def _clear_stale_render_artifacts(report_path: Path) -> None:
    """Remove any `.ipynb`/`.failed.ipynb`/`.html`/`.pdf` left by a previous, executed run.

    Regeneration must replace previously generated artifacts rather than leaving stale ones
    beside them (FR-028). `render_report` performs the same cleanup itself when it runs, but
    when the caller passes `execute=False` it never runs at all -- without this, a report
    previously generated with execution enabled and then regenerated with `--no-execute`
    would keep an `.ipynb`/`.html`/`.pdf` that no longer matches the freshly written `.py`.

    A writability preflight runs first (the same one `render_report` uses), so a locked
    sibling (e.g. a PDF open in a preview tab) raises before anything is deleted, rather than
    partway through -- leaving the previous set of artifacts fully intact.

    Args:
        report_path: The just-written jupytext `.py` report.

    Raises:
        ArtifactWriteError: When a sibling artifact cannot be removed.
    """
    lock_reason = check_artifacts_writable(report_path)
    if lock_reason is not None:
        raise ArtifactWriteError(lock_reason)

    for suffix_path in (
        report_path.with_suffix(".ipynb"),
        report_path.with_name(f"{report_path.stem}.failed.ipynb"),
        report_path.with_suffix(".html"),
        report_path.with_suffix(".pdf"),
    ):
        try:
            suffix_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ArtifactWriteError(
                _sanitize_reason(
                    f"cannot remove existing '{suffix_path}' -- it is likely open in another "
                    f"program (e.g. a preview tab or editor); close it and regenerate the report: {exc}"
                )
            ) from exc


def _render_and_build_artifacts(
    report_path: Path,
    run_table: RunTable,
    *,
    execute: bool,
    render_pdf: bool,
    progress: Callable[[str], None] | None,
) -> ReportArtifacts:
    """Execute (or skip executing) a generated `.py` report and build its `ReportArtifacts`.

    Shared by the individual and comparative generation paths so the `execute`/`render_pdf`
    decision is made in exactly one place (T101). When `execute` is `False`, `render_report`
    is never invoked -- no kernel starts, no `.ipynb`/`.html` is produced -- and the artifacts
    fall back to the `.py` file itself for both `notebook` and `html`, matching today's
    scaffold-only behaviour (`--no-execute` implies `--no-pdf`, contracts C1).

    Args:
        report_path: The just-written jupytext `.py` report.
        run_table: The `RunTable` already built for this group, for the loaded/skipped counts.
        execute: Whether the notebook should be executed.
        render_pdf: Whether a PDF should be rendered; ignored when `execute` is `False`.
        progress: Optional sink for human-readable progress lines.

    Returns:
        The `ReportArtifacts` describing every artifact produced for this report.
    """
    if not execute:
        _clear_stale_render_artifacts(report_path)
        return ReportArtifacts(
            source=report_path,
            notebook=report_path,
            html=report_path,
            pdf=None,
            pdf_skip_reason=_NOT_EXECUTED_REASON,
            runs_loaded=run_table.runs_loaded,
            runs_skipped=run_table.runs_skipped,
        )

    if progress is not None:
        progress(f"Executing report: {report_path}")

    render_result = render_report(report_path, render_pdf=render_pdf, progress=progress)

    if progress is not None:
        if render_result.pdf is not None:
            progress(f"PDF rendered: {render_result.pdf}")
        else:
            progress(f"PDF skipped: {render_result.pdf_skip_reason}")

    return ReportArtifacts(
        source=report_path,
        notebook=render_result.notebook,
        html=render_result.html,
        pdf=render_result.pdf,
        pdf_skip_reason=render_result.pdf_skip_reason,
        runs_loaded=run_table.runs_loaded,
        runs_skipped=run_table.runs_skipped,
    )


def generate_individual_report(
    experiment_path: Path,
    output_path: Path | None = None,
    *,
    execute: bool = True,
    render_pdf: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportArtifacts:
    """
    Generate an individual Jupyter notebook report for a single experiment.

    The generated report is a Python file (.py) in Jupytext format with cell markers (# %%).
    It can be opened directly as a Jupyter notebook using Jupytext or any IDE that supports it.

    Args:
        experiment_path: Path to the experiment directory containing JSON files
        output_path: Path where to save the generated report (default: experiment_path/report.py)
        execute: Whether the notebook should be executed via `render_report` (jupytext ->
            execute -> HTML -> PDF, `reports/render.py`). When `False`, only the `.py` scaffold
            is written and `render_pdf` is ignored.
        render_pdf: Whether a PDF should be rendered alongside the notebook; ignored when
            `execute` is `False`. A PDF failure never raises -- it is reported as
            `pdf_skip_reason` (FR-026).
        progress: Optional sink for human-readable progress lines.

    Returns:
        ReportArtifacts describing the generated notebook and why the PDF was skipped.

    Raises:
        ValueError: If experiment data cannot be loaded from experiment_path.
        ArtifactWriteError: If a render artifact at `output_path` (or one of its
            `.ipynb`/`.html`/`.pdf` siblings) cannot be replaced or removed -- most commonly
            because it is open in another program. Checked before anything is written, so a
            failure here leaves any previously generated artifacts untouched.
    """
    if output_path is None:
        output_path = experiment_path / "report.py"

    lock_reason = check_artifacts_writable(output_path)
    if lock_reason is not None:
        raise ArtifactWriteError(lock_reason)

    if progress is not None:
        progress(f"Generating individual report for {experiment_path}")

    # build_run_table is the single loading routine (FR-004, FR-009); it also validates the run
    # is readable without ever opening model.json (FR-007, SC-010).
    run_table = build_run_table(experiment_path)
    if run_table.runs_loaded == 0:
        reason = run_table.skipped[0].reason if run_table.skipped else "no run could be loaded"
        raise ValueError(f"Failed to load experiment data from {experiment_path}: {reason}")

    # Create template environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.py.j2")

    # The environment is named in prose with values baked in as literal text at generation
    # time (FR-001, FR-002, FR-003) — the sole record already loaded above carries them.
    record = run_table.records[0]
    env_summary = format_environment_summary(record.env_id, record.env_kwargs, record.max_episode_steps)

    # Prepare template context. The generation-time absolute path is baked in as a last-resort
    # fallback for `_locate_report_dir()` (research R9); forward-slashed so it is a valid raw
    # Python string literal on Windows.
    context = {
        "experiment_path": str(experiment_path),
        "experiment_path_posix": str(experiment_path.resolve()).replace("\\", "/"),
        "environment_file_name": environment_file_name,
        "model_file_name": model_file_name,
        "run_info_file_name": run_info_file_name,
        "tag_remove_cell": TAG_REMOVE_CELL,
        "tag_remove_input": TAG_REMOVE_INPUT,
        "tag_remove_output": TAG_REMOVE_OUTPUT,
        "env_id": record.env_id,
        "env_summary": env_summary,
    }

    # Generate report
    report_content = template.render(**context)

    # Write report file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Individual report generated: {output_path}")
    if progress is not None:
        progress(f"Report written: {output_path}")

    return _render_and_build_artifacts(
        output_path, run_table, execute=execute, render_pdf=render_pdf, progress=progress
    )


def generate_report(
    experiment_path: Path,
    output_path: Path | None = None,
    *,
    execute: bool = True,
    render_pdf: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportBundle:
    """
    Generate a Jupyter notebook report for an experiment or multiple experiments.

    This function automatically detects the type of report to generate:
    - If the given directory contains a valid experiment structure (environment.json,
      model.json, run_info.json), generates an individual report.
    - If the given directory contains subdirectories with valid experiment structures
      (searched up to MAX_DEPTH levels deep), generates a comparative report per
      environment-settings group.

    Args:
        experiment_path: Path to the experiment directory or parent directory
        output_path: Path where to save the generated report. Only supported for an individual
            report (default: experiment_path/report.py); rejected with a `ValueError` for a
            comparative run, since a parent directory can expand into several report groups,
            each of which writes its own `comparative_report.py` beside its runs.
        execute: Whether generated notebooks should be executed via `render_report` (jupytext
            -> execute -> HTML -> PDF, `reports/render.py`). When `False`, only the `.py`
            scaffold is written per group and `render_pdf` is ignored.
        render_pdf: Whether a PDF should be rendered alongside each notebook; ignored when
            `execute` is `False`. A PDF failure never raises -- it is reported as
            `pdf_skip_reason` (FR-026).
        progress: Optional sink for human-readable progress lines, so a caller can satisfy a
            progress cadence without this module importing a CLI framework.

    Returns:
        ReportBundle listing every report group actually rendered and every candidate group
        that was skipped, with a reason. A group whose artifacts could not be written (a
        locked PDF, most commonly) is recorded here as a skipped group -- it never aborts the
        remaining groups (Defect 1's robustness contract).

    Raises:
        ValueError: If no valid experiment directories are found, no qualifying report group
            could be generated (fewer than 2 loadable runs in every candidate group), or
            `output_path` was given for a comparative (multi-group) run.
        FileNotFoundError: If the experiment_path doesn't exist.
        ArtifactWriteError: If this is a single-group (individual) run and its artifacts cannot
            be written, or if this is a multi-group run and *every* candidate group failed to
            write its artifacts (an OSError-shaped problem, never "invalid experiment data" --
            Defect 3). A multi-group run where at least one group succeeds never raises this;
            the failure is reported via `ReportBundle.skipped_groups` instead.
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Directory not found: {experiment_path}")

    if not experiment_path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_path}")

    # Check if the root directory itself is a valid experiment directory
    if is_valid_experiment_directory(experiment_path):
        logger.info("Root directory is a valid experiment directory, generating individual report")
        artifacts = generate_individual_report(
            experiment_path, output_path, execute=execute, render_pdf=render_pdf, progress=progress
        )
        return ReportBundle(reports=[artifacts])

    if output_path is not None:
        raise ValueError(
            "output_path is not supported when experiment_path expands into a comparative report: "
            "each environment-settings group writes its own comparative_report.py beside its runs"
        )

    # Search for experiment directories recursively
    logger.info(f"Searching for experiment directories in {experiment_path} (max depth: {MAX_DEPTH})")
    experiment_dirs = find_experiment_directories(experiment_path, max_depth=MAX_DEPTH)

    if not experiment_dirs:
        logger.warning(
            f"No valid experiment directories found in {experiment_path} (searched up to {MAX_DEPTH} levels deep)"
        )
        raise ValueError(
            f"No valid experiment directories found in {experiment_path}. "
            f"A valid experiment directory must contain: {environment_file_name}, "
            f"{model_file_name}, and {run_info_file_name}"
        )

    logger.info(f"Found {len(experiment_dirs)} experiment directory(ies), generating comparative reports")

    # Group experiments by environment + environment parametrization
    # Structure: root/env/env_params/model/model_params/
    # We group by env/env_params (2 levels up from experiment directory)
    experiment_groups: dict[Path, list[Path]] = {}
    for exp_dir in experiment_dirs:
        # Go up 2 levels to get to the environment parametrization level
        # exp_dir is at model_params level, so:
        # parent is model, parent.parent is env_params
        if len(exp_dir.parts) >= 2:
            env_params_dir = exp_dir.parent.parent
            if env_params_dir not in experiment_groups:
                experiment_groups[env_params_dir] = []
            experiment_groups[env_params_dir].append(exp_dir)
        else:
            logger.warning(f"Cannot determine environment parametrization for {exp_dir}, skipping")

    if not experiment_groups:
        raise ValueError("Could not group experiments by environment parametrization")

    logger.info(f"Found {len(experiment_groups)} environment parametrization group(s)")

    # Generate a comparative report for each group
    generated_reports: list[ReportArtifacts] = []
    skipped_groups: list[SkippedGroup] = []
    # Tracked separately from `skipped_groups` so that, if every group fails, the exception
    # raised below can distinguish "every group had too few runs" (ValueError, invalid input)
    # from "every group's output was locked" (an OSError -- an output problem, not an invalid
    # experiment) rather than collapsing both into the same generic message (Defect 3).
    write_failed_groups: list[SkippedGroup] = []
    for env_params_dir, group_experiment_dirs in experiment_groups.items():
        if len(group_experiment_dirs) < 2:
            reason = f"only {len(group_experiment_dirs)} run(s), need at least 2 for comparison"
            logger.info(f"Skipping {env_params_dir}: {reason}")
            skipped_groups.append(SkippedGroup(path=env_params_dir, reason=reason))
            continue

        logger.info(f"Generating comparative report for {len(group_experiment_dirs)} runs in {env_params_dir}")
        if progress is not None:
            progress(f"Generating comparative report for {len(group_experiment_dirs)} runs in {env_params_dir}")

        # build_run_table is the single loading routine (FR-004, FR-009), reused unchanged by
        # the generated notebook at execution time.
        run_table = build_run_table(env_params_dir)
        if run_table.runs_loaded < 2:
            reason = f"only {run_table.runs_loaded} run(s) loaded successfully, need at least 2 for comparison"
            logger.warning(f"{reason} in {env_params_dir}, skipping")
            skipped_groups.append(SkippedGroup(path=env_params_dir, reason=reason))
            continue

        report_path = env_params_dir / "comparative_report.py"

        # Verify every render artifact for this group can be replaced/removed BEFORE writing
        # anything -- including the manifest and the `.py` itself (FR-028, "robustness in
        # report generation"). A group whose PDF is locked open in another program (a preview
        # tab, an editor) must not abort the whole `hercule report` invocation: it is recorded
        # as a skipped group with a reason naming the locked file, and the remaining groups are
        # still generated.
        lock_reason = check_artifacts_writable(report_path)
        if lock_reason is not None:
            reason = f"cannot write report artifacts: {lock_reason}"
            logger.warning(f"{reason} in {env_params_dir}, skipping")
            skipped_entry = SkippedGroup(path=env_params_dir, reason=reason)
            skipped_groups.append(skipped_entry)
            write_failed_groups.append(skipped_entry)
            continue

        # Write the manifest the generated notebook's directory search verifies against
        # (research R9, contracts C5) — the environment-settings level has no naturally
        # occurring anchor file the way an individual report has environment.json.
        manifest = ReportManifest(
            root=env_params_dir,
            env_id=run_table.env_id or "",
            env_kwargs=run_table.env_kwargs or {},
            max_episode_steps=run_table.max_episode_steps,
            model_names=run_table.model_names,
            runs_loaded=run_table.runs_loaded,
            runs_skipped=run_table.runs_skipped,
        )
        manifest_path = env_params_dir / "report_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        # Create template environment
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("comparative_report_template.py.j2")

        # The environment is named in prose with values baked in as literal text at generation
        # time (FR-001, FR-002) — the run table already built for the manifest above carries
        # them, so no extra disk read is needed.
        env_summary = format_environment_summary(
            run_table.env_id or "", run_table.env_kwargs or {}, run_table.max_episode_steps
        )

        # Prepare template context. The generation-time absolute path is baked in as a
        # last-resort fallback for `_locate_report_dir()` (research R9); forward-slashed so it
        # is a valid raw Python string literal on Windows.
        context = {
            "root_path": str(env_params_dir),
            "root_path_posix": str(env_params_dir.resolve()).replace("\\", "/"),
            "environment_file_name": environment_file_name,
            "model_file_name": model_file_name,
            "run_info_file_name": run_info_file_name,
            "tag_remove_cell": TAG_REMOVE_CELL,
            "tag_remove_input": TAG_REMOVE_INPUT,
            "tag_remove_output": TAG_REMOVE_OUTPUT,
            "env_id": run_table.env_id or "",
            "env_summary": env_summary,
        }

        # Generate report
        report_content = template.render(**context)

        # Write report file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Comparative report generated: {report_path}")
        if progress is not None:
            progress(f"Report written: {report_path}")

        try:
            artifacts = _render_and_build_artifacts(
                report_path, run_table, execute=execute, render_pdf=render_pdf, progress=progress
            )
        except OSError as exc:
            # A per-group write failure (e.g. a race between the preflight check above and this
            # render, or an execution that outlives a lock appearing mid-run) must not abort the
            # whole invocation: record it and keep generating the remaining groups (FR-028's
            # robustness contract -- one locked file is not "invalid experiment data").
            reason = f"cannot write report artifacts: {_sanitize_reason(str(exc))}"
            logger.warning(f"{reason} in {env_params_dir}")
            skipped_entry = SkippedGroup(path=env_params_dir, reason=reason)
            skipped_groups.append(skipped_entry)
            write_failed_groups.append(skipped_entry)
            continue

        generated_reports.append(artifacts)

    if not generated_reports:
        if write_failed_groups:
            # Every candidate group failed to write its output -- an OSError-shaped problem
            # (Defect 3), never "invalid experiment data": the data loaded fine, only the
            # destination could not be written (most commonly a locked PDF/notebook/HTML).
            raise ArtifactWriteError("; ".join(f"{entry.path}: {entry.reason}" for entry in write_failed_groups))
        raise ValueError(
            "No comparative reports could be generated. "
            "Ensure there are at least 2 experiments per environment parametrization group."
        )

    return ReportBundle(reports=generated_reports, skipped_groups=skipped_groups)


__all__ = [
    "MAX_DEPTH",
    "TAG_REMOVE_CELL",
    "TAG_REMOVE_INPUT",
    "TAG_REMOVE_OUTPUT",
    "TOP_TABLE_COLUMN_LABELS",
    "ArtifactWriteError",
    "ConstantMetric",
    "HyperparameterEtaSquared",
    "HyperparameterGridCardinality",
    "HyperparameterGridDimension",
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
    "MetricRedundancy",
    "PairwiseInteraction",
    "RankShift",
    "RankedRun",
    "RankingMetric",
    "RenderResult",
    "ReplicationStatus",
    "ReportArtifacts",
    "ReportBundle",
    "ReportManifest",
    "RunRecord",
    "RunTable",
    "SelectedSeries",
    "SeriesBucket",
    "SeriesSelection",
    "SkippedGroup",
    "SkippedRun",
    "TopDecileComparisonResult",
    "TopDecileComparisonUnavailable",
    "VarianceDecompositionEntry",
    "VarianceDecompositionResult",
    "VarianceDecompositionUnavailable",
    "build_run_table",
    "check_artifacts_writable",
    "detect_constant_metrics",
    "detect_redundant_metrics",
    "find_experiment_directories",
    "format_environment_summary",
    "format_relative_run_path",
    "format_series_labels",
    "format_top_table_hyperparameter_cells",
    "format_varying_hyperparameters",
    "generate_individual_report",
    "generate_report",
    "hyperparameter_grid_cardinality",
    "hyperparameter_importance",
    "hyperparameter_main_effects",
    "interaction_grid",
    "interaction_ranking",
    "is_valid_experiment_directory",
    "max_performance_is_saturated",
    "order_varying_hyperparameters_by_importance",
    "rank_runs_by_performance",
    "render_report",
    "replication_status",
    "select_series",
    "select_top_table_metric_columns",
    "top_decile_comparison",
    "variance_decomposition",
]
