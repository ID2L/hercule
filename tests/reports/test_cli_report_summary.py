"""CLI tests for the `hercule report` summary block (FR-027).

`ReportArtifacts` gained a `source` field pointing at the generated jupytext `.py` -- the
durable, re-runnable artifact -- because the previous summary listed only the executed
`.ipynb`/`.html`/`.pdf` and silently dropped the `.py` from the "here is what I produced"
statement. These tests run with `--no-execute` (no kernel, no browser) so they stay fast like
the rest of `tests/reports`; the executed (`execute=True`) path is covered by the `slow`-marked
tests in `test_render.py`.
"""

from hercule.cli.main import cli

from .conftest import RunTreeBuilder


def test_report_summary_lists_the_py_source_first(run_tree_builder: RunTreeBuilder, runner) -> None:
    """The final summary must list the `.py` source, and list it before the notebook (FR-027)."""
    for i in range(2):
        run_tree_builder.add_run(model_name="simple_q_learning", model_signature=f"sig_{i}")

    group_dir = run_tree_builder.root / "FrozenLake-v1" / "env_sig"
    result = runner.invoke(cli, ["report", str(group_dir), "--no-execute"])

    assert result.exit_code == 0, result.output

    report_path = group_dir / "comparative_report.py"
    assert report_path.exists()
    assert str(report_path) in result.output

    lines = result.output.splitlines()
    summary_index = next(i for i, line in enumerate(lines) if "report(s) generated" in line)
    source_index = next(i for i, line in enumerate(lines) if str(report_path) in line and i > summary_index)

    # The .py source is the primary artifact and must come first among the per-group listing.
    assert source_index == summary_index + 1


def test_no_execute_summary_does_not_duplicate_the_source_path(run_tree_builder: RunTreeBuilder, runner) -> None:
    """With --no-execute, notebook/html fall back to the source itself and must not be
    printed twice in the final per-group listing (the .py is legitimately echoed once more,
    earlier, by the "Report written: ..." progress line -- only the summary block's own
    listing must be deduplicated).
    """
    for i in range(2):
        run_tree_builder.add_run(model_name="simple_q_learning", model_signature=f"sig_{i}")

    group_dir = run_tree_builder.root / "FrozenLake-v1" / "env_sig"
    result = runner.invoke(cli, ["report", str(group_dir), "--no-execute"])

    assert result.exit_code == 0, result.output

    report_path = group_dir / "comparative_report.py"
    lines = result.output.splitlines()
    summary_index = next(i for i, line in enumerate(lines) if "report(s) generated" in line)
    listing_lines = lines[summary_index + 1 :]
    occurrences_in_listing = sum(1 for line in listing_lines if str(report_path) in line)
    assert occurrences_in_listing == 1

    assert "PDF skipped" in result.output
    assert "not executed" in result.output.lower() or "--no-execute" in result.output


def test_individual_report_summary_lists_the_py_source(run_tree_builder: RunTreeBuilder, runner) -> None:
    """The same first-listed-.py-source contract holds for an individual (single-run) report."""
    run_dir = run_tree_builder.add_run(model_name="simple_q_learning", model_signature="sig")

    result = runner.invoke(cli, ["report", str(run_dir), "--no-execute"])

    assert result.exit_code == 0, result.output

    report_path = run_dir / "report.py"
    assert report_path.exists()
    assert str(report_path) in result.output
