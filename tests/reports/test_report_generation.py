"""Integration tests for `hercule.reports.generate_report` (User Story 2).

These exercise the generator end to end over a synthetic run tree: the generated document's
size must not scale with run count (SC-001), the emitted Python must be syntactically valid,
and `output_path` must be honoured for an individual report but rejected — not silently
discarded — for a comparative (multi-group) run (contracts C1, T034).
"""

import ast
from pathlib import Path

import matplotlib
import pytest


matplotlib.use("Agg")  # must precede any `import matplotlib.pyplot` — including the executed report's

import matplotlib.pyplot as plt  # noqa: E402

from hercule.reports import generate_report  # noqa: E402

from .conftest import RunTreeBuilder  # noqa: E402


def _generate_comparative(builder: RunTreeBuilder, run_count: int) -> Path:
    for i in range(run_count):
        builder.add_run(
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
            hyperparameters={"learning_rate": 0.0001 * (i + 1), "seed": i},
        )
    bundle = generate_report(builder.root / "FrozenLake-v1" / "env_sig", execute=False, render_pdf=False)
    assert bundle.report_count == 1
    return bundle.reports[0].notebook


def test_comparative_report_line_count_is_independent_of_run_count(tmp_path_factory) -> None:
    """A 2-run group and a 50-run group must produce a document within 10% line count (SC-001).

    The old per-run `{% for %}` loop emitted ~35 lines per run; the rewritten template emits a
    single loading loop whose size is fixed regardless of how many runs it walks.
    """
    small_root = tmp_path_factory.mktemp("small")
    large_root = tmp_path_factory.mktemp("large")

    small_report = _generate_comparative(RunTreeBuilder(small_root), 2)
    large_report = _generate_comparative(RunTreeBuilder(large_root), 50)

    small_lines = len(small_report.read_text(encoding="utf-8").splitlines())
    large_lines = len(large_report.read_text(encoding="utf-8").splitlines())

    assert small_lines > 0
    assert abs(large_lines - small_lines) / small_lines < 0.10, (
        f"expected line counts within 10%: {small_lines} (2 runs) vs {large_lines} (50 runs)"
    )


def test_comparative_report_is_syntactically_valid_python(run_tree_builder: RunTreeBuilder) -> None:
    """The generated comparative document must parse as valid Python (no bare backslash paths)."""
    report_path = _generate_comparative(run_tree_builder, 3)
    ast.parse(report_path.read_text(encoding="utf-8"))


def test_individual_report_is_syntactically_valid_python(run_tree_builder: RunTreeBuilder) -> None:
    """The generated individual document must parse as valid Python."""
    run_dir = run_tree_builder.add_run(model_name="simple_q_learning", model_signature="sig")
    bundle = generate_report(run_dir, execute=False, render_pdf=False)
    assert bundle.report_count == 1
    report_path = bundle.reports[0].notebook
    ast.parse(report_path.read_text(encoding="utf-8"))


def test_individual_report_honours_output_path(run_tree_builder: RunTreeBuilder) -> None:
    """output_path is honoured for an individual (single-run) report."""
    run_dir = run_tree_builder.add_run(model_name="simple_q_learning", model_signature="sig")
    custom_path = run_dir / "custom_report.py"

    bundle = generate_report(run_dir, custom_path, execute=False, render_pdf=False)

    assert bundle.reports[0].notebook == custom_path
    assert bundle.reports[0].source == custom_path
    assert custom_path.exists()


def test_report_artifacts_source_is_the_generated_py_file(run_tree_builder: RunTreeBuilder) -> None:
    """`ReportArtifacts.source` always points at the generated jupytext `.py` (FR-027).

    With `execute=False`, `notebook` and `html` fall back to the `.py` scaffold itself
    (documented on `ReportArtifacts`), so all three fields coincide -- but `source` is the
    field callers should rely on to mean "the durable, re-runnable artifact" unambiguously.
    """
    for i in range(2):
        run_tree_builder.add_run(model_name="simple_q_learning", model_signature=f"sig_{i}")

    bundle = generate_report(run_tree_builder.root / "FrozenLake-v1" / "env_sig", execute=False, render_pdf=False)

    artifact = bundle.reports[0]
    expected_source = run_tree_builder.root / "FrozenLake-v1" / "env_sig" / "comparative_report.py"
    assert artifact.source == expected_source
    assert artifact.source.exists()
    assert artifact.source.suffix == ".py"
    # --no-execute semantics: no kernel ran, so notebook/html fall back to the source itself.
    assert artifact.notebook == artifact.source
    assert artifact.html == artifact.source


def test_comparative_report_rejects_output_path(run_tree_builder: RunTreeBuilder) -> None:
    """output_path must be rejected, not silently discarded, for a multi-group comparative run."""
    for i in range(2):
        run_tree_builder.add_run(model_name="simple_q_learning", model_signature=f"sig_{i}")

    with pytest.raises(ValueError, match="output_path"):
        generate_report(run_tree_builder.root, run_tree_builder.root / "wherever.py", execute=False, render_pdf=False)


def test_comparative_report_writes_manifest(run_tree_builder: RunTreeBuilder) -> None:
    """A report_manifest.json is written beside the comparative report (contracts C5)."""
    _generate_comparative(run_tree_builder, 2)

    manifest_path = run_tree_builder.root / "FrozenLake-v1" / "env_sig" / "report_manifest.json"
    assert manifest_path.exists()


def test_comparative_report_names_environment_in_prose(run_tree_builder: RunTreeBuilder) -> None:
    """FR-001/FR-002: the environment id and its settings appear as prose text, not only inside
    a raw config block, and without needing to execute the notebook (the values are baked in
    at generation time from the run table already built for the manifest)."""
    for i in range(3):
        run_tree_builder.add_run(
            env_id="FrozenLake-v1",
            env_kwargs={"map_name": "4x4", "is_slippery": True},
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
        )

    bundle = generate_report(run_tree_builder.root / "FrozenLake-v1" / "env_sig", execute=False, render_pdf=False)
    content = bundle.reports[0].notebook.read_text(encoding="utf-8")

    assert "FrozenLake-v1" in content
    assert "map_name=4x4" in content
    assert "is_slippery=True" in content


def test_comparative_report_states_no_override_explicitly(run_tree_builder: RunTreeBuilder) -> None:
    """FR-002 scenario 2: empty env_kwargs must produce an explicit statement rather than an
    empty structure."""
    for i in range(2):
        run_tree_builder.add_run(
            env_id="CartPole-v1",
            env_kwargs={},
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
        )

    bundle = generate_report(run_tree_builder.root / "CartPole-v1" / "env_sig", execute=False, render_pdf=False)
    content = bundle.reports[0].notebook.read_text(encoding="utf-8")

    assert "no environment-specific setting was overridden" in content


def test_individual_report_names_environment_in_prose(run_tree_builder: RunTreeBuilder) -> None:
    """FR-003: the individual report states the environment identifier and settings in the
    same prose form as the comparative report."""
    run_dir = run_tree_builder.add_run(
        env_id="FrozenLake-v1",
        env_kwargs={"map_name": "8x8"},
        model_name="simple_q_learning",
        model_signature="sig",
    )

    bundle = generate_report(run_dir, execute=False, render_pdf=False)
    content = bundle.reports[0].notebook.read_text(encoding="utf-8")

    assert "FrozenLake-v1" in content
    assert "map_name=8x8" in content


def test_no_chart_draws_more_than_nine_series(run_tree_builder: RunTreeBuilder) -> None:
    """SC-002: no comparative chart draws more than 9 series, even over a group well past the
    cap. The generated document is executed directly (a plain top-to-bottom script; the `# %%`
    markers are comments) so the real `select_series`-driven chart code runs and the resulting
    matplotlib figures can be inspected — this does not depend on the render pipeline
    (`reports/render.py`), which is a later phase."""
    for i in range(20):
        run_tree_builder.add_run(
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
            hyperparameters={"seed": i},
            learning_episode_count=15,
            testing_episode_count=5,
        )
    bundle = generate_report(run_tree_builder.root / "FrozenLake-v1" / "env_sig", execute=False, render_pdf=False)
    report_path = bundle.reports[0].notebook
    source = report_path.read_text(encoding="utf-8")

    plt.close("all")
    exec(compile(source, str(report_path), "exec"), {})

    figures_with_legends = 0
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                figures_with_legends += 1
                assert len(legend.get_texts()) <= 9, (
                    f"chart legend lists {len(legend.get_texts())} series, exceeding the 9-series cap"
                )
    assert figures_with_legends > 0, "expected at least one chart with a legend to check"
    plt.close("all")


def test_never_reads_model_json_contents(run_tree_builder: RunTreeBuilder) -> None:
    """Generation succeeds even when every model.json in the group is corrupt (SC-010)."""
    for i in range(3):
        run_tree_builder.add_run(
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
            corrupt_model=True,
        )

    bundle = generate_report(run_tree_builder.root / "FrozenLake-v1" / "env_sig", execute=False, render_pdf=False)

    assert bundle.reports[0].runs_loaded == 3
    report_content = bundle.reports[0].notebook.read_text(encoding="utf-8")
    assert "q_network_state_dict" not in report_content
    assert "q_table" not in report_content
