"""Robustness tests for report generation under a locked output artifact (Defects 1-3).

Reproduces, without depending on any real OS-level file lock, the incident where a PDF held
open by a preview tab (e.g. a VSCode tab, `WinError 32` on Windows) used to: (1) abort the
*entire* `hercule report` invocation, skipping sibling groups that had nothing wrong with
them; (2) leave a freshly regenerated `.py` beside the stale, un-updated `.pdf` with no
`.ipynb`/`.html` at all; and (3) surface as "Invalid experiment data" even though the
experiment data loaded perfectly fine -- only an *output* file could not be written.

The lock is simulated by monkeypatching the exact primitives the pipeline calls to detect and
perform artifact replacement (`Path.rename`, `Path.unlink`, and the shared
`check_artifacts_writable` preflight), which is deterministic and portable, unlike holding a
real exclusive file handle open on Windows.
"""

from pathlib import Path

import pytest

import hercule.reports as reports_module
from hercule.cli.main import cli
from hercule.controller import generate_experiment_report
from hercule.reports import render as render_module

from .conftest import RunTreeBuilder


# ---------------------------------------------------------------------------
# render.py level: the writability preflight itself, and render_report's use of it.
# ---------------------------------------------------------------------------


def test_check_artifacts_writable_detects_locked_pdf_and_touches_nothing(tmp_path: Path, monkeypatch) -> None:
    """A rename failure on the stale `.pdf` is reported, and the file is left byte-identical.

    `_probe_removable` renames a candidate aside and immediately back -- a lock that would
    break the real replace also breaks this probe, but nothing is deleted or truncated by the
    probe itself (defect 2's "abort before deleting anything" alternative).
    """
    py_path = tmp_path / "comparative_report.py"
    py_path.write_text("# scaffold", encoding="utf-8")
    pdf_path = py_path.with_suffix(".pdf")
    pdf_path.write_bytes(b"%PDF-1.4 two-days-old-content")
    original_bytes = pdf_path.read_bytes()
    original_mtime = pdf_path.stat().st_mtime

    original_rename = Path.rename

    def fake_rename(self: Path, target):
        if self == pdf_path:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fake_rename)

    reason = render_module.check_artifacts_writable(py_path)

    assert reason is not None
    assert "comparative_report.pdf" in reason or str(pdf_path) in reason
    # Non-destructive: the file was never actually removed or replaced by the probe.
    assert pdf_path.exists()
    assert pdf_path.read_bytes() == original_bytes
    assert pdf_path.stat().st_mtime == original_mtime


def test_check_artifacts_writable_passes_when_nothing_is_locked(tmp_path: Path) -> None:
    """No siblings on disk at all, or an unlocked sibling: no reason, nothing touched."""
    py_path = tmp_path / "report.py"

    assert render_module.check_artifacts_writable(py_path) is None

    pdf_path = py_path.with_suffix(".pdf")
    pdf_path.write_bytes(b"unlocked")
    assert render_module.check_artifacts_writable(py_path) is None
    assert pdf_path.read_bytes() == b"unlocked"


def test_render_report_raises_artifact_write_error_and_writes_nothing_when_locked(tmp_path: Path, monkeypatch) -> None:
    """`render_report` must raise before touching disk when the preflight finds a lock.

    No kernel needs to actually run for this: the preflight is the very first thing
    `render_report` does, so stubbing it out keeps this test fast (not `slow`-marked).
    """
    py_path = tmp_path / "report.py"
    py_path.write_text("# %%\nprint('hello')\n", encoding="utf-8")
    monkeypatch.setattr(render_module, "check_artifacts_writable", lambda p: "report.pdf is locked")

    with pytest.raises(render_module.ArtifactWriteError, match="locked"):
        render_module.render_report(py_path, execute_timeout=10)

    assert not (tmp_path / "report.ipynb").exists()
    assert not (tmp_path / "report.html").exists()
    assert not (tmp_path / "report.pdf").exists()


def test_render_report_raises_when_stale_cleanup_unlink_fails(tmp_path: Path, monkeypatch) -> None:
    """A failure at the actual `unlink` step (not just the rename-based preflight) still
    raises `ArtifactWriteError` rather than letting a raw `OSError` escape unclassified."""
    py_path = tmp_path / "report.py"
    py_path.write_text("# %%\nprint('hello')\n", encoding="utf-8")
    stale_pdf = py_path.with_suffix(".pdf")
    stale_pdf.write_bytes(b"stale")

    original_unlink = Path.unlink

    def fake_unlink(self: Path, missing_ok: bool = False):
        if self == stale_pdf:
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    with pytest.raises(render_module.ArtifactWriteError):
        render_module.render_report(py_path, execute_timeout=10)


# ---------------------------------------------------------------------------
# reports/__init__.py level: per-group isolation (Defect 1) and the fresh/stale mix (Defect 2).
# ---------------------------------------------------------------------------


def _fake_check_locks_group(marker: str):
    """Build a `check_artifacts_writable` stub that only locks the group named `marker`.

    Matches on the report's *parent directory name* (the env-signature segment, e.g.
    "locked" or "ok"), not a plain substring of the full path -- pytest's `tmp_path` is
    itself derived from the test's own name, which can innocently contain the marker text.
    """

    def fake_check(py_path: Path) -> str | None:
        if py_path.parent.name == marker:
            return f"'{py_path.with_suffix('.pdf')}' is open in another program (a preview tab); close it and retry"
        return None

    return fake_check


def test_locked_group_is_skipped_but_sibling_group_still_generated(
    run_tree_builder: RunTreeBuilder, monkeypatch
) -> None:
    """Defect 1: one group's locked output must not abort the other, unrelated group."""
    for i in range(2):
        run_tree_builder.add_run(env_signature="locked", model_name="simple_q_learning", model_signature=f"sig_{i}")
    for i in range(2):
        run_tree_builder.add_run(env_signature="ok", model_name="simple_q_learning", model_signature=f"sig_{i}")

    monkeypatch.setattr(reports_module, "check_artifacts_writable", _fake_check_locks_group("locked"))

    bundle = reports_module.generate_report(run_tree_builder.root / "FrozenLake-v1", execute=False, render_pdf=False)

    assert bundle.report_count == 1
    assert bundle.reports[0].source == run_tree_builder.root / "FrozenLake-v1" / "ok" / "comparative_report.py"

    assert len(bundle.skipped_groups) == 1
    skipped = bundle.skipped_groups[0]
    assert "locked" in str(skipped.path)
    assert "open in another program" in skipped.reason

    # The locked group's `.py`/manifest were never written -- the preflight runs before any
    # write for that group (Defect 2: never a fresh scaffold beside a stale render artifact).
    locked_dir = run_tree_builder.root / "FrozenLake-v1" / "locked"
    assert not (locked_dir / "comparative_report.py").exists()
    assert not (locked_dir / "report_manifest.json").exists()


def test_locked_group_leaves_previous_artifacts_untouched_not_a_fresh_stale_mix(
    run_tree_builder: RunTreeBuilder, monkeypatch
) -> None:
    """Defect 2's core invariant: a locked group's artifacts stay bit-identical to the
    previous successful run -- never a freshly regenerated `.py` beside a stale `.pdf`."""
    for i in range(2):
        run_tree_builder.add_run(env_signature="locked", model_name="simple_q_learning", model_signature=f"sig_{i}")
    for i in range(2):
        run_tree_builder.add_run(env_signature="ok", model_name="simple_q_learning", model_signature=f"sig_{i}")

    locked_dir = run_tree_builder.root / "FrozenLake-v1" / "locked"
    stale_py = locked_dir / "comparative_report.py"
    stale_pdf = locked_dir / "comparative_report.pdf"
    stale_py.write_text("# stale scaffold from a previous successful run\n", encoding="utf-8")
    stale_pdf.write_bytes(b"%PDF-1.4 two-days-old-content")
    stale_py_snapshot = (stale_py.read_text(encoding="utf-8"), stale_py.stat().st_mtime)
    stale_pdf_snapshot = (stale_pdf.read_bytes(), stale_pdf.stat().st_mtime)

    monkeypatch.setattr(reports_module, "check_artifacts_writable", _fake_check_locks_group("locked"))

    bundle = reports_module.generate_report(run_tree_builder.root / "FrozenLake-v1", execute=False, render_pdf=False)

    assert bundle.report_count == 1
    assert len(bundle.skipped_groups) == 1

    # Neither the stale .py NOR the stale .pdf were touched: content and mtime unchanged.
    assert (stale_py.read_text(encoding="utf-8"), stale_py.stat().st_mtime) == stale_py_snapshot
    assert (stale_pdf.read_bytes(), stale_pdf.stat().st_mtime) == stale_pdf_snapshot


def test_all_groups_locked_raises_os_error_not_value_error(run_tree_builder: RunTreeBuilder, monkeypatch) -> None:
    """Defect 3 at the `generate_report` boundary: when *every* group fails to write, the
    exception must be an `OSError` (an output problem), never a `ValueError` (an input
    problem) -- the two are handled very differently by the CLI."""
    for i in range(2):
        run_tree_builder.add_run(env_signature="locked", model_name="simple_q_learning", model_signature=f"sig_{i}")

    monkeypatch.setattr(reports_module, "check_artifacts_writable", _fake_check_locks_group("locked"))

    with pytest.raises(OSError) as exc_info:
        reports_module.generate_report(run_tree_builder.root / "FrozenLake-v1", execute=False, render_pdf=False)
    assert not isinstance(exc_info.value, ValueError)


def test_controller_reports_locked_output_as_os_error(run_tree_builder: RunTreeBuilder, monkeypatch) -> None:
    """Defect 3 at the controller boundary: a locked individual report raises `OSError` with
    an actionable message, never gets rewrapped into "Failed to generate report" `ValueError`."""
    run_dir = run_tree_builder.add_run(model_name="simple_q_learning", model_signature="sig")
    monkeypatch.setattr(reports_module, "check_artifacts_writable", lambda p: "report.pdf is open in another program")

    with pytest.raises(OSError) as exc_info:
        generate_experiment_report(run_dir, execute=False)
    assert not isinstance(exc_info.value, ValueError)
    assert "open in another program" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CLI level: exit codes and the corrected error message (all three defects, end to end).
# ---------------------------------------------------------------------------


def test_cli_partial_group_failure_still_exits_zero_and_reports_the_reason(
    run_tree_builder: RunTreeBuilder, runner, monkeypatch
) -> None:
    """Defect 1 + Defect 2 through the CLI: one locked group is reported, its sibling is
    still generated, and the process exits 0 because at least one group succeeded."""
    for i in range(2):
        run_tree_builder.add_run(env_signature="locked", model_name="simple_q_learning", model_signature=f"sig_{i}")
    for i in range(2):
        run_tree_builder.add_run(env_signature="ok", model_name="simple_q_learning", model_signature=f"sig_{i}")

    monkeypatch.setattr(reports_module, "check_artifacts_writable", _fake_check_locks_group("locked"))

    result = runner.invoke(cli, ["report", str(run_tree_builder.root / "FrozenLake-v1"), "--no-execute"])

    assert result.exit_code == 0, result.output
    assert "1 report(s) generated" in result.output
    assert "skipped" in result.output.lower()
    assert "open in another program" in result.output


def test_cli_all_groups_locked_exits_nonzero(run_tree_builder: RunTreeBuilder, runner, monkeypatch) -> None:
    """When no group succeeds at all, the CLI must exit non-zero."""
    for i in range(2):
        run_tree_builder.add_run(env_signature="locked", model_name="simple_q_learning", model_signature=f"sig_{i}")

    monkeypatch.setattr(reports_module, "check_artifacts_writable", _fake_check_locks_group("locked"))

    result = runner.invoke(cli, ["report", str(run_tree_builder.root / "FrozenLake-v1"), "--no-execute"])

    assert result.exit_code != 0


def test_cli_locked_output_message_names_the_real_problem_not_invalid_data(
    run_tree_builder: RunTreeBuilder, runner, monkeypatch
) -> None:
    """Defect 3 through the CLI: a locked output artifact must be reported as a write
    problem ("Cannot write report output"), never as "Invalid experiment data"."""
    run_dir = run_tree_builder.add_run(model_name="simple_q_learning", model_signature="sig")
    monkeypatch.setattr(
        reports_module,
        "check_artifacts_writable",
        lambda p: "report.pdf is open in another program (e.g. a preview tab); close it and retry",
    )

    result = runner.invoke(cli, ["report", str(run_dir), "--no-execute"])

    assert result.exit_code != 0
    assert "Cannot write report output" in result.output
    assert "Invalid experiment data" not in result.output
    assert "open in another program" in result.output
