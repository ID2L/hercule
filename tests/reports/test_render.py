"""Tests for the render pipeline (User Story 5): tag filtering, PDF success/skip paths.

`render_report` executes a real Jupyter kernel, so most of these tests are unavoidably
slower than the rest of `tests/reports` -- they are marked `slow` accordingly, except the
pure tag-filtering and browser-discovery-stub tests, which need neither a kernel nor a
browser and stay fast.
"""

import importlib.util
from pathlib import Path

import jupytext
import pytest
from nbformat.v4 import new_code_cell, new_notebook, new_output
from traitlets.config import Config

from hercule.reports import (
    TAG_REMOVE_CELL,
    TAG_REMOVE_INPUT,
    TAG_REMOVE_OUTPUT,
    generate_report,
)
from hercule.reports import render as render_module

from .conftest import RunTreeBuilder


_PERCENT_HEADER = (
    "# ---\n"
    "# jupyter:\n"
    "#   jupytext:\n"
    "#     text_representation:\n"
    "#       extension: .py\n"
    "#       format_name: percent\n"
    "#       format_version: '1.3'\n"
    "#       jupytext_version: 1.14.5\n"
    "#   kernelspec:\n"
    "#     display_name: Python 3\n"
    "#     language: python\n"
    "#     name: python3\n"
    "# ---\n\n"
)


def test_percent_tags_round_trip_to_cell_metadata(tmp_path: Path) -> None:
    """A `# %% tags=["remove_input"]` marker parses to `{"tags": ["remove_input"]}` (T079)."""
    source = _PERCENT_HEADER + f'# %% tags=["{TAG_REMOVE_INPUT}"]\nprint("hello")\n'
    py_path = tmp_path / "probe.py"
    py_path.write_text(source, encoding="utf-8")

    nb = jupytext.read(py_path, fmt="py:percent")

    assert nb.cells[-1].metadata.get("tags") == [TAG_REMOVE_INPUT]


def test_percent_remove_cell_tag_round_trips(tmp_path: Path) -> None:
    """The `remove_cell` tag round-trips identically to `remove_input` (T079)."""
    source = _PERCENT_HEADER + f'# %% tags=["{TAG_REMOVE_CELL}"]\nimport os\n'
    py_path = tmp_path / "probe.py"
    py_path.write_text(source, encoding="utf-8")

    nb = jupytext.read(py_path, fmt="py:percent")

    assert nb.cells[-1].metadata.get("tags") == [TAG_REMOVE_CELL]


def _tag_removal_config() -> Config:
    """Build the exact `TagRemovePreprocessor` configuration research R7 verified.

    Constructed independently of `render.py`'s internals so this test exercises the
    documented contract (contracts C4), not an implementation detail.
    """
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_cell_tags = (TAG_REMOVE_CELL,)
    config.TagRemovePreprocessor.remove_input_tags = (TAG_REMOVE_INPUT,)
    config.TagRemovePreprocessor.remove_all_outputs_tags = (TAG_REMOVE_OUTPUT,)
    config.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
    return config


def test_tag_remove_preprocessor_drops_input_but_keeps_output() -> None:
    """`remove_input` drops source, keeps output; `remove_cell` drops both entirely (T080)."""
    from nbconvert import HTMLExporter  # noqa: PLC0415 -- keep the import local to this test

    nb = new_notebook()

    cell_removed = new_code_cell("MARKER_SOURCE_REMOVE_CELL = 1")
    cell_removed.metadata["tags"] = [TAG_REMOVE_CELL]
    cell_removed.outputs = [new_output("stream", name="stdout", text="MARKER_OUTPUT_REMOVE_CELL\n")]

    cell_input_removed = new_code_cell("MARKER_SOURCE_REMOVE_INPUT = 2")
    cell_input_removed.metadata["tags"] = [TAG_REMOVE_INPUT]
    cell_input_removed.outputs = [new_output("stream", name="stdout", text="MARKER_OUTPUT_REMOVE_INPUT\n")]

    cell_untagged = new_code_cell("MARKER_SOURCE_UNTAGGED = 3")
    cell_untagged.outputs = [new_output("stream", name="stdout", text="MARKER_OUTPUT_UNTAGGED\n")]

    nb.cells = [cell_removed, cell_input_removed, cell_untagged]

    body, _resources = HTMLExporter(config=_tag_removal_config()).from_notebook_node(nb)

    # remove_cell: source AND output absent.
    assert "MARKER_SOURCE_REMOVE_CELL" not in body
    assert "MARKER_OUTPUT_REMOVE_CELL" not in body

    # remove_input: source absent, output retained -- exactly FR-025.
    assert "MARKER_SOURCE_REMOVE_INPUT" not in body
    assert "MARKER_OUTPUT_REMOVE_INPUT" in body

    # untagged: fully retained.
    assert "MARKER_SOURCE_UNTAGGED" in body
    assert "MARKER_OUTPUT_UNTAGGED" in body


def test_print_css_wraps_output_pre_without_touching_break_inside_rule() -> None:
    """The injected print CSS makes an output `<pre>` wrap (defect: a `white-space: pre` line
    longer than the printable page width is CLIPPED by the printer, not wrapped -- a 92-char
    run-path line lost its `__seed_42` tail with no error). Scoped to `.jp-OutputArea-output
    pre` so it cannot disturb the existing `break-inside: avoid` rule or figures."""
    html = render_module._inject_print_css("<html><head></head><body></body></html>")

    assert ".jp-OutputArea-output pre" in html
    assert "white-space: pre-wrap" in html
    assert "overflow-wrap: anywhere" in html
    # The pre-existing page-break rule must survive untouched alongside the new one.
    assert "break-inside: avoid" in html


def test_strip_stderr_preprocessor_drops_warning_but_keeps_other_outputs() -> None:
    """`_StripStderrPreprocessor` removes only `stream`/`stderr` output entries (e.g. a
    matplotlib `UserWarning` printed during a chart cell) -- stdout, text/plain results and
    figures on the same cell survive untouched (defect: a warning leaking onto stderr became
    visible report body text, including a local temp path and kernel PID)."""
    cell = new_code_cell("plt.show()")
    cell.outputs = [
        new_output("stream", name="stdout", text="loaded 42 runs\n"),
        new_output(
            "stream",
            name="stderr",
            text="/tmp/x.py:44: UserWarning: Tight layout not applied.\n",
        ),
        new_output("display_data", data={"image/png": "not-really-png-data"}),
    ]

    processed_cell, _resources = render_module._StripStderrPreprocessor().preprocess_cell(cell, {}, 0)

    output_types = [(output.output_type, output.get("name")) for output in processed_cell.outputs]
    assert ("stream", "stdout") in output_types
    assert ("stream", "stderr") not in output_types
    assert ("display_data", None) in output_types


def test_render_pdf_gracefully_skips_with_no_browser_and_no_playwright(tmp_path: Path, monkeypatch) -> None:
    """No browser + no playwright: `(None, reason)`, no exception, nothing written (T081, FR-026)."""
    monkeypatch.setattr(render_module, "_discover_browser", lambda: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>probe</body></html>", encoding="utf-8")
    pdf_path = tmp_path / "report.pdf"
    nb = new_notebook()

    pdf_result, reason = render_module._render_pdf(nb, html_path, pdf_path, Config())

    assert pdf_result is None
    assert reason is not None and reason.strip() != ""
    assert not pdf_path.exists()


def test_render_pdf_gracefully_skips_when_browser_print_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """A browser that "succeeds" without writing a file is still a skip (research R6/R11):
    success is judged by the file existing and being non-empty, never the return code."""
    monkeypatch.setattr(render_module, "_discover_browser", lambda: Path("fake-browser"))
    monkeypatch.setattr(render_module, "_print_with_browser", lambda *a, **k: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>probe</body></html>", encoding="utf-8")
    pdf_path = tmp_path / "report.pdf"
    nb = new_notebook()

    pdf_result, reason = render_module._render_pdf(nb, html_path, pdf_path, Config())

    assert pdf_result is None
    assert reason is not None
    assert not pdf_path.exists()


@pytest.mark.slow
def test_cell_execution_failure_writes_failed_notebook_with_sanitised_reason(tmp_path: Path) -> None:
    """A failing cell writes `<name>.failed.ipynb` and a sanitised, ANSI-free reason (T082)."""
    # The raised message deliberately carries a raw ANSI escape and non-ASCII characters,
    # mirroring the reproduced UnicodeEncodeError incident (research R11).
    source = _PERCENT_HEADER + '# %%\nraise ValueError("boom: caf\\u00e9 \\u00b1 \\u2192 \\x1b[31mred\\x1b[0m")\n'
    py_path = tmp_path / "failing_report.py"
    py_path.write_text(source, encoding="utf-8")

    result = render_module.render_report(py_path, execute_timeout=60)

    failed_path = tmp_path / "failing_report.failed.ipynb"
    assert result.notebook == failed_path
    assert failed_path.exists()
    assert result.html.exists()
    assert result.pdf is None
    assert result.pdf_skip_reason is not None
    assert "\x1b" not in result.pdf_skip_reason
    assert all(ord(ch) < 128 for ch in result.pdf_skip_reason)
    assert "cell execution failed" in result.pdf_skip_reason


@pytest.mark.slow
def test_render_report_gracefully_skips_pdf_with_no_browser_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: with browser discovery stubbed to find nothing, the executed notebook and
    HTML still exist, `pdf` is `None`, and a reason is set -- no exception (FR-026)."""
    monkeypatch.setattr(render_module, "_discover_browser", lambda: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    source = _PERCENT_HEADER + '# %%\nprint("hello from a report that never sees a browser")\n'
    py_path = tmp_path / "no_browser_report.py"
    py_path.write_text(source, encoding="utf-8")

    result = render_module.render_report(py_path, execute_timeout=60)

    assert result.notebook == tmp_path / "no_browser_report.ipynb"
    assert result.notebook.exists()
    assert result.html.exists()
    assert result.pdf is None
    assert result.pdf_skip_reason is not None


def _browser_or_playwright_available() -> bool:
    return render_module._discover_browser() is not None or importlib.util.find_spec("playwright") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _browser_or_playwright_available(), reason="no PDF rendering engine available")
def test_generated_pdf_contains_tables_but_no_mechanical_code(run_tree_builder: RunTreeBuilder) -> None:
    """SC-006: the PDF has content, a known table heading, and omits mechanical code such as
    `import matplotlib` -- exactly the FR-024/FR-025 contract, verified by reading the PDF
    back with pypdf (research R12: never assert on `image/png` appearing in the HTML, since a
    CodeMirror CSS artifact false-positives on that string)."""
    import pypdf  # noqa: PLC0415 -- dev-only dependency, kept local to this test

    for i in range(2):
        run_tree_builder.add_run(
            model_name="simple_q_learning",
            model_signature=f"sig_{i}",
            hyperparameters={"learning_rate": 0.001 * (i + 1), "seed": i},
            learning_episode_count=8,
            testing_episode_count=4,
        )

    bundle = generate_report(run_tree_builder.root / "FrozenLake-v1" / "env_sig", execute=True, render_pdf=True)
    artifact = bundle.reports[0]

    # The .py source is the durable, re-runnable artifact and is distinct from the executed
    # .ipynb it was derived from (FR-027) -- unlike the --no-execute path, where they coincide.
    assert artifact.source.exists()
    assert artifact.source.suffix == ".py"
    assert artifact.source != artifact.notebook
    assert artifact.notebook.exists()
    assert artifact.html.exists()
    assert artifact.pdf is not None, f"PDF was skipped: {artifact.pdf_skip_reason}"
    assert artifact.pdf.exists()
    assert artifact.pdf.stat().st_size > 0

    reader = pypdf.PdfReader(str(artifact.pdf))
    assert len(reader.pages) >= 1

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Performance Metrics" in text
    assert "import matplotlib" not in text
    assert "q_network_state_dict" not in text
    assert "q_table" not in text
