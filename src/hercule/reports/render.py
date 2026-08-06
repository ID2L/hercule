"""The notebook -> execute -> HTML -> PDF render pipeline (User Story 5).

`render_report(py_path)` runs the jupytext -> execute -> HTML -> browser-print pipeline
(plan D5, research R6-R8, R11, R12): read the generated `.py` as a notebook, execute it
in a real kernel, export a tag-filtered HTML, then print that HTML to PDF with a system
Chromium-family browser, falling back to an optional `WebPDFExporter` when no browser is
present. Every failure path degrades gracefully (FR-026): the notebook and the HTML are
always written, and a PDF failure is reported as a reason rather than an exception.
"""

import importlib.util
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import jupytext
import nbformat
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, Preprocessor
from pydantic import BaseModel, field_validator, model_validator
from traitlets.config import Config


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sanitize_reason(value: str) -> str:
    """Strip ANSI escapes and coerce to ASCII text safe for any console encoding.

    Duplicated from `hercule.reports.run_table`/`hercule.reports.__init__` rather than
    imported: both of those modules import this one (directly or via re-export) to expose
    `render_report`, so a top-level import in the other direction would be circular. The
    helper is a few lines of pure text handling with no state, so duplication is cheaper
    than restructuring the import graph (research R11) -- the same trade-off `run_table.py`
    already makes for the identical reason.

    Args:
        value: The raw reason text.

    Returns:
        The stripped, ANSI-free, ASCII-safe text.
    """
    stripped = _ANSI_ESCAPE_RE.sub("", value).strip()
    return stripped.encode("ascii", errors="replace").decode("ascii")


class ArtifactWriteError(OSError):
    """Raised when a render artifact's destination cannot be replaced or removed.

    Distinct from the graceful, never-raising degrade paths documented on `RenderResult`
    (FR-026, e.g. "no browser found"): this is an *output* problem, most commonly a stale
    `.pdf`/`.ipynb`/`.html` held open by another process (a PDF preview tab, an editor).
    A subclass of `OSError` on purpose, so a caller that only wants "was this a write
    problem, not an invalid-input problem" can catch `OSError` without importing this name.
    """


def _probe_removable(path: Path) -> str | None:
    """Non-destructively verify that `path` can be replaced or removed.

    Renames `path` aside and immediately back. A rename requires the same delete/rename
    permission an actual install (`Path.replace`) or cleanup (`Path.unlink`) would need, so a
    failure here means the same failure would happen during the real write -- most commonly
    because another process (e.g. a PDF preview tab) holds the file open without
    share-delete access (measured: Windows `WinError 32`). Nothing is deleted or truncated by
    this probe itself, so a caller can use it to decide *before* touching anything (FR-028).

    Args:
        path: Candidate artifact path; a non-existent path is trivially removable.

    Returns:
        `None` when `path` is safe to replace/remove, or a sanitized reason naming it otherwise.
    """
    if not path.exists():
        return None
    probe_path = path.with_name(path.name + ".hercule-lock-probe")
    try:
        path.rename(probe_path)
    except OSError as exc:
        return _sanitize_reason(
            f"'{path}' is open in another program (e.g. a preview tab or editor) and cannot be "
            f"replaced; close it and regenerate the report: {exc}"
        )
    probe_path.rename(path)
    return None


def check_artifacts_writable(py_path: Path) -> str | None:
    """Verify every render artifact sibling to `py_path` can be replaced or removed.

    Intended to be called *before* anything is written for a report group -- including
    `py_path` itself -- so a locked destination is detected and reported up front rather than
    discovered mid-write with some artifacts already replaced and others not (FR-028: never a
    mix of fresh and stale artifacts).

    Args:
        py_path: The jupytext `.py` report whose siblings (`.ipynb`, `.failed.ipynb`, `.html`,
            `.pdf`) share its stem.

    Returns:
        `None` when every sibling artifact is safe to replace, or a sanitized reason naming the
        first offending path otherwise.
    """
    for candidate in (
        py_path,
        py_path.with_suffix(".ipynb"),
        py_path.with_name(f"{py_path.stem}.failed.ipynb"),
        py_path.with_suffix(".html"),
        py_path.with_suffix(".pdf"),
    ):
        reason = _probe_removable(candidate)
        if reason is not None:
            return reason
    return None


# Cell-tag vocabulary shared with the generated templates (contracts C4). Duplicated as
# literals rather than imported from `hercule.reports` (which imports this module to
# re-export `render_report`) to avoid a circular import; `reports/__init__.py` asserts
# these match its own constants via a test.
TAG_REMOVE_CELL = "remove_cell"
TAG_REMOVE_INPUT = "remove_input"
TAG_REMOVE_OUTPUT = "remove_output"

# Print CSS so charts and tables do not straddle a page break (research R6), and so a
# preformatted output line longer than the printable page width wraps instead of silently
# overflowing the page box. A `<pre>` output defaults to `white-space: pre`: the browser's
# print engine then clips whatever does not fit on the line rather than reflowing it, with no
# error and no visible sign of the loss (measured: a 92-character run-path line was clipped to
# 88 characters in the PDF, silently dropping its `__seed_42` suffix). Scoped to
# `.jp-OutputArea-output pre` -- text/stream output only -- so it never touches a figure
# (`jp-RenderedImage` output has no `<pre>` inside) or the `break-inside: avoid` rule above; a
# table row that already fits the printable width (~88 characters, measured) is untouched and
# still renders as one line, since wrapping only engages once a line actually overflows.
_PRINT_CSS = (
    "<style>\n"
    "@page { size: A4; margin: 14mm; }\n"
    "@media print {\n"
    "  .jp-Cell, .jp-CodeCell, .jp-MarkdownCell, .jp-OutputArea-output { break-inside: avoid; }\n"
    "  img { max-width: 100%; }\n"
    "  .jp-OutputArea-output pre {\n"
    "    white-space: pre-wrap;\n"
    "    overflow-wrap: anywhere;\n"
    "    word-break: break-word;\n"
    "  }\n"
    "}\n"
    "</style>\n"
)

# Standard Chromium-family browser names probed via shutil.which, and standard install
# locations probed directly -- both are required: `shutil.which()` finds neither `msedge`
# nor `chrome` on a machine where both are installed only at their default paths (research
# R12), which is the common case on a workstation where the browser was launched from the
# Start Menu rather than added to PATH.
_BROWSER_NAMES = ("msedge", "chrome", "chromium", "chromium-browser")
_STANDARD_BROWSER_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path("/usr/bin/microsoft-edge"),
    Path("/usr/bin/microsoft-edge-stable"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)

_PDF_REMEDIATION = "uv sync --extra pdf && uv run playwright install chromium"


class RenderResult(BaseModel):
    """The outcome of `render_report(py_path)`: paths on disk, nothing about run counts.

    `pdf` is `None` on every failure path (FR-026); `pdf_skip_reason` is set exactly then.
    """

    notebook: Path
    html: Path
    pdf: Path | None = None
    pdf_skip_reason: str | None = None

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
    def _validate_pdf_xor_reason(self) -> "RenderResult":
        if (self.pdf is None) == (self.pdf_skip_reason is None):
            raise ValueError("exactly one of pdf or pdf_skip_reason must be set")
        return self


def _make_cell_start_hook(progress):
    """Build the `on_cell_start` traitlets callback for `ExecutePreprocessor`.

    Progress hooks are traitlets `Callable` traits invoked with **keyword** arguments, not
    overridable methods (research R8) -- `cell`/`cell_index` must be accepted as keywords
    with defaults, and any other keyword nbclient may pass is absorbed by `**_`.

    Args:
        progress: Optional sink for human-readable progress lines; `None` disables reporting.

    Returns:
        A callable suitable for `ExecutePreprocessor(on_cell_start=...)`.
    """

    def _on_cell_start(cell=None, cell_index=None, **_) -> None:
        if progress is not None:
            progress(f"Executing cell {cell_index}" if cell_index is not None else "Executing cell")

    return _on_cell_start


class _StripStderrPreprocessor(Preprocessor):
    """Drop stderr stream output from every cell before HTML/PDF export.

    `remove_input`-tagged cells keep their *output* by design (FR-025 -- it is how the PDF
    keeps "loaded N runs, skipped M" while hiding the loading code), so a warning emitted
    during that cell's execution (e.g. matplotlib's `UserWarning` when a layout cannot
    accommodate all Axes decorations) lands on stderr as an output too and would otherwise
    surface as visible report body text -- in the incident that motivated this, a local temp
    path and a kernel PID. `remove_all_outputs_tags` was considered and rejected: it drops
    *every* output of a tagged cell, which would also delete that cell's chart or table, not
    just the stray warning sharing its cell. Stripping only `output_type == "stream", name ==
    "stderr"` entries is narrower and leaves stdout (`print(...)`) and rich outputs (figures,
    text/plain results) untouched, so it is the least surprising of the two mechanisms. This
    is a systemic safety net alongside the narrow `warnings.filterwarnings(...)` call in each
    template's own setup cell (which stops the specific known warning at its source) -- an
    unanticipated warning class still cannot leak through.
    """

    def preprocess_cell(self, cell, resources, index):
        """Remove stderr stream outputs from `cell`, keeping every other output untouched."""
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            cell.outputs = [
                output
                for output in cell.outputs
                if not (output.get("output_type") == "stream" and output.get("name") == "stderr")
            ]
        return cell, resources


def _build_tag_removal_config() -> Config:
    """Build the shared `TagRemovePreprocessor` configuration (contracts C4, research R7).

    `TagRemovePreprocessor.enabled` defaults to `False` -- setting it is mandatory, not
    redundant. `HTMLExporter.preprocessors` is set here; the optional `WebPDFExporter`
    fallback sets its own `preprocessors` trait separately, since traitlets config is
    per-class (T096). `_StripStderrPreprocessor` rides alongside it so a stray warning never
    becomes report body text regardless of which exporter renders the notebook.
    """
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_cell_tags = (TAG_REMOVE_CELL,)
    config.TagRemovePreprocessor.remove_input_tags = (TAG_REMOVE_INPUT,)
    config.TagRemovePreprocessor.remove_all_outputs_tags = (TAG_REMOVE_OUTPUT,)
    config.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor",
        "hercule.reports.render._StripStderrPreprocessor",
    ]
    config.HTMLExporter.exclude_input_prompt = True
    config.HTMLExporter.exclude_output_prompt = True
    return config


def _inject_print_css(html: str) -> str:
    """Insert the print stylesheet before `</head>`, or prepend it if no head tag is found."""
    marker = "</head>"
    if marker in html:
        return html.replace(marker, _PRINT_CSS + marker, 1)
    return _PRINT_CSS + html


def _export_html(nb: nbformat.NotebookNode, config: Config) -> str:
    """Export `nb` to tag-filtered HTML, with print CSS injected.

    Best-effort: `html` is a required, always-written `RenderResult` field, so a failure
    here falls back to a minimal document stating the failure rather than propagating --
    an executed (or partially executed) notebook object always exists by this point.
    """
    try:
        body, _resources = HTMLExporter(config=config).from_notebook_node(nb)
    except Exception as exc:  # noqa: BLE001 -- HTML export must never block artifact delivery
        reason = _sanitize_reason(str(exc))
        return f"<html><body><p>HTML export failed: {reason}</p></body></html>"
    return _inject_print_css(body)


def _discover_browser() -> Path | None:
    """Find a Chromium-family browser via `PATH`, then the standard install locations.

    `shutil.which()` alone is not sufficient: verified on this project's own development
    machine, neither `msedge` nor `chrome` is on `PATH`, yet both exist at their standard
    install paths (research R12) -- the fallback is load-bearing, not a nicety.

    Returns:
        The browser executable's path, or `None` when none was found anywhere.
    """
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    for candidate in _STANDARD_BROWSER_PATHS:
        if candidate.exists():
            return candidate
    return None


def _print_with_browser(browser: Path, html_path: Path, pdf_out: Path, timeout: int = 180) -> bool:
    """Print `html_path` to `pdf_out` with a headless Chromium-family browser.

    `--user-data-dir` pointed at a fresh temporary directory is mandatory: if the browser is
    already running elsewhere on the machine (near-certain on a dev workstation), a bare
    `--print-to-pdf` hands off to the running instance and exits 0 having written nothing
    (research R6/R11). Success is judged by the output file existing and being non-empty,
    never by the return code, for the same reason.

    Args:
        browser: Path to the browser executable.
        html_path: The HTML file to print; passed as a `file://` URI.
        pdf_out: Destination PDF path -- should be a short temporary path (MAX_PATH, R11).
        timeout: Seconds to wait for the browser subprocess.

    Returns:
        Whether `pdf_out` now exists and is non-empty.
    """
    profile_dir = tempfile.mkdtemp(prefix="hercule-pdf-profile-")
    try:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={profile_dir}",
            "--no-pdf-header-footer",
            "--virtual-time-budget=10000",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            f"--print-to-pdf={pdf_out}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
    return pdf_out.exists() and pdf_out.stat().st_size > 0


def _render_with_webpdf(nb: nbformat.NotebookNode, config: Config) -> bytes | None:
    """Render `nb` to PDF bytes via the optional Playwright-backed `WebPDFExporter`.

    Gated by the caller on `importlib.util.find_spec("playwright")` (T096); imported lazily
    here so the module stays importable when the `pdf` extra is not installed.

    Args:
        nb: The executed notebook.
        config: The shared tag-removal `Config`; `WebPDFExporter.preprocessors` is set on it
            separately from `HTMLExporter.preprocessors`, since traitlets config is per-class.

    Returns:
        The rendered PDF bytes, or `None` on any failure.
    """
    from nbconvert.exporters.webpdf import WebPDFExporter  # noqa: PLC0415 -- optional dependency

    config.WebPDFExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor",
        "hercule.reports.render._StripStderrPreprocessor",
    ]
    try:
        pdf_bytes, _resources = WebPDFExporter(config=config).from_notebook_node(nb)
    except Exception:  # noqa: BLE001 -- any failure here is just another skip reason
        return None
    return pdf_bytes if pdf_bytes else None


def _render_pdf(
    nb: nbformat.NotebookNode, html_path: Path, pdf_path: Path, config: Config
) -> tuple[Path | None, str | None]:
    """Produce the PDF, trying a system browser first and an optional Playwright fallback.

    Returns:
        `(pdf_path, None)` on success, or `(None, reason)` on every failure path -- never
        raises (FR-026).
    """
    browser = _discover_browser()
    if browser is not None:
        with tempfile.TemporaryDirectory(prefix="hercule-pdf-out-") as tmp_dir:
            # Intermediate PDF goes to a short temp path and is moved into place: the output
            # tree already approaches MAX_PATH before a filename is appended (R11).
            temp_pdf = Path(tmp_dir) / "report.pdf"
            if _print_with_browser(browser, html_path, temp_pdf):
                try:
                    shutil.move(str(temp_pdf), str(pdf_path))
                except OSError as exc:
                    # Extremely unlikely once `render_report`'s own preflight check has
                    # already confirmed `pdf_path` is replaceable, but a lock can still
                    # reappear in the window between that check and this move (T-O-C-T-O-U).
                    # Degrade gracefully here rather than raising: the notebook and HTML are
                    # already installed by this point, and a rare late PDF hiccup should not
                    # discard them (FR-026).
                    return None, _sanitize_reason(f"the rendered PDF could not be installed at '{pdf_path}': {exc}")
                return pdf_path, None
        browser_reason = "the browser print produced no file"
    else:
        browser_reason = (
            f"no Chromium-family browser (msedge/chrome/chromium) was found; run `{_PDF_REMEDIATION}`, "
            "or install Edge/Chrome/Chromium"
        )

    if importlib.util.find_spec("playwright") is not None:
        pdf_bytes = _render_with_webpdf(nb, config)
        if pdf_bytes:
            pdf_path.write_bytes(pdf_bytes)
            return pdf_path, None
        return None, f"{browser_reason}; the optional WebPDFExporter fallback also failed"

    return None, f"{browser_reason}; the optional WebPDFExporter fallback is unavailable (playwright not installed)"


def render_report(
    py_path: Path,
    *,
    execute_timeout: int = 1800,
    render_pdf: bool = True,
    progress=None,
) -> RenderResult:
    """Execute a generated jupytext `.py` report and print it to PDF.

    Pipeline (research R6-R8, R11, R12): `jupytext.read` the `.py` as a notebook, execute it
    with a real kernel (`timeout=execute_timeout`, `startup_timeout=120`,
    `interrupt_on_timeout=True`, `allow_errors=False`, `record_timing=False`), export a
    tag-filtered HTML, then print that HTML to PDF with a system Chromium-family browser,
    falling back to an optional `WebPDFExporter`. Every failure path still returns a
    `RenderResult` with `pdf=None` and a reason (FR-026) -- the executed (or partially
    executed) notebook and the HTML are always written.

    Regeneration replaces artifacts in place rather than accumulating stale ones (FR-028), but
    never at the cost of leaving a fresh artifact beside a stale one it failed to replace: a
    writability preflight (`check_artifacts_writable`) runs first and, when it finds a locked
    sibling (e.g. a PDF open in a preview tab), this raises `ArtifactWriteError` *before*
    anything on disk is touched, leaving the previous, fully consistent set of artifacts intact.
    That is a distinct failure mode from every other path here, which never raises (FR-026).

    Args:
        py_path: The generated jupytext-percent `.py` report. Sibling artifacts
            (`.ipynb`/`.failed.ipynb`/`.html`/`.pdf`) share its stem and directory.
        execute_timeout: Per-cell execution timeout in seconds. The default `timeout=None`
            on `ExecutePreprocessor` is unbounded, which would hang forever on a stuck cell.
        render_pdf: Whether to attempt the PDF step at all after a successful execution
            (the `--no-pdf` CLI escape hatch). When `False`, the notebook is still executed
            and the HTML still exported, but `pdf` is `None` with an explicit reason.
        progress: Optional sink for human-readable progress lines, invoked once per cell
            (SC-008's 30-second cadence).

    Returns:
        `RenderResult` describing every artifact written and, when the PDF was skipped, why.

    Raises:
        ArtifactWriteError: When a sibling artifact cannot be replaced or removed (most
            commonly because it is open in another program). Nothing is written in that case.
    """
    report_dir = py_path.parent
    notebook_path = py_path.with_suffix(".ipynb")
    failed_notebook_path = py_path.with_name(f"{py_path.stem}.failed.ipynb")
    html_path = py_path.with_suffix(".html")
    pdf_path = py_path.with_suffix(".pdf")

    # Verify every sibling can be replaced/removed BEFORE touching any of them (FR-028): a
    # locked destination must abort cleanly, not partway through the clearing loop below with
    # some artifacts already gone and others not.
    lock_reason = check_artifacts_writable(py_path)
    if lock_reason is not None:
        raise ArtifactWriteError(lock_reason)

    # Regeneration replaces artifacts in place rather than accumulating stale ones (FR-028):
    # clear every possible sibling before writing this run's outputs. The preflight check just
    # above makes this expected to succeed; still guarded, since a lock can in principle
    # reappear in the tiny window between the check and this loop.
    for stale in (notebook_path, failed_notebook_path, html_path, pdf_path):
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            raise ArtifactWriteError(
                _sanitize_reason(
                    f"cannot remove existing '{stale}' -- it is likely open in another program "
                    f"(e.g. a preview tab or editor); close it and regenerate the report: {exc}"
                )
            ) from exc

    nb = jupytext.read(py_path, fmt="py:percent")

    executor = ExecutePreprocessor(
        timeout=execute_timeout,
        startup_timeout=120,
        interrupt_on_timeout=True,
        allow_errors=False,
        record_timing=False,
        kernel_name="python3",
        on_cell_start=_make_cell_start_hook(progress),
    )

    execution_reason: str | None = None
    with warnings.catch_warnings():
        # The Proactor event loop's "does not implement add_reader" RuntimeWarning is benign
        # on Windows -- pyzmq handles it with a selector thread (research R11). Do NOT change
        # the global asyncio event loop policy to silence it: that would strip asyncio
        # subprocess support, which the optional Playwright fallback needs.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            executor.preprocess(nb, {"metadata": {"path": str(report_dir)}})
            written_notebook_path = notebook_path
        except CellExecutionError as exc:
            execution_reason = _sanitize_reason(f"cell execution failed: {exc}")
            written_notebook_path = failed_notebook_path
        except CellTimeoutError as exc:
            execution_reason = _sanitize_reason(f"cell execution timed out: {exc}")
            written_notebook_path = failed_notebook_path
        except DeadKernelError as exc:
            execution_reason = _sanitize_reason(f"kernel died during execution: {exc}")
            written_notebook_path = failed_notebook_path

    # nbconvert does not save the notebook when it raises, so the traceback would otherwise
    # be lost entirely (research R8) -- write it ourselves on both branches.
    nbformat.write(nb, str(written_notebook_path))

    config = _build_tag_removal_config()
    html_body = _export_html(nb, config)
    html_path.write_text(html_body, encoding="utf-8")

    if execution_reason is not None:
        return RenderResult(
            notebook=written_notebook_path,
            html=html_path,
            pdf=None,
            pdf_skip_reason=execution_reason,
        )

    if not render_pdf:
        return RenderResult(
            notebook=written_notebook_path,
            html=html_path,
            pdf=None,
            pdf_skip_reason="PDF rendering was skipped (--no-pdf)",
        )

    pdf_result_path, pdf_skip_reason = _render_pdf(nb, html_path, pdf_path, config)

    return RenderResult(
        notebook=written_notebook_path,
        html=html_path,
        pdf=pdf_result_path,
        pdf_skip_reason=pdf_skip_reason,
    )


__all__ = [
    "ArtifactWriteError",
    "RenderResult",
    "check_artifacts_writable",
    "render_report",
]
