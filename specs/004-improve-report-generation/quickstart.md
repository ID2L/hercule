# Quickstart: Improved Experiment Report Generation

**Feature**: `004-improve-report-generation` | **Phase**: 1 | **Date**: 2026-07-28

For a developer picking this up. Assumes `D:\code\hercule` as cwd and the existing `outputs/`
tree as the validation corpus — **no training run is needed**.

---

## 1. Setup

Dependencies decided in research R10. Run these literally:

```bash
# main deps: the notebook -> execute -> HTML pipeline
uv add "jupytext>=1.16,<2" "nbformat>=5.10,<6" "nbconvert>=7.16,<8"

# ipykernel moves dev -> main: `hercule report` now executes the notebook for the PDF,
# so an end-user install must have a kernel
uv remove --group dev ipykernel
uv add "ipykernel>=7.1,<8"

# dev only: SC-006 is verified by reading the PDF back
uv add --group dev "pypdf>=5.1,<6"

# optional extra `pdf`: fallback WebPDFExporter engine for headless CI with no system browser
uv add --optional pdf "playwright>=1.49,<2"

uv sync
```

`[tool.uv] default-groups = "all"` governs **groups**, not **extras** — so `playwright` stays
out of a plain `uv sync`, which is the intent. Only opt in when you need engine 2:

```bash
uv sync --extra pdf && uv run playwright install chromium
```

That command string is also the remediation text FR-026 must print when the PDF is skipped.

**`scikit-learn` is NOT added.** PCA is implemented with `numpy.linalg.svd` (R4). Do not reach
for sklearn: it would pull 4 packages including scipy (~35-60 MB of wheels), it does not scale
(so `StandardScaler` would be needed anyway), and its SVD sign convention is an unpinned
internal that already changed once — generated notebooks run on the *reader's* machine, so
delegating the sign rule would make plot orientation depend on their sklearn version and break
FR-014/SC-009. Also not added: `seaborn`, `papermill`, explicit `nbclient`, pandoc, TeX,
`weasyprint`.

---

## 2. Validating against the real datasets

Three result sets exist. Work them in this order.

### 2a. `outputs/simple_games` — start here

```bash
uv run hercule report outputs/simple_games
```

22 runs, 2 environment-settings groups of 11 runs each:

```text
outputs/simple_games/FrozenLake-v1/is_sli_False__map_nam_4x4__max_epi_ste_200   # 11 runs
outputs/simple_games/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200    # 11 runs
```

3 model families per group: `dummy`, `simple_q_learning`, `simple_sarsa`. Smallest and fastest
round trip — best first target while iterating on the templates.

What it exercises:

- **SC-003 / FR-001-003** — the environment prose must state `FrozenLake-v1` plus
  `is_slippery`, `map_name`, `max_episode_steps`, and the two groups must differ visibly.
- **FR-020 / SC-007 negative branch** — `dummy` has only a `seed` hyperparameter, which is
  constant, so after the zero-variance drop `p_kept == 0`. PCA **must** render
  "not applicable for this family" as text and keep going (FR-021). If it raises, every cell
  below it dies. This is the family to check first.
- **FR-011 / SC-002 lower bound** — 11 runs per group is just above the 9-series cap; drop to a
  single model family (3-4 runs) to check the "all drawn, none omitted" path.
- **FR-015 is NOT exercised here.** Measured: every run in both `simple_games` groups has 200
  episodes, and every run in `frozenlake_4x4` has 5000 — lengths differ only *between* configs,
  never inside a group. Unequal lengths within one group must therefore be tested with a
  synthetic fixture (`run_tree_builder` takes a per-run episode count).
- **FR-030** — any group skipped for holding too few runs must be named with a reason.

### 2b. `outputs/frozenlake_4x4` — the PCA target

```bash
uv run hercule report outputs/frozenlake_4x4
```

135 runs, **one** environment-settings group:

```text
outputs/frozenlake_4x4/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200   # 135 runs
```

2 model families: `deep_q_learning` (108 runs) and `simple_q_learning` (27 runs).

What it exercises:

- **FR-016-019 / SC-007 positive branch** — both families vary ≥ 2 hyperparameters, so this is
  where the projection, its explained variance and its loadings are actually read.
- **FR-020** — two families with disjoint hyperparameter sets in one group: they must be
  projected separately, never merged.
- **Spec edge case "disjoint hyperparameter sets"** — the grid table spanning DQN and tabular Q
  must show absent hyperparameters as absent, not as a value.
- **FR-010/FR-012/FR-013** — 135 runs is well over the cap, and FrozenLake ties heavily (many
  runs score a mean reward of exactly 0), so this is the real test of the
  `(-metric, directory_name)` tie-break and of the omitted-count message.
- **FR-014 / SC-009** — regenerate twice and diff (see §3).

### 2c. `outputs/dq_cartpole` — the performance target

```bash
uv run hercule report outputs/dq_cartpole
```

218 runs, ~211 MB of episode metrics, ~110 MB of model weights. Two environment-settings
groups of 109 runs each:

```text
outputs/dq_cartpole/CartPole-v1/default             # 109 runs, ~163 MB on disk
outputs/dq_cartpole/CartPole-v1/sut_bar_rew_True    # 109 runs
```

Slowest by a wide margin. Run it last, and expect minutes.

What it exercises:

- **SC-008** — must complete within 10 minutes and emit progress at least every 30 s. Time it:
  `time uv run hercule report outputs/dq_cartpole`.
- **FR-007 / SC-010** — this is the set where the current template dumps
  `q_network_state_dict` verbatim into the document. The run table must never open
  `model.json`.
- **FR-002 acceptance scenario 2** — `CartPole-v1/default` takes no extra settings, so the
  report must say "no environment-specific setting was overridden" rather than printing an
  empty structure.
- **SC-004/SC-005** — no errors, no unexplained skipped group, executes end to end. This one
  invocation must produce **2** comparative reports.

> Group vs. invocation: the 218 CartPole runs are split across two env-settings groups of 109,
> so no single report covers 218 runs. The largest single **report group** in the repo is
> `frozenlake_4x4` at 135 runs — that is the SC-001 size-comparison target. The 218 is the
> per-**invocation** total and is what SC-008's 10-minute bound applies to. The spec was
> corrected to match; if you see "218-run group" anywhere, it is stale.

---

## 3. Verifying the acceptance criteria by hand

### SC-001 — document size must not scale with run count

Compare the generated `.py` for the largest group against a tiny one. Build the tiny one by
copying two run directories out of a group:

```bash
mkdir -p /tmp/hercule-sc001/CartPole-v1/default/deep_q_learning
cd D:/code/hercule/outputs/dq_cartpole/CartPole-v1/default/deep_q_learning
ls -d */ | head -2 | while read d; do cp -r "$d" /tmp/hercule-sc001/CartPole-v1/default/deep_q_learning/; done

cd D:/code/hercule
uv run hercule report /tmp/hercule-sc001
wc -l /tmp/hercule-sc001/CartPole-v1/default/comparative_report.py
wc -l outputs/dq_cartpole/CartPole-v1/default/comparative_report.py
```

The two line counts must agree within 10%. Today the second grows by ~35 lines per run, so
before the change expect roughly 70 vs ~3800 lines; after it, the same number twice.

### SC-009 — deterministic series selection

```bash
uv run hercule report outputs/frozenlake_4x4
cp outputs/frozenlake_4x4/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200/comparative_report.py /tmp/run1.py
uv run hercule report outputs/frozenlake_4x4
diff /tmp/run1.py outputs/frozenlake_4x4/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200/comparative_report.py
```

`diff` must be empty. (This needs `record_timing=False` on the `ExecutePreprocessor`, per R8,
or the executed notebook carries per-cell ISO timestamps.)

### SC-002 — at most 9 series per chart

Eyeball: open the generated HTML (or the notebook) and count legend rows on every
learning-progress and evaluation chart — at most 9, each labelled with its bucket (best /
median / worst), with a line stating how many runs were not drawn.

Assert in a test — do it on the pure helper, not the figure:

```python
# tests/reports/test_selection.py
import pytest
from hercule.reports.run_table import build_run_table
from hercule.reports.selection import select_series


@pytest.mark.integration
def test_no_chart_exceeds_nine_series():
    group = "outputs/frozenlake_4x4/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200"
    table = build_run_table(group)
    for metric in ("mean_learning_reward", "learning_success_rate", "mean_testing_reward"):
        result = select_series(table.records, metric)
        assert len(result.selected) <= 9
        assert result.omitted_count == len(table.records) - len(result.selected)
        assert {s.bucket for s in result.selected} <= {"best", "median", "worst"}
```

Every chart in the template must go through `select_series`; a chart that plots
`table.records` directly is the failure mode to grep for.

### SC-006 — PDF has all charts and tables, no import code

```bash
uv run python - <<'PY'
from pathlib import Path
from pypdf import PdfReader

pdf = Path("outputs/simple_games/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200/comparative_report.pdf")
print("size:", pdf.stat().st_size)

reader = PdfReader(pdf)
print("pages:", len(reader.pages))

text = "\n".join(page.extract_text() or "" for page in reader.pages)
images = sum(len(page.images) for page in reader.pages)
print("images:", images)

# FR-025: mechanical code is gone
for banned in ("import matplotlib", "import pandas", "build_run_table", "def _locate_report_dir"):
    assert banned not in text, f"mechanical code leaked into the PDF: {banned}"

# FR-024: narrative, informative output and tables survived
for expected in ("FrozenLake-v1", "is_slippery", "loaded", "runs"):
    assert expected in text, f"missing from the PDF: {expected}"

# one image per chart in the notebook
assert images >= 4, f"charts missing from the PDF: only {images} images"
print("SC-006 OK")
PY
```

Cross-check `images` against the number of `plt.show()` calls in the generated `.py`. Note
`page.images` counts embedded XObjects, so a chart split across a page break still counts once.

### SC-010 — no model weights in any output artifact

```bash
grep -rl "q_network_state_dict\|q_table" \
  outputs/dq_cartpole/CartPole-v1/default/comparative_report.py \
  outputs/dq_cartpole/CartPole-v1/default/comparative_report.ipynb \
  outputs/dq_cartpole/CartPole-v1/default/comparative_report.html
```

Must return nothing. `model.json` itself of course contains those keys — scope the grep to the
generated artifacts only, never to the whole tree. A stronger check: assert in
`tests/reports/test_run_table.py` that `build_run_table` never opens `model.json` (monkeypatch
`Path.open` or `json.load` and record the paths).

---

## 4. Gotchas already discovered

Condensed from research; each of these cost time to find.

- **`Path(__file__)` is undefined in a kernel** (R9). Confirmed: `HAS_FILE: False`, and
  `sys.argv[0]` is `.../site-packages/ipykernel_launcher.py`, so an `__file__`-derived path
  resolves *into site-packages*. This is why the current templates cannot execute at all. Use a
  verified candidate search anchored on a file that must exist, prefer `Path.cwd()` inside a
  kernel, and always pass `resources={"metadata": {"path": str(report_dir)}}` when executing.
- **`TagRemovePreprocessor.enabled` defaults to `False`** (R7). Set
  `c.TagRemovePreprocessor.enabled = True` or nothing is stripped and you will conclude the
  tags are wrong. traitlets config is per-exporter-class, so the optional `WebPDFExporter`
  needs its own `c.WebPDFExporter.preprocessors`.
- **`--user-data-dir=<fresh temp dir>` is mandatory, not hygiene** (R6/R11). If Edge or Chrome
  is already running — near-certain on a dev workstation — a bare `--print-to-pdf` hands off to
  the running instance and **exits 0 having written nothing**.
- **Verify the PDF by file size, not by return code**: success is
  `pdf.exists() and pdf.stat().st_size > 0`. See above for why the return code lies.
- **Always pass `encoding="utf-8"` when writing the HTML** (R11). The nbconvert HTML is ~308 KB
  of UTF-8 with non-cp1252 characters; `Path.write_text(body)` defaults to
  `locale.getpreferredencoding()` and raises on Windows. Same reflex for any log line carrying
  a `CellExecutionError` message — strip ANSI with
  `re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", ...)` and coerce to ASCII first.
- **The Proactor asyncio `RuntimeWarning` is benign and must NOT be "fixed"** (R11). Every
  nbclient execution on Windows emits `Proactor event loop does not implement add_reader ... Use
  asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`. **Do not follow that
  advice**: setting a global event-loop policy from library code is a side effect on any host
  embedding `hercule.controller`, and `WindowsSelectorEventLoopPolicy` removes asyncio
  subprocess support — exactly what the optional Playwright engine needs. If it pollutes
  output, suppress narrowly with `warnings.catch_warnings()` around `preprocess`.
- **LaTeX PDF export needs pandoc too, not just pdflatex** (R6). Verified failure:
  `PandocMissing: Pandoc wasn't found`. nbconvert routes every markdown cell through pandoc, so
  `--to pdf` needs a multi-GB TeX install *plus* a separate binary on every platform — and
  still gives the worst fidelity, because the LaTeX templates prefer `text/plain` and degrade
  every DataFrame to monospace text. Not used.
- **`--to webpdf` fails out of the box too**: `RuntimeError: No suitable chromium executable
  found`. Playwright pins an exact chromium revision, so unrelated cached browsers do not
  satisfy it. Opt-in extra only.
- **`ExecutePreprocessor` defaults are all traps** (R8): `timeout=None` (unbounded per cell),
  `interrupt_on_timeout=False` (loses the offending line in the traceback), `record_timing=True`
  (kills byte-identical regeneration). Set `timeout=1800`, `startup_timeout=120`,
  `interrupt_on_timeout=True`, `record_timing=False`, and keep `allow_errors=False`.
- **nbconvert does not save the notebook when it raises** — write it in the failure branch or
  the traceback is lost. Catch three unrelated types: `CellExecutionError`, `CellTimeoutError`
  (**not** a subclass — its MRO goes `TimeoutError → OSError`) and `DeadKernelError` (a plain
  `RuntimeError`).
- **Never `raise` from a middle notebook cell** — it blocks every cell below it. All degradation
  branches (PCA not applicable, no evaluation phase, constant performance colour scale) render
  as printed text.
- **MAX_PATH** (R11): the output tree already approaches 260 characters before a filename.
  Write intermediates (HTML, temp browser profile) to a short temp dir and move the finished PDF
  into place.
- **Paths baked into generated Python must be raw strings** (`Path(r"...")`) or
  forward-slashed — a bare `"C:\Users\..."` is both a `\U` escape error and a ruff violation.
- **Expectation-setting on PCA** (R4): Hercule grids are full cartesian products, so
  hyperparameter columns are orthogonal by construction and the correlation matrix is
  near-identity — PC1+PC2 typically capture only ~2/p of the variance. That is not a bug in the
  implementation; it is why FR-018 requires printing explained variance prominently.

---

## 5. Linting

```bash
uv run ruff check . --fix && uv run ruff format .
```

Line length is 120. The repo is ruff-clean — a new violation is yours. Two notes:

- `ruff format` also reformats Python snippets inside `.md` files, including the snippets in
  this document and in `specs/`. Check `git diff` after formatting so you do not ship an
  unrelated reflow.
- Ruff does **not** lint `.j2` template bodies. That is precisely why the table, selection and
  PCA logic lives in `reports/run_table.py`, `reports/selection.py` and `reports/pca.py` rather
  than inside the templates — code in a template is neither linted nor unit-testable.

```bash
uv run pytest tests/reports -m "not slow"     # new unit tests
uv run pytest                                 # full suite before the PR
```
