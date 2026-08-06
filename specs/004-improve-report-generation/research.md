# Phase 0 Research: Improved Experiment Report Generation

**Feature**: `004-improve-report-generation` | **Date**: 2026-07-28

All findings below were verified against this repository and this machine (nbconvert 7.17.1,
nbclient 0.11.0, nbformat 5.10.4, jupytext 1.19.5, playwright 1.61.0 probed in a scratch
environment). Where a claim was tested, the observed output is quoted.

---

## R1. Reverse impact of changing `reports/`

**Decision**: the change is contained; only one production call site must widen.

**Findings**:

- `spaghetti-compass explore src/hercule/reports/__init__.py -c src/` reports
  `circularDependencies: []`. Outbound imports: `json`, `logging`, `pathlib`, `typing`,
  `base64`, `io`, `matplotlib.pyplot`, `pandas`, `jinja2`, and `hercule.{config,
  environnements, models, run, supervisor, models.epoch_result}`.
- The installed `spaghetti-compass` build exposes only `explore` (no `impact` subcommand),
  and it cannot resolve `hercule.*` absolute imports onto the `src/` layout
  (`internalNodes: 1`). The reverse set was therefore established by search instead.
- Reverse dependents of `hercule.reports` (authoritative):
  - `src/hercule/controller/__init__.py:21,198` — the only production dependent.
  - `src/hercule/reports/cli.py:7,29` — an unregistered, unreachable Click group (dead).
  - `src/hercule/reports/example_usage.py:13,33` — standalone script with a hardcoded path.
  - `src/hercule/reports/README.md:17,24` and `CLAUDE.md:127` — documentation.
- `src/hercule/cli/main.py:211` consumes the controller wrapper and echoes a single path.

**Rationale**: a one-dependent blast radius means the module can be restructured freely as
long as `generate_report`'s contract is updated deliberately, in one place, with the CLI
message updated alongside.

**Constitution impact**: none. `reports/` and `controller/` are absent from the Root Class
Registry (`RLModel`, `TDModel`, `BaseConfig`, `HyperParamsBase`, `HerculeConfig`,
`EpochResult`, `Runner`, `Supervisor`). No amendment is triggered. The widening of
`generate_experiment_report`'s return type is nonetheless a public controller API change and
must be called out in the PR description.

---

## R2. Source of truth for hyperparameters

**Decision**: read `run_info.json → model_hyperparameters` directly. Do not parse directory
signatures, and do not change how training records results.

**Findings**:

- `run_info.json` already stores `model_hyperparameters` as a flat `name → scalar` mapping,
  with full (unabbreviated) names: `{"learning_rate": 0.0001, "discount_factor": 0.95,
  "epsilon": 1.0, "epsilon_decay": 0.0005, "epsilon_min": 0.01, "replay_buffer_size": 10000,
  "batch_size": 32, "step_modulo": 1, "weight_decay": 0.0, "seed": 42}`.
- A documented hazard was tested and **does not materialise**: `epsilon` is mutable state
  written back on each update, so it could have held the final decayed value. For
  `frozenlake_4x4/.../simple_q_learning/dis_fac_0.95__eps_1.0__eps_dec_0.005__...` (5000
  episodes, decay 0.005) the stored value is `epsilon: 1.0` — the **configured** value,
  matching the directory signature. Verified across three runs.

**Rationale**: the stored mapping is complete, uses readable names, and needs no decoding of
the 3-letter signature abbreviations. Avoiding a `Runner` change also keeps the Root Class
Registry untouched.

**Alternatives considered**: parsing the signature directory name (lossy — `lea_rat` must be
mapped back to `learning_rate` via the model registry); persisting hyperparameters in a new
file (would require re-training and would touch `Runner`).

---

## R3. Not loading `model.json`

**Decision**: never read `model.json` when building the run table; take the model name from
the directory layout.

**Findings**: a single report group holds **~55 MB across 109–135 `model.json` files**
(`q_network_state_dict`); `outputs/` holds 110.6 MB of them in total. The current comparative
template parses all of it per run and cell 18 prints it in full, skipping only `['q_table']` —
so DQN weights are dumped verbatim into the document. The model family is already the run
directory's parent name, which `ExperimentData` itself relies on today (`__init__.py:198`).

**Rationale**: satisfies FR-007 and SC-010, and removes the dominant avoidable cost in
loading a large group.

---

## R4. PCA: numpy SVD, not scikit-learn

**Decision**: implement PCA with `numpy.linalg.svd`. Do **not** add `scikit-learn`.

**Findings**:

- `uv.lock` locks 98 packages; `scikit-learn`, `scipy`, `sklearn`, `joblib` and
  `threadpoolctl` are all absent. `torch` does not pull `scipy` here. Adding scikit-learn
  means 4 new packages, with scipy wheels at ~35–60 MB.
- `sklearn.decomposition.PCA` centers but does **not** scale, so correct hyperparameter PCA
  would need `StandardScaler` anyway, plus manual zero-variance handling, plus manual
  derivation of loadings from `components_` and `explained_variance_`. It is not less code.
- **Determinism is decisive.** sklearn's SVD sign convention is an unpinned internal that has
  already changed once (scikit-learn#28826, PCA output changed in 1.5). Generated notebooks
  are executed by end users on their own machines with their own sklearn version, so
  delegating the sign rule makes plot orientation a function of the reader's environment.
  FR-014/SC-009 require deterministic regeneration.

**Recipe** (each property verified numerically):

1. Build the matrix per model family. Columns = hyperparameter keys sorted alphabetically
   (fixed order — the sign rule and tie-breaking depend on it). Keep a key only if its value
   is `int | float` in every run; exclude `bool` explicitly, since `isinstance(True, int)` is
   `True`.
2. Standardize on the **correlation** matrix, not covariance: `learning_rate` ~1e-4 against
   `replay_buffer_size` ~1e4 means covariance PCA would return "PC1 = replay_buffer_size" as
   a pure units artifact.
   `sd = X.std(axis=0, ddof=1)` is used to standardize the columns that are kept; the keep
   decision itself uses the exact range, not `sd`: `column_range = np.ptp(X, axis=0)`,
   `column_scale = np.maximum(np.abs(X).max(axis=0), 1.0)`,
   `keep = column_range > 1e-12 * column_scale`. Drop zero-variance columns **before** the
   SVD — standardizing them yields `nan`/`inf`. `ddof=1` makes the loading formula exactly
   Pearson correlation; the explained-variance ratio is invariant to that choice.
   **Corrected during implementation**: this section originally prescribed `keep = sd > 0`
   ("exact test, no epsilon"). That is wrong on real data — measured on
   `outputs/frozenlake_4x4` / `deep_q_learning` (108 runs), `epsilon_decay` is a genuinely
   constant column (single raw value `0.0005`, exact `np.ptp` of `0.0`), yet
   `column.std(ddof=1)` returns `2.1785135109378093e-19`, not `0.0`, because `std` accumulates
   floating-point rounding error in its sum of squares for a small repeated value. `sd > 0` is
   therefore `True` and silently retains a constant column, diluting the explained-variance
   ratio (1/6 instead of 1/5 per component for this family) and breaking the
   loadings/Pearson-correlation identity. `np.ptp` is exact for identical values and is immune
   to this failure mode; the relative-tolerance term additionally rejects columns whose spread
   is pure numerical noise rather than a real grid axis.
3. `U, S, Vt = np.linalg.svd(Xs, full_matrices=False)` — prefer SVD of `Xs` over `eigh` of the
   correlation matrix, which squares the condition number and can return small negative
   eigenvalues that produce `nan` under `sqrt`.
4. **Pin the sign** so plots do not flip between runs: make the largest-magnitude entry of
   each row of `Vt` positive, and apply the same signs to `U` or the scores desynchronise from
   the components. This matches sklearn's `svd_flip(..., u_based_decision=False)` convention,
   so cross-checks agree — but the project owns it, so it cannot drift.
5. Outputs: `scores = U * S`; `evr = S**2 / sum(S**2)`;
   `loadings = Vt * (S / sqrt(n - 1))[:, None]`. Verified: `scores == Xs @ Vt.T`; `evr` equals
   sklearn's; `loadings[i, j]` equals `corrcoef(Xs[:, j], scores[:, i])` to 1e-12; squared
   loadings per feature sum to 1 across all components, so the first two give the communality
   — the fraction of that hyperparameter captured by the 2-D plot, worth displaying.

**Residual non-determinism, stated honestly**: tied singular values make the eigenvector
*basis* non-unique, so BLAS/LAPACK differences can rotate the plot. A 2-column case producing
singular values `[1.732051, 1.732051]` was constructed. The notebook must state this; the
project must not promise bit-identical figures across platforms. SC-009 is scoped to series
selection, not figure geometry.

**Expectation-setting — important**: Hercule grids are full cartesian products, so
hyperparameter columns are orthogonal by construction and the correlation matrix is
near-identity. A simulated 4-varying-column Hercule-style grid gave
`evr = [0.303, 0.253, 0.241, 0.202]` — PC1+PC2 capture only ~56%, roughly what random
directions would give. The 2-D projection of a complete factorial design is mostly a lattice,
not a discovery. This is why FR-018 (print explained variance prominently) matters: it stops
readers over-interpreting the geometry. PCA becomes genuinely informative on ragged or partial
grids and resumed subsets, where the design is unbalanced.

**Degenerate shapes**: after centering, `rank(Xs) <= min(n - 1, p_kept)`, but
`full_matrices=False` still returns `min(n, p)` singular values — an `n=3, p=8` probe returned
`evr = [0.618, 0.382, 0.000]`, where the trailing zero component is noise with meaningless
loadings. Truncate to `k = min(n - 1, p_kept)`. Hard guard for the 2-D plot: `p_kept >= 2`
**and** `n >= 3`; with `p_kept == 1` there is no PC2 and `scores[:, 1]` raises `IndexError`.
Degrade with `if/elif/else` in the emitted cell, never `raise` — an exception in a middle cell
blocks every cell below it.

---

## R5. Performance encoding on the projection

**Decision**: `viridis` sequential colormap with a labelled colorbar, and an explicit
constant-value branch.

**Findings**: with a constant `c` array, matplotlib autoscales `clim` to `(0.7, 0.7)`,
`Normalize` maps every point to `0.0` — the **darkest** colour, reading as "all runs are the
worst" — and the colorbar renders **fabricated ticks 0.62 … 0.78** for a range that does not
exist. Silently misleading, no exception. This is not hypothetical: many FrozenLake runs score
a mean reward of 0.

**Consequences**: guard on `np.ptp(finite) == 0` and fall back to a single fixed colour with
no colorbar and an explicit annotation; filter `nan`/`inf` before computing `vmin`/`vmax`, or
one unfinished run collapses the scale. Use `edgecolors="black"` so light-yellow best
performers stay visible on white. Label axes with the variance (`PC1 (30.3% of variance)`) and
set `aspect="equal"` so plotted distance is honest.

---

## R6. Notebook → PDF pipeline

**Decision**: `jupytext.read` → `ExecutePreprocessor` → `HTMLExporter` +
`TagRemovePreprocessor` → print the HTML with a system Chromium-family browser
(`--headless=new --print-to-pdf`), falling back to an optional `WebPDFExporter`, and finally
to "HTML kept, PDF skipped with a reason".

**Rejected — (a) `--to pdf` (LaTeX)**: verified failure `PandocMissing: Pandoc wasn't found`.
`pdflatex`/`xelatex` on PATH is **not sufficient** — nbconvert routes every markdown cell
through pandoc. So it needs a multi-GB TeX install *plus* a separate binary, on every
platform, and still yields the worst fidelity here: the LaTeX templates prefer
`text/latex`/`text/plain`, so every pandas DataFrame degrades to monospace text.

**Rejected as default — (b) `--to webpdf`**: verified failure `RuntimeError: No suitable
chromium executable found on the system`. The cached browsers do not help — `playwright
1.61.0` pins chromium revision 1228 while the cache holds `chromium-1140` and `chromium-1234`
(left by the Node-based MCP). A ~170 MB download plus a 36 MB wheel, version-locked. Kept as
an **opt-in extra** for headless CI where no browser exists.

**Rejected — (d) PdfPages/reportlab**: would reimplement the whole document as a second
renderer that then drifts from the notebook, and discards the fact that the executed notebook
is already a complete structured representation.

**Verified working (c)**: printing a tag-filtered `HTMLExporter` output with the pre-installed
Edge returned `returncode: 0`, wrote 78,365 bytes, and extracted text confirmed exactly the
FR-024/FR-025 contract — narrative heading present; the `remove_input` cell's printed output
present **with no source**; the `remove_cell` cell entirely absent; an untagged cell's code
retained; a pandas DataFrame rendered as a real HTML table; the chart embedded as an image.

**Required flags** (each earned):

- `--user-data-dir=<fresh temp dir>` is **mandatory, not hygiene**: if Edge or Chrome is
  already running — near-certain on a dev workstation — a bare `--print-to-pdf` hands off to
  the running instance and **exits 0 having written nothing**. Verify
  `pdf.exists() and pdf.stat().st_size > 0` rather than trusting the return code.
- `--host-resolver-rules=MAP * ~NOTFOUND` plus `--virtual-time-budget=10000`: the default
  `lab` template references `cdnjs.cloudflare.com` for MathJax, RequireJS and Mermaid. CSS is
  inlined (308 KB), so blocking the network costs nothing for these reports and makes printing
  deterministic instead of dependent on a CDN or a hanging corporate proxy.
- `html_path.as_uri()` for the URL — verified to produce `file:///C:/...` correctly on Windows.
- Print CSS (`@page { size: A4; margin: 14mm }`, `.jp-Cell { break-inside: avoid }`,
  `img { max-width: 100% }`) so charts do not straddle pages.

---

## R7. Cell filtering

**Decision**: bake tags into the `# %%` markers at generation time; remove with
`TagRemovePreprocessor`.

**Verified jupytext percent syntax** — the delimiter grammar is
`# %% Optional title [cell type] key="value"`:

| literal generated line | resulting `cell.metadata` |
|---|---|
| `# %% tags=["remove_input"]` | `{"tags": ["remove_input"]}` |
| `# %% tags=["remove_cell"]` | `{"tags": ["remove_cell"]}` |
| `# %% [markdown] tags=["remove_cell"]` | markdown cell with those tags |

`jupytext.writes(nb, fmt="py:percent")` re-emits `# %% tags=["remove_input"]` byte-identically,
so generator output and jupytext round-trips agree.

**Preprocessor traits**, and the one that matters:

| trait | effect |
|---|---|
| `remove_cell_tags` | drops source **and** outputs |
| **`remove_input_tags`** | **drops source, keeps outputs** ← exactly FR-025 |
| `remove_all_outputs_tags` | drops outputs, keeps source |

`remove_input_tags` keeping the output was **confirmed, not assumed**: against a probe,
`contains 'import matplotlib': False`, `contains 'MPL_BACKEND': True`, `contains base64 png:
True`, while a `remove_cell`-tagged cell's `SHOULD NOT APPEAR` was absent.

**Programmatic config** (no CLI flags): `c.TagRemovePreprocessor.enabled = True` is
**required — the default is `False`**; set `c.HTMLExporter.preprocessors =
["nbconvert.preprocessors.TagRemovePreprocessor"]`. The extra
`exporter.register_preprocessor(...)` line from the nbconvert docs is redundant. traitlets
config is per-class, so wiring the optional `WebPDFExporter` needs
`c.WebPDFExporter.preprocessors` set as well. `exclude_input_prompt` /
`exclude_output_prompt = True` removes the `In [12]:` gutters.

Tag names are the project's choice — nbconvert ships no defaults. Adopt nbconvert's own
documented names (`remove_cell`, `remove_input`, `remove_output`), noting they differ from the
Jupyter Book convention (`remove-input`, hyphens) and must match the config strings exactly.

**Mapping for FR-025**: imports and model reconstruction → `remove_cell`; discovery and
loading → `remove_input` when their printed output is informative ("loaded 218 runs, skipped
2"); analysis code stays untagged. Note `exclude_input = True` is too blunt — it would strip
every input, contradicting FR-025's retention of informative analysis code.

---

## R8. Execution settings

**Decision**: explicit timeouts, `allow_errors=False`, `interrupt_on_timeout=True`, progress
via constructor hooks.

**Findings — every default is a trap**: `ExecutePreprocessor()` ships `timeout: None`
(per-cell, unbounded — a hung cell hangs `hercule` forever), `startup_timeout: 60`,
`allow_errors: False`, `interrupt_on_timeout: False`, `record_timing: True`.

- `timeout=1800` per cell (the run-loading cell is the long one), `startup_timeout=120` for a
  cold `torch`/`gymnasium` import in the kernel.
- `interrupt_on_timeout=True` is materially better for diagnostics. Verified on a 10 s cell
  with `timeout=2`: `True` raises `CellExecutionError` with a traceback that **points at the
  offending line**; `False` raises `CellTimeoutError` with a source preview only.
- **`allow_errors` stays `False`.** FR-031/SC-005 make "the report executes clean" a tested
  property; `True` would bury a broken report as a red `error` output silently embedded in the
  PDF. Verified: with `False` execution stops at the bad cell and later cells never run; with
  `True` they continue. FR-026 is satisfied at the *command* level instead — catch, write the
  partial notebook for debugging, report the reason, exit 0.
- **nbconvert does not save the notebook when it raises** — write it in the failure branch or
  the traceback is lost.
- Catch three unrelated types: `CellExecutionError`, `CellTimeoutError` (**not** a subclass —
  MRO goes `TimeoutError → OSError`), and `DeadKernelError` (a plain `RuntimeError`).
- A cell legitimately allowed to fail should be tagged `raises-exception` rather than flipping
  the global; verified that with `allow_errors=False` such a cell raises nothing, keeps its
  error output, and the following cell still runs.
- Progress hooks are traitlets `Callable` **traits, not overridable methods**, and are invoked
  with **keyword** arguments: pass `on_cell_start=lambda cell=None, cell_index=None, **_: ...`
  to the constructor. This is how SC-008's 30-second progress requirement is met.
- `record_timing=False` if the executed notebook is persisted, or per-cell ISO timestamps make
  byte-identical regeneration impossible.

**Two free wins**: no kernel registration is needed —
`KernelSpecManager().find_kernel_specs()` returned `['python3']` in a bare venv with only
`ipykernel` installed and no `ipykernel install` ever run, and it is the same interpreter, so
`import hercule` resolves. And no `%matplotlib inline` is needed — the kernel auto-selects
`module://matplotlib_inline.backend_inline` and `plt.show()` yields inline PNGs. **Do not add
magics**: they would break running the report as a plain script.

---

## R9. Locating the report's data directory

**Decision**: search a candidate list anchored on a file that must exist, preferring
`__file__` when defined and `Path.cwd()` otherwise; always pass
`resources["metadata"]["path"]` when executing.

**Findings**: `Path(__file__)` is **confirmed broken** in a kernel — `HAS_FILE: False`, and
`sys.argv[0]` is `.../site-packages/ipykernel_launcher.py`, so any `__file__`-derived path
resolves into site-packages rather than the run directory. The current templates
(`report_template.py.j2:41`) therefore cannot execute under nbconvert at all.

nbclient sets the working directory to exactly `resources["metadata"]["path"]` — verified. If
omitted it inherits the shell's cwd, so it must always be passed. Jupyter and JupyterLab also
start the kernel in the notebook's directory, which makes `Path.cwd()` correct for both the
framework-executed and manually-opened contexts, while `__file__` covers `python report.py`.

The candidate search must **verify** rather than hope: test each candidate for an anchor file
(`environment.json` for an individual report; a small generator-written manifest for a
comparative report, since the env-params level has no naturally occurring file). Bake the
generation-time absolute path as a last-resort fallback — the templates already receive it —
to rescue a notebook opened with a stray cwd.

**Caveat for the plain-script context**: with `__file__` defined, matplotlib picks a GUI
backend and `plt.show()` **blocks**. Guard with `if "get_ipython" not in globals():
matplotlib.use("Agg")` (verified: `get_ipython` is present inside the kernel, so the guard
discriminates correctly).

---

## R10. Dependencies

**Decision**:

| package | constraint | group | why |
|---|---|---|---|
| `jupytext` | `>=1.16,<2` | main | `.py` percent → `NotebookNode`; also what a user needs to open the `.py` as a notebook (FR-031) |
| `nbformat` | `>=5.10,<6` | main | `nbformat.write` for the executed/failed notebook; declare rather than lean on it transitively |
| `nbconvert` | `>=7.16,<8` | main | `ExecutePreprocessor`, `TagRemovePreprocessor`, `HTMLExporter` |
| `ipykernel` | `>=7.1,<8` | **dev → main** | `hercule report` is an end-user command and FR-023 makes PDF production part of it; without a kernel there is no execution |
| `pypdf` | `>=5.1,<6` | dev | SC-006 is only testable by reading the PDF back — pages, extracted text, image counts |
| `playwright` | `>=1.49,<2` | optional extra `pdf` | engine 2, for headless CI with no system browser |

**Not added**: `scikit-learn` (see R4), `seaborn` (matplotlib suffices, and it would become
another main dep for end users), `papermill` (no parameter injection needed), `nbclient`
explicitly (nbconvert pins it; its exception classes are re-exported).

**No system dependency is added** — no pandoc, no TeX, no GTK/Pango. `weasyprint` was also
ruled out: on Windows it still needs a separately installed GTK runtime, and Windows is this
project's primary dev platform.

Note `[tool.uv] default-groups = "all"` governs *groups*, not *extras*, so the `pdf` extra
stays out of `uv sync` unless `--extra pdf` is passed — which is the intent. FR-026's
remediation string is `uv sync --extra pdf && uv run playwright install chromium`.

---

## R11. Windows-specific hazards

**Decision**: sanitise all exception text, force UTF-8 on every write, and do **not** change
the asyncio event loop policy.

**Findings**:

- Every nbclient execution emits `RuntimeWarning: Proactor event loop does not implement
  add_reader ... Use asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`. It is
  **benign** — pyzmq registers a selector thread and execution worked correctly. **Do not
  follow the warning's advice**: setting a global event-loop policy from library code is a
  side effect on any host embedding `hercule.controller`, and `WindowsSelectorEventLoopPolicy`
  removes asyncio subprocess support, which is exactly what the optional Playwright engine
  needs. Suppress narrowly with `warnings.catch_warnings()` around `preprocess` if it pollutes
  output.
- **The emoji incident has a second surface.** `CellExecutionError` messages carry arbitrary
  user text *and* raw ANSI escapes. Reproduced: printing the value for a cell raising
  `ValueError("boom: café ± →")` died with `UnicodeEncodeError: 'charmap' codec can't encode
  character '\u2192'`. `harden_output_streams()` neutralises it for the CLI, but the string may
  also be logged — strip ANSI with `re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", ...)` and coerce to
  ASCII before logging.
- **`Path.write_text(body)` for the HTML will raise on Windows**: the nbconvert HTML is 308 KB
  of UTF-8 containing non-cp1252 characters, and `write_text` defaults to
  `locale.getpreferredencoding()`. Always pass `encoding="utf-8"`.
- **MAX_PATH**: the output tree is
  `{base}/{config}/{env}/{env_sig}/{model}/{model_sig}/` with signatures like
  `lea_rat_0.0001__bat_siz_32__...`, approaching 260 characters before a filename is appended.
  Write intermediates (HTML, temp profile) to a short temp dir and move the finished PDF into
  place.
- Paths baked into generated Python must be raw strings (`Path(r"...")`) or forward-slashed —
  a bare `"C:\Users\..."` is both a `\U` escape error and a ruff violation.

---

## R12. Verified during implementation: the Agg guard is load-bearing for SC-006

**Decision**: the `Agg` backend must be selected **only** outside a kernel. The guarded form is
mandatory, not stylistic.

**Findings** (measured on this machine during Phase 6):

- An **unconditional** `matplotlib.use("Agg")` in a report cell makes `plt.show()` a silent
  no-op: the figure never becomes a `display_data` output, so it is absent from the notebook,
  absent from the exported HTML (0 `<img>` tags), and absent from the PDF. **No error is
  raised** — the charts simply vanish.
- This produced a false-positive in an early probe: the check "`image/png` in html" matched a
  CodeMirror **CSS** artifact (`.cm-trailingspace` background-image), not the figure. Asserting
  on `image/png` appearing anywhere in the HTML is therefore an invalid test for SC-006.
- With the guarded form `if "get_ipython" not in globals(): matplotlib.use("Agg")`, the same
  report yields a 27 KB base64 PNG output, exactly 1 `<img>` tag in the HTML, and a PDF
  containing 1 embedded image — verified with `pypdf` (`sum(len(p.images) for p in reader.pages)`).

**Consequences for the tests**: SC-006 must be asserted by counting **embedded images in the
PDF** (`pypdf`), never by string-matching `image/png` in the HTML. The full chain was verified:
tagged `.py` → executed notebook (1 figure) → `TagRemovePreprocessor` HTML (1 `<img>`, no
import listing, informative output retained) → headless Edge → 52 KB PDF, 1 page, 1 image,
narrative and table text intact.

**Also confirmed**: `shutil.which()` finds **no** browser on this machine — neither `msedge` nor
`chrome` is on `PATH`. Both exist only at their standard install paths
(`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`,
`C:\Program Files\Google\Chrome\Application\chrome.exe`). The standard-paths fallback in
discovery is therefore required, not a nicety.

---

## Open risks carried into the plan

1. **PCA is near-degenerate on full factorial grids** (R4). Delivered as specified, with
   explained variance printed prominently and a stated caveat. Not a blocker; an honesty
   requirement.
2. **Figure geometry is not reproducible across platforms** when singular values tie (R4).
   SC-009 is read as covering series *selection*, which is fully deterministic.
3. **PDF depends on a system browser** being present. FR-026 makes this a graceful skip, and
   the HTML remains a shareable artifact. CI without a browser uses the `pdf` extra.
4. **Execution cost**: the largest single group parses ~65 MB of JSON inside the kernel (135
   runs), and one `dq_cartpole` invocation covers ~211 MB across two groups. Mitigated by
   loading only the two needed metric fields per episode and never touching `model.json` (R3),
   with progress hooks for SC-008.
