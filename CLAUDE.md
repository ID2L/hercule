# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hercule is a framework for benchmarking Reinforcement Learning and Deep RL algorithms on
[Gymnasium](https://gymnasium.farama.org/) environments. A YAML file declares *(environments × models ×
hyperparameter grids)*; the framework expands that into a cartesian product of runs, trains each one, saves
metrics/weights into a deterministic directory tree, and can generate Jupyter reports from the results.

Package lives in `src/hercule` (import as `hercule.*`, **never** `src.hercule.*` — banned via ruff `banned-api`).
`pyscaf/` is vendored scaffolding tooling (pdoc wrappers only), not part of the framework.

**Governance — read this before touching a base class.** `AGENTS.md` defines a *Root Class Registry* (`RLModel`,
`TDModel`, `BaseConfig`, `HyperParamsBase`, `HerculeConfig`, `EpochResult`, `Runner`, `Supervisor`). Any *semantic*
change to their public API (abstract method added/removed/renamed, signature change, lifecycle contract, `@final`
methods, `ClassVar` subclasses depend on) **must** trigger a review of `.specify/memory/constitution.md`, a version
bump if amended, and a "Constitution Impact" section in the PR. `AGENTS.md` is the shared human/agent reference and
overlaps this file; the sections below add the operational detail it does not cover.

## Commands

```bash
uv sync                                  # install deps (dev group included: default-groups = "all")

uv run hercule learn experiments/frozenlake_4x4.yaml       # train + test from a config
uv run hercule learn <config.yaml> -o <dir> -vv            # override output dir, DEBUG logging
uv run hercule play <model.json> <environment.json>        # replay a trained model (Ctrl+C to stop)
uv run hercule play <model.json> <environment.json> --no-render
uv run hercule report outputs/frozenlake_4x4              # individual or comparative report (auto-detected)

uv run pytest                            # 50 tests, ~25s
uv run pytest tests/config/test_config_expansion.py::TestConfigExpansion::test_expand_model_variants
uv run pytest -m "not slow"              # markers: slow, integration, unit (--strict-markers is on)

ruff check . --fix && ruff format .      # line-length 120
uv run gen-doc                           # pdoc -> docs/
uv run serve-doc
```

Ready-made configs live in `experiments/`; results land in `outputs/`.

## Architecture

### Execution pipeline

`cli/main.py` (Click, presentation only) → `controller/` (frontend-agnostic business API, also usable from a web
API; provides `CancellationToken`) → `supervisor/` (double loop over environments × models) → `run.Runner` (epoch
loop + persistence) → `models.RLModel.run_epoch()` (the algorithm).

One **epoch = one episode**. `Supervisor.execute_learn_phase()` runs all combinations to `learn_max_epoch`, then
`execute_test_phase()` re-loads each model and runs `test_epoch` episodes with `train_mode=False`.

### Config expansion and output layout (the core mechanic)

`load_config_from_yaml()` always calls `HerculeConfig.expand_variants()`: any hyperparameter whose value is a
**list** is expanded into the cartesian product of variants (`BaseConfig.expand_variants`). So a config with
`learning_rate: [1e-4, 2.5e-4, 1e-3]` and `batch_size: [32, 64]` produces 6 independent runs.

Each run gets a unique directory from `HerculeConfig.get_directory_for()`:

```
{base_output_dir}/{config.name}/{env_name}/{env_signature}/{model_name}/{model_signature}/
    environment.json   # gym.make kwargs, from environnements.save_environment()
    model.json         # weights/Q-table + "model_name" key, from RLModel.save()
    run_info.json      # epoch counters + learning_metrics/testing_metrics, from Runner.save()
```

The signature is `BaseConfig.get_hyperparameters_signature()`: first 3 letters of each word of the param name +
its value, sorted alphabetically (`lea_rat_0.0001__bat_siz_32__…`). It raises if a value is still a list, i.e.
`expand_variants()` must run first.

**Runs are resumable and this is by design**: `Runner.load(directory)` restores epoch counters from
`run_info.json` and `model.load(directory)` restores weights, so re-running the same config continues from where
it stopped (`Runner.learn` iterates `range(learning_ongoing_epoch, max_epoch)`). Increase `learn_max_epoch` to
train further; delete the directory to start clean. `save_every_n_epoch` checkpoints mid-training.

### Models

`models/__init__.py` defines `RLModel(BaseConfig, ABC, Generic[HyperParamsType])` — a Pydantic model, so `env` is
a field and internal state uses `PrivateAttr`. Subclasses must implement `act`, `run_epoch`, `predict`, `_export`,
`_import`; `save`/`load`/`check_environment_or_raise` are `@final` (JSON via `model.json`).

Hyperparameters are dual-representation:
- typed: a `HyperParamsBase` subclass declared as `hyperparams_class: ClassVar`, reachable via
  `self.get_hyperparameters()` (autocompletion, defaults) — **prefer this inside algorithms**;
- generic: `self.hyperparameters: list[HyperParameter]`, kept in sync so it can be serialized/signed.
`configure()` merges provided values over defaults and populates both. Mutable state such as decaying `epsilon`
lives in the typed hyperparameters and is written back to both on each update (see `TDModel.run_epoch`).

Model discovery is **directory-based**: `get_available_models()` scans `src/hercule/models/*/`, imports each
package and registers every `RLModel` subclass under its `model_name` `ClassVar`. `create_model("simple_sarsa")`
resolves through that registry, and YAML `models[].name` is the same key.

Current hierarchy:
- `td_models/` — abstract `TDModel` (Q-table, ε-greedy, epoch loop); subclasses only implement `update()`:
  `simple_q_learning/` (off-policy), `simple_sarsa/` (on-policy). Requires **discrete** action *and* observation
  spaces (`environnements/spaces_checker.py`).
- `deep_q_learning/` — DQN (PyTorch): `QNetwork` picks MLP for 1-D observations / CNN for 3-D, plus
  `ExperienceReplayBuffer`.
- `dummy/` — random baseline, works on any space.

### Adding a model

1. `src/hercule/models/<name>/__init__.py`, subclass `RLModel[MyHyperParams]` (or `TDModel` for tabular TD).
2. Set `model_name: ClassVar[str] = "<name>"` and `hyperparams_class: ClassVar = MyHyperParams`.
3. Implement the abstract methods, plus `load_from_dict(model_data)` if the model must be usable with
   `hercule play` (see gotcha below).
4. Reference `<name>` in a YAML under `models:` — no registration code needed.

### Environments

`environnements/` wraps Gymnasium: `EnvironmentFactory` (validated `gym.make` + cache keyed on name+kwargs),
`EnvironmentRegistry` (static registry queries with similar-name suggestions), `EnvironmentInspector` (metadata
from `env.spec`), and `save_environment`/`load_environment` which round-trip an env through JSON by keeping only
the keys accepted by `gym.make`. That JSON is what `hercule play` consumes to rebuild the env with
`render_mode="human"`.

### Reports

`reports/generate_report()` auto-detects: a directory holding the three JSON files → individual `report.py`; a
parent directory → recursive search (`MAX_DEPTH`), grouped by environment+env-params, producing
`comparative_report.py`. Output is a Jupytext-format `.py` (`# %%` cells) rendered from Jinja2 templates in
`reports/templates/`, meant to be opened as a notebook.

## Conventions

- Code and comments in **English** (some legacy French comments/docstrings remain in `run/` and `supervisor/`).
- Python 3.10+ syntax: `X | Y` unions, including in `isinstance()`; avoid `typing.Union`/`Optional`/`Any`; be
  explicit rather than falling back to `Any`.
- Pydantic v2 with `@field_validator` + `@classmethod` (never `@validator`). Structured data is a Pydantic model,
  not a dict.
- Google-style docstrings on public functions/classes.
- Ruff selects `B, C4, E, F, N, W, I, UP, TID, TC, PLC, PLE, PLW`; isort `known-first-party = ["hercule"]`,
  2 blank lines after imports; relative imports to parents are banned.
- To exercise behaviour manually, drive the **CLI** with a config in `experiments/` — do not add ad-hoc test
  scripts at the repo root.
- Tests use `tests/conftest.py` fixtures (`temp_test_dir`, `change_to_temp_dir`, Click `runner`) and YAML fixtures
  in `tests/fixtures/`.

## Gotchas

- **`.cursor/rules/*.mdc` and `src/hercule/run/README.md` are partly stale.** They document a `TrainingRunner` /
  `ModelExecutor` / `RunManager` API and a `benchmark/` module that no longer exist — the real orchestration is
  `Supervisor` + `Runner`. Their *style* and *design-rationale* sections are still authoritative; their code
  samples are not. Check the source before following them.
- `load_from_dict()` is used by `controller.play_interactive()` but is **not** declared on `RLModel`; it is
  implemented per model (`TDModel`, `DummyModel`, DQN). A new model without it breaks `hercule play` only.
- `TDModel.configure()` returns `False` (instead of raising) when the env spaces are not discrete, and
  `Supervisor` ignores the return value — pairing a tabular model with e.g. `CartPole-v1` fails later with an
  unrelated error. Validate the pairing in the config.
- `EnvironmentManager.load_environment()` still contains debug `print()` calls; `Supervisor` bypasses that class
  and uses `EnvironmentFactory` directly.
- Model persistence is JSON, so `save_every_n_epoch` on a large DQN writes big files; tune it per experiment.
- Speckit workflow: feature specs live in `specs/`, commands in `.cursor/commands/speckit.*.md`.
