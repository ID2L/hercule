# AGENTS.md — Hercule Development Guidance

This file provides runtime guidance for AI agents (and human developers)
working on the Hercule codebase. It complements the project constitution at
`.specify/memory/constitution.md`.

## Project Overview

Hercule is a reinforcement learning framework that lets you **define, train,
and benchmark RL algorithms in batch** against a collection of Gymnasium
environments. Experiments are fully described by a single YAML configuration
file; the framework handles the Cartesian product of
(models × environments × hyperparameter variants), training, evaluation, model
persistence, and automated report generation.

## Architecture at a Glance

```text
src/hercule/
├── config/           # YAML parsing, Pydantic V2 models, ParameterValue type
├── environnements/   # Gymnasium registry, factory, inspector, manager
├── models/           # RLModel base class + algorithm sub-packages
│   ├── td_models/    #   └─ TDModel (tabular TD base class)
│   ├── simple_q_learning/
│   ├── simple_sarsa/
│   ├── deep_q_learning/
│   └── dummy/
├── run/              # Runner (train/test loop), result storage
├── supervisor/       # Orchestrates learn & test phases across configs
├── controller/       # Business-logic entry points (learn, play, report)
├── reports/          # Report generation: run table -> Jinja2/Jupytext .py -> execute -> HTML/PDF
└── cli/              # Click CLI — thin wrapper over controller
```

## Root Class Registry — CRITICAL

The classes below are architectural foundations. **Any change to their public
API MUST trigger a review of the constitution** (`.specify/memory/constitution.md`)
and, if semantics change, a constitutional amendment.

| Class              | File                                       | Purpose                                   |
|--------------------|--------------------------------------------|-------------------------------------------|
| `RLModel`          | `src/hercule/models/__init__.py`           | Abstract base for ALL RL algorithms       |
| `TDModel`          | `src/hercule/models/td_models/__init__.py` | Abstract base for tabular TD algorithms   |
| `OffPolicyReplayModel` | `src/hercule/models/off_policy/__init__.py` | Abstract base for replay-based off-policy algorithms |
| `ContinuousActorCriticModel` | `src/hercule/models/continuous_actor_critic/__init__.py` | Abstract base for continuous actor-critic algorithms |
| `BaseConfig`       | `src/hercule/config/__init__.py`           | Base Pydantic model for named configs     |
| `HyperParamsBase`  | `src/hercule/config/__init__.py`           | Base for typed hyperparameter classes     |
| `HerculeConfig`    | `src/hercule/config/__init__.py`           | Top-level experiment configuration        |
| `EpochResult`      | `src/hercule/models/epoch_result.py`       | Standardised epoch return type            |
| `Runner`           | `src/hercule/run/__init__.py`              | Training/testing execution engine         |
| `Supervisor`       | `src/hercule/supervisor/__init__.py`       | High-level phase orchestrator             |

### What counts as a "semantic change"?

- Adding, removing, or renaming an abstract method
- Changing a method signature (parameters, return type)
- Altering the lifecycle contract (e.g. `configure → train → save`)
- Modifying `@final` methods
- Changing `ClassVar` declarations that subclasses depend on

## How to Add a New RL Algorithm

1. Create `src/hercule/models/<algorithm_name>/__init__.py`.
2. Define a `HyperParamsBase` subclass for the algorithm's hyperparameters.
3. Create a class inheriting from `RLModel` (or `TDModel` for tabular TD
   algorithms).
4. Set `model_name: ClassVar[str]` and `hyperparams_class: ClassVar[...]`.
4b. Set `supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]]` — the
   `(observation_kind, action_kind)` pairs the algorithm accepts, using `SpaceKind`
   from `hercule.environnements.spaces_checker`. **There is no default**, and a
   concrete model that omits it is *not registered* by `get_available_models()`.
   That is deliberate: a permissive default would let a new model be paired with any
   environment and fail later with an unrelated error. `Supervisor` checks the pair
   before calling `configure()`, skips a mismatched combination with a message naming
   both the expected and the actual kinds, and carries on with the rest.
5. Implement all abstract methods: `act()`, `run_epoch()`, `predict()`,
   `_export()`, `_import()`.
6. If — and only if — the policy carries state ACROSS steps within an episode
   (observation stacking, recurrent policy), override `begin_episode()` to reset
   it. It is concrete with a default no-op, so a policy that depends only on the
   current observation ignores it. Skipping it when you do need it is silent:
   `predict()` sees one observation and cannot detect an episode boundary, so the
   state would span two episodes.
7. **No other file needs modification** — `get_available_models()` discovers
   the new sub-package automatically.

### RLModel Abstract Interface

```python
class RLModel(BaseConfig, ABC, Generic[HyperParamsType]):
    model_name: ClassVar[str]
    hyperparams_class: ClassVar[type[HyperParamsBase] | None]

    def configure(self, env, hyperparameters) -> bool: ...
    @abstractmethod def act(self, observation, training=False): ...
    @abstractmethod def run_epoch(self, train_mode=False) -> EpochResult: ...
    @abstractmethod def predict(self, observation): ...
    @abstractmethod def _export(self) -> dict: ...
    @abstractmethod def _import(self, model_data: dict) -> None: ...
    @final def save(self, path: Path) -> None: ...
    @final def load(self, path: Path) -> None: ...
    @final def check_environment_or_raise(self) -> gym.Env: ...
    def evaluate(self, num_episodes=10) -> dict[str, float]: ...
```

### TDModel Extension Points

`TDModel` extends `RLModel` for Q-table-based algorithms. Subclasses only
need to implement `update()`:

```python
@abstractmethod
def update(self, state, action, reward, next_state, next_action) -> None: ...
```

Everything else (Q-table init, epsilon-greedy, serialisation) is handled by
`TDModel`.

### OffPolicyReplayModel Extension Points

`OffPolicyReplayModel` (`src/hercule/models/off_policy/__init__.py`) extends `RLModel` for
algorithms that learn from a replay buffer — the episode loop, the buffer itself, frame stacking,
observation rescaling, device/seeding and checkpoint assembly are all handled by the ancestor.
Subclasses implement six abstract hooks:

```python
@abstractmethod def _build_networks(self) -> None: ...          # construct every parameterised module
@abstractmethod def _build_optimizers(self) -> None: ...        # called right after _build_networks()
@abstractmethod def _select_action(self, observation, training) -> tuple: ...  # (env_action, stored_action)
@abstractmethod def _update(self, batch: list) -> None: ...     # one gradient step
@abstractmethod def _networks(self) -> Mapping[str, nn.Module]: ...
@abstractmethod def _optimizers(self) -> Mapping[str, optim.Optimizer]: ...
```

`_select_action` owns **all** exploration, warmup included — the ancestor never samples an action
itself — and must return `(env_action, stored_action)`: the first drives `env.step()`, the second is
what the replay buffer holds in the model's own coordinates (the same object when those coordinates
are the environment's, e.g. any discrete-action model).

A handful of further hooks are concrete with a default and only need overriding when they apply:
`_ready_to_update()` (add a warmup condition), `_on_training_step()` (advance per-step state such as
a decaying epsilon), `_sync_targets()` / `_target_pairs()` / `_target_sync_interval()` (a hard,
periodic copy onto delayed networks — `None` by default, meaning never), and `_extra_state()` /
`_load_extra_state()` (subclass-specific checkpoint payload, e.g. `epsilon` or a learned temperature,
plus anything a network's shape depends on such as `frame_stack`). `_migrate_parameter_keys()` is
identity by default and must be overridden whenever a module is renamed or re-nested, since a
`state_dict`'s keys are attribute paths and `load_state_dict` is strict (see `CLAUDE.md`'s Gotchas).

**Continuous action spaces**: `ContinuousActorCriticModel` (`src/hercule/models/continuous_actor_critic/`)
extends `OffPolicyReplayModel` for actor-critic algorithms over a `Box` action space (an actor, two
value estimators and their delayed copies, gradually averaged — no delayed actor). It inherits the
ancestor's normalised-to-environment action mapping (`to_env_action`/`to_policy_action`, per
dimension, cached only when the action space is a `Box`) and must store actions in the policy's own
normalised coordinates in the replay buffer, converting to the environment's bounds only where the
environment is actually stepped. See `src/hercule/models/sac/__init__.py` (`SACModel`) for a concrete
example of both hook surfaces.

## Coding Standards

- **Language**: Python 3.10+ — all code, comments, docstrings, and error
  messages in English.
- **Typing**: `X | Y` (not `Union[X, Y]`), avoid `Any`.
- **Validation**: Pydantic V2, `@field_validator` + `@classmethod`.
- **Linter**: Ruff (B, C4, E, F, N, W, I, UP, TID, TC, PLC, PLE, PLW),
  line length 120.
- **Imports**: absolute from `hercule.*`, no relative parent imports.
- **Tests**: run via `uv run pytest` (test files in `tests/`).
- **Toolchain**: uv manages dependencies and the `.venv/`; prefix every command
  with `uv run` (`uv sync` to install, `uv add` / `uv remove` to change deps).

## Configuration System

Experiments are defined in YAML:

```yaml
name: my_experiment
environments:
  - "CartPole-v1"
  - name: "FrozenLake-v1"
    hyperparameters:
      - key: "is_slippery"
        value: false
models:
  - name: "simple_q_learning"
    hyperparameters:
      - key: "learning_rate"
        value: [0.01, 0.1]   # list → expanded to 2 variants
learn_max_epoch: 5000
test_epoch: 100
```

List-valued hyperparameters produce the Cartesian product of all variants.

## CLI Commands

| Command                              | Description                              |
|--------------------------------------|------------------------------------------|
| `hercule learn <config.yaml>`        | Train all model×env combinations         |
| `hercule play <model.json> <env.json>` | Interactive visual playback            |
| `hercule report <output_dir>`        | Generate analysis report (Jupytext .py)  |

## Report Generation

`hercule report <output_dir>` (`reports/generate_report()`) auto-detects a single run directory
(one `report.py`) vs. a parent directory (recursive search, grouped by environment + env-params,
one `comparative_report.py` per group). The pipeline:

1. `build_run_table(root)` walks the run directories **once** and reads only `environment.json` +
   `run_info.json` per run — `model.json` (stored weights) is **never opened**; `model_name` comes
   from the run directory's parent name. The same function runs both at generation time and inside
   the generated notebook, so there is exactly one loading loop regardless of run count.
2. `select_series()` caps every multi-run chart at 9 ranked series (3 best/3 median/3 worst on
   that chart's metric), deterministic via a `(-metric_value, directory_name)` sort key.
3. `variance_decomposition()` (`reports/sensitivity.py`) attributes performance variance to each
   model family's hyperparameters via eta-squared by grouping (ANOVA, no PCA, exact for a balanced
   design) — one consolidated table of main effects and pure two-way interactions, plus a residual.
   `hyperparameter_main_effects()` (mean AND max per level) and `top_decile_comparison()` (the same
   decomposition on the top-scoring subset) round out the sensitivity analysis; every function
   returns an "…Unavailable" result with a reason instead of raising when a family has too few
   varying hyperparameters or runs.
4. `render_report()` executes the generated `.py` (jupytext → `ExecutePreprocessor`), exports a
   tag-filtered HTML (mechanical cells removed, informative output kept), and prints it to PDF via
   a system Chromium-family browser — degrading to `pdf=None` + a reason, never an exception, when
   no browser is available.

`generate_report()` / `generate_individual_report()` / `controller.generate_experiment_report()`
all return a `ReportBundle` (`reports: list[ReportArtifacts]`, `skipped_groups: list[SkippedGroup]`)
rather than a bare `Path`, so every artifact produced and every candidate group skipped (e.g. too
few runs to compare) is reported back to the caller. See `src/hercule/reports/README.md` for the
full module breakdown and `CLAUDE.md` for the Windows/notebook-execution gotchas (`Path(__file__)`
undefined in a kernel, `TagRemovePreprocessor.enabled` defaulting to `False`, PDF success judged by
file size not return code, etc.).

## Key Patterns

- **Context managers** for `EnvironmentManager` / `EnvironmentFactory`.
- **Factory pattern** for environments (cached) and models (`create_model()`).
- **Supervisor → Runner → Model** delegation chain for training.
- **JSON serialisation** for model weights, environment specs, and run info.

## When to Update the Constitution

If your change touches any Root Class Registry entry in a semantic way, you
MUST:

1. Check `.specify/memory/constitution.md`.
2. Determine if an amendment is needed (MAJOR / MINOR / PATCH).
3. Update the constitution, bump the version, and note the change in the
   Sync Impact Report comment at the top of the file.
4. Include a "Constitution Impact" section in the PR description.
