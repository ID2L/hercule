<!-- Sync Impact Report
Version change: 1.1.0 → 1.2.0 (MINOR: additive ClassVar contract on RLModel)
Modified principles:
  - I. Generic Algorithm Architecture: a concrete model MUST now declare
    `supported_spaces`, and model discovery registers concrete classes only.
Added sections: none
Removed sections: none
Rationale:
  `check_space_is_box()` returned `not check_space_is_discrete(space)`, so every
  non-`Discrete` space — `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`, `Text`,
  `Sequence`, `Graph`, `OneOf` — was accepted as a `Box`. Compounding it,
  `configure()` returned `False` on a space mismatch and `Supervisor` ignored the
  return value, so pairing a tabular model with e.g. `CartPole-v1` failed later
  with an unrelated error instead of a clear message. A model must be able to
  state which (observation, action) space pair it accepts, and the orchestrator
  must check it before configuring.
  Separately, `get_available_models()` registered every `RLModel` subclass it
  found, including abstract ones: the registry returned `tdmodel -> TDModel`,
  whose instantiation raises `TypeError`. This blocks the intermediate abstract
  classes the continuous-action roadmap introduces.
Backward compatibility: ADDITIVE for the abstract-method surface, which is
  unchanged; `save()` and `load()` remain `@final` and untouched. BREAKING for a
  concrete model that declares no `supported_spaces`: it is no longer registered.
  All four existing models declare it as part of this change.
Callers updated: `get_available_models()`, `Supervisor.execute_learn_phase()` /
  `execute_test_phase()`, `deep_q_learning`, `td_models`, `simple_q_learning`,
  `simple_sarsa`, `dummy`.
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (no conflict)
  - .specify/templates/spec-template.md ✅ (no conflict)
  - .specify/templates/tasks-template.md ✅ (no conflict)
Follow-up TODOs: none

Previous: 1.0.0 → 1.1.0 (MINOR: additive lifecycle hook on RLModel)
Modified principles:
  - I. Generic Algorithm Architecture: the per-episode lifecycle now includes
    `begin_episode()` on `RLModel`.
Added sections: none
Removed sections: none
Rationale:
  `predict()` receives a single observation and nothing else, so a model whose
  policy carries state ACROSS steps (observation stacking, recurrent policy)
  cannot detect an episode boundary. Without a hook that state silently spans
  two episodes and the policy acts on an observation that never existed. This
  was a real defect in `deep_q_learning`'s frame stacking, in `predict()` and
  therefore in `hercule play`; `run_epoch()` was already resetting correctly.
Backward compatibility: ADDITIVE, no migration needed. `begin_episode()` is
  concrete with a default no-op body, not abstract, so no existing model needs
  changing (`TDModel`, `simple_q_learning`, `simple_sarsa`, `dummy` are
  unaffected) and no existing signature moved.
Callers updated: `RLModel.evaluate()`, `controller.play_interactive()`,
  `DeepQLearningModel.run_epoch()`.
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (no conflict)
  - .specify/templates/spec-template.md ✅ (no conflict)
  - .specify/templates/tasks-template.md ✅ (no conflict)
Follow-up TODOs: none

Previous: 0.0.0 → 1.0.0 (initial ratification) added principles I-VI, the Root
Class Registry and the Development Workflow section.
-->

# Hercule Constitution

## Core Principles

### I. Generic Algorithm Architecture (NON-NEGOTIABLE)

Every reinforcement learning algorithm MUST inherit from the `RLModel` abstract
base class (`src/hercule/models/__init__.py`). Intermediate abstract classes
(e.g. `TDModel` for tabular temporal-difference methods) MAY be introduced to
factor common behaviour, but they MUST themselves extend `RLModel`.

**Rules:**

- A new algorithm is implemented as a sub-package under `src/hercule/models/<algorithm_name>/`
  with an `__init__.py` that exports exactly one concrete `RLModel` subclass.
- Every concrete model MUST declare a unique `model_name: ClassVar[str]` and
  a `hyperparams_class: ClassVar[type[HyperParamsBase]]`.
- **Space support**: every concrete model MUST declare
  `supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]]`, the set of
  `(observation_kind, action_kind)` pairs it accepts. There is deliberately **no
  default**: a permissive one would silently pair a new model with any environment,
  which is the failure this contract exists to remove. `Supervisor` compares the
  declaration against the environment **before** calling `configure()`, skips a
  mismatched combination with a message naming both the expected and the actual
  kinds, and continues with the remaining combinations. The check MUST NOT live
  behind `configure()`: an override that calls `super().configure(...)` and
  discards its return value would swallow the rejection.
- **Model discovery registers concrete classes only.** `get_available_models()`
  registers a class only when it is not abstract AND declares `model_name` in its
  own `__dict__`. Intermediate abstract classes are therefore free to live in a
  scanned package and to be imported by concrete models without appearing as
  phantom entries in the registry.
- Every concrete model MUST implement the abstract methods: `act()`,
  `run_epoch()`, `predict()`, `_export()`, `_import()`.
- The `configure() → save() → load()` lifecycle defined by `RLModel` MUST NOT
  be bypassed; `save()` and `load()` are `@final`.
- **Per-episode state**: a model whose policy carries state ACROSS steps within
  an episode (observation stacking, recurrent policy) MUST reset that state in
  `begin_episode()`, and every caller driving its own episode loop MUST call
  `begin_episode()` right after `env.reset()`. `begin_episode()` is concrete
  with a default no-op body, so a model whose policy depends only on the current
  observation needs no change. This exists because `predict()` receives one
  observation and nothing else: an episode boundary is not observable from it,
  so without the hook a stacked state spans two episodes.
- **Any modification to a root class listed in the Root Class Registry below
  MUST trigger a review of this constitution and, if semantics change, a
  constitutional amendment (MINOR or MAJOR version bump).**

### II. Configuration-Driven Design

All experiment parameters MUST be expressible in a single YAML file parsed by
`HerculeConfig` (Pydantic V2).

**Rules:**

- Hyperparameter types are constrained to `ParameterValue` (defined in
  `hercule.config`). Extending this union requires a constitution amendment.
- Model and environment hyperparameters MUST be provided via
  `HerculeConfig.get_hyperparameters_for_model()` /
  `get_hyperparameters_for_environment()` — never hard-coded.
- List-valued hyperparameters are expanded via `expand_variants()` to produce
  the Cartesian product of all combinations for batch runs.
- Validator decorators MUST use `@field_validator` (Pydantic V2), never the
  deprecated `@validator`.

### III. Gymnasium-First Integration

Hercule targets **Gymnasium** (`gymnasium` package) as its sole environment
interface. All environments MUST be loadable via `gym.make()`.

**Rules:**

- Environments MUST always be loaded through `EnvironmentManager` or
  `EnvironmentFactory` — direct `gym.make()` calls in business logic are
  forbidden.
- Environment metadata MUST be extracted from `env.spec` (kwargs,
  max_episode_steps, reward_threshold…), never duplicated in configuration.
- The `EnvironmentFactory` caches environments by `(name, hyperparameters)`;
  callers MUST NOT close environments obtained from the factory manually.

### IV. Module Separation

Each top-level package under `src/hercule/` has a single, well-defined
responsibility:

| Package           | Responsibility                                    |
|-------------------|---------------------------------------------------|
| `config`          | YAML parsing, Pydantic models, type aliases       |
| `environnements`  | Gymnasium registry, factory, inspector, manager   |
| `models`          | `RLModel` base class, algorithm implementations   |
| `run`             | `Runner`, training/testing loop, result storage   |
| `supervisor`      | Orchestration of learn & test phases              |
| `controller`      | Business-logic entry points (learn, play, report) |
| `reports`         | Jinja2-based experiment report generation         |
| `cli`             | Click CLI — thin layer delegating to `controller` |

- Cross-module imports MUST follow the dependency order above (top → bottom).
  Circular imports are forbidden.
- New top-level packages MUST be justified and approved via constitution
  amendment (MINOR bump).

### V. Modern Python & Code Quality

- Python **3.10+** is the minimum supported version.
- Type annotations MUST use modern union syntax (`X | Y`), never
  `typing.Union` or `typing.Optional`.
- `Any` MUST be avoided; use explicit union types.
- **Ruff** is the single linter/formatter (rules: B, C4, E, F, N, W, I, UP,
  TID, TC, PLC, PLE, PLW). Line length is **120** characters.
- All public classes and functions MUST have docstrings.
- Relative parent imports (`from .. import`) are banned; use absolute imports
  from `hercule.*`.

### VI. Extensibility & Discoverability

Adding a new RL algorithm MUST NOT require modifying any existing file outside
the new algorithm's sub-package.

**Rules:**

- `get_available_models()` dynamically discovers model sub-packages via
  filesystem introspection — no manual registry.
- `create_model(name)` is the single factory entry point for instantiation.
- Each algorithm sub-package MUST be self-contained: model class,
  hyperparameters class, and optional helpers.

## Root Class Registry

The following classes are **architectural foundations**. Any semantic change to
their public API (added/removed/renamed abstract methods, changed signatures,
altered lifecycle contracts) MUST trigger a constitution review.

| Class              | Location                                   | Role                                      |
|--------------------|--------------------------------------------|-------------------------------------------|
| `RLModel`          | `src/hercule/models/__init__.py`           | Abstract base for all RL algorithms       |
| `TDModel`          | `src/hercule/models/td_models/__init__.py` | Abstract base for tabular TD algorithms   |
| `BaseConfig`       | `src/hercule/config/__init__.py`           | Base Pydantic model for configurations    |
| `HyperParamsBase`  | `src/hercule/config/__init__.py`           | Base class for typed hyperparameters      |
| `HerculeConfig`    | `src/hercule/config/__init__.py`           | Top-level experiment configuration        |
| `EpochResult`      | `src/hercule/models/epoch_result.py`       | Standardised epoch return type            |
| `Runner`           | `src/hercule/run/__init__.py`              | Training/testing execution engine         |
| `Supervisor`       | `src/hercule/supervisor/__init__.py`       | High-level phase orchestrator             |

## Development Workflow

1. **Adding an algorithm**: create `src/hercule/models/<name>/__init__.py`,
   implement `RLModel` (or a subclass like `TDModel`), declare `model_name`
   and `hyperparams_class`. No other file needs changing.
2. **Adding an environment**: add its Gymnasium ID (and optional
   hyperparameters) to the YAML config. No code change required.
3. **Running experiments**: `hercule learn <config.yaml>` trains all
   model×environment combinations, then `hercule report <output_dir>`
   generates analysis notebooks.
4. **Playing a trained model**: `hercule play <model.json> <environment.json>`
   renders the agent in real time.

## Governance

- This constitution supersedes all other development practices for the Hercule
  project.
- **Amendment procedure**: propose change → update constitution → bump version
  (MAJOR for breaking removals/redefinitions, MINOR for additions/expansions,
  PATCH for clarifications) → update dependent templates if needed.
- Any PR that modifies a Root Class Registry entry MUST include a
  "Constitution Impact" section in its description explaining whether an
  amendment is required.
- Use `AGENTS.md` at the repository root for runtime AI-agent development
  guidance.

**Version**: 1.2.0 | **Ratified**: 2026-02-26 | **Last Amended**: 2026-09-10
