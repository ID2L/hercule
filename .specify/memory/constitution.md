<!-- Sync Impact Report
Version change: 0.0.0 → 1.0.0 (initial ratification)
Added principles:
  - I. Generic Algorithm Architecture (NON-NEGOTIABLE)
  - II. Configuration-Driven Design
  - III. Gymnasium-First Integration
  - IV. Module Separation
  - V. Modern Python & Code Quality
  - VI. Extensibility & Discoverability
Added sections:
  - Root Class Registry (critical reference)
  - Development Workflow
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (Constitution Check gates defined)
  - .specify/templates/spec-template.md ✅ (no conflict)
  - .specify/templates/tasks-template.md ✅ (no conflict)
Follow-up TODOs: none
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
- Every concrete model MUST implement the abstract methods: `act()`,
  `run_epoch()`, `predict()`, `_export()`, `_import()`.
- The `configure() → save() → load()` lifecycle defined by `RLModel` MUST NOT
  be bypassed; `save()` and `load()` are `@final`.
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

**Version**: 1.0.0 | **Ratified**: 2026-02-26 | **Last Amended**: 2026-02-26
