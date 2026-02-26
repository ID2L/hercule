# Hercule

A reinforcement learning framework for exploring, implementing, and
benchmarking RL algorithms in batch against collections of
[Gymnasium](https://gymnasium.farama.org/) environments.

## Motivation

Hercule provides a **generic, configuration-driven** approach to RL
experimentation. Define your models and environments in a single YAML file,
and the framework handles:

- Cartesian product of (models × environments × hyperparameter variants)
- Training and evaluation loops
- Model persistence and checkpoint resumption
- Automated comparative report generation

## Quick Start

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) 2.x

### Installation

```bash
poetry install
```

### Run an Experiment

```bash
# Train all model × environment combinations defined in the config
poetry run hercule learn experiments/simple_games.yaml

# Generate analysis report
poetry run hercule report outputs/<experiment_name>

# Interactively play a trained model
poetry run hercule play outputs/.../model.json outputs/.../environment.json
```

## Configuration

Experiments are fully described by a YAML configuration file:

```yaml
name: frozenlake_benchmark
environments:
  - "CartPole-v1"
  - name: "FrozenLake-v1"
    hyperparameters:
      - key: "is_slippery"
        value: false
      - key: "map_name"
        value: "4x4"
models:
  - name: "simple_q_learning"
    hyperparameters:
      - key: "learning_rate"
        value: [0.01, 0.05, 0.1]    # list values produce variants
      - key: "discount_factor"
        value: 0.95
  - name: "simple_sarsa"
  - name: "deep_q_learning"
learn_max_epoch: 5000
test_epoch: 100
save_every_n_epoch: 1000
```

List-valued hyperparameters are expanded via Cartesian product, so the
configuration above produces **3 variants** of Q-Learning (one per learning
rate) plus one SARSA and one DQN, each trained on both environments.

## Architecture

```text
src/hercule/
├── config/           # YAML parsing, Pydantic V2 models, ParameterValue type
├── environnements/   # Gymnasium registry, factory, inspector, manager
├── models/           # RLModel base class + algorithm sub-packages
│   ├── td_models/    #   └─ TDModel (tabular TD base)
│   ├── simple_q_learning/
│   ├── simple_sarsa/
│   ├── deep_q_learning/
│   └── dummy/
├── run/              # Runner (train/test loop), result persistence
├── supervisor/       # Orchestrates learn & test phases
├── controller/       # Business-logic entry points
├── reports/          # Jinja2-based report generation
└── cli/              # Click CLI
```

### Class Hierarchy

All RL algorithms inherit from a common abstract base class:

```text
RLModel (ABC, Generic[HyperParamsType])
├── TDModel (tabular TD algorithms with Q-table)
│   ├── SimpleQLearningModel   (off-policy TD)
│   └── SimpleSarsaModel       (on-policy TD)
├── DeepQLearningModel         (DQN with neural network)
└── DummyModel                 (random baseline)
```

**`RLModel`** defines the full algorithm lifecycle:

| Method             | Type       | Purpose                                  |
|--------------------|------------|------------------------------------------|
| `configure()`      | virtual    | Bind environment + hyperparameters       |
| `act()`            | abstract   | Select action given observation          |
| `run_epoch()`      | abstract   | Run one episode (train or eval)          |
| `predict()`        | abstract   | Inference-mode action selection          |
| `_export()`        | abstract   | Serialize model state to dict            |
| `_import()`        | abstract   | Deserialize model state from dict        |
| `save()` / `load()`| final     | JSON persistence (delegates to export/import) |
| `evaluate()`       | concrete   | Multi-episode evaluation with metrics    |

**`TDModel`** adds Q-table management and epsilon-greedy exploration. Concrete
TD algorithms only need to implement `update()`.

### Adding a New Algorithm

1. Create `src/hercule/models/<name>/__init__.py`
2. Subclass `RLModel` (or `TDModel`), set `model_name` and `hyperparams_class`
3. Implement the abstract methods
4. Done — auto-discovered at runtime, no registration needed

## Available Algorithms

| Name                 | Class                    | Type          | Description                    |
|----------------------|--------------------------|---------------|--------------------------------|
| `simple_q_learning`  | `SimpleQLearningModel`   | Tabular TD    | Off-policy Q-Learning          |
| `simple_sarsa`       | `SimpleSarsaModel`       | Tabular TD    | On-policy SARSA                |
| `deep_q_learning`    | `DeepQLearningModel`     | Deep RL       | DQN (Mnih et al., 2013)        |
| `dummy`              | `DummyModel`             | Baseline      | Random action selection         |

## CLI Reference

| Command                                          | Description                              |
|--------------------------------------------------|------------------------------------------|
| `hercule learn <config.yaml> [-o dir] [-v]`      | Train and evaluate all combinations      |
| `hercule play <model.json> <env.json> [--no-render]` | Interactive visual playback         |
| `hercule report <experiment_dir> [-o path]`      | Generate Jupytext analysis report        |

## Development

```bash
# Install with dev dependencies
poetry install

# Run tests
poetry run pytest

# Lint
poetry run ruff check .

# Format
poetry run ruff format .

# Generate API documentation
poetry run gen-doc

# Serve documentation locally
poetry run serve-doc
```

### Code Standards

- Python 3.10+ with modern type union syntax (`X | Y`)
- Pydantic V2 with `@field_validator`
- Ruff linter (line length: 120)
- Docstrings on all public classes and functions
- All code, comments, and error messages in English

## Project Governance

The project is governed by a constitution at `.specify/memory/constitution.md`.
Key rule: **any modification to a root class (`RLModel`, `TDModel`,
`BaseConfig`, `HyperParamsBase`, `HerculeConfig`, `EpochResult`, `Runner`,
`Supervisor`) MUST trigger a review of the constitution**. See `AGENTS.md`
for detailed development guidance.

## License

See [LICENSE](LICENSE) for details.
