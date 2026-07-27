"""Hercule — a configuration-driven framework for training and benchmarking
reinforcement learning algorithms on Gymnasium environments.

A single YAML file describes the Cartesian product of
(models x environments x hyperparameter variants); the framework trains each
combination, checkpoints it, and generates comparative reports.

Entry points:

- `hercule.supervisor` — orchestrates the learn and test phases
- `hercule.models` — `RLModel` base class and the algorithm sub-packages
- `hercule.config` — YAML parsing and hyperparameter variant expansion
- `hercule.cli` — the `hercule` command-line interface
"""

__version__ = "0.0.0"
