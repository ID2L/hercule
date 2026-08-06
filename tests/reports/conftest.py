"""Shared fixtures for tests/reports.

Provides a builder that writes a synthetic run tree mirroring the on-disk layout
documented in specs/004-improve-report-generation/data-model.md:
`env_id/env_signature/model_name/model_signature/{environment,model,run_info}.json`.
Kept parameterisable in run count, model family, hyperparameter grid and episode
counts so later phases (run table, selection, PCA) can build the fixtures they need
without touching real `outputs/` data.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest


def _write_json(path: Path, data: dict[str, object]) -> None:
    """Write a dict as pretty-printed JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _episode_metrics(
    count: int,
    *,
    reward_fn: Callable[[int], float],
    steps_fn: Callable[[int], int],
) -> list[dict[str, object]]:
    """Build a list of `EpochResult`-shaped dicts for `learning_metrics`/`testing_metrics`."""
    metrics: list[dict[str, object]] = []
    for i in range(count):
        reward = reward_fn(i)
        metrics.append(
            {
                "reward": reward,
                "steps_number": steps_fn(i),
                "final_state": "terminated" if reward > 0 else "truncated",
            }
        )
    return metrics


class RunTreeBuilder:
    """Writes synthetic run leaf directories under a fresh temporary root.

    Each call to `add_run` writes one `env_id/env_signature/model_name/model_signature/`
    directory containing `environment.json`, `run_info.json` and `model.json`, matching the
    schema `build_run_table` reads (see data-model.md "Existing structures read from disk").
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def add_run(
        self,
        *,
        env_id: str = "FrozenLake-v1",
        env_kwargs: dict[str, object] | None = None,
        max_episode_steps: int | None = 200,
        env_signature: str = "env_sig",
        model_name: str = "simple_q_learning",
        model_signature: str = "model_sig",
        hyperparameters: dict[str, object] | None = None,
        learning_episode_count: int = 10,
        testing_episode_count: int = 5,
        learning_reward_fn: Callable[[int], float] = lambda i: float(i % 2),
        learning_steps_fn: Callable[[int], int] = lambda i: 10 + i,
        testing_reward_fn: Callable[[int], float] = lambda i: float(i % 2),
        testing_steps_fn: Callable[[int], int] = lambda i: 10 + i,
        corrupt_run_info: bool = False,
        corrupt_model: bool = False,
        corrupt_environment: bool = False,
        omit_environment: bool = False,
    ) -> Path:
        """Write one run leaf directory and return its path.

        Args:
            env_id: Gymnasium environment identifier.
            env_kwargs: Environment settings; defaults to `{}` (nothing overridden).
            max_episode_steps: Episode step cap, or `None` for no limit.
            env_signature: Directory segment for the environment-settings level.
            model_name: Model family; becomes the run's parent directory name.
            model_signature: Directory segment for the hyperparameter-signature level.
            hyperparameters: Flat `model_hyperparameters` mapping; defaults to `{}`.
            learning_episode_count: Number of learning-phase episodes to synthesize.
            testing_episode_count: Number of testing-phase episodes to synthesize (0 for none).
            learning_reward_fn: Maps an episode index to its learning reward.
            learning_steps_fn: Maps an episode index to its learning step count.
            testing_reward_fn: Maps an episode index to its testing reward.
            testing_steps_fn: Maps an episode index to its testing step count.
            corrupt_run_info: Write invalid JSON to `run_info.json` instead of valid data.
            corrupt_model: Write invalid JSON to `model.json` instead of valid data.
            corrupt_environment: Write invalid JSON to `environment.json` instead of valid data.
                Unlike `omit_environment`, the file still exists, so the directory is still
                discovered by `find_experiment_directories` (which only checks existence) and
                the failure surfaces as a `SkippedRun` instead of the directory being invisible.
            omit_environment: Skip writing `environment.json` entirely. Note this makes the
                directory fail `is_valid_experiment_directory`'s existence check, so it is never
                visited at all — it will not appear as a `SkippedRun` either.

        Returns:
            The path to the created run leaf directory.
        """
        env_kwargs = {} if env_kwargs is None else env_kwargs
        hyperparameters = {} if hyperparameters is None else hyperparameters

        run_dir = self.root / env_id / env_signature / model_name / model_signature
        run_dir.mkdir(parents=True, exist_ok=True)

        if omit_environment:
            pass
        elif corrupt_environment:
            (run_dir / "environment.json").write_text("{not valid json", encoding="utf-8")
        else:
            _write_json(
                run_dir / "environment.json",
                {
                    "id": env_id,
                    "max_episode_steps": max_episode_steps,
                    "disable_env_checker": False,
                    "kwargs": env_kwargs,
                },
            )

        if corrupt_run_info:
            (run_dir / "run_info.json").write_text("{not valid json", encoding="utf-8")
        else:
            _write_json(
                run_dir / "run_info.json",
                {
                    "learning_ongoing_epoch": learning_episode_count,
                    "testing_ongoing_epoch": testing_episode_count,
                    "learning_metrics": _episode_metrics(
                        learning_episode_count,
                        reward_fn=learning_reward_fn,
                        steps_fn=learning_steps_fn,
                    ),
                    "testing_metrics": _episode_metrics(
                        testing_episode_count,
                        reward_fn=testing_reward_fn,
                        steps_fn=testing_steps_fn,
                    ),
                    "model_hyperparameters": hyperparameters,
                },
            )

        if corrupt_model:
            (run_dir / "model.json").write_text("{not valid json", encoding="utf-8")
        else:
            _write_json(run_dir / "model.json", {"model_name": model_name, "weights": "deliberately-not-read"})

        return run_dir


@pytest.fixture
def run_tree_builder(tmp_path: Path) -> RunTreeBuilder:
    """A builder that writes a synthetic run tree under a fresh temporary directory per test."""
    return RunTreeBuilder(tmp_path)
