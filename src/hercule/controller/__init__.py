"""Controller layer exposing business logic decoupled from CLI.

This module provides reusable functions to orchestrate learning runs and
interactive model execution, without any CLI-specific input/output. It can be
imported by alternative frontends (e.g., web API) to trigger the same actions.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym

from hercule.config import HerculeConfig, load_config_from_yaml
from hercule.environnements import load_environment
from hercule.models import RLModel, create_model
from hercule.supervisor import Supervisor


logger = logging.getLogger(__name__)


class CancellationToken:
    """Simple thread-safe cancellation token.

    A web API or CLI can hold a reference to this token and request graceful
    stop of a long-running operation.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def run_learning(config_file: Path, output_dir: Path | None = None) -> None:
    """Run the learning then testing phases from a YAML configuration file.

    Args:
        config_file: Path to YAML configuration.
        output_dir: Optional override for output directory.
    """
    config: HerculeConfig = load_config_from_yaml(config_file)

    if output_dir is not None:
        config.base_output_dir = output_dir

    # Save configuration summary to the appropriate location
    config.save()

    supervisor = Supervisor(config)
    supervisor.execute_learn_phase()
    supervisor.execute_test_phase()


@dataclass
class PlayResult:
    total_episodes: int
    total_reward: float

    @property
    def average_reward(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_reward / float(self.total_episodes)


def play_interactive(
    model_file: Path,
    environment_file: Path,
    cancel_token: CancellationToken | None = None,
) -> PlayResult:
    """Run an interactive simulation of a trained model until cancelled.

    This function is responsible for the environment/model setup and execution
    loop. It returns aggregate metrics so callers can format final messages.

    Args:
        model_file: JSON file containing trained model parameters.
        environment_file: JSON file containing saved environment configuration.
        render_mode: Gymnasium render mode.
        cancel_token: Optional cancellation token for graceful stop.

    Returns:
        PlayResult: aggregate statistics of the session.
    """
    # Load environment description and instantiate render-capable environment
    environment = load_environment(environment_file)
    env_id = environment.spec.id if getattr(environment, "spec", None) else "Unknown"
    kwargs = getattr(getattr(environment, "spec", None), "kwargs", {}) or {}

    env_with_render = gym.make(env_id, render_mode="human", **kwargs)

    try:
        # Load model payload
        with open(model_file, encoding="utf-8") as f:
            model_data = json.load(f)

        model_name = model_data.get("model_name", "simple_sarsa")
        model: RLModel = create_model(model_name)

        # Configure then hydrate weights
        model.configure(env_with_render, {})
        model.load_from_dict(model_data)

        total_reward: float = 0.0
        episode_count: int = 0

        while True:
            # Cancellation check at episode boundary
            if cancel_token is not None and cancel_token.is_cancelled():
                break

            obs, _ = env_with_render.reset()
            episode_reward: float = 0.0
            done = False
            episode_count += 1

            try:
                while not done:
                    # Periodic cancellation check inside episode
                    if cancel_token is not None and cancel_token.is_cancelled():
                        # Stop current episode early and exit gracefully
                        done = True
                        break

                    action = model.predict(obs)
                    obs, reward, terminated, truncated, _ = env_with_render.step(action)
                    episode_reward += float(reward)
                    done = bool(terminated or truncated)

                    env_with_render.render()

            except KeyboardInterrupt:
                # Graceful interruption inside an episode
                break

            total_reward += episode_reward

        return PlayResult(total_episodes=episode_count, total_reward=total_reward)

    except KeyboardInterrupt:
        # Interruption at outer loop level
        return PlayResult(total_episodes=0, total_reward=0.0)
    finally:
        env_with_render.close()
