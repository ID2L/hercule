"""Dummy model implementation for testing purposes."""

import logging
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from pydantic import Field, PrivateAttr

from hercule.config import HyperParamsBase, ParameterValue
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


logger = logging.getLogger(__name__)


class DummyModelHyperParams(HyperParamsBase):
    """Type-safe hyperparameters for Dummy model."""

    seed: int = Field(default=42, description="Random seed")


class DummyModel(RLModel[DummyModelHyperParams]):
    """
    Dummy model that takes random actions from the environment's action space.

    This model serves as a simple baseline and does not perform any learning.
    It randomly selects actions from the available action space on each step.
    """

    # Class attribute for model name (static, immutable)
    model_name: ClassVar[str] = "dummy"
    # Type-safe hyperparameters class
    hyperparams_class: ClassVar[type[HyperParamsBase]] = DummyModelHyperParams

    @classmethod
    def default_hyperparameters_typed(cls) -> DummyModelHyperParams:
        """
        Get default hyperparameters as a typed instance.

        Returns a type-safe instance of DummyModelHyperParams with default values.
        This provides autocomplete and type checking in IDEs.

        Returns:
            Typed hyperparameters instance with default values
        """
        return DummyModelHyperParams()

    # Private attributes (not Pydantic fields, use PrivateAttr to avoid validation)
    _action_space: gym.Space | None = PrivateAttr(default=None)
    _rng: np.random.Generator | None = PrivateAttr(default=None)
    _is_trained: bool = PrivateAttr(default=True)

    def __init__(self) -> None:
        """
        Initialize the dummy model.

        Args:
            name: Model name identifier (if None, uses class model_name)
        """
        super().__init__()

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """
        Configure the dummy model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (will be merged with defaults)
        """
        # Configure base class (this will merge with defaults and store in self.hyperparameters)
        super().configure(env, hyperparameters)
        self._action_space = env.action_space

        # Get hyperparameters - use typed hyperparameters (type-safe)
        typed_params = self.get_hyperparameters()
        seed_value = typed_params.seed
        if isinstance(seed_value, int):
            self._rng = np.random.default_rng(seed_value)

        logger.info(
            f"Dummy model '{self.model_name}' configured for environment with action space: {self._action_space}"
        )

    def run_epoch(self, train_mode=False) -> EpochResult:
        env = self.check_environment_or_raise()

        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action = self.act(observation, training=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1
            observation = next_observation

        return EpochResult(
            reward=float(episode_reward),
            steps_number=episode_length,
            final_state="truncated" if truncated else "terminated",
        )

    def act(self, observation: np.ndarray, training: bool = False) -> int | np.ndarray:
        """
        Select a random action from the environment's action space.

        Args:
            observation: Environment observation (unused for dummy model)
            training: Whether the model is in training mode (unused for dummy model)

        Returns:
            Randomly selected action

        Raises:
            ValueError: If model is not configured
        """
        if self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        # Use the environment's action space to sample a random action
        return self._action_space.sample()

    def predict(self, observation: np.ndarray | int) -> int | np.ndarray:
        """
        Predict the best action for a given observation (inference mode).

        For the dummy model, this returns a random action.

        Args:
            observation: Current observation from the environment (unused)

        Returns:
            Randomly selected action
        """
        return self.act(observation, training=False)

    def learn(
        self, observation: np.ndarray, action: int | np.ndarray, reward: float, next_observation: np.ndarray, done: bool
    ) -> None:
        """
        Dummy model does not learn from experience.

        Args:
            observation: Current observation (unused)
            action: Action taken (unused)
            reward: Reward received (unused)
            next_observation: Next observation (unused)
            done: Whether episode is finished (unused)
        """
        # Dummy model doesn't learn - this is a no-op
        pass

    def train(self, env: gym.Env, config: dict[str, ParameterValue], max_iterations: int) -> dict[str, ParameterValue]:
        """
        Train the dummy model (runs episodes with random actions for evaluation).

        Args:
            env: Gymnasium environment
            config: Model hyperparameters
            max_iterations: Maximum number of training iterations (episodes for dummy model)

        Returns:
            Training results and metrics
        """
        if not self.env:
            self.configure(env, config)

        if self.env is None:
            msg = "Model not configured with an environment. Call configure() first."
            raise ValueError(msg)

        rewards = []
        episode_lengths = []

        logger.info(f"Running {max_iterations} episodes with dummy model '{self.model_name}'")

        for episode in range(max_iterations):
            observation, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.act(observation, training=True)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                episode_length += 1
                observation = next_observation

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Log progress every 20% of episodes
            if (episode + 1) % max(1, max_iterations // 5) == 0:
                logger.info(f"Completed {episode + 1}/{max_iterations} episodes")

        self.is_trained = True

        # Calculate training metrics
        metrics: dict[str, ParameterValue] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
            "episodes_run": max_iterations,
        }

        self._training_metrics.update(metrics)

        logger.info(
            f"Dummy model training completed. Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
        )

        return metrics

    def _export(self) -> dict:
        """
        Export dummy model data for serialization.

        Returns:
            Dictionary containing model data ready for JSON serialization
        """
        # Dummy model doesn't have any state to export, return empty dict
        return {}

    def _import(self, model_data: dict) -> None:
        """
        Import dummy model data from serialized format.

        Args:
            model_data: Dictionary containing model data from JSON
        """
        # Dummy model doesn't have any state to import, this is a no-op
        pass

    def load_from_dict(self, model_data: dict) -> None:
        """
        Load a trained dummy model from a dictionary.

        Args:
            model_data: Dictionary containing model data
        """
        self._import(model_data)
        logger.info(f"Loaded {self.model_name} model from dictionary")


__all__ = ["DummyModel"]
