"""Dummy model implementation for testing purposes."""

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


logger = logging.getLogger(__name__)


class DummyModel(RLModel):
    """
    Dummy model that takes random actions from the environment's action space.

    This model serves as a simple baseline and does not perform any learning.
    It randomly selects actions from the available action space on each step.
    """

    # Class attribute for model name
    model_name: str = "dummy"

    def __init__(self) -> None:
        """
        Initialize the dummy model.

        Args:
            name: Model name identifier (if None, uses class model_name)
        """
        super().__init__()
        self._action_space: gym.Space | None = None
        self._rng = np.random.default_rng(42)  # Default seed for reproducibility
        self.is_trained = True

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """
        Configure the dummy model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (unused for dummy model)
        """
        super().configure(env, hyperparameters)
        self._action_space = env.action_space

        # Set random seed if provided in hyperparameters
        if "seed" in hyperparameters:
            seed_value = hyperparameters["seed"]
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

    def save(self, path: Path) -> None:
        """
        Save dummy model state to disk.

        Args:
            path: Path where to save the model
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save minimal model information
        model_file = path / "dummy_model.txt"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(f"Dummy model: {self.model_name}\n")
            f.write(f"Trained: {self.is_trained}\n")
            f.write("Model type: Random action selection\n")
            f.write("Learning: None (dummy model)\n")

        # Save training metrics if available
        if self._training_metrics:
            metrics_file = path / "training_metrics.txt"
            with open(metrics_file, "w", encoding="utf-8") as f:
                f.write("Training Metrics:\n")
                for key, value in self._training_metrics.items():
                    f.write(f"{key}: {value}\n")

        logger.info(f"Dummy model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load dummy model state from disk.

        Args:
            path: Path to the saved model

        Raises:
            FileNotFoundError: If model file is not found
        """
        model_file = path / "dummy_model.txt"
        if not model_file.exists():
            msg = f"No dummy model found at {path}"
            raise FileNotFoundError(msg)

        # Load basic model state
        with open(model_file, encoding="utf-8") as f:
            content = f.read()
            if "Trained: True" in content:
                self.is_trained = True

        # Load training metrics if available
        metrics_file = path / "training_metrics.txt"
        if metrics_file.exists():
            # Parse metrics file - simple implementation
            # For a more robust implementation, could use JSON
            pass

        logger.info(f"Loaded dummy model from {path}")


__all__ = ["DummyModel"]
