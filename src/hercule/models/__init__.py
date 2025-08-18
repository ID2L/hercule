"""Abstract base classes and interfaces for reinforcement learning models."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue

logger = logging.getLogger(__name__)


class RLModel(ABC):
    """Abstract base class for reinforcement learning models."""

    def __init__(self, name: str) -> None:
        """
        Initialize the RL model.

        Args:
            name: Model name identifier
        """
        self.name = name
        self.is_trained = False
        self._training_metrics: dict[str, ParameterValue] = {}

    @abstractmethod
    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """
        Configure the model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters
        """
        pass

    @abstractmethod
    def act(self, observation: np.ndarray, training: bool = False) -> int | np.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Action to take in the environment
        """
        pass

    @abstractmethod
    def learn(
        self, observation: np.ndarray, action: int | np.ndarray, reward: float, next_observation: np.ndarray, done: bool
    ) -> None:
        """
        Update the model based on a transition.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is finished
        """
        pass

    @abstractmethod
    def train(self, env: gym.Env, config: dict[str, ParameterValue], max_iterations: int) -> dict[str, ParameterValue]:
        """
        Train the model on the given environment.

        Args:
            env: Gymnasium environment
            config: Model hyperparameters
            max_iterations: Maximum number of training iterations

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path where to save the model
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        pass

    def get_training_metrics(self) -> dict[str, ParameterValue]:
        """
        Get training metrics.

        Returns:
            Dictionary of training metrics
        """
        return self._training_metrics.copy()

    def reset_training_metrics(self) -> None:
        """Reset training metrics."""
        self._training_metrics.clear()

    def add_metric(self, key: str, value: ParameterValue) -> None:
        """
        Add a training metric.

        Args:
            key: Metric name
            value: Metric value
        """
        self._training_metrics[key] = value

    def evaluate(self, env: gym.Env, num_episodes: int = 10) -> dict[str, float]:
        """
        Evaluate the model on an environment.

        Args:
            env: Gymnasium environment
            num_episodes: Number of episodes to run

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            logger.warning(f"Model {self.name} has not been trained yet")

        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.act(observation, training=False)
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
        }

        logger.info(f"Evaluation results for {self.name}: {metrics}")
        return metrics

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()


class BaselineModel(RLModel):
    """
    Simple baseline model for testing purposes.

    This model takes random actions and serves as a baseline for comparison.
    """

    def __init__(self, name: str = "baseline") -> None:
        """Initialize the baseline model."""
        super().__init__(name)
        self._action_space: gym.Space | None = None

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """Configure the baseline model."""
        self._action_space = env.action_space
        logger.info(f"Baseline model configured for environment with action space: {self._action_space}")

    def act(self, observation: np.ndarray, training: bool = False) -> int | np.ndarray:
        """Take a random action."""
        if self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)
        return self._action_space.sample()

    def learn(
        self, observation: np.ndarray, action: int | np.ndarray, reward: float, next_observation: np.ndarray, done: bool
    ) -> None:
        """Baseline model doesn't learn."""
        pass

    def train(self, env: gym.Env, config: dict[str, ParameterValue], max_iterations: int) -> dict[str, ParameterValue]:
        """
        Train the baseline model (which just runs random episodes).

        Args:
            env: Gymnasium environment
            config: Model hyperparameters (unused for baseline)
            max_iterations: Maximum number of training iterations

        Returns:
            Training results and metrics
        """
        self.configure(env, config)

        num_episodes_value = config.get("num_episodes", 100)
        if isinstance(num_episodes_value, int):
            num_episodes = min(max_iterations, num_episodes_value)
        else:
            num_episodes = min(max_iterations, 100)
        rewards = []

        for _ in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.act(observation, training=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                observation = next_observation

            rewards.append(episode_reward)

        self.is_trained = True
        metrics: dict[str, ParameterValue] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "num_episodes": num_episodes,
        }
        self._training_metrics.update(metrics)

        return metrics

    def save(self, path: Path) -> None:
        """Save baseline model (minimal save since it's stateless)."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "baseline_model.txt", "w", encoding="utf-8") as f:
            f.write(f"Baseline model: {self.name}\n")
            f.write(f"Trained: {self.is_trained}\n")

    def load(self, path: Path) -> None:
        """Load baseline model."""
        model_file = path / "baseline_model.txt"
        if model_file.exists():
            self.is_trained = True
            logger.info(f"Loaded baseline model from {path}")
        else:
            msg = f"No baseline model found at {path}"
            raise FileNotFoundError(msg)
