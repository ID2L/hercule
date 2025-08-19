"""Abstract base classes and interfaces for reinforcement learning models."""

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue


logger = logging.getLogger(__name__)


class RLModel(ABC):
    """Abstract base class for reinforcement learning models."""

    # Class attribute for model name
    model_name: str = "abstract_RL_model"

    def __init__(self) -> None:
        """
        Initialize the RL model.

        Args:
            name: Model name identifier (if None, uses class model_name)
        """
        self.is_trained = False
        self._training_metrics: dict[str, ParameterValue] = {}
        self.env: gym.Env | None = None

    @abstractmethod
    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """
        Configure the model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters
        """
        self.env = env

    @abstractmethod
    def act(self, observation: np.ndarray | int, training: bool = False) -> int | float | np.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Action to take in the environment (int for discrete, float/array for continuous)
        """
        pass

    @abstractmethod
    def learn(
        self,
        observation: np.ndarray | int,
        action: int | float | np.ndarray,
        reward: float,
        next_observation: np.ndarray | int,
        done: bool,
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

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """
        Evaluate the model on its configured environment.

        Args:
            num_episodes: Number of episodes to run

        Returns:
            Evaluation metrics

        Raises:
            ValueError: If model is not configured with an environment
        """
        if self.env is None:
            msg = "Model not configured with an environment. Call configure() first."
            raise ValueError(msg)

        if not self.is_trained:
            logger.warning(f"Model {self.model_name} has not been trained yet")

        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.act(observation, training=False)
                observation, reward, terminated, truncated, _ = self.env.step(action)
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

        logger.info(f"Evaluation results for {self.model_name}: {metrics}")
        return metrics

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', trained={self.is_trained})"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()


def get_available_models() -> dict[str, type[RLModel]]:
    """
    Discover and import all available models dynamically.

    This function scans the models directory for subdirectories containing
    model implementations and imports them automatically.

    Returns:
        Dictionary mapping model names to their class types
    """
    models_dict: dict[str, type[RLModel]] = {}

    # Get the path to the models directory
    models_dir = Path(__file__).parent

    # Iterate through all subdirectories in the models directory
    for item in models_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_") and item.name != "__pycache__":
            module_name = f"hercule.models.{item.name}"

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Look for classes that inherit from RLModel
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a class that inherits from RLModel
                    if isinstance(attr, type) and issubclass(attr, RLModel) and attr != RLModel:
                        # Use the model_name class attribute or fallback to class name
                        model_name = getattr(attr, "model_name", attr.__name__.lower())
                        models_dict[model_name] = attr

                        logger.debug(f"Discovered model: {model_name} -> {attr.__name__}")

            except ImportError as e:
                logger.warning(f"Failed to import module {module_name}: {e}")
            except Exception as e:
                logger.warning(f"Error processing module {module_name}: {e}")

    logger.info(f"Discovered {len(models_dict)} models: {list(models_dict.keys())}")
    return models_dict


def create_model(model_name: str, **kwargs) -> RLModel:
    """
    Create a model instance by name.

    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not found
    """
    available_models = get_available_models()

    if model_name not in available_models:
        available_names = list(available_models.keys())
        msg = f"Model '{model_name}' not found. Available models: {available_names}"
        raise ValueError(msg)

    model_class = available_models[model_name]
    return model_class(**kwargs)
