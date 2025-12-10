"""Abstract base classes and interfaces for reinforcement learning models."""

import importlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Final, cast, final

import gymnasium as gym
import numpy as np

from hercule.config import BaseConfig, HyperParameter, ParameterValue
from hercule.models.epoch_result import EpochResult


logger = logging.getLogger(__name__)

model_file_name: Final = "model.json"


class RLModel(BaseConfig, ABC):
    """Abstract base class for reinforcement learning models."""

    # Class attribute for model name (static, immutable)
    model_name: ClassVar[str]
    # Class attribute for default hyperparameters (static, immutable) - must be overridden by subclasses
    default_hyperparameters: ClassVar[dict[str, ParameterValue]] = {}

    def __init__(self, **kwargs) -> None:
        """
        Initialize the RL model.

        The model inherits from BaseConfig, so it has:
        - name: str (initialized from model_name)
        - hyperparameters: list[HyperParameter] (empty by default)
        """
        # Initialize BaseConfig with model_name as name if not provided
        if "name" not in kwargs:
            kwargs["name"] = self.model_name
        super().__init__(**kwargs)
        self.env: gym.Env | None = None

    def get_default_hyperparameters(self) -> dict[str, ParameterValue]:
        """
        Get default hyperparameters for this model.

        Returns the class attribute default_hyperparameters, which should be
        defined by each model subclass.

        Returns:
            Dictionary of default hyperparameters
        """
        return self.__class__.default_hyperparameters.copy()

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure the model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (will be merged with defaults)

        Note:
            This method merges provided hyperparameters with defaults and stores
            them in self.hyperparameters (list[HyperParameter]) for later retrieval.
        """
        # Merge with defaults
        defaults = self.get_default_hyperparameters()
        merged = defaults.copy()
        merged.update(hyperparameters)

        # Store hyperparameters in BaseConfig format
        self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in merged.items()]

        self.env = env
        return True

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

    @final
    def check_environment_or_raise(self) -> gym.Env:
        if self.env is None:
            raise ValueError(f"Environment not configured for {self.model_name}. Call configure() first.")
        return cast("gym.Env", self.env)

    @abstractmethod
    def run_epoch(self, train_mode=False) -> EpochResult:
        pass

    @final
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path where to save the model
        """
        path.mkdir(parents=True, exist_ok=True)

        # Get model data from implementation
        model_data = self._export()

        # Save as JSON
        model_file = path / model_file_name
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        logger.info(f"'{self.model_name}' model saved to {path} (JSON: {model_file})")

    @abstractmethod
    def _export(self) -> dict:
        """
        Export model data for serialization.

        Returns:
            Dictionary containing model data ready for JSON serialization
        """
        pass

    @final
    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        model_file = path / model_file_name

        if not model_file.exists():
            logger.info(f"No {self.model_name} model found at {path} (looked for {model_file})")
            return

        with open(model_file, encoding="utf-8") as f:
            model_data = json.load(f)

        # Import model data using implementation
        self._import(model_data)

        logger.info(f"Loaded {self.model_name} model from JSON: {model_file}")

    @abstractmethod
    def _import(self, model_data: dict) -> None:
        """
        Import model data from serialized format.

        Args:
            model_data: Dictionary containing model data from JSON
        """
        pass

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
        return f"{self.__class__.__name__}(name='{self.model_name}')"

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
