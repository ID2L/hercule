"""Abstract base classes and interfaces for reinforcement learning models."""

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Final, Generic, TypeVar, cast, final

import gymnasium as gym
import numpy as np
from pydantic import ConfigDict, Field

from hercule.config import BaseConfig, HyperParameter, HyperParamsBase, ParameterValue
from hercule.environnements.spaces_checker import SpaceKind, classify_space
from hercule.models.epoch_result import EpochResult


# Type variable for hyperparameters
HyperParamsType = TypeVar("HyperParamsType", bound=HyperParamsBase)


logger = logging.getLogger(__name__)

model_file_name: Final = "model.json"


class RLModel(BaseConfig, ABC, Generic[HyperParamsType]):
    """
    Abstract base class for reinforcement learning models.

    This class is generic over HyperParamsType, allowing type-safe hyperparameters
    for each model subclass.

    Type Parameters:
        HyperParamsType: The type of hyperparameters for this model (must inherit from HyperParamsBase)
    """

    # Pydantic configuration to allow arbitrary types (like gym.Env, np.ndarray, etc.)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class attribute for model name (static, immutable)
    model_name: ClassVar[str]
    # Class attribute for the hyperparameters type class (for type-safe access)
    hyperparams_class: ClassVar[type[HyperParamsBase] | None] = None
    # The (observation_kind, action_kind) pairs this model can be configured on.
    #
    # Deliberately declared WITHOUT a default. A permissive default ("supports
    # everything") would recreate the silent-accept this mechanism exists to remove:
    # a new model whose author forgets the declaration would be paired with any
    # environment and fail later with an unrelated error. `get_available_models()`
    # refuses to register a concrete model that does not declare it.
    supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]]

    # Environment field (Pydantic field to allow gym.Env type)
    env: gym.Env | None = Field(default=None, description="Gymnasium environment")

    # Instance attribute for typed hyperparameters (set after configure)
    _typed_hyperparameters: HyperParamsType | None = None

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
        self._typed_hyperparameters = None

    def get_default_hyperparameters(self) -> dict[str, ParameterValue]:
        """
        Get default hyperparameters for this model as a dictionary.

        Returns default hyperparameters converted from the typed instance.

        Returns:
            Dictionary of default hyperparameters
        """
        return self.get_default_hyperparameters_typed().to_dict()

    def get_default_hyperparameters_typed(self) -> HyperParamsType:
        """
        Get default hyperparameters for this model as a typed instance.

        Returns a type-safe instance of hyperparameters with default values.
        This provides autocomplete and type checking in IDEs.

        Returns:
            Typed hyperparameters instance with default values

        Raises:
            ValueError: If hyperparams_class is not defined
        """
        if self.hyperparams_class is None:
            msg = f"hyperparams_class not defined for {self.model_name}"
            raise ValueError(msg)
        # Create instance with default values (Pydantic will use Field defaults)
        return self.hyperparams_class()

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue] | HyperParamsType) -> bool:
        """
        Configure the model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (dict or typed HyperParamsBase instance)
                           Will be merged with defaults.

        Note:
            This method merges provided hyperparameters with defaults and stores
            them in self.hyperparameters (list[HyperParameter]) for later retrieval.
            It also stores typed hyperparameters in self._typed_hyperparameters.
        """
        # Convert to dict if it's a HyperParamsBase instance
        if isinstance(hyperparameters, HyperParamsBase):
            hyperparams_dict = hyperparameters.to_dict()
            # Store typed instance for type-safe access
            self._typed_hyperparameters = hyperparameters
        else:
            hyperparams_dict = hyperparameters
            # Create typed instance if hyperparams_class is defined
            if self.hyperparams_class is not None:
                # Merge with defaults first
                defaults_dict = self.get_default_hyperparameters()
                merged = defaults_dict.copy()
                merged.update(hyperparams_dict)
                # Create typed instance (should always succeed if hyperparams_class is defined)
                self._typed_hyperparameters = self.hyperparams_class(**merged)
            else:
                # No hyperparams_class defined, keep None
                self._typed_hyperparameters = None

        # Merge with defaults
        defaults_dict = self.get_default_hyperparameters()
        merged = defaults_dict.copy()
        merged.update(hyperparams_dict)

        # Store hyperparameters in BaseConfig format (for generic access)
        self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in merged.items()]

        self.env = env
        return True

    def get_hyperparameters(self) -> HyperParamsType:
        """
        Get typed hyperparameters for this model.

        Returns the type-safe hyperparameters instance.
        This provides autocomplete and type checking in IDEs.

        Returns:
            Typed hyperparameters instance

        Raises:
            ValueError: If model is not configured or hyperparams_class is not defined
        """
        if self._typed_hyperparameters is None:
            msg = f"Model {self.model_name} not configured or hyperparams_class not defined. Call configure() first."
            raise ValueError(msg)
        return self._typed_hyperparameters

    @classmethod
    def supports_environment(cls, env: gym.Env) -> bool:
        """
        Whether this model declares support for the environment's space pair.

        Args:
            env: The environment the model would be configured on.

        Returns:
            True when `(observation_kind, action_kind)` is in `supported_spaces`.

        Raises:
            AttributeError: If the class never declared `supported_spaces`. That is a
                programming error, not a runtime condition, so it is not softened into
                a permissive True.
        """
        pair = (classify_space(env.observation_space), classify_space(env.action_space))
        return pair in cls.supported_spaces

    @classmethod
    def describe_space_mismatch(cls, env: gym.Env) -> str:
        """
        One line naming what the model expects and what the environment offers.

        The message has to carry both halves: "requires a discrete action space" alone
        does not tell the reader what the environment actually provides, which is the
        thing they need in order to fix their YAML.
        """
        observation_kind = classify_space(env.observation_space)
        action_kind = classify_space(env.action_space)
        expected = ", ".join(sorted(f"(observation={o}, action={a})" for o, a in cls.supported_spaces))
        return (
            f"Model '{cls.model_name}' does not support this environment: "
            f"got (observation={observation_kind}, action={action_kind}), supports {expected}"
        )

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

    def begin_episode(self) -> None:
        """
        Signal the start of a new episode, before its first observation is fed.

        The default implementation does nothing: a policy that depends only on the
        current observation has no per-episode state. Models carrying state ACROSS
        steps -- observation stacking, recurrent policies -- MUST override this and
        reset that state here, and every caller driving its own episode loop MUST
        call it right after `env.reset()`.

        Without it, `predict()` cannot tell where an episode begins: it receives one
        observation and nothing else, so a stacked state would silently span two
        episodes and the policy would act on an observation that never existed.
        """

    @final
    def check_environment_or_raise(self) -> gym.Env:
        if self.env is None:
            raise ValueError(f"Environment not configured for {self.model_name}. Call configure() first.")
        return cast("gym.Env", self.env)

    @abstractmethod
    def run_epoch(self, train_mode=False) -> EpochResult:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray | int) -> int | float | np.ndarray:
        """
        Predict the best action for a given observation (inference mode).

        This method should be used for inference/evaluation, not during training.
        It typically calls act() with training=False.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action (int for discrete, float/array for continuous)
        """
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

        # Add model name to exported data for proper deserialization
        model_data["model_name"] = self.model_name

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
            self.begin_episode()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # predict(), not act(): it is the stateful inference entry point, so
                # a model with observation stacking gets its history maintained. For
                # a stateless model predict() is exactly act(training=False).
                action = self.predict(observation)
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


def _is_registrable(attr: object) -> bool:
    """
    Whether a scanned module attribute should be registered as an available model.

    A registrable attribute is a concrete `RLModel` subclass (not `RLModel` itself,
    not abstract) that declares its own `model_name` -- never inherited from an
    abstract parent, since that would collapse every subclass onto the same name --
    and declares `supported_spaces`, own or inherited (e.g. `simple_q_learning` and
    `simple_sarsa` share the declaration on `TDModel`).

    Args:
        attr: Candidate object found while scanning a models subpackage module.

    Returns:
        True if `attr` should be registered as an available model.
    """
    if not (isinstance(attr, type) and issubclass(attr, RLModel) and attr is not RLModel):
        return False
    if inspect.isabstract(attr):
        return False
    if "model_name" not in attr.__dict__:
        logger.warning(f"Skipping '{attr.__module__}.{attr.__name__}': no own 'model_name' declared.")
        return False
    if not hasattr(attr, "supported_spaces"):
        logger.warning(
            f"Skipping model '{attr.__dict__['model_name']}' ({attr.__module__}.{attr.__name__}): "
            "no 'supported_spaces' declared."
        )
        return False
    return True


def get_available_models() -> dict[str, type[RLModel]]:
    """
    Discover and import all available models dynamically.

    This function scans the models directory for subdirectories containing
    model implementations and imports them automatically. Only concrete
    `RLModel` subclasses that declare their own `model_name` and a
    `supported_spaces` declaration are registered; see `_is_registrable`.

    Returns:
        Dictionary mapping model names to their class types
    """
    models_dict: dict[str, type[RLModel]] = {}

    # Get the path to the models directory
    models_dir = Path(__file__).parent

    # Iterate through all subdirectories in the models directory
    for item in models_dir.iterdir():
        if not (item.is_dir() and not item.name.startswith("_") and item.name != "__pycache__"):
            continue

        module_name = f"hercule.models.{item.name}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import module {module_name}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error processing module {module_name}: {e}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not _is_registrable(attr):
                continue

            model_name = attr.__dict__["model_name"]
            models_dict[model_name] = attr
            logger.debug(f"Discovered model: {model_name} -> {attr.__name__}")

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
