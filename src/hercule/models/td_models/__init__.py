"""Abstract Temporal Difference (TD) model for reinforcement learning algorithms."""

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import gymnasium as gym
import numpy as np
from pydantic import Field, PrivateAttr

from hercule.config import HyperParameter, HyperParamsBase, ParameterValue
from hercule.environnements.spaces_checker import check_space_is_discrete
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


if TYPE_CHECKING:
    from gymnasium.spaces import Discrete
else:
    Discrete = object  # Placeholder for runtime


logger = logging.getLogger(__name__)


class TDModelHyperParams(HyperParamsBase):
    """Type-safe hyperparameters for TD learning models."""

    learning_rate: float = Field(default=0.1, description="Learning rate (alpha)")
    discount_factor: float = Field(default=0.95, description="Discount factor (gamma)")
    epsilon: float = Field(default=1.0, description="Initial epsilon for epsilon-greedy")
    epsilon_decay: float = Field(default=0.0, description="Epsilon decay rate per epoch")
    epsilon_min: float = Field(default=0.0, description="Minimum epsilon value")
    seed: int = Field(default=42, description="Random seed")


class TDModel(RLModel[TDModelHyperParams], ABC):
    """
    Abstract base class for Temporal Difference (TD) learning algorithms.

    This class provides the common structure and functionality for TD algorithms
    like SARSA and Q-Learning, which both use Q-tables and temporal difference
    updates. The main difference between implementations is in the update method.
    """

    # Default hyperparameters for TD models (static, immutable) - SARSA, Q-Learning, etc.
    default_hyperparameters: ClassVar[dict[str, ParameterValue]] = {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.0,
        "epsilon_min": 0.0,
        "seed": 42,
    }
    # Type-safe hyperparameters class
    hyperparams_class: ClassVar[type[HyperParamsBase]] = TDModelHyperParams

    # Private attributes (not Pydantic fields, use PrivateAttr to avoid validation)
    _q_table: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    _action_space: Discrete | None = PrivateAttr(default=None)
    _observation_space: Discrete | None = PrivateAttr(default=None)
    _seed: np.random.Generator = PrivateAttr(default_factory=lambda: np.random.default_rng(42))

    def __init__(self) -> None:
        """Initialize the TD model."""
        super().__init__()
        # Hyperparameters will be initialized in configure() from self.hyperparameters

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure the TD model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (will be merged with defaults)

        Returns:
            True if configuration successful, False otherwise

        Raises:
            ValueError: If environment does not have discrete action and observation spaces
        """
        # Validate environment has discrete spaces
        if not check_space_is_discrete(env.action_space) or not check_space_is_discrete(env.observation_space):
            return False

        # Configure base class (this will merge with defaults and store in self.hyperparameters)
        super().configure(env, hyperparameters)
        # Store environment spaces
        self._action_space = cast("Discrete", env.action_space)
        self._observation_space = cast("Discrete", env.observation_space)

        self._q_table = np.zeros((self._observation_space.n, self._action_space.n), dtype=np.float64)

        # Get typed hyperparameters (type-safe access)
        typed_params = self.get_hyperparameters()
        lr = typed_params.learning_rate
        df = typed_params.discount_factor
        eps = typed_params.epsilon
        eps_decay = typed_params.epsilon_decay
        eps_min = typed_params.epsilon_min
        seed_value = typed_params.seed

        # Validate hyperparameters
        validated_lr = self._validate_hyperparameter(float(lr), "learning_rate", 0.0, 1.0)
        validated_df = self._validate_hyperparameter(float(df), "discount_factor", 0.0, 1.0)
        validated_eps = self._validate_hyperparameter(float(eps), "epsilon", 0.0, 1.0)
        validated_eps_decay = self._validate_hyperparameter(float(eps_decay), "epsilon_decay", 0.0, 1.0)
        validated_eps_min = self._validate_hyperparameter(float(eps_min), "epsilon_min", 0.0, 1.0)

        # Update typed hyperparameters with validated values
        typed_params.learning_rate = validated_lr
        typed_params.discount_factor = validated_df
        typed_params.epsilon = validated_eps
        typed_params.epsilon_decay = validated_eps_decay
        typed_params.epsilon_min = validated_eps_min

        # Set random seed from hyperparameters
        if isinstance(seed_value, int):
            self._seed = np.random.default_rng(seed_value)

        # Update self.hyperparameters with validated values from typed params
        self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]

        logger.info(
            f"'{self.model_name}' configured for environment with "
            f"discrete action space ({self._action_space.n} actions) and "
            f"discrete observation space ({self._observation_space.n} states)"
        )
        return True

    def run_epoch(self, train_mode=False) -> EpochResult:
        """
        Run a single epoch/episode using the TD algorithm.

        Args:
            train_mode: Whether to update the model during the episode

        Returns:
            EpochResult containing episode statistics
        """
        env = self.check_environment_or_raise()

        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        action = self.act(observation, training=train_mode)
        while not done:
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_action = self.act(next_observation, training=train_mode)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            if train_mode:
                self.update(observation, action, reward, next_observation, next_action)
                # Update epsilon (decay) - use typed hyperparameters
                typed_params = self.get_hyperparameters()
                current_epsilon = typed_params.epsilon
                epsilon_decay = typed_params.epsilon_decay
                epsilon_min = typed_params.epsilon_min
                new_epsilon = max(epsilon_min, current_epsilon * (1 - epsilon_decay))
                # Update typed hyperparameters
                typed_params.epsilon = new_epsilon
                # Update self.hyperparameters list to reflect the change
                self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]

            observation = next_observation
            action = next_action

        return EpochResult(
            reward=float(episode_reward),
            steps_number=episode_length,
            final_state="truncated" if truncated else "terminated",
        )

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        """
        Abstract method for updating Q-values using temporal difference learning.

        This is the core method that differs between TD algorithms:
        - SARSA: Uses the actual next action taken (on-policy)
        - Q-Learning: Uses the best action for the next state (off-policy)

        Args:
            state: Current state
            action: Action taken in current state
            reward: Reward received
            next_state: Next state reached
            next_action: Next action taken (for SARSA) or best action (for Q-Learning)
        """
        pass

    def _validate_hyperparameter(self, value: float, name: str, min_val: float, max_val: float) -> float:
        """
        Validate that a hyperparameter is within the specified range.

        Args:
            value: Hyperparameter value to validate
            name: Name of the hyperparameter for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated value

        Raises:
            ValueError: If value is outside the valid range
        """
        if not (min_val <= value <= max_val):
            msg = f"{name} must be between {min_val} and {max_val}, got {value}"
            raise ValueError(msg)
        return value

    def _get_state_key(self, observation: np.ndarray | int) -> tuple:
        """
        Convert observation to a hashable state key for Q-table lookup.

        Args:
            observation: Environment observation (can be numpy array or int for discrete spaces)

        Returns:
            Hashable state key
        """
        # For discrete spaces, the observation can be an integer or numpy array
        if isinstance(observation, int):
            return (observation,)
        elif isinstance(observation, np.ndarray):
            if observation.ndim == 0:  # Scalar array
                return (int(observation),)
            else:  # Multi-dimensional array
                return tuple(int(x) for x in observation.flatten())
        else:
            # Fallback: convert to int and wrap in tuple
            return (int(observation),)

    def _get_q_value(self, state: int, action: int) -> float:
        """
        Get Q-value for a state-action pair.

        Args:
            state: State key
            action: Action index

        Returns:
            Q-value (0.0 if not initialized)
        """
        return self._q_table[state][action]

    def _set_q_value(self, state: int, action: int, value: float) -> None:
        """
        Set Q-value for a state-action pair.

        Args:
            state: State key
            action: Action index
            value: Q-value to set
        """
        self._q_table[state][action] = value

    def exploit(self, state: int) -> int:
        """
        Select the best action for a given state (exploitation).

        Args:
            state: Current state

        Returns:
            Action index with highest Q-value
        """
        if self._action_space.n is None:
            raise ValueError("Number of actions not set")
        action_state_value = self._q_table[state]
        maximum_value = max(action_state_value)
        possible_action_index = [i for i, j in enumerate(action_state_value) if j == maximum_value]
        random_between_maximum_value = random.randint(0, len(possible_action_index) - 1)
        return possible_action_index[random_between_maximum_value]

    def _epsilon_greedy_action(self, state: int, training: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (affects epsilon usage)

        Returns:
            Selected action index
        """
        # Get epsilon from hyperparameters - use typed hyperparameters
        typed_params = self.get_hyperparameters()
        epsilon = typed_params.epsilon
        rand = random.random()
        if rand < epsilon:
            # Explore: choose random action
            if self._action_space.n is None:
                raise ValueError("Number of actions not set")
            return random.randint(0, self._action_space.n - 1)
        else:
            return self.exploit(state)

    def act(self, observation: int, training: bool = False) -> int:
        """
        Select an action given an observation using epsilon-greedy policy.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Selected action index

        Raises:
            ValueError: If model is not configured
        """
        if self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        state = observation
        if training:
            return self._epsilon_greedy_action(state, training)
        else:
            return self.exploit(state)

    def _export(self) -> dict:
        """
        Export TD model data for serialization.

        Returns:
            Dictionary containing model data ready for JSON serialization
        """
        # Convert Q-table to JSON-serializable format
        json_q_table = self._q_table.tolist()

        # Prepare model data for JSON serialization
        model_data = {
            "q_table": json_q_table,
        }

        return model_data

    def _import(self, model_data: dict) -> None:
        """
        Import TD model data from serialized format.

        Args:
            model_data: Dictionary containing model data from JSON
        """
        # Convert Q-table back from JSON format
        self._q_table = np.array(model_data["q_table"])

        # Note: Environment needs to be configured separately after loading
        # as it cannot be serialized
        logger.info("Note: Call configure() with environment before using loaded model")

    def load_from_dict(self, model_data: dict) -> None:
        """
        Load a trained TD model from a dictionary.

        Args:
            model_data: Dictionary containing model data

        Raises:
            KeyError: If required keys are missing from model_data
        """
        # Convert Q-table back from JSON format
        self.q_table = np.array(model_data["q_table"])

        logger.info(f"Loaded {self.model_name} model from dictionary")

        # Note: Environment needs to be configured separately after loading
        # as it cannot be serialized
        logger.info("Note: Call configure() with environment before using loaded model")

    def predict(self, observation: np.ndarray | int) -> int:
        """
        Predict the best action for a given observation (inference mode).

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action index
        """
        return self.act(observation, training=False)


__all__ = ["TDModel"]
