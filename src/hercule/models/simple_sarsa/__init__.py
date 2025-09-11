"""Simple SARSA implementation with Q-table for discrete environments."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue
from hercule.environnements.spaces_checker import check_space_is_discrete
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


if TYPE_CHECKING:
    from gymnasium.spaces import Discrete


logger = logging.getLogger(__name__)


class SimpleSarsaModel(RLModel):
    """
    Simple SARSA (State-Action-Reward-State-Action) implementation with Q-table.

    This model is designed for environments with discrete action and observation spaces.
    It uses a Q-table to store state-action values and updates them using the SARSA algorithm.

    SARSA is an on-policy temporal difference learning algorithm that learns the Q-values
    for the policy being followed, rather than the optimal policy.
    """

    # Class attribute for model name
    model_name: str = "simple_sarsa"

    def __init__(self) -> None:
        """Initialize the SARSA model."""
        super().__init__()
        self._q_table: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._action_space: Discrete
        self._observation_space: Discrete
        self.seed = np.random.default_rng(42)

        # Hyperparameters
        self._learning_rate: float = 0.1
        self._discount_factor: float = 0.95
        self._epsilon: float = 1.0  # Default to 1.0 for no decay behavior
        self._epsilon_decay: float = 0.0  # Default to 0.0 for no decay
        self._epsilon_min: float = 0.0  # Default to 0.0

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure the SARSA model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters

        Raises:
            ValueError: If environment does not have discrete action and observation spaces
        """
        # Validate environment has discrete spaces
        if not check_space_is_discrete(env.action_space) or not check_space_is_discrete(env.observation_space):
            return False

        super().configure(env, hyperparameters)
        # Store environment spaces
        self._action_space = cast("Discrete", env.action_space)
        self._observation_space = cast("Discrete", env.observation_space)

        self._q_table = np.zeros((self._observation_space.n, self._action_space.n), dtype=np.float64)
        # Set hyperparameters with validation
        lr = hyperparameters.get("learning_rate", 0.1)
        self._learning_rate = self._validate_hyperparameter(
            float(lr) if isinstance(lr, int | float) else 0.1, "learning_rate", 0.0, 1.0
        )

        df = hyperparameters.get("discount_factor", 0.95)
        self._discount_factor = self._validate_hyperparameter(
            float(df) if isinstance(df, int | float) else 0.95, "discount_factor", 0.0, 1.0
        )

        eps = hyperparameters.get("epsilon", 1.0)
        self._epsilon = self._validate_hyperparameter(
            float(eps) if isinstance(eps, int | float) else 1.0, "epsilon", 0.0, 1.0
        )

        eps_decay = hyperparameters.get("epsilon_decay", 0.0)
        self._epsilon_decay = self._validate_hyperparameter(
            float(eps_decay) if isinstance(eps_decay, int | float) else 0.0, "epsilon_decay", 0.0, 1.0
        )

        eps_min = hyperparameters.get("epsilon_min", 0.0)
        self._epsilon_min = self._validate_hyperparameter(
            float(eps_min) if isinstance(eps_min, int | float) else 0.0, "epsilon_min", 0.0, 1.0
        )

        # Set random seed if provided
        if "seed" in hyperparameters:
            seed_value = hyperparameters["seed"]
            if isinstance(seed_value, int):
                self.seed = np.random.default_rng(seed_value)

        logger.info(
            f"SARSA model '{self.model_name}' configured for environment with "
            f"discrete action space ({self._action_space.n} actions) and "
            f"discrete observation space ({self._observation_space.n} states)"
        )
        return True

    def run_epoch(self, train_mode=False) -> EpochResult:
        env = self.check_environment_or_raise()

        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        action = self.act(observation, training=train_mode)
        while not done:
            next_action = self.act(observation, training=train_mode)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            if train_mode:
                self._q_table[observation][action] += self._learning_rate * (
                    float(reward)
                    + self._discount_factor
                    + self._q_table[next_observation, next_action]
                    - self._q_table[observation, action]
                )

            observation = next_observation
            action = next_action

        return EpochResult(
            reward=float(episode_reward),
            steps_number=episode_length,
            final_state="truncated" if truncated else "goal_reached",
        )

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

    def _epsilon_greedy_action(self, state: int, training: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (affects epsilon usage)

        Returns:
            Selected action index
        """
        if not training or self.seed.random() > self._epsilon:
            # Exploit: choose best action
            if self._action_space.n is None:
                raise ValueError("Number of actions not set")
            q_values = [self._get_q_value(state, a) for a in range(self._action_space.n - 1)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return self.seed.choice(best_actions)
        else:
            # Explore: choose random action
            if self._action_space.n is None:
                raise ValueError("Number of actions not set")
            return int(self.seed.integers(0, self._action_space.n))

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
        return self._epsilon_greedy_action(state, training)

    def save(
        self,
        path: Path,
        environment_name: str | None = None,
        environment_hyperparameters: dict | None = None,
        model_hyperparameters: dict | None = None,
    ) -> None:
        """
        Save the trained SARSA model to disk in JSON format.

        Args:
            path: Path where to save the model
            environment_name: Name of the environment used for training
            environment_hyperparameters: Environment hyperparameters used
            model_hyperparameters: Model hyperparameters used for training
        """
        path.mkdir(parents=True, exist_ok=True)

        # Convert Q-table to JSON-serializable format
        json_q_table = self._q_table.tolist()

        # Prepare model data for JSON serialization
        model_data = {
            "q_table": json_q_table,
        }

        # Save as JSON
        model_file = path / "model.json"
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        logger.info(f"SARSA model saved to {path} (JSON: {model_file})")

    def load(self, path: Path) -> None:
        """
        Load a trained SARSA model from disk.

        Args:
            path: Path to the saved model

        Raises:
            FileNotFoundError: If model file is not found
        """
        json_file = path / "sarsa_model.json"

        # Try to load JSON format first, fall back to pickle
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                model_data = json.load(f)

            # Convert Q-table back from JSON format
            self._q_table = np.array(model_data["q_table"])

            logger.info(f"Loaded SARSA model from JSON: {json_file}")

        else:
            msg = f"No SARSA model found at {path} (looked for {json_file})"
            logger.info(msg)

        # Note: Environment needs to be configured separately after loading
        # as it cannot be serialized
        logger.info("Note: Call configure() with environment before using loaded model")

    def load_from_dict(self, model_data: dict) -> None:
        """
        Load a trained SARSA model from a dictionary.

        Args:
            model_data: Dictionary containing model data

        Raises:
            KeyError: If required keys are missing from model_data
        """
        # Convert Q-table back from JSON format
        self._q_table = np.array(model_data["q_table"])

        logger.info("Loaded SARSA model from dictionary")

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


__all__ = ["SimpleSarsaModel"]
