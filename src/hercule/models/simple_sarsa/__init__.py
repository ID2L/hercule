"""Simple SARSA implementation with Q-table for discrete environments."""

import logging
import pickle
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue
from hercule.environnements import DiscreteSpaceInfo, EnvironmentInspector
from hercule.models import RLModel


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
        self._q_table: dict[tuple, dict[int, float]] = {}
        self._action_space: gym.Space | None = None
        self._observation_space: gym.Space | None = None
        self._num_actions: int | None = None
        self._rng = np.random.default_rng(42)

        # Hyperparameters
        self._learning_rate: float = 0.1
        self._discount_factor: float = 0.95
        self._epsilon: float = 1.0  # Default to 1.0 for no decay behavior
        self._epsilon_decay: float = 0.0  # Default to 0.0 for no decay
        self._epsilon_min: float = 0.0  # Default to 0.0

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> None:
        """
        Configure the SARSA model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters

        Raises:
            ValueError: If environment does not have discrete action and observation spaces
        """
        super().configure(env, hyperparameters)

        # Validate environment has discrete spaces
        self._validate_discrete_environment(env)

        # Store environment spaces
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        # Get number of actions from discrete action space
        # Use cast to inform type checker that action_space has 'n' attribute after validation
        action_space = cast("gym.spaces.Discrete", env.action_space)
        self._num_actions = int(action_space.n)

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
                self._rng = np.random.default_rng(seed_value)

        logger.info(
            f"SARSA model '{self.model_name}' configured for environment with "
            f"discrete action space ({self._num_actions} actions) and "
            f"discrete observation space"
        )

    def _validate_discrete_environment(self, env: gym.Env) -> None:
        """
        Validate that the environment has discrete action and observation spaces.

        Args:
            env: Gymnasium environment to validate

        Raises:
            ValueError: If environment does not have discrete spaces
        """
        inspector = EnvironmentInspector()
        env_info = inspector.get_environment_info(env)

        # Check action space
        if not isinstance(env_info.action_space, DiscreteSpaceInfo):
            msg = f"SARSA requires discrete action space, got {env_info.action_space.type}"
            raise ValueError(msg)

        # Check observation space
        if not isinstance(env_info.observation_space, DiscreteSpaceInfo):
            msg = f"SARSA requires discrete observation space, got {env_info.observation_space.type}"
            raise ValueError(msg)

        logger.debug(f"Environment validation passed: {env_info.name}")

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

    def _get_q_value(self, state: tuple, action: int) -> float:
        """
        Get Q-value for a state-action pair.

        Args:
            state: State key
            action: Action index

        Returns:
            Q-value (0.0 if not initialized)
        """
        return self._q_table.get(state, {}).get(action, 0.0)

    def _set_q_value(self, state: tuple, action: int, value: float) -> None:
        """
        Set Q-value for a state-action pair.

        Args:
            state: State key
            action: Action index
            value: Q-value to set
        """
        if state not in self._q_table:
            self._q_table[state] = {}
        self._q_table[state][action] = value

    def _epsilon_greedy_action(self, state: tuple, training: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (affects epsilon usage)

        Returns:
            Selected action index
        """
        if not training or self._rng.random() > self._epsilon:
            # Exploit: choose best action
            if self._num_actions is None:
                raise ValueError("Number of actions not set")
            q_values = [self._get_q_value(state, a) for a in range(self._num_actions)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return self._rng.choice(best_actions)
        else:
            # Explore: choose random action
            if self._num_actions is None:
                raise ValueError("Number of actions not set")
            return int(self._rng.integers(0, self._num_actions))

    def act(self, observation: np.ndarray | int, training: bool = False) -> int:
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

        state = self._get_state_key(observation)
        return self._epsilon_greedy_action(state, training)

    def learn(
        self, observation: np.ndarray | int, action: int, reward: float, next_observation: np.ndarray | int, done: bool
    ) -> None:
        """
        Update Q-values using SARSA algorithm.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is finished
        """
        if self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        current_state = self._get_state_key(observation)
        next_state = self._get_state_key(next_observation)

        # Get current Q-value
        current_q = self._get_q_value(current_state, action)

        if done:
            # Terminal state: no next action
            next_q = 0.0
        else:
            # Non-terminal state: get Q-value for next state-action pair
            next_action = self._epsilon_greedy_action(next_state, training=True)
            next_q = self._get_q_value(next_state, next_action)

        # SARSA update rule
        new_q = current_q + self._learning_rate * (reward + self._discount_factor * next_q - current_q)
        self._set_q_value(current_state, action, new_q)

    def train(self, env: gym.Env, config: dict[str, ParameterValue], max_iterations: int) -> dict[str, ParameterValue]:
        """
        Train the SARSA model on the given environment.

        Args:
            env: Gymnasium environment
            config: Model hyperparameters
            max_iterations: Maximum number of training iterations

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
        episode_rewards = []
        current_episode_reward = 0.0
        current_episode_length = 0

        logger.info(f"Starting SARSA training for {max_iterations} iterations")

        # Initialize episode
        observation, _ = self.env.reset()
        state = self._get_state_key(observation)
        action = self._epsilon_greedy_action(state, training=True)

        for iteration in range(max_iterations):
            # Take action and observe result
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Learn from this transition
            self.learn(observation, action, float(reward), next_observation, done)

            # Update episode tracking
            current_episode_reward += float(reward)
            current_episode_length += 1

            if done:
                # Episode finished
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                rewards.append(current_episode_reward)

                # Reset for next episode
                observation, _ = self.env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0
            else:
                # Continue episode
                observation = next_observation

            # Select next action using SARSA (on-policy)
            state = self._get_state_key(observation)
            action = self._epsilon_greedy_action(state, training=True)

            # Decay epsilon (only if epsilon_decay > 0)
            if self._epsilon_decay > 0:
                self._epsilon = max(self._epsilon_min, self._epsilon * (1.0 - self._epsilon_decay))

            # Log progress every 10% of iterations
            if (iteration + 1) % max(1, max_iterations // 10) == 0:
                recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                logger.info(
                    f"Iteration {iteration + 1}/{max_iterations}, "
                    f"Epsilon: {self._epsilon:.3f}, "
                    f"Avg reward (last 100): {avg_reward:.2f}"
                )

        self.is_trained = True

        # Calculate training metrics
        metrics: dict[str, ParameterValue] = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
            "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
            "episodes_completed": len(episode_rewards),
            "final_epsilon": self._epsilon,
            "q_table_size": len(self._q_table),
        }

        self._training_metrics.update(metrics)

        logger.info(
            f"SARSA training completed. "
            f"Episodes: {len(episode_rewards)}, "
            f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}, "
            f"Final epsilon: {self._epsilon:.3f}"
        )

        return metrics

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
        json_q_table = {}
        for state, actions in self._q_table.items():
            # Convert tuple state keys to strings
            state_key = "_".join(map(str, state))
            json_q_table[state_key] = actions

        # Prepare model data for JSON serialization
        model_data = {
            "model_info": {
                "model_name": self.model_name,
                "model_type": "simple_sarsa",
                "is_trained": self.is_trained,
                "q_table_size": len(self._q_table),
                "num_actions": self._num_actions,
            },
            "environment_info": {
                "environment_name": environment_name,
                "environment_hyperparameters": environment_hyperparameters or {},
            },
            "model_hyperparameters": {
                "learning_rate": self._learning_rate,
                "discount_factor": self._discount_factor,
                "epsilon": self._epsilon,
                "epsilon_decay": self._epsilon_decay,
                "epsilon_min": self._epsilon_min,
                **(model_hyperparameters or {}),
            },
            "training_metrics": self._training_metrics,
            "q_table": json_q_table,
        }

        # Save as JSON
        model_file = path / "sarsa_model.json"
        with open(model_file, "w", encoding="utf-8") as f:
            import json

            json.dump(model_data, f, indent=2, ensure_ascii=False)

        # Also save the legacy pickle format for backward compatibility
        legacy_data = {
            "q_table": self._q_table,
            "learning_rate": self._learning_rate,
            "discount_factor": self._discount_factor,
            "epsilon": self._epsilon,
            "epsilon_decay": self._epsilon_decay,
            "epsilon_min": self._epsilon_min,
            "num_actions": self._num_actions,
            "is_trained": self.is_trained,
            "training_metrics": self._training_metrics,
        }

        legacy_file = path / "sarsa_model.pkl"
        with open(legacy_file, "wb") as f:
            pickle.dump(legacy_data, f)

        logger.info(f"SARSA model saved to {path} (JSON: {model_file}, Legacy: {legacy_file})")

    def load(self, path: Path) -> None:
        """
        Load a trained SARSA model from disk.

        Args:
            path: Path to the saved model

        Raises:
            FileNotFoundError: If model file is not found
        """
        json_file = path / "sarsa_model.json"
        pkl_file = path / "sarsa_model.pkl"

        # Try to load JSON format first, fall back to pickle
        if json_file.exists():
            import json

            with open(json_file, encoding="utf-8") as f:
                model_data = json.load(f)

            # Convert Q-table back from JSON format
            self._q_table = {}
            for state_key, actions in model_data["q_table"].items():
                # Convert string state keys back to tuples
                state_parts = state_key.split("_")
                state = tuple(int(part) for part in state_parts)
                self._q_table[state] = actions

            # Restore model state from JSON structure
            model_hyperparams = model_data["model_hyperparameters"]
            self._learning_rate = model_hyperparams["learning_rate"]
            self._discount_factor = model_hyperparams["discount_factor"]
            self._epsilon = model_hyperparams["epsilon"]
            self._epsilon_decay = model_hyperparams["epsilon_decay"]
            self._epsilon_min = model_hyperparams["epsilon_min"]
            self._num_actions = model_data["model_info"]["num_actions"]
            self.is_trained = model_data["model_info"]["is_trained"]
            self._training_metrics = model_data.get("training_metrics", {})

            logger.info(f"Loaded SARSA model from JSON: {json_file}")

        elif pkl_file.exists():
            # Load legacy pickle format
            with open(pkl_file, "rb") as f:
                model_data = pickle.load(f)

            # Restore model state
            self._q_table = model_data["q_table"]
            self._learning_rate = model_data["learning_rate"]
            self._discount_factor = model_data["discount_factor"]
            self._epsilon = model_data["epsilon"]
            self._epsilon_decay = model_data["epsilon_decay"]
            self._epsilon_min = model_data["epsilon_min"]
            self._num_actions = model_data["num_actions"]
            self.is_trained = model_data["is_trained"]
            self._training_metrics = model_data.get("training_metrics", {})

            logger.info(f"Loaded SARSA model from pickle: {pkl_file}")
        else:
            msg = f"No SARSA model found at {path} (looked for {json_file} and {pkl_file})"
            raise FileNotFoundError(msg)

        # Note: Environment needs to be configured separately after loading
        # as it cannot be serialized
        logger.info("Note: Call configure() with environment before using loaded model")


__all__ = ["SimpleSarsaModel"]
