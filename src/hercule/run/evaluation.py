"""Evaluation module for running trained models."""

import logging
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np

from hercule.config import ParameterValue
from hercule.environnements import EnvironmentManager


logger = logging.getLogger(__name__)


class EvaluationProtocol(Protocol):
    """Protocol defining the interface for RL models during evaluation."""

    def act(self, observation: np.ndarray | int, training: bool = False) -> int | float | np.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Selected action (int for discrete, float/array for continuous)
        """
        ...


class EvaluationResult:
    """Container for model evaluation results."""

    def __init__(
        self,
        environment_name: str,
        model_name: str,
        episode_rewards: list[float],
        episode_lengths: list[int],
        total_episodes: int,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """
        Initialize evaluation result.

        Args:
            environment_name: Name of the environment used
            model_name: Name of the model used
            episode_rewards: List of rewards for each episode
            episode_lengths: List of episode lengths
            total_episodes: Total number of episodes executed
            success: Whether the evaluation was successful
            error_message: Error message if evaluation failed
        """
        self.environment_name = environment_name
        self.model_name = model_name
        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths
        self.total_episodes = total_episodes
        self.success = success
        self.error_message = error_message

    @property
    def mean_reward(self) -> float:
        """Calculate mean reward across all episodes."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        """Calculate standard deviation of rewards."""
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def min_reward(self) -> float:
        """Get minimum reward."""
        return float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def max_reward(self) -> float:
        """Get maximum reward."""
        return float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def mean_length(self) -> float:
        """Calculate mean episode length."""
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def std_length(self) -> float:
        """Calculate standard deviation of episode lengths."""
        return float(np.std(self.episode_lengths)) if self.episode_lengths else 0.0

    def to_dict(self) -> dict[str, ParameterValue | list[float] | list[int] | None]:
        """Convert result to dictionary for serialization."""
        return {
            "environment_name": self.environment_name,
            "model_name": self.model_name,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_episodes": self.total_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "success": self.success,
            "error_message": self.error_message,
        }


class ModelEvaluator:
    """Handles model evaluation after training."""

    def __init__(self, env_manager: EnvironmentManager, log_level: str = "INFO"):
        """
        Initialize the model evaluator.

        Args:
            env_manager: Environment manager for loading environments
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.env_manager = env_manager
        self.log_level = log_level.upper()

        # Configure logging level
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        logger.setLevel(numeric_level)

    def evaluate_model(
        self,
        model: EvaluationProtocol,
        environment_name: str,
        model_name: str,
        num_episodes: int = 10,
        max_steps_per_episode: int | None = None,
        render: bool = False,
    ) -> EvaluationResult:
        import logging

        evaluation_logger = logging.getLogger("hercule.evaluation")
        """
        Evaluate a trained model on an environment.

        Args:
            model: Trained model implementing EvaluationProtocol
            environment_name: Name of the environment to use
            model_name: Name of the model (for logging)
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode (None for no limit)
            render: Whether to render the environment (if supported)

        Returns:
            EvaluationResult containing evaluation metrics
        """
        try:
            evaluation_logger.info(f"Starting evaluation of {model_name} on {environment_name}")
            evaluation_logger.debug(
                f"Evaluation parameters: {num_episodes} episodes, max_steps={max_steps_per_episode}"
            )

            # Load environment
            env = self.env_manager.load_environment(environment_name)
            evaluation_logger.info(f"Environment {environment_name} loaded for evaluation")

            # Enable rendering if requested
            if render:
                try:
                    env = gym.wrappers.RenderCollection(env)
                    evaluation_logger.debug("Rendering enabled")
                except Exception as e:
                    evaluation_logger.warning(f"Could not enable rendering: {e}")

            episode_rewards = []
            episode_lengths = []

            for episode in range(num_episodes):
                evaluation_logger.debug(f"Starting episode {episode + 1}/{num_episodes}")

                # Reset environment
                observation, info = env.reset()
                episode_reward = 0.0
                episode_length = 0

                # Run episode
                while True:
                    # Select action (not in training mode)
                    action = model.act(observation, training=False)

                    # Ensure action is in the correct format for the environment
                    if hasattr(env.action_space, "n"):
                        # For discrete action spaces, ensure action is an integer
                        action = int(action)
                        # Validate action is within valid range
                        if action >= env.action_space.n:
                            evaluation_logger.error(
                                f"Invalid action {action} for environment with {env.action_space.n} actions"
                            )
                            raise ValueError(f"Action {action} is out of range [0, {env.action_space.n})")
                    else:
                        # For continuous action spaces, ensure action is a numpy array
                        action = np.array(action)

                    # Take step
                    next_observation, reward, terminated, truncated, info = env.step(action)

                    # Update episode tracking
                    episode_reward += float(reward)
                    episode_length += 1

                    # Check termination conditions
                    done = terminated or truncated
                    if done or (max_steps_per_episode and episode_length >= max_steps_per_episode):
                        break

                    observation = next_observation

                # Record episode results
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                evaluation_logger.debug(
                    f"Episode {episode + 1} completed: reward={episode_reward:.2f}, length={episode_length}"
                )

                # Log progress every 25% of episodes
                if (episode + 1) % max(1, num_episodes // 4) == 0:
                    recent_rewards = episode_rewards[-max(1, num_episodes // 4) :]
                    avg_reward = np.mean(recent_rewards)
                    evaluation_logger.info(
                        f"Evaluation progress: {episode + 1}/{num_episodes} episodes, "
                        f"avg reward (last {len(recent_rewards)}): {avg_reward:.2f}"
                    )

            # Calculate final metrics
            result = EvaluationResult(
                environment_name=environment_name,
                model_name=model_name,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                total_episodes=num_episodes,
                success=True,
            )

            evaluation_logger.info(
                f"Evaluation completed for {model_name} on {environment_name}. "
                f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}, "
                f"Mean length: {result.mean_length:.1f} ± {result.std_length:.1f}"
            )

            return result

        except Exception as e:
            evaluation_logger.error(f"Evaluation failed for {model_name} on {environment_name}: {e}")
            return EvaluationResult(
                environment_name=environment_name,
                model_name=model_name,
                episode_rewards=[],
                episode_lengths=[],
                total_episodes=0,
                success=False,
                error_message=str(e),
            )

    def evaluate_model_with_hyperparams(
        self,
        model: EvaluationProtocol,
        environment_name: str,
        model_name: str,
        environment_hyperparameters: dict[str, ParameterValue] | None = None,
        num_episodes: int = 10,
        max_steps_per_episode: int | None = None,
        render: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a trained model on an environment with specific hyperparameters.

        Args:
            model: Trained model implementing EvaluationProtocol
            environment_name: Name of the environment to use
            model_name: Name of the model (for logging)
            environment_hyperparameters: Environment hyperparameters to use
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode (None for no limit)
            render: Whether to render the environment (if supported)

        Returns:
            EvaluationResult containing evaluation metrics
        """
        import logging

        evaluation_logger = logging.getLogger("hercule.evaluation")
        """
        Evaluate a trained model on an environment.

        Args:
            model: Trained model implementing EvaluationProtocol
            environment_name: Name of the environment to use
            model_name: Name of the model (for logging)
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode (None for no limit)
            render: Whether to render the environment (if supported)

        Returns:
            EvaluationResult containing evaluation metrics
        """
        try:
            evaluation_logger.info(f"Starting evaluation of {model_name} on {environment_name}")
            evaluation_logger.debug(
                f"Evaluation parameters: {num_episodes} episodes, max_steps={max_steps_per_episode}"
            )

            # Load environment with specific hyperparameters
            if environment_hyperparameters:
                env = self.env_manager.load_environment_with_hyperparams(environment_name, environment_hyperparameters)
            else:
                env = self.env_manager.load_environment(environment_name)
            evaluation_logger.info(f"Environment {environment_name} loaded for evaluation")

            # Reconfigure model for this environment if needed
            # This ensures the model knows the correct action space for evaluation
            if hasattr(model, "configure"):
                # Get hyperparameters from model if available
                model_hyperparams = {}
                if hasattr(model, "_learning_rate"):
                    model_hyperparams["learning_rate"] = model._learning_rate
                if hasattr(model, "_discount_factor"):
                    model_hyperparams["discount_factor"] = model._discount_factor
                if hasattr(model, "_epsilon"):
                    model_hyperparams["epsilon"] = model._epsilon
                if hasattr(model, "_epsilon_decay"):
                    model_hyperparams["epsilon_decay"] = model._epsilon_decay
                if hasattr(model, "_epsilon_min"):
                    model_hyperparams["epsilon_min"] = model._epsilon_min

                # Reconfigure model for this environment
                model.configure(env, model_hyperparams)
                evaluation_logger.debug(f"Model reconfigured for environment {environment_name}")

            # Enable rendering if requested
            if render:
                try:
                    env = gym.wrappers.RenderCollection(env)
                    evaluation_logger.debug("Rendering enabled")
                except Exception as e:
                    evaluation_logger.warning(f"Could not enable rendering: {e}")

            episode_rewards = []
            episode_lengths = []

            for episode in range(num_episodes):
                evaluation_logger.debug(f"Starting episode {episode + 1}/{num_episodes}")

                # Reset environment
                observation, info = env.reset()
                episode_reward = 0.0
                episode_length = 0

                # Run episode
                while True:
                    # Select action (not in training mode)
                    action = model.act(observation, training=False)

                    # Ensure action is in the correct format for the environment
                    if hasattr(env.action_space, "n"):
                        # For discrete action spaces, ensure action is an integer
                        action = int(action)
                        # Validate action is within valid range
                        if action >= env.action_space.n:
                            evaluation_logger.error(
                                f"Invalid action {action} for environment with {env.action_space.n} actions"
                            )
                            raise ValueError(f"Action {action} is out of range [0, {env.action_space.n})")
                    else:
                        # For continuous action spaces, ensure action is a numpy array
                        action = np.array(action)

                    # Take step
                    next_observation, reward, terminated, truncated, info = env.step(action)

                    # Update episode tracking
                    episode_reward += float(reward)
                    episode_length += 1

                    # Check termination conditions
                    done = terminated or truncated
                    if done or (max_steps_per_episode and episode_length >= max_steps_per_episode):
                        break

                    observation = next_observation

                # Record episode results
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                evaluation_logger.debug(
                    f"Episode {episode + 1} completed: reward={episode_reward:.2f}, length={episode_length}"
                )

                # Log progress every 25% of episodes
                if (episode + 1) % max(1, num_episodes // 4) == 0:
                    recent_rewards = episode_rewards[-max(1, num_episodes // 4) :]
                    avg_reward = np.mean(recent_rewards)
                    evaluation_logger.info(
                        f"Evaluation progress: {episode + 1}/{num_episodes} episodes, "
                        f"avg reward (last {len(recent_rewards)}): {avg_reward:.2f}"
                    )

            # Calculate final metrics
            result = EvaluationResult(
                environment_name=environment_name,
                model_name=model_name,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                total_episodes=num_episodes,
                success=True,
            )

            evaluation_logger.info(
                f"Evaluation completed for {model_name} on {environment_name}. "
                f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}, "
                f"Mean length: {result.mean_length:.1f} ± {result.std_length:.1f}"
            )

            return result

        except Exception as e:
            evaluation_logger.error(f"Evaluation failed for {model_name} on {environment_name}: {e}")
            return EvaluationResult(
                environment_name=environment_name,
                model_name=model_name,
                episode_rewards=[],
                episode_lengths=[],
                total_episodes=0,
                success=False,
                error_message=str(e),
            )

    def close(self) -> None:
        """Close all resources."""
        self.env_manager.close_all()

    def __enter__(self) -> "ModelEvaluator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def save_evaluation_results(
    results: list[EvaluationResult], output_dir: Path, filename: str = "evaluation_results.json"
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of evaluation results to save
        output_dir: Directory to save results in
        filename: Name of the output file
    """
    import json

    output_path = output_dir / filename

    # Convert results to dictionaries
    results_data = [result.to_dict() for result in results]

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to {output_path}")


__all__ = [
    "EvaluationProtocol",
    "EvaluationResult",
    "ModelEvaluator",
    "save_evaluation_results",
]
