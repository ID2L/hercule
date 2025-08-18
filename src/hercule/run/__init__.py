"""Run module for training execution and result management."""

import logging
from pathlib import Path
from typing import Protocol

import gymnasium as gym

from hercule.config import HerculeConfig, ParameterValue
from hercule.environnements import EnvironmentManager

logger = logging.getLogger(__name__)


class TrainingProtocol(Protocol):
    """Protocol defining the interface for RL models."""

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
        ...

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        ...


class RunResult:
    """Container for training run results."""

    def __init__(
        self,
        environment_name: str,
        model_name: str,
        hyperparameters: dict[str, ParameterValue],
        metrics: dict[str, ParameterValue],
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """
        Initialize run result.

        Args:
            environment_name: Name of the environment used
            model_name: Name of the model used
            hyperparameters: Hyperparameters used for training
            metrics: Training metrics and results
            success: Whether the run was successful
            error_message: Error message if run failed
        """
        self.environment_name = environment_name
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.success = success
        self.error_message = error_message

    def to_dict(self) -> dict[str, ParameterValue | dict[str, ParameterValue] | None]:
        """Convert result to dictionary for serialization."""
        return {
            "environment_name": self.environment_name,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "success": self.success,
            "error_message": self.error_message,
        }


class TrainingRunner:
    """Handles training execution across models and environments."""

    def __init__(self, config: HerculeConfig):
        """
        Initialize the training runner.

        Args:
            config: Hercule configuration
        """
        self.config = config
        self.env_manager = EnvironmentManager(config)

    def run_single_training(
        self,
        model: TrainingProtocol,
        environment_name: str,
        model_name: str,
        hyperparameters: dict[str, ParameterValue],
    ) -> RunResult:
        """
        Run training for a single model-environment combination.

        Args:
            model: Model implementing TrainingProtocol
            environment_name: Name of the environment
            model_name: Name of the model
            hyperparameters: Model hyperparameters

        Returns:
            RunResult containing training results
        """
        try:
            # Load environment
            env = self.env_manager.load_environment(environment_name)

            # Run training
            metrics = model.train(env, hyperparameters, self.config.max_iterations)

            return RunResult(
                environment_name=environment_name,
                model_name=model_name,
                hyperparameters=hyperparameters,
                metrics=metrics,
                success=True,
            )
        except Exception as e:
            logger.error(f"Training failed for {model_name} on {environment_name}: {e}")
            return RunResult(
                environment_name=environment_name,
                model_name=model_name,
                hyperparameters=hyperparameters,
                metrics={},
                success=False,
                error_message=str(e),
            )

    def validate_configuration(self) -> bool:
        """
        Validate that all configured environments can be loaded.

        Returns:
            True if all environments are valid, False otherwise
        """
        try:
            self.env_manager.validate_environments()
            return True
        except ValueError as e:
            logger.error(f"Environment validation failed: {e}")
            return False

    def close(self) -> None:
        """Close all resources."""
        self.env_manager.close_all()

    def __enter__(self) -> "TrainingRunner":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def create_output_directory(config: HerculeConfig) -> Path:
    """
    Create output directory structure.

    Args:
        config: Hercule configuration

    Returns:
        Path to the created output directory
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    subdirs = ["models", "logs", "results"]
    for subdir in subdirs:
        (output_dir / subdir).mkdir(exist_ok=True)

    logger.info(f"Output directory created: {output_dir}")
    return output_dir


__all__ = [
    "TrainingProtocol",
    "RunResult",
    "TrainingRunner",
    "create_output_directory",
]
