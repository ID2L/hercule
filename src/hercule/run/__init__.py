"""Run module for managing Gymnasium environments and training execution."""

import logging
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np

from hercule.config import HerculeConfig, EnvironmentConfig, ParameterValue


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


class EnvironmentManager:
    """Manager for Gymnasium environments."""
    
    def __init__(self, config: HerculeConfig) -> None:
        """
        Initialize the environment manager.
        
        Args:
            config: Hercule configuration
        """
        self.config = config
        self._environments: dict[str, gym.Env] = {}
    
    def load_environment(self, env_name: str) -> gym.Env:
        """
        Load a Gymnasium environment.
        
        Args:
            env_name: Name of the environment to load
            
        Returns:
            Loaded Gymnasium environment
            
        Raises:
            ValueError: If environment name is not supported
        """
        if env_name in self._environments:
            return self._environments[env_name]
        
        try:
            env = gym.make(env_name)
            self._environments[env_name] = env
            logger.info(f"Successfully loaded environment: {env_name}")
            return env
        except gym.error.Error as e:
            msg = f"Failed to load environment '{env_name}': {e}"
            raise ValueError(msg) from e
    
    def get_environment_info(self, env_name: str) -> dict[str, str | int | list[float] | dict[str, str | int | list[float] | None] | None]:
        """
        Get information about an environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Dictionary containing environment information
        """
        env = self.load_environment(env_name)
        
        obs_space_info: dict[str, str | int | list[float] | None] = {
            "type": type(env.observation_space).__name__,
            "shape": getattr(env.observation_space, 'shape', None),
            "low": getattr(env.observation_space, 'low', None),
            "high": getattr(env.observation_space, 'high', None),
        }
        
        action_space_info: dict[str, str | int | list[float] | None] = {
            "type": type(env.action_space).__name__,
            "n": getattr(env.action_space, 'n', None),
            "shape": getattr(env.action_space, 'shape', None),
            "low": getattr(env.action_space, 'low', None),
            "high": getattr(env.action_space, 'high', None),
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for space_info in [obs_space_info, action_space_info]:
            for key in ["low", "high"]:
                value = space_info[key]
                if isinstance(value, np.ndarray):
                    space_info[key] = value.tolist()
        
        info: dict[str, str | int | list[float] | dict[str, str | int | list[float] | None] | None] = {
            "name": env_name,
            "observation_space": obs_space_info,
            "action_space": action_space_info,
            "max_episode_steps": getattr(env.spec, 'max_episode_steps', None) if env.spec else None,
        }
        
        return info
    
    def validate_environments(self) -> list[str]:
        """
        Validate all configured environments can be loaded.
        
        Returns:
            List of successfully validated environment names
            
        Raises:
            ValueError: If no environments can be loaded
        """
        valid_environments = []
        invalid_environments = []
        
        for env in self.config.environments:
            env_name = env.name if isinstance(env, EnvironmentConfig) else env
            try:
                self.load_environment(env_name)
                valid_environments.append(env_name)
            except ValueError as e:
                logger.warning(f"Environment '{env_name}' could not be loaded: {e}")
                invalid_environments.append(env_name)
        
        if not valid_environments:
            msg = f"No valid environments found. Invalid environments: {invalid_environments}"
            raise ValueError(msg)
        
        if invalid_environments:
            logger.warning(f"Some environments could not be loaded: {invalid_environments}")
        
        return valid_environments
    
    def close_all(self) -> None:
        """Close all loaded environments."""
        for env in self._environments.values():
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
        self._environments.clear()
    
    def __enter__(self) -> "EnvironmentManager":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close all environments."""
        self.close_all()


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