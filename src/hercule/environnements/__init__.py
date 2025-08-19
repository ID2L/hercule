"""Environment management module for Hercule reinforcement learning framework."""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import registry
from pydantic import BaseModel

from hercule.config import EnvironmentConfig, HerculeConfig, ParameterValue


logger = logging.getLogger(__name__)


class SpaceInfo(BaseModel):
    """Base class for space information."""

    type: str
    shape: tuple[int, ...] | None = None


class DiscreteSpaceInfo(SpaceInfo):
    """Information about a discrete action/observation space."""

    n: int | None = None


class BoxSpaceInfo(SpaceInfo):
    """Information about a box (continuous) action/observation space."""

    low: list[float] | None = None
    high: list[float] | None = None


class EnvironmentSpecInfo(BaseModel):
    """Information from Gymnasium environment spec."""

    id: str
    entry_point: str | None = None
    reward_threshold: float | None = None
    nondeterministic: bool = False
    max_episode_steps: int | None = None
    order_enforce: bool = True
    autoreset: bool = False
    kwargs: dict[str, Any] = {}


class EnvironmentInfo(BaseModel):
    """Complete information about a Gymnasium environment."""

    name: str
    observation_space: SpaceInfo
    action_space: SpaceInfo
    spec_info: EnvironmentSpecInfo | None = None

    @property
    def max_episode_steps(self) -> int | None:
        """Get max episode steps from spec info."""
        return self.spec_info.max_episode_steps if self.spec_info else None


class EnvironmentRegistry:
    """Registry for managing available Gymnasium environments."""

    @staticmethod
    def list_available_environments() -> list[str]:
        """
        List all available environments in the Gymnasium registry.

        Returns:
            List of available environment IDs
        """
        return list(registry.keys())

    @staticmethod
    def search_environments(search_term: str) -> list[str]:
        """
        Search for environments containing a specific term.

        Args:
            search_term: Term to search for in environment names

        Returns:
            List of matching environment IDs
        """
        available_envs = registry.keys()
        return [env for env in available_envs if search_term.lower() in env.lower()]

    @staticmethod
    def environment_exists(env_name: str) -> bool:
        """
        Check if an environment exists in the registry.

        Args:
            env_name: Name of the environment to check

        Returns:
            True if environment exists, False otherwise
        """
        return env_name in registry.keys()

    @staticmethod
    def get_similar_environments(env_name: str, limit: int = 5) -> list[str]:
        """
        Get environments with similar names.

        Args:
            env_name: Environment name to find similar names for
            limit: Maximum number of suggestions to return

        Returns:
            List of similar environment names
        """
        available_envs = registry.keys()
        similar_envs = [
            env for env in available_envs if env_name.lower() in env.lower() or env.lower() in env_name.lower()
        ]
        return list(similar_envs)[:limit]


class EnvironmentFactory:
    """Factory for creating and managing Gymnasium environments."""

    def __init__(self):
        """Initialize the environment factory."""
        self._environments: dict[str, gym.Env] = {}

    def create_environment(self, env_name: str, **kwargs) -> gym.Env:
        """
        Create a Gymnasium environment with validation.

        Args:
            env_name: Name of the environment to create
            **kwargs: Additional keyword arguments for environment creation

        Returns:
            Created Gymnasium environment

        Raises:
            ValueError: If environment name is not supported or creation fails
        """
        # Check if environment exists in registry
        if not EnvironmentRegistry.environment_exists(env_name):
            similar_envs = EnvironmentRegistry.get_similar_environments(env_name)
            if similar_envs:
                suggestions = ", ".join(similar_envs)
                msg = (
                    f"Environment '{env_name}' does not exist in Gymnasium registry. "
                    f"Similar environments: {suggestions}"
                )
            else:
                total_envs = len(EnvironmentRegistry.list_available_environments())
                msg = (
                    f"Environment '{env_name}' does not exist in Gymnasium registry. "
                    f"Available environments: {total_envs} total"
                )
            raise ValueError(msg)

        try:
            env = gym.make(env_name, **kwargs)
            logger.info(f"Successfully created environment: {env_name}")
            return env
        except gym.error.Error as e:
            msg = f"Failed to create environment '{env_name}' (exists in registry but creation failed): {e}"
            raise ValueError(msg) from e

    def get_or_create_environment(self, env_name: str, **kwargs) -> gym.Env:
        """
        Get cached environment or create new one.

        Args:
            env_name: Name of the environment
            **kwargs: Additional keyword arguments for environment creation

        Returns:
            Gymnasium environment (cached or newly created)
        """
        cache_key = f"{env_name}_{hash(frozenset(kwargs.items()))}"

        if cache_key in self._environments:
            return self._environments[cache_key]

        env = self.create_environment(env_name, **kwargs)
        self._environments[cache_key] = env
        return env

    def close_all(self) -> None:
        """Close all cached environments."""
        for env in self._environments.values():
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
        self._environments.clear()

    def __enter__(self) -> "EnvironmentFactory":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close all environments."""
        self.close_all()


class EnvironmentInspector:
    """Inspector for extracting detailed information from Gymnasium environments."""

    @staticmethod
    def _create_space_info(space: gym.Space) -> SpaceInfo:
        """
        Create space info from a Gymnasium space.

        Args:
            space: Gymnasium space object

        Returns:
            Appropriate SpaceInfo subclass instance
        """
        space_type = type(space).__name__

        if space_type == "Discrete":
            return DiscreteSpaceInfo(type=space_type, n=getattr(space, "n", None))
        elif space_type == "Box":
            low = getattr(space, "low", None)
            high = getattr(space, "high", None)
            return BoxSpaceInfo(
                type=space_type,
                shape=getattr(space, "shape", None),
                low=low.tolist() if isinstance(low, np.ndarray) else None,
                high=high.tolist() if isinstance(high, np.ndarray) else None,
            )
        else:
            return SpaceInfo(type=space_type, shape=getattr(space, "shape", None))

    @staticmethod
    def _create_spec_info(spec) -> EnvironmentSpecInfo | None:
        """
        Create spec info from a Gymnasium environment spec.

        Args:
            spec: Gymnasium environment spec

        Returns:
            EnvironmentSpecInfo instance or None if no spec
        """
        if spec is None:
            return None

        return EnvironmentSpecInfo(
            id=getattr(spec, "id", ""),
            entry_point=getattr(spec, "entry_point", None),
            reward_threshold=getattr(spec, "reward_threshold", None),
            nondeterministic=getattr(spec, "nondeterministic", False),
            max_episode_steps=getattr(spec, "max_episode_steps", None),
            order_enforce=getattr(spec, "order_enforce", True),
            autoreset=getattr(spec, "autoreset", False),
            kwargs=getattr(spec, "kwargs", {}),
        )

    @classmethod
    def get_environment_info(cls, env: gym.Env) -> EnvironmentInfo:
        """
        Extract complete information from a Gymnasium environment.

        Args:
            env: Gymnasium environment instance

        Returns:
            EnvironmentInfo object containing all environment details
        """
        # Get environment name from spec or use unknown
        env_name = env.spec.id if env.spec else "Unknown"

        # Create space information
        obs_space_info = cls._create_space_info(env.observation_space)
        action_space_info = cls._create_space_info(env.action_space)

        # Create spec information
        spec_info = cls._create_spec_info(env.spec)

        return EnvironmentInfo(
            name=env_name,
            observation_space=obs_space_info,
            action_space=action_space_info,
            spec_info=spec_info,
        )

    @classmethod
    def get_environment_hyperparameters(cls, env: gym.Env) -> dict[str, ParameterValue]:
        """
        Extract hyperparameters from environment spec.

        Args:
            env: Gymnasium environment instance

        Returns:
            Dictionary of hyperparameters from the environment spec
        """
        if env.spec is None:
            return {}

        # Get kwargs from spec which contain the hyperparameters
        hyperparams = getattr(env.spec, "kwargs", {})

        # Convert to ParameterValue types
        converted_params: dict[str, ParameterValue] = {}
        for key, value in hyperparams.items():
            if isinstance(value, str | int | float | bool | list):
                converted_params[key] = value
            else:
                # Convert other types to string representation
                converted_params[key] = str(value)

        return converted_params


class EnvironmentManager:
    """High-level manager for Gymnasium environments with configuration support."""

    def __init__(self, config: HerculeConfig):
        """
        Initialize the environment manager.

        Args:
            config: Hercule configuration
        """
        self.config = config
        self.factory = EnvironmentFactory()
        self.inspector = EnvironmentInspector()

    def load_environment(self, env_name: str) -> gym.Env:
        """
        Load a Gymnasium environment by name, always applying configuration.

        Args:
            env_name: Name of the environment to load

        Returns:
            Loaded Gymnasium environment with configuration applied

        Raises:
            ValueError: If environment name is not supported
        """
        # Get hyperparameters for this environment (empty dict if none configured)
        print(f"Loading environment !! : {env_name}")
        hyperparams = self.config.get_hyperparameters_for_environment(env_name)
        print(f"Hyperparameters: {hyperparams}")
        return self.factory.get_or_create_environment(env_name, **hyperparams)

    def load_environment_with_hyperparams(self, env_name: str, hyperparams: dict[str, ParameterValue]) -> gym.Env:
        """
        Load a Gymnasium environment by name with explicit hyperparameters.

        Args:
            env_name: Name of the environment to load
            hyperparams: Explicit hyperparameters to use

        Returns:
            Loaded Gymnasium environment with specified hyperparameters

        Raises:
            ValueError: If environment name is not supported
        """
        print(f"Loading environment !! : {env_name}")
        print(f"Hyperparameters: {hyperparams}")
        return self.factory.get_or_create_environment(env_name, **hyperparams)

    def get_environment_info(self, env_name: str) -> EnvironmentInfo:
        """
        Get complete information about an environment.

        Args:
            env_name: Name of the environment

        Returns:
            EnvironmentInfo object containing structured environment information
        """
        env = self.load_environment(env_name)
        return self.inspector.get_environment_info(env)

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
        self.factory.close_all()

    def __enter__(self) -> "EnvironmentManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close all environments."""
        self.close_all()


__all__ = [
    "SpaceInfo",
    "DiscreteSpaceInfo",
    "BoxSpaceInfo",
    "EnvironmentSpecInfo",
    "EnvironmentInfo",
    "EnvironmentRegistry",
    "EnvironmentFactory",
    "EnvironmentInspector",
    "EnvironmentManager",
]
