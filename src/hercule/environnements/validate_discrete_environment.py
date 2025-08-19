import logging
from typing import cast

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete


logger = logging.getLogger(__name__)


def validate_discrete_environment(env: gym.Env) -> Env[Discrete, Discrete]:
    """
    Validate that the environment has discrete action and observation spaces.

    Args:
        env: Gymnasium environment to validate

    Returns:
        The validated environment with explicit discrete space types

    Raises:
        ValueError: If environment does not have discrete spaces
    """
    if not isinstance(env.action_space, Discrete):
        msg = f"Algo requires discrete action space for environment {env.spec}"
        raise ValueError(msg)

    if not isinstance(env.observation_space, Discrete):
        msg = f"Algo requires discrete observation space for environment {env.spec}"
        raise ValueError(msg)

    return cast("Env[Discrete, Discrete]", env)
