"""Gymnasium wrappers Hercule applies on top of an environment, configured from the YAML file."""

import gymnasium as gym
import numpy as np


class DiscretizeActionWrapper(gym.ActionWrapper):
    """Expose a continuous Box action space as a Discrete one.

    Each action dimension is replaced by ``bins`` evenly spaced values between its bounds (both included); the
    Discrete action ``i`` indexes the cartesian product of those grids (C order, last dimension varying fastest).
    The wrapped environment is untouched: it still receives a continuous action.
    """

    def __init__(self, env: gym.Env, bins: int | list[int]):
        """Wrap ``env``.

        Args:
            env: Environment with a ``gym.spaces.Box`` action space with finite bounds.
            bins: Number of values per action dimension (one int for every dimension, or one per dimension).

        Raises:
            ValueError: If the action space is not a bounded Box, or ``bins`` does not match its dimensions.
        """
        super().__init__(env)
        space = env.action_space
        if not isinstance(space, gym.spaces.Box):
            raise ValueError(f"Action discretization requires a Box action space, got {type(space).__name__}")
        if not (np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))):
            raise ValueError("Action discretization requires finite action bounds")

        dimensions = int(np.prod(space.shape))
        bins_per_dimension = [bins] * dimensions if isinstance(bins, int) else list(bins)
        if len(bins_per_dimension) != dimensions:
            raise ValueError(f"bins has {len(bins_per_dimension)} entries but the action space has {dimensions} dims")

        low, high = space.low.reshape(-1), space.high.reshape(-1)
        self._grids = [
            np.linspace(low[d], high[d], bins_per_dimension[d], dtype=space.dtype) for d in range(dimensions)
        ]
        self._bins = tuple(bins_per_dimension)
        self._shape = space.shape
        self._dtype = space.dtype
        self.action_space = gym.spaces.Discrete(int(np.prod(self._bins)))

    def action(self, action: int | np.integer) -> np.ndarray:
        """Map a Discrete action index to the corresponding continuous action."""
        indices = np.unravel_index(int(action), self._bins)
        values = [self._grids[d][i] for d, i in enumerate(indices)]
        return np.asarray(values, dtype=self._dtype).reshape(self._shape)


__all__ = ["DiscretizeActionWrapper"]
