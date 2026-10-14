"""Minimal custom Gymnasium environment used by the custom-environment tests.

Imported by bare module name through the YAML ``python_path``, exactly like a user's own environment.
"""

import gymnasium as gym
import numpy as np


class LineEnv(gym.Env):
    """Move a point on [-1, 1] with a continuous action; the episode ends when it reaches +1."""

    metadata = {"render_modes": []}

    def __init__(self, step_size: float = 0.5):
        super().__init__()
        self.step_size = step_size
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._position = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._position = 0.0
        return np.array([self._position], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._position = float(np.clip(self._position + self.step_size * float(action[0]), -1.0, 1.0))
        terminated = self._position >= 1.0
        return np.array([self._position], dtype=np.float32), float(terminated), terminated, False, {}
