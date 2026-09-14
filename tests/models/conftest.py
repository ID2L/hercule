"""Shared fixtures for model tests.

The `StubEnv` factory exists because the defects this feature fixes are only observable
at an episode boundary: whether an episode ended by `terminated` or by `truncated`
changes the TD target, and no stock Gymnasium environment lets a test choose. Driving
these tests through a real environment would also make them slow enough to discourage
running them.
"""

import gymnasium as gym
import numpy as np
import pytest

from hercule.models.deep_q_learning import DeepQLearningModel


class StubEnv(gym.Env):
    """A deterministic environment whose episode ending is chosen by the test.

    Attributes:
        ends_by: ``"terminated"`` for a genuine MDP terminal state, ``"truncated"`` for a
            time limit. This is the distinction the whole S03 sub-spec turns on: a
            truncation is not terminal, so its successor state still has value.
    """

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        episode_length: int = 10,
        ends_by: str = "terminated",
        reward: float = 1.0,
    ) -> None:
        if ends_by not in ("terminated", "truncated"):
            raise ValueError(f"ends_by must be 'terminated' or 'truncated', got {ends_by!r}")
        self.observation_space = observation_space or gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = action_space or gym.spaces.Discrete(3)
        self.episode_length = episode_length
        self.ends_by = ends_by
        self.reward = reward
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        at_end = self._step >= self.episode_length
        terminated = at_end and self.ends_by == "terminated"
        truncated = at_end and self.ends_by == "truncated"
        return self._observation(), self.reward, terminated, truncated, {}

    def _observation(self) -> np.ndarray:
        """A distinct, deterministic observation per step, so states are distinguishable."""
        value = self._step / max(self.episode_length, 1)
        return np.full(self.observation_space.shape, value, dtype=np.float32)


@pytest.fixture
def stub_env_factory():
    """Return the `StubEnv` class so a test can parameterise its own instance."""
    return StubEnv


@pytest.fixture
def tiny_dqn_hyperparameters() -> dict:
    """Hyperparameters sized so a training step runs in milliseconds.

    `epsilon_min` equals `epsilon` so exploration stays constant, which keeps a test
    that asserts on the TD target independent of the decay schedule.
    """
    return {
        "learning_rate": 0.001,
        "batch_size": 2,
        "replay_buffer_size": 16,
        "epsilon": 1.0,
        "epsilon_decay": 0.0,
        "epsilon_min": 1.0,
        "step_modulo": 1,
        "target_update_frequency": 1000,
        "seed": 42,
    }


@pytest.fixture
def tiny_dqn(stub_env_factory, tiny_dqn_hyperparameters):
    """A `DeepQLearningModel` configured on a small discrete stub environment.

    Returns:
        A ``(model, env)`` pair. The environment is returned too because several tests
        need to drive it directly rather than through ``run_epoch``.
    """

    def _build(**overrides):
        env_kwargs = {k: overrides.pop(k) for k in ("episode_length", "ends_by", "reward") if k in overrides}
        env = stub_env_factory(**env_kwargs)
        hyperparameters = {**tiny_dqn_hyperparameters, **overrides}
        model = DeepQLearningModel()
        assert model.configure(env, hyperparameters), "stub environment must be configurable"
        model.env = env
        return model, env

    return _build
