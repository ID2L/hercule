"""A `Box` space of any shape, not just a flat one, must work end to end.

Two independent defects shared one root cause: `ContinuousActorCriticModel` and
`OffPolicyReplayModel` both worked in the FLAT coordinates `_action_dimensions`
derives (`int(np.prod(shape))`), but the code around them still carried the
space's ORIGINAL shape in places that mattered.

- `SACModel.supported_spaces` declares support for `Box` actions generally, but
  `OffPolicyReplayModel._cache_action_bounds` stored `_action_low`/`_action_high`
  with the space's original shape, so `to_env_action`'s `bias + scale * normalised`
  mixed a `(2, 2)`-shaped bound with a flat `(4,)` normalised action.
- `off_policy.Encoder`'s non-image branch sized its input from
  `int(np.prod(observation_shape))` for a rank other than 1 or 3, but `forward()`
  passed the tensor straight into the `Linear` stack without flattening it, so a
  `(2, 2)` observation arrived as `(batch, 2, 2)` against a `Linear` expecting 4
  features.

Both environments below are minimal on purpose: only the shape under test varies
from the flat cases `test_action_mapping.py` already covers.
"""

import gymnasium as gym
import numpy as np
import pytest

from hercule.models.sac import SACModel


# Deliberately asymmetric per element, like the oracle's own action bounds: a
# uniform low/high would let a transposed or otherwise misread shape pass by
# accident.
MULTI_AXIS_ACTION_LOW = np.array([[-1.0, 0.0], [-2.0, 1.0]], dtype=np.float32)
MULTI_AXIS_ACTION_HIGH = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)


class MultiAxisActionEnv(gym.Env):
    """A flat observation, but a rank-2 `Box` action space, shape `(2, 2)`.

    Exists to prove the action mapping handles a `Box` of any shape: SAC's policy
    and replay buffer work in the FLAT `_action_dimensions` coordinates, so the
    mapping must reshape back to `(2, 2)` before `env.step()` sees it, and every
    submitted action is recorded here to check exactly that.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        """Build the space pair and the recording list `env.step()` fills."""
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=MULTI_AXIS_ACTION_LOW, high=MULTI_AXIS_ACTION_HIGH, dtype=np.float32)
        self.render_mode = None
        self._step = 0
        self.submitted_actions: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode and clear the action log."""
        super().reset(seed=seed)
        self._step = 0
        self.submitted_actions.clear()
        return self.observation_space.sample(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Record the submitted action verbatim, then advance a short episode."""
        self.submitted_actions.append(np.array(action, copy=True))
        self._step += 1
        truncated = self._step >= 8
        return self.observation_space.sample(), 0.0, False, truncated, {}


class MultiAxisObservationEnv(gym.Env):
    """A rank-2 `Box` observation, shape `(2, 2)`, and a flat action space.

    Exists to prove `Encoder`'s non-image branch flattens before its first
    `Linear` layer: without it, a batch of `(2, 2)` observations reaches that
    layer as `(batch, 2, 2)` and torch raises a shape error.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        """Build the space pair; the action bounds are irrelevant here."""
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 2), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.render_mode = None
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode."""
        super().reset(seed=seed)
        self._step = 0
        return self.observation_space.sample(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance a short episode, ignoring the action's value."""
        self._step += 1
        truncated = self._step >= 8
        return self.observation_space.sample(), 0.0, False, truncated, {}


def _configured(env: gym.Env) -> SACModel:
    model = SACModel()
    assert model.configure(env, {"seed": 1, "learning_starts": 0})
    model.env = env
    return model


@pytest.mark.integration
def test_multi_axis_box_action_space_end_to_end() -> None:
    """Every action submitted to `env.step()` is in-bounds and correctly shaped.

    Runs a full training epoch (`train_mode=True`), so this exercises the warmup
    branch of `_select_action` (uniform draw, mapped) and, once past
    `learning_starts=0`, the actor-sampled branch too -- both call `to_env_action`.
    """
    env = MultiAxisActionEnv()
    model = _configured(env)

    model.run_epoch(train_mode=True)

    assert env.submitted_actions, "the episode produced no steps to check"
    for action in env.submitted_actions:
        assert action.shape == env.action_space.shape, f"{action.shape} != {env.action_space.shape}"
        assert env.action_space.contains(action), f"{action} outside {env.action_space}"


@pytest.mark.unit
def test_multi_axis_box_action_endpoints_map_exactly() -> None:
    """`-1` lands on each element's `low`, `+1` on its `high`, per element.

    The same guarantee `test_action_mapping.py` checks for a flat space, here
    against a shape the flat arithmetic must be reshaped back into.
    """
    env = MultiAxisActionEnv()
    model = _configured(env)

    lowest = model.to_env_action(np.full(4, -1.0, dtype=np.float32))
    highest = model.to_env_action(np.full(4, 1.0, dtype=np.float32))

    assert lowest.shape == env.action_space.shape
    assert highest.shape == env.action_space.shape
    assert np.allclose(lowest, MULTI_AXIS_ACTION_LOW), f"{lowest} should be exactly {MULTI_AXIS_ACTION_LOW}"
    assert np.allclose(highest, MULTI_AXIS_ACTION_HIGH), f"{highest} should be exactly {MULTI_AXIS_ACTION_HIGH}"


@pytest.mark.integration
def test_multi_axis_box_observation_configures_and_trains() -> None:
    """A rank-2 `Box` observation configures and completes a training epoch.

    Before the fix this raised a shape error out of the first `Linear` layer in
    `Encoder`'s non-image branch, which never flattened its input.
    """
    env = MultiAxisObservationEnv()
    model = _configured(env)

    result = model.run_epoch(train_mode=True)

    assert result.steps_number > 0
