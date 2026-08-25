"""Tests for Deep Q-Learning observation handling: frame stacking and rescaling.

Two behaviours are covered here:

* `frame_stack` concatenates the N previous observations onto the current one
  along the channel axis. `frame_stack: 0` must stay bit-compatible with the
  unstacked behaviour, since it is the default.
* observations are rescaled onto the range declared by the observation space
  itself, never a hard-coded divisor, so the same model works on an image
  environment (0-255), a bounded Box, an unbounded Box and a Discrete space.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from hercule.models.deep_q_learning import DeepQLearningModel
from hercule.models.dummy import DummyModel


IMAGE_SIDE = 48
EPISODE_LENGTH = 12


class _ImageEnv(gym.Env):
    """Environment whose observation is a uint8 image filled with the step index."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        return np.zeros((IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        obs = np.full((IMAGE_SIDE, IMAGE_SIDE, 3), self._step, dtype=np.uint8)
        return obs, 1.0, self._step >= EPISODE_LENGTH, False, {}


class _UnboundedEnv(_ImageEnv):
    """Vector observations with infinite bounds, like CartPole's velocities."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -np.inf], dtype=np.float32),
            high=np.array([4.8, np.inf], dtype=np.float32),
            dtype=np.float32,
        )


class _DiscreteObsEnv(_ImageEnv):
    """Discrete observations, like FrozenLake."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Discrete(16)


def _configure(env: gym.Env, **overrides: object) -> DeepQLearningModel:
    model = DeepQLearningModel()
    hyperparameters = {"learning_rate": 0.001, "batch_size": 4, "replay_buffer_size": 100, "seed": 42}
    hyperparameters.update(overrides)
    assert model.configure(env, hyperparameters)
    model.env = env
    return model


@pytest.mark.unit
def test_frame_stack_zero_keeps_the_unstacked_shape() -> None:
    """The default must not change the network the previous behaviour built."""
    model = _configure(_ImageEnv(), frame_stack=0)

    assert tuple(model._q_network.observation_shape) == (IMAGE_SIDE, IMAGE_SIDE, 3)


@pytest.mark.unit
def test_frame_stack_multiplies_the_channel_axis_only() -> None:
    """frame_stack=3 means current + 3 previous = 4 frames, stacked on channels."""
    model = _configure(_ImageEnv(), frame_stack=3)

    assert tuple(model._q_network.observation_shape) == (IMAGE_SIDE, IMAGE_SIDE, 12)


@pytest.mark.unit
def test_stacked_state_holds_the_previous_frames() -> None:
    """The stack must actually carry history, oldest first, newest last."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3)
    model.run_epoch(train_mode=True)

    # Each stored frame is uniformly filled with its own step index, so the
    # channel triplets of a stacked state read as consecutive step indices.
    state, _, _, next_state, _ = model._replay_buffer.buffer[-1]
    per_frame = [state[0, 0, 3 * i] for i in range(4)]
    assert per_frame == sorted(per_frame), f"frames out of order: {per_frame}"
    assert per_frame[-1] + 1 == next_state[0, 0, 9], "next_state must advance by exactly one frame"
    assert next_state[0, 0, :9].tolist() == state[0, 0, 3:].tolist(), "stack must slide by one frame"


@pytest.mark.unit
def test_replay_buffer_keeps_the_native_dtype() -> None:
    """Storing uint8 rather than float32 is what makes stacking affordable."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3)
    model.run_epoch(train_mode=True)

    state, _, _, _, _ = model._replay_buffer.buffer[-1]
    assert state.dtype == np.uint8
    # 4 frames of uint8 cost the same as 1 frame of float32 did.
    assert state.nbytes == IMAGE_SIDE * IMAGE_SIDE * 3 * 4


@pytest.mark.unit
def test_image_observations_are_rescaled_by_their_own_range() -> None:
    """A 0-255 space yields offset 0 / scale 255, read from the space not hard-coded."""
    model = _configure(_ImageEnv(), frame_stack=0)

    assert model._obs_offset == pytest.approx(0.0)
    assert model._obs_scale == pytest.approx(255.0)
    rescaled = model._as_network_input(np.full((IMAGE_SIDE, IMAGE_SIDE, 3), 255, dtype=np.uint8))
    assert rescaled.dtype == torch.float32
    assert rescaled.max().item() == pytest.approx(1.0)


@pytest.mark.unit
def test_unbounded_space_is_left_untouched() -> None:
    """An infinite bound cannot define a range, so rescaling must be the identity."""
    model = _configure(_UnboundedEnv(), frame_stack=0)

    assert model._obs_offset == pytest.approx(0.0)
    assert model._obs_scale == pytest.approx(1.0)
    values = np.array([3.0, -120.0], dtype=np.float32)
    assert model._as_network_input(values).tolist() == pytest.approx(values.tolist())


@pytest.mark.unit
def test_discrete_space_is_scaled_by_its_cardinality() -> None:
    """Discrete(16) indices run 0..15, so the scale is 15 and the top index maps to 1."""
    model = _configure(_DiscreteObsEnv(), frame_stack=0)

    assert model._obs_scale == pytest.approx(15.0)
    assert model._as_network_input(np.array([15])).item() == pytest.approx(1.0)


@pytest.mark.unit
def test_training_step_runs_on_stacked_rescaled_batches() -> None:
    """End to end: an episode with stacking must produce real gradient updates."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3, step_modulo=1, epsilon=0.0, epsilon_min=0.0)
    before = {k: v.clone() for k, v in model._q_network.state_dict().items()}

    model.run_epoch(train_mode=True)

    after = model._q_network.state_dict()
    assert any(not torch.equal(after[k], before[k]) for k in before), "no weights moved"


@pytest.mark.unit
def test_predict_stacks_across_successive_calls() -> None:
    """predict() has no reset hook, so it must carry its own history."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3)

    obs, _ = env.reset()
    action = model.predict(obs)
    assert 0 <= action < env.action_space.n
    assert len(model._predict_frames) == 4, "history must be primed on the first call"

    for _ in range(3):
        obs, _, _, _, _ = env.step(action)
        action = model.predict(obs)
    assert len(model._predict_frames) == 4, "history must stay capped at frame_stack + 1"


@pytest.mark.unit
def test_saved_stacked_model_loads_into_a_default_configured_model() -> None:
    """The `hercule play` path: configure() gets defaults, the weights carry the depth.

    play_interactive calls configure(env, {}), so without the saved frame_stack the
    rebuilt network would have the wrong in_channels and refuse the state dict.
    """
    env = _ImageEnv()
    trained = _configure(env, frame_stack=3)
    payload = trained._export()
    assert payload["frame_stack"] == 3

    fresh = DeepQLearningModel()
    assert fresh.configure(env, {})  # defaults -> frame_stack 0
    assert tuple(fresh._q_network.observation_shape) == (IMAGE_SIDE, IMAGE_SIDE, 3)

    fresh.load_from_dict(payload)

    assert tuple(fresh._q_network.observation_shape) == (IMAGE_SIDE, IMAGE_SIDE, 12)
    assert fresh.get_hyperparameters().frame_stack == 3
    for key, value in trained._q_network.state_dict().items():
        assert torch.equal(fresh._q_network.state_dict()[key], value)
    # And it can actually act afterwards.
    assert 0 <= fresh.predict(env.reset()[0]) < env.action_space.n


@pytest.mark.unit
def test_begin_episode_clears_the_frame_history() -> None:
    """A stacked state must never span two episodes."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3)

    obs, _ = env.reset()
    model.predict(obs)
    for _ in range(3):
        obs, _, _, _, _ = env.step(0)
        model.predict(obs)
    assert len(model._predict_frames) == 4

    model.begin_episode()

    assert len(model._predict_frames) == 0, "history must be dropped at an episode boundary"


@pytest.mark.unit
def test_predict_reprimes_after_begin_episode() -> None:
    """After a boundary the stack repeats the new episode's first frame only.

    The previous behaviour carried frames over, so the first `frame_stack` steps
    of an episode were decided on a state mixing two different episodes.
    """
    env = _ImageEnv()
    model = _configure(env, frame_stack=3)

    # Episode 1: advance far enough that the history is full of high step indices.
    obs, _ = env.reset()
    model.predict(obs)
    for _ in range(5):
        obs, _, _, _, _ = env.step(0)
        model.predict(obs)
    assert {int(f[0, 0, 0]) for f in model._predict_frames} != {0}

    # Episode 2 starts on an all-zero frame.
    model.begin_episode()
    obs, _ = env.reset()
    model.predict(obs)

    stacked = model._stack(model._predict_frames)
    assert stacked.shape == (IMAGE_SIDE, IMAGE_SIDE, 12)
    assert not stacked.any(), "the first state of a new episode must hold only its own frame"


@pytest.mark.unit
def test_run_epoch_never_leaks_frames_between_episodes() -> None:
    """run_epoch drives its own loop, so it must honour the same contract."""
    env = _ImageEnv()
    model = _configure(env, frame_stack=3, epsilon=0.0, epsilon_min=0.0)

    model.run_epoch(train_mode=True)
    model.run_epoch(train_mode=True)

    # The first transition of the second episode is the oldest one whose state is
    # entirely made of the priming frame; a leak would show a non-zero tail.
    first_of_second_episode = model._replay_buffer.buffer[EPISODE_LENGTH]
    state = first_of_second_episode[0]
    assert not state.any(), "first state of the second episode carries earlier frames"


@pytest.mark.unit
def test_begin_episode_is_a_noop_by_default() -> None:
    """Stateless models must be unaffected: the hook is additive."""
    env = _ImageEnv()
    model = DummyModel()
    model.configure(env, {"seed": 42})  # DummyModel.configure returns None, not True

    model.begin_episode()  # must not raise

    assert 0 <= model.predict(env.reset()[0]) < env.action_space.n
