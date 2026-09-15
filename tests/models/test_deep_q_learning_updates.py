"""Regression tests for when Deep Q-Learning performs its gradient updates.

The bug these guard against: `run_epoch()` used to call the gradient step *after*
the step loop, so an episode of N steps produced a single gradient update
derived from the whole episode return -- Monte-Carlo control, not the
per-transition TD update DQN is defined by. The target-network synchronisation
was chained to that same call, which also cancelled the target network out.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from hercule.models.deep_q_learning import DeepQLearningModel


EPISODE_LENGTH = 40


class _FixedLengthEnv(gym.Env):
    """Deterministic environment terminating after exactly EPISODE_LENGTH steps."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        obs = np.full(2, self._step / EPISODE_LENGTH, dtype=np.float32)
        return obs, 1.0, self._step >= EPISODE_LENGTH, False, {}


def _make_model(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> tuple[DeepQLearningModel, list[int]]:
    """Configure a model on the stub env, counting every `_update()` call."""
    env = _FixedLengthEnv()
    model = DeepQLearningModel()
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 4,
        "replay_buffer_size": 500,
        "epsilon": 1.0,
        "epsilon_decay": 0.0,
        "epsilon_min": 1.0,
        "step_modulo": 1,
        "target_update_frequency": 1000,
        "seed": 42,
    }
    hyperparameters.update(overrides)
    assert model.configure(env, hyperparameters)
    model.env = env

    calls: list[int] = []
    original = DeepQLearningModel._update

    def counting_update(self: DeepQLearningModel, batch: list) -> None:
        calls.append(self._step_count)
        original(self, batch)

    monkeypatch.setattr(DeepQLearningModel, "_update", counting_update)
    return model, calls


@pytest.mark.unit
def test_train_step_runs_every_environment_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """With step_modulo=1 the update is per transition, not one per episode."""
    model, calls = _make_model(monkeypatch)

    result = model.run_epoch(train_mode=True)

    assert result.steps_number == EPISODE_LENGTH
    # The buffer needs batch_size transitions before the first replay, so the
    # updates are every step from step batch_size onwards -- crucially many more
    # than the single end-of-episode update of the old behaviour.
    assert calls == list(range(4, EPISODE_LENGTH + 1))


@pytest.mark.unit
def test_train_step_honours_step_modulo(monkeypatch: pytest.MonkeyPatch) -> None:
    """step_modulo counts environment steps, not episodes."""
    model, calls = _make_model(monkeypatch, step_modulo=8)

    model.run_epoch(train_mode=True)

    assert calls == [8, 16, 24, 32, 40]


@pytest.mark.unit
def test_no_update_outside_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test-phase episode must never touch the weights."""
    model, calls = _make_model(monkeypatch)

    model.run_epoch(train_mode=False)

    assert calls == []


@pytest.mark.unit
def test_target_network_lags_the_online_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target network is frozen between two synchronisations.

    Chaining the copy to the gradient step made the bootstrap target come from the
    current weights, i.e. no target network at all.
    """
    model, calls = _make_model(monkeypatch, target_update_frequency=1000)
    frozen = {k: v.clone() for k, v in model._target_network.state_dict().items()}

    model.run_epoch(train_mode=True)

    assert len(calls) > 1, "the episode must contain several updates for this to mean anything"
    online = model._q_network.state_dict()
    target = model._target_network.state_dict()
    assert any(not torch.equal(online[k], frozen[k]) for k in frozen), "online network did not move"
    assert all(torch.equal(target[k], frozen[k]) for k in frozen), "target network followed the online one"


@pytest.mark.unit
def test_target_network_syncs_on_its_own_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reaching target_update_frequency steps does copy the online weights over."""
    model, _ = _make_model(monkeypatch, target_update_frequency=10)
    frozen = {k: v.clone() for k, v in model._target_network.state_dict().items()}

    model.run_epoch(train_mode=True)

    target = model._target_network.state_dict()
    assert any(not torch.equal(target[k], frozen[k]) for k in frozen), "target network never synchronised"
