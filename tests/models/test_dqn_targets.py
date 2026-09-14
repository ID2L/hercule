"""Tests for the DQN TD target: `terminated` masks the bootstrap, `truncated` never does.

Regression coverage for spec 006 sub-spec S03. `run_epoch` used to collapse `terminated`
and `truncated` into a single `done` flag before pushing a transition to the replay
buffer, and `_train_step` masked the bootstrap term with that flag. A Gymnasium
time-limit truncation is NOT an MDP terminal state -- `Pendulum-v1` always truncates and
never terminates -- so masking on it trained every truncating environment against a
target of `r` instead of `r + gamma * max_a Q_target(s', a)`.
"""

import numpy as np
import pytest
import torch

from hercule.models.deep_q_learning import DeepQLearningModel


def _capture_mse_targets(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Patch `torch.nn.MSELoss.forward` so a test can read the target `_train_step` built.

    `_train_step` computes `target_q_values` as a local variable and never returns or
    stores it anywhere, so intercepting the one place it is consumed is the only way to
    observe it from a test without changing production code.
    """
    captured: list[torch.Tensor] = []
    real_forward = torch.nn.MSELoss.forward

    def capturing_forward(self: torch.nn.MSELoss, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        captured.append(target.detach().clone())
        return real_forward(self, current, target)

    monkeypatch.setattr(torch.nn.MSELoss, "forward", capturing_forward)
    return captured


def _expected_target(model: DeepQLearningModel, reward: float, next_state: np.ndarray, *, bootstraps: bool) -> float:
    """Recompute the target the way `_train_step` is specified to build it."""
    with torch.no_grad():
        next_q = model._target_network(model._as_network_input(next_state).unsqueeze(0)).max().item()
    discount_factor = model.get_hyperparameters().discount_factor
    return reward + (discount_factor * next_q if bootstraps else 0.0)


@pytest.mark.unit
def test_truncation_keeps_the_bootstrap_term(tiny_dqn, monkeypatch: pytest.MonkeyPatch) -> None:
    """A time-limit truncation must not zero the bootstrap term (T023)."""
    model, _ = tiny_dqn(batch_size=1)
    state = np.array([0.1, -0.2], dtype=np.float32)
    next_state = np.array([0.3, 0.4], dtype=np.float32)
    model._replay_buffer.push(state, 0, 1.0, next_state, terminated=False, truncated=True)
    captured = _capture_mse_targets(monkeypatch)

    model._train_step()

    assert captured, "no MSE loss was computed, _train_step did not run"
    expected = _expected_target(model, reward=1.0, next_state=next_state, bootstraps=True)
    assert captured[-1].item() == pytest.approx(expected)


@pytest.mark.unit
def test_termination_zeroes_the_bootstrap_term(tiny_dqn, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine terminal state must zero the bootstrap term: target == reward exactly (T024)."""
    model, _ = tiny_dqn(batch_size=1)
    state = np.array([0.1, -0.2], dtype=np.float32)
    next_state = np.array([0.3, 0.4], dtype=np.float32)
    model._replay_buffer.push(state, 0, 1.0, next_state, terminated=True, truncated=False)
    captured = _capture_mse_targets(monkeypatch)

    model._train_step()

    assert captured[-1].item() == pytest.approx(1.0)


@pytest.mark.unit
def test_mid_episode_transition_is_unaffected_by_either_flag(tiny_dqn, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transition stored mid-episode (both flags False) always bootstraps (T025)."""
    model, _ = tiny_dqn(batch_size=1)
    state = np.array([0.1, -0.2], dtype=np.float32)
    next_state = np.array([0.3, 0.4], dtype=np.float32)
    model._replay_buffer.push(state, 0, 1.0, next_state, terminated=False, truncated=False)
    captured = _capture_mse_targets(monkeypatch)

    model._train_step()

    expected = _expected_target(model, reward=1.0, next_state=next_state, bootstraps=True)
    assert captured[-1].item() == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ends_by", "expected_terminated", "expected_truncated"),
    [("terminated", True, False), ("truncated", False, True)],
)
def test_run_epoch_pushes_terminated_and_truncated_separately(
    tiny_dqn, ends_by: str, expected_terminated: bool, expected_truncated: bool
) -> None:
    """The `push` call site (T021) must forward the env's own flags, never a collapsed `done`."""
    model, _ = tiny_dqn(episode_length=3, ends_by=ends_by)

    model.run_epoch(train_mode=True)

    *_, terminated, truncated = model._replay_buffer.buffer[-1]
    assert terminated is expected_terminated
    assert truncated is expected_truncated
    # Every earlier transition in the same episode ended neither way.
    for _, _, _, _, mid_terminated, mid_truncated in list(model._replay_buffer.buffer)[:-1]:
        assert mid_terminated is False
        assert mid_truncated is False
