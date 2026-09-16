"""Tests for Deep Q-Learning checkpoint persistence (spec 006, sub-spec S05).

Regression coverage for four independent defects in the previous
`tensor.tolist()` + `json.dump` encoding:

* it only ever saved the ONLINE network, and `_import()` loaded that single dict
  into the target network too, destroying target lag on every resume;
* it never saved the optimizer's state, so a resume restarted Adam's moment
  buffers from zero;
* it never saved `epsilon`, so a resume at epoch 5000 restarted exploration at
  the YAML default (1.0, fully random) instead of wherever it had decayed to;
* it never saved any RNG state, so a resumed run's randomness restarted from
  wherever `configure()` last seeded it rather than continuing.
"""

import copy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from hercule.models.deep_q_learning import DeepQLearningModel


def _build_and_train(tiny_dqn, *, epochs: int = 2, **hyperparameter_overrides: object):
    """Build a tiny model and train it briefly enough to move the online network
    away from the target network (real lag) without triggering a target sync.

    Returns:
        `(model, env, hyperparameters)` -- the hyperparameters dict is returned
        alongside so a caller can configure an equivalent *fresh* model later
        without picking up any mutation `run_epoch` made along the way (e.g. the
        decayed `epsilon`, which is exactly what a couple of these tests need to
        tell apart from a resumed value).
    """
    hyperparameters = {
        "episode_length": 6,
        "target_update_frequency": 1000,  # never syncs within this test's few steps
        **hyperparameter_overrides,
    }
    model, env = tiny_dqn(**hyperparameters)
    for _ in range(epochs):
        model.run_epoch(train_mode=True)
    return model, env, hyperparameters


@pytest.mark.unit
def test_export_import_round_trip_is_bit_identical(tiny_dqn) -> None:
    """Online weights, target weights, optimizer state and RNG state all survive intact (T044)."""
    trained, _, hyperparameters = _build_and_train(tiny_dqn)

    # Nothing in normal operation currently draws from torch's global RNG beyond
    # network initialisation (no dropout, no stochastic layers), so force a draw
    # to prove the round trip restores a torch RNG state that has moved, not
    # merely one a fresh configure() would reproduce anyway.
    torch.rand(3)

    online_before = {k: v.clone() for k, v in trained._q_network.state_dict().items()}
    target_before = {k: v.clone() for k, v in trained._target_network.state_dict().items()}
    optimizer_before = copy.deepcopy(trained._optimizer.state_dict())
    torch_rng_before = torch.get_rng_state().clone()
    python_random_before = random.getstate()
    numpy_generator_before = copy.deepcopy(trained._rng.bit_generator.state)

    payload = trained._export()

    fresh, _ = tiny_dqn(**hyperparameters)
    fresh._import(payload)

    for key, before_value in online_before.items():
        assert torch.equal(fresh._q_network.state_dict()[key], before_value), f"online weight '{key}' diverged"
    for key, before_value in target_before.items():
        assert torch.equal(fresh._target_network.state_dict()[key], before_value), f"target weight '{key}' diverged"

    fresh_optimizer_state = fresh._optimizer.state_dict()
    assert fresh_optimizer_state["param_groups"] == optimizer_before["param_groups"]
    for param_id, before_state in optimizer_before["state"].items():
        after_state = fresh_optimizer_state["state"][param_id]
        for buffer_key, before_value in before_state.items():
            after_value = after_state[buffer_key]
            if isinstance(before_value, torch.Tensor):
                assert torch.equal(after_value, before_value), f"optimizer buffer '{buffer_key}' diverged"
            else:
                assert after_value == before_value

    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() == python_random_before
    assert fresh._rng.bit_generator.state == numpy_generator_before


@pytest.mark.unit
def test_legacy_list_encoded_format_still_loads(tiny_dqn) -> None:
    """A pre-S05 (list-encoded, online-network-only) checkpoint must keep loading (T045).

    Built inline rather than as a committed fixture file: this task's file
    ownership is scoped to `src/hercule/models/deep_q_learning/__init__.py` plus
    three named test modules, not a new fixtures directory. The payload shape
    matches exactly what the removed `tensor.tolist()` + `json.dump` encoding
    used to produce -- a plain list-of-lists per tensor, no `format_version` key,
    no target network, no optimizer state, no RNG state -- so it exercises the
    same `_import_legacy_format` branch a real file already under `outputs/`
    would.
    """
    model, _ = tiny_dqn()
    reference_state = {k: v.clone() for k, v in model._q_network.state_dict().items()}
    legacy_payload = {
        "q_network_state_dict": {k: v.cpu().tolist() for k, v in reference_state.items()},
        "epoch_count": 7,
        "step_count": 70,
        "frame_stack": 0,
        "observation_shape": list(model._q_network.observation_shape),
    }

    fresh, _ = tiny_dqn()
    fresh.load_from_dict(legacy_payload)

    for key, value in reference_state.items():
        assert torch.equal(fresh._q_network.state_dict()[key], value)
    # The legacy format never saved the target network: the old (lag-destroying)
    # behaviour of loading the online dict into the target too must be reproduced
    # exactly for a checkpoint that never had anything better to offer.
    for key, value in reference_state.items():
        assert torch.equal(fresh._target_network.state_dict()[key], value)
    assert fresh._epoch_count == 7
    assert fresh._step_count == 70


@pytest.mark.unit
def test_epsilon_resumes_from_its_saved_value_not_the_yaml_default(tiny_dqn) -> None:
    """After save-then-load, `epsilon` continues from its saved value (T046)."""
    trained, _, hyperparameters = _build_and_train(tiny_dqn, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.01, epochs=1)
    decayed_epsilon = trained.get_hyperparameters().epsilon
    assert decayed_epsilon < 1.0, "epsilon must have decayed for this test to mean anything"

    payload = trained._export()
    fresh, _ = tiny_dqn(**hyperparameters)  # a fresh configure() resets epsilon to the YAML value
    assert fresh.get_hyperparameters().epsilon == pytest.approx(1.0)

    fresh._import(payload)

    assert fresh.get_hyperparameters().epsilon == pytest.approx(decayed_epsilon)


@pytest.mark.unit
def test_target_network_lag_survives_a_save_and_load(tiny_dqn) -> None:
    """After save-then-load, the target network stays unequal to the online one (T047)."""
    trained, _, hyperparameters = _build_and_train(tiny_dqn)
    online_state = trained._q_network.state_dict()
    target_state = trained._target_network.state_dict()
    assert any(not torch.equal(online_state[k], target_state[k]) for k in online_state), (
        "target must have real lag before saving, for this test to mean anything"
    )

    payload = trained._export()
    fresh, _ = tiny_dqn(**hyperparameters)
    fresh._import(payload)

    fresh_online = fresh._q_network.state_dict()
    fresh_target = fresh._target_network.state_dict()
    assert any(not torch.equal(fresh_online[k], fresh_target[k]) for k in fresh_online), (
        "target network lost its lag across save/load"
    )
    for key, before_value in target_state.items():
        assert torch.equal(fresh_target[key], before_value), f"restored target weight '{key}' does not match"


class _CarRacingShapedEnv(gym.Env):
    """A single-frame stand-in for CarRacing-v3's own (96, 96, 3) uint8 observation.

    `frame_stack=3` below reproduces the real 4-frame-stacked network this repo
    trains on (`experiments/dq_car_racing.yaml`): `QNetwork((96, 96, 12), 5)`,
    2,194,597 parameters -- the exact shape the roadmap's 133.9 MB measurement
    and the `_encode_state_dict` benchmark in `_export`'s docstring were taken
    from.
    """

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros((96, 96, 3), dtype=np.uint8), 0.0, True, False, {}


@pytest.mark.unit
def test_carracing_shaped_checkpoint_size(tmp_path) -> None:
    """Measure `model.json` for a TRAINED, stacked CarRacing-shaped network (T049).

    This is a REGRESSION GUARD against a return to the pre-S05 encoding (133.9 MB
    measured on disk, `tensor.tolist()` + `json.dump`), not an enforcement of the
    roadmap's original "under 15 MB" acceptance bar: measured here at 22-45 MB
    (untrained / trained), that bar is unreachable while also satisfying T038
    (the target network must be serialised in its own right, not copied from the
    online one) and T039 (the optimizer state must be serialised too).

    The assertion measures the TRAINED state on purpose, not the state right
    after `configure()`. Adam allocates its `exp_avg`/`exp_avg_sq` buffers
    lazily, on the first `step()` -- an untrained checkpoint contains only the
    two networks (~22.3 MB) and an empty optimizer `state_dict()["state"]`, so
    it never exercises the optimizer-state half of what this guard is meant to
    protect. Training for a couple of epochs first (batch_size=2, step_modulo=1
    so the second epoch's push already meets the batch size) forces one real
    gradient step, populating those buffers before `save()` runs; the size
    roughly doubles as a result (measured ~44.7 MB), since the Adam moment
    buffers are the same size as the network they shadow. The
    `state_dict()["state"]` assertion below exists so this test cannot silently
    regress back to measuring the untrained, optimizer-empty case.
    """
    env = _CarRacingShapedEnv()
    model = DeepQLearningModel()
    assert model.configure(env, {"frame_stack": 3, "seed": 42, "batch_size": 2, "step_modulo": 1})
    assert tuple(model._q_network.observation_shape) == (96, 96, 12)

    # `_CarRacingShapedEnv` terminates after one step, so each `run_epoch` pushes
    # exactly one transition; two epochs fill the batch of 2 and trigger exactly
    # one gradient step.
    model.run_epoch(train_mode=True)
    model.run_epoch(train_mode=True)

    optimizer_state = model._optimizer.state_dict()["state"]
    assert optimizer_state, "expected a gradient step to have populated Adam's optimizer state before measuring"

    model.save(tmp_path)

    model_file = tmp_path / "model.json"
    size_mb = model_file.stat().st_size / (1024 * 1024)
    # 50 MB still leaves an ample margin under the legacy 133.9 MB while catching
    # a real regression (e.g. a reintroduced tolist()+json.dump path).
    assert size_mb < 50, f"model.json is {size_mb:.2f} MB, expected well under the legacy 133.9 MB"
