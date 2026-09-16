"""SAC-specific persistence coverage (spec 007, Phase 6, T059-T060).

`tests/models/test_persistence.py` already pins the shared `OffPolicyReplayModel`
checkpoint machinery (round-trip fidelity, the legacy format, the 50 MB size guard)
against `DeepQLearningModel`, a model with **two** networks and **one** optimizer.
SAC holds **five** networks (actor, two critics, two delayed copies of those
critics) and **four** optimizers (actor, critic_1, critic_2, temperature -- three
of them network-sized) plus a learned scalar temperature that is not itself a
module parameter. Two things follow that the DQN suite cannot exercise:

* SC-008's checkpoint-size arithmetic is entirely different (`5 x 11.70 + 3 x 2 x
  11.70` ~= 129 MB, see the spec's Measured Baselines section) and the existing
  50 MB guard is, by that same arithmetic, unreachable by this model -- it is
  arithmetically impossible for a SAC checkpoint to satisfy it, not merely
  untested.
* SC-005's resume contract names four specific things -- the temperature, every
  optimizer's state, the delayed copies' lag, and all three RNG streams -- and a
  broken `_load_extra_state()`/`_import()` could restore any strict subset of
  them while still passing a test that checks only one.
"""

import copy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

import hercule.environnements  # noqa: F401 -- registers AsymmetricOracle-v0
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.sac import SACModel


class _CarRacingShapedEnv(gym.Env):
    """A single-frame stand-in for CarRacing-v3(continuous=True)'s own shapes.

    Observation matches `tests/models/test_persistence.py::_CarRacingShapedEnv`
    exactly ((96, 96, 3) uint8), so `frame_stack=3` below reproduces the same
    stacked (96, 96, 12) input the deep model's own checkpoint-size guard was
    measured on. The action space is the one difference that matters here: SAC
    needs a `Box`, and CarRacing's real one is deliberately ASYMMETRIC per
    dimension (`low = [-1, 0, 0]`, `high = [1, 1, 1]` -- steer/gas/brake), which
    is what the real environment declares.
    """

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros((96, 96, 3), dtype=np.uint8), 0.0, True, False, {}


@pytest.mark.unit
def test_carracing_shaped_sac_checkpoint_stays_under_150mb(tmp_path) -> None:
    """SC-008: a checkpoint on a shaped CarRacing-like environment stays under 150 MB.

    150 MB is the spec's own derived figure (Measured Baselines): five networks of
    roughly the single-network size measured for feature 006 (11.70 MB base64-
    encoded) plus optimizer state -- two Adam moment buffers (`exp_avg`,
    `exp_avg_sq`), each network-sized -- for the three TRAINED ones (actor,
    critic_1, critic_2; the temperature optimizer's own moments hold two floats
    and are negligible), `5 x 11.70 + 3 x 2 x 11.70` ~= 129 MB, with headroom added
    for the heads (which differ between the actor and the two critics). This is a
    DIFFERENT bound from `test_persistence.py`'s 50 MB guard, by construction, not
    by a looser tolerance: that guard was measured on a model holding two networks
    and one optimizer, this one holds five networks and three network-sized
    optimizers, so 50 MB is arithmetically unreachable here regardless of how well
    this model is implemented -- see the spec's SC-008 for the full arithmetic.
    Running the existing `test_persistence.py::test_carracing_shaped_checkpoint_size`
    alongside this test confirms that guard is untouched and still green.

    The checkpoint MUST be measured after at least one real gradient step, not on
    an untrained model. Adam allocates its `exp_avg`/`exp_avg_sq` buffers lazily,
    on each parameter's first `optimizer.step()` -- an untrained model's
    `model.json` therefore contains the five networks and NO optimizer state at
    all, regardless of how many optimizers `_optimizers()` reports. Measured on
    this exact environment: an untrained checkpoint is only ~58.50 MB (exactly
    `5 x 11.70`, the five networks alone -- Adam simply never allocated anything),
    against ~128.16 MB once every optimizer has taken a step, matching the 129 MB
    derivation above almost exactly. A test that saves before training measures
    roughly 45% of what the 150 MB bar is meant to bound, and would stay green
    even if the optimizer state doubled or were written twice. So below, the model
    trains for a few steps -- a small `batch_size`, `learning_starts=0` and a
    small `replay_buffer_size` keep it fast -- and the optimizer state is asserted
    non-empty BEFORE saving, so a regression back to measuring the untrained case
    fails loudly here instead of silently passing this test again.
    """
    env = _CarRacingShapedEnv()
    model = SACModel()
    assert model.configure(
        env,
        {
            "frame_stack": 3,
            "seed": 42,
            "batch_size": 2,
            "learning_starts": 0,
            "replay_buffer_size": 16,
            "step_modulo": 1,
        },
    )
    assert model._stacked_observation_shape == (96, 96, 12)
    model.env = env

    # `_CarRacingShapedEnv.step()` always terminates after one step, so each
    # `run_epoch()` call contributes exactly one environment step and one replay
    # push. `batch_size=2` needs two pushes before the first update can fire
    # (`_ready_to_update()`'s `len(replay_buffer) >= batch_size`), so five epochs
    # leaves several spare gradient steps rather than the bare minimum.
    for _ in range(5):
        model.run_epoch(train_mode=True)

    optimizer_states = {name: optimizer.state_dict() for name, optimizer in model._optimizers().items()}
    assert set(optimizer_states) == {"actor", "critic_1", "critic_2", "temperature"}
    for name, state in optimizer_states.items():
        assert state["state"], (
            f"{name} optimizer never took a gradient step; this test would then measure the untrained "
            "checkpoint size again, exactly the defect it exists to catch"
        )

    model.save(tmp_path)

    model_file = tmp_path / "model.json"
    size_mb = model_file.stat().st_size / (1024 * 1024)
    assert size_mb < 150, f"model.json is {size_mb:.2f} MB, expected under the derived 150 MB bound"


def _build_and_train_sac(*, epochs: int, seed: int, **hyperparameter_overrides: object):
    """Configure a small SAC on the oracle and train it briefly.

    `learning_starts: 10` is deliberately nonzero (unlike the oracle/Pendulum
    experiment configs' own warmup values, which would also work): the first few
    steps of every episode must draw a UNIFORM warmup action from `model._rng`
    (`SACModel._select_action`), which is the only thing that ever advances that
    stream at all -- once warmup ends, action selection samples from the actor via
    torch, not from `self._rng`. Without a nonzero warmup, `_rng` never moves away
    from the state `configure()` gives it, and comparing it across save/load would
    pass against a load that restored nothing.

    Returns:
        `(model, hyperparameters)` -- the hyperparameters are returned too, so a
        fresh model can be configured identically without inheriting any state
        this training call mutated (e.g. the adapted temperature).
    """
    hyperparameters = {
        "learning_starts": 10,
        "batch_size": 8,
        "replay_buffer_size": 256,
        "step_modulo": 1,
        "seed": seed,
        **hyperparameter_overrides,
    }
    env = gym.make(ENVIRONMENT_ID)
    model = SACModel()
    assert model.configure(env, hyperparameters)
    model.env = env
    for _ in range(epochs):
        model.run_epoch(train_mode=True)
    return model, hyperparameters


@pytest.mark.integration
def test_sac_resume_restores_every_adapted_quantity_not_just_some(tmp_path) -> None:
    """SC-005: temperature, all four optimizers, target lag and all three RNG
    streams each continue from their stored value across a save/load, none of
    them reverting to what a fresh `configure()` would give it.

    Eight tests already shipped with this feature that could not fail (see the
    round-2 review history in the spec). The discipline followed here to avoid a
    ninth: every restored quantity gets its own "moved during training" sanity
    check (so a no-op training loop could not make the later assertion vacuous)
    AND its own "the fresh baseline actually differs from the stored value"
    check taken BEFORE `load()` runs (so a `load()` that silently did nothing
    could not make the final assertion pass by coincidence).
    """
    trained, hyperparameters = _build_and_train_sac(epochs=5, seed=5)

    # Force an extra torch draw, exactly as test_persistence.py's own round-trip
    # test does: nothing in ordinary operation otherwise proves the restored torch
    # stream is one that has genuinely moved, rather than one a fresh configure()
    # with the same seed would reproduce on its own.
    torch.rand(3)

    log_alpha_before = float(trained._log_alpha.item())
    optimizer_states_before = {
        name: copy.deepcopy(optimizer.state_dict()) for name, optimizer in trained._optimizers().items()
    }
    assert set(optimizer_states_before) == {"actor", "critic_1", "critic_2", "temperature"}
    critic_1_before = {k: v.clone() for k, v in trained._critic_1.state_dict().items()}
    critic_1_target_before = {k: v.clone() for k, v in trained._critic_1_target.state_dict().items()}
    critic_2_before = {k: v.clone() for k, v in trained._critic_2.state_dict().items()}
    critic_2_target_before = {k: v.clone() for k, v in trained._critic_2_target.state_dict().items()}
    torch_rng_before = torch.get_rng_state().clone()
    python_random_before = random.getstate()
    numpy_generator_before = copy.deepcopy(trained._rng.bit_generator.state)
    step_count_before = trained._step_count
    epoch_count_before = trained._epoch_count

    # --- sanity: every quantity below must actually have MOVED during training,
    # or the corresponding assertion after load() would pass vacuously. ---
    assert epoch_count_before == 5
    assert step_count_before > 0
    default_log_alpha = float(np.log(SACModel().get_default_hyperparameters_typed().init_temperature))
    assert log_alpha_before != pytest.approx(default_log_alpha), "temperature never adapted; widen training"
    assert any(not torch.equal(critic_1_before[k], critic_1_target_before[k]) for k in critic_1_before), (
        "critic_1 has no lag from its target yet; this test cannot tell a correct resume from a broken one"
    )
    assert any(not torch.equal(critic_2_before[k], critic_2_target_before[k]) for k in critic_2_before), (
        "critic_2 has no lag from its target yet; this test cannot tell a correct resume from a broken one"
    )
    for name, state in optimizer_states_before.items():
        assert state["state"], f"{name} optimizer never took a step; this test cannot exercise its restored state"
    scratch_numpy_generator = np.random.default_rng(hyperparameters["seed"]).bit_generator.state
    assert numpy_generator_before != scratch_numpy_generator, (
        "model._rng never advanced during training (the warmup path never ran); widen learning_starts"
    )

    trained.save(tmp_path)

    # A FRESH model, configured IDENTICALLY -- the exact baseline `load()` must
    # overwrite. Every quantity checked below is asserted to differ from its
    # stored counterpart HERE, before load() runs, so a load() that silently did
    # nothing could not make the post-load assertions pass by coincidence.
    fresh_env = gym.make(ENVIRONMENT_ID)
    fresh = SACModel()
    assert fresh.configure(fresh_env, hyperparameters)
    fresh.env = fresh_env

    assert float(fresh._log_alpha.item()) == pytest.approx(default_log_alpha)
    assert float(fresh._log_alpha.item()) != pytest.approx(log_alpha_before)

    fresh_optimizer_states_before_load = {
        name: optimizer.state_dict() for name, optimizer in fresh._optimizers().items()
    }
    for name in optimizer_states_before:
        assert fresh_optimizer_states_before_load[name]["state"] == {}, (
            f"{name} optimizer already carries state before any load"
        )

    fresh_critic_1_before_load = {k: v.clone() for k, v in fresh._critic_1.state_dict().items()}
    assert any(
        not torch.equal(fresh_critic_1_before_load[k], critic_1_before[k]) for k in fresh_critic_1_before_load
    ), "a freshly configured critic_1 already matches the trained one; reseed or train longer"
    fresh_critic_2_before_load = {k: v.clone() for k, v in fresh._critic_2.state_dict().items()}
    assert any(
        not torch.equal(fresh_critic_2_before_load[k], critic_2_before[k]) for k in fresh_critic_2_before_load
    ), "a freshly configured critic_2 already matches the trained one; reseed or train longer"

    assert not torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() != python_random_before
    assert fresh._rng.bit_generator.state != numpy_generator_before

    assert fresh._step_count == 0
    assert fresh._epoch_count == 0

    fresh.load(tmp_path)

    # --- the learned temperature ---
    assert float(fresh._log_alpha.item()) == pytest.approx(log_alpha_before)

    # --- every optimizer's state, all four, tensor for tensor ---
    fresh_optimizer_states_after = {name: optimizer.state_dict() for name, optimizer in fresh._optimizers().items()}
    for name, before_state in optimizer_states_before.items():
        after_state = fresh_optimizer_states_after[name]
        assert after_state["param_groups"] == before_state["param_groups"], f"{name} param_groups diverged"
        assert set(after_state["state"]) == set(before_state["state"]), f"{name} optimizer state keys diverged"
        for param_id, before_param_state in before_state["state"].items():
            after_param_state = after_state["state"][param_id]
            for buffer_key, before_value in before_param_state.items():
                after_value = after_param_state[buffer_key]
                if isinstance(before_value, torch.Tensor):
                    assert torch.equal(after_value, before_value), f"{name} optimizer buffer '{buffer_key}' diverged"
                else:
                    assert after_value == before_value

    # --- the delayed copies' LAG: the exact same amount as before saving, not a
    # fresh hard copy of the live network ---
    fresh_critic_1_after = fresh._critic_1.state_dict()
    fresh_critic_1_target_after = fresh._critic_1_target.state_dict()
    for key, before_value in critic_1_before.items():
        assert torch.equal(fresh_critic_1_after[key], before_value), f"critic_1 weight '{key}' diverged"
        assert torch.equal(fresh_critic_1_target_after[key], critic_1_target_before[key]), (
            f"critic_1_target weight '{key}' diverged -- the target's lag was not preserved"
        )
    fresh_critic_2_after = fresh._critic_2.state_dict()
    fresh_critic_2_target_after = fresh._critic_2_target.state_dict()
    for key, before_value in critic_2_before.items():
        assert torch.equal(fresh_critic_2_after[key], before_value), f"critic_2 weight '{key}' diverged"
        assert torch.equal(fresh_critic_2_target_after[key], critic_2_target_before[key]), (
            f"critic_2_target weight '{key}' diverged -- the target's lag was not preserved"
        )

    # --- all three random streams ---
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() == python_random_before
    assert fresh._rng.bit_generator.state == numpy_generator_before

    # --- the step and epoch counters ---
    assert fresh._step_count == step_count_before
    assert fresh._epoch_count == epoch_count_before
