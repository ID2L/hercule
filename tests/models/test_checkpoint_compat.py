"""Checkpoints written before feature 007 still load after it (User Story 2, contract C4).

The golden fixture cannot cover this, and the distinction is the whole reason
this file exists. The fixture compares weight *tensors* produced by fresh
training; it never opens a stored file. But a `state_dict`'s keys are attribute
paths, so splitting `QNetwork` into an encoder and a head renames every key in
every `model.json` already on disk -- and `load_state_dict` raises by default on
a key it did not expect. A refactor can therefore be bit-identical under the
fixture and still make every stored model unreadable.

The three inputs were captured from the pre-refactor code, at the same time and
by the same script as the fixture, and are committed alongside it. Unlike the
fixture they are **not** deleted at feature closure: the migration path they test
outlives the refactor that introduced it.
"""

import base64
import copy
import hashlib
import importlib.util
import io
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from hercule.models.deep_q_learning import DeepQLearningModel


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
CHECKPOINTS = FIXTURE_DIR / "checkpoints"
BASELINE = json.loads((FIXTURE_DIR / "golden" / "dqn_baseline.json").read_text(encoding="utf-8"))


def _hashes(module: torch.nn.Module) -> list[str]:
    return [hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for p in module.parameters()]


def _shaped_image_env() -> gym.Env:
    """The environment the image checkpoint was captured on."""
    spec = importlib.util.spec_from_file_location("capture_baselines", FIXTURE_DIR / "capture_baselines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ShapedImageEnv()


CASES = {
    "vector": (lambda: gym.make("CartPole-v1"), "cartpole"),
    "image": (_shaped_image_env, "shaped_image"),
}


@pytest.mark.unit
@pytest.mark.parametrize("branch", sorted(CASES))
def test_a_version_2_checkpoint_still_loads(branch: str) -> None:
    """Both networks come back exactly as they were written, on both branches."""
    build_env, baseline_name = CASES[branch]
    payload = json.loads((CHECKPOINTS / branch / "model.json").read_text(encoding="utf-8"))
    assert payload["format_version"] == 2, "the captured input must be the OLD format, or this proves nothing"

    model = DeepQLearningModel()
    assert model.configure(build_env(), {"seed": 1})
    model.load_from_dict(payload)

    expected = BASELINE["configurations"][baseline_name]
    assert _hashes(model._q_network) == expected["online_parameter_hashes"]
    assert _hashes(model._target_network) == expected["target_parameter_hashes"], (
        "the target network must be restored in its OWN right, never rebuilt from the online weights: "
        "doing so destroys the lag that is its entire purpose"
    )


@pytest.mark.unit
@pytest.mark.parametrize("branch", sorted(CASES))
def test_a_version_2_optimizer_state_attaches_to_the_right_parameters(branch: str) -> None:
    """Adam's moments are keyed by parameter INDEX, so enumeration order is load-bearing.

    A version-2 `optimizer_state_b64` carries `{0: {...}, 1: {...}}` against the
    order `parameters()` yielded before the refactor. If the encoder/head split had
    changed that order, every tensor would still load, every weight hash would still
    match, and Adam's moments would silently attach to the wrong parameters --
    diverging only on the next gradient step, long after any test looked.

    Shapes are the check: on the convolutional branch the ten parameters have ten
    different shapes, so a permutation cannot survive it.
    """
    build_env, _ = CASES[branch]
    payload = json.loads((CHECKPOINTS / branch / "model.json").read_text(encoding="utf-8"))
    assert isinstance(payload["optimizer_state_b64"], str), "version 2 stored one bare, unnamed state"

    model = DeepQLearningModel()
    assert model.configure(build_env(), {"seed": 1})
    model.load_from_dict(payload)

    parameters = list(model._q_network.parameters())
    state = model._optimizer.state_dict()["state"]
    assert state, "no optimizer state was restored at all"
    for index, entry in state.items():
        assert entry["exp_avg"].shape == parameters[index].shape, (
            f"{branch}: Adam's first moment for parameter {index} has shape {tuple(entry['exp_avg'].shape)} "
            f"but that parameter is {tuple(parameters[index].shape)} -- the enumeration order changed"
        )
        assert entry["exp_avg_sq"].shape == parameters[index].shape


@pytest.mark.unit
@pytest.mark.parametrize("branch", sorted(CASES))
def test_the_migration_actually_renames_keys(branch: str) -> None:
    """Guard against the migration quietly becoming a no-op.

    If the stored keys were already the current ones, the test above would pass
    without exercising the migration at all -- so assert directly that the file
    carries the OLD attribute paths and the live module carries the new ones.
    """
    payload = json.loads((CHECKPOINTS / branch / "model.json").read_text(encoding="utf-8"))
    buffer = io.BytesIO(base64.b64decode(payload["networks_b64"]["online"]))
    stored_keys = set(torch.load(buffer, weights_only=True).keys())

    build_env, _ = CASES[branch]
    model = DeepQLearningModel()
    assert model.configure(build_env(), {"seed": 1})
    live_keys = set(model._q_network.state_dict().keys())

    assert stored_keys != live_keys, f"{branch}: stored and live key sets already agree, nothing is being migrated"
    assert all(k.startswith(("encoder.", "head.")) for k in live_keys), f"{branch}: unexpected current layout"
    assert not any(k.startswith(("encoder.", "head.")) for k in stored_keys), (
        f"{branch}: stored file is not the old layout"
    )


@pytest.mark.unit
def test_the_image_checkpoint_carries_the_duplicate_registration() -> None:
    """The image branch registered every parameter twice, and the migration drops the alias.

    `self.network = nn.Sequential(self.conv_layers, self.fc_layers)` put every
    parameter into the state dict a second time under a `network.*` path. The
    current modules register each once, so those aliases must be discarded rather
    than mapped -- mapping them would collide with the canonical names.
    """
    payload = json.loads((CHECKPOINTS / "image" / "model.json").read_text(encoding="utf-8"))
    buffer = io.BytesIO(base64.b64decode(payload["networks_b64"]["online"]))
    stored = torch.load(buffer, weights_only=True)

    aliases = [k for k in stored if k.startswith("network.")]
    canonical = [k for k in stored if k.startswith(("conv_layers.", "fc_layers."))]
    assert aliases and canonical, "the captured file does not carry the duplicate registration"
    assert len(stored) == len(aliases) + len(canonical)

    migrated = DeepQLearningModel()._migrate_parameter_keys(stored, "online")
    assert len(migrated) == len(canonical), "the aliases were not dropped"
    assert not any(k.startswith("network.") for k in migrated)


@pytest.mark.unit
def test_the_pre_006_legacy_format_still_loads() -> None:
    """The oldest format on record: one list-encoded network, nothing else.

    Its keys are the old attribute paths too, so it goes through the same migration
    a version-2 payload does. A legacy branch that loaded them verbatim would fail
    against the current modules -- the opposite of the compatibility it exists for.
    """
    payload = json.loads((CHECKPOINTS / "legacy_pre006.json").read_text(encoding="utf-8"))
    assert "q_network_state_dict" in payload and "networks_b64" not in payload

    model = DeepQLearningModel()
    assert model.configure(gym.make("CartPole-v1"), {"seed": 1})
    before = _hashes(model._q_network)
    model.load_from_dict(payload)

    expected = BASELINE["configurations"]["cartpole"]["online_parameter_hashes"]
    assert before != expected, "the freshly configured weights already match, so the load proves nothing"
    assert _hashes(model._q_network) == expected


@pytest.mark.unit
def test_a_version_3_checkpoint_round_trips(tmp_path) -> None:
    """What this feature writes today comes back whole: networks, optimizer, streams, counters."""
    model = DeepQLearningModel()
    assert model.configure(gym.make("CartPole-v1"), {"seed": 3, "batch_size": 2, "epsilon_decay": 0.01})
    for _ in range(3):
        model.run_epoch(train_mode=True)

    online_before = _hashes(model._q_network)
    target_before = _hashes(model._target_network)
    epsilon_before = model.get_hyperparameters().epsilon
    steps_before = model._step_count
    epochs_before = model._epoch_count
    optimizer_before = copy.deepcopy(model._optimizer.state_dict())
    torch_rng_before = torch.get_rng_state().clone()
    python_rng_before = random.getstate()
    numpy_rng_before = copy.deepcopy(model._rng.bit_generator.state)

    model.save(tmp_path)

    # Move every stream on, so restoring them is distinguishable from never having
    # touched them: without this the assertions below would pass on a no-op import.
    torch.rand(5)
    random.random()
    model._rng.random()
    payload = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert payload["format_version"] == 3
    assert isinstance(payload["optimizer_state_b64"], dict), (
        "version 3 keys optimizer state by name, because a model may now hold several"
    )

    restored = DeepQLearningModel()
    assert restored.configure(gym.make("CartPole-v1"), {"seed": 99})
    restored.load_from_dict(payload)

    assert _hashes(restored._q_network) == online_before
    assert _hashes(restored._target_network) == target_before
    assert restored.get_hyperparameters().epsilon == pytest.approx(epsilon_before, abs=0.0)
    assert restored._step_count == steps_before
    assert restored._epoch_count == epochs_before
    assert not restored._needs_seeded_reset, "a resumed run must not re-issue the seeded first reset"

    # Adam's moments, tensor for tensor -- comparing only the state's KEYS would
    # pass against buffers that were zeroed, swapped or corrupted.
    restored_state = restored._optimizer.state_dict()["state"]
    assert restored_state.keys() == optimizer_before["state"].keys()
    for index, entry in optimizer_before["state"].items():
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(restored_state[index][field], entry[field]), f"optimizer {field}[{index}] differs"
    assert restored._optimizer.param_groups[0]["lr"] == pytest.approx(model._optimizer.param_groups[0]["lr"])

    # All three random streams, which nothing above would notice the absence of:
    # deleting the RNG restore entirely would leave every assertion so far passing.
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert random.getstate() == python_rng_before
    assert restored._rng.bit_generator.state == numpy_rng_before


@pytest.mark.unit
def test_a_stacked_model_loads_into_a_default_configured_one(tmp_path) -> None:
    """The `hercule play` path: configure with DEFAULTS, then load.

    `hercule play` has no access to the training config, so it configures with the
    model's default hyperparameters -- `frame_stack=0` -- and only then loads the
    checkpoint. If the checkpoint did not carry the stack depth the network's shape
    was built from, the freshly configured network would be the wrong shape and the
    load would fail. This is why `frame_stack` and `observation_shape` travel with
    the weights, and the test that would notice if they stopped.
    """
    trained = DeepQLearningModel()
    assert trained.configure(gym.make("CartPole-v1"), {"seed": 5, "frame_stack": 3})
    assert tuple(trained._q_network.observation_shape) == (16,), "4 observations x 4 frames"
    trained.run_epoch(train_mode=True)
    trained.save(tmp_path)

    replayed = DeepQLearningModel()
    assert replayed.configure(gym.make("CartPole-v1"), {})  # defaults: frame_stack = 0
    assert tuple(replayed._q_network.observation_shape) == (4,), "an unstacked network, before loading"

    replayed.load_from_dict(json.loads((tmp_path / "model.json").read_text(encoding="utf-8")))

    assert tuple(replayed._q_network.observation_shape) == (16,), "the network was not rebuilt for the saved stack"
    assert _hashes(replayed._q_network) == _hashes(trained._q_network)

    # And it can actually act, which is what `hercule play` does with it.
    observation, _ = gym.make("CartPole-v1").reset(seed=0)
    replayed.begin_episode()
    action = replayed.predict(observation)
    assert action in range(2)
