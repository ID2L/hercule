"""Capture the pre-refactor baselines that feature 007 is verified against.

Run this **once, from `main`, before a single source file is touched**:

    uv run python tests/fixtures/capture_baselines.py

Both artifacts it writes are evidence *about* the pre-refactor code, so only the
pre-refactor code can produce them. Once the refactor of feature 007 begins they
cannot be regenerated, which is precisely what makes them evidence.

Two artifacts, two different jobs:

* ``golden/dqn_baseline.json`` -- per-episode reward series and one SHA-256 per
  parameter tensor, for three configurations. Certifies that the refactor
  computes the same **numbers**. Hashes are taken in ``parameters()`` **order**
  and are deliberately NOT keyed by parameter name: the refactor renames every
  attribute path (`QNetwork` becomes an encoder plus a head), so a name-keyed
  fixture would fail the refactor it exists to certify, for a reason that has
  nothing to do with the numbers.
* ``checkpoints/`` -- real, loadable `model.json` files. Certify that the
  refactor leaves those numbers **readable**. The fixture cannot do this: it
  compares tensors and never opens a file, so a rename that breaks every stored
  checkpoint is invisible to it.

The three configurations cover both network branches. `CartPole-v1` is a 1-D
`Box` observation and `FrozenLake-v1` is `Discrete` -- whose
`_single_frame_shape()` returns `(1,)` -- so **both take the MLP branch**;
FrozenLake earns its place by covering the `Discrete` observation rescaling path
instead. Only the shaped image environment reaches `_build_cnn`, which is the
branch with five parameterised layers rather than three, and the only one whose
`state_dict` registers each parameter twice.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from hercule.models.deep_q_learning import DeepQLearningModel


FIXTURES = Path(__file__).parent
GOLDEN = FIXTURES / "golden"
CHECKPOINTS = FIXTURES / "checkpoints"

# Chosen so the capture exercises the machinery the refactor moves, rather than
# leaving it dormant: a non-zero `epsilon_decay` so `run_epoch`'s per-step decay
# actually runs (the default is 0.0, which would leave the hook the refactor
# relocates untested), and a short `target_update_frequency` so at least one
# target synchronisation happens inside the captured run.
HYPERPARAMETERS: dict[str, object] = {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.002,
    "epsilon_min": 0.05,
    "replay_buffer_size": 500,
    "batch_size": 8,
    "step_modulo": 1,
    "target_update_frequency": 50,
    "seed": 42,
}


class ShapedImageEnv(gym.Env):
    """A small 3-D `Box` observation, purely to reach `QNetwork._build_cnn`.

    36x36 is the smallest square that survives the three hard-coded convolutions
    (kernel 8 stride 4, then 4/2, then 3/1) without collapsing to a zero-sized
    feature map. The real CarRacing shape (96, 96, 3) would work too and produce
    a ~45 MB checkpoint; this one produces well under a megabyte, which matters
    because `checkpoints/` is committed and, unlike the golden fixture, is not
    deleted at feature closure.

    The observation is a deterministic function of the step index so the run
    stays reproducible without drawing from any random stream.
    """

    SIZE = 36
    EPISODE_LENGTH = 10

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.SIZE, self.SIZE, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0

    def _observation(self) -> np.ndarray:
        return np.full((self.SIZE, self.SIZE, 3), (self._step * 17) % 256, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        reward = 1.0 if int(action) == self._step % 4 else 0.0
        truncated = self._step >= self.EPISODE_LENGTH
        return self._observation(), reward, False, truncated, {}


def _hash_parameters(module: torch.nn.Module) -> list[str]:
    """SHA-256 per parameter tensor, in `parameters()` order.

    Order rather than name: `parameters()` yields the same sequence before and
    after the encoder/head split, while every key in `state_dict()` is renamed by
    it. De-duplication is `parameters()`' own behaviour and is what makes the
    convolutional branch's double registration invisible here -- which is correct,
    since the duplicate is the same tensor.
    """
    return [hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for p in module.parameters()]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover - provenance only
        return "unknown"


def _capture(name: str, env: gym.Env, overrides: dict[str, object], epochs: int) -> tuple[dict, DeepQLearningModel]:
    """Train one configuration and return its fixture entry plus the trained model."""
    model = DeepQLearningModel()
    hyperparameters = {**HYPERPARAMETERS, **overrides}
    if not model.configure(env, hyperparameters):
        msg = f"{name}: configure() refused the environment"
        raise RuntimeError(msg)
    model.env = env

    rewards = [model.run_epoch(train_mode=True).reward for _ in range(epochs)]

    entry = {
        "epochs": epochs,
        "hyperparameters": hyperparameters,
        "rewards": rewards,
        "online_parameter_hashes": _hash_parameters(model._q_network),
        "target_parameter_hashes": _hash_parameters(model._target_network),
        "epsilon": model.get_hyperparameters().epsilon,
        "step_count": model._step_count,
        "epoch_count": model._epoch_count,
        "observation_shape": list(model._q_network.observation_shape),
    }
    return entry, model


def _write_legacy_checkpoint(model: DeepQLearningModel, destination: Path) -> None:
    """Project a state dict into the pre-006 encoding.

    No code in the tree writes this format any more -- feature 006 replaced it --
    so it cannot be captured from a run. It can be reproduced exactly, because the
    old encoding was a pure function of the state dict: every tensor rendered as
    nested lists under a single `q_network_state_dict` key. That is what
    `_import_legacy_format` reads, so a file built this way exercises the real
    legacy path rather than a guess at it.
    """
    payload = {
        "model_name": model.model_name,
        "q_network_state_dict": {k: v.detach().cpu().tolist() for k, v in model._q_network.state_dict().items()},
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    configurations = [
        ("cartpole", lambda: gym.make("CartPole-v1"), {}, 12),
        ("frozenlake", lambda: gym.make("FrozenLake-v1", is_slippery=False), {}, 20),
        ("shaped_image", ShapedImageEnv, {}, 8),
    ]

    fixture: dict[str, object] = {
        "captured_from_commit": _git_head(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gymnasium_version": gym.__version__,
        "note": (
            "Parameter hashes are in parameters() order, never keyed by name: the feature-007 "
            "refactor renames every attribute path, and a name-keyed fixture would fail the "
            "refactor it exists to certify."
        ),
        "configurations": {},
    }

    trained: dict[str, DeepQLearningModel] = {}
    for name, build_env, overrides, epochs in configurations:
        entry, model = _capture(name, build_env(), overrides, epochs)
        fixture["configurations"][name] = entry
        trained[name] = model
        print(f"  {name}: {epochs} epochs, {len(entry['online_parameter_hashes'])} tensors, rewards={entry['rewards']}")

    (GOLDEN / "dqn_baseline.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {GOLDEN / 'dqn_baseline.json'}")

    # Two real checkpoints, one per network branch. `cartpole` covers the MLP
    # path, `shaped_image` the convolutional one -- the branch whose state dict
    # registers each parameter twice and whose migration is therefore the one
    # that can go wrong.
    for name, directory in (("cartpole", "vector"), ("shaped_image", "image")):
        target = CHECKPOINTS / directory
        target.mkdir(parents=True, exist_ok=True)
        trained[name].save(target)
        size_kb = (target / "model.json").stat().st_size / 1024
        print(f"wrote {target / 'model.json'} ({size_kb:.0f} KB)")

    _write_legacy_checkpoint(trained["cartpole"], CHECKPOINTS / "legacy_pre006.json")
    print(f"wrote {CHECKPOINTS / 'legacy_pre006.json'}")


if __name__ == "__main__":
    main()
