"""The refactor of feature 007 changed no number (FR-002, SC-004).

`OffPolicyReplayModel` was extracted out of `DeepQLearningModel`, and the deep
model rebuilt on it. The requirement is that this changed nothing observable: the
same configuration at the same seed must produce the same episode rewards and the
same weights, tensor for tensor.

The fixture this reads was captured from the pre-refactor code and committed
before a single source file was touched, because it is evidence *about* that code
and cannot be regenerated afterwards.

**This file is temporary by design.** At feature closure it is deleted along with
the fixture and replaced by a determinism property test -- same seed twice gives
the same result, different seeds give different ones -- which stores no recorded
expectation and therefore survives later deliberate behaviour changes without
needing regeneration. A committed fixture asserts *today's numbers*; keeping one
past the refactor it certifies means regenerating it on every intentional change,
at which point it certifies nothing.

The comparison is CPU-only. Floating-point reduction order differs between CPU
and CUDA kernels, so bit-identity across devices is not achievable, and asserting
it would produce a test that fails on exactly the machines that have a GPU.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_FILE = FIXTURE_DIR / "golden" / "dqn_baseline.json"


def _load_capture_module():
    """Import the capture script by path.

    `tests/` is not a package, so a plain import would not resolve. Loading the
    module the capture was produced by -- rather than restating its environments
    and hyperparameters here -- is what makes this a comparison of the code under
    test and not a comparison of two transcriptions.
    """
    spec = importlib.util.spec_from_file_location("capture_baselines", FIXTURE_DIR / "capture_baselines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()
FIXTURE = json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))

BUILDERS = {
    "cartpole": lambda: __import__("gymnasium").make("CartPole-v1"),
    "frozenlake": lambda: __import__("gymnasium").make("FrozenLake-v1", is_slippery=False),
    "shaped_image": capture.ShapedImageEnv,
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(FIXTURE["configurations"]))
def test_refactor_reproduces_the_pre_refactor_baseline(name: str) -> None:
    """Rewards, weights, epsilon and counters all match the captured baseline."""
    expected = FIXTURE["configurations"][name]

    with torch.device("cpu"):
        entry, _ = capture._capture(name, BUILDERS[name](), {}, expected["epochs"])

    assert entry["rewards"] == expected["rewards"], (
        f"{name}: the episode reward series changed. If the weights below also differ, suspect "
        "module construction ORDER before suspecting the algorithm -- a reordered nn.Linear draws "
        "different numbers from the same seed."
    )
    assert entry["online_parameter_hashes"] == expected["online_parameter_hashes"], f"{name}: online weights differ"
    assert entry["target_parameter_hashes"] == expected["target_parameter_hashes"], (
        f"{name}: target weights differ. The target network must keep being CONSTRUCTED rather than "
        "copied from the online one: its constructor consumes a second sequence of torch RNG draws, "
        "and that state is checkpointed even though the weights it produced are overwritten."
    )
    assert entry["epsilon"] == pytest.approx(expected["epsilon"], abs=0.0), (
        f"{name}: the decayed epsilon differs, so the per-step hook moved relative to the episode loop"
    )
    assert entry["step_count"] == expected["step_count"], f"{name}: step count differs"
    assert entry["epoch_count"] == expected["epoch_count"], f"{name}: epoch count differs"
    assert entry["observation_shape"] == expected["observation_shape"], f"{name}: network input shape differs"


@pytest.mark.unit
def test_fixture_covers_both_network_branches() -> None:
    """The fixture is only evidence if it reaches both encoder branches.

    `CartPole-v1` is a 1-D `Box` and `FrozenLake-v1` is `Discrete`, whose
    `_single_frame_shape()` returns `(1,)` -- so both take the MLP branch, three
    parameterised layers, six tensors. Only the shaped environment reaches the
    convolutional branch, five layers, ten tensors. A fixture built on the first
    two alone would leave the branch with the most to get wrong uncovered.
    """
    tensor_counts = {name: len(entry["online_parameter_hashes"]) for name, entry in FIXTURE["configurations"].items()}
    assert tensor_counts["cartpole"] == 6
    assert tensor_counts["frozenlake"] == 6
    assert tensor_counts["shaped_image"] == 10
