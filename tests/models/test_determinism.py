"""Tests that `seed` actually controls Deep Q-Learning's randomness (spec 006, S04).

Before this fix `DeepQLearningModelHyperParams.seed` was declared but never read:
`__init__` hard-coded `torch.manual_seed(42)`, and Python's `random` module (which
drives epsilon-greedy exploration and replay sampling) was never seeded at all. Two
YAML runs with `seed: 1` and `seed: 999` differed only through uncontrolled RNG.
"""

import ast
import inspect
from pathlib import Path

import pytest
import torch

import hercule.models.deep_q_learning as dqn_module
from hercule.models.deep_q_learning import DeepQLearningModel


def _run_training(stub_env_factory, seed: int, episode_length: int = 6, epochs: int = 3):
    """Configure a fresh model with `seed` and train it for a few short episodes.

    `epsilon=0.5` (rather than 0 or 1) matters here: it makes exploration a coin
    flip on every step, so the test actually exercises the seeded `random` stream
    instead of always taking the same branch regardless of seed.
    """
    env = stub_env_factory(episode_length=episode_length)
    model = DeepQLearningModel()
    hyperparameters = {
        "learning_rate": 0.01,
        "batch_size": 2,
        "replay_buffer_size": 16,
        "epsilon": 0.5,
        "epsilon_decay": 0.0,
        "epsilon_min": 0.5,
        "step_modulo": 1,
        "target_update_frequency": 1000,
        "seed": seed,
    }
    assert model.configure(env, hyperparameters)
    model.env = env

    total_reward = 0.0
    for _ in range(epochs):
        result = model.run_epoch(train_mode=True)
        total_reward += result.reward
    return model, total_reward


@pytest.mark.unit
def test_same_seed_produces_bit_identical_weights_and_rewards(stub_env_factory) -> None:
    """Two independently built-and-trained models with the same seed must match exactly."""
    model_a, reward_a = _run_training(stub_env_factory, seed=42)
    model_b, reward_b = _run_training(stub_env_factory, seed=42)

    assert reward_a == pytest.approx(reward_b)
    weights_a = model_a._q_network.state_dict()
    weights_b = model_b._q_network.state_dict()
    assert weights_a.keys() == weights_b.keys()
    for key in weights_a:
        assert torch.equal(weights_a[key], weights_b[key]), f"weight '{key}' diverged despite identical seed"


@pytest.mark.unit
def test_different_seed_produces_different_weights(stub_env_factory) -> None:
    """Two different seeds must diverge.

    This assertion already passed before S04, but for the wrong reason:
    `torch.manual_seed(42)` was hard-coded in `__init__`, so every model had
    identical initial weights regardless of what `seed` said, and only the
    (also unseeded) `random`-driven exploration could make two runs differ. It is
    meaningful now that `configure()` actually seeds torch from the hyperparameter.
    """
    model_a, _ = _run_training(stub_env_factory, seed=1)
    model_b, _ = _run_training(stub_env_factory, seed=2)

    weights_a = model_a._q_network.state_dict()
    weights_b = model_b._q_network.state_dict()
    assert any(not torch.equal(weights_a[key], weights_b[key]) for key in weights_a), (
        "weights are identical across different seeds"
    )


@pytest.mark.unit
def test_no_bare_global_numpy_random_call() -> None:
    """Every NumPy draw in this module must go through the model-owned `_rng`.

    S05 checkpoints `_rng.bit_generator.state`, which is a complete record of
    consumed randomness only if nothing bypasses the owned generator by calling a
    global `np.random.*` function instead. `np.random.default_rng(...)` is the sole
    permitted call: it is how `_rng` itself is constructed, and unlike every other
    `np.random.*` function it neither reads nor mutates global state.

    Parses the module as an AST rather than grepping for the text `np.random.` so
    that prose mentioning a forbidden call by name (e.g. this module's own
    docstring explaining why `np.random.get_state()` must not be used) cannot
    trip a false positive: a comment or docstring is never an `ast.Call` node.
    """
    source_path = Path(inspect.getfile(dqn_module))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    disallowed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_np_random_attr = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
            and func.value.attr == "random"
        )
        if is_np_random_attr and func.attr != "default_rng":
            disallowed.append(f"np.random.{func.attr}")

    assert not disallowed, f"bare global np.random calls found: {disallowed}"
