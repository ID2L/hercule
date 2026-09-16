"""The normalised-to-environment action map, per dimension (SC-003).

Checked against the real `CarRacing-v3(continuous=True)` action space, because the
asymmetric case is the one that matters and it is the project's own target: the
bounds are `low = [-1, 0, 0]` and `high = [1, 1, 1]`, so a policy whose normalised
output is forwarded unchanged applies negative throttle and negative brake.
Gymnasium does not reject that. The car simply never accelerates, and the run still
produces a plausible-looking reward curve.

Both halves of the check are here. A correct map is not enough if the methods
callers actually use return the policy's coordinates instead of the environment's:
`hercule play` and the evaluation loop both drive the environment through
`predict()`, so returning the wrong representation there reintroduces the same
failure on the path the mapping exists to protect.
"""

import gymnasium as gym
import numpy as np
import pytest

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.deep_q_learning import DeepQLearningModel
from hercule.models.sac import SACModel


CAR_RACING = ("CarRacing-v3", {"continuous": True})


def _configured(environment_id: str, **kwargs) -> SACModel:
    model = SACModel()
    env = gym.make(environment_id, **kwargs)
    assert model.configure(env, {"seed": 1, "learning_starts": 0})
    model.env = env
    return model


@pytest.mark.integration
def test_the_endpoints_map_exactly_onto_the_real_carracing_bounds() -> None:
    """`-1` lands on each dimension's `low`, `+1` on its `high`, per dimension."""
    model = _configured(CAR_RACING[0], **CAR_RACING[1])
    space = model.env.action_space
    assert np.allclose(space.low, [-1.0, 0.0, 0.0]), "the environment's bounds are not what this test assumes"
    assert np.allclose(space.high, [1.0, 1.0, 1.0])

    lowest = model.to_env_action(np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    highest = model.to_env_action(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    assert np.allclose(lowest, space.low), f"{lowest} should be exactly {space.low}"
    assert np.allclose(highest, space.high), f"{highest} should be exactly {space.high}"


@pytest.mark.integration
def test_an_unmapped_action_would_be_out_of_bounds() -> None:
    """Evidence the map is doing work rather than being an identity in disguise.

    The one-sided dimensions are the whole point: a normalised `-0.5` on throttle is
    a legal normalised value and an illegal environment action.
    """
    model = _configured(CAR_RACING[0], **CAR_RACING[1])
    normalised = np.array([-0.3, -0.5, -0.2], dtype=np.float32)

    assert not model.env.action_space.contains(normalised), "the unscaled action is already in bounds here"
    assert model.env.action_space.contains(model.to_env_action(normalised))


@pytest.mark.integration
def test_the_map_round_trips() -> None:
    """`to_policy_action` inverts `to_env_action`, per dimension."""
    model = _configured(CAR_RACING[0], **CAR_RACING[1])
    for normalised in ([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.3, -0.7, 0.9]):
        original = np.array(normalised, dtype=np.float32)
        assert np.allclose(model.to_policy_action(model.to_env_action(original)), original, atol=1e-6)


@pytest.mark.integration
def test_act_and_predict_return_environment_coordinates() -> None:
    """The second half of the oracle, and the one `hercule play` depends on.

    `RLModel.evaluate()` and `controller.play_interactive()` both drive their own
    episode loop through `predict()`. Returning the policy's own coordinates there
    would send raw normalised values to the environment -- exactly the failure the
    map exists to prevent, reintroduced on the evaluation path.
    """
    model = _configured(CAR_RACING[0], **CAR_RACING[1])
    space = model.env.action_space
    observation, _ = model.env.reset(seed=0)

    model.begin_episode()
    for training in (False, True):
        action = np.asarray(model.act(observation, training=training), dtype=np.float32)
        assert space.contains(action), f"act(training={training}) returned {action}, outside {space}"

    predicted = np.asarray(model.predict(observation), dtype=np.float32)
    assert space.contains(predicted), f"predict() returned {predicted}, outside {space}"


@pytest.mark.unit
def test_the_map_is_refused_on_a_discrete_action_space() -> None:
    """A model whose coordinates are already the environment's has no reason to call it.

    Returning the identity instead would make that defect silent, which is the
    opposite of what the guard is for.
    """
    model = DeepQLearningModel()
    assert model.configure(gym.make("CartPole-v1"), {"seed": 1})
    assert model._action_low is None, "bounds were read from a Discrete space, which has none"

    with pytest.raises(ValueError, match="Box action space"):
        model.to_env_action(np.array([0.0], dtype=np.float32))


@pytest.mark.unit
def test_the_stored_action_is_in_policy_coordinates_and_the_executed_one_is_not() -> None:
    """The pair `_select_action` returns, on an environment where the two differ.

    The oracle's second dimension spans `[0, 4]`, so the two coordinate systems are
    genuinely different there and storing the wrong one is detectable.
    """
    model = _configured(ENVIRONMENT_ID)
    observation, _ = model.env.reset(seed=0)
    model.begin_episode()

    env_action, stored_action = model._select_action(observation, training=True)

    assert model.env.action_space.contains(np.asarray(env_action, dtype=np.float32))
    assert np.all(np.abs(stored_action) <= 1.0), "the stored action is not in normalised coordinates"
    assert np.allclose(model.to_env_action(stored_action), env_action, atol=1e-6)
    assert not np.allclose(stored_action, env_action), (
        "the two coordinate systems coincide on this observation, so the distinction is untested"
    )
