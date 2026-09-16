"""The oracle can detect what it exists to detect (SC-002, contract C2).

This is the cheapest test in the feature and the one with the most leverage. It
does not train anything; it establishes that the environment's five closed-form
policies land where the contract says they do, and therefore that a training run
against it means something.

A first design of this environment did not pass its own third row and could not
have: with the target on `[1.5, 2.0]`, any distribution on an interval that narrow
has variance at most `0.0625`, so clamping that dimension cost too little to fall
below the bar however the rest was tuned. No amount of training would have revealed
that -- the run would simply have succeeded while the defect it was meant to catch
went unnoticed.
"""

import gymnasium as gym
import numpy as np
import pytest

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import (
    ACTION_HIGH,
    ACTION_LOW,
    ENVIRONMENT_ID,
    EPISODE_LENGTH,
    OPTIMAL_RETURN,
    SUCCESS_BAR,
    TARGET_HIGH,
    TARGET_LOW,
)


EPISODES = 200  # enough that each row's empirical mean is within ~0.2 of its closed form


def _mean_return(policy, episodes: int = EPISODES, seed: int = 0) -> float:
    """Mean return over several episodes, for a policy of the observed target."""
    env = gym.make(ENVIRONMENT_ID)
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total, done = 0.0, False
        while not done:
            observation, reward, terminated, truncated, _ = env.step(policy(observation))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


@pytest.mark.unit
def test_the_optimal_policy_reaches_the_closed_form_optimum() -> None:
    """Acting the observed target scores exactly the analytic optimum."""
    assert _mean_return(lambda target: target) == pytest.approx(OPTIMAL_RETURN, abs=1e-4)
    assert SUCCESS_BAR == pytest.approx(45.0)


@pytest.mark.unit
def test_the_optimum_is_inside_the_action_bounds() -> None:
    """A criterion the agent cannot satisfy would be a defect in the environment."""
    assert np.all(TARGET_LOW >= ACTION_LOW)
    assert np.all(TARGET_HIGH <= ACTION_HIGH)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dimension", "best_constant", "expected"),
    [(0, 0.0, 50.0 * (1 - 0.5 * 1 / 3)), (1, 3.0, 50.0 * (1 - 0.5 * 1 / 3))],
)
def test_clamping_either_dimension_falls_below_the_bar(dimension: int, best_constant: float, expected: float) -> None:
    """The property the oracle exists for: BOTH dimensions must matter.

    The constant is the best one available -- the coordinate's mean -- and the other
    dimension is tracked perfectly. Even so the policy falls short, which is what
    makes an agent that learns only one dimension detectable. Quantified over the
    *best* constant and not an arbitrary one: clamping a dimension to the value it
    takes in the optimal action would leave the optimum reachable, so a criterion
    phrased over any constant would be unsatisfiable rather than merely demanding.
    """

    def policy(target: np.ndarray) -> np.ndarray:
        action = np.array(target, dtype=np.float32)
        action[dimension] = best_constant
        return action

    achieved = _mean_return(policy)
    assert achieved == pytest.approx(expected, abs=0.5)
    assert achieved < SUCCESS_BAR, f"clamping dimension {dimension} still reaches {achieved:.2f}"


@pytest.mark.unit
def test_the_best_constant_action_falls_below_the_bar() -> None:
    """An agent that learned nothing at all cannot pass."""
    best_constant = np.array([0.0, 3.0], dtype=np.float32)
    achieved = _mean_return(lambda _target: best_constant)
    assert achieved == pytest.approx(50.0 * (1 - 0.5 * 2 / 3), abs=0.5)
    assert achieved < SUCCESS_BAR


@pytest.mark.unit
def test_an_unscaled_policy_falls_far_below_the_bar() -> None:
    """A policy forwarding normalised output unscaled cannot reach the optimum.

    Its second coordinate is capped at `1.0` while the optimal one is never below
    `2.0`, so the mis-mapping shows up in the SCORE and not merely in an
    out-of-bounds count. That matters: an implementation that clips its actions into
    the bounds would satisfy a bounds check while still being mis-mapped.
    """
    achieved = _mean_return(lambda target: np.array([target[0], 1.0], dtype=np.float32))
    assert achieved == pytest.approx(50.0 * (1 - 0.5 * 13 / 3), abs=0.5)
    assert achieved < SUCCESS_BAR


@pytest.mark.unit
def test_every_episode_truncates_and_none_terminates() -> None:
    """Exercises the terminated/truncated separation on every episode, like Pendulum."""
    env = gym.make(ENVIRONMENT_ID)
    observation, _ = env.reset(seed=1)
    for step in range(EPISODE_LENGTH):
        observation, _, terminated, truncated, _ = env.step(observation)
        assert not terminated
        assert truncated == (step == EPISODE_LENGTH - 1)
    env.close()
