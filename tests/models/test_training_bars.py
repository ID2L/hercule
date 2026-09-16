"""End-to-end training bars: the only checks that can catch a missed numeric target.

Every other SAC test in this package pins one clause of the algorithm against a
hand-computed value or a structural property. None of them can catch a defect that
leaves every clause individually correct but the combination unable to actually
solve anything -- and a YAML config plus a bare `hercule learn` trains a model and
asserts nothing back. These two tests are the ones that close that gap: they train
a real `SACModel` for its full budget and assert the numeric bar the corresponding
experiment config (`experiments/sac_oracle.yaml`, `experiments/sac_pendulum.yaml`)
exists to reach.

Both are marked `slow` -- 200 and 500 full episodes respectively -- and are not
part of the default test run.
"""

import gymnasium as gym
import numpy as np
import pytest

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID, OPTIMAL_RETURN, SUCCESS_BAR
from hercule.models.sac import SACModel


@pytest.mark.slow
def test_sac_clears_the_oracle_success_bar() -> None:
    """SC-002: SAC must reach 90% of the oracle's closed-form optimum.

    `AsymmetricOracle-v0`'s optimal return is known exactly (`OPTIMAL_RETURN`,
    50.0): the observation IS the target, so the task is trivially learnable and
    any failure to clear the bar points at the algorithm, not at the environment's
    difficulty. The bar (`SUCCESS_BAR`, 45.0) is 90% of that optimum.

    The oracle's two action dimensions are deliberately asymmetric (`[-1, 1]` and
    `[0, 4]`, with the second dimension's target confined to its upper half), which
    is exactly what makes a per-dimension or action-mapping defect -- e.g. a
    normalised policy output forwarded unscaled to the environment -- visible here
    and invisible on the symmetric, one-dimensional `Pendulum-v1`. Reaching the bar
    is therefore evidence about both the learning algorithm and the action mapping
    at once; the second assertion below isolates the mapping half explicitly.
    """
    env = gym.make(ENVIRONMENT_ID)
    model = SACModel()
    assert model.configure(
        env,
        {"learning_starts": 500, "batch_size": 128, "replay_buffer_size": 20000, "seed": 42},
    )
    model.env = env

    for _ in range(200):
        model.run_epoch(train_mode=True)

    total_reward = 0.0
    for _ in range(20):
        result = model.run_epoch(train_mode=False)
        total_reward += result.reward
    mean_reward = total_reward / 20

    assert mean_reward >= SUCCESS_BAR, (
        f"mean test reward {mean_reward:.2f} is below the success bar {SUCCESS_BAR:.2f} "
        f"(closed-form optimum {OPTIMAL_RETURN:.2f}); SAC failed to solve an environment "
        "whose optimal policy is known exactly"
    )

    # Isolate the action-mapping half of the bar: drive evaluation episodes by hand
    # so every individual action submitted to the environment can be checked, not
    # just the aggregate reward it produced. A policy that occasionally emits an
    # out-of-bounds action would be silently clamped by some environments and would
    # not necessarily show up as a reward shortfall, so this is checked separately.
    out_of_bounds = []
    for _ in range(5):
        observation, _ = env.reset()
        model.begin_episode()
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.asarray(model.predict(observation), dtype=np.float32)
            if not env.action_space.contains(action):
                out_of_bounds.append(action)
            observation, _, terminated, truncated = env.step(action)[:4]

    assert not out_of_bounds, (
        f"{len(out_of_bounds)} action(s) fell outside {env.action_space} during evaluation, e.g. "
        f"{out_of_bounds[0]}; the normalised-to-environment action map is not confining the policy "
        "to the environment's (asymmetric) bounds"
    )


@pytest.mark.slow
def test_sac_clears_the_pendulum_success_bar() -> None:
    """SC-001: SAC must reach a mean test reward strictly above -200 on Pendulum-v1.

    `Pendulum-v1` NEVER terminates -- every one of its 200 steps ends in a
    truncation, never a termination -- so this is the first bar on which treating a
    truncated step as terminal would bootstrap the learning target from a wrong
    value (zero continuation) on every single training episode, rather than on an
    occasional edge case.
    """
    env = gym.make("Pendulum-v1")
    model = SACModel()
    assert model.configure(env, {"learning_starts": 1000, "batch_size": 256, "seed": 42})
    model.env = env

    for _ in range(500):
        model.run_epoch(train_mode=True)

    total_reward = 0.0
    for _ in range(20):
        result = model.run_epoch(train_mode=False)
        total_reward += result.reward
    mean_reward = total_reward / 20

    assert mean_reward > -200, (
        f"mean test reward {mean_reward:.2f} does not clear the -200 bar over 20 evaluation episodes"
    )
