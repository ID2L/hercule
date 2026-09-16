"""A two-dimensional continuous environment whose optimum is known in closed form.

`Pendulum-v1` is one-dimensional and symmetric, so an entire class of defect is
structurally invisible on it: anything that only misbehaves once an action has more
than one dimension, or once two dimensions have different bounds. A policy that
learns one coordinate and ignores the other, or one whose normalised output is
forwarded to the environment unscaled, both score exactly as well there as a
correct one.

This environment exists to fail loudly on both. It is **not** a performance
benchmark and no conclusion about an algorithm's quality should be drawn from it.

## Why the numbers are what they are

Two requirements pull in opposite directions, and only a wide enough range
satisfies both. Making a clamped dimension *detectable* is a lower bound on that
dimension's target **variance**; making its optimum unreachable under a mis-mapped
policy is a lower bound on the target's **location**. A first design used a target
on `[1.5, 2.0]` over an action range of `[0, 2]` and was **unsatisfiable**: any
distribution on an interval of width `0.25` has variance at most `0.0625`, so
clamping that dimension at its mean cost too little to fall below the bar however
the rest was tuned -- a criterion no correct implementation could pass.

With dimension 1 spanning `[0, 4]` and its target drawn from `[2, 4]`, both hold:
the variance is `1/3`, and the optimal action there is always at least `2.0` while
a policy emitting normalised values unscaled can never exceed `1.0`.

## The closed form

With `reward = 1 - 0.5 * squared error`, `L` steps and both target coordinates
uniform and independent, a policy costing an expected per-step penalty `p` returns
`L * (1 - p)`. Against a bar at 90% of the optimum it clears exactly when
`p <= 0.1` -- inclusive, because the bar is "at least 90%".

| Policy | Expected squared error | Return | Below 45? |
|---|---|---|---|
| Optimal, `a = T` | 0 | 50.00 | -- |
| Best constant on dim 0 (`0`), dim 1 tracked | `Var(T_0) = 1/3` | 41.67 | yes |
| Best constant on dim 1 (`3`), dim 0 tracked | `Var(T_1) = 1/3` | 41.67 | yes |
| Best fully constant action | `2/3` | 33.33 | yes |
| Normalised output forwarded unscaled | `Var(T_1) + (3-1)^2 = 13/3` | -58.33 | yes |
"""

import gymnasium as gym
import numpy as np


ENVIRONMENT_ID = "AsymmetricOracle-v0"

EPISODE_LENGTH = 50
REWARD_OFFSET = 1.0
ERROR_WEIGHT = 0.5

OPTIMAL_RETURN = EPISODE_LENGTH * REWARD_OFFSET
SUCCESS_BAR = 0.9 * OPTIMAL_RETURN

# Dimension 0 is symmetric and spans the same range as a normalised policy output.
# Dimension 1 is one-sided, twice as wide, and its target lives in the UPPER HALF of
# its range -- which is what makes a mis-mapped policy unable to reach the optimum.
ACTION_LOW = np.array([-1.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 4.0], dtype=np.float32)
TARGET_LOW = np.array([-1.0, 2.0], dtype=np.float32)
TARGET_HIGH = np.array([1.0, 4.0], dtype=np.float32)


class AsymmetricOracleEnv(gym.Env):
    """Track a target that moves every step, on two deliberately unequal dimensions.

    The observation IS the target, so the task is learnable in a few thousand steps
    and any failure to reach the bar points at the algorithm rather than at the
    difficulty of the problem.

    Episodes always truncate and never terminate, which exercises the
    terminated/truncated separation on every single episode -- the same property
    that makes `Pendulum-v1` a good first rung.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: str | None = None) -> None:
        """
        Args:
            render_mode: Accepted and ignored; the environment has nothing to draw.
        """
        self.observation_space = gym.spaces.Box(low=TARGET_LOW, high=TARGET_HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self.render_mode = render_mode
        self._target = np.zeros(2, dtype=np.float32)
        self._step = 0

    def _draw_target(self) -> np.ndarray:
        """Draw a fresh target from this environment's own generator."""
        return self.np_random.uniform(TARGET_LOW, TARGET_HIGH).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode and expose the first target."""
        super().reset(seed=seed)
        self._step = 0
        self._target = self._draw_target()
        return self._target.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Score the action against the current target, then draw the next one."""
        action = np.asarray(action, dtype=np.float32).reshape(2)
        error = float(np.sum((action - self._target) ** 2))
        reward = REWARD_OFFSET - ERROR_WEIGHT * error

        self._step += 1
        self._target = self._draw_target()
        truncated = self._step >= EPISODE_LENGTH
        return self._target.copy(), reward, False, truncated, {}


def register() -> None:
    """Register the oracle with Gymnasium, once.

    Called from `hercule.environnements.__init__` rather than left to whoever
    imports this module: placing a file under a package does not execute it, so
    without that import the registration never runs and a config naming this
    environment fails in a fresh process with an unknown-environment error.
    """
    if ENVIRONMENT_ID not in gym.registry:
        gym.register(id=ENVIRONMENT_ID, entry_point="hercule.environnements.oracle:AsymmetricOracleEnv")


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "ENVIRONMENT_ID",
    "EPISODE_LENGTH",
    "OPTIMAL_RETURN",
    "SUCCESS_BAR",
    "TARGET_HIGH",
    "TARGET_LOW",
    "AsymmetricOracleEnv",
    "register",
]
