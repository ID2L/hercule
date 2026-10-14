"""RaceEnv — gymnasium wrapper for the Crash&Learn Grand Prix simulation.

Copied from the crash-and-learn repository (``4pedagos/test_RL/env.py``) so Hercule can benchmark it as a custom
environment. Changes from the original, behaviour otherwise identical:

- the per-step ``print()`` calls are gone; the reward breakdown is logged at DEBUG level instead;
- the pure-pursuit controller is no longer the default: ``controller="rl"`` applies the agent's action (the
  original module-level ``USE_PURSUIT = True`` silently ignored it), ``controller="pure_pursuit"`` keeps the
  scripted driver for debugging;
- the episode step cap is a constructor argument (``step_limit``) instead of a module constant;
- the ``test_obs_scale()`` debugging helper was dropped.

The simulation backend (``env_simulation.py``) and the vendored ``f110_gym`` engine are copied unchanged, except
for ``env_simulation``'s ``sys.path`` bootstrap, which points at the ``f110_gym`` package next to it.

``env_simulation`` is imported by its bare module name, so this directory must be on ``sys.path`` — Hercule's YAML
``python_path`` does that.
"""

import logging
import math

import gymnasium as gym
import numpy as np
from env_simulation import (
    _DECISION_FREQ_HZ,
    apply_action,
    close,
    get_obs,
    get_space_info,
    get_step_info,
    reset,
    simulation_step,
)


logger = logging.getLogger(__name__)

DEFAULT_STEP_LIMIT = 8000  # decision steps; 8000 / 20Hz = 400s sim time per episode cap

REWARD_WEIGHTS: dict[str, float] = {
    "progress": 100.0,
    "smoothing": 0.1,
    "collision_wall": 1.0,
    "stagnation": 1.0,
    "lap_bonus": 1000.0,
}


class RaceEnv(gym.Env):
    """Single-agent racing environment wrapping env_simulation.

    Observation: [lidar / 15, velocity / 5, steering / 0.42]   shape=(102,)
    Action (raw): [-1, 1] denormalized to target_speed (-2.0 m/s reverse .. 10.0 m/s forward),
                   steering (-0.42 .. 0.42 rad)                     shape=(2,), bounds=[-1, 1]
    """

    metadata = {"render_modes": [], "step_limit": DEFAULT_STEP_LIMIT}

    def __init__(self, controller: str = "rl", step_limit: int = DEFAULT_STEP_LIMIT):
        """Build the environment.

        Args:
            controller: ``"rl"`` to apply the agent's action, ``"pure_pursuit"`` to drive with the scripted
                waypoint follower (the action is then ignored).
            step_limit: Number of decision steps after which the episode is truncated.
        """
        super().__init__()
        if controller not in ("rl", "pure_pursuit"):
            raise ValueError(f"controller must be 'rl' or 'pure_pursuit', got {controller!r}")

        self.space_info = get_space_info()

        # Extract observation bounds from space_info for per-component normalization
        obs_bounds = self.space_info["observations"]
        self._obs_lo = {key: obs_bounds[key]["bounds"][0] for key in ("lidar", "velocity", "steering")}
        self._obs_hi = {key: obs_bounds[key]["bounds"][1] for key in ("lidar", "velocity", "steering")}

        lidar_dim = obs_bounds["lidar"]["shape"][0]
        total_dim = lidar_dim + 1 + 1  # 102 = 100 + 1 + 1

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(total_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        self._speed_bounds = self.space_info["actions"]["target_speed"]["bounds"]
        self._steering_bounds = self.space_info["actions"]["steering"]["bounds"]

        self._controller = controller
        self._step_limit = step_limit
        self._step_count = 0
        self._last_progress = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._pp_lookahead = 1.0  # pure pursuit lookahead distance (m)
        self._pp_speed = 2.0  # pure pursuit target speed (m/s)

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        reset()
        raw = get_obs(0)
        self._last_progress = float(raw["progress"])
        self._step_count = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        return self._get_processed_obs(raw), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._controller == "pure_pursuit":
            speed, steer = self._pure_pursuit_step()
            # Express the scripted action in policy space so the smoothing reward stays comparable
            speed_raw = self._to_policy_space(speed, self._speed_bounds)
            steering_raw = self._to_policy_space(steer, self._steering_bounds)
        else:
            speed_raw = float(action[0])
            steering_raw = float(action[1])
            speed = self._from_policy_space(speed_raw, self._speed_bounds)
            steer = self._from_policy_space(steering_raw, self._steering_bounds)

        apply_action(0, speed, steer)
        simulation_step()

        info = get_step_info()
        raw = get_obs(0)

        # ── Reward ──────────────────────────────────────────────────────────────
        rewards = {
            "progress": 0.0,  # normalized to [0, 1]
            "smoothing": 0.0,  # normalized to [-1, 0]
            "collision_wall": 0.0,  # value {-1, 0}
            "stagnation": 0.0,  # value {-1, 0}
            "lap_bonus": 0.0,  # value {0, 1}
        }
        curr_progress = float(raw["progress"])
        delta = curr_progress - self._last_progress
        # Big enough to detect jump from ~0.99 (end) to ~0.01 (start)
        delta += 1 if delta < -0.5 else 0
        if delta > 0.0:
            _, t_hi = self._speed_bounds  # Suppose that t_hi > |t_lo|
            ratio = 1 / (t_hi / _DECISION_FREQ_HZ)
            rewards["progress"] = delta * ratio
            self._last_progress = curr_progress  # update only if going forward

        # Smoothing penalty — compare raw actions in [-1, 1] policy space
        action_delta = np.array([speed_raw - self._last_action[0], steering_raw - self._last_action[1]])
        rewards["smoothing"] = -float(np.sum(action_delta**2) / 2)
        self._last_action = np.array([speed_raw, steering_raw])

        # Only agent id=0 here
        rewards["collision_wall"] = -1.0 if info["collisions"]["wall"].get(0, False) else 0.0
        is_stagnant = info["stagnation"].get(0, False)
        rewards["stagnation"] = -1.0 if is_stagnant else 0.0
        lap_finished = info["lap_complete"].get(0, False)
        rewards["lap_bonus"] = 1.0 if lap_finished else 0.0

        reward = sum(REWARD_WEIGHTS[key] * rewards[key] for key in REWARD_WEIGHTS)

        # ── Termination condition ────────────────────────────────────────
        self._step_count += 1
        terminated = bool(lap_finished or is_stagnant)
        truncated = self._step_count >= self._step_limit

        logger.debug(
            "step=%d reward=%.2f progress=%.3f terminated=%s truncated=%s rewards=%s",
            self._step_count,
            reward,
            curr_progress,
            terminated,
            truncated,
            rewards,
        )
        return self._get_processed_obs(raw), float(reward), terminated, truncated, info

    def close(self) -> None:
        close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _from_policy_space(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return low + (value + 1.0) * (high - low) / 2.0

    @staticmethod
    def _to_policy_space(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return (value - low) / ((high - low) / 2.0) - 1.0

    def _pure_pursuit_step(self) -> tuple[float, float]:
        from env_simulation import _PARAMS, _get_sim  # noqa: PLC0415 - private internals, debug controller only

        sim_obj = _get_sim()
        wp = sim_obj._waypoints
        # .agents lives on the internal Simulator (_sim), not on the wrapper
        agent = sim_obj._sim.agents[0]
        x, y, yaw = agent.state[0], agent.state[1], agent.state[4]
        idx = int(np.argmin((wp[:, 0] - x) ** 2 + (wp[:, 1] - y) ** 2))
        s_lo, s_hi = _PARAMS["s_min"], _PARAMS["s_max"]
        target_x, target_y = wp[idx, 0], wp[idx, 1]
        for i in range(1, len(wp)):
            wi = (idx + i) % len(wp)
            jx = (idx + i - 1) % len(wp)
            if math.hypot(wp[wi, 0] - wp[jx, 0], wp[wi, 1] - wp[jx, 1]) >= self._pp_lookahead:
                target_x, target_y = wp[wi, 0], wp[wi, 1]
                break
        angle = math.atan2(target_y - y, target_x - x) - yaw
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return self._pp_speed, max(s_lo, min(s_hi, angle))

    def _get_processed_obs(self, raw: dict) -> np.ndarray:
        """Return normalized observation array of shape (102,) dtype=float32.

        Each component is normalized to [-1, 1] using its own lo/hi bounds from space_info:
            normalized = (val - lo) / (hi - lo) * 2 - 1
        """

        def normalize(key: str, value: np.ndarray | float) -> np.ndarray:
            lo, hi = self._obs_lo[key], self._obs_hi[key]
            return np.atleast_1d((np.asarray(value) - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)

        obs = np.concatenate(
            [
                normalize("lidar", raw["lidar"]),
                normalize("velocity", raw["velocity"]),
                normalize("steering", raw["steering"]),
            ]
        )
        return np.clip(obs, -1.0, 1.0)
