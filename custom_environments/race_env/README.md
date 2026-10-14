# RaceEnv — Crash&Learn Grand Prix (custom Gymnasium environment)

Copy of the Crash&Learn `RaceEnv` (single-agent F1Tenth racing) used to exercise Hercule's custom-environment
support. Source: `crash-and-learn` repository, `docs/content-CrashAndLearn-master/resources/`.

| File | Origin | Changes |
| --- | --- | --- |
| `race_env.py` | `4pedagos/test_RL/env.py` | per-step `print()` removed (DEBUG log instead); `controller="rl"` by default (the original `USE_PURSUIT = True` ignored the agent's action); `step_limit` constructor argument; `test_obs_scale()` dropped |
| `env_simulation.py` | `4students/simulation/env_simulation.py` | only the `sys.path` bootstrap, pointing at `f110_gym/` next to it |
| `f110_gym/` | `4students/simulation/gym/f110_gym/` (engine only, MIT, see `LICENSE`) | none |
| `maps/examples/` | `4students/simulation/maps/examples/` | the default track only (other circuits not copied) |

- Observation: `Box(-1, 1, (102,))` (100 LiDAR beams, speed, steering, normalized).
- Action: `Box(-1, 1, (2,))` (target speed, steering). Use `action_discretization` in the Hercule YAML for
  discrete-action models.
- Dependencies: `numba`, `scipy`, `pillow` (`custom-environments` uv dependency group).
- The simulator is a process-wide singleton: two `RaceEnv` instances in one process share the same car.
- The LiDAR noise is not seeded by `reset(seed=...)`, so episodes are not reproducible.

Usage: see `experiments/race_env_dummy.yaml` and `experiments/race_env_dqn.yaml`.
