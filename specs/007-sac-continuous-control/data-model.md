# Phase 1 — Data model: classes, hooks, hyperparameters, checkpoint

## Class hierarchy

```text
RLModel                                   [ABC, Pydantic]   unchanged
└── OffPolicyReplayModel                  [ABC]             NEW — models/off_policy/
    ├── DeepQLearningModel                                  REBUILT
    └── ContinuousActorCriticModel        [ABC]             NEW — models/continuous_actor_critic/
        └── SACModel                                        NEW — models/sac/
```

`TDModel` and its subclasses are untouched and remain a sibling branch under `RLModel`.

Neither new abstract class is registered by model discovery: `_is_registrable` already requires a
class to be non-abstract and to declare `model_name` in its own `__dict__`, which feature 006 put in
place precisely so intermediate abstractions could exist without appearing as phantom entries.

## `OffPolicyReplayModel` — what it owns

State, all `PrivateAttr`, moved from `DeepQLearningModel` unchanged:

| Attribute | Role |
|---|---|
| `_replay_buffer` | transitions; action widened from `int` to `int \| np.ndarray` |
| `_frames`, `_predict_frames` | the two independent frame histories, one per episode driver |
| `_obs_offset`, `_obs_scale`, and their tensor operands | affine rescaling derived from the observation space |
| `_device`, `_rng`, `_needs_seeded_reset` | device, the model-owned NumPy generator, first-reset seeding |
| `_step_count`, `_epoch_count` | counters |
| `_action_space`, `_observation_space` | the spaces |
| `_action_low`, `_action_high` | NEW — per-dimension bounds, cached at configure time **only when the action space is a `Box`**. A `Discrete` space has no `low`/`high`; the rebuilt deep model would fail at `configure()` if they were read unconditionally |

Behaviour it owns: `configure()` (seeding, then network construction, then optimizers, then the
replay store), `run_epoch()` (the episode loop below), `begin_episode()`, frame priming and
stacking, observation rescaling, the action mapping of FR-008, checkpoint assembly, and
`act()`/`predict()` which return environment-coordinate actions per FR-009.

### The episode loop

Transcribed from the roadmap's §2.3, whose two ordering constraints are load-bearing and are both
present in today's source:

```text
run_epoch(train_mode):
    observation = env.reset(seed=... on the first reset of a fresh run only)
    begin_episode()                                    # clears the frame histories...
    obs = prime(frames, to_frame(observation))         # ...so priming MUST follow it
    while not done:
        env_action, stored_action = _select_action(obs, training=train_mode)
        next_raw, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated                 # episode END still ORs both
        frames.append(to_frame(next_raw)); next_obs = stack(frames)
        if train_mode:
            push(obs, stored_action, reward, next_obs, terminated, truncated)
            _on_training_step()                        # no-op by default; DQN decays epsilon here
            step_count += 1
            if _ready_to_update():
                _update(sample_batch())                # gradual averaging lives in here
            _sync_targets()                            # once per ENV step; no-op by default
        obs = next_obs
```

`begin_episode()` before priming, and `push()` inside `if train_mode`, are the two orderings the
roadmap records as having been got wrong on a first draft. Inverting the first wipes the primed
history; hoisting the second writes greedy evaluation transitions into the replay store.

**On `_on_training_step()`, which replaces the roadmap's `_advance_exploration_state()`.** The
roadmap's version defaults to epsilon decay, which SAC does not have and which would raise on its
first training step. This one has a **no-op** default and a name that makes no claim about what a
subclass advances. FR-003 forbids the shared abstraction from *assuming an exploration scheme* and
from *requiring a hyperparameter a subclass does not use*; a no-op default does neither, so the hook
is compliant as written.

An intermediate draft went further and removed the hook entirely, having the deep model decay epsilon
at the end of its own `_select_action`. Review showed that trades one problem for another: `act()`
delegates to `_select_action`, so a caller invoking the **public** `act(obs, training=True)` outside
the episode loop would then mutate epsilon, where today it does not. The golden fixture drives
`run_epoch()` and would never see it. Keeping the hook at the call site the decay already occupies
leaves `act()` non-mutating and the deep model's behaviour untouched.

### Hook surface

| Hook | Default | Notes |
|---|---|---|
| `_build_networks()` | abstract | must construct parameterised modules in a fixed order — see the plan's refactor constraint |
| `_build_optimizers()` | abstract | **constructs** the optimizers; `configure()` calls it immediately after `_build_networks()`. Distinct from `_optimizers()`, which only *enumerates* already-built ones for checkpointing and cannot create anything — without this hook the ancestor cannot own `configure()` at all |
| `_select_action(obs, training) -> (env_action, stored_action)` | abstract | owns **all** exploration, so the ancestor never samples an action itself. A warmup action is already in storage coordinates when it returns |
| `_ready_to_update() -> bool` | `step_count % step_modulo == 0 and len(buffer) >= batch_size` | both conjuncts; the config `dq_car_racing.yaml` sets `step_modulo: 4`, so dropping the first quadruples DQN's update rate. SAC **overrides** it to add `step_count >= learning_starts`, so no gradient step is taken during the warmup |
| `_on_training_step()` | **no-op** | called once per environment step **on the training path only**, at the point the deep model decays epsilon today. Deliberately not named for exploration: the ancestor makes no claim about what a subclass advances here |
| `_target_pairs() -> Iterable[tuple[nn.Module, nn.Module]]` | `()` | which live network each delayed copy shadows. Required input to `_sync_targets()`'s default body, which cannot otherwise know what to copy onto what without a naming convention |
| `_update(batch) -> None` | abstract | the losses. For the continuous family, gradual averaging of the delayed copies happens **here**, because it is per gradient step |
| `_sync_targets()` | hard copy of every declared target pair, on the interval `_target_sync_interval()` returns | called once per environment step. The ancestor therefore *carries* the deep model's schedule as its default, per FR-007's letter |
| `_target_sync_interval() -> int \| None` | `None`, meaning never | DQN overrides it to return `target_update_frequency`. This is the indirection that lets the ancestor own the hard-copy mechanism without naming a hyperparameter only one subclass declares |
| `_networks() -> Mapping[str, nn.Module]` | abstract | every module to checkpoint, delayed copies included |
| `_optimizers() -> Mapping[str, optim.Optimizer]` | abstract | every optimizer's state to checkpoint |
| `_extra_state() -> dict` / `_load_extra_state(d)` | `{}` | **subclass-specific** non-module state only: the deep model's `epsilon`, SAC's log-temperature, **and every value the network's shape was built from** (`frame_stack`, `observation_shape`), because `hercule play` configures with defaults and rebuilds the network from the checkpoint. **Not** the random streams and **not** the counters: the ancestor owns and exports those itself, and a subclass emitting them too would write the same state under two keys |

## Action mapping

Owned by the ancestor because DecQN will need it as much as SAC does. It is **not** the observation
rescaling, which runs the other way.

```text
to_env(u)    = bias + scale * u          bias = (high + low) / 2,  scale = (high - low) / 2
to_policy(a) = (a - bias) / scale
```

`u = -1` maps to `low` and `u = +1` to `high`, per dimension (FR-008, SC-003). On
`CarRacing-v3(continuous=True)` the bounds are `low = [-1, 0, 0]`, `high = [1, 1, 1]`, so the second
and third dimensions have `bias = scale = 0.5` — the case that makes a forgotten map produce a car
that never accelerates while the reward curve still looks plausible.

The replay store holds `u`; `to_env` is applied only where `env.step()` is called and where
`act()`/`predict()` return (FR-006, FR-009).

## `ContinuousActorCriticModel` — what it adds

| Attribute | Role |
|---|---|
| `_actor` | policy network, its own encoder |
| `_critic_1`, `_critic_2` | value estimators, each with its own encoder |
| `_critic_1_target`, `_critic_2_target` | delayed copies, each a deep copy of its live counterpart at construction |
| `_actor_optimizer`, `_critic_1_optimizer`, `_critic_2_optimizer` | one per trained network, plus the temperature's in `SACModel` — **four** in total, matching the count the specification's Measured Baselines section uses. A single optimizer spanning both estimators would be numerically equivalent for Adam but would change the checkpoint layout the specification describes |

**No target actor** (FR-019). Gradual averaging, coefficient `tau`, applied inside `_update()` once
per gradient step (FR-007).

The delayed copies have `requires_grad = False` on every parameter. That is not an optimisation: it
is one half of the mechanism by which the learning target cannot leak gradient (FR-015); the other
half is computing the target under `torch.no_grad()`.

## `SACModel`

### Class declarations

```python
model_name: ClassVar[str] = "sac"
hyperparams_class: ClassVar[type[HyperParamsBase]] = SACHyperParams
supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]] = frozenset({(SpaceKind.BOX, SpaceKind.BOX)})
```

All three are required, and none is optional or inferred. `_is_registrable` refuses to register a
concrete model that does not declare `model_name` **in its own `__dict__`** and does not have
`supported_spaces`; `supports_environment` compares `classify_space()` results against that
frozenset, so the members must be `SpaceKind` values and not strings — a set of string pairs
type-checks, registers, and then rejects every continuous environment as unsupported.

`SACModel` also owns `_log_alpha` (a `torch.Tensor` with `requires_grad=True`, not a module) and its
own `_temperature_optimizer`. Both are SAC-specific and neither belongs on
`ContinuousActorCriticModel`, which is the shared family-B abstraction and has no temperature: a
deterministic-actor sibling added later would inherit one it never uses.

### Networks

```text
GaussianTanhActor:   Encoder(obs) → Linear(h, 2*d)      → (mean, log_std), log_std clamped to [-20, 2]
ContinuousCritic:    Encoder(obs) → cat(feature, u) → Linear(f+d, 256) → ReLU → Linear(256, 1)
```

`d` is the action-space dimension. The critic takes `u` in normalised coordinates, matching what the
replay store holds (FR-011, FR-012).

### Hyperparameters — `SACHyperParams(HyperParamsBase)`

| Field | Default | Note |
|---|---|---|
| `learning_rate` | `3e-4` | one rate for actor, both estimators and the temperature (research R3) |
| `discount_factor` | `0.99` | |
| `tau` | `0.005` | the gradual-averaging fraction; strictly between 0 and 1 |
| `batch_size` | `256` | |
| `replay_buffer_size` | `100000` | |
| `step_modulo` | `1` | environment steps between gradient steps |
| `learning_starts` | `1000` | length of the uniform-random warmup (FR-018) |
| `init_temperature` | `1.0` | the initial temperature; the optimised quantity is its logarithm |
| `frame_stack` | `0` | the deep model's default and the deep model's **semantics**: the number of *previous* observations concatenated to the current one, so `0` means the current frame alone and `3` means four frames. Gymnasium's own `FrameStackObservation` and Stable-Baselines3 count the total instead; this field deliberately does not |
| `weight_decay` | `0.0` | |
| `seed` | `42` | |

**Not** a hyperparameter: the target entropy. FR-017 fixes it at `-d`, derived from the action space
at configure time. Exposing it would let a grid sweep contradict a requirement.

`supported_spaces = {(BOX, BOX)}` — vector or image observations, continuous actions only (FR-020).
`DISCRETE` observations are excluded: a tabular observation with a continuous action is not a
combination any target environment presents, and admitting it would mean specifying an embedding
this feature does not need.

### The four learned quantities

Stated as a table because the spec's closure argument turns on this set being exhaustive, and
because each row's last two columns are what a criterion asserts.

| Quantity | Objective | Gradient must not reach |
|---|---|---|
| `_critic_1`, `_critic_2` | regression toward the FR-015 target, at the **stored** observation and action | — (the target is a constant) |
| `_actor` | lesser of the two **live** estimators at a differentiable sample, minus temperature × log-density | the estimators' parameters, the temperature |
| temperature (as `log_alpha`) | gradient `−(log-density + target entropy)`, log-density held constant | the actor's parameters |
| `_critic_*_target` | not trained — gradual averaging only | anything (no gradient path exists) |

## Checkpoint

`model.json` keeps every top-level key it has today and the same base64 encoding. **Two** payload
changes sit underneath, and contract C4 owns both:

1. `optimizer_state_b64` becomes a name-keyed mapping, since a model can now hold four optimizers
   rather than one. The format version bumps to **3** to announce it, per the rule the code already
   states for itself.
2. Every parameter key inside `networks_b64` is renamed, because the encoder/head split changes the
   attribute paths a `state_dict()` is keyed by.

Both older forms keep loading, and the pre-006 legacy form keeps its own dispatch branch — but *not*
untouched: its parameter keys go through the same migration table as a version-2 file, because they
are the old attribute paths too. A legacy branch that loaded its keys verbatim would fail against the
rebuilt module, which is the opposite of the compatibility it exists to provide. Contract C4 states this as an obligation
because a deep-model checkpoint already sitting in `outputs/` must load after this feature, and the
golden fixture cannot show that: a fixture is written by fresh training and never exercises loading.

| Key | Contents for SAC |
|---|---|
| `format_version` | **3** — `optimizer_state_b64` becomes a name-keyed mapping, which is a payload-shape change, and the code's own rule bumps the version for exactly that. `_import` dispatches: 3 reads the mapping, 2 the bare single state, pre-006 the legacy form |
| `networks_b64` | one entry per module `_networks()` names: actor, both estimators, both delayed copies |
| `optimizer_state_b64` | one entry per optimizer `_optimizers()` names: actor, estimator 1, estimator 2, temperature |
| `rng_state_b64` | torch's RNG tensor, Python `random.getstate()`, the owned NumPy generator's `bit_generator.state` dict — the same three streams, encoded the same way, **written by the ancestor**, not by `_extra_state()` |
| `log_alpha` | the learned log-temperature |
| `epoch_count`, `step_count` | counters |
| `frame_stack`, `observation_shape` | the values the network's shape was built from, carried for the same reason the deep model already carries them |

`weights_only=True` on load is non-negotiable — feature 006 established that `weights_only=False`
makes `hercule play` on a shared model an arbitrary-code-execution path, and that numpy's **legacy**
RNG tuple cannot round-trip under it while the `Generator`'s state dict can.

Replay contents are not persisted (FR-028). Size: ~129 MB on the image-observation environment, bar
at 150 MB (SC-008); the shipped config sets its checkpoint interval accordingly.
