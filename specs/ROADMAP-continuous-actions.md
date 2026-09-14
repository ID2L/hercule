# Roadmap — Continuous action spaces in Hercule

Revision after round 2 of adversarial review (3 of 4 reviewers returned NOT CONVERGED). Every new
claim was re-verified against the source or tested empirically before adoption; two were tested by
running code, and the results are quoted.

## 0. What round 2 changed

| Round-2 finding | Verified how | v3 response |
|---|---|---|
| v2's hook table **dropped** `_networks()`, leaving no way to enumerate modules for checkpointing | v2 §2.1 read back — the hook is genuinely absent | Restored (§2.1). This was a regression I introduced. |
| Target networks are rebuilt from the online net on import, destroying the lag | `deep_q_learning/__init__.py:654-657` — `_target_network.load_state_dict(state_dict)` using the **online** dict | Target weights serialised in their own right (0.5) |
| `_ready_to_update()` default omitted `step_modulo` | `deep_q_learning/__init__.py:518-523` is a conjunction of `step_count % step_modulo == 0` **and** `len(buffer) >= batch_size`; `experiments/dq_car_racing.yaml:112` sets `step_modulo: 4` | Default corrected (§2.1) |
| A `supported_spaces` check behind `configure()` can never propagate | `deep_q_learning/__init__.py:257` calls `super().configure(...)` and **discards** the return | Check moved into `Supervisor`, against the ClassVar directly (0.2) |
| `weights_only=True` is incompatible with numpy RNG state | **Tested.** `torch.save`/`torch.load(weights_only=True)` round-trip: torch RNG state OK, `random.getstate()` OK, `np.random.get_state()` **UnpicklingError**, `default_rng().bit_generator.state` OK | 0.5 stores the Generator's `bit_generator.state` dict, never the legacy tuple |
| The ancestor's loop *call sequence* was never written down — only the hooks | v2 §2.1 read back | Written out as pseudocode (§2.3) |
| `act()`/`predict()` unbound to the two-value `_select_action` | `models/__init__.py:158-211`, `evaluate()` at `:309` and `play_interactive` both use `predict` | `act`/`predict` return the **env** action (§2.3) |
| v2 §2.2's "-3 on Pendulum" example is arithmetically incoherent | Pendulum's `low` **is** -2; "maps u=-1 to low" is the *correct* behaviour, so the sentence contradicted itself | Example replaced with the real failure (§2.2) |
| `bins_per_dimension` undefined at `k = 1` | `(k-1)` divisor | `Field(ge=2)` (Phase 2) |
| `RLModel.load()` returns silently when `model.json` is absent | `models/__init__.py:256-258` | Addressed in `Runner`, not `RLModel` — see 0.5 |
| 133.9 MB and 140.9 MB both presented as "verified" without reconciliation | Two different artifacts | Reconciled below |

**Size reconciliation:** 133.9 MB is the real `model.json` on disk from
`outputs/dq_car_racing/.../fra_sta_3/`, whose network has 5 actions. 140.9 MB is the benchmark
file written from a freshly constructed `QNetwork((96, 96, 12), 5)`. The gap is JSON float
repr length on trained vs freshly-initialised weights. Both are pre-base64; the base64 figure
(11.7 MB) is measured against the 140.9 MB one, so the 12.0x ratio is the conservative reading.

## 1. Confirmed pre-existing defects (unchanged from v2, all re-verified)

1. **Truncation treated as termination.** `done = terminated or truncated` (line 488) pushed as one
   flag (line 499); `target = rewards + gamma * next_q * ~dones` (line 578) suppresses the bootstrap
   for both. `Pendulum-v1` *always* truncates at 200 steps and never terminates; `CarRacing-v3`
   truncates at `max_episode_steps`. Every time-limit transition gets a target of `r` instead of
   `r + gamma * V(s')`.
2. **`_export()` omits mutated state.** `run_epoch` mutates `typed_params.epsilon` every training
   step (line 507); `_export()` (lines 609-619) does not write it. A run resumed at epoch 5000 keeps
   its step count and restarts exploration at `1.0`. Optimizer state and target-network weights are
   absent for the same reason.
3. **`seed` is dead.** Declared at line 57, never read; `torch.manual_seed(42)` hardcoded at line
   235; `random` (epsilon-greedy at line 449, replay sampling at line 189) never seeded;
   `env.reset()` unseeded (line 474).
4. **Registry admits abstract classes.** Verified by running it: `tdmodel -> TDModel abstract=True`.
5. **`check_space_is_box` returns `not is_discrete`** and has **zero callers** (`grep` over `src/`
   and `tests/`), so correcting it is free.

## 2. Class hierarchy

```
RLModel                                    [abstract surface unchanged; see 0.2 for the one ClassVar]
└── OffPolicyReplayModel           [ABC]   frame stacking, observation rescaling, generic replay,
    │                                      device+seeding, action-space mapping, episode loop,
    │                                      checkpoint assembly
    ├── DeepQLearningModel                 epsilon-greedy, single Q head, hard target sync
    ├── DecQNModel                         d x k heads, value decomposition, bang-bang
    └── ContinuousActorCriticModel [ABC]   actor + twin critics + TARGET CRITICS ONLY + Polyak
        ├── TD3Model                       adds its own target actor; deterministic policy
        └── SACModel                       no target actor; stochastic policy, learned temperature
```

SAC samples `a' ~ pi(.|s')` from the **current** actor and uses target critics only; a target actor
is TD3-specific (it exists to be smoothed). `QNetwork` maps `s -> R^|A|` (`:67`) and cannot serve as
a continuous critic `Q(s,a)`; what is shared is the **trunk** (MLP for 1-D, CNN for 3-D), so the
shared building block is an `Encoder` plus per-algorithm heads.

### 2.1 Hook surface

| Hook | Default | Rationale |
|---|---|---|
| `_build_networks()` | abstract | topology differs |
| `_select_action(obs, training) -> (env_action, stored_action)` | abstract | DecQN sends floats to `env.step()` but stores bin indices; TD3/SAC store the normalised action. **Owns all exploration** — epsilon-greedy, warmup, Gaussian noise — so the ancestor never samples actions itself and a warmup action is already converted to storage coordinates before it returns. |
| `_ready_to_update() -> bool` | `step_count % step_modulo == 0 and len(buffer) >= batch_size` | v2 omitted the first conjunct. `experiments/dq_car_racing.yaml:112` uses `step_modulo: 4`, so the omission would have quadrupled DQN's update rate. |
| `_advance_exploration_state()` | epsilon decay | TD3/SAC hyperparameters have no `epsilon` field; the current loop would raise `AttributeError` on their first training step |
| `_update(batch) -> None` | abstract | the loss. **For TD3/SAC, Polyak averaging happens here**, because it is per *gradient* step, which differs from per *env* step whenever `step_modulo > 1`. |
| `_sync_targets()` | hard copy every `target_update_frequency` **env** steps | DQN's schedule, which is deliberately not chained to `_train_step` (source comment at `:527-529`). No-op for TD3/SAC. |
| `_networks() -> Mapping[str, nn.Module]` | abstract | **Restored.** Enumerates every module to checkpoint, target copies included. |
| `_extra_state() -> dict` / `_load_extra_state(d)` | `{}` | non-module state: `epsilon`, `log_alpha`, optimizer state dicts, RNG state |

`ExperienceReplayBuffer.push` widens `action` from `int` to `int | np.ndarray`; collation stacks
rather than assuming scalars.

### 2.2 Action-space mapping lives in the ancestor

DecQN (Phase 2) needs it as much as TD3/SAC (Phase 3), so it cannot sit in
`ContinuousActorCriticModel`. It is **not** the observation rescaling, which runs the other way
(`[low, high] -> [0, 1]`, `:294-320`). Two maps:

```
bins  (DecQN):    a_i = low_i + j * (high_i - low_i) / (k - 1),      j in 0..k-1,  k >= 2
tanh  (TD3/SAC):  a_i = (high_i + low_i)/2 + u_i * (high_i - low_i)/2,   u in [-1, 1]
```

Both send the lower end to `low_i` and the upper end to `high_i`.

**The failure this prevents** (v2's illustration of it was arithmetically incoherent and is
withdrawn): the bug is *omitting* the map, not misapplying it. A tanh actor emitting `u = [-0.3,
-0.5, -0.2]` on `CarRacing-v3` and stepping the environment with `u` directly applies gas `-0.5`
and brake `-0.2`, both outside `Box(low=[-1,0,0], high=1.0)`. Gymnasium does not reject it; the car
simply never accelerates, and the run still produces a plausible-looking reward curve.

### 2.3 The ancestor's episode loop — call sequence, written down

v2 listed hooks without saying who calls them or in what order, which is what let round 2 argue
Phase 1 could still freeze DQN's shape. The contract:

```
run_epoch(train_mode):
    observation = env.reset(seed=... on first reset of a fresh run only)
    begin_episode()                                   # clears the frame history...
    obs = prime(frames, to_frame(observation))        # ...so priming MUST follow it
    while not done:
        env_action, stored_action = _select_action(obs, training=train_mode)
        next_obs_raw, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated                # episode END still ORs both
        frames.append(to_frame(next_obs_raw)); next_obs = stack(frames)
        if train_mode:
            push(obs, stored_action, reward, next_obs, terminated)  # only `terminated` kills the bootstrap
            _advance_exploration_state()
            step_count += 1
            if _ready_to_update():
                _update(sample_batch())               # Polyak lives in here for TD3/SAC
            _sync_targets()                           # hard copy for DQN, no-op otherwise
        obs = next_obs
```

Two ordering constraints in there are load-bearing, and v3's first draft got both wrong:

- **`begin_episode()` precedes priming.** It *clears* `_frames` and `_predict_frames`
  (`:362-365`), so priming first and calling it second wipes the primed history and every
  subsequent stacked `next_obs` is short. The source has this order (`:482-483`); the pseudocode
  must not invert it.
- **`push()` is inside `if train_mode`.** The source stores only on the training path
  (`:496-499`). Hoisting it out writes greedy evaluation transitions into the replay buffer, which
  the next training epoch then samples — and Phase 1's golden fixture is a *training* run, so it
  would not catch it.

`act(observation, training)` and `predict(observation)` return **`env_action`** — the first element.
`RLModel.evaluate()` (`models/__init__.py:309`) and `controller.play_interactive()` both drive their
own loops through `predict`, so returning the stored representation would make `hercule play` send
bin indices or raw `u` to CarRacing — the exact failure §2.2 exists to prevent, reintroduced on the
evaluation path.

## 3. Roadmap

Each phase is one spec: implementation + tests + one `experiments/*.yaml`, leaving `uv run pytest`
and `uv run ruff check .` clean.

### Phase 0 — Foundations

Ordering matters: 0.3 and 0.4 change DQN's behaviour, so Phase 1's regression baseline is captured
**after** them. 0.1 and 0.2 are independent of the other three and can run in parallel.

**0.1 — Concrete-only model discovery.** Register only if `not inspect.isabstract(cls)` **and**
`"model_name" in cls.__dict__`; drop the `__name__.lower()` fallback; warn on a concrete model
missing its own `model_name`. Also require `supported_spaces` (0.2) so it cannot be forgotten
silently. *Acceptance:* registry returns exactly the four concrete models; a test asserts no
abstract class is ever returned.

**0.2 — Enumerable space taxonomy.** `SpaceKind` StrEnum over all 10 Gymnasium 1.2.3 space classes,
`classify_space()`, and `supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]]` on
`RLModel`, declared explicitly by all four existing models — **no permissive default**, which would
recreate the silent-accept it removes.

*Enforcement point, corrected:* the comparison runs in **`Supervisor`**, against the model class's
ClassVar, before `configure()` is called. Round 2 showed a check behind `configure()` cannot work:
`DeepQLearningModel.configure` calls `super().configure(...)` and **discards its return**
(`:257`), so a base-class rejection never reaches the caller. Checking in `Supervisor` needs no
propagation from any override and no signature change.

*Constitution Impact: MINOR bump.* Adding a `ClassVar` subclasses depend on is a semantic change per
AGENTS.md. The abstract-method surface is untouched. (v2 claimed "RLModel unchanged" in one section
while adding this in another — that contradiction is resolved in favour of declaring the impact.)

`check_space_is_box` corrected to a real `isinstance(space, gym.spaces.Box)`; zero callers, zero
risk. *Acceptance:* parametrised classification test over the five target environments; a test that
`simple_q_learning` + `CartPole-v1` is skipped with a message naming both kinds while the other
combinations in the same config still run.

**0.3 — Terminated vs truncated.** Transitions carry both flags; the bootstrap is suppressed on
`terminated` only. *Acceptance:* stub environment that truncates without terminating asserts the
target equals `r + gamma * Q(s')`; one that terminates asserts `r`. Plus a measured before/after on
`CartPole-v1`. *Consequence stated in the spec:* existing `outputs/` results become non-comparable
with post-fix runs — the correct trade, since the old numbers were computed against a wrong target.

**0.4 — Make `seed` live.** `configure()` seeds torch and Python `random` from the `seed`
hyperparameter, **before network construction**; `run_epoch` seeds `env.reset(seed=...)` on the
first reset of a **fresh** run only. `torch.manual_seed(42)` in `__init__` goes away.

*NumPy gets one owned `Generator`, not the global functions.* The model holds a single
`np.random.default_rng(seed)` and **all** NumPy randomness in the model routes through it. This is
not a style preference: 0.5 checkpoints `Generator.bit_generator.state`, and that is only the state
that was actually consumed if nothing calls `np.random.*` behind its back. Seeding the global
legacy RNG and checkpointing an unrelated `Generator` would produce a resume that looks correct and
is not. (`TDModel` already holds such a generator, `td_models/__init__.py:67`; DQN holds none.)

*Interaction with 0.5, stated explicitly:* on **resume**, the restored RNG state from the checkpoint
wins; `env.reset(seed=original)` is **not** re-issued. Otherwise 0.5's RNG round-trip is dead on
arrival on the path 0.4 writes. *Acceptance:* same seed twice gives bit-identical weights; different
seeds do not. Today the second half passes for the wrong reason and the first half fails.

**0.5 — Complete, compact checkpoints.**
- Weights as base64 of a `torch.save` buffer. Measured: 140.9 MB / 3.95 s -> 11.7 MB / 0.23 s
  (12.0x smaller, 17x faster to write; 1.59 s -> 0.31 s to read). Legacy
  `q_network_state_dict` list format stays readable so the models already in `outputs/` keep loading.
- **Every module from `_networks()` is serialised, target copies included.** Today `_import`
  overwrites the target with the *online* weights (`:654-657`), destroying the lag that is the
  target network's entire purpose — so a resumed run restarts with zero lag.
- `_extra_state()` carries optimizer state dicts, mutated hyperparameters (`epsilon`), and RNG state.
- **RNG state is stored as torch's RNG tensor, Python's `random.getstate()`, and numpy's
  `Generator.bit_generator.state` dict.** Tested: under `torch.load(..., weights_only=True)` the
  first three round-trip fine and `np.random.get_state()` (the legacy tuple, which contains an
  ndarray) raises `UnpicklingError`. `weights_only=True` is non-negotiable — `weights_only=False`
  would make `hercule play` on a shared model an arbitrary-code-execution path.
- **Replay buffer contents are explicitly out of scope**, with the consequence stated: a resumed
  off-policy run restarts with an empty buffer and re-fills it, a real discontinuity in the learning
  curve. Storing it means gigabytes per checkpoint.
- **Missing-checkpoint detection lives in `Runner`, not `RLModel`.** `RLModel.load()` returns
  silently when `model.json` is absent (`models/__init__.py:256-258`), and that tolerance is what
  makes a *fresh* run work, so it must stay. Instead `Runner` — which already reads `run_info.json`
  to restore epoch counters — raises when `run_info.json` reports a non-zero epoch while
  `model.json` is missing. That is a corrupt run directory, and it is currently silent. No
  Constitution impact: `Runner` is a registry class but this is additive validation, not an API
  change.
- **0.5 blocks Phase 1**, since the Phase 1 ancestor owns checkpoint assembly.
- *Acceptance:* round-trip (export -> import -> bit-identical weights, target weights, optimizer
  state, RNG state); legacy fixture still loads; resume test asserting `epsilon` continues rather
  than resetting and that the target network keeps its lag; measured CarRacing `model.json` under
  15 MB.

### Phase 1 — `OffPolicyReplayModel`, DQN refactored onto it

Extract per §2, with the hook surface of §2.1 **and the call sequence of §2.3** fixed up front.

*Regression method:* a **golden fixture** (episode rewards + weight hashes) generated from the code
as of 0.4's completion and committed; the refactor must reproduce it bit-identically. Two versions
cannot coexist in one process, so "run before and after" was never implementable. This is only
meaningful because 0.4 made seeding real.

*Fixture retirement:* at Phase 1 closure the fixture is replaced by a durable determinism property
test (same seed twice -> identical; different seeds -> different), which stores no expectation and
therefore survives later behaviour changes without regeneration. The fixture never outlives the
refactor it certifies.

Note "move-only diff" is **not** an acceptance criterion — hook dispatch *is* the extract. The
fixture is the criterion.

### Phase 2 — DecQN (Family A)

Seyde et al., ICLR 2023, `arXiv:2210.12566`. One head per action dimension, `k` bins each; joint Q
as the mean of per-dimension utilities, so outputs grow as `d x k`, not `k^d`.

Both halves of the decomposition: online value `mean_i Q_i(s, a_i)` gathered per dimension; target
`r + gamma * (1 - terminated) * mean_i max_a Q_i(s', a)`. Bin-to-action mapping per §2.2.
Hyperparameter `bins_per_dimension: int = Field(default=3, ge=2)` — the `(k-1)` divisor is
undefined at `k = 1`.

### Phase 3 — TD3 (Family B, deterministic actor) — the first actor network

Fujimoto et al., ICML 2018, `arXiv:1802.09477`. `ContinuousActorCriticModel` provides actor, twin
critics, **target critics**, Polyak (`tau`); the critic takes the action as an input concatenated
after the trunk. TD3 adds its own target actor.

Target policy smoothing is **two** operations: clipped Gaussian noise on the target action, then a
**clamp to the legal action bounds**. Near CarRacing's gas/brake floor at `0`, correctly-clipped
noise still produces an illegal target action.

Exploration: Gaussian noise plus a `learning_starts` uniform-random warmup, both inside
`_select_action` (§2.1), never through epsilon fields TD3 does not have.

**Trunk decision, made now and applied to both TD3 and SAC: separate trunks.** Actor gets its own;
each twin critic gets its own. This matches the reference implementations the Pendulum bars are
calibrated against, and avoids the stop-gradient failure mode of DrQ-v2-style sharing. Critically,
the twin critics must **never** share a trunk: shared features collapse the decorrelation that
clipped double-Q exists to provide. Sharing can be revisited later as a pure optimisation behind
the same hook surface.

DDPG is not a separate model: it is TD3 with `policy_delay=1`, no target noise, one critic.

### Phase 4 — SAC (Family B, stochastic actor)

Haarnoja et al., `arXiv:1812.05905` — the automatic-temperature version, not 1801.01290, whose
fixed `alpha` must compensate each environment's reward scale, the wrong property for a
multi-environment benchmark.

Sibling of TD3: reuses twin critics, target critics, Polyak. **No target actor.** Delta:
`(mean, log_std)` actor, reparameterisation trick, tanh squashing with the `log(1 - tanh^2 + eps)`
correction, entropy term in the critic target, learned `log_alpha` against `target_entropy = -dim(A)`.

**The policy operates in normalised `[-1, 1]` space; rescaling to the environment's bounds happens
only at the `env.step()` boundary.** With `a = bias + scale * tanh(u)` the change of variables
carries an extra `-sum(log(scale))` term; it is constant in the actor's parameters, so it does not
change the policy gradient, but it shifts the entropy that `alpha` adapts against — Pendulum's scale
is `2` (`log 2 ~ 0.69` against a target of `-1`), CarRacing's gas/brake scale is `0.5`. Keeping the
policy normalised makes `target_entropy = -dim(A)` correct as written.

The tanh log-prob correction gets its own unit test against a closed-form value: omit it and
training still runs and still improves, just to a worse policy. `log_alpha` travels in
`_extra_state()`.

### Phase 5 — NAF: dropped

Unanimous across all four reviewers. The settling argument: the roadmap's own selection rule is "one
best-in-class representative of each family", and NAF is best-in-class of nothing. It is also not
"the cheapest possible model" — it needs a `d(d+1)/2` Cholesky head and its own exploration noise
process. Nothing downstream depends on it.

## 4. Acceptance criteria

Four tiers per algorithm phase, cheapest first. "Runs end-to-end" and "a non-flat learning curve"
are withdrawn as unfalsifiable — this roadmap itself establishes that a car applying negative gas
still trains and still produces a curve.

1. **Action-mapping unit test against the real `CarRacing-v3` action space.** Per dimension, bin `0`
   -> `low_i`, bin `k-1` -> `high_i`; for tanh policies `u = -1 -> low_i`, `u = +1 -> high_i`. Also
   asserts `act()`/`predict()` return the env-coordinate action. No training, milliseconds.
2. **A 2-D asymmetric `Box` toy environment** as the multi-dimensional oracle. `Pendulum-v1` cannot
   serve alone: 1-D and symmetric, so DecQN's `mean_i` is a no-op (`d = 1`, degenerating to
   discretised DQN) and no per-dimension asymmetric mapping bug can surface.
3. **`Pendulum-v1` smoke training with a numeric bar**: mean test reward over 20 episodes above
   `-300` for DecQN, `-200` for TD3 and SAC.
4. **`CarRacing-v3` against a measured baseline**: mean test reward exceeding the random-policy
   score by a stated margin at a stated epoch, the baseline measured once and recorded in the spec.

## 5. Adjudicated in round 2 (no longer open)

- **Two-value `_select_action`; the buffer stores `stored_action` only.** Unanimous. `env_action` is
  a deterministic function of `(stored_action, bounds)` that the ancestor already owns, so storing
  both only creates a divergence channel. `stored_action` must be the action actually executed, in
  network coordinates.
- **Golden fixture, then retire it** into a durable determinism property test at Phase 1 closure.
- **All five Phase 0 specs stay before Phase 1.** Unanimous: freezing first and fixing later either
  certifies the truncation bug into the ancestor or forces re-opening it after closure.
- **Separate trunks**, decided once in `ContinuousActorCriticModel` for both algorithms (3 of 4
  reviewers; the dissent argued for critic-only-gradient sharing, which adds a stop-gradient failure
  mode for a parameter-count win).

## 6. Adjudicated in round 3 — nothing open

- **`Runner` raises** on a non-zero recorded epoch with a missing `model.json`. Unanimous, 4 of 4.
  No legitimate workflow deletes `model.json` while keeping `run_info.json`: the result is a
  directory that *looks* resumable and silently restarts from scratch, concatenating two unrelated
  runs into one metric series. A deliberate restart deletes the run directory. The error names the
  missing file and the reported epoch.
- **`_sync_targets()` stays a per-env-step hook** with a no-op default on the actor-critic branch;
  Polyak stays inside `_update()`. 3 of 4 (the dissent proposed folding it into `_update()` to avoid
  a hook one branch never uses). The deciding fact is in the source: DQN's target sync is
  deliberately *not* chained to the training step, with a comment saying why (`:527-529`). Folding
  it into `_update()` would couple DQN's target lag to `step_modulo` — at `step_modulo: 4`, which
  `experiments/dq_car_racing.yaml:112` uses, the lag would silently quadruple. One no-op call per
  env step is the cheaper price.
