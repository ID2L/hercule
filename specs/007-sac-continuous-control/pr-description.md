# SAC continuous actor-critic on a shared off-policy ancestor

## Summary

This feature gives Hercule its first algorithm for `Box` (continuous) action spaces: Soft
Actor-Critic (SAC, arXiv:1812.05905, the automatic-temperature version — see `src/hercule/models/sac/__init__.py`
for why the fixed-temperature 1801.01290 variant was rejected for a multi-environment benchmark).
Getting there required extracting the scaffolding every replay-based algorithm needs — the episode
loop, the replay buffer, frame stacking, observation rescaling, device/seeding, the action mapping
between a policy's own coordinates and the environment's, and checkpoint assembly — into a new
abstract ancestor, `OffPolicyReplayModel`, and rebuilding the existing `DeepQLearningModel` onto it.
A second abstract layer, `ContinuousActorCriticModel`, factors what any actor/twin-critic algorithm
in this family needs (delayed critic copies, gradual `tau`-averaging, no delayed actor); SAC is its
first and only concrete instance today.

Both the refactor and the new algorithm are pinned by direct tests rather than by "it trains and the
curve goes up" — the spec's own reasoning for why (`specs/007-sac-continuous-control/spec.md`) is
uncomfortable rather than academic: a wrong RL implementation still trains, still improves, and
still produces a plausible reward curve. Bootstrapping from live estimators instead of their delayed
copies, taking the greater of two value estimates instead of the lesser, adding the entropy term
instead of subtracting it — none of these announce themselves in a training curve.

## What changed

- **`src/hercule/models/off_policy/__init__.py`** (new): `OffPolicyReplayModel(RLModel)`, abstract.
  Owns the episode loop (`run_epoch`), `ExperienceReplayBuffer` (widened from a scalar `int` action
  to `int | np.ndarray` so a continuous action stacks correctly), `Encoder` (MLP or CNN feature
  extractor, shared by every replay-based model), per-dimension action mapping (`to_env_action`
  /`to_policy_action`, cached only for a `Box` action space), and checkpoint assembly. Concrete
  models implement six hooks (`_build_networks`, `_build_optimizers`, `_select_action`, `_update`,
  `_networks`, `_optimizers`); several more have defaults (`_ready_to_update`, `_on_training_step`,
  `_sync_targets`/`_target_pairs`/`_target_sync_interval`, `_extra_state`/`_load_extra_state`,
  `_migrate_parameter_keys`). See `AGENTS.md`'s "OffPolicyReplayModel Extension Points" for the full
  surface.
- **`src/hercule/models/continuous_actor_critic/__init__.py`** (new):
  `ContinuousActorCriticModel(OffPolicyReplayModel)`, abstract. An actor (subclass-supplied), two
  value estimators and two delayed copies of those estimators only — deliberately no delayed actor —
  gradually averaged by `tau` once per gradient step.
- **`src/hercule/models/sac/__init__.py`** (new): `SACModel`, the concrete algorithm. A squashed
  Gaussian policy (`GaussianTanhActor`) with the log-std clamped as a module constant, a numerically
  stable squashing correction (`2*(log 2 - u - softplus(-2u))`, avoiding the `log(1-tanh^2)` underflow
  the naive form hits for `|u| >~ 9` in float32), twin critics taking the action as an input, a
  learned temperature optimised as `log_alpha` (so it cannot cross zero and invert the entropy term),
  and `target_entropy = -d` fixed at configure time rather than exposed as a sweepable hyperparameter.
  Registered as `model_name = "sac"`, `supported_spaces = {(BOX, BOX)}`.
- **`src/hercule/models/deep_q_learning/__init__.py`** (rebuilt, not rewritten): `DeepQLearningModel`
  now sits on `OffPolicyReplayModel`; `QNetwork` becomes `Encoder` + `head`. This is a quality choice,
  not a requirement of adding SAC — see the Constitution Impact section below — done to avoid
  duplicating the ancestor's scaffolding in a second sub-package.
- **`src/hercule/environnements/oracle.py`** (new): `AsymmetricOracleEnv`, a deterministic
  two-dimensional `Box` environment with a closed-form optimum (50.0 over 50 always-truncated steps).
  It exists because `Pendulum-v1` — the primary validation environment — is one-dimensional and
  symmetric, so an entire class of defect is invisible on it: a policy that learns only one action
  dimension, or one whose normalised output is forwarded to the environment unscaled, both score
  exactly as well there as a correct policy. The oracle's action bounds are deliberately asymmetric
  (`low=[-1,0]`, `high=[1,4]`) with the target drawn from the upper half of the wide dimension's
  range, so a mis-mapped policy provably cannot reach the optimum, and `tests/environnements/test_asymmetric_oracle.py`
  asserts the closed-form rows (optimum 50.0; every degenerate policy — best constant on either
  dimension, best fully constant, unscaled-normalised-output — below 45.0) directly.
- Checkpoint format bumped to **version 3**: `optimizer_state_b64` becomes a name-keyed mapping
  rather than one bare state, because a model may now hold several optimizers (SAC holds four: actor,
  two critics, temperature) where the deep model held one. `_import()` still reads every earlier
  format — version 2, and the pre-006 legacy `q_network_state_dict` form — so no checkpoint already on
  disk breaks.
- **Parameter-key migration** (`DeepQLearningModel._migrate_parameter_keys`, `_VECTOR_KEY_MIGRATION`
  /`_IMAGE_KEY_MIGRATION`): a `state_dict`'s keys are attribute paths, so splitting `QNetwork` into an
  `encoder`/`head` layout renamed every key in every `model.json` already on disk. The image branch
  additionally carried every convolutional/dense parameter *twice* pre-refactor (`network.*` was an
  alias of `conv_layers.*`/`fc_layers.*`, from `self.network = nn.Sequential(self.conv_layers, self.fc_layers)`)
  — 20 state-dict entries for 10 tensors — and the migration selects the branch first (`conv_layers.`
  present vs. not) before applying either table, then drops the image branch's aliases rather than
  mapping them, since no single flat rule is correct for both branches.

## The refactor is bit-identical — how that is evidenced

Two independent artifacts, captured from the pre-refactor code on `main` *before any source file was
touched* (`tests/fixtures/capture_baselines.py`, run once and committed alongside the refactor):

1. **`tests/fixtures/golden/dqn_baseline.json`**: per-episode reward series plus one SHA-256 per
   parameter tensor, hashed in `parameters()` enumeration order rather than keyed by parameter name —
   a name-keyed fixture would have broken under the encoder/head rename for a reason unrelated to the
   actual numbers. `tests/models/test_golden_fixture.py` re-trains the same three configurations
   (CartPole's MLP branch, FrozenLake's MLP branch plus its cardinality rescaling, a shaped 3-D `Box`
   environment's CNN branch) at `seed=42` and asserts an exact match, CPU-pinned since bit-identity is
   not achievable across devices.
2. **Two real, loadable pre-refactor checkpoints** (`tests/fixtures/checkpoints/`, one per network
   branch) plus a hand-built pre-006-format file (`legacy_pre006.json`, which no code in the current
   tree can still produce). These are *not* deleted at feature closure, unlike the golden fixture:
   `tests/models/test_checkpoint_compat.py` loads each one and asserts it loads without error and that
   the restored weights, optimizer state (by parameter *identity*, not just by name or shape — Adam's
   moments in a version-2 checkpoint are keyed by integer index into `parameters()`, so an
   order-preserving-but-renamed migration would silently attach the wrong moments to the wrong
   tensor), and all three RNG streams come back exactly as written. This is the test the golden
   fixture structurally cannot be, since the fixture never opens a file — it re-trains fresh and would
   stay green through a rename that broke every checkpoint already on disk.

Per the spec, the golden fixture and its `tests/fixtures/golden/` directory are polish-phase items
staged for deletion in favour of a determinism property test (same seed twice identical, different
seeds different) that stores no baked-in expectation and survives later deliberate behaviour changes;
`tests/fixtures/checkpoints/` stays, since the compatibility path it exercises outlives the fixture
that helped build it.

## What remains outstanding

Two long-running recorded experiments from Phase 6 (image-observation environments) have not been
run as part of this PR:

- **SC-007 / `experiments/sac_car_racing.yaml`** — SAC on `CarRacing-v3(continuous=True)` at epoch
  700 (matching the existing `dq_car_racing.yaml` budget for direct comparison), required to beat the
  measured random-policy baseline of -34.37 by at least 100 points (i.e. reach at least +65.63).
- **SC-006 / `experiments/sac_lunarlander.yaml`** — SAC on `LunarLanderContinuous`, required to reach
  the environment's conventional "solved" bar of 200 over 100 evaluation episodes within a 3000-episode
  budget.

Both are long enough that they are tracked as separate follow-up runs rather than blocking this PR;
if either misses its bar, the spec's own instruction is to record the finding rather than lower it.

## Constitution Impact

**MINOR amendment, 1.2.0 → 1.3.0** (already applied in `.specify/memory/constitution.md`, commit
`163053c`, ahead of the source changes as the spec's Phase 1 requires).

Two entries were added to the Root Class Registry: `OffPolicyReplayModel`
(`src/hercule/models/off_policy/__init__.py`) and `ContinuousActorCriticModel`
(`src/hercule/models/continuous_actor_critic/__init__.py`). Both occupy exactly the position
`TDModel` already occupies in the registry — an abstract intermediate whose lifecycle contract
concrete models depend on — which is why their addition is MINOR rather than MAJOR: it is additive,
not a redefinition of an existing entry.

One clarifying sentence was added to **Principle VI** (Extensibility & Discoverability), whose text
reads "Adding a new RL algorithm MUST NOT require modifying any existing file outside the new
algorithm's sub-package." This feature rebuilds `DeepQLearningModel` onto the new ancestor, which
touches a file outside SAC's own sub-package — but that rebuild is a quality choice, not something
adding SAC *requires*: SAC could have duplicated the off-policy scaffolding standalone in its own
sub-package and satisfied the principle as originally worded. The amendment states explicitly that
extracting shared scaffolding into an intermediate abstract class is not a violation, since the
principle's actual test is whether the *next* algorithm can be added inside its own sub-package —
which holds more strongly after this rebuild than before it, since the next off-policy or
continuous-actor-critic algorithm now inherits the scaffolding entirely.

`RLModel`'s own abstract surface is untouched by this feature: no abstract method was added, removed
or renamed on it, and `save()`/`load()`/`check_environment_or_raise()` remain `@final`. The two new
classes are inserted *beneath* `RLModel`, which Principle I already permits for intermediate abstract
classes; neither is itself registered by `get_available_models()`, since model discovery registers
concrete classes only.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
