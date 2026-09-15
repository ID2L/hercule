# Tasks: SAC Continuous Actor-Critic on a Shared Off-Policy Ancestor

**Feature**: `007-sac-continuous-control` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)
**Input**: spec.md (revision 10, converged), plan.md (converged), research.md, data-model.md,
contracts/model-contracts.md, quickstart.md

**Tests are in scope and are not optional here.** The specification's criteria SC-003 and SC-010 to
SC-017 each demand a *direct* test, and say why: the defects they cover leave training running and
the reward curve rising. A task that implements one of those without its test has not delivered it.

## Phase ordering, and why it is not spec order

User stories 1 and 2 are both P1. The phases below run US2 before US1, because US2 verifies the
refactor that US1's code is built on: a SAC bug and a refactor bug are indistinguishable if the
refactor was never certified. The specification's own reasoning for making US2 a P1 — "the condition
under which Story 1's numbers can be trusted at all" — is the same argument.

---

## Phase 1: Setup — the irreversible captures

**Nothing in this phase can be redone once Phase 2 starts.** Both artifacts are evidence *about* the
pre-refactor code, so they must be produced by the pre-refactor code.

- [X] T001 Write `tests/fixtures/capture_baselines.py`: a script, run once from `main`, that trains
      `DeepQLearningModel` at `seed=42` on three configurations — `CartPole-v1` (1-D `Box`
      observation, MLP branch), `FrozenLake-v1` (`Discrete` observation, MLP branch plus the
      cardinality rescaling path), and a shaped 3-D `Box` environment (CNN branch) — and writes both
      artifacts named in T002 and T003
- [X] T002 Emit the golden fixture to `tests/fixtures/golden/dqn_baseline.json`: per-episode reward
      series plus one SHA-256 per parameter tensor, **hashed in `parameters()` order and not keyed by
      parameter name** (research R8 — a name-keyed fixture breaks under T016's rename, for a reason
      unrelated to the numbers)
- [X] T003 Emit two real, loadable pre-refactor checkpoints to `tests/fixtures/checkpoints/`, one per
      network branch, by calling `save()` on the trained models. These are contract C4's test input
      and, unlike the fixture, are **not** deleted at feature closure
- [X] T004 Hand-build `tests/fixtures/checkpoints/legacy_pre006.json`, a minimal file with a
      `q_network_state_dict` key. It cannot be captured: feature 006 replaced that format, so no code
      in the tree can still produce one (research R8)
- [X] T005 Run T001 and commit all four artifacts **before any source file is touched**
- [X] T006 [P] Amend `.specify/memory/constitution.md` to 1.3.0: add `OffPolicyReplayModel` and
      `ContinuousActorCriticModel` to the Root Class Registry, and add the one clarifying sentence to
      Principle VI stated in the spec's Constitution Impact section

**Checkpoint**: four fixture files committed, constitution at 1.3.0, `src/` untouched.

---

## Phase 2: Foundational — the shared ancestor

**Blocks every user story.** Nothing below this line can start until the ancestor exists and the deep
model is rebuilt on it.

- [X] T007 Create `src/hercule/models/off_policy/__init__.py` with `OffPolicyReplayModel(RLModel)`,
      abstract, declaring the hook surface of `data-model.md`: `_build_networks`,
      `_build_optimizers`, `_select_action`, `_update`, `_networks`, `_optimizers` abstract;
      `_ready_to_update`, `_on_training_step`, `_sync_targets`, `_target_pairs`,
      `_target_sync_interval`, `_extra_state`, `_load_extra_state` concrete with the stated defaults
- [X] T008 Move `ExperienceReplayBuffer` into `off_policy`, widening the stored action from `int` to
      `int | np.ndarray` and stacking rather than assuming scalars on collation (FR-004)
- [X] T009 Add `Encoder` to `off_policy`: MLP branch `Linear(in,128) → ReLU → Linear(128,128) → ReLU`,
      CNN branch `Conv2d ×3 with ReLUs → Flatten → Linear(flat,512) → ReLU`, layers held in one
      `nn.Sequential` named `layers`. **Construction order is observable behaviour** — see the plan's
      refactor constraint; the shape probe stays between the convolutions and the linear layer
- [X] T010 Implement the per-dimension action mapping in `off_policy` (`to_env`, `to_policy`), caching
      `_action_low`/`_action_high` **only when the action space is a `Box`** (guarantee G8 — a
      `Discrete` space has neither, and reading them unconditionally breaks the rebuilt deep model)
- [X] T011 Move the episode loop into `OffPolicyReplayModel.run_epoch()`, transcribing the call
      sequence in `data-model.md` exactly: `begin_episode()` **before** priming, `push()` **inside**
      the training branch, `_on_training_step()` at the point the epsilon decay occupies today,
      `_sync_targets()` once per environment step
- [X] T012 Move frame stacking, priming, observation rescaling, device handling, seeding and
      `_needs_seeded_reset` into the ancestor, keeping `configure()`'s order: seed, then
      `_build_networks()`, then `_build_optimizers()`, with nothing drawing from torch's RNG in
      between (guarantee G1)
- [X] T013 Move checkpoint assembly into the ancestor: `networks_b64` from `_networks()`,
      `optimizer_state_b64` as a **name-keyed mapping** from `_optimizers()`, `rng_state_b64` and the
      counters written **by the ancestor itself**, `_extra_state()` merged in (guarantee G7)
- [X] T014 Bump `_CHECKPOINT_FORMAT_VERSION` to 3 and make `_import` dispatch across 3, 2 and the
      pre-006 legacy form (contract C4)
- [X] T015 Implement the parameter-key migration of contract C4: **select the branch first** — image
      if any key begins `conv_layers.`, else vector — then apply only that branch's rows, dropping the
      image branch's `network.*` aliases. No flat application order is correct for both branches
- [X] T016 Rebuild `DeepQLearningModel` on `OffPolicyReplayModel`: `QNetwork` becomes
      `Encoder` + `head`, constructed in that order; the target network is still **constructed**, not
      copied, so it consumes the same second RNG sequence it does today
- [X] T017 Implement the deep model's hooks: `_select_action` (epsilon-greedy, returning the action
      twice since its own coordinates are the environment's), `_on_training_step` (epsilon decay),
      `_target_pairs` (online → target), `_target_sync_interval` (`target_update_frequency`),
      `_extra_state` (`epsilon`, `frame_stack`, `observation_shape`)
- [X] T018 Verify `ruff check .` and `ruff format --check .` are clean, and that `get_available_models()`
      still returns exactly the four concrete models — the two new abstract classes must not appear

**Checkpoint**: the ancestor exists, the deep model runs on it, nothing is verified yet.

---

## Phase 3 (US2, P1): Existing results and models keep their meaning

**Goal**: prove the refactor changed no number and broke no stored artifact.
**Independent test**: `uv run pytest tests/models/test_golden_fixture.py tests/models/test_checkpoint_compat.py`

- [X] T019 [US2] Write `tests/models/test_golden_fixture.py`: re-run each of T001's three
      configurations at `seed=42` and assert the reward series and the order-keyed weight hashes match
      `tests/fixtures/golden/dqn_baseline.json` **exactly**. CPU-pinned (research R9 — bit-identity
      across devices is not achievable and asserting it fails on machines that have a GPU)
- [X] T020 [US2] Write `tests/models/test_checkpoint_compat.py`: load each of T003's two pre-refactor
      checkpoints and T004's legacy file, assert each loads without error and that the loaded weights
      equal the stored ones. **This is the test the fixture cannot be** — the fixture compares tensors
      and never opens a file (contract C4)
- [X] T021 [US2] Extend `test_checkpoint_compat.py` with a round-trip at version 3: save, load, assert
      every network including the delayed copies, every optimizer's state, and all three RNG streams
      come back identical
- [X] T022 [P] [US2] Assert `hercule play` renders a pre-refactor checkpoint, including a stacked one,
      exercising the `frame_stack`/`observation_shape` keys that let a default-configured model rebuild
      the right network shape (User Story 2 scenario 3)
- [X] T023 [P] [US2] Assert the tabular and random-baseline models are untouched: the existing tests in
      `tests/models/` must pass unmodified

**Checkpoint**: the refactor is certified. Only now is a failure in Phase 4 attributable to SAC.

---

## Phase 4 (US1, P1): Benchmark a continuous-control algorithm from a config

**Goal**: `hercule learn` trains SAC on a `Box` action space and writes ordinary run directories.
**Independent test**: SC-001's bar on `Pendulum-v1`, asserted by `tests/models/test_training_bars.py`

### The family-B abstraction

- [ ] T024 [US1] Create `src/hercule/models/continuous_actor_critic/__init__.py` with
      `ContinuousActorCriticModel(OffPolicyReplayModel)`, abstract: actor, two value estimators, two
      delayed copies, three optimizers. **No target actor** (FR-019)
- [ ] T025 [US1] Set `requires_grad = False` on every delayed-copy parameter at construction — one
      half of the mechanism by which the learning target cannot leak gradient (FR-015); the other half
      is T032's `no_grad`
- [ ] T026 [US1] Implement gradual averaging by `tau` inside `_update()`, once per **gradient** step,
      and leave `_target_sync_interval()` at its `None` default so the ancestor's hard copy never runs
      (FR-007)

### SAC's networks and declarations

- [ ] T027 [US1] Create `src/hercule/models/sac/__init__.py` with `SACHyperParams`: `learning_rate`
      3e-4, `discount_factor` 0.99, `tau` 0.005, `batch_size` 256, `replay_buffer_size` 100000,
      `step_modulo` 1, `learning_starts` 1000, `init_temperature` 1.0, `frame_stack` **0**,
      `weight_decay` 0.0, `seed` 42. Target entropy is **not** here: FR-017 fixes it at `-d`
- [ ] T028 [US1] Declare `model_name = "sac"`, `hyperparams_class`, and
      `supported_spaces = frozenset({(SpaceKind.BOX, SpaceKind.BOX)})` — members must be `SpaceKind`
      values; a set of string pairs registers fine and then rejects every continuous environment
- [ ] T029 [P] [US1] Implement `GaussianTanhActor`: `Encoder` → `Linear(h, 2*d)` → `(mean, log_std)`,
      `log_std` clamped to `[-20, 2]` as module constants, not hyperparameters (research R4)
- [ ] T030 [P] [US1] Implement `ContinuousCritic`: `Encoder(obs)` → concatenate the action →
      `Linear(f+d, 256)` → ReLU → `Linear(256, 1)`. It takes the action as an **input** and returns one
      value (FR-012); the deep model's observation-to-vector network cannot serve here
- [ ] T031 [US1] Implement the squashing correction in its numerically stable form,
      `Σ 2·(log 2 − u − softplus(−2u))` — the naive `Σ log(1 − tanh²)` underflows to `log(0)` for
      `|u| ≳ 9` in float32, which ordinary training reaches (research R5)

### The four learned quantities

- [ ] T032 [US1] Implement the learning target of FR-015 exactly: reward, plus — suppressed **only**
      on a terminal successor, never on a time-limit cut-off — discount × [lesser of the two **delayed**
      copies at an action **resampled from the current policy**, minus temperature × its log-density].
      Computed under `torch.no_grad()` so it is a constant for learning
- [ ] T033 [US1] Implement the estimators' objective (FR-023): **both** regress toward that target, at
      the observation and action **as stored** in the replay history — not at a resampled action
- [ ] T034 [US1] Implement the actor's objective (FR-022): lesser of the two **live** estimators at a
      differentiable sample, minus temperature × log-density, **confined to the actor's parameters** —
      no gradient onto the estimators or the temperature
- [ ] T035 [US1] Implement the temperature (FR-024): learned by gradient descent on an objective whose
      gradient is `−(log-density + target entropy)`, the log-density held constant, and the optimised
      quantity is `log_alpha` so the temperature cannot cross zero and invert the entropy term
- [ ] T036 [US1] Set `target_entropy = -d` from the action space at configure time (FR-017)
- [ ] T037 [US1] Implement `_select_action`: uniform in normalised `[-1,1]^d` while
      `step_count < learning_starts`, then the policy — stochastic when training, **deterministic when
      not** (FR-014). Returns `(env_action, normalised_action)`; the warmup action is already in
      storage coordinates when it returns (research R6, obligation O3)
- [ ] T038 [US1] Override `_ready_to_update()` to add `step_count >= learning_starts`, so no gradient
      step is taken during the warmup
- [ ] T039 [US1] Implement `_networks`, `_optimizers`, `_target_pairs`, `_extra_state` (`log_alpha`,
      `frame_stack`, `observation_shape`) and `load_from_dict` — without the last, `hercule play` does
      not work on SAC and nothing else fails (FR-027)

### The direct tests

- [ ] T040 [P] [US1] `tests/models/test_action_mapping.py` — SC-003: against the real
      `CarRacing-v3(continuous=True)` action space, `u = -1` maps exactly to each dimension's `low` and
      `u = +1` to its `high`; **and** `act()`/`predict()` return environment coordinates
- [ ] T041 [P] [US1] `tests/models/test_sac_objectives.py` — SC-010: the correction of T031 against the
      naive closed form on moderate inputs, computed in **normalised** coordinates, so both dropping it
      and measuring it in environment coordinates fail
- [ ] T042 [US1] SC-011: the target of T032 clause by clause against a hand-computed value on a fixed
      batch. Six independent cases — live instead of delayed estimators; greater or mean instead of
      lesser; stored instead of resampled action; entropy added or omitted; bootstrap suppressed on a
      cut-off; **and** a non-value-level case asserting the actor, temperature and delayed copies
      received no gradient from the target
- [ ] T043 [US1] SC-012: the actor and temperature objectives against hand-computed values, with each
      wrong form failing independently — delayed instead of live estimators, one instead of the lesser
      of two, entropy omitted, a sample carrying no gradient to the policy, a wrong temperature
      gradient, a temperature optimised directly and driven across zero, a reversed direction, and the
      non-value-level case that the estimators and the temperature received no gradient from the actor
- [ ] T044 [US1] SC-013: both estimators move toward the target, each at the **stored** observation and
      action. Training only one, or regressing at a resampled action, leaves every other criterion green
- [ ] T045 [P] [US1] `tests/models/test_sac_structure.py` — SC-014: `target_entropy == -d`, asserted on
      an environment of **at least two** action dimensions so `-d` is distinguishable from `-1`
- [ ] T046 [P] [US1] SC-015: the actor and the two estimators hold **pairwise disjoint** parameters —
      all three pairs. Sharing makes the checkpoint smaller, so no size criterion can catch it
- [ ] T047 [P] [US1] `tests/models/test_polyak.py` — SC-016: the delayed copies advance once per
      gradient step and not once per environment step, under a configuration where the two clocks
      differ; each advance is the configured fraction toward the live parameters, hand-computed, so a
      hard copy on the right clock fails; and, from delayed copies **equal** to the live parameters,
      after one known non-zero update with `0 < tau < 1`, they differ
- [ ] T048 [P] [US1] SC-017: evaluation is deterministic — two evaluation passes over the same
      observation sequence give the same actions, two training steps do not
- [ ] T049 [P] [US1] SC-009: extend `tests/supervisor/test_space_gating.py` — SAC paired with a
      discrete-action environment is skipped with a message naming both kinds, while the other
      combinations in the same config still run

### The oracle and the first bars

- [ ] T050 [US1] Create `src/hercule/environnements/oracle.py` with `AsymmetricOracleEnv` per contract
      C2: observation `Box([-1,2],[1,4])`, action `Box([-1,0],[1,4])`, target redrawn every step,
      `reward = 1 − 0.5·squared error`, 50 steps, always truncated
- [ ] T051 [US1] Add the import to `src/hercule/environnements/__init__.py` so the registration runs.
      **Placing the module under the package does not execute it**, and without this line the config in
      T053 fails in a fresh process
- [ ] T052 [P] [US1] `tests/environnements/test_asymmetric_oracle.py` — assert contract C2's five
      closed-form rows: optimum 50.0, and each of the four degenerate policies below 45.0. This
      establishes that the oracle can detect what it exists to detect, and is the cheapest test in the
      feature
- [ ] T053 [P] [US1] Ship `experiments/sac_oracle.yaml` and `experiments/sac_pendulum.yaml` (FR-029)
- [ ] T054 [US1] `tests/models/test_training_bars.py`, marked `slow` — SC-002: 90% of the oracle's
      optimum within 200 episodes, and **zero** out-of-bounds actions across the evaluation episodes;
      SC-001: strictly above `-200` over 20 evaluation episodes on `Pendulum-v1` within 500 episodes.
      A YAML plus `hercule learn` trains and asserts nothing — only this file can fail

**Checkpoint**: Hercule trains a continuous-action agent and every clause of the algorithm is pinned.

---

## Phase 5 (US3, P2): Inspect, replay and compare a continuous agent

- [ ] T055 [P] [US3] Assert `hercule play` runs a trained SAC agent and that every action it submits is
      within the environment's declared bounds for a full episode (FR-010)
- [ ] T056 [P] [US3] Assert `hercule report` produces a comparative report over a grid of SAC runs and
      ranks them, with **no change to `reports/`** (FR-030)
- [ ] T057 [P] [US3] Assert a SAC hyperparameter list expands to independent run directories under
      distinct signatures (FR-021, User Story 1 scenario 1) and that a re-run at a higher epoch ceiling
      resumes rather than restarting (scenario 2)

---

## Phase 6 (US4, P3): Reach the image-observation environment

- [ ] T058 [US4] Ship `experiments/sac_car_racing.yaml` at `learn_max_epoch: 700`, matching
      `dq_car_racing.yaml` so the two runs are comparable, with the checkpoint interval set for a
      ~129 MB write
- [ ] T059 [US4] `tests/models/test_sac_persistence.py` — SC-008: a checkpoint written on a shaped
      CarRacing-like environment stays under **150 MB**, and the deep model's own **50 MB** guard is
      still green
- [ ] T060 [US4] SC-005: interrupt and resume — the temperature, every optimizer's state, the delayed
      copies' lag and all three RNG streams continue from their stored values, none reverting to an
      initial one
- [ ] T061 [US4] Run the recorded experiment and check SC-007: at epoch 700, mean test reward over 20
      evaluation episodes at least **+65.63**, against the measured random baseline of `-34.37`. If it
      misses, record the finding — do not lower the bar
- [ ] T062 [US4] Run the recorded experiment for SC-006 on `LunarLanderContinuous`: at least **200**
      over 100 evaluation episodes within 3000 episodes, and ship `experiments/sac_lunarlander.yaml`

---

## Phase 7: Polish and closure

- [ ] T063 Retire the golden fixture: delete `tests/models/test_golden_fixture.py` and
      `tests/fixtures/golden/`, replacing them with a determinism property test — same seed twice
      identical, different seeds different — which stores no expectation and survives later deliberate
      behaviour changes (FR-002, SC-004). `tests/fixtures/checkpoints/` **stays**
- [ ] T064 [P] Document in `CLAUDE.md` that a resumed off-policy run restarts with an empty replay
      buffer, and that the step in the learning curve is expected (FR-028)
- [ ] T065 [P] Add a `CLAUDE.md` gotcha recording that module **construction order** is observable
      behaviour whenever a seeded RNG initialises weights — the single most expensive thing to
      rediscover in this feature
- [ ] T066 [P] Add a `CLAUDE.md` gotcha recording that a `state_dict`'s keys are attribute paths, so
      renaming a module breaks every stored checkpoint while changing no number, and that a
      weight-comparing fixture is structurally blind to it
- [ ] T067 [P] Update `AGENTS.md`'s "How to Add a New RL Algorithm" with the `OffPolicyReplayModel`
      hook surface
- [ ] T068 `uv run ruff check . --fix && uv run ruff format .`
- [ ] T069 `uv run pytest` green and `uv run gen-doc` still succeeding (SC-018)
- [ ] T070 Write the PR's **Constitution Impact** section: two Root Class Registry additions and one
      Principle VI clarification, 1.2.0 → 1.3.0

---

## Dependencies

```text
Phase 1 (T001-T006)   captures + constitution
   |                  T005 MUST precede every source change in Phase 2
Phase 2 (T007-T018)   the ancestor, the rebuild, the migration
   |
Phase 3 (T019-T023)   US2 — the refactor is certified
   |                  a failure after this point is attributable to SAC
Phase 4 (T024-T054)   US1 — SAC
   |                  T024-T026 -> T027-T039 -> T040-T049 -> T050-T054
   +-- Phase 5 (T055-T057)   US3 — play and report
   +-- Phase 6 (T058-T062)   US4 — the image environment
          |
Phase 7 (T063-T070)   closure
```

Within Phase 2, T009 (Encoder) blocks T016 (the rebuilt network); T013 (checkpoint assembly) blocks
T014 and T015. Within Phase 4, T032 blocks T042, T034 and T035 block T043, T033 blocks T044.

## Parallel opportunities

- **Phase 1**: T006 runs alongside T001-T005; it touches no source.
- **Phase 3**: T022 and T023 are independent of T019-T021.
- **Phase 4**: T029 and T030 are two different files; T040, T041, T045, T046, T047, T048, T049 are
  seven independent test files; T052 and T053 are independent of both.
- **Phase 5**: all three tasks.
- **Phase 7**: T064-T067 are four different documents.

## MVP

**Phases 1 through 4.** That is the feature: Hercule trains a continuous-action agent, and the
refactor underneath it is certified. Phases 5 and 6 refine what can be done with the result; Phase 7
closes the fixture out.

Phases 1-3 alone deliver nothing a user can see — they produce a refactor that changes no behaviour,
which is exactly the point. They are not a shippable increment and should not be presented as one.

## Task summary

| Phase | Story | Tasks | Count |
|---|---|---|---|
| 1 Setup | — | T001-T006 | 6 |
| 2 Foundational | — | T007-T018 | 12 |
| 3 | US2 (P1) | T019-T023 | 5 |
| 4 | US1 (P1) | T024-T054 | 31 |
| 5 | US3 (P2) | T055-T057 | 3 |
| 6 | US4 (P3) | T058-T062 | 5 |
| 7 Polish | — | T063-T070 | 8 |
| **Total** | | | **70** |

Of the 70, **24 are test tasks**. That ratio is not an accident of style: the specification's own
standard is that a wrong reinforcement-learning implementation still trains, still improves, and still
produces a plausible curve, so for most of this feature a direct test is the only thing that
distinguishes correct from incorrect.
