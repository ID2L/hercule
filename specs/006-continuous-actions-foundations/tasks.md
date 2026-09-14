---
description: "Task list for feature 006 — Foundations for continuous action spaces"
---

# Tasks: Foundations for Continuous Action Spaces

**Input**: `specs/ROADMAP-continuous-actions.md` (Phase 0, specs 0.1–0.5)
**Prerequisites**: none — this is the first phase of the roadmap and blocks every later one.

**Scope**: this feature contains **no new algorithm**. It fixes five defects that would otherwise
be frozen into the shared ancestor built in feature 007, and it establishes the contracts that
DecQN, TD3 and SAC depend on. Every defect below was verified against the source or reproduced by
running code; the roadmap's §0 and §1 record the evidence.

**Tests**: Test tasks ARE included. Four of the five defects are silent — they produce a plausible
learning curve while training against a wrong target, restarting exploration, or destroying target
lag — so a test is the only thing that distinguishes fixed from unfixed.

**Behaviour changes**: 0.3 and 0.4 deliberately change how `deep_q_learning` learns. Results
already in `outputs/` become non-comparable with post-fix runs. This is intentional and must be
stated in the PR: the previous numbers were computed against a wrong TD target.

## Format: `[ID] [P?] [Spec] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Spec]**: which roadmap sub-spec the task serves (S01–S05)
- Exact file paths and line references are given in every task

## Path Conventions

Single project: `src/hercule/`, `tests/` at repository root.

## Ordering constraints (from the roadmap)

- S01 and S02 are independent of the other three and of each other.
- S03 and S04 **must both land before** the golden fixture is captured in feature 007.
- S05 **must land before** feature 007, which builds checkpoint assembly into the ancestor.

---

## Phase 1: Setup

**Purpose**: test scaffolding for models

**Correction to the original task list**: `tests/models/` already exists and already holds **20
passing tests** across `test_deep_q_learning_observations.py` and
`test_deep_q_learning_updates.py`. The claim that models have no test coverage was wrong.

**Baseline before any change: `uv run pytest` = 224 passed in 55 s.** Every task below must leave
that suite green. Three existing tests interact directly with this feature's changes and must be
updated **deliberately**, not deleted:

- `test_deep_q_learning_observations.py::test_replay_buffer_keeps_the_native_dtype` calls
  `ExperienceReplayBuffer.push` positionally — T020 changes its signature
- `test_deep_q_learning_observations.py::test_saved_stacked_model_loads_into_a_default_configured_model`
  exercises the export/import round-trip that T035–T041 re-encode
- `test_deep_q_learning_updates.py::test_target_network_lags_the_online_network` asserts the lag
  that T038 must preserve across a save/load, not just within a run

- [X] T001 ~~Create the test package `tests/models/__init__.py`~~ — already exists; no action
- [X] T002 [P] Add `tests/models/conftest.py` fixtures: a `StubEnv` factory parameterised on
      `(observation_space, action_space)` and on whether an episode ends by `terminated` or by
      `truncated`, so S03 can assert the TD target without a real Gymnasium environment
- [X] T003 [P] Add a `tiny_dqn` fixture to `tests/models/conftest.py`: a `DeepQLearningModel`
      configured on a 1-D observation / 2-action stub with `batch_size=2`, `replay_buffer_size=16`,
      so a training step runs in milliseconds

**Checkpoint**: `uv run pytest tests/models` collects and passes with zero tests

---

## Phase 2: S01 — Concrete-only model discovery

**Purpose**: stop the registry exposing abstract classes. Blocks the ancestor in feature 007, which
would otherwise be registered as a phantom model.

**Evidence**: running `get_available_models()` today returns
`tdmodel -> hercule.models.td_models.TDModel abstract=True`. `create_model("tdmodel")` raises
`TypeError: Can't instantiate abstract class`.

- [X] T004 [S01] Add `import inspect` and filter the discovery loop in
      `src/hercule/models/__init__.py:368` so a class is registered only if
      `not inspect.isabstract(attr)` **and** `"model_name" in attr.__dict__`
- [X] T005 [S01] Remove the `attr.__name__.lower()` fallback at
      `src/hercule/models/__init__.py:371-374`; log `logger.warning` naming the class and its module
      when a concrete `RLModel` subclass declares no own `model_name`
- [X] T006 [S01] Require `supported_spaces` (T010) to be declared on any registered class; a
      concrete model missing it is warned about and **not** registered
- [X] T007 [P] [S01] `tests/models/test_registry.py`: the registry returns exactly
      `{deep_q_learning, dummy, simple_q_learning, simple_sarsa}` and `tdmodel` is absent
- [X] T008 [P] [S01] `tests/models/test_registry.py`: no returned class satisfies
      `inspect.isabstract`; a locally defined abstract `RLModel` subclass imported into a scanned
      package is not registered
- [X] T009 [P] [S01] `tests/models/test_registry.py`: `create_model("tdmodel")` raises `ValueError`
      with the available names, not `TypeError` from abstract instantiation

**Checkpoint**: `uv run hercule learn experiments/frozenlake_4x4.yaml` still resolves its models

---

## Phase 3: S02 — Enumerable space taxonomy

**Purpose**: make (observation × action) space support declarable and checkable. Today
`check_space_is_box` returns `not is_discrete`, so `MultiDiscrete`, `Dict`, `Tuple`, `Text`,
`Sequence`, `Graph`, `MultiBinary` and `OneOf` all read as `Box`.

**Evidence**: `grep -rn check_space_is_box src/ tests/` returns **only its own definition** — zero
callers, so correcting it carries no regression risk. Gymnasium 1.2.3 ships 10 space classes.

- [X] T010 [S02] Add `SpaceKind` covering all 10 Gymnasium space classes and
      `classify_space(space) -> SpaceKind` to `src/hercule/environnements/spaces_checker.py`.
      **Correction: this task said `StrEnum`, which is Python 3.11+.** `pyproject.toml` declares
      `requires-python = ">=3.10"` and the interpreter in use is 3.10.11, so the import fails outright.
      Implemented as `(str, Enum)` with an explicit `__str__` returning the value — without it the mixin
      renders as `SpaceKind.BOX`, putting the enum's own name into the user-facing mismatch messages
      instead of `Box`.
- [X] T011 [S02] Correct `check_space_is_box` at
      `src/hercule/environnements/spaces_checker.py:10-11` to
      `isinstance(space, gym.spaces.Box)`. Zero callers — no audit needed
- [X] T012 [S02] Add `supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]]` to
      `RLModel` in `src/hercule/models/__init__.py`. **No permissive default** — a default of
      "everything" recreates exactly the silent-accept this spec removes
- [X] T013 [S02] Declare `supported_spaces` explicitly on all four existing models:
      `deep_q_learning` (`{(BOX, DISCRETE), (DISCRETE, DISCRETE)}`), `td_models`
      (`{(DISCRETE, DISCRETE)}`), `simple_q_learning` and `simple_sarsa` (inherited from `TDModel`),
      `dummy` (every pair)
- [X] T014 [S02] Enforce the pair check in `src/hercule/supervisor/__init__.py` **before**
      `model.configure(...)` at `:58` and `:80`, comparing
      `(classify_space(env.observation_space), classify_space(env.action_space))` against the model
      class's ClassVar. **Not behind `configure()`**: `DeepQLearningModel.configure` calls
      `super().configure(env, hyperparameters)` at
      `src/hercule/models/deep_q_learning/__init__.py:257` and **discards its return value**, so a
      base-class rejection would never reach the caller
- [X] T015 [S02] On mismatch, `Supervisor` skips that (environment, model) combination, logs a
      message naming the model, the environment, the expected kinds and the actual kinds, and
      **continues with the remaining combinations** rather than aborting the run
- [X] T016 [P] [S02] `tests/environnements/test_spaces_checker.py`: parametrised classification over
      `FrozenLake-v1`, `CartPole-v1`, `Pendulum-v1`, `CarRacing-v3(continuous=True)` and
      `CarRacing-v3(continuous=False)` asserting the `(observation, action)` pair
- [X] T017 [P] [S02] `tests/environnements/test_spaces_checker.py`: `check_space_is_box` is `False`
      for `MultiDiscrete`, `MultiBinary`, `Dict` and `Tuple` — the cases the old implementation
      accepted
- [X] T018 [P] [S02] `tests/supervisor/test_space_gating.py`: a config pairing `simple_q_learning`
      with `CartPole-v1` **and** a valid combination runs the valid one and skips the invalid one
      with a message naming both kinds
- [X] T019 [S02] **Constitution Impact**: adding a `ClassVar` subclasses depend on is a semantic
      change per `AGENTS.md`. Amend `.specify/memory/constitution.md`, bump 1.1.0 → 1.2.0, write the
      Sync Impact Report, and include a "Constitution Impact" section in the PR. The abstract-method
      surface is unchanged; say so explicitly

**Checkpoint**: an invalid model/environment pairing is reported at supervision time instead of
failing later with an unrelated error

---

## Phase 4: S03 — Terminated vs truncated

**Purpose**: stop destroying the bootstrap at every time limit. **This changes what DQN learns.**

**Evidence**: `run_epoch` computes `done = terminated or truncated`
(`src/hercule/models/deep_q_learning/__init__.py:488`) and pushes that single flag (`:499`);
`_train_step` applies `rewards + (discount_factor * next_q_values * ~dones)` (`:578`). A Gymnasium
time-limit truncation is not an MDP terminal state. `Pendulum-v1` **always** truncates at 200 steps
and never terminates; `CarRacing-v3` truncates at `max_episode_steps` (1000 in
`experiments/dq_car_racing.yaml`). Every time-limit transition therefore gets a target of `r`
instead of `r + gamma * V(s')`.

- [X] T020 [S03] Widen `ExperienceReplayBuffer.push` at
      `src/hercule/models/deep_q_learning/__init__.py:166` to take `terminated: bool` and
      `truncated: bool` as separate fields, and update the docstring
- [X] T021 [S03] Update the `push` call at `:499` to pass both flags; keep
      `done = terminated or truncated` at `:488` for the **episode-end** test only — the loop must
      still stop on either
- [X] T022 [S03] In `_train_step` at `:570` and `:578`, build the mask from `terminated` alone:
      `target = rewards + discount_factor * next_q_values * ~terminateds`
- [X] T023 [P] [S03] `tests/models/test_dqn_targets.py`: on a stub that truncates without
      terminating, the computed target equals `r + gamma * max_a Q_target(s', a)`
- [X] T024 [P] [S03] `tests/models/test_dqn_targets.py`: on a stub that terminates, the target
      equals `r` exactly
- [X] T025 [P] [S03] `tests/models/test_dqn_targets.py`: a transition stored mid-episode is
      unaffected by either flag
- [X] T026 [S03] ~~Record a measured before/after on `CartPole-v1`~~ — **measured, and the task as
      written cannot be satisfied. `CartPole-v1` is structurally incapable of demonstrating this
      fix.** Controlled experiment, same seed, same hyperparameters, "before" reproduced by
      collapsing the two flags at the push site exactly as the old code did:

      | Horizon | before | after | truncated episodes |
      |---|---|---|---|
      | default (500 steps), 150 epochs | mean 85.41, last50 167.28 | mean 85.41, last50 167.28 | **0 / 150** |
      | capped at 100 steps, 200 epochs | mean 69.33, last50 99.96 | mean 68.05, last50 99.96 | 109-111 / 200 |

      Both readings are dead ends, for opposite reasons. At the default horizon the agent never
      reaches 500 steps within a reasonable run (max 300), so **no truncation ever occurs** and the
      corrected code path is never taken — identical numbers to the digit. Capped at 100 the
      truncations do happen, but the task becomes trivial and both conditions saturate the ceiling
      (`last50` 99.96 of a possible 100), leaving no room to separate them; the 1.3 gap in the
      overall mean is single-seed noise.

      The fix is nonetheless correct — T023-T025 prove the TD target itself — but a *behavioural*
      delta needs an environment that truncates **and** has value left to estimate at the horizon.
      `Pendulum-v1` is that environment (it always truncates, never terminates) and is exactly the
      correctness oracle features 008-010 use — but it is continuous-action, so no model in this
      repo can run it yet. Re-measure there once DecQN lands.
- [ ] T027 [S03] State in the PR that existing `outputs/` results are **non-comparable** with
      post-fix runs, and why that is the correct trade. Note T026's finding sharpens this rather
      than weakening it: the divergence is invisible on the runs already recorded because those
      never truncated, so "non-comparable" is a statement about future runs on truncating
      environments, not a claim that the existing CartPole numbers are wrong.

**Checkpoint**: `uv run pytest tests/models/test_dqn_targets.py` passes; a short `Pendulum`-style
truncating run no longer collapses its value estimates at the horizon

---

## Phase 5: S04 — Make `seed` live

**Purpose**: the `seed` hyperparameter currently controls nothing. Feature 007's golden-fixture
regression test is unachievable until it does.

**Evidence**: `DeepQLearningModelHyperParams.seed` is declared at
`src/hercule/models/deep_q_learning/__init__.py:57` and **never read**; `__init__` hardcodes
`torch.manual_seed(42)` at `:235`; `random` — which drives epsilon-greedy at `:449` and replay
sampling at `:189` — is never seeded; `run_epoch` calls `env.reset()` with no seed at `:474`. Two
YAML runs with `seed: 1` and `seed: 999` differ only through uncontrolled RNG.

- [X] T028 [S04] In `DeepQLearningModel.configure`, seed `torch.manual_seed(seed)` and
      `random.seed(seed)` from `typed_params.seed`, **before** `_build_from_spaces()` at `:269` —
      network initialisation consumes the torch RNG, so seeding after it is a no-op for the weights
- [X] T029 [S04] Delete `torch.manual_seed(42)` from `__init__` at
      `src/hercule/models/deep_q_learning/__init__.py:234-235`
- [X] T030 [S04] Add a model-owned `_rng: np.random.Generator = PrivateAttr(...)` seeded from the
      same value, and route **all** NumPy randomness in the model through it. Never call the global
      `np.random.*` functions: S05 checkpoints `Generator.bit_generator.state`, and that is only the
      state actually consumed if nothing bypasses the owned generator. `TDModel` already holds such
      a generator (`src/hercule/models/td_models/__init__.py:67`); `deep_q_learning` holds none
- [X] T031 [S04] Seed the environment on the **first reset of a fresh run only**:
      `env.reset(seed=typed_params.seed)` at `:474`, subsequent resets unseeded. On **resume** the
      RNG state restored by S05 wins and the seeded reset is **not** re-issued — otherwise S05's
      round-trip is dead on the path S04 writes
- [X] T032 [P] [S04] `tests/models/test_determinism.py`: two runs with the same `seed` produce
      bit-identical weights and identical episode rewards
- [X] T033 [P] [S04] `tests/models/test_determinism.py`: two runs with different `seed` values
      produce different weights. Today this passes for the wrong reason — assert it still passes
      **after** T028–T031, which is what makes it meaningful
- [X] T034 [P] [S04] `tests/models/test_determinism.py`: a grep-style guard asserting
      `src/hercule/models/deep_q_learning/__init__.py` contains no bare `np.random.` call

**Checkpoint**: `seed` is a real hyperparameter; the grid expansion over `seed` in
`experiments/*.yaml` stops producing runs that differ only by directory name

---

## Phase 6: S05 — Complete, compact checkpoints

**Purpose**: make a checkpoint (a) small enough for a three-network model, and (b) actually
resumable. Blocks feature 007, whose ancestor owns checkpoint assembly.

**Evidence, measured**: `outputs/dq_car_racing/.../fra_sta_3/model.json` is **133.9 MB** on disk.
Benchmarked on `QNetwork((96, 96, 12), 5)` (2,194,597 parameters, 8.8 MB as raw float32):

| Path | write | size | read+rebuild |
|---|---|---|---|
| current (`tensor.tolist()` + `json.dump(indent=2)`) | 3.95 s | 140.9 MB | 1.59 s |
| `torch.save` to buffer → base64 inside the same JSON | 0.23 s | 11.7 MB | 0.31 s |

12.0× smaller, 17× faster to write. A three-network actor-critic would be **423 MB vs 35 MB** per
checkpoint, per run, per grid variant. All of this lives inside `_export()`/`_import()`, which are
abstract hooks — **not** inside the `@final` `save()`/`load()` — so Constitution impact is nil.

- [X] T035 [S05] Add `_encode_state_dict` / `_decode_state_dict` helpers: `torch.save` a
      `state_dict` into a `BytesIO`, base64-encode, and decode with
      `torch.load(..., weights_only=True)`. `weights_only=False` would make `hercule play` on a
      shared model an arbitrary-code-execution path
- [X] T036 [S05] Emit weights under a **versioned** key (e.g. `format_version: 2`,
      `networks_b64`) in `_export()` at `src/hercule/models/deep_q_learning/__init__.py:588-619`
- [X] T037 [S05] Keep reading the legacy `q_network_state_dict` list format in `_import()` at
      `:621-664`, so the 133 MB models already in `outputs/` keep loading and `hercule play` keeps
      working on them
- [X] T038 [S05] Serialise the **target network in its own right**. `_import` currently loads the
      *online* dict into `_target_network` at `:654-657`, destroying the lag that is the target
      network's entire purpose — a resumed run restarts with zero lag
- [X] T039 [S05] Serialise the optimizer `state_dict` (Adam's moment buffers), so a resume does not
      restart with cold momentum
- [X] T040 [S05] Serialise mutated hyperparameters, `epsilon` first. `run_epoch` mutates
      `typed_params.epsilon` on every training step at `:507`; `_export()` at `:609-619` does not
      write it, so a run resumed at epoch 5000 keeps its step count and restarts exploration at the
      YAML value (default `1.0`, fully random)
- [X] T041 [S05] Serialise RNG state as **torch's RNG tensor, Python's `random.getstate()`, and
      NumPy's `Generator.bit_generator.state`**. Verified by running the round-trip: under
      `torch.load(..., weights_only=True)` those three load cleanly and `np.random.get_state()`
      (the legacy tuple, which contains an `ndarray`) raises `UnpicklingError`. Do not store the
      legacy tuple
- [X] T042 [S05] Document in the `_export` docstring that **replay buffer contents are out of
      scope**, with the consequence: a resumed off-policy run restarts with an empty buffer and
      re-fills it, a real discontinuity in the learning curve. Storing it means gigabytes per
      checkpoint
- [X] T043 [S05] Raise in `Runner` when `run_info.json` reports a non-zero epoch while `model.json`
      is missing. `RLModel.load()` returns silently in that case
      (`src/hercule/models/__init__.py:256-258`) and that tolerance is what makes a **fresh** run
      work, so it must stay. The error names the missing file and the reported epoch. No
      Constitution impact: `Runner` is a registry class but this is additive validation, not an API
      change
- [X] T044 [P] [S05] `tests/models/test_persistence.py`: export → import round-trip gives
      bit-identical online weights, target weights, optimizer state and RNG state
- [X] T045 [P] [S05] `tests/models/test_persistence.py`: a committed legacy-format fixture (list
      encoding) still loads
- [X] T046 [P] [S05] `tests/models/test_persistence.py`: after save-then-load, `epsilon` continues
      from its saved value instead of resetting to the YAML default
- [X] T047 [P] [S05] `tests/models/test_persistence.py`: after save-then-load, the target network
      is **not** equal to the online network when it had lag before saving
- [X] T048 [P] [S05] `tests/run/test_resume_guard.py`: a run directory with
      `run_info.json` at epoch > 0 and no `model.json` raises, naming both
- [X] T049 [S05] Measure and record the resulting `model.json` size for a CarRacing DQN.
      **The original "under 15 MB" bar was arithmetically impossible and is corrected to 50 MB.**
      15 MB was derived from the single-network benchmark (11.70 MB) without propagating T038 and
      T039, which require storing two more parameter-sized structures each:

      | Contents | size |
      |---|---|
      | online network alone (2,194,597 params, base64) | 11.70 MB |
      | + target network (T038 — its lag cannot be reconstructed) | 23.41 MB |
      | + Adam `exp_avg` and `exp_avg_sq` (T039 — one buffer per parameter, twice) | 46.82 MB |

      Measured: **22.35 MB untrained, 44.69 MB trained**, against 46.82 MB theoretical. gzip was
      tried and rejected — float32 weights are near-random, yielding ~9%.

      The corrected framing: 133.9 MB → 44.69 MB is a 3.0x reduction **while storing three times
      more information** (target network, optimizer state and RNG state, none of which the 133.9 MB
      file contained). Per unit of information retained the improvement is ~9x. The regression
      guard is set at 50 MB. Reaching 15 MB would require dropping the target network — which
      reintroduces the lag-destruction bug T038 exists to fix — or half-precision weights, which
      trades numerical precision for disk and was not requested.

**Checkpoint**: `uv run hercule learn experiments/dq_car_racing.yaml` writes checkpoints under
50 MB, and re-running the same config resumes exploration and target lag where it left off

---

## Phase 7: Polish

- [X] T050 [P] Update `CLAUDE.md`'s Gotchas: remove the stale "`TDModel.configure()` returns
      `False` … and `Supervisor` ignores the return value" entry, replaced by T014–T015
- [X] T051 [P] Add a Gotcha recording the `weights_only=True` / `np.random.get_state()`
      incompatibility measured in T041 — it is not discoverable from the torch docs
- [X] T052 [P] Add a Gotcha recording that a truncation is not a termination, with the
      `Pendulum-v1`-always-truncates evidence, so the fix is not silently reverted
- [X] T053 Update `AGENTS.md`'s "How to Add a New RL Algorithm" with the `supported_spaces`
      requirement from T012–T013
- [X] T054 `uv run ruff check . --fix && uv run ruff format .` — the repo is Ruff-clean; keep it so
- [X] T055 `uv run pytest` green, and `uv run gen-doc` still succeeds (a PR that breaks it fails the
      docs check)

---

## Dependencies

```
T001-T003  setup
   |
   +-- T004-T009   S01  registry          (independent)
   +-- T010-T019   S02  spaces            (independent; T012 needs T010)
   +-- T020-T027   S03  truncation        ──┐
   +-- T028-T034   S04  seeding           ──┼─> both must land before feature 007
   +-- T035-T049   S05  checkpoints       ──┘   captures its golden fixture
   |
   +-- T050-T055   polish
```

`T006` depends on `T012` (the ClassVar must exist before the registry can require it).
`T031` depends on `T041` (the resume path must know the RNG state is restored).
`T038`, `T040`, `T041` all depend on `T035` (the encode/decode helpers).

## Out of scope

- The `OffPolicyReplayModel` ancestor and the `deep_q_learning` refactor onto it — feature 007.
- DecQN, TD3, SAC — features 008, 009, 010.
- Replay buffer persistence — deliberately excluded, see T042.
- NAF — dropped from the roadmap; see `specs/ROADMAP-continuous-actions.md` Phase 5.
