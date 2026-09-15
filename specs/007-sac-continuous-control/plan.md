# Implementation Plan: SAC Continuous Actor-Critic on a Shared Off-Policy Ancestor

**Branch**: `007-sac-continuous-control` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/007-sac-continuous-control/spec.md` (revision 10,
converged after nine rounds of adversarial review)

## Summary

Extract from `deep_q_learning` the machinery every replay-based model needs — episode loop, replay
store, frame stacking, observation rescaling, device and seeding, action mapping, checkpoint
assembly — into `OffPolicyReplayModel`; rebuild `DeepQLearningModel` on it with **bit-identical**
behaviour; then add `ContinuousActorCriticModel` and `SACModel` beneath it, giving Hercule its first
model that can act on a `Box` action space.

The technical spine of the work is not the algorithm. SAC is well-specified and the spec pins every
clause of it. The spine is **FR-002's bit-identity requirement**, which turns an ordinary refactor
into a constrained one, in a way explained under "The constraint that governs the refactor" below —
it was not identified in the spec or the roadmap, and it determines the shape of the extraction.

## Technical Context

**Language/Version**: Python 3.10+ (`X | Y` unions, no `typing.Any`)
**Primary Dependencies**: existing only — torch, gymnasium 1.2.3, numpy, pydantic v2, click.
**No new dependency.** Box2D is already enabled (feature 005) and covers `LunarLanderContinuous`.
**Storage**: `outputs/` JSON tree, unchanged. `model.json` keeps feature 006's keys and base64
encoding; `optimizer_state_b64` becomes a name-keyed mapping and the format version therefore goes
to **3**, with `_import` dispatching across 3, 2 and the pre-006 legacy form (contract C4).
**Testing**: pytest, `--strict-markers`, markers `unit` / `integration` / `slow`. SC-001 and SC-002's
numeric bars are asserted by `slow` tests; only SC-006 and SC-007, which need hours, are recorded
experiments outside the suite. A shipped YAML is not a test: it trains, and asserts nothing.
**Target Platform**: Windows 11 primary dev, cross-platform library. CPU is the reference device;
CUDA is opportunistic and must not change results at a fixed seed on CPU.
**Project Type**: single Python package (`src/hercule`) with a Click CLI
**Performance Goals**: `Pendulum-v1` to SC-001's bar inside 500 episodes; CarRacing checkpoint
≤ 150 MB (derived, spec Measured Baselines); DQN's own 50 MB guard unchanged
**Constraints**: ruff-clean at line-length 120; `uv run gen-doc` keeps succeeding; DQN behaviour
bit-identical at a fixed seed
**Scale/Scope**: 4 new source packages, 1 rebuilt, ~30 requirements, ~18 success criteria

## The constraint that governs the refactor

FR-002 requires the rebuilt `DeepQLearningModel` to produce **identical stored weights** at a fixed
seed. Weights are initialised by `nn.Linear` / `nn.Conv2d` constructors, each of which draws from
torch's global RNG, seeded once in `configure()` before `_build_from_spaces()`
([deep_q_learning/__init__.py:296](../../src/hercule/models/deep_q_learning/__init__.py#L296)).

Therefore **the order in which parameterised modules are constructed is part of the observable
behaviour this feature must preserve.** Any extraction that creates the same layers in a different
order produces different initial weights from the same seed, and the golden fixture fails — with a
diff that looks like a deep numerical bug and is in fact a constructor-ordering change.

Two consequences, both load-bearing for every design choice below:

1. **The encoder/head split must preserve construction order.** Today's `QNetwork` builds, for a
   vector observation, `Linear(in,128) → Linear(128,128) → Linear(128,n)`, and for an image,
   `Conv2d ×3 → Linear(flat,512) → Linear(512,n)`. Splitting at the last layer — everything before
   it becomes `Encoder`, the last becomes the head — reproduces that order exactly in both branches,
   provided the encoder is constructed before the head. This is why the split goes *there* and not
   anywhere else, and it is why `Encoder` cannot lazily build its layers on first forward.
2. **Nothing may consume torch's global RNG between the seeding call and network construction**,
   and no new draw may be inserted among the existing ones. The ancestor's `configure()` must keep
   seeding immediately before `_build_networks()`, with no intervening allocation that draws.

3. **The deep model's target network must keep being *constructed*, not copied.** `_build_from_spaces()`
   builds a second complete `QNetwork` and only then overwrites it with the online weights
   ([deep_q_learning/__init__.py](../../src/hercule/models/deep_q_learning/__init__.py)). That second
   constructor consumes a full second sequence of torch RNG draws whose results are immediately
   discarded — but the RNG *state* afterwards is not discarded, and feature 006 checkpoints it.
   Replacing it with a copy, which is what `ContinuousActorCriticModel` legitimately does for its own
   delayed copies, leaves the weights identical and the RNG state different, so a resumed run diverges
   while a fresh one does not. The rebuilt deep model keeps the construction.

The dummy-input shape probe in `_build_cnn` runs under `torch.no_grad()` and draws nothing, so it is
safe to keep where it is. It must stay after the convolutions and before the fully-connected layers,
because that is where it is today.

## Constitution Check

*GATE: evaluated before implementation and re-evaluated at completion.*

| Principle | Verdict | Evidence |
|---|---|---|
| **I. Generic Algorithm Architecture** | **Engaged — amendment required** | Two abstract intermediates are added beneath `RLModel`, which Principle I explicitly permits. They take the position `TDModel` already occupies, and the spec's Constitution Impact section decides on that symmetry that both belong in the Root Class Registry. `RLModel`'s own abstract surface is untouched: no abstract method added, removed or renamed; `save()`/`load()` stay `@final` and unmodified. Bump 1.2.0 → **1.3.0** (MINOR, additive registry entries plus one clarifying sentence in Principle VI). |
| **II. Configuration-Driven Design** | Satisfied | Every SAC hyperparameter is a `HyperParamsBase` field reachable from YAML and participating in grid expansion and the directory signature. No `ParameterValue` extension: all fields are `float`/`int`. |
| **III. Gymnasium-First Integration** | Satisfied | Environments still come from `EnvironmentFactory`. The asymmetric oracle is a test-local `gym.Env`, registered through Gymnasium's own registry so it is reached the same way as any other environment. |
| **IV. Module Separation** | Satisfied | New code lives under `models/`. `run/`, `supervisor/`, `config/`, `reports/` are untouched. |
| **V. Modern Python & Code Quality** | Satisfied | 3.10+ unions, no `Any`, Google docstrings, ruff-clean at 120. |
| **VI. Extensibility & Discoverability** | Satisfied, and strengthened | Adding SAC touches only `models/sac/`. Rebuilding `deep_q_learning` is a refactor, not an algorithm addition — the spec's Constitution Impact section records the reasoning and the clarifying sentence the amendment carries. After this feature, DecQN and TD3 each touch only their own sub-package. |

**Root Class Registry entries touched**: `RLModel` (not modified; two subclasses added beneath it),
plus two new entries. `Runner` and `Supervisor` unmodified. The PR carries a "Constitution Impact"
section.

## Project Structure

### Documentation (this feature)

```text
specs/007-sac-continuous-control/
├── spec.md                  # revision 10, converged
├── plan.md                  # this file
├── research.md              # Phase 0: decisions, with the alternatives rejected
├── data-model.md            # Phase 1: class hierarchy, hook surface, hyperparameters, checkpoint
├── contracts/
│   └── model-contracts.md   # Phase 1: the hook surface as a contract, and the oracle environment
├── quickstart.md            # Phase 1: how to run each validation rung
├── checklists/requirements.md
└── tasks.md                 # Phase 2, by /speckit.tasks — not created here
```

### Source Code

```text
src/hercule/environnements/
├── __init__.py                 # ONE LINE ADDED: import the oracle module, so its gym.register runs.
│                               #      Placing the file under the package does not execute it.
└── oracle.py                   # NEW  AsymmetricOracleEnv (FR-029, SC-002)

src/hercule/models/
├── __init__.py                 # unchanged; concrete-only discovery already skips the new ABCs
├── off_policy/
│   └── __init__.py             # NEW  OffPolicyReplayModel, ExperienceReplayBuffer, Encoder,
│                               #      action mapping, episode loop, checkpoint assembly
├── deep_q_learning/
│   └── __init__.py             # REBUILT onto OffPolicyReplayModel; QNetwork keeps its construction
│                               #      order (see "The constraint that governs the refactor")
├── continuous_actor_critic/
│   └── __init__.py             # NEW  ContinuousActorCriticModel (ABC): twin critics, target
│                               #      critics, Polyak; no target actor
├── sac/
│   └── __init__.py             # NEW  SACModel, SACHyperParams, GaussianTanhActor, ContinuousCritic
├── td_models/, simple_q_learning/, simple_sarsa/, dummy/   # untouched
└── epoch_result.py             # untouched

tests/
├── models/
│   ├── test_golden_fixture.py      # NEW  FR-002 / SC-004, retired at closure
│   ├── test_action_mapping.py      # NEW  SC-003
│   ├── test_sac_objectives.py      # NEW  SC-010 to SC-013: hand-computed values and gradient edges
│   ├── test_sac_structure.py       # NEW  SC-014 target entropy, SC-015 pairwise-disjoint extractors,
│   │                               #      SC-017 deterministic evaluation
│   ├── test_polyak.py              # NEW  SC-016
│   ├── test_training_bars.py       # NEW  [slow]  SC-001's Pendulum bar and budget, SC-002's trained
│   │                               #      oracle bar and its zero-out-of-bounds check — the asserting
│   │                               #      tests. A YAML plus a bare `hercule learn` trains but asserts
│   │                               #      nothing and cannot fail when a bar is missed
│   ├── test_checkpoint_compat.py   # NEW  the existing-checkpoint load of User Story 2 scenario 3,
│   │                               #      through contract C4's migration table
│   ├── test_sac_persistence.py     # NEW  SC-005 (resume continues every adapted quantity), SC-008
│   └── ...                         # existing files unchanged
├── environnements/
│   └── test_asymmetric_oracle.py   # NEW  the oracle's five closed-form assertions (contract C2)
└── fixtures/
    ├── capture_baselines.py        # NEW  run ONCE from main, before any source change, to produce
    │                               #      both artifacts below. Under tests/ rather than at the repo
    │                               #      root, which CLAUDE.md reserves against ad-hoc scripts
    ├── golden/                     # NEW  committed fixture, deleted at feature closure
    └── checkpoints/                # NEW  two real pre-refactor model.json files, one per network
                                    #      branch. NOT deleted at closure: they outlive the fixture,
                                    #      because the migration path they test does

tests/supervisor/
└── test_space_gating.py        # EXTENDED  SC-009: SAC paired with a discrete-action environment

experiments/
├── sac_oracle.yaml             # NEW  SC-002, the oracle rung (FR-029 wants one config per rung)
├── sac_pendulum.yaml           # NEW  SC-001
├── sac_lunarlander.yaml        # NEW  SC-006
└── sac_car_racing.yaml         # NEW  SC-007, budget 700 epochs to match dq_car_racing.yaml
```

**Structure Decision**: single project, the existing `src/hercule` layout. Each new class gets its
own sub-package under `models/`, matching the convention every existing model follows and keeping
model discovery's filesystem scan meaningful. The two abstract classes live in sub-packages of their
own rather than inside a concrete model's, so neither concrete model imports the other.

## Ordering

The sequence is forced in two places and free elsewhere.

1. **Capture both pre-refactor artifacts first**, from `main` as it stands, before a single line
   moves: the golden fixture (reward series and order-keyed weight hashes) *and* two real, loadable
   `model.json` files, one per network branch, for contract C4's compatibility test. Neither can be
   regenerated afterwards — that is the whole point of both — and capturing them late is the one
   mistake in this plan that cannot be undone. They go in different directories: the fixture is
   deleted at closure, the checkpoints outlive it.
2. **Extract the ancestor and rebuild DQN on it, then verify against the fixture**, before any SAC
   code exists. A failure here must be attributable to the extraction alone.
3. Then the continuous side: action mapping → actor and critic modules → `ContinuousActorCriticModel`
   → `SACModel` → objectives and their tests → persistence → configs and the numeric bars.

The oracle environment can be built at any point before the objective tests need it, and is a good
candidate for parallel work since it touches nothing else.

## Deviations from the specification: none

An earlier draft of this plan declared two, and adversarial review showed both were avoidable. They
are recorded because each replacement is a design decision in its own right, and because "we deviated
for a good reason" is the kind of claim that should have to survive an attempt to remove it.

| Withdrawn deviation | Why it was wrong | What replaced it |
|---|---|---|
| Give `_advance_exploration_state()` a no-op default instead of the roadmap's epsilon decay, and override it in the deep model | The roadmap's own default would raise on SAC's first training step, since SAC has no `epsilon`. That much the draft got right. What it got wrong was the *fix*: it removed the hook altogether, which review then showed makes the public `act(obs, training=True)` mutate epsilon | The hook is **kept**, renamed `_on_training_step()`, with a **no-op** default and a name that makes no claim about what a subclass advances. FR-003 forbids the ancestor *assuming an exploration scheme* and *requiring a hyperparameter a subclass lacks*; a no-op default does neither, so there is nothing to deviate from. The call site is the one the decay already occupies, so the deep model is untouched |
| Give `_sync_targets()` a no-op default and move the hard copy into the deep model | The justification again invoked FR-003, which governs exploration and says nothing about target synchronisation, while FR-007 affirmatively requires the ancestor to carry the deep model's schedule **as its default** | The ancestor's default body performs the hard copy on the interval `_target_sync_interval()` returns; that hook returns `None` — never — by default, and the deep model overrides it to return `target_update_frequency`. The ancestor owns the mechanism without naming a hyperparameter only one subclass declares |

Both replacements cost one indirection each and remove an argument from the plan. That is the right
trade: an argument has to be re-made every time someone reads the code, and an indirection does not.

## Complexity Tracking

| Choice | Why needed | Simpler alternative rejected because |
|---|---|---|
| Two abstract intermediates rather than one | The off-policy concerns (replay, stacking, episode loop) are shared with DecQN, which is family A and has no actor; the actor-critic concerns are not | A single `ContinuousModel` would force DecQN, when it arrives, either to inherit an actor it does not have or to duplicate the replay machinery |
| Three separate feature extractors | FR-025, and the roadmap's adjudicated decision: shared features between the twin estimators collapse the decorrelation that taking the lesser of two exists to provide | Sharing is measurably cheaper — it is the only route to the existing 50 MB guard — and is exactly why FR-025 is a requirement with a direct parameter-disjointness assertion rather than a preference |
| A purpose-built oracle environment | `Pendulum-v1` is one-dimensional and symmetric, so every per-dimension and asymmetry defect is invisible there | Using `LunarLanderContinuous` as the multi-dimensional oracle instead: its optimum is not known in closed form, so SC-002's "90% of the analytic optimum" would have no referent |

## Plan review history

The plan is reviewed by the same method as the specification: four models of distinct lineages,
Anthropic excluded, given the plan, the closed specification **and the actual source it proposes to
refactor**, so that every claim about the existing code is checkable rather than taken on trust.

**Round 1** — four reviewers, one returning an empty completion. The three that answered returned
NOT CONVERGED with, between them, thirteen distinct findings. All thirteen adopted. The three
load-bearing claims about the existing code were independently verified as **correct** by two
reviewers each: the within-network construction order, that the dummy shape probe draws nothing, and
that nothing between the seeding call and network construction consumes torch's RNG.

| Finding | Verified how | Response |
|---|---|---|
| **The oracle environment was unsatisfiable by a correct implementation** (3 of 3). Clamping its one-sided dimension scored 39.58 against a 36.0 bar — and no tuning could fix it, since any distribution on an interval of width 0.25 has variance at most 0.0625 | Arithmetic, computed independently by all three | Adopted, and the environment resized: dimension 1 now spans `[0, 4]` with its target in `[2, 4]`, episodes are 50 steps. Contract C2 carries the five closed-form rows and the general condition, so a future change to the numbers can be checked rather than guessed |
| The golden fixture covered only one of the two network branches: `CartPole-v1` is a 1-D `Box` and `FrozenLake-v1` is `Discrete`, whose `_single_frame_shape()` returns `(1,)` — **both take `_build_mlp`** (2 of 3) | Read against `_single_frame_shape` and `QNetwork.__init__` | Adopted — a third, shaped-observation configuration covers the convolutional branch, which has five parameterised layers to the vector branch's three |
| The deep model **constructs** its target network rather than copying it, consuming a second full RNG sequence. The plan pinned order *within* a network but never forbade replacing that construction with the copy it prescribes for the continuous family's delayed copies (2 of 3) | Read against `_build_from_spaces` | Adopted — stated as the third consequence of the refactor constraint, and as obligation O1's failure mode |
| The checkpoint section described a format the code does not have — one payload with `networks`/`optimizers`/`extra` — where the real format is `networks_b64`, `optimizer_state_b64`, `rng_state_b64` and top-level fields, **and** has a legacy import path the plan never mentioned (1 of 3) | Read against `_export`/`_import` | Adopted — contract C4 states format compatibility as an obligation, and the existing-checkpoint load gets a test, which the fixture cannot provide since a fixture never exercises loading |
| Checkpoint assembly moved into the ancestor would drop `frame_stack` and `observation_shape`, which the deep model carries **because `hercule play` configures with defaults** and rebuilds the network from them (1 of 3) | Read against `_export`'s own comment | Adopted — obligation O5 and its note |
| `SC-014` and `SC-015` had no test home; `SC-009` and `SC-017` were assigned only to "the suite" with no artifact (2 of 3) | Read back against the Project Structure | Adopted — `test_sac_structure.py`, and `test_space_gating.py` named explicitly |
| FR-029 wants a ready-to-run config per validation rung; the oracle rung had none, and could not have one while the environment lived under `tests/` (1 of 3) | Read against FR-029 | Adopted — the oracle moves to `src/hercule/environnements/oracle.py`, registered on import, and gets `experiments/sac_oracle.yaml` |
| The plan used one optimizer spanning both estimators; the specification's arithmetic describes **four** (1 of 3) | Read against Measured Baselines | Adopted — one optimizer per trained network plus the temperature's. Numerically equivalent for Adam, but the checkpoint layout is what the specification describes |
| `frame_stack` default given as `1` and labelled "as DQN's": the deep model's default is `0`, and it counts **previous** frames, so the plan's value silently stacked two observations and changed every network's input shape (1 of 3) | Read against the field definition | Adopted |
| Caching `_action_low`/`_action_high` unconditionally breaks the rebuilt deep model, whose action space is `Discrete` and has neither (1 of 3) | Elementary | Adopted — guarantee G8 |
| `SACModel`'s `model_name`, `hyperparams_class` and `supported_spaces` were never declared, and a `supported_spaces` of string pairs would register fine and then reject every continuous environment (1 of 3) | Read against `_is_registrable` and `supports_environment` | Adopted — the declarations are spelled out, with the failure mode |
| The temperature's optimizer had no home (1 of 3) | Read back | Adopted — it and `_log_alpha` belong to `SACModel`, not to the shared family-B abstraction, which a deterministic-actor sibling would otherwise inherit unused |
| `_ready_to_update` never mentioned `learning_starts`, so SAC could take gradient steps during its warmup (1 of 3) | Read back | Adopted — SAC overrides it with the additional conjunct |
| **Both declared deviations from the specification were unjustified** (2 of 3, one each) | FR-003 governs exploration and is *stronger* than the draft used; FR-007 affirmatively requires the ancestor to carry the deep model's schedule | Adopted — both withdrawn, replaced as described above. The plan now deviates from the specification in no place |

The first finding is the one worth carrying forward: it is the same failure mode the specification hit
in its own rounds 6, 8 and 9 — a criterion, or here an environment, that a **correct** implementation
cannot satisfy. Three rounds of that in the spec and one more here suggests it is not an accident of
one document but a standing hazard of writing falsifiable criteria at all: it is much easier to check
that a bad implementation fails than that a good one passes.

**Round 2** — two CONVERGED (Grok, Gemini), one NOT CONVERGED with five findings, one empty
completion. All five adopted. Both reviewers who converged recomputed the oracle's five rows
independently and confirmed them, as did the dissenting one; the environment is now sound.

| Finding | Verified how | Response |
|---|---|---|
| **Round 1's own fix broke something.** Removing the per-step hook and decaying epsilon inside `_select_action` means the public `act(obs, training=True)`, called outside the episode loop, now mutates epsilon — where today it does not. The golden fixture drives `run_epoch()` and would never see it | Read against the current `act()` and `run_epoch()` | Adopted, and the round-1 finding it came from **partly reversed**: re-reading FR-003, it forbids the ancestor *assuming an exploration scheme* and *requiring a hyperparameter a subclass does not use*, and a no-op default does neither. The hook returns as `_on_training_step()` — no-op default, no exploration in its name — at the call site the decay already occupies. `act()` stays non-mutating (guarantee G9) |
| `_sync_targets()`'s default body was required to hard-copy "every declared target pair", and no hook declared any. `_networks()` returns names and modules; it does not say which live network each delayed copy shadows | Read back | Adopted — `_target_pairs()`, defaulting to empty. Without it the default body needs a naming convention over `_networks()`, which would be an undocumented contract |
| The oracle module would never be imported. Placing a file under a package does not execute it, so its `gym.register` would not run and the config naming it would fail in a fresh process | Elementary, and load-bearing | Adopted — `environnements/__init__.py` imports it, stated as the one-line change it is |
| SC-001 and SC-002's *trained* halves had no asserting test. A YAML plus `hercule learn` trains and asserts nothing, so a missed bar fails nothing | Read back against the Project Structure | Adopted — `test_training_bars.py`, marked slow. Only SC-006 and SC-007, which need hours, stay recorded experiments |
| C2's general condition said a policy clears the bar "exactly when `p < 0.1`"; SC-002 says *at least* 90%, so the bound is inclusive | Elementary | Adopted |

The first row is the fourth time in this feature that a fix adopted from a review round turned out
to be a defect, and the second time the right response was to partly reverse the earlier finding
rather than patch its consequence. The lesson that generalises: when a reviewer's argument rests on
a requirement's wording, re-read the requirement rather than the argument.

**Round 3** — two CONVERGED (Grok, Gemini), two NOT CONVERGED with six findings between them, three
of which the two dissenters found independently. All adopted. Both dissenters recomputed the oracle's
five rows from scratch rather than reading the table, making three independent confirmations across
three rounds; one also re-verified every load-bearing claim about the existing source and found them
correct.

| Finding | Verified how | Response |
|---|---|---|
| **The Deviations table still prescribed the design round 2 withdrew** — "the hook is removed, the deep model decays epsilon in `_select_action`" — with the justification round 2 falsified, while the rest of the plan said the opposite (2 of 2, independently) | Read back against the round-2 history row | Adopted. This is an editing failure, not a design one, and the worst kind: the section whose stated purpose is to record each decision was the section left describing the rejected one. An implementer reading it would have rebuilt the mutating `act()` |
| **G9's "never mutate model state" is violated by a correct implementation** (2 of 2, independently). `predict()` must advance the frame history — that is what observation stacking *is*, and `hercule play` depends on it — and a stochastic `act()` consumes the random streams by definition | Read against the existing `predict()` and `RLModel.predict`'s docstring | Adopted — G9 now forbids mutating *adaptive* state (hyperparameters, schedules, temperature, counters) and explicitly permits the frame history and the random streams. Round 2's repair was right about the problem and over-general about the cure |
| **The ancestor owns `configure()` but no hook constructs the optimizers.** `_optimizers()` enumerates already-built ones for checkpointing; it cannot create anything, so the hook surface could not actually be implemented as described (1 of 2) | Read back against the hook table | Adopted — `_build_optimizers()`, abstract, called immediately after `_build_networks()` |
| **The checkpoint contract contradicted itself.** It claimed the format was unchanged *including* `format_version`, while restructuring `optimizer_state_b64` from a bare state into a name-keyed mapping — and the code's own rule is that the version bumps whenever `_export()`'s payload shape changes (1 of 2) | Read against `_CHECKPOINT_FORMAT_VERSION`'s docstring and `_export` | Adopted — the version goes to **3** and `_import` dispatches across 3, 2 and the pre-006 legacy form. Keeping one version across two payload shapes would make a v2 and a v3 file indistinguishable, which is what the field exists to prevent |
| **Ownership of the random streams was stated two incompatible ways**: obligation O5 told a subclass to export every non-module quantity it mutates, while the checkpoint table and G1 made the ancestor own all three streams. A subclass following O5 literally double-writes them (1 of 2) | Read back across G1, G7, O5 and the checkpoint table | Adopted — the ancestor owns and exports the streams and the counters under G7; `_extra_state()` is subclass-specific quantities plus the shape keys |
| A stale `FR-024` where `FR-025` was meant, in Complexity Tracking (1 of 2, non-blocking) | Read back | Adopted |

Two of these six are consequences of round 2's own fix — one left un-propagated, one over-generalised.
That is now the pattern rather than the exception: **five of this feature's review rounds have found
a defect in a fix adopted from the round before**. The practical conclusion, and the reason this plan
records its own history rather than presenting a clean face: after adopting a finding, the next thing
to check is not the next finding but every other place the adopted one touches.

**Round 4** — one CONVERGED, three NOT CONVERGED with five findings between them, three of them found
by two reviewers independently. All adopted. The round was run as a targeted question — "check the
six round-3 adoptions for incomplete propagation and over-generalisation specifically" — and three of
the five findings were exactly that, including one in the propagation of the fix for incomplete
propagation.

| Finding | Verified how | Response |
|---|---|---|
| **The encoder/head split renames every parameter key, and `load_state_dict` is strict by default** (1 of 4). A `state_dict()`'s keys are attribute paths; today's convolutional branch even registers each parameter twice, under `conv_layers.*` and again under `network.0.*`. Every `model.json` in `outputs/` would fail to load after a refactor that changes not one weight — and **the golden fixture cannot see it**, because it compares tensors and never opens a file | Read against `QNetwork._build_mlp` / `_build_cnn` and `_import`'s strict `load_state_dict` | Adopted — research R11 and contract C4: an explicit key-migration map applied to version 2 and the legacy form before loading, tested against a real pre-refactor checkpoint. The alternative, contorting the new modules to reproduce the old paths, was rejected: it carries a quirk of today's code into the ancestor every future model inherits |
| G9's closing sentence, "everything adaptive advances in `_on_training_step()`", is false of the plan's own design — SAC's temperature advances in `_update()` (2 of 4) | Read back against the hook table | Adopted |
| G9 still asserted that *both* `act()` and `predict()` advance the frame history and consume the random streams. Neither is true of both: the deep model's `act()` touches no frame history, and a deterministic `predict()` consumes nothing — which FR-014 and SC-017 positively require (1 of 4) | Read against the existing `act()` and `predict()` | Adopted — G9 now grants two *permissions*, each scoped to the call that needs it, instead of asserting a behaviour of both methods |
| `plan.md`'s Technical Context still said the checkpoint format version is "reused as-is", contradicting round 3's bump to 3 (2 of 4) | Read back | Adopted |
| The FR-024 → FR-025 correction was applied in Complexity Tracking and not in `research.md` R10 (1 of 4, non-blocking) | Read back | Adopted |

Two observations worth keeping. The first: the round explicitly hunting incomplete propagation found
two more instances of it, one of them in a fix whose whole content was a reference correction. The
second matters more for the implementation than for the process — **the golden fixture has a blind
spot with a name**. It certifies that the refactor computes the same numbers; it certifies nothing
about whether the artifacts those numbers are stored in can still be read. Every checkpoint-shaped
requirement needs its own test, and contract C4 is where they are enumerated.

**Round 5** — one CONVERGED, three NOT CONVERGED, converging on the same three findings. All adopted.

| Finding | Verified how | Response |
|---|---|---|
| **The golden fixture would have failed the refactor it exists to certify.** R11 renames every parameter path; a fixture whose hashes are keyed by parameter *name* breaks on the rename — for a reason that has nothing to do with the numbers, which is the most misleading way a test can break (1 of 4) | Read R8 against R11 | Adopted — the hashes are taken in `parameters()` **order**, which R11's table shows is stable across the rename. The two instruments are now cleanly separated: the fixture certifies the numbers, contract C4's load test certifies the names |
| **The migration test's input has no capture step and nowhere to live.** C4 claimed the pre-refactor checkpoint was "the same artifact the golden fixture is generated from" — it is not; the fixture is a reward series and hashes, not a loadable `model.json`. Worse, `tests/fixtures/golden/` is deleted at closure, and the migration path outlives it. And the pre-006 **legacy** form cannot be captured at all, because feature 006 replaced it (2 of 4) | Read against R8's own definition of the fixture and the closure step | Adopted — two real `model.json` files, one per network branch, captured in the same pre-refactor step and committed to a **separate** directory that survives closure; the legacy branch is tested against a hand-built minimal file, stated so nobody hunts for a capture step that cannot exist |
| **"The explicit migration map" was named four times and never written.** An implementer would have had to derive the new attribute paths themselves, mid-refactor, under a failing test (2 of 4) | Read back; one reviewer enumerated the real keys — twenty entries for ten tensors on the image branch, because of the double registration | Adopted — C4 now carries the table, old prefix to new, with the aliases explicitly dropped and the encoder's layer indices explained. Also recorded: parameter *enumeration order* is preserved by that table, which a v2 optimizer state depends on, since its moment buffers are keyed by parameter index |
| G9's closing sentence, on its third attempt, still over-generalised: it named `_on_training_step()` and `_update()` as the only sites where adaptive state advances, while the counters advance in the loop and the delayed copies in `_sync_targets()` (3 of 4) | Read against the loop pseudocode and G7 | Adopted — the guarantee now says only what `act()` and `predict()` may and may not do, and explicitly makes no claim about where mutation happens. Three drafts were spent trying to enumerate mutation sites; the fix was to stop trying |

The first two findings share a shape worth naming, because it is not the incomplete-propagation
pattern of rounds 2 to 4. **Both are instruments that would have reported success while the thing
they certify was broken** — a fixture that fails for the wrong reason, and a compatibility test whose
input does not exist. A plan is easier to check for what it *says* than for whether its evidence can
actually be produced, and five rounds of review reached that question only at the fifth.

**Round 6** — one CONVERGED, three NOT CONVERGED, all three naming the same first finding. All
adopted.

| Finding | Verified how | Response |
|---|---|---|
| **The runbook still said "the one thing to do before touching any code" and named only the fixture.** Round 5 added a second, equally irreversible capture and did not propagate it to the section whose entire job is to say what is irreversible (3 of 4) | Read back | Adopted — the section is now "the two things", with one script producing both and an explicit note about why the second exists |
| **The migration table's rows collide across branches.** On an image checkpoint `network.0.0.weight` matches both the vector rename prefix and the image drop rule; on a vector one `network.0.weight` matches both as well. Rename-first leaves an unexpected key on every image file, drop-first a missing key on every vector one — there is no flat order that is correct for both (1 of 4) | Read against the real key structure of both branches | Adopted — the branch is selected first, by the presence of any `conv_layers.` key, and only that branch's rows are applied. Without it the selector would have been invented mid-refactor under a failing test, which is exactly what writing the table down was supposed to prevent |
| **`data-model.md`'s checkpoint paragraph was stale on two counts**: it said the legacy path is "untouched" — but the legacy form's parameter keys are the old attribute paths too, so loading them verbatim would fail against the rebuilt module — and it said "one key's internal shape changes" when C4 now pins two changes, the second being the parameter rename it never mentions (1 of 4) | Read across C4 and the paragraph | Adopted |

Sixth round, sixth incomplete propagation — and this one landed in the runbook that gives the
irreversible instruction, which is the worst available place for it. Worth stating plainly, since the
pattern has now held for every round of this plan: **a fix is not finished when the defect it names is
gone; it is finished when every document that repeats the old claim has been found.** The failure is
never in the reasoning, always in the sweep.

**Round 7** — three CONVERGED (GLM, Grok, Gemini), one NOT CONVERGED with a single finding, adopted:
contract C4 still opened its payload section with "**One** payload shape does change" while the same
contract, two paragraphs later, pins a second one and `data-model.md` had already been corrected to
say two. Reconciled.

That is the seventh consecutive round of this plan to find an incomplete propagation, and it was
found in the round whose brief was, in so many words, "sweep for incomplete propagation". The pattern
is not going to be argued away, so it is recorded as a working conclusion instead: on this feature,
the probability that a fix is fully propagated on the first attempt is low enough that the sweep
should be treated as a separate step with its own pass, not as the tail of the fix.

**Round 8** — unanimous CONVERGED across the three reviewers run, on top of Gemini's round-7
convergence. The plan is closed.

## Convergence

Eight rounds, four reviewers of distinct lineages, Anthropic excluded throughout because an Anthropic
model drafted this. Findings per round: 13, 5, 6, 5, 4, 3, 1, 0 — 37 in all, every one adopted; none
was rejected, which is itself a difference from the specification's rounds and says the plan started
weaker than the spec did.

Three results worth keeping, none of them about SAC:

1. **Every round found an incomplete propagation**, including the round whose brief was to sweep for
   incomplete propagation. On this evidence the sweep is a separate step needing its own pass, not
   the tail end of a fix.
2. **The two most valuable findings were about instruments, not designs.** The oracle environment
   could not detect the defect it existed for, and the golden fixture would have failed the refactor
   it existed to certify while being blind to the one failure mode — unreadable checkpoints — that
   the refactor actually threatened. A plan is easy to check for what it claims and hard to check for
   whether its evidence can be produced at all.
3. **Giving reviewers the source changed the review.** Three of the thirteen round-1 findings were
   claims about the existing code that were simply false, and could only be caught by reading it:
   that two fixture configurations covered both network branches when both took the same one, that
   `frame_stack` defaults to 1 when it defaults to 0 and counts something else, and that the deep
   model copies its target network when it constructs one.
