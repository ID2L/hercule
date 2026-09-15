# Feature Specification: SAC Continuous Actor-Critic on a Shared Off-Policy Ancestor

**Feature Branch**: `007-sac-continuous-control`
**Created**: 2026-09-15
**Status**: **CONVERGED** — revision 10, unanimous at round 10 after ten rounds of adversarial review
**Input**: User description: "SAC (Soft Actor-Critic, arXiv:1812.05905) — implement an actor-critic
model for continuous (`Box`) action spaces in Hercule, validated on Pendulum-v1, then
LunarLander continuous, then CarRacing-v3(continuous=True)"

## Context

Hercule can benchmark algorithms on environments whose action space is `Discrete`. Every model it
ships — tabular Q-learning, SARSA, DQN, the random baseline — selects an action by enumerating the
action set. On a `Box` action space that enumeration does not exist, so the whole catalogue is
unavailable on the class of environments continuous control is actually about.

`specs/ROADMAP-continuous-actions.md` establishes why, and splits the literature into two families:
make the maximisation computable (family A), or delegate it to a learned actor (family B). This
feature delivers **one best-in-class representative of family B, SAC**, and the shared off-policy
scaffolding both families need.

The roadmap sequences five phases. This feature collapses phases 1 and 4 into one deliverable and
defers phases 2 (DecQN) and 3 (TD3). The reason is a scope decision — one representative of family
B, chosen for reliability — not a claim that family A is already covered: `CarRacing(continuous=False)`
is a `Discrete` action space and is therefore not a family-A method at all, since family A is about
maximising over a `Box`.

**This document is consequently the sole carrier of the family-B contracts the roadmap had placed
in phase 3.** Every one of them is restated below rather than inherited by reference, because the
phase that would have written them down is not being executed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Benchmark a continuous-control algorithm from a config (Priority: P1)

A researcher writes a YAML config naming a continuous-action environment and the new model, with a
hyperparameter grid, and runs the same `hercule learn` command they already use. The framework
expands the grid, trains every variant, and writes each run into the deterministic output tree with
its metrics and weights — exactly as it does for a discrete environment today.

**Why this priority**: This is the feature. Without it the framework has no reach into continuous
control at all, and every other story is a refinement of this one.

**Independent Test**: Run `hercule learn` on a config pairing the new model with the primary
one-dimensional continuous environment and check the outcome against **SC-001's numeric bar**. An
improving learning curve is deliberately *not* the test: the roadmap withdrew that criterion because
a policy driving its pedals out of bounds still trains and still produces a plausible curve.

**Acceptance Scenarios**:

1. **Given** a config naming a continuous-action environment and the new model with a two-value
   hyperparameter list, **When** `hercule learn` runs, **Then** two independent run directories are
   created under distinct hyperparameter signatures, each holding `environment.json`, `model.json`
   and `run_info.json`.
2. **Given** a completed training run, **When** the same config is run again with a higher epoch
   ceiling, **Then** training continues from the recorded epoch rather than restarting, and the
   quantities that were being adapted during training continue from their stored values rather than
   reverting to their initial ones.
3. **Given** a config that pairs the new model with a discrete-action environment, **When**
   `hercule learn` runs, **Then** that combination is skipped with a message naming both the
   environment's space kind and the kinds the model accepts, and every other combination in the same
   config still runs to completion.

---

### User Story 2 - Existing results and models keep their meaning (Priority: P1)

A researcher who already has trained runs and reports in `outputs/` upgrades to this version. The
existing models behave identically: the same configuration at the same seed produces the same
numbers, and previously written model files still load.

**Why this priority**: The scaffolding this feature introduces is extracted from the existing deep
Q-learning model. A refactor that silently perturbs it would invalidate every benchmark the project
has produced — and, because a wrong reinforcement-learning implementation still produces a plausible
learning curve, would do so with no visible symptom. It is P1 alongside Story 1 because it is the
condition under which Story 1's numbers can be trusted at all.

**Independent Test**: Apply the roadmap's adjudicated method — a **golden fixture** (episode reward
series plus weight hashes) generated from the current, pre-refactor code and committed **before** any
refactoring begins; the refactored model must reproduce it bit-identically. Running the old and new
code side by side is not an available method: two versions of the same class cannot coexist in one
process.

**Acceptance Scenarios**:

1. **Given** a golden fixture captured from the pre-refactor code at a fixed seed, **When** the
   refactored model is run on the same configuration and seed, **Then** the per-episode reward series
   and the weight hashes match the fixture exactly.
2. **Given** the refactor is complete and the fixture has served its purpose, **When** the feature
   closes, **Then** the fixture is **retired** and replaced by a determinism property test — same
   seed twice gives identical results, different seeds give different results — which stores no
   recorded expectation and therefore survives later deliberate behaviour changes without
   regeneration.
3. **Given** a model file written by a previous version, **When** it is loaded, **Then** it loads
   successfully and `hercule play` renders it.
4. **Given** the existing tabular and random-baseline models, **When** the test suite runs, **Then**
   their behaviour is unchanged.

---

### User Story 3 - Inspect, replay and compare a continuous agent (Priority: P2)

A researcher who has trained continuous-action agents replays one interactively to watch what it
learned, and generates a comparative report over a hyperparameter grid to see which settings drive
performance.

**Why this priority**: Training that cannot be inspected or compared does not close the loop the
framework exists for — but it is only reachable once Story 1 works.

**Independent Test**: Run `hercule play` on a trained continuous agent, then `hercule report` on the
parent output directory, and confirm a rendered report is produced for the group.

**Acceptance Scenarios**:

1. **Given** a trained continuous agent's `model.json` and `environment.json`, **When**
   `hercule play` runs, **Then** the environment renders and the agent acts within the environment's
   declared action bounds for the whole episode.
2. **Given** an output directory holding a grid of continuous-action runs, **When**
   `hercule report` runs, **Then** a comparative report is generated and ranks the runs by
   performance.

---

### User Story 4 - Reach an image-observation continuous environment (Priority: P3)

A researcher points the new model at a continuous-action environment whose observations are images
rather than a small state vector, and gets a run that trains rather than failing on shape or memory.

**Why this priority**: This is the project's stated target environment, but it is the hardest rung
and the one whose outcome is least controllable — image-based continuous control is a research
problem in its own right. Treating it as P3 keeps the feature deliverable even if the score achieved
there is modest.

**Independent Test**: Train on the image-observation continuous environment for the epoch budget
stated in SC-007 and compare the mean test reward against the random-policy baseline **measured and
recorded in this specification**.

**Acceptance Scenarios**:

1. **Given** a config naming the image-observation continuous environment, **When** training runs,
   **Then** it completes without shape or memory errors and writes a checkpoint within the
   model-specific size bar of SC-008.
2. **Given** that trained agent, **When** it is evaluated at the epoch stated in SC-007, **Then** its
   mean test reward exceeds the recorded random-policy baseline by the margin stated there.

---

### Edge Cases

- **Asymmetric per-dimension bounds.** The target environment's three action dimensions do not share
  a range: steering spans a symmetric interval while throttle and brake are one-sided. A policy that
  emits values in a single normalised range and forwards them unchanged drives the one-sided
  dimensions out of bounds. The environment does not reject such an action — it simply behaves as if
  the pedal were never pressed, and the run still produces a plausible reward curve. Mapping must
  therefore be per dimension, and must be covered by a test against the real environment's bounds.
- **Density measured in the wrong coordinate system.** Rescaling the policy's output to the
  environment's bounds *before* computing its density adds a per-environment constant to the measured
  entropy — one direction on an environment whose bounds are wider than the normalised range, the
  other on one whose bounds are narrower. The adapted temperature then converges against a shifted
  target. Nothing fails; the agent merely explores too much or too little, differently on every
  environment, which is precisely the property automatic temperature adjustment was chosen to remove.
- **Time-limit truncation.** The primary validation environment never terminates; it is cut off by a
  time limit on every single episode. An implementation that treats being cut off as reaching a
  terminal state trains against a wrong target on 100% of its episodes.
- **One-dimensional action space.** The primary validation environment has a single action
  dimension, so any quantity defined per dimension degenerates there. A second, multi-dimensional
  validation environment with deliberately asymmetric bounds is required as the oracle for anything
  that only misbehaves once more than one dimension exists.
- **A learning target formed from the wrong estimators.** Bootstrapping from the live value
  estimators rather than their delayed copies, or evaluating them at the action stored in the replay
  history rather than at one resampled from the current policy, both leave training running and a
  curve moving. The first is a feedback loop; the second quietly turns policy improvement into policy
  evaluation.
- **Evaluation versus training behaviour.** The policy is stochastic during training; during testing
  it must act deterministically. A model that samples during evaluation reports noisier and lower
  scores than the policy it actually learned.
- **Delayed copies updated on the wrong clock.** Synchronising the delayed copies once per
  environment step rather than once per gradient step changes their lag by the ratio between the two,
  which is a hyperparameter. Nothing fails; the lag is simply not what was configured.
- **Resume with no stored replay history.** Replay contents are deliberately not persisted, so a
  resumed run restarts with an empty history. The resulting discontinuity in the learning curve is
  expected behaviour and must be documented rather than discovered.
- **Checkpoint volume.** This model carries five networks and four optimizers where the existing
  deep model carries two networks and one optimizer. On the image-observation environment that is
  roughly a threefold difference — ~129 MB against 44.69 MB measured — which is a change in operating
  cost rather than a marginal one, and is the reason the existing guard cannot be reused. See SC-008.
- **Space pairing refusal.** Requesting this model on a discrete-action environment must be refused
  before training starts, with a message naming both sides, and must not abort the other
  combinations in the same config.

## Requirements *(mandatory)*

### Functional Requirements

#### Shared off-policy scaffolding

- **FR-001**: The framework MUST provide a shared abstraction for off-policy, replay-based models
  that owns the concerns common to them: observation history and rescaling, the replay store, device
  and seeding, the episode loop, the mapping between the policy's own action representation and the
  environment's, and checkpoint assembly.
- **FR-002**: The existing deep Q-learning model MUST be rebuilt on that abstraction with **no
  change to its observable behaviour**, verified against a committed golden fixture captured from the
  pre-refactor code, and that fixture MUST be retired into a determinism property test when the
  feature closes.
- **FR-003**: Exploration MUST be owned entirely by the per-algorithm action-selection step. The
  shared abstraction MUST NOT assume any particular exploration scheme, and MUST NOT require an
  algorithm to declare hyperparameters belonging to a scheme it does not use.
- **FR-004**: The replay store MUST accept a multi-dimensional action per transition, not only a
  single scalar.
- **FR-005**: The separation between reaching a terminal state and being cut off by a time limit,
  established by the preceding feature, MUST be preserved through the refactor: only reaching a
  terminal state suppresses the bootstrap.
- **FR-006**: The replay store MUST hold the action in the policy's own coordinates, and the action
  actually executed in the environment MUST be reconstructible from it plus the environment's bounds.
  The two representations MUST NOT both be persisted, since a second copy is a channel through which
  they can diverge.
- **FR-007**: Synchronisation of delayed network copies MUST remain a shared step evaluated **once
  per environment step**, carrying the existing deep model's schedule as its default, so that its
  synchronisation interval stays independent of how often a gradient step is taken. The continuous
  actor-critic family MUST disable that step and instead update its delayed copies by gradual
  averaging **once per gradient step**. This distinction is load-bearing: the two clocks differ
  whenever a gradient step is not taken on every environment step, which the project's own
  image-observation config already configures.

  The averaging MUST be **gradual**, each delayed copy moving toward its live counterpart by a
  configurable fraction and never being replaced by it outright. Both halves are required and the
  clock alone is not enough: a hard copy performed once per gradient step satisfies the schedule
  while making each delayed copy identical to its live counterpart from the first step onward, at
  which point FR-015's target — "the lesser of the two **delayed** copies" — is numerically the
  lesser of the two live ones. That is the first wrong form FR-015 enumerates, reached without
  violating a single word of it.

#### Action representation

- **FR-008**: The framework MUST map a policy's normalised action onto each action dimension's own
  declared bounds, independently per dimension, such that the lowest normalised value maps exactly
  to that dimension's lower bound and the highest exactly to its upper bound.
- **FR-009**: The methods used for acting and for prediction MUST return an action expressed in the
  environment's coordinates, so that interactive replay and evaluation drive the environment
  correctly.
- **FR-010**: Every action the model submits to an environment MUST lie within that environment's
  declared bounds, in training and in evaluation.
- **FR-011**: The policy, its density, and the value estimators MUST all operate in the **normalised**
  action coordinates. Rescaling to the environment's bounds MUST happen only at the boundary where
  the environment is stepped. The change-of-variables term contributed by that rescaling MUST NOT
  enter the density against which the temperature is adapted. This is what makes the target entropy
  of FR-017 correct as stated, independently of each environment's bounds.
- **FR-012**: Each value estimator MUST take the action as an **input** and return a single value,
  rather than mapping an observation to one value per action. The existing deep model's network maps
  an observation to a vector indexed by action and therefore cannot serve as a continuous value
  estimator; what the two families share is the feature extractor, not the head.

#### The continuous actor-critic algorithm

- **FR-013**: The framework MUST provide a model implementing Soft Actor-Critic with automatic
  temperature adjustment, as described in arXiv:1812.05905 — **not** the fixed-temperature variant,
  whose temperature must be retuned for each environment's reward scale and is therefore unusable in
  a multi-environment benchmark.
- **FR-014**: The policy MUST be stochastic during training and deterministic during evaluation.
- **FR-015**: The learning target MUST be exactly: the observed reward, plus — **suppressed if and
  only if the successor observation is terminal**, never when the episode was merely cut off by a
  time limit — the discount factor times the quantity formed by taking the **lesser** of the two
  **delayed copies** of the value estimators, evaluated at an action **resampled from the current
  policy** at the successor observation, **minus** the temperature times the log-density of that
  resampled action under the current policy. Every clause is load-bearing and each is separately
  pinned because each has a plausible-looking wrong form: the live estimators instead of the delayed
  copies (a feedback loop); the greater instead of the lesser, or their mean (the overestimation this
  construction exists to control); the action stored in the replay history instead of a resampled one
  (policy evaluation instead of policy improvement); an entropy term added instead of subtracted, or
  omitted (an agent rewarded for being predictable, which converges to a deterministic policy while
  the temperature adapts in the wrong direction); and the bootstrap suppressed on a time-limit
  cut-off as well as on a terminal successor (the defect the preceding feature removed, which every
  episode of the primary validation environment would re-trigger). All five still train and still
  produce a curve.

  The target MUST additionally be a **constant for the purpose of learning**: no gradient may flow
  out of it into the actor's parameters, the temperature, or the delayed copies. This is a property
  of how the target is computed, not of its value — detaching it changes no number, so every
  value-level check above passes identically either way, and the damage appears only as spurious
  gradients deposited at the optimizer step. The natural implementation invites the defect precisely
  because FR-022 *requires* the actor's own sample to stay differentiable: one sampler reused for
  both, with nothing detached on the target side, makes the actor optimise against its own target
  and the temperature against a quantity it helped produce.
- **FR-016**: The squashing applied to bound the policy's output MUST be accompanied by its
  corresponding density correction. This requirement is stated explicitly because omitting the
  correction leaves training running and still improving, merely to a worse policy — so only a
  dedicated test distinguishes correct from incorrect.
- **FR-017**: The target entropy the temperature is adapted against MUST be exactly the **negative
  of the number of action dimensions**. A value merely "derived from" the dimension count is not
  sufficient: a wrong constant, or a wrong sign, still produces a moving temperature and a plausible
  curve.
- **FR-018**: Action selection MUST begin with a bounded phase of uniformly random actions before
  the learned policy takes over, the length of which is a hyperparameter.
- **FR-019**: The family-B abstraction MUST carry delayed copies of the **two value estimators
  only**. It MUST NOT carry a delayed copy of the actor: that construct exists to be smoothed in the
  deterministic-actor algorithm this feature defers, and here it would be a network trained, averaged,
  checkpointed and never read.
- **FR-020**: The model MUST declare the observation and action space kinds it accepts — image or
  vector observations, continuous actions only — so that the orchestrator refuses an unsupported
  pairing before configuring it.
- **FR-021**: Every hyperparameter of the model MUST be expressible in the YAML config and MUST
  participate in the existing grid-expansion and directory-signature mechanisms.
- **FR-022**: The actor's objective MUST be to maximise the entropy-regularised value: the **lesser
  of the two live estimators** — the live ones here, not the delayed copies, which are a target-side
  construct — evaluated at an action drawn from the current policy in a way that keeps the sample
  differentiable with respect to the policy's parameters, **minus** the temperature times that
  action's log-density. Sampling in a way that breaks that differentiability, omitting the entropy
  term, or using one estimator instead of the lesser of two, each still trains and still produces a
  curve, and none is detected by FR-015's criteria, which govern the target side only.

  The actor's objective MUST likewise be **confined to the actor's parameters**: within it, the two
  live estimators and the temperature are evaluated, not trained, so no gradient may flow out of it
  onto their parameters. This is the mirror of FR-015's constant-for-learning clause, and it is
  stated because the natural implementation violates it — summing the three objectives and taking
  one backward pass produces values that are bit-identical to the correct ones and two distinct
  silent defects. The estimators are stepped to *raise* the value the actor is chasing, which
  reintroduces the overestimation the lesser-of-two construction exists to control, and does so
  identically on both estimators, so neither that construction nor FR-025's disjoint extractors can
  see it. And the gradient the actor's objective deposits on the temperature exactly cancels the
  log-density term of the temperature's own objective, leaving the temperature driven by the target
  entropy alone — a countdown decoupled from the policy's measured entropy, which silently degrades
  FR-013's headline property into the fixed-temperature variant that requirement rejects.
- **FR-023**: **Both** live value estimators MUST be trained, each by regression of its own output
  toward the target of FR-015, evaluated at the observation and the action **as stored in the replay
  history** — not at an action resampled from the current policy, which is the target side's
  construct and belongs only there. Training one estimator and leaving the other to drift, or
  regressing at a resampled action, each leaves the target, the actor objective, the temperature
  objective, the disjointness assertions and every gradient-isolation assertion passing unchanged.
  This requirement exists because the document pinned what the estimators are *used for* three times
  over before it pinned what they are *trained on*.
- **FR-024**: The temperature MUST be **learned by gradient descent**, not driven by a hand-written
  controller, on an objective whose gradient with respect to the temperature is the negative of the
  sum of the policy's log-density at the sampled action and the target entropy of FR-017. The
  log-density MUST be treated as a constant in that objective: it depends on the policy's parameters,
  and letting the temperature's gradient flow back into them makes the actor optimise against its own
  exploration schedule. The quantity actually optimised MUST be the temperature's **logarithm**, so
  that the temperature stays strictly positive for any step size — an unconstrained temperature that
  crosses zero inverts the sign of the entropy term in FR-015 and FR-022 without any error being
  raised. The resulting behaviour, which is what the criterion checks end to end, is that the
  temperature **rises when the policy's measured entropy falls below the target** and **falls when it
  rises above**. The direction is stated as well as the objective because the opposite sign produces
  a temperature that moves, a policy that trains, and a curve that rises — while exploration collapses
  or diverges. It is the same class of defect as FR-015's, on the one remaining learned quantity.
- **FR-025**: The actor and each of the two value estimators MUST have their **own** feature
  extractor — three extractors, pairwise disjoint, no parameter shared between any two of them. The
  two value estimators in particular MUST NOT share one: shared features collapse the decorrelation
  that taking the lesser of the two exists to provide, which would silently defeat FR-015 while every
  other requirement and every checkpoint-size bar still passed. Actor-to-estimator sharing has a
  different failure mode — representation interference between two objectives pulling in different
  directions — and is forbidden by the same roadmap decision. This is stated as a requirement rather
  than left in Assumptions precisely because any sharing makes the checkpoint *smaller*, so no size
  criterion can catch it.

#### Persistence and lifecycle

- **FR-026**: A checkpoint MUST carry every network including the delayed copies, every optimizer's
  state, the adapted temperature, and **all three** random-number-generator streams the project
  seeds — the tensor library's, the standard library's, and the model's own numerical generator —
  such that a resumed run continues rather than restarting those quantities.
- **FR-027**: The model MUST be loadable from a stored model file for interactive replay.
- **FR-028**: Replay history is explicitly NOT persisted; the resulting discontinuity on resume MUST
  be documented where a user will encounter it.

#### Configuration and reporting

- **FR-029**: The feature MUST ship one ready-to-run config per validation rung, under
  `experiments/`.
- **FR-030**: Runs produced by the new model MUST be consumable by the existing report generation
  without modification to it.

### Key Entities

- **Off-policy replay model**: The shared abstraction underlying every model that learns from a
  stored history of transitions. Owns the episode loop, the replay store, observation handling,
  seeding, checkpoint assembly, and the per-environment-step synchronisation step of FR-007 with its
  existing default schedule; delegates network construction, action selection and the learning update
  to the concrete algorithm.
- **Continuous actor-critic model**: The shared abstraction for family-B algorithms — an actor, two
  value estimators, and delayed copies **of those two estimators only**, gradually averaged once per
  gradient step. There is deliberately no delayed copy of the actor (FR-019).
- **Soft actor-critic model**: The concrete algorithm — a stochastic bounded policy in normalised
  coordinates, an entropy term in the learning target, and an adapted temperature.
- **Action mapping**: The correspondence between the policy's normalised action representation and
  the environment's per-dimension bounds, applied only where the environment is stepped.
- **Asymmetric two-dimensional oracle environment**: A deterministic test environment introduced by
  this feature, with two action dimensions whose bounds deliberately differ — one symmetric, one
  one-sided — and whose optimal return is computable in closed form. It exists because the primary
  validation environment is one-dimensional and symmetric, so no per-dimension or asymmetry defect
  can surface there. Three properties are required of it, and without them it certifies nothing:
  its return MUST depend on **both** action dimensions in a way that varies with the observation, so
  that no single constant on either dimension — not even the best one — can reach the optimum, and a
  policy that learns only one dimension therefore cannot approach it; the optimal action on the
  one-sided dimension MUST lie at a point that is **unreachable under a mis-mapped policy** — at or
  near that dimension's bound, which a policy emitting normalised values forwarded unscaled would
  never produce; and a constant-action policy MUST score materially below the optimum, so that an
  agent which has learned nothing cannot pass. The first property is stated as observation-dependent
  variation rather than as plain dependence because a return that depends on a dimension but whose
  optimum on it is the same in every state is still reachable by a policy that never learned it.
- **Checkpoint**: The complete on-disk state of a model, sufficient to resume training with no
  silent reinitialisation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On `Pendulum-v1`, the trained agent achieves a mean test reward **strictly above -200
  over 20 evaluation episodes**, within a training budget of **500 episodes**. The bar is the one
  recorded in the project roadmap; the budget is stated here so that a run which never reaches the
  bar can be declared failed.
- **SC-002**: On the asymmetric two-dimensional oracle environment, within a training budget of
  **200 episodes**, the trained agent achieves a mean return over 20 evaluation episodes of at least
  **90% of that environment's analytically known optimal return**, and **zero** actions submitted
  across those episodes fall outside the declared bounds. The environment is introduced by this
  feature, so its learnability within that budget is a design constraint on it, not an assumption
  about the algorithm. The criterion is only meaningful if the environment satisfies the three
  properties required of it in Key Entities, so **all three** are themselves asserted: a test MUST
  show that the best constant-action policy scores below the 90% bar; that a policy whose normalised
  output is forwarded to the environment unscaled also scores below it; and that, **for each
  dimension taken separately, the best policy that holds that dimension at a constant while choosing
  the other freely** scores below it. The third assertion is quantified over the *best* such
  constant, not over any constant: clamping a dimension to the value it takes in the optimal action
  leaves the optimum reachable, so a criterion phrased over an arbitrary constant would be
  unsatisfiable by a correct implementation rather than merely demanding. Phrased over the best
  constant it is satisfiable exactly when the environment's optimal action on that dimension varies
  with the observation, which Key Entities requires of it.
  Without the first two assertions a constant-reward environment would satisfy SC-002 with an agent
  that learned nothing; without the third, an environment whose second dimension barely moves the
  return would let an agent that never learned that dimension at all reach the bar — which is the one
  defect this environment exists to catch, since the primary validation environment is
  one-dimensional.
- **SC-003**: Both halves of the mapping oracle hold against the real `CarRacing-v3` action space:
  for every dimension, the lowest normalised action value maps exactly to that dimension's lower
  bound and the highest exactly to its upper bound; **and** the acting and prediction methods return
  the action in the environment's coordinates rather than in the policy's.
- **SC-004**: The existing deep Q-learning model, at a fixed seed, reproduces the committed golden
  fixture's episode reward series and weight hashes **exactly**; at feature closure that fixture is
  replaced by a determinism property test and is no longer present in the repository.
- **SC-005**: A run interrupted mid-training and resumed continues its adapted temperature, its
  optimizer state, the lag of its delayed value estimators, and all three random-number-generator
  streams from their stored values — none of these reverts to an initial value.
- **SC-006**: On `LunarLanderContinuous`, the trained agent achieves a mean reward of at least
  **200 over 100 evaluation episodes** — the score at which this environment is conventionally
  considered solved — within a training budget of **3000 episodes**. The 100-episode protocol is that
  convention's own, and is deliberately not the 20-episode protocol used for the roadmap's rungs.
- **SC-007**: On `CarRacing-v3(continuous=True)`, at **epoch 700** — the same training budget the
  project's existing deep-model config uses on this environment, so the two are directly comparable —
  the trained agent's mean test reward over 20 evaluation episodes is at least **+65.63**, that is,
  it exceeds the measured random-policy baseline of **-34.37** by at least **100 reward points**. The
  margin is roughly 80 times the standard error of that baseline (σ = 5.45 over 20 episodes, so
  SE ≈ 1.22), which is far outside anything noise can produce. If the run misses it, that is a
  finding to record, not a bar to lower after the fact.
- **SC-008**: A checkpoint written by the new model on the image-observation environment stays within
  **150 MB**, and the existing deep Q-learning model's **50 MB** guard remains green. The two numbers
  differ by construction and the second does not bound the first: the existing guard was measured on
  a model holding two networks and one optimizer, while this one holds five networks and four
  optimizers — three of them network-sized, the fourth the temperature's two floats. The arithmetic
  behind 150 MB is recorded in the Measured Baselines section.
- **SC-009**: Pairing the new model with a discrete-action environment is refused before training,
  with a message naming both the environment's space kind and the accepted kinds, while every other
  combination in the same config completes.
- **SC-010**: The density correction of FR-016 is verified against an independently computed
  reference value, computed in the **normalised** coordinates of FR-011, so that both removing the
  correction and measuring it in the environment's coordinates fail the suite.
- **SC-011**: The learning target of FR-015 is verified **clause by clause** against a hand-computed
  value on a small fixed batch, not only through end-to-end scores. Each of the five wrong forms
  FR-015 enumerates fails the suite independently: live estimators instead of delayed copies; the
  greater of the two, or their mean, instead of the lesser; the stored action instead of a resampled
  one; the entropy term added instead of subtracted, or omitted; and the bootstrap suppressed on a
  time-limit cut-off instead of only on a terminal successor. A sixth case is **not** value-level and
  must be checked differently: after a single learning step, the actor's parameters, the temperature
  and the delayed copies MUST have received no gradient from the target. A target left attached to
  the learning graph produces bit-identical values and passes all five cases above.
- **SC-012**: The actor objective of FR-022 and the temperature objective of FR-024 are verified the
  same way as the target: against hand-computed values on a small fixed batch. Each wrong form fails
  independently — an actor objective evaluated on the delayed copies instead of the live ones, or on
  one estimator instead of the lesser of two, or with the entropy term omitted; a sample that does
  not carry gradient back to the policy's parameters; a temperature objective whose gradient is not
  the one FR-024 states, or one that lets that gradient reach the policy's parameters; a temperature
  optimised directly rather than through its logarithm, shown by driving it to cross zero under a
  large step size; and a temperature that moves in the direction opposite to the one FR-024 states.
  One further case is **not** value-level, mirroring SC-011's sixth: after the actor's update, the
  two value estimators' parameters and the temperature MUST have received no gradient from the
  actor's objective. A single summed backward pass over the three objectives is value-identical and
  passes every case above. Without this criterion the entire actor and temperature side of the
  algorithm is pinned only by end-to-end scores, which the roadmap withdrew as evidence.
- **SC-013**: The value estimators' own objective of FR-023 is verified the same way: on a small
  fixed batch, **both** estimators move toward the target, and each does so at the observation and
  action **as stored in the replay history**. Training only one, or regressing at an action resampled
  from the current policy, leaves every other criterion in this document passing unchanged. With this
  criterion, every quantity the model learns — the two estimators, the actor, the temperature and the
  delayed copies — has both its objective and its gradient boundary pinned by a direct test. That set
  is exhaustive: there is nothing else in the model that is trained.
- **SC-014**: The target entropy of FR-017 is asserted to equal the negative of the action-space
  dimension, on an environment of at least two action dimensions so that the assertion distinguishes
  `-dim(A)` from `-1`. Without this, FR-017's own stated failure mode — a wrong constant or a wrong
  sign, producing a moving temperature and a plausible curve — has no falsifying test.
- **SC-015**: The actor and the two value estimators hold **pairwise disjoint** parameters — all
  three pairs asserted, not only the estimator-to-estimator one — directly rather than inferred from
  behaviour. Any sharing makes the checkpoint smaller and the curve no worse in the short run, so no
  size or performance criterion can detect a violation of FR-025; and an assertion covering only the
  two estimators would leave an actor sharing an extractor with one of them passing every criterion
  in this document.
- **SC-016**: The delayed copies advance once per gradient step and not once per environment step,
  verified under a configuration where the two clocks differ; **and** each advance is the configured
  gradual fraction of the way toward the live parameters, asserted against a hand-computed value, so
  that a hard copy on the right clock fails. The end-state form of the same check is available too,
  but only under a **controlled precondition**: starting from delayed copies equal to the live
  parameters, after one known non-zero live update and with the averaging fraction strictly between
  0 and 1, the delayed copies must differ from the live parameters. The preconditions are not
  decoration, and two successive attempts at this criterion were wrong without them. An unconditional
  "they must differ" fails for a *correct* implementation whenever the live update was zero, or when
  the two coincide numerically at convergence. Merely requiring that the live parameters moved is
  also not enough: with a live parameter moving from 0 to 1 while the delayed copy already sits at 1,
  every fraction leaves the delayed copy at 1, equal to the parameter that just moved.
- **SC-017**: Evaluation is deterministic: two evaluation episodes from the same observation sequence
  produce the same actions, while two training steps from it do not.
- **SC-018**: The full test suite passes, the linter reports no violations, and documentation
  generation still succeeds.

## Measured Baselines

Numbers measured for this specification, so that the criteria above are falsifiable at the moment
they are written rather than at the moment someone chooses to define them.

| Quantity | Value | How obtained |
|---|---|---|
| Random-policy baseline, `CarRacing-v3(continuous=True)` | **mean -34.37**, σ 5.45, min -46.09, max -25.65 | Measured 2026-09-15: 20 episodes, uniform sampling from the action space, seeds 1000-1019, `gymnasium` as pinned by this repository |
| Training budget on that environment | 700 epochs | The existing `experiments/dq_car_racing.yaml`, so SC-007 is comparable to the deep-model run already on record |
| Base64 state-dict size, one CarRacing convolutional network | **11.70 MB** | Feature 006 `tasks.md` T049, benchmark table (2,194,597 parameters); the `~11.2 MB` in `test_persistence.py`'s docstring is a rounding of the same figure |
| Existing deep-model checkpoint guard | 50 MB | `tests/models/test_persistence.py`; measured 44.69 MB trained |

**SC-008's 150 MB, derived rather than guessed.** With separate feature extractors — the roadmap's
adjudicated decision — this model holds five networks of roughly the measured single-network size
(actor, two value estimators, two delayed copies) and optimizer state for the three that are
trained. A fourth optimizer, the temperature's, holds two floats and does not enter the arithmetic.
An optimizer of the kind used here holds two buffers the size of the network it shadows. That gives
`5 x 11.70 + 3 x 2 x 11.70` ≈ **129 MB**, and 150 MB is that figure with headroom for the heads, which
differ between the actor and the value estimators. **The existing 50 MB guard is therefore not merely
inapplicable to this model — it is arithmetically unreachable by it.** Dropping one component does
not rescue it: the five networks alone are ~58.5 MB, and the three trained networks plus their
optimizer state are ~105 MB, so both would have to go — and both were added deliberately by the
preceding feature. The one remaining way to reach 50 MB is sharing feature extractors between the
two value estimators, which FR-025 forbids for a reason that has nothing to do with size.

**Consequence to carry into planning**: a 150 MB checkpoint written on every checkpoint interval is a
real operational cost, and the project's own documentation already warns that checkpoint frequency
must be tuned per experiment for the existing, far smaller deep model. The shipped image-observation
config must set that frequency accordingly.

## Assumptions

- **Base state**: `main` carries features 005 and 006. The enumerable space taxonomy with per-model
  declared support, the terminated/truncated separation, a live seed hyperparameter, and complete
  compact checkpoints all exist already and are treated as given, not re-specified here.
- **Roadmap collapse**: This feature covers roadmap phases 1 and 4. Phase 2 (DecQN) and phase 3
  (TD3) are deferred, not cancelled; their roadmap entries stand and will be renumbered when taken
  up. Because phase 3 is where the family-B abstraction was to be defined, **its contracts are
  restated here as requirements** — delayed copies of the value estimators only (FR-019), gradual
  averaging once per gradient step (FR-007), separate feature extractors with no sharing between the
  two estimators (FR-025), and value estimators that take the action as an input (FR-012) in the
  normalised coordinates of FR-011 — rather than inherited by reference from a phase nobody will
  execute. Each is a requirement, not an assumption: the first draft of this section left the
  extractor decision in Assumptions and claimed FR-011 carried the estimator-shape contract, which
  it did not.
- **Separate trunks**: Not an assumption — a requirement, FR-025, promoted out of this section in
  revision 3 because nothing in an assumption is testable. What remains an assumption is only that
  the resulting checkpoint size (Measured Baselines) is an acceptable price, and that sharing may be
  revisited later as a pure optimisation behind the same interface.
- **Image-observation ambition**: Competitive performance on the image-observation environment is
  **not** claimed. SC-007 is a beat-the-measured-baseline bar, not a state-of-the-art one.
- **Evaluation cost**: The budgets in SC-001, SC-002, SC-006 and SC-007 are long-running experiments,
  not default test-suite runs. SC-001 and SC-002 are cheap enough to run as marked slow tests;
  SC-006 and SC-007 are recorded experiments whose results are committed as part of the feature.
- **Reward-scale independence**: Automatic temperature adjustment, **together with FR-011's
  normalised density**, is assumed to remove the need for per-environment tuning of the entropy
  weighting. FR-011 is what makes this assumption defensible; without it the assumption is false by
  construction, since the shift is a function of each environment's own bounds.

## Dependencies

- `specs/ROADMAP-continuous-actions.md` — the specification of record for the phase sequencing, the
  hook surface, the action-mapping contract and the acceptance tiers.
- Feature 006 (`specs/006-continuous-actions-foundations/`) — complete and merged; this feature is
  unimplementable without it. Its `tasks.md` carries the measured checkpoint figures this
  specification's Measured Baselines section reuses.
- The Box2D environment family, already enabled as a dependency by feature 005.

## Constitution Impact

**Decided here, not deferred: a MINOR amendment is required, 1.2.0 → 1.3.0.**

The reasoning is a symmetry with what the registry already contains. The Root Class Registry lists
`TDModel` — an abstract intermediate class that factors behaviour common to the tabular algorithms
and is not itself instantiable. The two abstractions this feature introduces occupy exactly that
position for the off-policy and continuous actor-critic families, and the concrete models will depend
on their lifecycle contracts in the same way the tabular models depend on `TDModel`'s. Adding
registry entries is an addition, not a removal or redefinition, which the constitution's own
amendment procedure classifies as MINOR.

Two related questions, answered so they are not reopened during planning:

- **Is `RLModel` itself modified?** Not expected. The abstractions are inserted beneath it, and
  Principle I explicitly permits intermediate abstract classes provided they extend `RLModel`. If
  planning discovers that a hook must be added to `RLModel` after all, that is a separate amendment
  and must be raised then.
- **Does rebuilding `deep_q_learning` violate Principle VI?** Principle VI reads: "Adding a new RL
  algorithm MUST NOT require modifying any existing file outside the new algorithm's sub-package."
  The operative word is *require*. Adding SAC does not require it: SAC could be written standalone
  in its own sub-package, duplicating the scaffolding, and the principle would be satisfied. The
  rebuild is chosen to avoid that duplication, and after it the principle holds *more* strongly than
  before — the next algorithm in either family inherits the scaffolding and touches only its own
  sub-package. Note also that the shared abstractions live in new packages of their own and are not
  registered, since model discovery registers concrete classes only; the one existing file this
  feature modifies for reasons of quality rather than necessity is `deep_q_learning`'s.

  This reading was contested across two review rounds by one reviewer, while three upheld it. Rather
  than leave the disagreement to be rediscovered, **the 1.3.0 amendment carries one clarifying
  sentence in Principle VI**: that extracting shared scaffolding, which strictly reduces the
  per-algorithm coupling the principle exists to protect, does not by itself constitute a violation.
  The sentence is deliberately narrow. It does **not** move the compliance test to a future
  changeset, weaken the obligation, or exempt anything labelled "refactoring" — the dissenting
  reviewer read an earlier draft of it as doing all three, which would indeed have been a
  redefinition and so a MAJOR change. As written it adds one exemption to an existing rule, which
  the amendment procedure classifies as an addition; the registry entries are additions too, so the
  amendment is MINOR throughout.

## Out of Scope

- **DecQN and TD3.** Deferred to later features; see the roadmap.
- **A discrete-action variant of this algorithm.** The framework already has three discrete-action
  algorithms; a fourth would not extend its reach. The cost is that this model cannot appear in a
  comparative report alongside them, since report grouping is per environment.
- **Image augmentation.** The technique the literature identifies as decisive for image-based
  continuous control (DrQ-v2, arXiv:2107.09645). The roadmap does not itself make that claim — it
  references DrQ-v2 only for its trunk-sharing decision — so the attribution is to the literature,
  not to the roadmap. Deliberately left to a later feature.
- **Replay history persistence.** Excluded in feature 006 on size grounds; unchanged here.
- **Changes to report generation.** Existing report generation must consume these runs as they are.

## Review History

**Round 1** — four independent adversarial reviewers of distinct lineages (GPT, Grok, Gemini, Kimi),
all four returning NOT CONVERGED. Every finding was verified against the repository before being
adopted; three were rejected as incorrect, and the rejection is recorded so it is not re-litigated.

| Finding | Verified how | Response |
|---|---|---|
| The learning target need not use the delayed estimators or a resampled action (4 of 4) | Roadmap §2, Phase 4 | Adopted — FR-015, SC-011 |
| Target entropy "derived from" the dimension count admits a wrong sign or constant (3 of 4) | Roadmap Phase 4 fixes `-dim(A)` | Adopted — FR-017 |
| The family-B entity's wording carried a delayed copy of the actor (3 of 4) | Roadmap §2: "TARGET CRITICS ONLY"; Phase 4: "No target actor" | Adopted — FR-019, Key Entities |
| The normalised-density contract was omitted entirely (2 of 4) | Roadmap Phase 4's `-sum(log(scale))` paragraph | Adopted — FR-011, SC-010, Edge Cases |
| "Before and after the refactor" reverses an adjudicated decision (4 of 4) | Roadmap Phase 1: "two versions cannot coexist in one process" | Adopted — US2, FR-002, SC-004 |
| Story 1's independent test reinstated the withdrawn "improving curve" criterion (3 of 4) | Roadmap §4 withdraws it explicitly | Adopted — US1 Independent Test |
| CarRacing's baseline, margin and epoch were deferred to the plan (4 of 4) | Roadmap §4 tier 4: "recorded in the spec" | Adopted — baseline measured, SC-007 |
| Numeric bars had no training budget, so failure was undeclarable (2 of 4) | — | Adopted — budgets in SC-001, SC-002, SC-006, SC-007 |
| Polyak frequency was unpinned (2 of 4) | Roadmap §2.1, §6 | Adopted — FR-007, SC-016 |
| Only one of the three seeded RNG streams was named (1 of 4) | Roadmap Phase 0.4, 0.5 | Adopted — FR-026, SC-005 |
| SC-003 dropped half of the roadmap's tier-1 oracle (1 of 4) | Roadmap §4 tier 1 asserts both halves | Adopted — SC-003 |
| The LunarLander threshold cited a protocol that is not its own (2 of 4) | The 200 convention is over 100 episodes, not 20 | Adopted — SC-006 corrected |
| "Image augmentation, named in the roadmap" is a fabricated attribution (1 of 4) | The roadmap mentions DrQ-v2 only for trunk sharing | Adopted — Out of Scope reworded |
| Family A is not covered by discrete CarRacing (1 of 4) | `continuous=False` is a `Discrete` space | Adopted — Context reworded |
| **The 50 MB guard is fabricated** (4 of 4) | **Rejected.** It exists: `tests/models/test_persistence.py`, set by feature 006's tasks, which also records why the roadmap's earlier 15 MB bar was arithmetically impossible. All four reviewers were given the roadmap but not that file. | Rejected as stated — **but the underlying arithmetic is correct and was adopted**: the guard is deep-Q-specific and unreachable by this model. SC-008 now carries a derived, model-specific bar. |
| Tier 2 is an analytic oracle evaluated "in milliseconds, no training" (1 of 4) | **Rejected.** That sentence belongs to tier 1. Tier 2 is explicitly the multi-dimensional oracle that tier 3's Pendulum cannot serve as. | Rejected; SC-002 keeps a training bar, now quantified |
| Constitution impact must be decided, and Principle VI is violated (2 of 4) | **Partly rejected.** Principle VI constrains adding an algorithm, which SAC satisfies. | Decision adopted, reasoning replaced: the registry already contains `TDModel`, an abstract intermediate — the symmetry, which no reviewer raised, is what settles it at MINOR |

**Round 2** — the same four reviewers, given the two files whose absence caused round 1's three
false findings (feature 006's `tasks.md` and `tests/models/test_persistence.py`). Verdicts: one
CONVERGED, two NOT CONVERGED with a combined four findings, one timed out and was re-run. Findings
per reviewer dropped from a range of 9-16 to a range of 0-4, and every round-1 rejection was upheld
by the reviewers who re-examined it with the missing files in hand.

| Finding | Verified how | Response |
|---|---|---|
| SC-007's threshold `+65.6` is 99.97 above the baseline, not the 100 it claims | `-34.37 + 100 = 65.63` | Adopted — arithmetic corrected to `+65.63` |
| FR-015 pinned the estimators and the action but not the algebra: the greater instead of the lesser, or an entropy term added instead of subtracted, still satisfied it | Read back against Roadmap Phase 4 | Adopted — FR-015 now states the target in full, naming all five wrong forms; SC-011 tests each independently |
| FR-017's own stated failure mode had no falsifying criterion | Read back: no SC referenced FR-017 | Adopted — SC-014, on an environment of at least two action dimensions so `-dim(A)` is distinguishable from `-1` |
| The oracle environment was underspecified: a constant-reward environment satisfies every property demanded of it, and a learner that learned nothing then passes SC-002 | Construction exhibited by the reviewer; the properties as written do not exclude it | Adopted — Key Entities now requires return to depend on both dimensions, the one-sided optimum to be unreachable under a mis-mapped policy, and a constant-action policy to score below the bar; SC-002 asserts the last two |
| "Separate trunks" was an assumption, not a requirement, and a shared extractor makes the checkpoint *smaller* — so no size bar can catch it | Read back: SC-008 is an upper bound | Adopted — FR-025 and SC-015, asserting disjoint parameters directly |
| Assumptions claimed FR-011 carried the "value estimator takes the action as an input" contract; it did not, so the Context's completeness claim was false | Read back against Roadmap §2 (`QNetwork: s -> R^|A|` cannot serve as a continuous critic) | Adopted — FR-012 added; Assumptions corrected and the omission recorded there |
| Measured Baselines said meeting 50 MB would mean dropping *either* the delayed copies *or* the optimizer state; both would be needed | `5 x 11.2 = 56 > 50`; `3 x 11.2 + 3 x 2 x 11.2 = 101 > 50` | Adopted — sentence corrected, and the one remaining route to 50 MB named as the thing FR-025 forbids |
| Principle VI is violated because the feature does modify `deep_q_learning` (1 of 2 who examined it; the other upheld the rejection) | Constitution Principle VI reads "MUST NOT **require**" | Rejection upheld on the reading, but the disagreement is closed structurally rather than by argument: the 1.3.0 amendment now also clarifies Principle VI |


**Round 3** — same four reviewers against revision 3. Verdicts: two CONVERGED (Grok, Gemini), two
NOT CONVERGED with four findings between them, of which **two reviewers of distinct lineages
independently found the same defect** — the strongest signal available from this method, and the one
finding here that a single reviewer might have been talked out of.

| Finding | Verified how | Response |
|---|---|---|
| SC-015 asserted disjoint parameters only between the two value estimators, while FR-025 requires three separate extractors. An actor sharing an extractor with one estimator violates the requirement and passes every criterion in the document — including the size bar, since sharing makes the checkpoint *smaller* (2 of 4, independently) | Read back: FR-025 says "the actor and each of the two"; SC-015 said "the two value estimators" | Adopted — SC-015 now asserts all three pairs; FR-025 states "three extractors, pairwise disjoint" and names the distinct failure mode of actor-to-estimator sharing |
| The actor objective and the temperature objective were never stated, so the whole learned side of the algorithm other than the critic target was pinned by end-to-end scores alone — which this document's own standard rejects (1 of 4) | Read back: FR-013 named the algorithm, FR-015 pinned only the target, FR-017 only the entropy constant | Adopted — FR-022, FR-024, SC-012 |
| FR-015's prose enumerated four wrong forms while SC-011 and the round-2 history both said five (1 of 4) | Counted | Adopted — the truncation form is now enumerated in FR-015 too, making it five in both places |
| The Principle VI clarification is a redefinition, so the bump should be MAJOR (1 of 4; the other three upheld MINOR) | Constitution's amendment procedure: MAJOR for removals/redefinitions, MINOR for additions | **Rejected on the substance** — one narrow added exemption is an addition. But the earlier draft's "the test applies to the next algorithm added" did read as a scope change, and that wording is withdrawn: the amendment sentence is now narrowed to say only what it needs to. |

**Convergence**: reviewer findings across the three rounds ran 12/4/9/8, then 4/3/0/—, then 3/0/0/1,
with every round-1 and round-2 rejection independently upheld once the missing files were supplied.
The two round-3 blocking findings are adopted; the one disputed governance finding is rejected by
three reviewers to one and its triggering wording removed regardless. No finding in round 3 touched
anything established in round 1.

**Round 4 (closure check)** — two CONVERGED (Grok, Gemini), one NOT CONVERGED with a single
finding, one returned an empty completion and was excluded. The finding, adopted: FR-024 pinned only
the *direction* in which the temperature moves, not its objective, so SC-012 could not verify it
against a hand-computed value the way it verifies the target and the actor — a sign-correct
hand-written controller would have satisfied it. FR-024 now states the objective, that the
log-density is held constant within it, and that the logarithm of the temperature is what is
optimised so the temperature cannot cross zero and silently invert the entropy term; SC-012 gains
three corresponding falsifying cases. Both reviewers who returned CONVERGED confirmed the round-3
fixes present in the text.

**Round 5 (final confirmation)** — GPT, Grok and Gemini all returned CONVERGED, GPT having dissented
in every previous round. A fourth lineage was substituted for one that had failed twice on this
context size, and — reviewing the document for the first time — returned two findings that five
rounds had missed. Both are adopted. That a fresh reviewer beat three converged ones is the result
worth recording here: convergence measures a panel's exhaustion as much as a document's readiness.

| Finding | Verified how | Response |
|---|---|---|
| The learning target was pinned as a **value** only. Nothing forbade it from remaining attached to the learning graph, which leaks gradient into the actor, the temperature and the delayed copies. Detaching changes no number, so all five of SC-011's cases pass identically either way — and FR-022 actively invites the defect by requiring the actor's own sample to stay differentiable | Read back: every clause of FR-015 and SC-011 is value-level, while SC-012 already contained graph-level cases, so the document's own scope admitted this and simply omitted it | Adopted — FR-015 gains the constant-for-learning clause; SC-011 gains a sixth, non-value-level case |
| SC-002 said "the three properties … are themselves asserted" and then asserted two. The unasserted one — that the return depends on **both** dimensions — is the property the oracle exists for, and without it an environment whose second dimension barely matters lets an agent that never learned that dimension pass | Read back; structurally identical to the SC-015 defect round 3 blocked on | Adopted — SC-002 now asserts all three, the third by clamping either dimension in the closed-form optimum |
| Measured Baselines quoted ~11.2 MB per network, from a test docstring's rounding, where the recorded measurement is 11.70 MB | `tasks.md` T049 benchmark table and roadmap §0 both say 11.70 | Adopted — figure and provenance corrected; the derived total moves 123 → 129 MB and every conclusion holds (150 MB bar unchanged, 58.5 and 105 both still above 50) |
| "An order-of-magnitude difference" describes a ~2.8x one | Arithmetic | Adopted — rhetoric replaced by the measured ratio |


**Round 6** — two CONVERGED, two NOT CONVERGED with one finding each. Both findings are adopted,
and both are of a kind only a late round produces: one is the exact mirror of round 5's, on the side
of the computation round 5 did not look at, and the other is a defect **introduced by** round 5's own
fix.

| Finding | Verified how | Response |
|---|---|---|
| Round 5 confined the *target's* gradient and left the *actor objective's* unconfined. A single summed backward pass over the three objectives is value-identical, passes every criterion, steps both estimators to raise the value the actor is chasing — identically, so neither the lesser-of-two construction nor the disjoint extractors can see it — and deposits on the temperature a gradient that exactly cancels the log-density term of its own objective, reducing automatic temperature adjustment to a countdown | Edge inventory over the document: target → actor/temperature/delayed copies pinned by FR-015; temperature → policy pinned by FR-024; actor sample → policy pinned by FR-022; actor objective → estimators/temperature unpinned | Adopted — FR-022 gains the confinement clause; SC-012 gains a non-value-level case mirroring SC-011's sixth |
| Round 5's own new SC-002 assertion — that the optimum with either dimension clamped to **any** constant falls below the bar — is **unsatisfiable by a correct implementation**: clamping a dimension to the coordinate it takes in the optimal action leaves the optimum reachable | Elementary, and sharpened by Key Entities requiring the one-sided optimum to sit at or near the bound | Adopted — the assertion is now quantified over the *best* such constant, which is satisfiable exactly when the environment's per-dimension optimum varies with the observation; Key Entities' first property is restated to require that variation |

The second row is the more instructive of the two: a fix adopted from an adversarial review, written
directly into a success criterion, was itself wrong — and wrong in the direction that would have
blocked a correct implementation rather than admitted a broken one. It is the argument for running
the round after the one that looked finished.


**Round 7** — two CONVERGED, two NOT CONVERGED. Both round-6 adoptions were confirmed correct in the
text, including that the repaired SC-002 is now satisfiable. Two further findings, both adopted, and
both instances of the same pattern the last three rounds have been working through: the document
pinned what a learned quantity is *used for* before it pinned what it is *trained on*.

| Finding | Verified how | Response |
|---|---|---|
| The value estimators' own objective was never stated (2 of 4, independently — one of them ranking it below blocking). FR-015 pinned the target, FR-022 the actor's objective, FR-024 the temperature's; nothing said the estimators regress toward that target, at the action **as stored**, and that **both** do. Training one and letting the other drift, or regressing at a resampled action, passes every criterion | Read back: three objectives stated, four learned quantities | Adopted — FR-023, SC-013 |
| FR-007 pinned the *clock* of the delayed copies' advance but not its *graduality*. A hard copy performed once per gradient step satisfies the schedule and makes each delayed copy identical to its live counterpart from the first step, at which point FR-015's "lesser of the two delayed copies" **is** the lesser of the two live ones — the first wrong form FR-015 enumerates, reached without violating a word of it. SC-015 checked only the clock (1 of 4) | Read back against FR-015's own enumeration | Adopted — FR-007 requires gradual averaging explicitly; SC-015 asserts the fraction against a hand-computed value and that a trained model's delayed copies differ from its live ones |

**Closure argument, now available and stated in SC-013.** The model learns exactly four things: the
two value estimators, the actor, the temperature, and the delayed copies. As of revision 8 each has
both its objective and its gradient boundary pinned by a direct, non-end-to-end test. That set is
exhaustive by construction, so the class of defect the last four rounds kept finding — a learned
quantity whose update was unspecified — has no remaining member. This is the first round at which
that can be said, and it is a stronger statement than any individual reviewer's verdict.


**Round 8** — three CONVERGED, one NOT CONVERGED with a single finding, adopted. The round was run
as a closed question rather than an open invitation: verify the two round-7 adoptions, then test the
closure argument.

| Finding | Verified how | Response |
|---|---|---|
| SC-016's end-state check — that a trained model's delayed copies differ from its live ones — is a criterion a **correct** implementation can fail. Gradual averaging legitimately leaves them equal when the live update was zero, and they may coincide numerically at convergence | Elementary; the second instance of this error class, after round 6's | Adopted — the check is now conditioned on a step in which the live parameters are known to have moved, and the reason is recorded in the criterion itself |
| Edge Cases and Measured Baselines said "three optimizers" while four quantities are learned (non-blocking) | The temperature's optimizer holds two floats | Adopted — count corrected; the arithmetic is unaffected and says so |

**The closure argument was tested and held.** One reviewer verified it parameter by parameter rather
than accepting it: the inventory is the actor's mean, log-standard-deviation and extractor, the two
estimators with their extractors, the two delayed copies, and the temperature — exactly the four
quantities named. It then checked each gradient-boundary clause for *completeness of its
enumeration*, on the reasoning that a boundary naming fewer parameters than its loss actually touches
would be the remaining hole, and found each complete. It searched for a fifth learned quantity and
rejected the candidates with reasons: optimizer moment buffers are adapted bookkeeping rather than
trained and are covered by the checkpoint requirement; the warmup counter rides on the step count;
observation rescaling and the action map are not learned.

**Round 9** — a narrow confirmation of the two round-8 fixes, and both were still wrong.

| Finding | Verified how | Response |
|---|---|---|
| SC-016's repaired precondition — "after a step in which the live parameters are known to have moved" — is still insufficient. Counterexample: a live parameter moving from 0 to 1 while its delayed copy already sits at 1 leaves the delayed copy at 1 for every averaging fraction, equal to the parameter that just moved | Arithmetic, supplied by the reviewer | Adopted — the precondition is now the controlled one: delayed copies equal to live parameters beforehand, one known non-zero live update, fraction strictly between 0 and 1. Both failed attempts are recorded inside the criterion so a third does not repeat them |
| The optimizer count was corrected in two of its three occurrences; SC-008 still said three | Grep | Adopted |

This is the third consecutive round in which a fix adopted from the previous round was itself
defective, and the second in which the defect was a criterion a correct implementation would fail.
The pattern is now explicit in the Convergence section.

## Convergence

Nine rounds, four reviewers of distinct lineages, Anthropic excluded throughout on the ground that
this document was drafted by an Anthropic model. Findings per round: 33, 7, 4, 1, 2, 2, 2, 1, 2. Of
54 findings, 50 were adopted and 4 rejected — the four rejections all being reviewers working without a
file they had not been given, or misattributing one section of the roadmap to another, each recorded
above with the evidence rather than merely dismissed.

Three results from the process are worth keeping, because they are not about this feature:

1. **A fresh reviewer at round 5 found two real defects that three converged reviewers had missed.**
   Convergence measures a panel's exhaustion at least as much as a document's readiness. Rotating in
   a lineage that has not seen the document is worth more than another pass from one that has.
2. **Four findings were defects in fixes adopted from earlier rounds**, in rounds 6, 8 and 9 — and
   three of them were criteria a *correct* implementation would have failed, the opposite direction
   from the one adversarial review is usually pointed in. SC-016's end-state check was wrong twice in
   a row before it was right. A review loop that only asks "what would a broken implementation pass"
   will not catch these; the question has to be asked in both directions, and a fix deserves the same
   scrutiny as the text it replaces.
**Round 10** — a two-question confirmation of the round-9 repairs, unanimous CONVERGED across the
three reviewers run (GPT, which had raised both, plus GLM and Grok). SC-016's precondition was
verified algebraically in both directions: with delayed copies equal to the live parameters
beforehand, a non-zero live update and a fraction strictly between 0 and 1, gradual averaging gives
`D + tau*(L - D) != L`, while a hard copy gives exactly `L` on every step the check applies to. The
specification is closed at revision 10.

3. **The last four rounds all found the same pattern**, a learned quantity whose objective or
   gradient boundary was unstated. That pattern is what made a closure argument possible, and the
   closure argument is what ends the loop — not a run of unanimous verdicts, which rounds 5 and 7
   show can be wrong.
