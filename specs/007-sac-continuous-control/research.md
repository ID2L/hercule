# Phase 0 — Research: SAC on a shared off-policy ancestor

Decisions taken before design, each with what it was chosen over. Where a decision was already
settled by the specification or by `specs/ROADMAP-continuous-actions.md`, it is not re-opened here;
only the ones this plan had to make appear below.

## R1 — Where the encoder/head split falls

**Decision**: `Encoder` is everything up to and including the last hidden activation; the head is
the final `Linear`. For a vector observation the encoder is
`Linear(in,128) → ReLU → Linear(128,128) → ReLU`; for an image it is
`Conv2d ×3 (with ReLUs) → Flatten → Linear(flat,512) → ReLU`.

**Rationale**: this is the only split point that preserves the order in which parameterised modules
are constructed, and that order is observable behaviour under FR-002 — see the plan's "The
constraint that governs the refactor". Every parameterised layer is created in exactly the sequence
it is today, so torch's global RNG is consumed identically and the initial weights are bit-identical
from the same seed.

**Alternatives rejected**:

- *Splitting after the convolutions only, leaving both fully-connected layers in the head.* Equally
  valid for image observations, but there is then no encoder at all for vector observations, and
  SAC's critic needs a place to concatenate the action that exists in both branches.
- *A lazily built encoder that infers its input shape on first forward.* Attractive because it
  removes the dummy-input probe, and fatal: it moves parameter construction out of `configure()` to
  an arbitrary later point, which changes when the RNG is drawn from and breaks the fixture.

## R2 — Where the action enters the value estimator

**Decision**: the estimator encodes the observation with its own `Encoder`, concatenates the action
to the encoding, and passes the result through a two-layer head to a scalar.

**Rationale**: it is the one arrangement that works unchanged for both observation branches. For an
image, concatenating a 3-vector to a 96×96×12 input before the convolutions is meaningless; after
the encoder both branches present a flat feature vector and the action concatenates the same way.

**Alternatives rejected**:

- *Concatenating action to observation at the input.* Only definable for vector observations.
- *Tiling the action across the image's spatial dimensions.* Used in some vision-based work, costs a
  great deal of compute for an effect nothing in this feature's criteria would detect.

## R3 — One learning rate or three

**Decision**: one `learning_rate` hyperparameter, applied to the actor, both estimators and the
temperature.

**Rationale**: Hercule expands list-valued hyperparameters into a full cartesian product. Three
independent rates turn a three-value sweep into 27 runs instead of 3, and the project's own
sensitivity tooling then reports on a grid whose cost is cubic in a distinction the reference
implementations do not make. SAC's published results use one rate across all three.

**Alternative rejected**: separate rates as a later addition if a sweep shows the single rate
binding. Adding a hyperparameter later is additive and cheap; removing one from a grid already run
invalidates the runs.

## R4 — Bounding the policy's spread

**Decision**: the actor outputs an unbounded log-standard-deviation which is clamped to `[-20, 2]`
before use; the clamp bounds are module constants, not hyperparameters.

**Rationale**: without a lower bound the spread collapses toward zero and the log-density diverges;
without an upper bound early training produces spreads large enough that the squashing correction
saturates. The range is the reference implementations' and is not the kind of quantity a benchmark
should sweep — it is a numerical guard, not a design choice.

**Alternative rejected**: a soft bound via `tanh` rescaling of the raw output. Equivalent in effect,
but it makes the log-density's closed form harder to write, and SC-010 requires that closed form to
be computable independently for the test.

## R5 — Computing the squashing correction

**Decision**: implement the correction in its numerically stable form, and have the test compare it
against the **naive** closed form on inputs where the naive form is accurate.

**Rationale**: the correction is `Σ log(1 − tanh²(u))`, which underflows to `log(0)` for `|u| ≳ 9`
in float32 — reachable in ordinary training. The stable rewriting, `Σ 2·(log 2 − u − softplus(−2u))`,
is exact and does not underflow. SC-010 demands verification "against an independently computed
reference value": the naive expression *is* that independent reference, and comparing the two on
moderate inputs tests the rewriting rather than restating it. A test that recomputed the stable form
would assert the implementation against itself.

**Alternative rejected**: the naive form plus an epsilon, `log(1 − tanh² + 1e-6)`. It is what most
tutorials show; it silently biases the density wherever the epsilon dominates, in exactly the
saturated region where the actor spends time once it has learned to push an action to a bound.

## R6 — Warmup actions

**Decision**: the warmup phase samples uniformly in the **normalised** `[-1, 1]^d` cube and maps to
environment coordinates on the way out, so the stored action is already in policy coordinates.

**Rationale**: FR-006 requires the replay store to hold policy coordinates. Sampling from
`env.action_space` instead would require inverting the map before storing — a second code path
through the mapping, and a channel for the two to disagree. The roadmap makes the same point about
`_select_action` returning a converted action.

**Note**: for a symmetric dimension, uniform-in-normalised and uniform-in-environment coincide. For
a one-sided dimension they also coincide, because the map is affine. So nothing is lost.

## R7 — The asymmetric oracle environment

**Decision**: a tracking task. The observation is a target point, redrawn every step; the action must
match it. Two action dimensions with deliberately different bounds — `[-1, 1]` and `[0, 4]` — and the
target on the one-sided dimension drawn from the upper half of its range, `[2, 4]`, so the optimal
action there always exceeds what an unscaled normalised output can reach. Reward
`1 - 0.5*(squared error)`, episodes of 50 steps, optimum 50.0, SC-002 bar 45.0. The full arithmetic
is in contract C2.

**It lives in `src/hercule/environnements/oracle.py`, not under `tests/`.** FR-029 requires a
ready-to-run config per validation rung, and a config can only name an environment the package
registers with Gymnasium at import time. A test-local environment would have made that rung
unreachable from `hercule learn`.

**The sizing is forced, not chosen.** A first draft used a target on `[1.5, 2.0]` over an action
range of `[0, 2]`, and was **unsatisfiable**: SC-002 requires that clamping either dimension leave a
policy below 90% of the optimum, which is a lower bound on that dimension's target *variance*, while
the mis-map property is a lower bound on its target's *location*. On an interval of width `0.25` the
variance is at most `0.0625` and the clamped policy scores above the bar however the rest is tuned —
a criterion no correct implementation could pass. The two requirements are only jointly satisfiable
once the interval is wide enough to hold both, which is why dimension 1 spans `[0, 4]` with its
target in `[2, 4]`: variance `1/3`, and an optimum always at least `2.0` against an unscaled policy
capped at `1.0`.

The reward is shifted positive so that "90% of the optimal return" is a meaningful fraction; on a
purely negative scale a percentage of the optimum is ambiguous. Episodes are 50 steps rather than 20
so that SC-002's 200-episode budget is 10,000 environment steps — enough for SAC to converge on a
task this simple, where 4,000 would have been thin.

**Alternatives rejected**:

- *`MountainCarContinuous-v0`.* One action dimension — the defect class the oracle exists for cannot
  appear.
- *`LunarLanderContinuous`.* Two dimensions, but its optimum has no closed form, so SC-002's bar
  would have nothing to be 90% of. It is used for SC-006 instead, where the bar is the published
  solved threshold rather than a fraction of an analytic optimum.
- *Normalising each dimension's error by its own half-range*, so both contribute equally to the
  reward. It makes the reward prettier and breaks the clamp test: dividing dimension 1's error by its
  half-range divides its variance by four, dropping it back below the detection threshold.

## R8 — The golden fixture's contents and lifetime

**Decision**: a JSON file holding the per-episode reward series and a SHA-256 hash per parameter
tensor, produced by a short committed script from `main` before any refactoring, at a fixed seed, for
**three** configurations: `CartPole-v1`, `FrozenLake-v1`, and a shaped-observation environment with a
three-dimensional `Box`. It is deleted at feature closure and replaced by a determinism property test.

**The hashes are taken in parameter *order*, not keyed by parameter name.** R11 renames every
attribute path, so a name-keyed fixture would fail the refactor it exists to certify — and fail for a
reason that has nothing to do with the numbers, which is the most misleading way a test can break.
Ordering by `parameters()` is stable across the rename, as R11's table shows. This also keeps the two
instruments cleanly separated: the fixture certifies the **numbers**, contract C4's load test
certifies the **names**.

**A second artifact is captured in the same step and is not the fixture**: two real, loadable
pre-refactor `model.json` files, one from a vector-observation run and one from an image-observation
run, committed under `tests/fixtures/checkpoints/`. They are what C4's compatibility test loads, and
they must be captured before any source change for exactly the reason the fixture must. They live in
a **different directory** from the golden fixture because they outlive it: the fixture is deleted at
feature closure, these stay for as long as the migration path does.

The pre-006 legacy form cannot be produced by today's code at all — feature 006 replaced it — so the
legacy branch of the dispatch is tested against a hand-built minimal file with a
`q_network_state_dict` key, not against a captured one. That is stated so nobody spends time looking
for a capture step that cannot exist.

**Rationale**: hashes rather than the weights themselves keep the fixture small and make a mismatch
unambiguous. Three configurations because the split of R1 is the change most likely to perturb one
network branch and not the other, and **two of the three exercise the same branch**: `CartPole-v1` is
a one-dimensional `Box` and `FrozenLake-v1` is `Discrete`, whose `_single_frame_shape()` returns
`(1,)` — both therefore take `_build_mlp`, and neither reaches `_build_cnn` at all. A first draft of
this decision claimed FrozenLake covered the image path "via a shaped observation"; it does not, and
a fixture built on those two alone would have left the convolutional branch — the one with five
parameterised layers rather than three — entirely uncovered. The third configuration uses the shaped
environment `tests/models/test_persistence.py` already defines for exactly this purpose.

`FrozenLake-v1` is kept rather than replaced because it covers a third thing neither other config
does: the `Discrete`-observation rescaling path, which divides by the space's cardinality.

**On the retirement**: the roadmap requires it, and the reason is worth restating. A committed
fixture asserts *today's numbers*, so every later deliberate behaviour change requires regenerating
it, at which point it certifies nothing. The determinism property it is replaced by — same seed
twice identical, different seeds different — stores no expectation and survives.

## R9 — Device policy

**Decision**: CPU is the reference. CUDA is used when available for the image-observation runs, and
the bit-identity requirement of FR-002 is asserted on CPU only.

**Rationale**: floating-point reduction order differs between CPU and CUDA kernels, so bit-identity
across devices is not achievable and asserting it would produce a test that fails on exactly the
machines that have a GPU. The fixture is a CPU artifact and the test that reads it is CPU-pinned.

## R11 — How an existing checkpoint survives the encoder/head split

**Decision**: `_import` carries an explicit map from the pre-refactor parameter key names to the
post-refactor ones, applied to `format_version` 2 and to the pre-006 legacy form before
`load_state_dict` runs.

**Rationale**: a `state_dict()`'s keys are attribute paths. R1's split renames all of them, and
`load_state_dict` raises by default on any key it did not expect. So every `model.json` already in
`outputs/` would fail to load after a refactor that changes not one weight — the failure mode is
total and the cause is invisible, since the weights are right and the names are not.

**Why the golden fixture does not cover it**: the fixture compares weight tensors produced by fresh
training. It never opens a checkpoint. A refactor can be bit-identical under the fixture and still
make every stored model unreadable, which is exactly why contract C4 requires a separate test that
loads a real pre-refactor file.

**Alternative rejected**: reproducing the old attribute paths in the new modules, duplicate
convolutional registrations included. It would keep the fixture and the loader both green with no
migration code — at the price of carrying a quirk of today's `QNetwork` into the shared ancestor that
every future model inherits, forever, to avoid writing one mapping once.

## R10 — What is *not* researched, because the spec settled it

Recorded so no reader mistakes their absence for an oversight: the learning target's full algebra
and its gradient boundary (FR-015), the actor's objective and its confinement (FR-022), the value
estimators' objective (FR-023), the temperature's objective, direction and log-parameterisation
(FR-024), the target entropy's exact value (FR-017), the normalised coordinate system for the
density (FR-011), three pairwise-disjoint feature extractors (FR-025), gradual averaging once per
gradient step (FR-007). Each was pinned during the specification's nine review rounds, several after
an implementation that got it wrong was shown to pass every criterion then in place. None is a plan
decision.
