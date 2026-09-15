# Phase 1 — Contracts

Hercule is a library with a CLI, not a service, so its contracts are the interfaces a new algorithm
implements and the ones the framework guarantees back. Two are specified here: the hook surface a
concrete model must satisfy, and the oracle environment this feature introduces.

## C1 — `OffPolicyReplayModel`: the contract a concrete model implements

A subclass is correct if and only if every clause below holds. Each is stated as an obligation on
the subclass or a guarantee from the ancestor, so that a reviewer can check them one at a time.

### The ancestor guarantees

| # | Guarantee |
|---|---|
| G1 | `configure()` seeds torch, Python `random` and the model-owned NumPy generator from the `seed` hyperparameter **before** calling `_build_networks()`, and draws from none of them in between |
| G2 | `run_epoch()` calls the hooks in the order given in `data-model.md`, with `begin_episode()` before frame priming and `push()` only on the training path |
| G3 | `_sync_targets()` is called once per **environment** step, never chained to whether a gradient step was taken. Its default body hard-copies each pair `_target_pairs()` declares, on the interval `_target_sync_interval()` returns; that hook returns `None` — meaning never — by default. Both hooks are needed: an interval alone does not tell the ancestor what to copy onto what, and a naming convention over `_networks()` would be an undocumented contract |
| G4 | `_update()` is called only when `_ready_to_update()` returns true |
| G5 | Transitions carry `terminated` and `truncated` **separately**; the ancestor never collapses them into one flag on the stored transition |
| G6 | `act()` and `predict()` return the environment-coordinate action — the first element of `_select_action`'s pair |
| G7 | `save()`/`load()` serialise every module `_networks()` names, every optimizer `_optimizers()` names, **the three random streams and the step and epoch counters, which the ancestor owns and exports itself**, and whatever `_extra_state()` returns; delayed copies keep their own weights rather than being rebuilt from their live counterparts |
| G8 | Per-dimension action bounds are cached **only when the action space is a `Box`**. On a `Discrete` action space the mapping is the identity and no bounds are read — `Discrete` has no `low` or `high`, and reading them unconditionally would break the rebuilt deep model at `configure()` |
| G9 | Neither `act()` nor `predict()` mutates **adaptive** state: no hyperparameter, no exploration schedule, no temperature, no counter. Two mutations are *permitted* where the call requires them, and neither is required of both methods: `predict()` advances the frame history, because that is what observation stacking *is* and `hercule play` depends on it, while `act()` does not; and a **stochastic** selection consumes the random streams, while a deterministic one — every `predict()`, and `act(training=False)` — consumes none. Where adaptive state *does* advance is not this guarantee's business and it makes no claim about it — an earlier draft tried to enumerate the sites and was wrong twice, omitting the counters the loop advances directly and the delayed copies `_sync_targets()` advances. G9 says only what `act()` and `predict()` may and may not do |

### The subclass must

| # | Obligation |
|---|---|
| O1 | Construct every parameterised module inside `_build_networks()`, in a deterministic order fixed by the code, never lazily on first forward |
| O2 | Return from `_select_action` a pair whose second element is the action **as it will be stored**, in the model's own coordinates, and whose first is that action mapped to the environment's |
| O3 | Perform all exploration inside `_select_action` — including any warmup — so the ancestor never samples an action |
| O4 | Name in `_networks()` every module whose weights must survive a resume, delayed copies included |
| O5 | Put into `_extra_state()` every **subclass-specific** non-module quantity it mutates during training — the deep model's `epsilon`, SAC's log-temperature — **and every quantity the network's shape depends on** (see the note below). Not the random streams and not the counters: the ancestor owns and exports those under G7, and a subclass emitting them too would write the same state under two keys |
| O6 | Declare `supported_spaces`; discovery refuses to register a concrete model without it |
| O7 | Declare `_target_pairs()` whenever it holds delayed copies, and `_target_sync_interval()` whenever it wants the ancestor's hard-copy default |

**On O5 and network shape.** `hercule play` configures a model with **default** hyperparameters and
then loads the checkpoint. If the checkpoint does not carry the values the network's shape was built
from, the rebuilt network has the wrong shape and the load fails. Today's deep model carries
`frame_stack` and `observation_shape` in its checkpoint for exactly this reason; the ancestor's
checkpoint assembly must keep carrying them, and SAC must carry them too. A design that moved
checkpoint assembly into the ancestor and quietly dropped these two keys would break `hercule play`
on every stacked model, including ones already on disk.

### What a violation looks like

Recorded because each of these was found during the specification's review rounds to pass every
criterion then in place:

- O2 violated by storing the environment-coordinate action: the ancestor maps it a second time, so
  the action the estimator is trained on is not the action that was executed. Evaluation still looks
  in-bounds.
- O3 violated by sampling warmup actions in the ancestor: the ancestor must then know the action
  space's shape and bounds in a second place, and a warmup action reaches the replay store in
  environment coordinates while a policy action reaches it in normalised ones.
- O4 violated by omitting the delayed copies: a resumed run rebuilds them from the live weights, so
  it restarts with zero lag — which is the same as having no delayed copies at all for as long as
  the averaging takes to re-establish the gap.
- O1 violated by replacing the deep model's second `QNetwork(...)` construction with a copy of the
  first: see the plan's refactor constraint. The weights are the same; the RNG state afterwards is
  not.

## C2 — `AsymmetricOracleEnv`: the two-dimensional oracle

A `gym.Env` introduced by this feature. It lives in **`src/hercule/environnements/oracle.py`**, not
under `tests/` — FR-029 requires a ready-to-run config per validation rung, and a config can only
name an environment the package registers with Gymnasium.

**Placing the module under the package is not enough to register it.** `oracle.py`'s
`gym.register(...)` call runs only when that module is imported, and nothing imports it by merely
importing `hercule.environnements`. `environnements/__init__.py` must import it explicitly, so that
any process reaching the environment factory — which always goes through that package — has the
registration in place. Without that one line, `hercule learn experiments/sac_oracle.yaml` fails in a
fresh process with an unknown-environment error, and SC-002's training half is unreachable.

### Spaces and dynamics

```text
observation_space = Box(low=[-1.0, 2.0], high=[1.0, 4.0], shape=(2,))   the target to track
action_space      = Box(low=[-1.0, 0.0], high=[1.0, 4.0], shape=(2,))   deliberately asymmetric

target drawn uniformly over the observation space, redrawn EVERY step and exposed as the observation
reward_t = 1.0 - 0.5 * ((a_0 - T_0)^2 + (a_1 - T_1)^2)
episode  = 50 steps, always truncated, never terminated
optimal return J* = 50.0        (a = T at every step)
SC-002 bar = 0.9 * J* = 45.0
```

Dimension 0 is symmetric and spans the same range as the policy's own normalised output. Dimension 1
is one-sided, twice as wide, and its target lives in the **upper half** of its range — so the
optimal action there is always at least `2.0`, while a policy whose normalised output is forwarded
unscaled can never exceed `1.0`.

Truncating and never terminating is deliberate: it exercises the terminated/truncated separation of
G5 on every episode, the same way `Pendulum-v1` does.

### The three properties, verified

Both coordinates are uniform and independent, so `Var(T_0) = (1-(-1))^2/12 = 1/3` and
`Var(T_1) = (4-2)^2/12 = 1/3`. Expected return for a policy is `50 * (1 - 0.5 * E[squared error])`.

| Policy | Expected squared error | Expected return | Below 45? |
|---|---|---|---|
| Optimal, `a = T` | 0 | **50.00** | — |
| Best constant on dim 0 (`a_0 = 0`), dim 1 tracked | `Var(T_0) = 1/3` | **41.67** | yes |
| Best constant on dim 1 (`a_1 = 3`), dim 0 tracked | `Var(T_1) = 1/3` | **41.67** | yes |
| Best fully constant action | `2/3` | **33.33** | yes |
| Normalised output forwarded unscaled (best is `a_1 = 1`) | `Var(T_1) + (3-1)^2 = 13/3` | **-58.33** | yes |

Each row maps to one of the spec's required properties:

| Property (spec Key Entities) | Row that establishes it |
|---|---|
| Return depends on both dimensions, varying with the observation | rows 2 and 3 — clamping **either** dimension, even at its best constant and with the other tracked perfectly, falls short of the bar |
| The one-sided optimum is unreachable under a mis-mapped policy | row 5 |
| A constant-action policy scores materially below the optimum | row 4 |

**Why the ranges are what they are.** A first draft of this environment used
`T_1 ~ U[1.5, 2.0]` on an action range of `[0, 2]`, and it was **unsatisfiable**: any distribution
on an interval of width `0.25` has variance at most `0.0625`, so clamping dimension 1 at its mean
costs at most `20 * 0.0625 / 2` and the clamped policy scores above the bar no matter how the
environment is otherwise tuned. The requirement that clamping a dimension be *detectable* is a lower
bound on that dimension's target **variance**, while the requirement that its optimum be unreachable
under a mis-map is a lower bound on its target's **location**. Those two pull in opposite directions
inside a narrow range, and are only jointly satisfiable once the range is wide enough to hold both —
which is why dimension 1 spans `[0, 4]` with its target in `[2, 4]`.

The general condition, so a future change to these numbers can be checked rather than guessed: with
a per-step optimum of `1`, an episode of `L` steps and a bar at 90%, a policy costing an expected
per-step penalty `p` returns `L(1 - p)` and clears the bar exactly when `p <= 0.1`. The bound is
**inclusive**, because SC-002 says "at least 90%": a policy costing exactly `0.1` scores exactly the
bar and passes. Each row above is that inequality, and each has margin — the tightest, the two
single-dimension clamps, sits at `p = 1/6`.

All five rows are closed-form, so the assertions are arithmetic and need no training run to
establish that the oracle is a valid oracle.

### What the oracle does **not** do

It is not a performance benchmark and no conclusion about SAC's quality should be drawn from it. Its
only job is to fail loudly when a per-dimension or asymmetry defect exists — the class of defect that
is structurally invisible on `Pendulum-v1`, which is one-dimensional and symmetric.

## C3 — CLI surface

Unchanged. `hercule learn`, `hercule play` and `hercule report` gain no flags and no new output
shapes. A SAC run is an ordinary run: same directory layout, same three JSON files, same report
pipeline. This is stated as a contract because FR-030 makes it one — report generation must consume
these runs without modification, which means nothing in this feature may change what a run directory
looks like.

## C4 — Checkpoint format compatibility

The deep model's checkpoint keeps every top-level **key** it has today — `format_version`,
`networks_b64`, `optimizer_state_b64`, `rng_state_b64`, `epsilon`, `epoch_count`, `step_count`,
`frame_stack`, `observation_shape` — and the entries inside `networks_b64` keep the names they have,
`online` and `target`. The legacy pre-006 import path (`q_network_state_dict`) keeps working.

**Two things underneath do change, and the version bumps to announce both. The first:** today `optimizer_state_b64`
holds a single bare encoded state, because the deep model has exactly one optimizer; generalising it
to one entry per optimizer `_optimizers()` names makes it a name-keyed mapping. That is a change to
the payload shape of `_export()`, and `_CHECKPOINT_FORMAT_VERSION`'s own documented rule is that it
is bumped whenever that shape changes. So **`format_version` goes to 3**, and `_import` dispatches on
it: 3 reads the mapping, 2 reads the bare state, and the pre-006 legacy form keeps its existing path.

An earlier draft of this contract claimed the format was unchanged *while* restructuring that key
under the same version number. That would have left a v2 file and a v3 file indistinguishable — which
is exactly what the version field exists to prevent — so the claim is withdrawn rather than the
restructuring.

**The second, and subtler: the parameter key names inside `networks_b64`.** A `state_dict()`'s keys
are attribute paths, so today's deep model stores `network.0.weight` and — on the convolutional
branch — the *same* parameters a second time under `conv_layers.*` and `fc_layers.*`, because
`self.network = nn.Sequential(self.conv_layers, self.fc_layers)` registers them twice. Splitting the
network into an encoder and a head renames every one of those paths, and `load_state_dict` is strict
by default: it raises on a missing or unexpected key. **An existing `model.json` would therefore fail
to load even though its weights are bit-identical to what the rebuilt model expects.**

The golden fixture cannot catch this. It compares weight *tensors* and never loads a file, so it
would stay green while every checkpoint in `outputs/` became unreadable.

**Decision: `_import` carries an explicit key-migration map** from the pre-refactor attribute paths
to the post-refactor ones, applied to any checkpoint at `format_version` 2 or the legacy form before
`load_state_dict` sees it.

The post-refactor paths are fixed here so the map is a table rather than a decision left to
implementation. `Encoder` holds its layers in one `nn.Sequential` named `layers`; `QNetwork` holds
`self.encoder` and `self.head`.

| Branch | Old key prefix | New key prefix |
|---|---|---|
| vector | `network.0.` | `encoder.layers.0.` |
| vector | `network.2.` | `encoder.layers.2.` |
| vector | `network.4.` | `head.` |
| image | `conv_layers.0.` | `encoder.layers.0.` |
| image | `conv_layers.2.` | `encoder.layers.2.` |
| image | `conv_layers.4.` | `encoder.layers.4.` |
| image | `fc_layers.1.` | `encoder.layers.7.` |
| image | `fc_layers.3.` | `head.` |
| image | `network.0.*`, `network.1.*` | **dropped** — aliases of the rows above |

**The branch is selected first, and only that branch's rows are then applied.** The two row-sets are
not jointly applicable: on an image checkpoint `network.0.0.weight` matches both the vector rename's
`network.0.` prefix and the image drop rule, and on a vector checkpoint `network.0.weight` matches
both as well. Renaming first leaves an unexpected key on every image file; dropping first leaves a
missing key on every vector one. There is no flat application order that is correct for both, so the
selector is part of the contract rather than a decision left to implementation:

```text
image branch  <=>  the checkpoint contains any key beginning "conv_layers."
otherwise     ->   vector branch
```

`conv_layers` exists only on the convolutional path, so the test is exact, and it reads a key the
migration itself never writes. The image branch's encoder `Sequential` is `Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU, Flatten,
Linear, ReLU`, which is where indices 0/2/4/7 come from. Its checkpoint today carries **twenty** key
entries for **ten** parameter tensors, because `self.network = nn.Sequential(self.conv_layers,
self.fc_layers)` registers every one of them a second time; the migration keeps the canonical set and
discards the aliases, which is why the row above says "dropped" rather than mapping them.

**Parameter enumeration order is preserved by this table, and that is load-bearing for a second
reason**: an optimizer's `state_dict` keys its moment buffers by parameter *index*, so a v2 optimizer
state only reattaches to the right tensors if `parameters()` yields them in the same order. It does:
today's order is `conv_layers.{0,2,4}` then `fc_layers.{1,3}` (the `network` alias adds nothing, since
`parameters()` de-duplicates), and the new order is `encoder.layers.{0,2,4,7}` then `head` — the same
sequence.

The map lives beside the version dispatch and is tested against real pre-refactor checkpoints, which
must be **captured before any source changes** — see research R8, which owns that capture step.

The alternative — contorting the rebuilt modules to reproduce the old attribute paths, duplicate
registrations included — was rejected: it preserves a quirk nobody wants, in code every future model
inherits, to avoid writing a dozen-line mapping once.

Everything else is unchanged: a deep-model checkpoint written before this feature loads after it, and
one written after differs only in that key's internal shape, in those parameter names, and in the
version that announces both.

This is a contract and not merely a design note because User Story 2's third acceptance scenario
tests it from the outside: an existing `model.json` must still load and still render under
`hercule play`. The golden fixture does not cover it — a fixture is written by fresh training and
never exercises the loading path.
