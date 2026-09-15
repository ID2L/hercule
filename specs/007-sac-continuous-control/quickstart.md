# Phase 1 — Quickstart: running each validation rung

Every command below is runnable from the repository root once the feature is implemented. They are
listed cheapest first, which is also the order in which a failure is most informative: a failure at
rung 1 makes every later rung meaningless, so there is no point running rung 3 until rung 1 is green.

## Rung 0 — the unit criteria (seconds)

```bash
uv run pytest tests/models/test_action_mapping.py tests/models/test_sac_objectives.py \
              tests/models/test_polyak.py -q
```

Covers SC-003 (mapping endpoints and env-coordinate return), SC-010 to SC-013 (the density
correction, the learning target clause by clause, the actor and temperature objectives, the value
estimators' objective, and every gradient boundary), and SC-016 (the gradual averaging and its
clock). No training; hand-computed values on fixed batches.

These are the criteria that matter most and cost least. A wrong SAC that trains and improves fails
here and nowhere else.

## Rung 1 — the refactor is behaviour-preserving (seconds)

```bash
uv run pytest tests/models/test_golden_fixture.py -q
```

SC-004. Reads the committed fixture and asserts the rebuilt `DeepQLearningModel` reproduces its
reward series and weight hashes exactly, on CPU.

**If this fails with weights that differ but a reward series that looks plausible**, suspect module
construction order before suspecting the algorithm — see the plan's "The constraint that governs the
refactor". A reordered `nn.Linear` draws different numbers from the same seed.

At feature closure this file is deleted along with the fixture, and the determinism property test
takes over.

## Rung 2 — the asymmetric oracle (~1 minute)

```bash
uv run pytest tests/environnements/test_asymmetric_oracle.py -q     # the oracle is a valid oracle
uv run pytest -m slow -k oracle tests/models/test_training_bars.py  # SAC reaches 90% of optimum
uv run hercule learn experiments/sac_oracle.yaml                    # the same run, by hand
```

SC-002. The first command asserts the five closed-form rows of contract C2 — that the optimum is
50.0, and that each of the four degenerate policies scores below the 45.0 bar. The second trains for
200 episodes (10,000 environment steps) and checks the score plus zero out-of-bounds actions.

The first command is the one that matters most and is the cheapest thing in this document: it
establishes that the oracle can detect what it exists to detect. A first draft of this environment
could not — clamping its one-sided dimension still scored above the bar — and no amount of training
would have revealed that.

This is the rung that catches per-dimension and asymmetry defects. `Pendulum-v1` cannot: it has one
action dimension, symmetric.

## Rung 3 — Pendulum (~10 minutes)

```bash
uv run pytest -m slow -k pendulum tests/models/test_training_bars.py   # the asserting test
uv run hercule learn experiments/sac_pendulum.yaml                     # the same run, by hand
```

SC-001: mean test reward strictly above `-200` over 20 evaluation episodes, within 500 episodes of
training. The shipped YAML trains but asserts nothing — only the `slow` test can fail when the bar is
missed, which is why both exist. Also the first rung on which the time-limit handling matters — `Pendulum-v1` never
terminates, it truncates at 200 steps on every single episode, so an implementation that treats a
cut-off as terminal trains against a wrong target 100% of the time.

To inspect what it learned:

```bash
uv run hercule play outputs/sac_pendulum/Pendulum-v1/<env_sig>/sac/<model_sig>/model.json \
                    outputs/sac_pendulum/Pendulum-v1/<env_sig>/environment.json
```

## Rung 4 — LunarLander continuous (hours)

```bash
uv run hercule learn experiments/sac_lunarlander.yaml
```

SC-006: mean reward at least `200` over **100** evaluation episodes — the protocol that convention
attaches to that threshold, deliberately not the 20-episode protocol the roadmap's own rungs use —
within 3000 episodes.

A recorded experiment, not a suite test. Its result is committed with the feature.

## Rung 5 — CarRacing continuous (overnight)

```bash
uv run hercule learn experiments/sac_car_racing.yaml
```

SC-007: at epoch 700 — the same budget `experiments/dq_car_racing.yaml` uses, so the two runs are
directly comparable — mean test reward at least `+65.63`, which is the measured random-policy
baseline of `-34.37` plus the stated margin of 100.

The baseline was measured for the specification and is recorded there; it is not re-measured here.

Two things to watch on this rung:

- **Checkpoint size.** ~129 MB per write against a 150 MB bar (SC-008). The shipped config sets its
  checkpoint interval with that in mind; do not lower the interval without recomputing the disk cost.
- **Resume discontinuity.** Replay contents are deliberately not persisted (FR-028), so a resumed run
  refills an empty buffer and the learning curve shows a step. That is expected behaviour, not a bug,
  and is the single most likely thing to be misread as one.

## Rung 6 — everything else

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run gen-doc
```

SC-009 (the pairing refusal, in `tests/supervisor/test_space_gating.py`) and User Story 2's third
scenario (an existing `model.json` still loads and still renders, in `tests/models/test_sac_persistence.py`
per contract C4) run inside the suite. SC-018 is this command being green.

## The two things to do before touching any code

Both are captured from `main` as it stands, and **neither can be regenerated once the refactor
begins** — which is precisely what makes them evidence. A refactor started without them cannot
demonstrate FR-002 or User Story 2 at all.

```bash
uv run python tests/fixtures/capture_baselines.py
```

One script, both artifacts, run once from `main`. It lives under `tests/fixtures/` rather than at the
repository root, which `CLAUDE.md` reserves against ad-hoc scripts, and beside the files it writes.

1. **The golden fixture**, into `tests/fixtures/golden/`. Certifies that the refactor computes the
   same **numbers**. Deleted at feature closure and replaced by a determinism property test.
2. **Two real pre-refactor `model.json` files**, one per network branch, into
   `tests/fixtures/checkpoints/`. Certifies that the refactor leaves those numbers **readable** —
   contract C4's migration table renames every parameter key, and the fixture cannot see that,
   because it compares tensors and never opens a file. These are **not** deleted at closure: they
   outlive the fixture, because the migration path they test does.

The second was missing from an earlier version of this runbook, which named only the fixture. That is
the worst place for such an omission: this section is the one that says what is irreversible.
