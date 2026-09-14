# Implementation Plan: Foundations for Continuous Action Spaces

**Branch**: `006-continuous-actions-foundations` | **Date**: 2026-09-10 | **Spec**: [ROADMAP-continuous-actions.md](../ROADMAP-continuous-actions.md)
**Input**: `specs/ROADMAP-continuous-actions.md`, Phase 0 (sub-specs 0.1–0.5)

## Summary

This feature contains **no new algorithm**. It fixes five defects in the existing codebase that
would otherwise be frozen into the shared off-policy ancestor built in feature 007, and it
establishes the contracts DecQN, TD3 and SAC depend on.

Four of the five defects are silent: they produce a plausible learning curve while training against
a wrong TD target, restarting exploration from scratch on resume, destroying target-network lag, or
letting a model accept an environment it cannot handle. The fifth makes an actor-critic checkpoint
unusable on disk grounds alone.

Technically: a concrete-only model registry; an enumerable `SpaceKind` taxonomy with per-model
declared support enforced in `Supervisor`; separation of `terminated` from `truncated` in the replay
transition and the TD target; a real `seed` hyperparameter wired through torch, Python `random`, a
model-owned NumPy `Generator` and the environment; and a base64 checkpoint carrying every network,
the optimizer state, mutated hyperparameters and RNG state.

Every defect was verified against the source or reproduced by running code before being written
down; the roadmap's §0–§1 records the evidence, and every line reference in `tasks.md` was checked.

## Technical Context

**Language/Version**: Python 3.10+ (`X | Y` unions, no `typing.Any`)
**Primary Dependencies**: existing only — torch, gymnasium 1.2.3, numpy, pydantic v2, click. **No
new dependency is added by this feature.**
**Storage**: `outputs/` JSON tree. `model.json` gains a versioned encoding; the legacy list format
stays readable, so existing artifacts keep loading and `hercule play` keeps working on them.
**Testing**: pytest with `--strict-markers`; markers `unit`, `integration`, `slow`. `tests/models/`
does not exist today and is created here.
**Target Platform**: Windows 11 primary dev, cross-platform library
**Project Type**: single Python package (`src/hercule`) with a Click CLI
**Performance Goals**: a CarRacing DQN `model.json` under **15 MB**, against 133.9 MB measured on
disk today; checkpoint write under 0.5 s, against 3.95 s measured
**Constraints**: ruff-clean at line-length 120; `uv run gen-doc` must keep succeeding; no change to
any `@final` method on a Root Class Registry entry
**Scale/Scope**: 5 sub-specs, 55 tasks, 4 source packages touched (`models`, `environnements`,
`supervisor`, `run`)

## Constitution Check

*GATE: evaluated before implementation and re-evaluated at completion.*

| Principle | Verdict | Evidence |
|---|---|---|
| **I. Generic Algorithm Architecture** | **Engaged — amendment required** | S02 adds `supported_spaces: ClassVar` to `RLModel`. AGENTS.md lists "changing `ClassVar` declarations that subclasses depend on" as a semantic change. The abstract-method surface is untouched; `save()`/`load()` remain `@final` and unmodified. Bump 1.1.0 → **1.2.0** (MINOR, additive). Task T019. |
| **II. Configuration-Driven Design** | Satisfied | `seed` becomes a real hyperparameter rather than a new one; no `ParameterValue` extension; no hard-coded parameter introduced. |
| **III. Deterministic output layout** | Satisfied | No directory-layout or signature change. `model.json` changes encoding, not location. |
| **IV. Test coverage** | Satisfied | 21 of 55 tasks are tests. Four defects are silent, so a test is the only thing distinguishing fixed from unfixed. |
| **V. Resumability** | **Restored** | Resumability is a stated design property (`Runner.learn` iterates `range(learning_ongoing_epoch, max_epoch)`) that is currently broken in three ways — epsilon, optimizer state, target lag. S05 fixes all three. |
| **VI. English source** | Satisfied | All new code, comments and docs in English. |

**Root Class Registry entries touched**: `RLModel` (additive ClassVar, S02), `Runner` (additive
validation, T043 — no API change). Both require the "Constitution Impact" PR section.

## Project Structure

### Documentation (this feature)

```
specs/006-continuous-actions-foundations/
├── plan.md          # this file
└── tasks.md         # 55 tasks across 5 sub-specs
```

`spec.md` is not duplicated: `specs/ROADMAP-continuous-actions.md` is this feature's specification,
and it carries the evidence, the adversarial-review history and the phase sequencing.

### Source Code (repository root)

```
src/hercule/
├── models/
│   ├── __init__.py              # S01 registry filter; S02 supported_spaces ClassVar
│   ├── deep_q_learning/         # S03 truncation; S04 seeding; S05 checkpoints
│   ├── td_models/               # S02 declaration
│   ├── simple_q_learning/       # S02 declaration (inherited)
│   ├── simple_sarsa/            # S02 declaration (inherited)
│   └── dummy/                   # S02 declaration
├── environnements/
│   └── spaces_checker.py        # S02 SpaceKind, classify_space, corrected check_space_is_box
├── supervisor/__init__.py       # S02 pair gating before configure()
└── run/__init__.py              # S05 resume guard

tests/
├── models/                      # created here: registry, targets, determinism, persistence
├── environnements/              # spaces classification
├── supervisor/                  # space gating
└── run/                         # resume guard
```

**Structure Decision**: single project, unchanged. No new package is introduced; the shared
ancestor `OffPolicyReplayModel` belongs to feature 007, not here.

## Execution Strategy

File ownership drives parallelism, not the `[P]` markers alone: three of the five sub-specs
(S03, S04, S05) all edit `src/hercule/models/deep_q_learning/__init__.py`, so they cannot run
concurrently regardless of how their tasks are marked.

| Wave | Work | Owner |
|---|---|---|
| 0 | T001–T003 scaffolding, T010–T011 `spaces_checker.py`, T012 `RLModel` ClassVar | main session — small, foundational, every stream depends on them |
| 1a | S01 registry + S02 declarations on the three non-DQN models + Supervisor gating + tests | parallel agent |
| 1b | S03 + S04 + S05 on `deep_q_learning` + its `supported_spaces` declaration + tests | parallel agent |
| 1c | T043 + T048 `Runner` resume guard + tests | parallel agent |
| 2 | T019 constitution amendment, T050–T055 polish, full-suite verification | main session |

Wave 1's three streams touch disjoint file sets, which is what makes them safe to run
concurrently.

## Complexity Tracking

| Concern | Justification |
|---|---|
| Five sub-specs before the first new algorithm | Adjudicated unanimously in adversarial review: freezing first and fixing later either certifies the truncation bug into the shared ancestor or forces re-opening it after closure. |
| S03 and S04 change what `deep_q_learning` learns | Deliberate. Results already under `outputs/` become non-comparable with post-fix runs; the previous numbers were computed against a wrong TD target. Stated in the PR (T026–T027). |
| Replay buffer excluded from checkpoints | Storing it costs gigabytes per checkpoint. The consequence — a resumed run refills an empty buffer — is documented rather than left unsaid (T042). |
