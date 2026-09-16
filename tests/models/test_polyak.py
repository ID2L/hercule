"""The delayed copies advance gradually, on the gradient clock (SC-016).

Two properties, and the clock alone is not enough. A hard copy performed once per
gradient step satisfies the schedule perfectly while making each delayed copy
identical to its live counterpart from the first step onward -- at which point "the
lesser of the two delayed copies" in the learning target is numerically the lesser
of the two live ones, which is the first wrong form that target enumerates, reached
without violating a word of it.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.sac import SACModel


TAU = 0.25


@pytest.fixture
def sac() -> SACModel:
    """A SAC whose gradient clock and environment clock deliberately differ."""
    model = SACModel()
    env = gym.make(ENVIRONMENT_ID)
    assert model.configure(
        env,
        {
            "seed": 4,
            "tau": TAU,
            "batch_size": 2,
            "replay_buffer_size": 32,
            "learning_starts": 0,
            "step_modulo": 3,  # a gradient step every third environment step
        },
    )
    model.env = env
    return model


@pytest.mark.unit
def test_each_advance_is_the_configured_fraction(sac: SACModel) -> None:
    """Hand-computed: `delayed <- (1 - tau) * delayed + tau * live`.

    A hard copy on the right clock fails this, which is the point: it would land on
    `live` rather than a fraction of the way toward it.
    """
    with torch.no_grad():
        for parameter in sac._critic_1.parameters():
            parameter.add_(torch.randn_like(parameter))

    live_before = [p.detach().clone() for p in sac._critic_1.parameters()]
    delayed_before = [p.detach().clone() for p in sac._critic_1_target.parameters()]

    sac._polyak_update(TAU)

    for live, delayed_old, delayed_new in zip(
        live_before, delayed_before, sac._critic_1_target.parameters(), strict=True
    ):
        expected = (1.0 - TAU) * delayed_old + TAU * live
        assert torch.allclose(delayed_new, expected, atol=1e-6)
        assert not torch.allclose(delayed_new, live), "a fraction of the way is not all the way -- this is a hard copy"


@pytest.mark.unit
def test_a_moved_live_network_leaves_its_delayed_copy_behind(sac: SACModel) -> None:
    """The end-state form, under a CONTROLLED precondition.

    The preconditions are not decoration, and two earlier attempts at this check
    were wrong without them. An unconditional "they must differ" fails for a correct
    implementation whenever the live update was zero, and they may coincide
    numerically at convergence. Merely requiring that the live parameters moved is
    also not enough: a live parameter moving from 0 to 1 while its delayed copy
    already sits at 1 leaves the copy at 1 for every fraction -- equal to the
    parameter that just moved.

    So: start from delayed EQUAL to live, apply one known non-zero live update, with
    `0 < tau < 1`.
    """
    for live, delayed in zip(sac._critic_1.parameters(), sac._critic_1_target.parameters(), strict=True):
        assert torch.equal(live, delayed), "the precondition does not hold: they do not start equal"

    with torch.no_grad():
        for parameter in sac._critic_1.parameters():
            parameter.add_(1.0)  # a known, non-zero, uniform move

    assert 0.0 < TAU < 1.0
    sac._polyak_update(TAU)

    for live, delayed in zip(sac._critic_1.parameters(), sac._critic_1_target.parameters(), strict=True):
        assert not torch.allclose(live, delayed), "the delayed copy tracked the live network exactly"


@pytest.mark.unit
def test_the_copies_advance_on_the_gradient_clock_not_the_environment_clock(sac: SACModel) -> None:
    """With `step_modulo = 3`, EVERY gradient-eligible environment step produces exactly one advance.

    The previous version of this check only asserted that at least one advance
    happened and that every advance landed on a multiple of `step_modulo` -- a fact
    guaranteed by `_ready_to_update()`'s own eligibility condition regardless of how
    many times `_polyak_update()` actually runs. An implementation that advanced
    exactly ONCE during the two full episodes below would satisfy both of those
    checks (one advance is non-empty, and it necessarily lands on-clock), and so
    would one that advanced twice per eligible step.

    So this counts two things independently: every time `_ready_to_update()`
    reports a step eligible (the ground truth for how many gradient steps SHOULD
    have advanced the copies), and every time `_polyak_update()` actually runs. The
    counts, and the exact step numbers, must match -- not merely "on-clock".
    """
    eligible_steps = []
    original_ready_to_update = SACModel._ready_to_update

    def counting_ready_to_update(self) -> bool:
        ready = original_ready_to_update(self)
        if ready:
            eligible_steps.append(self._step_count)
        return ready

    advances = []
    original_polyak_update = SACModel._polyak_update

    def counting_polyak_update(self, tau: float) -> None:
        advances.append(self._step_count)
        original_polyak_update(self, tau)

    SACModel._ready_to_update = counting_ready_to_update
    SACModel._polyak_update = counting_polyak_update
    steps_before = sac._step_count
    try:
        for _ in range(2):
            sac.run_epoch(train_mode=True)
    finally:
        SACModel._ready_to_update = original_ready_to_update
        SACModel._polyak_update = original_polyak_update

    environment_steps = sac._step_count - steps_before
    assert environment_steps > len(eligible_steps) > 0, (
        "either no step was gradient-eligible, or every environment step was -- "
        "this run does not exercise the gradient-clock-vs-environment-clock distinction"
    )
    assert all(step % 3 == 0 for step in advances), f"advances happened off the gradient clock: {advances[:10]}"
    assert len(advances) == len(eligible_steps), (
        f"{len(advances)} advances for {len(eligible_steps)} gradient-eligible steps out of "
        f"{environment_steps} environment steps -- an implementation advancing once per episode, "
        "or more than once per eligible step, is not caught by an on-clock check alone"
    )
    assert advances == eligible_steps, "advances did not occur on exactly the eligible steps"


@pytest.mark.unit
def test_the_ancestor_hard_copy_never_runs_for_this_family(sac: SACModel) -> None:
    """The per-environment-step hook stays dormant; the gradual averaging is the only path.

    Both mechanisms firing would mean the delayed copies were periodically snapped
    onto the live networks, undoing whatever lag the averaging had built.
    """
    assert sac._target_sync_interval() is None

    with torch.no_grad():
        for parameter in sac._critic_1.parameters():
            parameter.add_(1.0)
    before = [p.detach().clone() for p in sac._critic_1_target.parameters()]

    for step in range(1, 50):
        sac._step_count = step
        sac._sync_targets()

    for old, new in zip(before, sac._critic_1_target.parameters(), strict=True):
        assert torch.equal(old, new), "the ancestor's hard copy fired for a continuous actor-critic model"


@pytest.mark.unit
def test_the_delayed_copies_take_no_gradient(sac: SACModel) -> None:
    """Half of the mechanism that keeps the learning target a constant for learning."""
    for parameter in list(sac._critic_1_target.parameters()) + list(sac._critic_2_target.parameters()):
        assert not parameter.requires_grad


@pytest.mark.unit
def test_a_full_training_epoch_keeps_the_copies_lagging(sac: SACModel) -> None:
    """End to end: after real training the delayed copies differ from the live ones."""
    for _ in range(3):
        sac.run_epoch(train_mode=True)

    differences = [
        float((live - delayed).abs().max())
        for live, delayed in zip(sac._critic_1.parameters(), sac._critic_1_target.parameters(), strict=True)
    ]
    assert max(differences) > 0.0, "the delayed copy is identical to the live network after training"
    assert np.isfinite(differences).all()
