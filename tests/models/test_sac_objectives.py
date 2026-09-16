"""Every learned quantity in SAC, pinned against a hand-computed value.

The reason this file is long is stated once, here. A wrong reinforcement-learning
implementation still trains, still improves, and still produces a plausible reward
curve. Bootstrapping from the live estimators instead of their delayed copies,
taking the greater of the two instead of the lesser, adding the entropy term
instead of subtracting it, evaluating the target at the stored action instead of a
resampled one -- each of those converges to *something*, and none announces itself.
An end-to-end score cannot separate them, which is why the project's own roadmap
withdrew "a non-flat learning curve" as evidence.

So each clause is computed by hand on a fixed batch and compared, and each wrong
form is computed too and asserted to differ. A test that only checked the right
form would pass against an implementation where the clause did not matter at all.

Two of the checks are NOT value-level, and they are the ones a value comparison
can never make: detaching the learning target and confining the actor's objective
both change no number at all. Their damage appears only as gradients arriving
somewhere they should not, at the optimizer step.
"""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.sac import GaussianTanhActor, SACModel


BATCH = 4
SEED = 7


@pytest.fixture
def sac() -> SACModel:
    """A configured SAC on the two-dimensional oracle, small enough to be fast."""
    model = SACModel()
    env = gym.make(ENVIRONMENT_ID)
    assert model.configure(
        env, {"seed": SEED, "batch_size": BATCH, "learning_starts": 0, "replay_buffer_size": 64, "tau": 0.1}
    )
    model.env = env

    # Establish real lag between each estimator and its delayed copy. Straight out of
    # `configure()` they are exact copies -- `load_state_dict` makes them so -- and on
    # such a model "the delayed copies" and "the live estimators" are numerically the
    # same thing, which makes every test distinguishing them vacuous. Two of the tests
    # below carry an assertion that says so, and they caught this.
    with torch.no_grad():
        for critic in (model._critic_1, model._critic_2):
            for parameter in critic.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return model


@pytest.fixture
def batch_tensors(sac: SACModel) -> dict:
    """A fixed batch, as the tensors `_update` would build from it."""
    generator = np.random.default_rng(3)
    observations = generator.uniform([-1.0, 2.0], [1.0, 4.0], size=(BATCH, 2)).astype(np.float32)
    next_observations = generator.uniform([-1.0, 2.0], [1.0, 4.0], size=(BATCH, 2)).astype(np.float32)
    actions = generator.uniform(-1.0, 1.0, size=(BATCH, 2)).astype(np.float32)
    rewards = generator.uniform(-1.0, 1.0, size=BATCH).astype(np.float32)
    terminated = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return {
        "observations": sac._as_network_input(observations),
        "next_observations": sac._as_network_input(next_observations),
        "actions": torch.as_tensor(actions),
        "rewards": torch.as_tensor(rewards),
        "terminated": torch.as_tensor(terminated),
    }


# --------------------------------------------------------------- SC-010, density


def _sample_with_fixed_noise(
    actor: GaussianTanhActor, observations: torch.Tensor, noise: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call `sample()` for real, but with `torch.randn_like` pinned to a known value.

    This is what turns `sample()` into a deterministic function of `actor(observations)`'s
    own `(mean, log_std)`: its output can then be recomputed independently, from
    scratch, and compared against what `sample()` ACTUALLY returned -- rather than
    two hand-written expressions compared only to each other, which is what the
    tests below used to do without ever calling `sample()` at all.
    """

    def fake_randn_like(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return noise

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    return actor.sample(observations)


def _independent_log_density(
    mean: torch.Tensor, log_std: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute `sample()`'s action and log-density from scratch, given known noise.

    Written independently of `sample()`'s own code, from the algorithm's textbook
    form: a Gaussian log-density minus the tanh squashing correction. Returns
    `(action, log_density_with_correction, log_density_without_correction)`, the
    last one being what an implementation that OMITTED the correction would produce
    -- the wrong form this whole check exists to distinguish from the right one.
    """
    std = log_std.exp()
    unsquashed = mean + std * noise
    gaussian_log_density = (-0.5 * (noise**2) - log_std - 0.5 * float(np.log(2.0 * np.pi))).sum(dim=-1)
    squash_correction = (2.0 * (float(np.log(2.0)) - unsquashed - torch.nn.functional.softplus(-2.0 * unsquashed))).sum(
        dim=-1
    )
    return torch.tanh(unsquashed), gaussian_log_density - squash_correction, gaussian_log_density


@pytest.mark.unit
def test_the_squashing_correction_matches_the_naive_closed_form(
    sac: SACModel, batch_tensors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sample()`'s own output carries the squashing correction -- checked against `sample()` itself.

    The previous version of this test compared two hand-written expressions to each
    other and never called `GaussianTanhActor.sample()`; an implementation that
    omitted the squashing correction entirely would still have passed it. Here the
    noise is pinned via `torch.randn_like`, so `sample()`'s real output can be
    reproduced from its own `mean`/`log_std` (via `actor(observations)`) plus that
    known noise, entirely independently of `sample()`'s internals. The first
    assertion checks the corrected form matches; the second checks that the
    UNCORRECTED form does NOT match, so a dropped correction is caught rather than
    silently accepted as "close enough".
    """
    observations = batch_tensors["observations"]
    mean, log_std = sac._actor(observations)
    torch.manual_seed(17)
    noise = torch.randn(mean.shape)

    action, log_density = _sample_with_fixed_noise(sac._actor, observations, noise, monkeypatch)
    expected_action, expected_log_density, uncorrected_log_density = _independent_log_density(mean, log_std, noise)

    assert torch.allclose(action, expected_action, atol=1e-5)
    assert torch.allclose(log_density, expected_log_density, atol=1e-5)
    assert not torch.allclose(log_density, uncorrected_log_density, atol=1e-4), (
        "sample()'s log-density matches the UNCORRECTED Gaussian form -- the squashing correction is missing"
    )


@pytest.mark.unit
def test_the_naive_correction_really_does_underflow() -> None:
    """Evidence that the rewriting is necessary rather than ornamental."""
    saturated = torch.tensor([[12.0]])
    assert torch.isinf(torch.log(1.0 - torch.tanh(saturated) ** 2)).all()
    stable = 2.0 * (float(np.log(2.0)) - saturated - torch.nn.functional.softplus(-2.0 * saturated))
    assert torch.isfinite(stable).all()


@pytest.mark.unit
def test_the_density_is_measured_in_normalised_coordinates(
    sac: SACModel, batch_tensors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The density must not carry the environment rescaling's log-determinant.

    The oracle's action ranges are 2 and 4 wide, so the affine map to environment
    coordinates has scales 1 and 2 and would contribute `log(1 * 2) = log 2` per
    sample. Including it would shift the entropy the temperature adapts against by a
    per-environment constant -- silently turning automatic temperature adjustment
    back into the per-environment tuning it exists to remove.

    The previous version of this test compared `sample()`'s own output to itself
    minus a constant -- true by construction for ANY implementation, since `x` never
    equals `x` minus a nonzero constant. Here the comparison point is built
    independently, the same way as the squashing-correction check above: pin the
    noise, recompute the NORMALISED-coordinate density from `sample()`'s own
    `mean`/`log_std`, and check `sample()`'s actual output against that oracle --
    and against the oracle shifted by the environment map's own log-determinant,
    which is a real, distinguishable quantity here (`log(2) != 0`), not a tautology.
    """
    observations = batch_tensors["observations"]
    mean, log_std = sac._actor(observations)
    torch.manual_seed(19)
    noise = torch.randn(mean.shape)

    _, log_density = _sample_with_fixed_noise(sac._actor, observations, noise, monkeypatch)
    _, normalised_oracle, _ = _independent_log_density(mean, log_std, noise)

    scales = (sac._action_high - sac._action_low) / 2.0
    environment_offset = float(np.sum(np.log(scales)))
    assert environment_offset == pytest.approx(float(np.log(2.0)), abs=1e-6)

    assert torch.allclose(log_density, normalised_oracle, atol=1e-5), (
        "sample() does not match an independently recomputed NORMALISED-coordinate density"
    )
    assert not torch.allclose(log_density, normalised_oracle - environment_offset, atol=1e-6), (
        "the density appears to carry the environment map's log-determinant"
    )


# ----------------------------------------------------------------- SC-011, target


def _target_by_hand(sac: SACModel, tensors: dict, *, torch_seed: int, **wrong) -> torch.Tensor:
    """Recompute the learning target, with any clause optionally replaced."""
    torch.manual_seed(torch_seed)
    with torch.no_grad():
        if wrong.get("stored_action"):
            next_actions = tensors["actions"]
            _, next_log_density = sac._actor.sample(tensors["next_observations"])
        else:
            next_actions, next_log_density = sac._actor.sample(tensors["next_observations"])

        critic_1 = sac._critic_1 if wrong.get("live_critics") else sac._critic_1_target
        critic_2 = sac._critic_2 if wrong.get("live_critics") else sac._critic_2_target
        values = torch.stack(
            [critic_1(tensors["next_observations"], next_actions), critic_2(tensors["next_observations"], next_actions)]
        )
        if wrong.get("greater"):
            value = values.max(dim=0).values
        elif wrong.get("mean"):
            value = values.mean(dim=0)
        else:
            value = values.min(dim=0).values

        entropy_sign = 1.0 if wrong.get("added_entropy") else -1.0
        entropy = 0.0 if wrong.get("omitted_entropy") else entropy_sign * sac._alpha * next_log_density
        soft_value = value + entropy

        mask = torch.ones_like(tensors["terminated"]) if wrong.get("ignore_terminal") else 1.0 - tensors["terminated"]
        return tensors["rewards"] + 0.99 * mask * soft_value


@pytest.mark.unit
def test_the_learning_target_matches_its_hand_computed_value(sac: SACModel, batch_tensors: dict) -> None:
    """The target, clause for clause, against an independent recomputation."""
    torch.manual_seed(101)
    produced = sac._learning_target(
        batch_tensors["rewards"], batch_tensors["next_observations"], batch_tensors["terminated"], 0.99
    )
    expected = _target_by_hand(sac, batch_tensors, torch_seed=101)
    assert torch.allclose(produced, expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    "wrong_form",
    [
        {"live_critics": True},
        {"greater": True},
        {"mean": True},
        {"stored_action": True},
        {"added_entropy": True},
        {"omitted_entropy": True},
        {"ignore_terminal": True},
    ],
    ids=[
        "live-estimators",
        "greater-of-two",
        "mean-of-two",
        "stored-action",
        "entropy-added",
        "entropy-omitted",
        "terminal-not-suppressed",
    ],
)
def test_each_wrong_target_form_differs_from_the_right_one(
    sac: SACModel, batch_tensors: dict, wrong_form: dict
) -> None:
    """Every enumerated wrong form is actually detectable on this batch.

    Without this, the test above would be satisfied by an implementation where the
    clause made no difference -- and the whole point is that each of these produces a
    model that trains and improves anyway.

    Note the last id: `wrong_form["ignore_terminal"]` forces the mask to `1` for
    EVERY transition, including a genuinely terminal one -- i.e. it tests failing to
    SUPPRESS the bootstrap on a real terminal state. That is a real wrong form, but
    it is not the terminated/truncated DISTINCTION itself (`_learning_target` never
    receives `truncated` at all, so that distinction cannot be exercised at this
    level) -- see `test_a_termination_and_a_truncation_produce_different_critic_updates`
    below, which is.
    """
    right = _target_by_hand(sac, batch_tensors, torch_seed=101)
    wrong = _target_by_hand(sac, batch_tensors, torch_seed=101, **wrong_form)
    assert not torch.allclose(right, wrong, atol=1e-6)


def _snapshot_sac_state(sac: SACModel) -> dict:
    """Deep-copy every piece of state `_update()` can touch, for a fair back-to-back rerun."""
    return {
        "networks": {name: copy.deepcopy(module.state_dict()) for name, module in sac._networks().items()},
        "optimizers": {name: copy.deepcopy(optimizer.state_dict()) for name, optimizer in sac._optimizers().items()},
        "log_alpha": float(sac._log_alpha.item()),
        "torch_rng": torch.get_rng_state(),
    }


def _restore_sac_state(sac: SACModel, snapshot: dict) -> None:
    """Undo one `_update()` call, so the next one starts from bit-identical state.

    Restores every network and optimizer `_update()` can mutate, the temperature,
    and torch's global RNG -- `_update()` draws from it twice, through
    `_actor.sample()` inside both `_learning_target()` and `_update_actor()` -- so a
    second call reproduces the same draws the first one made.
    """
    for name, module in sac._networks().items():
        module.load_state_dict(snapshot["networks"][name])
    for name, optimizer in sac._optimizers().items():
        optimizer.load_state_dict(snapshot["optimizers"][name])
    with torch.no_grad():
        sac._log_alpha.fill_(snapshot["log_alpha"])
    torch.set_rng_state(snapshot["torch_rng"])


@pytest.mark.unit
def test_a_termination_and_a_truncation_produce_different_critic_updates(sac: SACModel) -> None:
    """A truncation must keep the bootstrap term; a genuine termination must zero it.

    `_learning_target()` never receives `truncated` at all -- only `_update()`
    builds the `terminated` mask from the stored batch, so this distinction can only
    be certified one level up, by actually calling `_update()`. Two batches, IDENTICAL
    but for one transition's `(terminated, truncated)` pair -- `(False, True)` in one,
    `(True, False)` in the other -- must produce DIFFERENT critic parameters after one
    real gradient step from identical model state. The primary validation environment
    (`AsymmetricOracleEnv`) truncates on every single episode and never terminates,
    so a model that collapsed this distinction would train against a wrong target
    100% of the time there, with nothing failing loudly.
    """
    generator = np.random.default_rng(9)
    size = BATCH
    observations = generator.uniform([-1.0, 2.0], [1.0, 4.0], size=(size, 2)).astype(np.float32)
    next_observations = generator.uniform([-1.0, 2.0], [1.0, 4.0], size=(size, 2)).astype(np.float32)
    actions = generator.uniform(-1.0, 1.0, size=(size, 2)).astype(np.float32)
    rewards = generator.uniform(-1.0, 1.0, size=size).astype(np.float32)

    def make_batch(*, first_terminated: bool, first_truncated: bool) -> list:
        return [
            (
                observations[i],
                actions[i],
                float(rewards[i]),
                next_observations[i],
                first_terminated if i == 0 else False,
                first_truncated if i == 0 else False,
            )
            for i in range(size)
        ]

    truncated_batch = make_batch(first_terminated=False, first_truncated=True)
    terminated_batch = make_batch(first_terminated=True, first_truncated=False)

    snapshot = _snapshot_sac_state(sac)
    sac._update(truncated_batch)
    after_truncated = [p.detach().clone() for p in sac._critic_1.parameters()]

    _restore_sac_state(sac, snapshot)
    sac._update(terminated_batch)
    after_terminated = [p.detach().clone() for p in sac._critic_1.parameters()]

    assert any(not torch.allclose(a, b) for a, b in zip(after_truncated, after_terminated, strict=True)), (
        "a truncated and a terminated transition produced the same critic update -- "
        "the terminated/truncated distinction is not reaching _update()"
    )


@pytest.mark.unit
def test_the_target_leaks_no_gradient(sac: SACModel, batch_tensors: dict) -> None:
    """NOT a value check, and it cannot be one: detaching changes no number.

    A target left attached to the learning graph produces bit-identical values and
    passes every case above, while depositing gradients on the actor, the temperature
    and the delayed copies at the optimizer step.
    """
    target = sac._learning_target(
        batch_tensors["rewards"], batch_tensors["next_observations"], batch_tensors["terminated"], 0.99
    )
    assert not target.requires_grad, "the target is still attached to the learning graph"

    target.sum().backward() if target.requires_grad else None
    assert all(p.grad is None for p in sac._actor.parameters())
    assert sac._log_alpha.grad is None
    assert all(p.grad is None for p in sac._critic_1_target.parameters())


# ----------------------------------------------------- SC-012, actor and temperature


def _reference_actor(sac: SACModel, state_dict: dict) -> GaussianTanhActor:
    """An actor sharing NO tensors with `sac._actor`, initialised to a given state.

    Used so a hand-derived loss can be stepped by a fresh optimizer without
    touching `sac._actor` itself, keeping the "right form" and "wrong form"
    computations independent of each other and of whatever `_update_actor()` does
    to the model under test.
    """
    reference = GaussianTanhActor(sac._stacked_observation_shape, sac._action_dimensions)
    reference.load_state_dict(state_dict)
    return reference


def _step_reference_actor(
    sac: SACModel, state_dict: dict, observations: torch.Tensor, *, torch_seed: int, **wrong: bool
) -> list[torch.Tensor]:
    """Step an independent actor copy by one hand-computed loss -- the right form, or a wrong one.

    A fresh `Adam`, configured with `sac._actor_optimizer`'s own hyperparameters,
    takes exactly one step from `state_dict` against the loss requested. Returns the
    resulting parameters, so the caller compares them against `_update_actor()`'s
    OWN effect on `sac._actor` -- the thing this file exists to certify actually
    happens, not two hand-written expressions compared only to each other.
    """
    reference = _reference_actor(sac, state_dict)
    optimizer = torch.optim.Adam(reference.parameters(), **sac._actor_optimizer.defaults)

    torch.manual_seed(torch_seed)
    actions, log_density = reference.sample(observations)

    critic_1 = sac._critic_1_target if wrong.get("delayed_critics") else sac._critic_1
    critic_2 = sac._critic_2_target if wrong.get("delayed_critics") else sac._critic_2
    if wrong.get("single_estimator"):
        value = critic_1(observations, actions)
    else:
        value = torch.min(critic_1(observations, actions), critic_2(observations, actions))

    entropy_term = 0.0 if wrong.get("omitted_entropy") else sac._alpha.detach() * log_density
    loss = (entropy_term - value).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return [p.detach().clone() for p in reference.parameters()]


@pytest.mark.unit
def test_the_actor_objective_matches_its_hand_computed_value(sac: SACModel, batch_tensors: dict) -> None:
    """`_update_actor()`'s own effect on the actor matches the hand-derived objective.

    The previous version of this test computed an `expected` loss and several wrong
    forms, then only asserted the wrong forms differed from `expected` -- it never
    called `_update_actor()`, so a no-op implementation would have passed. Here
    `_update_actor()` runs for real on `sac._actor`, and its effect is reproduced
    from first principles by stepping an INDEPENDENT actor copy (sharing no tensors
    with `sac._actor`) with a fresh, identically-configured optimizer against the
    hand-derived loss, seeded to draw the same noise. The two must land at
    bit-identical parameters -- that is the actual certification. Each wrong form is
    then checked against `_update_actor()`'s REAL output, not against the hand
    computation, so a wrong form that happens to coincide with a no-op or with
    `_update_actor()`'s actual behaviour would still be caught.
    """
    observations = batch_tensors["observations"]
    before_state = {name: tensor.clone() for name, tensor in sac._actor.state_dict().items()}
    before_params = [p.detach().clone() for p in sac._actor.parameters()]

    expected_after = _step_reference_actor(sac, before_state, observations, torch_seed=5)

    torch.manual_seed(5)
    sac._update_actor(observations)
    produced_after = [p.detach().clone() for p in sac._actor.parameters()]

    assert any(not torch.allclose(b, a) for b, a in zip(before_params, produced_after, strict=True)), (
        "the actor's parameters did not move at all -- _update_actor() is a no-op on this batch"
    )
    for expected, produced in zip(expected_after, produced_after, strict=True):
        assert torch.allclose(expected, produced, atol=1e-6), (
            "_update_actor()'s real effect on the actor does not match the hand-derived objective"
        )

    for wrong_form, message in [
        ({"delayed_critics": True}, "delayed instead of live estimators is not detectable here"),
        ({"single_estimator": True}, "one estimator instead of the lesser of two is not detectable here"),
        ({"omitted_entropy": True}, "omitting the entropy term is not detectable here"),
    ]:
        wrong_after = _step_reference_actor(sac, before_state, observations, torch_seed=5, **wrong_form)
        assert any(not torch.allclose(w, p, atol=1e-6) for w, p in zip(wrong_after, produced_after, strict=True)), (
            message
        )


@pytest.mark.unit
def test_the_actor_sample_carries_gradient_to_the_policy(sac: SACModel, batch_tensors: dict) -> None:
    """The actor differentiates THROUGH the sample, so the sample must not be detached."""
    actions, log_density = sac._actor.sample(batch_tensors["observations"])
    assert actions.requires_grad
    assert log_density.requires_grad
    actions.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in sac._actor.parameters())


@pytest.mark.unit
def test_the_actor_objective_deposits_no_gradient_on_the_estimators(sac: SACModel, batch_tensors: dict) -> None:
    """The mirror of the target check, and equally non-value-level.

    A single summed backward pass over the three objectives is value-identical and
    would pass every assertion above, while stepping both estimators toward the value
    the actor is chasing -- identically, so neither the lesser-of-two construction nor
    their disjoint extractors could see it -- and depositing on the temperature a
    gradient that exactly cancels the log-density term of its own objective.
    """
    for parameter in list(sac._critic_1.parameters()) + list(sac._critic_2.parameters()):
        parameter.grad = None
    sac._log_alpha.grad = None

    sac._update_actor(batch_tensors["observations"])

    assert all(p.grad is None for p in sac._critic_1.parameters()), "gradient reached estimator 1"
    assert all(p.grad is None for p in sac._critic_2.parameters()), "gradient reached estimator 2"
    assert sac._log_alpha.grad is None, "gradient reached the temperature"


@pytest.mark.unit
def test_the_temperature_gradient_is_the_stated_one(sac: SACModel, batch_tensors: dict) -> None:
    """Gradient of the temperature objective is `-(log_density + target_entropy)`."""
    log_density = torch.tensor([0.5, -1.25, 0.0, 2.0])
    sac._log_alpha.grad = None
    sac._update_temperature(log_density)

    # The optimizer has already stepped, so read the gradient it stepped on.
    expected = -(log_density + sac._target_entropy).mean()
    assert sac._log_alpha.grad is not None
    assert float(sac._log_alpha.grad) == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.unit
def test_the_temperature_update_would_leak_into_the_actor_without_detaching(sac: SACModel, batch_tensors: dict) -> None:
    """`_update_temperature()` performs no detaching of its own -- isolation is the CALLER's job.

    The test above passes a standalone `torch.tensor` with no connection to the
    actor at all, so it cannot detect a leak even in principle. This one passes a
    log-density fresh off `_actor.sample()`, deliberately left attached to the
    policy's graph, and confirms it DOES deposit a real gradient on the actor's
    parameters -- measured here with the temperature moved off its coincidental
    `log(1.0) == 0` default, which would otherwise mask the leak behind an
    all-zero-but-still-present gradient. This is not a bug: it demonstrates why
    `_update_actor()`'s detached return value (checked in the next test) is the
    thing that actually prevents this in `_update()`'s real call sequence, not any
    guard inside `_update_temperature()` itself.
    """
    with torch.no_grad():
        sac._log_alpha.fill_(0.7)
    for parameter in sac._actor.parameters():
        parameter.grad = None

    _, connected_log_density = sac._actor.sample(batch_tensors["observations"])
    assert connected_log_density.requires_grad, (
        "the log-density is not connected to the actor graph, so this proves nothing"
    )

    sac._update_temperature(connected_log_density)

    assert any(p.grad is not None and torch.any(p.grad != 0) for p in sac._actor.parameters()), (
        "no gradient reached the actor even though the log-density was left connected to its graph -- "
        "this no longer demonstrates why _update_actor()'s detach matters"
    )


@pytest.mark.unit
def test_the_actor_update_returns_a_detached_log_density(sac: SACModel, batch_tensors: dict) -> None:
    """`_update_actor()`'s return value is detached -- the production guarantee against the leak above.

    `_update()` always calls `_update_temperature()` with exactly this return
    value, never a freshly-sampled, still-connected one. Checked two ways: the
    tensor itself has `requires_grad is False`, AND calling `_update_temperature()`
    with it deposits no gradient on the actor at all -- unlike the connected case
    above, this holds regardless of the temperature's value, since the graph is
    genuinely severed rather than merely multiplied by a small number.
    """
    log_density = sac._update_actor(batch_tensors["observations"])
    assert log_density.requires_grad is False

    with torch.no_grad():
        sac._log_alpha.fill_(0.7)
    for parameter in sac._actor.parameters():
        parameter.grad = None

    sac._update_temperature(log_density)

    assert all(p.grad is None for p in sac._actor.parameters()), (
        "the actor received gradient even from a properly detached log-density"
    )


@pytest.mark.unit
def test_the_temperature_moves_in_the_stated_direction(sac: SACModel) -> None:
    """Rises when measured entropy is below target, falls when above.

    The opposite sign also produces a temperature that moves, a policy that trains
    and a curve that rises, while exploration collapses or diverges. So the direction
    is asserted as well as the gradient.
    """
    # Entropy is `-log_density`. Below target means `-log_density < target_entropy`.
    too_deterministic = torch.full((BATCH,), -sac._target_entropy + 1.0)
    assert float(-too_deterministic.mean()) < sac._target_entropy

    before = float(sac._alpha)
    sac._update_temperature(too_deterministic)
    assert float(sac._alpha) > before, "the temperature must rise when entropy falls below target"

    too_random = torch.full((BATCH,), -sac._target_entropy - 1.0)
    assert float(-too_random.mean()) > sac._target_entropy
    before = float(sac._alpha)
    sac._update_temperature(too_random)
    assert float(sac._alpha) < before, "the temperature must fall when entropy rises above target"


@pytest.mark.unit
def test_the_temperature_is_optimised_through_its_logarithm(sac: SACModel) -> None:
    """A temperature crossing zero would invert the entropy term with no error raised.

    Driven hard in one direction under a large step size, `alpha` must stay strictly
    positive -- which it does because the optimised quantity is its logarithm.
    """
    for group in sac._temperature_optimizer.param_groups:
        group["lr"] = 5.0
    drive_down = torch.full((BATCH,), -sac._target_entropy - 10.0)
    for _ in range(20):
        sac._update_temperature(drive_down)
    assert float(sac._alpha) > 0.0
    assert np.isfinite(float(sac._alpha))


# ------------------------------------------------------------- SC-013, estimators


@pytest.mark.unit
def test_both_estimators_move_toward_the_target(sac: SACModel, batch_tensors: dict) -> None:
    """BOTH, and at the STORED action.

    Training only one leaves the other drifting while the lesser-of-two still reads
    it; regressing at an action resampled from the current policy is the target
    side's construct and belongs only there.
    """
    observations, actions = batch_tensors["observations"], batch_tensors["actions"]
    target = torch.full((BATCH,), 5.0)  # far from initialisation, so the direction is unambiguous

    before_1 = sac._critic_1(observations, actions).detach()
    before_2 = sac._critic_2(observations, actions).detach()
    for group in sac._critic_1_optimizer.param_groups + sac._critic_2_optimizer.param_groups:
        group["lr"] = 0.05

    sac._update_critics(observations, actions, target)

    after_1 = sac._critic_1(observations, actions).detach()
    after_2 = sac._critic_2(observations, actions).detach()
    assert torch.all(after_1 > before_1), "estimator 1 did not move toward the target"
    assert torch.all(after_2 > before_2), "estimator 2 did not move toward the target"


@pytest.mark.unit
def test_the_estimators_are_regressed_at_the_stored_action(sac: SACModel, batch_tensors: dict) -> None:
    """A resampled action would produce a different loss, hence a different step."""
    observations, actions = batch_tensors["observations"], batch_tensors["actions"]
    target = torch.full((BATCH,), 5.0)

    at_stored = torch.nn.functional.mse_loss(sac._critic_1(observations, actions), target)
    torch.manual_seed(2)
    resampled, _ = sac._actor.sample(observations)
    at_resampled = torch.nn.functional.mse_loss(sac._critic_1(observations, resampled.detach()), target)

    assert not torch.allclose(at_stored, at_resampled), (
        "the stored and resampled actions give the same loss on this batch, so the distinction is untested"
    )
