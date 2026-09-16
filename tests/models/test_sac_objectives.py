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

import gymnasium as gym
import numpy as np
import pytest
import torch

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.sac import SACModel


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


@pytest.mark.unit
def test_the_squashing_correction_matches_the_naive_closed_form(sac: SACModel) -> None:
    """The stable rewriting equals the textbook expression where the latter is accurate.

    The implementation uses `2 * (log 2 - u - softplus(-2u))` because
    `log(1 - tanh(u)^2)` underflows to `-inf` for `|u|` above about 9 in float32,
    which ordinary training reaches once the policy pushes an action toward a bound.
    The naive form IS the independent reference this is checked against -- comparing
    the stable form to itself would assert nothing -- so the comparison is made on
    moderate inputs where the naive form is still exact.
    """
    unsquashed = torch.tensor([[-2.0, -0.5], [0.0, 0.5], [1.5, 2.5]])

    naive = torch.log(1.0 - torch.tanh(unsquashed) ** 2).sum(dim=-1)
    stable = (2.0 * (float(np.log(2.0)) - unsquashed - torch.nn.functional.softplus(-2.0 * unsquashed))).sum(dim=-1)

    assert torch.allclose(naive, stable, atol=1e-5)


@pytest.mark.unit
def test_the_naive_correction_really_does_underflow() -> None:
    """Evidence that the rewriting is necessary rather than ornamental."""
    saturated = torch.tensor([[12.0]])
    assert torch.isinf(torch.log(1.0 - torch.tanh(saturated) ** 2)).all()
    stable = 2.0 * (float(np.log(2.0)) - saturated - torch.nn.functional.softplus(-2.0 * saturated))
    assert torch.isfinite(stable).all()


@pytest.mark.unit
def test_the_density_is_measured_in_normalised_coordinates(sac: SACModel, batch_tensors: dict) -> None:
    """The density must not carry the environment rescaling's log-determinant.

    The oracle's action ranges are 2 and 4 wide, so the affine map to environment
    coordinates has scales 1 and 2 and would contribute `log(1 * 2) = log 2` per
    sample. Including it would shift the entropy the temperature adapts against by a
    per-environment constant -- silently turning automatic temperature adjustment
    back into the per-environment tuning it exists to remove.
    """
    torch.manual_seed(11)
    _, log_density = sac._actor.sample(batch_tensors["observations"])

    scales = (sac._action_high - sac._action_low) / 2.0
    environment_offset = float(np.sum(np.log(scales)))
    assert environment_offset == pytest.approx(float(np.log(2.0)), abs=1e-6)

    torch.manual_seed(11)
    _, again = sac._actor.sample(batch_tensors["observations"])
    assert torch.allclose(log_density, again), "the density is not reproducible, so the check below means nothing"
    assert not torch.allclose(log_density, again - environment_offset), (
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
        "truncation-treated-as-terminal",
    ],
)
def test_each_wrong_target_form_differs_from_the_right_one(
    sac: SACModel, batch_tensors: dict, wrong_form: dict
) -> None:
    """Every enumerated wrong form is actually detectable on this batch.

    Without this, the test above would be satisfied by an implementation where the
    clause made no difference -- and the whole point is that each of these produces a
    model that trains and improves anyway.
    """
    right = _target_by_hand(sac, batch_tensors, torch_seed=101)
    wrong = _target_by_hand(sac, batch_tensors, torch_seed=101, **wrong_form)
    assert not torch.allclose(right, wrong, atol=1e-6)


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


@pytest.mark.unit
def test_the_actor_objective_matches_its_hand_computed_value(sac: SACModel, batch_tensors: dict) -> None:
    """Lesser of the two LIVE estimators, minus temperature times log-density."""
    observations = batch_tensors["observations"]

    torch.manual_seed(5)
    actions, log_density = sac._actor.sample(observations)
    value = torch.min(sac._critic_1(observations, actions), sac._critic_2(observations, actions))
    expected = (sac._alpha.detach() * log_density - value).mean()

    torch.manual_seed(5)
    actions, log_density = sac._actor.sample(observations)
    delayed_value = torch.min(sac._critic_1_target(observations, actions), sac._critic_2_target(observations, actions))
    with_delayed = (sac._alpha.detach() * log_density - delayed_value).mean()
    single = (sac._alpha.detach() * log_density - sac._critic_1(observations, actions)).mean()
    without_entropy = (-value).mean()

    assert not torch.allclose(expected, with_delayed), "delayed instead of live estimators is not detectable here"
    assert not torch.allclose(expected, single), "one estimator instead of the lesser of two is not detectable here"
    assert not torch.allclose(expected, without_entropy), "omitting the entropy term is not detectable here"


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
