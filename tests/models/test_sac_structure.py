"""Structural properties of SAC that no score or size criterion can detect.

Each of these is asserted directly because the alternative is asserting nothing.
A wrong target entropy produces a temperature that moves and a curve that rises; a
shared feature extractor makes the checkpoint *smaller* and the curve no worse in
the short run; a sampling evaluator merely reports lower scores. None of them
announces itself through the outcomes a benchmark records.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.environnements.oracle import ENVIRONMENT_ID
from hercule.models.sac import SACModel


def _configured(environment_id: str = ENVIRONMENT_ID, **overrides) -> SACModel:
    """A SAC configured on one environment."""
    model = SACModel()
    env = gym.make(environment_id)
    assert model.configure(env, {"seed": 3, "learning_starts": 0, **overrides})
    model.env = env
    return model


# ----------------------------------------------------------- SC-014, target entropy


@pytest.mark.unit
def test_the_target_entropy_is_the_negative_action_dimension() -> None:
    """Asserted on a TWO-dimensional action space, so `-d` is distinguishable from `-1`.

    On `Pendulum-v1` the two coincide, and a model hard-coding `-1` would pass. The
    oracle has two action dimensions precisely so this assertion can tell them apart.
    """
    model = _configured()
    assert model._action_dimensions == 2
    assert model._target_entropy == pytest.approx(-2.0)
    assert model._target_entropy != pytest.approx(-1.0)


@pytest.mark.unit
def test_the_target_entropy_is_not_a_hyperparameter() -> None:
    """Exposing it would let a grid sweep contradict the requirement that fixes it."""
    assert "target_entropy" not in SACModel().get_default_hyperparameters()


@pytest.mark.unit
def test_the_target_entropy_follows_the_action_space() -> None:
    """A one-dimensional environment gets `-1`, so the value is derived and not fixed."""
    assert _configured("Pendulum-v1")._target_entropy == pytest.approx(-1.0)


# ------------------------------------------------ SC-015, three disjoint extractors


@pytest.mark.unit
def test_the_actor_and_both_estimators_hold_pairwise_disjoint_parameters() -> None:
    """All THREE pairs, not just the two estimators.

    Sharing between the two estimators collapses the decorrelation that taking the
    lesser of them exists to provide. Sharing between the actor and either estimator
    has a different failure mode -- representation interference between two objectives
    pulling in different directions -- and is forbidden by the same decision. Neither
    is detectable by any other criterion here: sharing makes the checkpoint SMALLER,
    so no size bar can catch it, and the curve is no worse in the short run.
    """
    model = _configured()
    groups = {
        "actor": {id(p) for p in model._actor.parameters()},
        "critic_1": {id(p) for p in model._critic_1.parameters()},
        "critic_2": {id(p) for p in model._critic_2.parameters()},
    }
    assert all(groups.values()), "a module has no parameters, so disjointness is trivially true"

    names = sorted(groups)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = groups[first] & groups[second]
            assert not shared, f"{first} and {second} share {len(shared)} parameter tensors"


@pytest.mark.unit
def test_the_delayed_copies_are_separate_tensors_from_their_live_counterparts() -> None:
    """Equal in value at construction, but never the same objects."""
    model = _configured()
    live = {id(p) for p in model._critic_1.parameters()}
    delayed = {id(p) for p in model._critic_1_target.parameters()}
    assert not (live & delayed)
    for live_parameter, delayed_parameter in zip(
        model._critic_1.parameters(), model._critic_1_target.parameters(), strict=True
    ):
        assert torch.equal(live_parameter, delayed_parameter), "a delayed copy must start equal to its live network"


@pytest.mark.unit
def test_there_is_no_delayed_copy_of_the_actor() -> None:
    """It exists to be smoothed in the deterministic-actor algorithms, not here.

    Carrying one would mean a network trained, averaged, checkpointed and never read
    -- and it would show up in the checkpoint, which is where the cost lands.
    """
    model = _configured()
    assert set(model._networks()) == {"actor", "critic_1", "critic_2", "critic_1_target", "critic_2_target"}
    assert not any("actor" in name and "target" in name for name in model._networks())


# ------------------------------------------------- SC-017, deterministic evaluation


@pytest.mark.unit
def test_evaluation_is_deterministic_and_training_is_not() -> None:
    """Both halves matter: a deterministic trainer explores nothing."""
    model = _configured()
    observation = np.array([0.4, 3.1], dtype=np.float32)

    evaluated = [model.act(observation, training=False) for _ in range(5)]
    for action in evaluated[1:]:
        assert np.allclose(action, evaluated[0]), "evaluation sampled instead of taking the policy's mode"

    model._step_count = 10_000  # past any warmup, so this is the policy and not the uniform draw
    trained = [model.act(observation, training=True) for _ in range(5)]
    assert any(not np.allclose(action, trained[0]) for action in trained[1:]), "training did not explore"


@pytest.mark.unit
def test_evaluation_consumes_no_random_stream() -> None:
    """The stronger form: a deterministic action needs no draw at all.

    Checkpointed RNG state is part of what a resume restores, so an evaluation pass
    that quietly advanced it would make a resumed run diverge from an uninterrupted
    one for reasons having nothing to do with training.
    """
    model = _configured()
    observation = np.array([0.4, 3.1], dtype=np.float32)

    before = torch.get_rng_state().clone()
    model.act(observation, training=False)
    assert torch.equal(torch.get_rng_state(), before)


# ------------------------------------------------------------- bounds and mapping


@pytest.mark.unit
def test_every_action_submitted_is_within_the_environment_bounds() -> None:
    """Over a full episode, in training and in evaluation, warmup included."""
    model = _configured(learning_starts=25)
    space = model.env.action_space

    for training in (True, False):
        model.run_epoch(train_mode=training)

    model.begin_episode()
    observation, _ = model.env.reset(seed=0)
    for _ in range(50):
        action = model.predict(observation)
        assert space.contains(np.asarray(action, dtype=np.float32)), f"{action} is outside {space}"
        observation, _, terminated, truncated, _ = model.env.step(action)
        if terminated or truncated:
            break
