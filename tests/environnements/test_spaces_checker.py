"""Tests for the `SpaceKind` taxonomy and space-classification helpers (spec 006, S02).

`check_space_is_box` used to return `not check_space_is_discrete(space)`, which read
every non-`Discrete` space -- `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple` included --
as a `Box`. These tests pin the corrected behaviour and the `classify_space` pairs the
`Supervisor` gates on.
"""

import gymnasium as gym
import pytest

from hercule.environnements.spaces_checker import SpaceKind, check_space_is_box, classify_space


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_id", "env_kwargs", "expected_pair"),
    [
        ("FrozenLake-v1", {}, (SpaceKind.DISCRETE, SpaceKind.DISCRETE)),
        ("CartPole-v1", {}, (SpaceKind.BOX, SpaceKind.DISCRETE)),
        ("Pendulum-v1", {}, (SpaceKind.BOX, SpaceKind.BOX)),
        ("CarRacing-v3", {"continuous": True}, (SpaceKind.BOX, SpaceKind.BOX)),
        ("CarRacing-v3", {"continuous": False}, (SpaceKind.BOX, SpaceKind.DISCRETE)),
    ],
)
def test_classify_space_pair_matches_environment(
    env_id: str, env_kwargs: dict, expected_pair: tuple[SpaceKind, SpaceKind]
) -> None:
    env = gym.make(env_id, **env_kwargs)
    try:
        pair = (classify_space(env.observation_space), classify_space(env.action_space))
    finally:
        env.close()

    assert pair == expected_pair


@pytest.mark.unit
@pytest.mark.parametrize(
    "space",
    [
        gym.spaces.MultiDiscrete([2, 3]),
        gym.spaces.MultiBinary(4),
        gym.spaces.Dict({"a": gym.spaces.Discrete(2)}),
        gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
    ],
)
def test_check_space_is_box_rejects_non_box_spaces(space: gym.Space) -> None:
    """The old implementation (`not is_discrete`) accepted every one of these."""
    assert check_space_is_box(space) is False


@pytest.mark.unit
def test_check_space_is_box_accepts_box() -> None:
    assert check_space_is_box(gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("space", "expected_kind"),
    [
        (gym.spaces.Discrete(4), SpaceKind.DISCRETE),
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)), SpaceKind.BOX),
        (gym.spaces.MultiDiscrete([2, 3]), SpaceKind.MULTI_DISCRETE),
        (gym.spaces.MultiBinary(4), SpaceKind.MULTI_BINARY),
        (gym.spaces.Tuple((gym.spaces.Discrete(2),)), SpaceKind.TUPLE),
        (gym.spaces.Dict({"a": gym.spaces.Discrete(2)}), SpaceKind.DICT),
    ],
)
def test_classify_space_maps_every_gymnasium_space_class(space: gym.Space, expected_kind: SpaceKind) -> None:
    assert classify_space(space) == expected_kind
