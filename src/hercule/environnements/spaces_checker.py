"""Classification of Gymnasium observation and action spaces.

A model declares which `(observation_kind, action_kind)` pairs it supports, and the
`Supervisor` compares that declaration against the environment before configuring the
model. This exists because the previous `check_space_is_box` returned `not is_discrete`,
so every non-`Discrete` space -- `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`, `Text`,
`Sequence`, `Graph`, `OneOf` -- read as a `Box`.
"""

from enum import Enum

import gymnasium as gym


class SpaceKind(str, Enum):
    """One entry per Gymnasium space class, plus a fallback for anything unrecognised.

    The values are the class names so a message printed from a `SpaceKind` names the
    same thing the Gymnasium documentation does.

    `(str, Enum)` rather than `enum.StrEnum`: the latter is Python 3.11+, and
    `pyproject.toml` declares `requires-python = ">=3.10"`. `__str__` is defined
    explicitly because the `(str, Enum)` mixin otherwise renders as `SpaceKind.BOX`,
    which would put the enum's own name into user-facing messages instead of `Box`.
    """

    DISCRETE = "Discrete"
    BOX = "Box"
    MULTI_DISCRETE = "MultiDiscrete"
    MULTI_BINARY = "MultiBinary"
    TUPLE = "Tuple"
    DICT = "Dict"
    TEXT = "Text"
    SEQUENCE = "Sequence"
    GRAPH = "Graph"
    ONE_OF = "OneOf"
    UNKNOWN = "Unknown"

    def __str__(self) -> str:
        """Render as the bare space-class name, for messages."""
        return self.value


# Ordered most-specific first. `MultiDiscrete` and `MultiBinary` are checked before
# `Box` because a subclass relationship between them would otherwise be misread; the
# explicit order makes the classification independent of Gymnasium's own hierarchy.
_SPACE_CLASSES: tuple[tuple[type[gym.Space], SpaceKind], ...] = (
    (gym.spaces.Discrete, SpaceKind.DISCRETE),
    (gym.spaces.MultiDiscrete, SpaceKind.MULTI_DISCRETE),
    (gym.spaces.MultiBinary, SpaceKind.MULTI_BINARY),
    (gym.spaces.Box, SpaceKind.BOX),
    (gym.spaces.Tuple, SpaceKind.TUPLE),
    (gym.spaces.Dict, SpaceKind.DICT),
    (gym.spaces.Text, SpaceKind.TEXT),
    (gym.spaces.Sequence, SpaceKind.SEQUENCE),
    (gym.spaces.Graph, SpaceKind.GRAPH),
    (gym.spaces.OneOf, SpaceKind.ONE_OF),
)


def classify_space(space: gym.Space) -> SpaceKind:
    """
    Map a Gymnasium space onto its `SpaceKind`.

    Args:
        space: Any Gymnasium space instance.

    Returns:
        The matching `SpaceKind`, or `SpaceKind.UNKNOWN` when the space is not one of
        the classes Gymnasium ships. UNKNOWN is returned rather than raised so a model
        declaring support for it stays possible, and so an unfamiliar space produces a
        clear "unsupported" message instead of a crash inside the classifier.
    """
    for space_class, kind in _SPACE_CLASSES:
        if isinstance(space, space_class):
            return kind
    return SpaceKind.UNKNOWN


def check_space_is_discrete(space: gym.Space) -> bool:
    """Return True when the space is a Gymnasium `Discrete`."""
    return isinstance(space, gym.spaces.Discrete)


def check_space_is_box(space: gym.Space) -> bool:
    """
    Return True when the space is a Gymnasium `Box`.

    Note this is a real type test. It previously returned `not check_space_is_discrete(space)`,
    which accepted every non-`Discrete` space -- `MultiDiscrete` and `Dict` included.
    """
    return isinstance(space, gym.spaces.Box)
