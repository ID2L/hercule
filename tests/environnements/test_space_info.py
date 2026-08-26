"""Tests for space introspection, in particular Box bounds of any rank.

`BoxSpaceInfo.low/high` used to be typed `list[float]`, which made
`EnvironmentInspector.get_environment_info()` raise a ValidationError on every
pixel environment: a Box's bounds carry the space's own shape, so an image space
yields a nested (H, W, C) list. That crashed report generation on CarRacing.
"""

import gymnasium as gym
import numpy as np
import pytest

from hercule.environnements import BoxSpaceInfo, EnvironmentInspector


@pytest.mark.unit
def test_image_observation_space_is_introspectable() -> None:
    """The regression: a 3-D Box must not raise."""
    env = gym.make("CarRacing-v3", continuous=False)
    try:
        info = EnvironmentInspector.get_environment_info(env)
    finally:
        env.close()

    obs = info.observation_space
    assert isinstance(obs, BoxSpaceInfo)
    assert tuple(obs.shape) == (96, 96, 3)
    assert np.asarray(obs.low).shape == (96, 96, 3)


@pytest.mark.unit
def test_uniform_bounds_are_summarised_not_dumped() -> None:
    """27 648 numbers must never reach a report line."""
    space = BoxSpaceInfo(
        type="Box",
        shape=(96, 96, 3),
        low=np.zeros((96, 96, 3)).tolist(),
        high=np.full((96, 96, 3), 255.0).tolist(),
    )

    described = space.describe_bounds()

    assert described == "[0, 255] uniform over shape (96, 96, 3)"
    assert len(described) < 88, "must fit a monospace PDF line, which clips rather than wraps"


@pytest.mark.unit
def test_short_vector_bounds_are_printed_in_full() -> None:
    """A 3-element action range is genuinely informative, so it is kept."""
    space = BoxSpaceInfo(type="Box", shape=(3,), low=[-1.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])

    described = space.describe_bounds()

    assert "-1" in described and "high=" in described


@pytest.mark.unit
def test_non_uniform_high_rank_bounds_report_a_range() -> None:
    """Per-element bounds too large to list fall back to an overall range."""
    low = np.zeros((20, 20))
    low[0, 0] = -5.0
    space = BoxSpaceInfo(type="Box", shape=(20, 20), low=low.tolist(), high=np.ones((20, 20)).tolist())

    described = space.describe_bounds()

    assert "overall" in described and "not listed" in described
    assert "-5" in described


@pytest.mark.unit
def test_missing_bounds_report_unbounded() -> None:
    """A Box built without bounds must not crash the description."""
    assert BoxSpaceInfo(type="Box", shape=(2,)).describe_bounds() == "unbounded"
