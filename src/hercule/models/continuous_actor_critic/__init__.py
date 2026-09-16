"""Shared scaffolding for continuous actor-critic models.

`ContinuousActorCriticModel` holds what every family-B algorithm needs: an actor,
two value estimators, delayed copies **of those two estimators only**, and their
gradual averaging. A concrete algorithm supplies the heads, the objectives and the
action selection.

There is deliberately **no delayed copy of the actor**. That construct exists to
be smoothed in the deterministic-actor algorithms, and here it would be a network
trained, averaged, checkpointed and never read. A deterministic-actor sibling added
later declares its own.
"""

import logging
from abc import ABC
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import PrivateAttr

from hercule.models import HyperParamsType
from hercule.models.off_policy import Encoder, OffPolicyReplayModel


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


logger = logging.getLogger(__name__)


class ContinuousCritic(nn.Module):
    """One value estimator: `Q(observation, action) -> scalar`.

    The action enters **after** the encoder rather than at the input, which is the
    only arrangement that works unchanged on both observation branches: for an
    image, concatenating a 3-vector to a 96x96x12 input before the convolutions is
    meaningless, while after the encoder both branches present a flat feature
    vector and the action concatenates the same way.

    The action arrives in the policy's **normalised** coordinates, matching what
    the replay buffer holds.
    """

    HIDDEN_SIZE = 256

    def __init__(self, observation_shape: tuple[int, ...], action_dimensions: int) -> None:
        """
        Args:
            observation_shape: Stacked observation shape.
            action_dimensions: Size of the continuous action vector.
        """
        super().__init__()
        self.encoder = Encoder(observation_shape)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_size + action_dimensions, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_SIZE, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return one value per batch element, shaped `(batch,)`."""
        features = self.encoder(observation)
        return self.head(torch.cat([features, action], dim=-1)).squeeze(-1)


class ContinuousActorCriticModel(OffPolicyReplayModel[HyperParamsType], ABC, Generic[HyperParamsType]):
    """Abstract base for continuous actor-critic algorithms."""

    _actor: nn.Module | None = PrivateAttr(default=None)
    _critic_1: ContinuousCritic | None = PrivateAttr(default=None)
    _critic_2: ContinuousCritic | None = PrivateAttr(default=None)
    _critic_1_target: ContinuousCritic | None = PrivateAttr(default=None)
    _critic_2_target: ContinuousCritic | None = PrivateAttr(default=None)

    _actor_optimizer: optim.Optimizer | None = PrivateAttr(default=None)
    _critic_1_optimizer: optim.Optimizer | None = PrivateAttr(default=None)
    _critic_2_optimizer: optim.Optimizer | None = PrivateAttr(default=None)

    @property
    def _action_dimensions(self) -> int:
        """Size of the continuous action vector."""
        return int(np.prod(self._action_space.shape))

    def _build_critics(self) -> None:
        """Build both estimators and their delayed copies.

        Each of the three networks -- the actor is built by the subclass -- gets its
        **own** feature extractor. The two estimators in particular must never share
        one: shared features collapse the decorrelation that taking the lesser of the
        two exists to provide, and because sharing makes the checkpoint *smaller*, no
        size or performance criterion would ever catch it.
        """
        shape = self._stacked_observation_shape
        dimensions = self._action_dimensions

        self._critic_1 = ContinuousCritic(shape, dimensions)
        self._critic_2 = ContinuousCritic(shape, dimensions)
        self._critic_1_target = ContinuousCritic(shape, dimensions)
        self._critic_2_target = ContinuousCritic(shape, dimensions)

        self._critic_1_target.load_state_dict(self._critic_1.state_dict())
        self._critic_2_target.load_state_dict(self._critic_2.state_dict())

        # Half of the mechanism by which the learning target cannot leak gradient;
        # the other half is computing that target under `torch.no_grad()`.
        for delayed in (self._critic_1_target, self._critic_2_target):
            delayed.eval()
            for parameter in delayed.parameters():
                parameter.requires_grad = False

    def _polyak_update(self, tau: float) -> None:
        """Move each delayed copy a fraction `tau` toward its live counterpart.

        Called once per **gradient** step, from inside `_update()`, never from the
        ancestor's per-environment-step synchronisation hook: the two clocks differ
        whenever a gradient step is not taken on every environment step.

        The averaging is gradual and never a replacement. A hard copy on the right
        clock would make each delayed copy identical to its live counterpart from
        the first step onward, at which point the learning target's "lesser of the
        two delayed copies" is numerically the lesser of the two live ones -- the
        first wrong form the target requirement enumerates, reached without
        violating a word of it.
        """
        with torch.no_grad():
            for live, delayed in self._target_pairs():
                for live_parameter, delayed_parameter in zip(live.parameters(), delayed.parameters(), strict=True):
                    delayed_parameter.mul_(1.0 - tau).add_(live_parameter, alpha=tau)

    def _target_pairs(self) -> "Iterator[tuple[nn.Module, nn.Module]]":
        """`(live, delayed)` estimator pairs. Note: no actor pair, by design."""
        if self._critic_1 is None or self._critic_1_target is None:
            return iter(())
        return iter(
            (
                (self._critic_1, self._critic_1_target),
                (self._critic_2, self._critic_2_target),
            )
        )

    @contextmanager
    def _frozen_critics(self):
        """Detach both estimators from the graph for the duration of a block.

        This is how the actor's objective is *confined* to the actor's parameters.
        Within it the estimators are evaluated, not trained, so no gradient flows
        out of the actor's objective onto them. Zeroing their gradients afterwards
        would work too, and would be one ordering mistake away from silently
        stepping them toward the value the actor is chasing -- which reintroduces
        the overestimation that taking the lesser of two exists to control, and does
        so identically on both, where neither that construction nor their disjoint
        extractors can see it.
        """
        critics = [c for c in (self._critic_1, self._critic_2) if c is not None]
        parameters = [p for critic in critics for p in critic.parameters()]
        for parameter in parameters:
            parameter.requires_grad = False
        try:
            yield
        finally:
            for parameter in parameters:
                parameter.requires_grad = True

    def _networks(self) -> "Mapping[str, nn.Module]":
        """Every module whose weights must survive a resume, delayed copies included."""
        if self._actor is None:
            return {}
        return {
            "actor": self._actor,
            "critic_1": self._critic_1,
            "critic_2": self._critic_2,
            "critic_1_target": self._critic_1_target,
            "critic_2_target": self._critic_2_target,
        }


__all__ = ["ContinuousActorCriticModel", "ContinuousCritic"]
