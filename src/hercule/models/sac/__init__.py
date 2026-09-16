"""Soft Actor-Critic with automatic temperature adjustment (arXiv:1812.05905).

Deliberately the 1812.05905 version and not 1801.01290: the earlier one's fixed
temperature has to compensate each environment's reward scale, which makes it
unusable in a multi-environment benchmark -- the temperature would become one more
hyperparameter to sweep per environment, which is exactly what automatic
adjustment removes.

Every clause of the algorithm below is pinned by a requirement and covered by a
direct test, and the reason is uncomfortable rather than academic: a wrong
reinforcement-learning implementation still trains, still improves, and still
produces a plausible reward curve. Bootstrapping from the live estimators instead
of their delayed copies, taking the greater of the two instead of the lesser,
adding the entropy term instead of subtracting it, evaluating the target at the
stored action instead of a resampled one -- all four converge to something, and
none of them announces itself. So the objectives are written out in full and each
wrong form fails its own test.
"""

import logging
from typing import TYPE_CHECKING, ClassVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, PrivateAttr

from hercule.config import HyperParameter, HyperParamsBase, ParameterValue
from hercule.environnements.spaces_checker import SpaceKind, check_space_is_box
from hercule.models.continuous_actor_critic import ContinuousActorCriticModel
from hercule.models.off_policy import Encoder


if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Bounds on the policy's log-standard-deviation. Module constants, not
# hyperparameters: without a lower bound the spread collapses toward zero and the
# log-density diverges; without an upper bound early training produces spreads
# large enough to saturate the squashing correction. This is a numerical guard,
# not a design choice a benchmark should sweep.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# Bounds on the learned temperature's LOGARITHM. Also a numerical guard, exactly
# like the log-std clamp above, not an algorithmic choice a benchmark should sweep.
# Optimising the logarithm keeps the temperature positive in exact arithmetic for
# any step size, but float32 `exp()` underflows to exactly `0.0` once its argument
# drops below roughly -104 -- at which point the entropy term vanishes from both the
# learning target and the actor objective with no error raised. `exp(LOG_ALPHA_MIN)`
# is about 2e-9: small enough to be no exploration at all, and still strictly
# positive and representable. `exp(LOG_ALPHA_MAX)` bounds the other end against
# overflow.
LOG_ALPHA_MIN = -20.0
LOG_ALPHA_MAX = 20.0


class SACHyperParams(HyperParamsBase):
    """Type-safe hyperparameters for Soft Actor-Critic.

    The target entropy is deliberately absent. It is fixed at the negative of the
    action-space dimension and derived at configure time; exposing it would let a
    grid sweep contradict the requirement that pins it.
    """

    learning_rate: float = Field(default=3e-4, description="Learning rate for the actor, both critics and alpha")
    discount_factor: float = Field(default=0.99, description="Discount factor (gamma)")
    tau: float = Field(default=0.005, gt=0.0, lt=1.0, description="Gradual averaging fraction for the target critics")
    batch_size: int = Field(default=256, description="Batch size for experience replay")
    replay_buffer_size: int = Field(default=100000, description="Size of experience replay buffer")
    step_modulo: int = Field(default=1, description="Environment steps between two gradient steps")
    learning_starts: int = Field(
        default=1000, description="Environment steps of uniform-random warmup before the policy takes over"
    )
    init_temperature: float = Field(default=1.0, gt=0.0, description="Initial entropy temperature (alpha)")
    frame_stack: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of PREVIOUS observations concatenated to the current one "
            "(0 = current frame only; 3 = current plus the 3 preceding, i.e. 4 frames)."
        ),
    )
    seed: int = Field(default=42, description="Random seed")

    # Deliberately no `weight_decay` field, unlike `DeepQLearningModel`. Adam's
    # weight decay adds an L2 term straight into the gradient, so any non-zero
    # value would make the realised update differ from the actor and critic
    # objectives this module pins exactly (see `_update_actor`, `_learning_target`).
    # SAC's own reference implementations do not use it either, and admitting it
    # here would add a dimension to every hyperparameter grid sweep for no benefit
    # this benchmark can measure.


class GaussianTanhActor(nn.Module):
    """A squashed Gaussian policy: `Encoder` then one head producing mean and spread."""

    def __init__(self, observation_shape: tuple[int, ...], action_dimensions: int) -> None:
        """
        Args:
            observation_shape: Stacked observation shape.
            action_dimensions: Size of the continuous action vector.
        """
        super().__init__()
        self.action_dimensions = action_dimensions
        self.encoder = Encoder(observation_shape)
        self.head = nn.Linear(self.encoder.output_size, 2 * action_dimensions)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(mean, log_std)`, the latter clamped to its numerical guard."""
        mean, log_std = self.head(self.encoder(observation)).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a differentiable action and its log-density.

        The sample keeps its gradient path back to the policy's parameters -- it is
        drawn as `mean + std * noise` rather than from a distribution object that
        would detach it -- because the actor's objective differentiates *through* it.

        The density correction for the squashing is computed in its numerically
        stable form, `2 * (log 2 - u - softplus(-2u))`. The textbook expression,
        `log(1 - tanh(u)^2)`, underflows to `log(0)` for `|u|` above about 9 in
        float32, which ordinary training reaches once the policy learns to push an
        action toward a bound. The common patch of adding an epsilon inside the
        logarithm instead biases the density in exactly that saturated region.

        Returns:
            `(action, log_density)` in the policy's NORMALISED coordinates, the
            action shaped `(batch, d)` and the density `(batch,)`.
        """
        mean, log_std = self(observation)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        unsquashed = mean + std * noise

        gaussian_log_density = (-0.5 * (noise**2) - log_std - 0.5 * float(np.log(2.0 * np.pi))).sum(dim=-1)
        squash_correction = (2.0 * (float(np.log(2.0)) - unsquashed - nn.functional.softplus(-2.0 * unsquashed))).sum(
            dim=-1
        )

        return torch.tanh(unsquashed), gaussian_log_density - squash_correction

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """The policy's mode, for evaluation. No draw, so no random stream moves."""
        mean, _ = self(observation)
        return torch.tanh(mean)


class SACModel(ContinuousActorCriticModel[SACHyperParams]):
    """Soft Actor-Critic with a learned temperature."""

    model_name: ClassVar[str] = "sac"
    hyperparams_class: ClassVar[type[HyperParamsBase]] = SACHyperParams

    # Continuous actions only. A `Discrete` observation is excluded deliberately:
    # a tabular observation with a continuous action is not a combination any target
    # environment presents, and admitting it would mean specifying an embedding this
    # model does not need.
    supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]] = frozenset({(SpaceKind.BOX, SpaceKind.BOX)})

    # The learned temperature, held as its LOGARITHM. Optimising the logarithm keeps
    # the temperature strictly positive for any step size WITHIN the clamp
    # (`LOG_ALPHA_MIN`/`LOG_ALPHA_MAX`): an unconstrained temperature that crosses
    # zero inverts the sign of the entropy term in both the learning target and the
    # actor's objective, with no error raised anywhere. The clamp is what makes that
    # guarantee true in float32 rather than merely true in exact arithmetic --
    # without it, `exp()` underflows to exactly `0.0` once the logarithm drops below
    # roughly -104.
    _log_alpha: torch.Tensor | None = PrivateAttr(default=None)
    _temperature_optimizer: optim.Optimizer | None = PrivateAttr(default=None)
    _target_entropy: float = PrivateAttr(default=0.0)

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure SAC for one continuous-action environment.

        Returns:
            True on success; False when the action space is not a `Box`.
        """
        if not check_space_is_box(env.action_space):
            logger.error(f"SAC requires a Box action space, got {type(env.action_space)}")
            return False

        if not super().configure(env, hyperparameters):
            return False

        logger.info(
            f"'{self.model_name}' configured for a {self._action_dimensions}-dimensional Box action space, "
            f"target entropy {self._target_entropy}, observation shape {self._stacked_observation_shape}"
        )
        return True

    # --------------------------------------------------------------------- hooks

    def _build_networks(self) -> None:
        """Actor first, then both estimators and their delayed copies.

        The temperature is created here too rather than in `_build_optimizers()`,
        because it is a learned tensor and this is where learned things are made.
        """
        self._actor = GaussianTanhActor(self._stacked_observation_shape, self._action_dimensions)
        self._build_critics()

        typed_params = self.get_hyperparameters()
        self._log_alpha = torch.tensor(
            float(np.log(typed_params.init_temperature)), device=self._device, requires_grad=True
        )
        # Fixed at the negative of the action dimension, which is correct precisely
        # because the policy, its density and the estimators all operate in the
        # normalised coordinates: rescaling to the environment's bounds would add a
        # per-environment constant to the measured entropy and shift what the
        # temperature adapts against.
        self._target_entropy = -float(self._action_dimensions)

    def _build_optimizers(self) -> None:
        """One optimizer per trained network, plus the temperature's.

        Four in total. The estimators get one each rather than sharing one: it is
        numerically equivalent for Adam, whose state is per parameter, and it keeps
        the checkpoint's optimizer mapping one entry per trained thing.
        """
        typed_params = self.get_hyperparameters()
        rate = typed_params.learning_rate
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=rate)
        self._critic_1_optimizer = optim.Adam(self._critic_1.parameters(), lr=rate)
        self._critic_2_optimizer = optim.Adam(self._critic_2.parameters(), lr=rate)
        self._temperature_optimizer = optim.Adam([self._log_alpha], lr=rate)

    def _optimizers(self) -> "Mapping[str, optim.Optimizer]":
        """Every optimizer whose state must survive a resume."""
        if self._actor_optimizer is None:
            return {}
        return {
            "actor": self._actor_optimizer,
            "critic_1": self._critic_1_optimizer,
            "critic_2": self._critic_2_optimizer,
            "temperature": self._temperature_optimizer,
        }

    @property
    def _alpha(self) -> torch.Tensor:
        """The temperature itself."""
        return self._log_alpha.exp()

    def _select_action(self, observation: np.ndarray, training: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Choose an action, and say how it should be stored.

        Owns all exploration, warmup included, so the ancestor never samples an
        action. The warmup draws uniformly in the **normalised** cube and maps to
        the environment on the way out, so what reaches the replay buffer is already
        in the policy's coordinates -- sampling from `env.action_space` instead would
        need the map inverted before storing, which is a second path through the
        mapping and a channel for the two to disagree.

        Returns:
            `(env_action, normalised_action)`.
        """
        if self._actor is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        typed_params = self.get_hyperparameters()
        if training and self._step_count < typed_params.learning_starts:
            normalised = self._rng.uniform(-1.0, 1.0, size=self._action_dimensions).astype(np.float32)
            return self.to_env_action(normalised), normalised

        observation_tensor = self._as_network_input(self._to_frame(observation)).unsqueeze(0)
        self._actor.eval()
        with torch.no_grad():
            if training:
                action, _ = self._actor.sample(observation_tensor)
            else:
                # Evaluation is deterministic, and consumes no random stream.
                action = self._actor.deterministic(observation_tensor)
        self._actor.train()

        normalised = action.squeeze(0).cpu().numpy().astype(np.float32)
        return self.to_env_action(normalised), normalised

    def _ready_to_update(self) -> bool:
        """No gradient step before the warmup is over."""
        return super()._ready_to_update() and self._step_count >= self.get_hyperparameters().learning_starts

    def _update(self, batch: list) -> None:
        """One gradient step: both estimators, then the actor, then the temperature.

        Then the delayed copies advance, once per **gradient** step -- which is here
        and not in the ancestor's per-environment-step hook, because the two clocks
        differ whenever `step_modulo > 1`.
        """
        typed_params = self.get_hyperparameters()

        observations = self._as_network_input(np.array([s for s, _, _, _, _, _ in batch]))
        actions = torch.as_tensor(np.array([a for _, a, _, _, _, _ in batch], dtype=np.float32), device=self._device)
        rewards = torch.as_tensor(np.array([r for _, _, r, _, _, _ in batch], dtype=np.float32), device=self._device)
        next_observations = self._as_network_input(np.array([ns for _, _, _, ns, _, _ in batch]))
        # `terminated` alone, never `truncated`: a time-limit cut-off is not an MDP
        # terminal state, so its successor still has value and must keep contributing
        # the bootstrap term. The primary validation environment truncates on EVERY
        # episode and never terminates, so collapsing the two would train against a
        # wrong target 100% of the time.
        terminated = torch.as_tensor(np.array([t for _, _, _, _, t, _ in batch], dtype=np.float32), device=self._device)

        target = self._learning_target(rewards, next_observations, terminated, typed_params.discount_factor)
        self._update_critics(observations, actions, target)
        log_density = self._update_actor(observations)
        self._update_temperature(log_density)
        self._polyak_update(typed_params.tau)

    def _learning_target(
        self, rewards: torch.Tensor, next_observations: torch.Tensor, terminated: torch.Tensor, discount: float
    ) -> torch.Tensor:
        """
        The learning target, in full.

        The reward, plus -- suppressed if and only if the successor is terminal --
        the discount times the lesser of the two **delayed** copies, evaluated at an
        action **resampled from the current policy** at the successor observation,
        minus the temperature times that action's log-density.

        Computed under `no_grad`, so it is a **constant for learning**: no gradient
        flows out of it into the actor, the temperature or the delayed copies.
        Detaching changes no number, which is exactly why it needs saying -- every
        value-level check passes either way, and the damage appears only as spurious
        gradients at the optimizer step.
        """
        with torch.no_grad():
            next_actions, next_log_density = self._actor.sample(next_observations)
            next_value = torch.min(
                self._critic_1_target(next_observations, next_actions),
                self._critic_2_target(next_observations, next_actions),
            )
            soft_value = next_value - self._alpha * next_log_density
            return rewards + discount * (1.0 - terminated) * soft_value

    def _update_critics(self, observations: torch.Tensor, actions: torch.Tensor, target: torch.Tensor) -> None:
        """Regress **both** estimators toward the target, at the **stored** action.

        Both, and at the stored action: training only one leaves the other drifting
        while the lesser-of-two still reads it, and regressing at an action resampled
        from the current policy is the target side's construct and belongs only there.
        """
        for critic, optimizer in (
            (self._critic_1, self._critic_1_optimizer),
            (self._critic_2, self._critic_2_optimizer),
        ):
            loss = nn.functional.mse_loss(critic(observations, actions), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def _update_actor(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Maximise the entropy-regularised value, confined to the actor's parameters.

        The lesser of the two **live** estimators -- live here, not the delayed
        copies, which are a target-side construct -- at a differentiable sample,
        minus the temperature times its log-density. The estimators are frozen for
        the duration, so the objective deposits no gradient on them; the temperature
        is detached for the same reason.

        Returns:
            The log-density, detached, for the temperature's own objective.
        """
        with self._frozen_critics():
            actions, log_density = self._actor.sample(observations)
            value = torch.min(self._critic_1(observations, actions), self._critic_2(observations, actions))
            loss = (self._alpha.detach() * log_density - value).mean()

            self._actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._actor_optimizer.step()

        return log_density.detach()

    def _update_temperature(self, log_density: torch.Tensor) -> None:
        """
        Adapt the temperature by gradient descent, not by a hand-written controller.

        The objective's gradient with respect to the log-temperature is the negative
        of `log_density + target_entropy`. The log-density is held constant within
        it: it depends on the policy's parameters, and letting this gradient flow
        back into them would make the actor optimise against its own exploration
        schedule.

        The resulting behaviour, which is what an end-to-end check sees: the
        temperature **rises when the policy's measured entropy falls below** the
        target and **falls when it rises above**. The opposite sign also produces a
        temperature that moves, a policy that trains and a curve that rises, while
        exploration collapses or diverges -- which is why the direction is stated as
        well as the objective.

        After the optimizer step, `_log_alpha` is clamped in place to
        `[LOG_ALPHA_MIN, LOG_ALPHA_MAX]`. This leaves the objective and its gradient
        exactly as stated above -- it only bounds the state the optimizer carries
        forward -- and is what keeps the temperature strictly positive **in
        float32** for any step size, rather than merely in exact arithmetic: without
        it, enough gradient steps in one direction drive the logarithm past about
        -104, where `exp()` underflows to exactly `0.0` and the entropy term
        vanishes from both the learning target and the actor objective with no error
        raised.
        """
        loss = -(self._log_alpha * (log_density + self._target_entropy)).mean()
        self._temperature_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._temperature_optimizer.step()
        with torch.no_grad():
            self._log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)

    # ---------------------------------------------------------------- checkpoint

    def _extra_state(self) -> dict:
        """The learned temperature, plus the values the network shapes depend on."""
        typed_params = self.get_hyperparameters()
        return {
            "log_alpha": float(self._log_alpha.item()) if self._log_alpha is not None else 0.0,
            "frame_stack": typed_params.frame_stack,
            "observation_shape": list(self._stacked_observation_shape),
        }

    def _load_extra_state(self, model_data: dict) -> None:
        """Restore the temperature, rebuilding the networks first if the stack differs."""
        saved_stack = model_data.get("frame_stack")
        if saved_stack is not None and self._observation_space is not None:
            typed_params = self.get_hyperparameters()
            if int(saved_stack) != typed_params.frame_stack:
                logger.info(f"Saved model was trained with frame_stack={saved_stack}, rebuilding networks")
                typed_params.frame_stack = int(saved_stack)
                self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]
                self._rebuild_from_spaces()

        if "log_alpha" in model_data and self._log_alpha is not None:
            with torch.no_grad():
                self._log_alpha.fill_(float(model_data["log_alpha"]))

    def load_from_dict(self, model_data: dict) -> None:
        """Load a trained SAC model from a dictionary, for `hercule play`."""
        self._import(model_data)
        logger.info(f"Loaded {self.model_name} model from dictionary")

    def act(self, observation: np.ndarray, training: bool = False) -> np.ndarray:
        """Choose an action in the environment's coordinates."""
        return cast("np.ndarray", super().act(observation, training=training))


__all__ = ["GaussianTanhActor", "SACHyperParams", "SACModel"]
