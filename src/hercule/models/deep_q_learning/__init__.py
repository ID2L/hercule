"""Deep Q-Learning (DQN) implementation based on the 2013 paper 'Playing Atari with Deep Reinforcement Learning'."""

import logging
import random
from typing import TYPE_CHECKING, ClassVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, PrivateAttr

from hercule.config import HyperParameter, HyperParamsBase, ParameterValue
from hercule.environnements.spaces_checker import SpaceKind, check_space_is_discrete
from hercule.models.off_policy import Encoder, OffPolicyReplayModel


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from gymnasium.spaces import Discrete
else:
    Discrete = object  # Placeholder for runtime


logger = logging.getLogger(__name__)


class DeepQLearningModelHyperParams(HyperParamsBase):
    """Type-safe hyperparameters for Deep Q-Learning model."""

    learning_rate: float = Field(default=0.00025, description="Learning rate (alpha)")
    discount_factor: float = Field(default=0.99, description="Discount factor (gamma)")
    epsilon: float = Field(default=1.0, description="Initial epsilon for epsilon-greedy")
    epsilon_decay: float = Field(default=0.0, description="Epsilon decay rate per epoch")
    epsilon_min: float = Field(default=0.1, description="Minimum epsilon value")
    replay_buffer_size: int = Field(default=10000, description="Size of experience replay buffer")
    batch_size: int = Field(default=32, description="Batch size for experience replay")
    step_modulo: int = Field(
        default=1, description="Number of steps before performing experience replay (default: 1, every step)"
    )
    target_update_frequency: int = Field(
        default=1000, description="Number of steps between two target-network synchronisations"
    )
    frame_stack: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of PREVIOUS observations concatenated to the current one "
            "(0 = current frame only; 3 = current plus the 3 preceding, i.e. 4 frames). "
            "Note this counts previous frames, unlike gymnasium's FrameStackObservation "
            "and Stable-Baselines3, whose parameter is the total count."
        ),
    )
    weight_decay: float = Field(default=0.0, description="Weight decay (L2 regularization) for optimizer")
    seed: int = Field(default=42, description="Random seed")


class QNetwork(nn.Module):
    """
    Deep Q-Network: a shared `Encoder` followed by one linear head per action.

    The split is at the LAST layer and nowhere else, because the order in which
    parameterised modules are constructed is observable behaviour: the encoder's
    layers are created first and the head second, which is exactly the sequence
    this project's stored checkpoints were initialised under. Moving the split, or
    building either part lazily, changes the weights a given seed produces.
    """

    def __init__(self, observation_shape: tuple, num_actions: int) -> None:
        """
        Args:
            observation_shape: Stacked observation shape, e.g. `(4,)` or `(96, 96, 12)`.
            num_actions: Size of the discrete action space.
        """
        super().__init__()
        self.observation_shape = tuple(observation_shape)
        self.num_actions = num_actions
        self.encoder = Encoder(self.observation_shape)
        self.head = nn.Linear(self.encoder.output_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, *observation_shape)

        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        return self.head(self.encoder(x))


class DeepQLearningModel(OffPolicyReplayModel[DeepQLearningModelHyperParams]):
    """
    Deep Q-Learning (DQN) implementation.

    Based on the 2013 paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
    Uses a deep neural network to approximate Q-values and experience replay for stable learning.
    """

    # Class attribute for model name (static, immutable)
    model_name: ClassVar[str] = "deep_q_learning"

    # Type-safe hyperparameters class
    hyperparams_class: ClassVar[type[HyperParamsBase]] = DeepQLearningModelHyperParams

    # DQN needs a discrete action space (it outputs one Q-value per action) but
    # tolerates either a Box or a Discrete observation space (a vector/image, or a
    # tabular index like FrozenLake's).
    supported_spaces: ClassVar[frozenset[tuple[SpaceKind, SpaceKind]]] = frozenset(
        {(SpaceKind.BOX, SpaceKind.DISCRETE), (SpaceKind.DISCRETE, SpaceKind.DISCRETE)}
    )

    # Old attribute path -> new attribute path, applied when reading a checkpoint
    # written before `QNetwork` was split into an encoder and a head. The two
    # branches are NOT jointly applicable and the branch is selected first: on an
    # image checkpoint `network.0.0.weight` matches the vector prefix `network.0.`
    # as well as the image alias rule, and on a vector checkpoint `network.0.weight`
    # matches both too, so no flat application order is correct for both.
    _VECTOR_KEY_MIGRATION: ClassVar[tuple[tuple[str, str], ...]] = (
        ("network.0.", "encoder.layers.0."),
        ("network.2.", "encoder.layers.2."),
        ("network.4.", "head."),
    )
    _IMAGE_KEY_MIGRATION: ClassVar[tuple[tuple[str, str], ...]] = (
        ("conv_layers.0.", "encoder.layers.0."),
        ("conv_layers.2.", "encoder.layers.2."),
        ("conv_layers.4.", "encoder.layers.4."),
        ("fc_layers.1.", "encoder.layers.7."),
        ("fc_layers.3.", "head."),
    )

    _q_network: QNetwork | None = PrivateAttr(default=None)
    _target_network: QNetwork | None = PrivateAttr(default=None)
    _optimizer: optim.Optimizer | None = PrivateAttr(default=None)

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure the Deep Q-Learning model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (will be merged with defaults)

        Returns:
            True if configuration successful, False otherwise
        """
        if not check_space_is_discrete(env.action_space):
            logger.error(f"Deep Q-Learning requires discrete action space, got {type(env.action_space)}")
            return False

        if not super().configure(env, hyperparameters):
            return False

        typed_params = self.get_hyperparameters()
        logger.info(
            f"'{self.model_name}' configured for environment with "
            f"discrete action space ({cast('Discrete', env.action_space).n} actions) and "
            f"observation shape {self._q_network.observation_shape} "
            f"(frame_stack={typed_params.frame_stack})"
        )
        return True

    # --------------------------------------------------------------------- hooks

    def _build_networks(self) -> None:
        """Build the online network, then the target network, then align them.

        The target network is CONSTRUCTED rather than copied. That is not an
        oversight: its constructor consumes a second full sequence of torch RNG
        draws, and although the weights it produces are immediately overwritten,
        the RNG state afterwards is not -- and that state is checkpointed. A copy
        here would leave every fresh run identical and every resumed run different.
        """
        num_actions = int(cast("Discrete", self._action_space).n)
        self._q_network = QNetwork(self._stacked_observation_shape, num_actions)
        self._target_network = QNetwork(self._stacked_observation_shape, num_actions)
        self._target_network.load_state_dict(self._q_network.state_dict())
        self._target_network.eval()  # Target network is always in eval mode

    def _build_optimizers(self) -> None:
        """Create the optimizer over the current network parameters."""
        typed_params = self.get_hyperparameters()
        self._optimizer = optim.Adam(
            self._q_network.parameters(), lr=typed_params.learning_rate, weight_decay=typed_params.weight_decay
        )

    def _select_action(self, observation: np.ndarray | int, training: bool) -> tuple[int, int]:
        """
        Epsilon-greedy selection.

        DQN's own action coordinates ARE the environment's -- an index into a
        discrete space -- so both halves of the returned pair are the same value.

        Raises:
            ValueError: If the model is not configured.
        """
        if self._q_network is None or self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        # The observation is already stacked when it comes from run_epoch/predict;
        # rescaling onto the observation space's range happens here, at the single
        # point where a tensor is built.
        obs_tensor_source = self._to_frame(observation)

        typed_params = self.get_hyperparameters()
        epsilon = typed_params.epsilon if training else 0.0  # No exploration during evaluation

        if training and random.random() < epsilon:
            action = random.randint(0, int(cast("Discrete", self._action_space).n) - 1)
            return action, action

        self._q_network.eval()
        with torch.no_grad():
            obs_tensor = self._as_network_input(obs_tensor_source).unsqueeze(0)
            action = int(self._q_network(obs_tensor).argmax().item())
        self._q_network.train()
        return action, action

    def _on_training_step(self) -> None:
        """Decay epsilon, once per environment step on the training path.

        Written back to BOTH representations: the typed hyperparameters the
        algorithm reads, and the generic list that gets serialised and signed.
        """
        typed_params = self.get_hyperparameters()
        typed_params.epsilon = max(typed_params.epsilon_min, typed_params.epsilon * (1 - typed_params.epsilon_decay))
        self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]

    def _target_pairs(self) -> "Iterable[tuple[nn.Module, nn.Module]]":
        """The one pair DQN keeps: the online network and its delayed copy."""
        if self._q_network is None or self._target_network is None:
            return ()
        return ((self._q_network, self._target_network),)

    def _target_sync_interval(self) -> int | None:
        """Environment steps between hard copies.

        Deliberately a number of ENVIRONMENT steps, never gradient steps: copying
        the weights after every update would make the bootstrap target come from
        the current weights, which is exactly having no target network at all.
        """
        return int(self.get_hyperparameters().target_update_frequency)

    def _networks(self) -> "Mapping[str, nn.Module]":
        """Both networks, under the names every stored checkpoint already uses."""
        if self._q_network is None or self._target_network is None:
            return {}
        return {"online": self._q_network, "target": self._target_network}

    def _optimizers(self) -> "Mapping[str, optim.Optimizer]":
        """The single optimizer."""
        return {} if self._optimizer is None else {"main": self._optimizer}

    def _update(self, batch: list) -> None:
        """One gradient step on a sampled batch."""
        if self._q_network is None or self._target_network is None or self._optimizer is None:
            return

        typed_params = self.get_hyperparameters()

        # States are stored in their native dtype (uint8 for image observations) and
        # rescaled here, so the buffer stays compact and the network still sees
        # values in [0, 1].
        states = self._as_network_input(np.array([s for s, _, _, _, _, _ in batch]))
        actions = torch.LongTensor([a for _, a, _, _, _, _ in batch]).to(self._device)
        rewards = torch.FloatTensor([r for _, _, r, _, _, _ in batch]).to(self._device)
        next_states = self._as_network_input(np.array([ns for _, _, _, ns, _, _ in batch]))
        # The mask is built from `terminated` ALONE, never `truncated`: a time-limit
        # truncation is not an MDP terminal state, so its successor state still has
        # value and must keep contributing the bootstrap term `gamma * V(s')`.
        terminateds = torch.BoolTensor([t for _, _, _, _, t, _ in batch]).to(self._device)

        current_q_values = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._target_network(next_states).max(1)[0]
            target_q_values = rewards + (typed_params.discount_factor * next_q_values * ~terminateds)

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    # ---------------------------------------------------------------- checkpoint

    def _extra_state(self) -> dict:
        """The decayed epsilon, plus the two values the network's shape depends on.

        `frame_stack` and `observation_shape` travel with the weights because
        `hercule play` configures with DEFAULT hyperparameters: without them it
        would build a network of the wrong shape and fail to load a stacked model.
        """
        typed_params = self.get_hyperparameters()
        return {
            "epsilon": typed_params.epsilon,
            "frame_stack": typed_params.frame_stack,
            "observation_shape": list(self._q_network.observation_shape) if self._q_network else [],
        }

    def _load_extra_state(self, model_data: dict) -> None:
        """Restore epsilon, and rebuild the networks if the stack depth differs.

        The rebuild must happen BEFORE any weights are loaded, which is why the
        ancestor calls this first.
        """
        saved_stack = model_data.get("frame_stack")
        if saved_stack is not None and self._observation_space is not None:
            typed_params = self.get_hyperparameters()
            if int(saved_stack) != typed_params.frame_stack:
                logger.info(
                    f"Saved model was trained with frame_stack={saved_stack}, "
                    f"rebuilding networks (was {typed_params.frame_stack})"
                )
                typed_params.frame_stack = int(saved_stack)
                self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]
                self._rebuild_from_spaces()

        if "epsilon" in model_data:
            typed_params = self.get_hyperparameters()
            typed_params.epsilon = model_data["epsilon"]
            self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]

    def _migrate_parameter_keys(self, state_dict: dict, network: str) -> dict:
        """
        Rename pre-split attribute paths onto the encoder/head layout.

        The branch is selected first, from the payload's own keys, and only that
        branch's rules are applied. `conv_layers.` exists on the convolutional path
        and nowhere else, so the test is exact and reads a key the migration never
        writes.

        On the image branch the old `state_dict` carries every parameter TWICE --
        once under `conv_layers.*`/`fc_layers.*` and again under `network.*`,
        because `self.network = nn.Sequential(self.conv_layers, self.fc_layers)`
        registered them a second time. The aliases are dropped rather than mapped:
        the current modules register each parameter once.
        """
        is_image = any(key.startswith("conv_layers.") for key in state_dict)
        if not is_image and not any(key.startswith("network.") for key in state_dict):
            return state_dict  # already in the current layout

        rules = self._IMAGE_KEY_MIGRATION if is_image else self._VECTOR_KEY_MIGRATION
        migrated: dict = {}
        for key, value in state_dict.items():
            if is_image and key.startswith("network."):
                continue  # duplicate registration of a parameter kept under its canonical name
            for old_prefix, new_prefix in rules:
                if key.startswith(old_prefix):
                    migrated[new_prefix + key[len(old_prefix) :]] = value
                    break
            else:
                migrated[key] = value
        return migrated

    def _load_legacy_networks(self, state_dict: dict) -> None:
        """Place a migrated pre-006 state dict into both networks.

        The legacy format never saved the target network separately, so loading the
        online weights into both reproduces its previous (lag-destroying) behaviour
        exactly. There is no better information available from such a checkpoint.
        """
        if self._q_network is None:
            return
        self._q_network.load_state_dict(state_dict)
        if self._target_network is not None:
            self._target_network.load_state_dict(state_dict)

    def load_from_dict(self, model_data: dict) -> None:
        """
        Load a trained Deep Q-Learning model from a dictionary.

        Args:
            model_data: Dictionary containing model data
        """
        self._import(model_data)
        logger.info(f"Loaded {self.model_name} model from dictionary")


__all__ = ["DeepQLearningModel"]
