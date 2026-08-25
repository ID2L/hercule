"""Deep Q-Learning (DQN) implementation based on the 2013 paper 'Playing Atari with Deep Reinforcement Learning'."""

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, ClassVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, PrivateAttr

from hercule.config import HyperParameter, HyperParamsBase, ParameterValue
from hercule.environnements.spaces_checker import check_space_is_discrete
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


if TYPE_CHECKING:
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
    Deep Q-Network architecture.

    Standard CNN architecture for processing observations and outputting Q-values for each action.
    """

    def __init__(self, observation_shape: tuple, num_actions: int) -> None:
        """
        Initialize the Q-Network.

        Args:
            observation_shape: Shape of the observation space (e.g., (4,) for CartPole, (84, 84, 4) for Atari)
            num_actions: Number of possible actions
        """
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Determine if input is image-like (3D) or vector-like (1D)
        if len(observation_shape) == 1:
            # Vector input (e.g., CartPole)
            self._build_mlp(observation_shape[0], num_actions)
        elif len(observation_shape) == 3:
            # Image input (e.g., Atari)
            self._build_cnn(observation_shape, num_actions)
        else:
            # Flatten and use MLP for other cases
            input_size = int(np.prod(observation_shape))
            self._build_mlp(input_size, num_actions)

    def _build_mlp(self, input_size: int, num_actions: int) -> None:
        """Build a Multi-Layer Perceptron for vector inputs."""
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def _build_cnn(self, observation_shape: tuple, num_actions: int) -> None:
        """Build a Convolutional Neural Network for image inputs."""
        # Standard CNN architecture from the DQN paper
        # Calculate the size of the flattened feature map
        # This is a placeholder - in practice, you'd calculate this based on input size
        # For now, we'll use a dynamic calculation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calculate the size of the flattened feature map
        # Create a dummy input to determine the size
        with torch.no_grad():
            dummy_input = torch.zeros(1, observation_shape[2], observation_shape[0], observation_shape[1])
            conv_output = self.conv_layers(dummy_input)
            flattened_size = int(np.prod(conv_output.shape[1:]))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self.network = nn.Sequential(self.conv_layers, self.fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, *observation_shape)

        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        # Ensure input has correct shape
        if len(self.observation_shape) == 3 and x.dim() == 4:
            # Image input: if shape is (batch, H, W, C), convert to (batch, C, H, W)
            if x.shape[1] == self.observation_shape[0] and x.shape[3] == self.observation_shape[2]:
                x = x.permute(0, 3, 1, 2)
        return self.network(x)


class ExperienceReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores tuples of (state, action, reward, next_state, done) for experience replay.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the experience replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether the episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DeepQLearningModel(RLModel[DeepQLearningModelHyperParams]):
    """
    Deep Q-Learning (DQN) implementation.

    Based on the 2013 paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
    Uses a deep neural network to approximate Q-values and experience replay for stable learning.
    """

    # Class attribute for model name (static, immutable)
    model_name: ClassVar[str] = "deep_q_learning"

    # Type-safe hyperparameters class
    hyperparams_class: ClassVar[type[HyperParamsBase]] = DeepQLearningModelHyperParams

    # Private attributes (not Pydantic fields, use PrivateAttr to avoid validation)
    _q_network: QNetwork | None = PrivateAttr(default=None)
    _target_network: QNetwork | None = PrivateAttr(default=None)
    _optimizer: optim.Optimizer | None = PrivateAttr(default=None)
    _replay_buffer: ExperienceReplayBuffer | None = PrivateAttr(default=None)
    _action_space: Discrete | None = PrivateAttr(default=None)
    _observation_space: gym.Space | None = PrivateAttr(default=None)
    _device: torch.device = PrivateAttr(default_factory=lambda: torch.device("cpu"))
    _step_count: int = PrivateAttr(default=0)
    _epoch_count: int = PrivateAttr(default=0)
    # Frame history for observation stacking. Two independent deques: run_epoch()
    # drives its own episode loop, predict() is called step by step from outside.
    _frames: deque = PrivateAttr(default_factory=lambda: deque(maxlen=1))
    _predict_frames: deque = PrivateAttr(default_factory=lambda: deque(maxlen=1))
    # Affine rescaling of observations, derived from the observation space's own
    # range rather than hard-coded: scalars when the bounds are uniform.
    _obs_offset: np.ndarray | float = PrivateAttr(default=0.0)
    _obs_scale: np.ndarray | float = PrivateAttr(default=1.0)
    _obs_offset_operand: torch.Tensor | float | None = PrivateAttr(default=None)
    _obs_scale_operand: torch.Tensor | float | None = PrivateAttr(default=None)

    def __init__(self) -> None:
        """Initialize the Deep Q-Learning model."""
        super().__init__()
        # Set PyTorch random seed
        torch.manual_seed(42)

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Configure the Deep Q-Learning model for a specific environment.

        Args:
            env: Gymnasium environment
            hyperparameters: Model hyperparameters (will be merged with defaults)

        Returns:
            True if configuration successful, False otherwise

        Raises:
            ValueError: If environment does not have discrete action space
        """
        # Validate environment has discrete action space
        if not check_space_is_discrete(env.action_space):
            logger.error(f"Deep Q-Learning requires discrete action space, got {type(env.action_space)}")
            return False

        # Configure base class (this will merge with defaults and store in self.hyperparameters)
        super().configure(env, hyperparameters)

        # Store environment spaces
        self._action_space = cast("Discrete", env.action_space)
        self._observation_space = env.observation_space

        # Get typed hyperparameters
        typed_params = self.get_hyperparameters()

        # Network, observation rescaling and frame history all derive from the
        # spaces plus frame_stack; _import() reuses this when a saved model was
        # trained with a different stack depth.
        self._build_from_spaces()
        self._build_optimizer()

        # Initialize experience replay buffer
        self._replay_buffer = ExperienceReplayBuffer(typed_params.replay_buffer_size)

        # Reset counters
        self._step_count = 0
        self._epoch_count = 0

        logger.info(
            f"'{self.model_name}' configured for environment with "
            f"discrete action space ({self._action_space.n} actions) and "
            f"observation shape {self._q_network.observation_shape} "
            f"(frame_stack={typed_params.frame_stack})"
        )
        return True

    def _single_frame_shape(self) -> tuple[int, ...]:
        """Shape of ONE observation, before any stacking."""
        if isinstance(self._observation_space, gym.spaces.Discrete):
            return (1,)
        shape = getattr(self._observation_space, "shape", None)
        return tuple(shape) if shape else (1,)

    def _observation_rescaling(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Derive an affine rescaling to [0, 1] from the observation space's own range.

        Read from the space rather than hard-coded per environment: an image space
        gives (0, 255), a bounded Box gives its own bounds, and a Discrete space is
        scaled by its cardinality. An unbounded Box cannot be rescaled this way, so
        it is left untouched (identity) rather than clipped to an invented range.

        Returns:
            (offset, scale) such that (observation - offset) / scale lands in [0, 1].
            Scalars when the bounds are uniform over the array, arrays otherwise.
        """
        space = self._observation_space
        if isinstance(space, gym.spaces.Discrete):
            # Indices run 0..n-1; guard n == 1 against a zero span.
            return 0.0, float(max(int(space.n) - 1, 1))
        if isinstance(space, gym.spaces.Box):
            low = np.asarray(space.low, dtype=np.float64)
            high = np.asarray(space.high, dtype=np.float64)
            span = high - low
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(span > 0):
                # Uniform bounds (the image case) collapse to scalars.
                if np.all(low == low.flat[0]) and np.all(span == span.flat[0]):
                    return float(low.flat[0]), float(span.flat[0])
                return low.astype(np.float32), span.astype(np.float32)
        return 0.0, 1.0

    def _build_from_spaces(self) -> None:
        """(Re)create the networks, the observation rescaling and the frame history."""
        typed_params = self.get_hyperparameters()
        frame_count = typed_params.frame_stack + 1
        frame_shape = self._single_frame_shape()

        # Stacking concatenates along the LAST axis, so an image keeps its spatial
        # dimensions and only gains channels -- which is why QNetwork needs no
        # change: _build_cnn reads its in_channels from shape[2] and sizes its
        # flattened layer from a dummy forward pass.
        stacked_shape = (*frame_shape[:-1], frame_shape[-1] * frame_count)

        num_actions = self._action_space.n
        self._q_network = QNetwork(stacked_shape, num_actions)
        self._target_network = QNetwork(stacked_shape, num_actions)
        self._target_network.load_state_dict(self._q_network.state_dict())
        self._target_network.eval()  # Target network is always in eval mode

        offset, scale = self._observation_rescaling()
        if isinstance(offset, np.ndarray) and frame_count > 1:
            # Per-element bounds must be tiled to match the stacked observation.
            offset = np.concatenate([offset] * frame_count, axis=-1)
            scale = np.concatenate([scale] * frame_count, axis=-1)
        self._obs_offset, self._obs_scale = offset, scale

        # Operands for _as_network_input, kept as None when the term is a no-op so
        # an unbounded space costs no extra pass over the batch at all.
        self._obs_offset_operand = self._rescaling_operand(offset, neutral=0.0)
        self._obs_scale_operand = self._rescaling_operand(scale, neutral=1.0)

        self._frames = deque(maxlen=frame_count)
        self._predict_frames = deque(maxlen=frame_count)

    def _build_optimizer(self) -> None:
        """Create the optimizer over the current network parameters."""
        typed_params = self.get_hyperparameters()
        self._optimizer = optim.Adam(
            self._q_network.parameters(), lr=typed_params.learning_rate, weight_decay=typed_params.weight_decay
        )

    def _to_frame(self, observation: np.ndarray | int) -> np.ndarray:
        """
        Normalise an environment observation to one array, PRESERVING its dtype.

        Keeping the native dtype is what makes stacking affordable: a uint8 image
        frame stays 1 byte per value in the replay buffer instead of 4, so the
        buffer costs no more at frame_stack=3 than it did unstacked. The cast to
        float32 happens only where a tensor is built.
        """
        if isinstance(observation, (int, np.integer)):
            return np.array([observation])
        # np.array (not asarray) so the frame never aliases the environment's own
        # buffer: the transition is pushed to the replay buffer AFTER the next
        # env.step(), so an environment reusing its array in place would otherwise
        # corrupt the stored state.
        frame = np.array(observation)
        return frame.reshape(1) if frame.ndim == 0 else frame

    def _stack(self, frames: deque) -> np.ndarray:
        """Concatenate the frame history along the last axis."""
        return frames[0] if len(frames) == 1 else np.concatenate(list(frames), axis=-1)

    def _prime(self, frames: deque, frame: np.ndarray) -> np.ndarray:
        """Fill a history by repeating the first frame of an episode."""
        frames.clear()
        for _ in range(frames.maxlen or 1):
            frames.append(frame)
        return self._stack(frames)

    @staticmethod
    def _rescaling_operand(value: np.ndarray | float, neutral: float) -> torch.Tensor | float | None:
        """Prepare one rescaling term, or None when it would be a no-op."""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype(np.float32))
        return None if value == neutral else float(value)

    def _as_network_input(self, observation: np.ndarray) -> torch.Tensor:
        """
        Build the network input: cast to float32 and rescale, in place.

        Done in torch rather than numpy on purpose. The numpy form
        `(x.astype(float32) - offset) / scale` allocates three arrays and makes
        three passes; on a batch of 32 stacked (96, 96, 12) frames that is ~42 MB
        of churn per call, twice per gradient step, and it measured at 5.8 s of a
        19.2 s episode -- more than a quarter of the whole run. The in-place torch
        version fuses the conversion and reuses one buffer.
        """
        tensor = torch.from_numpy(np.ascontiguousarray(observation)).to(self._device, dtype=torch.float32)
        if self._obs_offset_operand is not None:
            tensor.sub_(self._obs_offset_operand)
        if self._obs_scale_operand is not None:
            tensor.div_(self._obs_scale_operand)
        return tensor

    def act(self, observation: np.ndarray | int, training: bool = False) -> int:
        """
        Select an action given an observation using epsilon-greedy policy.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Selected action index

        Raises:
            ValueError: If model is not configured
        """
        if self._q_network is None or self._action_space is None:
            msg = "Model not configured. Call configure() first."
            raise ValueError(msg)

        # The observation is already stacked when it comes from run_epoch/predict;
        # rescaling onto the observation space's range happens here, at the single
        # point where a tensor is built.
        obs_tensor_source = self._to_frame(observation)

        # Get epsilon from hyperparameters
        typed_params = self.get_hyperparameters()
        epsilon = typed_params.epsilon if training else 0.0  # No exploration during evaluation

        # Epsilon-greedy action selection
        if training and random.random() < epsilon:
            # Explore: choose random action
            return random.randint(0, self._action_space.n - 1)
        else:
            # Exploit: choose best action according to Q-network
            self._q_network.eval()
            with torch.no_grad():
                obs_tensor = self._as_network_input(obs_tensor_source).unsqueeze(0)
                q_values = self._q_network(obs_tensor)
                action = q_values.argmax().item()
            self._q_network.train()
            return action

    def run_epoch(self, train_mode: bool = False) -> EpochResult:
        """
        Run a single epoch/episode using Deep Q-Learning.

        Args:
            train_mode: Whether to update the model during the episode

        Returns:
            EpochResult containing episode statistics
        """
        env = self.check_environment_or_raise()

        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        # Prime the frame history by repeating the first observation, so a stacked
        # state is well-defined from the very first step of the episode.
        obs = self._prime(self._frames, self._to_frame(observation))

        while not done:
            action = self.act(obs, training=train_mode)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            # Advance the history, then read the stacked next state out of it.
            self._frames.append(self._to_frame(next_observation))
            next_obs = self._stack(self._frames)

            if train_mode:
                # Store transition in replay buffer
                if self._replay_buffer is not None:
                    self._replay_buffer.push(obs.copy(), action, float(reward), next_obs.copy(), done)

                # Update epsilon (decay) - done per step during training
                typed_params = self.get_hyperparameters()
                current_epsilon = typed_params.epsilon
                epsilon_decay = typed_params.epsilon_decay
                epsilon_min = typed_params.epsilon_min
                new_epsilon = max(epsilon_min, current_epsilon * (1 - epsilon_decay))
                typed_params.epsilon = new_epsilon
                # Update self.hyperparameters list to reflect the change
                self.hyperparameters = [HyperParameter(key=k, value=v) for k, v in typed_params.to_dict().items()]

                self._step_count += 1

                # Experience replay, every step_modulo *environment steps*. This has to
                # happen inside the step loop: performing it once per episode instead
                # would bootstrap from a single end-of-episode update, i.e. one gradient
                # step per episode return, which is Monte-Carlo control rather than the
                # per-transition TD update DQN is defined by.
                if (
                    self._step_count % typed_params.step_modulo == 0
                    and self._replay_buffer is not None
                    and len(self._replay_buffer) >= typed_params.batch_size
                ):
                    self._train_step()

                # The target network must LAG the online network, so its synchronisation
                # is on its own period and never chained to _train_step(): copying the
                # weights after every update would make the bootstrap target come from
                # the current weights, which is exactly having no target network at all.
                if (
                    self._step_count % typed_params.target_update_frequency == 0
                    and self._target_network is not None
                    and self._q_network is not None
                ):
                    self._target_network.load_state_dict(self._q_network.state_dict())

            obs = next_obs

        if train_mode:
            self._epoch_count += 1

        return EpochResult(
            reward=float(episode_reward),
            steps_number=episode_length,
            final_state="truncated" if truncated else "terminated",
        )

    def _train_step(self) -> None:
        """Perform one training step using experience replay."""
        if (
            self._q_network is None
            or self._target_network is None
            or self._optimizer is None
            or self._replay_buffer is None
        ):
            return

        typed_params = self.get_hyperparameters()

        # Sample batch from replay buffer
        batch = self._replay_buffer.sample(typed_params.batch_size)

        # Convert batch to tensors
        # States are stored in their native dtype (uint8 for image observations) and
        # rescaled here, so the buffer stays compact and the network still sees
        # values in [0, 1].
        states = self._as_network_input(np.array([s for s, _, _, _, _ in batch]))
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self._device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self._device)
        next_states = self._as_network_input(np.array([ns for _, _, _, ns, _ in batch]))
        dones = torch.BoolTensor([d for _, _, _, _, d in batch]).to(self._device)

        # Compute current Q-values
        current_q_values = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self._target_network(next_states).max(1)[0]
            target_q_values = rewards + (typed_params.discount_factor * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _export(self) -> dict:
        """
        Export Deep Q-Learning model data for serialization.

        Returns:
            Dictionary containing model data ready for JSON serialization
        """
        if self._q_network is None:
            return {}

        # Save network state dict
        state_dict = self._q_network.state_dict()
        # Convert tensors to lists for JSON serialization
        # Handle both single tensors and nested structures
        serialized_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                serialized_state_dict[k] = v.cpu().tolist()
            else:
                serialized_state_dict[k] = v

        return {
            "q_network_state_dict": serialized_state_dict,
            "epoch_count": self._epoch_count,
            "step_count": self._step_count,
            # The first conv layer's in_channels depends on frame_stack, so the
            # stack depth has to travel with the weights: `hercule play` configures
            # with DEFAULT hyperparameters and would otherwise build a network of
            # the wrong shape and fail to load a stacked model.
            "frame_stack": self.get_hyperparameters().frame_stack,
            "observation_shape": list(self._q_network.observation_shape),
        }

    def _import(self, model_data: dict) -> None:
        """
        Import Deep Q-Learning model data from serialized format.

        Args:
            model_data: Dictionary containing model data from JSON
        """
        # Rebuild for the saved stack depth before touching the weights. This is
        # the normal path for `hercule play`, which configures with defaults, not
        # an edge case.
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
                self._build_from_spaces()
                self._build_optimizer()

        if "q_network_state_dict" in model_data and self._q_network is not None:
            # Convert lists back to tensors
            state_dict = {}
            for k, v in model_data["q_network_state_dict"].items():
                if isinstance(v, list):
                    # Handle nested lists (for multi-dimensional tensors)
                    state_dict[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    state_dict[k] = v

            self._q_network.load_state_dict(state_dict)
            # Also update target network
            if self._target_network is not None:
                self._target_network.load_state_dict(state_dict)

        if "epoch_count" in model_data:
            self._epoch_count = model_data["epoch_count"]
        if "step_count" in model_data:
            self._step_count = model_data["step_count"]

        logger.info("Note: Call configure() with environment before using loaded model")

    def load_from_dict(self, model_data: dict) -> None:
        """
        Load a trained Deep Q-Learning model from a dictionary.

        Args:
            model_data: Dictionary containing model data

        Raises:
            KeyError: If required keys are missing from model_data
        """
        self._import(model_data)
        logger.info(f"Loaded {self.model_name} model from dictionary")

    def predict(self, observation: np.ndarray | int) -> int:
        """
        Predict the best action for a given observation (inference mode).

        Args:
            observation: Current observation from the environment

        Note:
            Frame stacking needs a history, but `RLModel` has no episode-reset hook
            and callers (e.g. `controller.play_interactive`) drive `env.reset()`
            themselves, so an episode boundary is not observable from here. The
            history is therefore carried across boundaries: for the first
            `frame_stack` steps of a new episode the stack still holds frames from
            the previous one. That is `frame_stack` steps out of an episode's
            length, and the alternative -- adding a lifecycle method to `RLModel` --
            is a Root Class Registry change requiring a constitution review.

        Returns:
            Selected action index
        """
        frame = self._to_frame(observation)
        if len(self._predict_frames) == 0:
            self._prime(self._predict_frames, frame)
        else:
            self._predict_frames.append(frame)
        return self.act(self._stack(self._predict_frames), training=False)


__all__ = ["DeepQLearningModel"]
