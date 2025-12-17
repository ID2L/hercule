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
    replay_modulo: int = Field(default=4, description="Number of epochs before experience replay step")
    target_update_frequency: int = Field(default=1000, description="Number of steps before updating target network")
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

        # Get observation shape
        if hasattr(self._observation_space, "shape"):
            observation_shape = self._observation_space.shape
        else:
            # For discrete observation space, convert to shape
            if isinstance(self._observation_space, gym.spaces.Discrete):
                observation_shape = (1,)
            else:
                # Try to get shape from space attributes
                observation_shape = getattr(self._observation_space, "shape", (1,))

        num_actions = self._action_space.n

        # Initialize Q-network and target network
        self._q_network = QNetwork(observation_shape, num_actions)
        self._target_network = QNetwork(observation_shape, num_actions)
        self._target_network.load_state_dict(self._q_network.state_dict())
        self._target_network.eval()  # Target network is always in eval mode

        # Initialize optimizer
        self._optimizer = optim.Adam(
            self._q_network.parameters(), lr=typed_params.learning_rate, weight_decay=typed_params.weight_decay
        )

        # Initialize experience replay buffer
        self._replay_buffer = ExperienceReplayBuffer(typed_params.replay_buffer_size)

        # Reset counters
        self._step_count = 0
        self._epoch_count = 0

        logger.info(
            f"'{self.model_name}' configured for environment with "
            f"discrete action space ({num_actions} actions) and "
            f"observation shape {observation_shape}"
        )
        return True

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

        # Convert observation to numpy array if needed
        if isinstance(observation, int):
            obs_array = np.array([observation], dtype=np.float32)
        else:
            obs_array = np.array(observation, dtype=np.float32)
            # Ensure correct shape
            if obs_array.ndim == 0:
                obs_array = obs_array.reshape(1)

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
                obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self._device)
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

        # Convert initial observation
        if isinstance(observation, int):
            obs = np.array([observation], dtype=np.float32)
        else:
            obs = np.array(observation, dtype=np.float32)
            # Ensure correct shape
            if obs.ndim == 0:
                obs = obs.reshape(1)

        while not done:
            action = self.act(obs, training=train_mode)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            if train_mode:
                # Convert next observation
                if isinstance(next_observation, int):
                    next_obs = np.array([next_observation], dtype=np.float32)
                else:
                    next_obs = np.array(next_observation, dtype=np.float32)
                    # Ensure correct shape
                    if next_obs.ndim == 0:
                        next_obs = next_obs.reshape(1)

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

            if train_mode:
                obs = next_obs
            else:
                if isinstance(next_observation, int):
                    obs = np.array([next_observation], dtype=np.float32)
                else:
                    obs = np.array(next_observation, dtype=np.float32)
                    # Ensure correct shape
                    if obs.ndim == 0:
                        obs = obs.reshape(1)

        if train_mode:
            # Perform experience replay every replay_modulo epochs
            typed_params = self.get_hyperparameters()
            if (
                self._epoch_count % typed_params.replay_modulo == 0
                and self._replay_buffer is not None
                and len(self._replay_buffer) >= typed_params.batch_size
            ):
                # Perform one training step with experience replay
                self._train_step()

                # Copy network weights to target network after experience replay
                if self._target_network is not None and self._q_network is not None:
                    self._target_network.load_state_dict(self._q_network.state_dict())

            # Update target network periodically (alternative to epoch-based update)
            if self._step_count % typed_params.target_update_frequency == 0:
                if self._target_network is not None and self._q_network is not None:
                    self._target_network.load_state_dict(self._q_network.state_dict())

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
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(self._device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self._device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self._device)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in batch])).to(self._device)
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
        }

    def _import(self, model_data: dict) -> None:
        """
        Import Deep Q-Learning model data from serialized format.

        Args:
            model_data: Dictionary containing model data from JSON
        """
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

        Returns:
            Selected action index
        """
        return self.act(observation, training=False)


__all__ = ["DeepQLearningModel"]
