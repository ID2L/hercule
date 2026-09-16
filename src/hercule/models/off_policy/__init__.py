"""Shared scaffolding for off-policy, replay-based models.

`OffPolicyReplayModel` owns what every model learning from a stored history of
transitions needs and none of them should own privately: the episode loop, the
replay store, observation history and rescaling, device and seeding, the mapping
between a policy's own action coordinates and the environment's, and checkpoint
assembly. A concrete algorithm supplies its networks, its action selection, its
learning update, and the handful of hooks listed below.

Two properties of this module are load-bearing and easy to break by accident.

**The order in which parameterised modules are constructed is observable
behaviour.** `nn.Linear` and `nn.Conv2d` each draw from torch's global RNG in
their constructor, and `configure()` seeds that RNG once, immediately before
`_build_networks()`. Any change to the order or number of those constructions
produces different initial weights from the same seed. `Encoder` therefore builds
its layers eagerly and in a fixed sequence, and nothing between the seeding call
and `_build_networks()` may draw from torch.

**A `state_dict`'s keys are attribute paths.** Renaming or re-nesting a module
renames every key in every checkpoint already written, and `load_state_dict` is
strict by default. `_migrate_parameter_keys()` exists for exactly that, and is
applied whenever an older payload is read.
"""

import base64
import io
import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Mapping
from typing import ClassVar, Generic

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import PrivateAttr

from hercule.config import ParameterValue
from hercule.models import HyperParamsType, RLModel
from hercule.models.epoch_result import EpochResult


logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Feature extractor: MLP for a vector observation, CNN for an image.

    The layers live in one `nn.Sequential` named `layers`, built eagerly in the
    constructor. Both choices are deliberate: a lazily built encoder would move
    parameter construction out of `configure()` to an arbitrary later point, which
    changes when torch's RNG is drawn from, and a differently named or nested
    container would rename every key in every stored checkpoint.
    """

    def __init__(self, observation_shape: tuple[int, ...]) -> None:
        """
        Build the extractor for one observation shape.

        Args:
            observation_shape: The stacked observation shape, e.g. `(4,)` for a
                vector or `(96, 96, 12)` for four stacked RGB frames.
        """
        super().__init__()
        self.observation_shape = tuple(observation_shape)

        if len(self.observation_shape) == 3:
            self._build_cnn()
        elif len(self.observation_shape) == 1:
            self._build_mlp(self.observation_shape[0], flatten=False)
        else:
            # PRE-EXISTING defect, fixed here: a rank other than 1 or 3 (e.g. a
            # multi-axis Box observation of shape (2, 2)) sized its input from
            # int(np.prod(observation_shape)) but forward() passed the tensor
            # straight into the Linear stack without flattening it, so a batch
            # arrived as (batch, 2, 2) against a Linear expecting 4 features.
            # nn.Flatten() has no parameters -- it draws nothing from torch's
            # RNG -- but it DOES occupy an index in the Sequential, so it is
            # added ONLY on this path: the rank-1 path above and the rank-3 CNN
            # path in _build_cnn() must keep their exact existing layer indices,
            # which the golden fixture and the checkpoint key migration tables
            # (_VECTOR_KEY_MIGRATION / _IMAGE_KEY_MIGRATION) both pin.
            self._build_mlp(int(np.prod(self.observation_shape)), flatten=True)

    def _build_mlp(self, input_size: int, flatten: bool) -> None:
        """Two hidden layers, matching the shape this project has always used."""
        layers: list[nn.Module] = []
        if flatten:
            layers.append(nn.Flatten())
        layers.extend(
            [
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.output_size = 128

    def _build_cnn(self) -> None:
        """The DQN-paper convolutional stack, followed by one dense layer.

        The convolutions are constructed first, then the flattened size is measured
        by a forward pass of zeros -- which draws nothing from any RNG -- and only
        then is the dense layer constructed. That sequence is the one this project's
        checkpoints were written under and must not be reordered.
        """
        height, width, channels = self.observation_shape
        convolutions = [
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        ]
        with torch.no_grad():
            probe = nn.Sequential(*convolutions)(torch.zeros(1, channels, height, width))
            flattened_size = int(np.prod(probe.shape[1:]))

        self.layers = nn.Sequential(
            *convolutions,
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
        )
        self.output_size = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations, moving channels first for an image."""
        if len(self.observation_shape) == 3 and x.dim() == 4:
            # (batch, H, W, C) -> (batch, C, H, W)
            if x.shape[1] == self.observation_shape[0] and x.shape[3] == self.observation_shape[2]:
                x = x.permute(0, 3, 1, 2)
        return self.layers(x)


class ExperienceReplayBuffer:
    """Stored transitions, sampled uniformly.

    `terminated` and `truncated` are kept separate rather than collapsed into one
    `done` flag, because they mean different things to a learning target:
    `terminated` is a genuine MDP terminal state and must zero the bootstrap term,
    while `truncated` is an external time limit whose successor state still has
    value and must not.

    The action is stored as whatever the model chose to store -- an index for a
    discrete model, an array for a continuous one -- so collation stacks rather
    than assuming a scalar.
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: Maximum number of transitions retained.
        """
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one transition."""
        self.buffer.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, int | np.ndarray, float, np.ndarray, bool, bool]]:
        """Sample uniformly without replacement, capped at the buffer's size."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return len(self.buffer)


class OffPolicyReplayModel(RLModel[HyperParamsType], ABC, Generic[HyperParamsType]):
    """Abstract base for models that learn from a replay buffer.

    Subclasses implement `_build_networks`, `_build_optimizers`, `_select_action`,
    `_update`, `_networks` and `_optimizers`. Everything else has a default.
    """

    # Bumped whenever the payload shape of `_export()` changes. Version 3 turned
    # `optimizer_state_b64` into a name-keyed mapping, because a model may now hold
    # several optimizers where the deep model held one. `_import()` keeps reading
    # every earlier version, so a checkpoint already on disk never breaks.
    _CHECKPOINT_FORMAT_VERSION: ClassVar[int] = 3

    _replay_buffer: ExperienceReplayBuffer | None = PrivateAttr(default=None)
    _action_space: gym.Space | None = PrivateAttr(default=None)
    _observation_space: gym.Space | None = PrivateAttr(default=None)
    _device: torch.device = PrivateAttr(default_factory=lambda: torch.device("cpu"))
    _step_count: int = PrivateAttr(default=0)
    _epoch_count: int = PrivateAttr(default=0)

    # Frame history for observation stacking. Two independent deques: run_epoch()
    # drives its own episode loop, predict() is called step by step from outside.
    _frames: deque = PrivateAttr(default_factory=lambda: deque(maxlen=1))
    _predict_frames: deque = PrivateAttr(default_factory=lambda: deque(maxlen=1))
    _stacked_observation_shape: tuple[int, ...] = PrivateAttr(default=())

    # Affine rescaling of observations, derived from the observation space's own
    # range rather than hard-coded: scalars when the bounds are uniform.
    _obs_offset: np.ndarray | float = PrivateAttr(default=0.0)
    _obs_scale: np.ndarray | float = PrivateAttr(default=1.0)
    _obs_offset_operand: torch.Tensor | float | None = PrivateAttr(default=None)
    _obs_scale_operand: torch.Tensor | float | None = PrivateAttr(default=None)

    # Per-dimension action bounds, cached ONLY for a Box action space. A Discrete
    # space has no `low`/`high` at all, so reading them unconditionally would break
    # every discrete model at configure() time.
    _action_low: np.ndarray | None = PrivateAttr(default=None)
    _action_high: np.ndarray | None = PrivateAttr(default=None)

    # Model-owned NumPy generator: every NumPy-side random draw a subclass makes
    # must go through it, never through the global `np.random.*` functions, so its
    # `bit_generator.state` is a complete, checkpointable record of what has been
    # consumed.
    _rng: np.random.Generator = PrivateAttr(default_factory=lambda: np.random.default_rng(42))

    # Whether the FIRST `env.reset()` of this run should carry `seed=`. True after
    # a fresh `configure()` (a new run); flipped to False by `_import()` (a resumed
    # run), so a reload never re-issues the seeded reset a fresh run gets once.
    _needs_seeded_reset: bool = PrivateAttr(default=True)

    # ------------------------------------------------------------------ lifecycle

    def configure(self, env: gym.Env, hyperparameters: dict[str, ParameterValue]) -> bool:
        """
        Prepare the model for one environment.

        The order here is a contract, not a convenience. Every random stream is
        seeded BEFORE `_build_networks()`, and nothing between the two draws from
        torch: network initialisation consumes torch's global RNG in the layer
        constructors, so seeding afterwards would leave the weights unseeded and
        inserting a draw in between would shift every weight that follows.

        Args:
            env: The environment to configure against.
            hyperparameters: Values merged over the model's defaults.

        Returns:
            True when configuration succeeded.
        """
        super().configure(env, hyperparameters)

        self._action_space = env.action_space
        self._observation_space = env.observation_space

        typed_params = self.get_hyperparameters()
        torch.manual_seed(typed_params.seed)
        random.seed(typed_params.seed)
        self._rng = np.random.default_rng(typed_params.seed)

        self._needs_seeded_reset = True

        self._cache_action_bounds()
        self._build_observation_pipeline()
        self._build_networks()
        self._build_optimizers()

        self._replay_buffer = ExperienceReplayBuffer(typed_params.replay_buffer_size)
        self._step_count = 0
        self._epoch_count = 0
        return True

    def _cache_action_bounds(self) -> None:
        """Cache per-dimension action bounds, for a `Box` action space only.

        Flattened (`.reshape(-1)`), regardless of the space's own shape: a
        multi-axis `Box` (e.g. shape `(2, 2)`) stores its bounds with that same
        shape, while `to_env_action`/`to_policy_action` work on the FLAT vector
        the policy and the replay buffer actually use (`_action_dimensions` is
        `int(np.prod(shape))`). Caching the bounds flat once here means the
        mapping arithmetic never has to reconcile the two shapes itself.
        """
        if isinstance(self._action_space, gym.spaces.Box):
            self._action_low = np.asarray(self._action_space.low, dtype=np.float32).reshape(-1)
            self._action_high = np.asarray(self._action_space.high, dtype=np.float32).reshape(-1)
        else:
            self._action_low = None
            self._action_high = None

    def _build_observation_pipeline(self) -> None:
        """Derive the stacked observation shape, the rescaling and the histories.

        Draws from no random stream, so it may run either side of
        `_build_networks()` without changing a single weight. It runs before,
        because `_build_networks()` needs the stacked shape.
        """
        typed_params = self.get_hyperparameters()
        frame_count = typed_params.frame_stack + 1
        frame_shape = self._single_frame_shape()

        # Stacking concatenates along the LAST axis, so an image keeps its spatial
        # dimensions and only gains channels -- which is why the encoder needs no
        # change: its CNN branch reads in_channels from shape[2] and sizes its
        # flattened layer from a dummy forward pass.
        self._stacked_observation_shape = (*frame_shape[:-1], frame_shape[-1] * frame_count)

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

    def _rebuild_from_spaces(self) -> None:
        """Rebuild the pipeline, the networks and the optimizers.

        Used by `_import()` when a checkpoint was written at a different frame
        stack than the one currently configured -- `hercule play` configures with
        DEFAULT hyperparameters, so this is the normal path there, not an edge case.
        """
        self._build_observation_pipeline()
        self._build_networks()
        self._build_optimizers()

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

    # ------------------------------------------------------------- observations

    def begin_episode(self) -> None:
        """Drop the frame history so a stacked state never spans two episodes."""
        self._frames.clear()
        self._predict_frames.clear()

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

    # ------------------------------------------------------------ action mapping

    def to_env_action(self, normalised: np.ndarray) -> np.ndarray:
        """
        Map a normalised action in `[-1, 1]^d` onto the environment's own bounds.

        Per dimension, so that `-1` lands exactly on that dimension's `low` and
        `+1` exactly on its `high`. The asymmetric case is the one that matters:
        on `CarRacing-v3(continuous=True)` the bounds are `low = [-1, 0, 0]` and
        `high = [1, 1, 1]`, so forwarding a normalised action unchanged would apply
        negative throttle and negative brake. Gymnasium does not reject that; the
        car simply never accelerates, and the run still produces a plausible reward
        curve.

        `normalised` is FLAT (`_action_dimensions` elements), matching what the
        policy and the replay buffer use. The result is reshaped back to the
        action space's own declared shape before returning, so a multi-axis `Box`
        (e.g. shape `(2, 2)`) gets back exactly what `env.step()` expects rather
        than the flat vector the arithmetic is done on.
        """
        if self._action_low is None or self._action_high is None:
            msg = (
                "Action mapping is only defined for a Box action space. A model whose own action "
                "coordinates are the environment's -- any Discrete-action model -- must return the "
                "same value for both halves of _select_action's pair rather than calling this."
            )
            raise ValueError(msg)
        bias = (self._action_high + self._action_low) / 2.0
        scale = (self._action_high - self._action_low) / 2.0
        env_action = bias + scale * np.asarray(normalised, dtype=np.float32).reshape(-1)
        return env_action.reshape(self._action_space.shape)

    def to_policy_action(self, env_action: np.ndarray) -> np.ndarray:
        """
        Inverse of `to_env_action`, per dimension.

        `env_action` carries the action space's own shape (e.g. `(2, 2)`); it is
        flattened before the arithmetic, matching the flat bounds cached by
        `_cache_action_bounds()`, and the result is the flat normalised vector the
        policy and the replay buffer use.
        """
        if self._action_low is None or self._action_high is None:
            msg = "Action mapping is only defined for a Box action space"
            raise ValueError(msg)
        bias = (self._action_high + self._action_low) / 2.0
        scale = (self._action_high - self._action_low) / 2.0
        return (np.asarray(env_action, dtype=np.float32).reshape(-1) - bias) / scale

    # -------------------------------------------------------------- episode loop

    def act(self, observation: np.ndarray | int, training: bool = False) -> int | float | np.ndarray:
        """
        Choose an action for one observation, in the environment's coordinates.

        Never mutates adaptive state: any per-step schedule a subclass keeps
        advances in `_on_training_step()`, which only the episode loop calls.
        """
        return self._select_action(observation, training=training)[0]

    def predict(self, observation: np.ndarray | int) -> int | float | np.ndarray:
        """
        Choose an action for inference, driving the caller-side frame history.

        Frame stacking needs a history, and an episode boundary is NOT observable
        from a single observation, so callers driving their own episode loop must
        call `begin_episode()` right after `env.reset()`. This method then re-primes
        the history from the episode's first observation instead of carrying frames
        over from the previous one.
        """
        frame = self._to_frame(observation)
        if len(self._predict_frames) == 0:
            self._prime(self._predict_frames, frame)
        else:
            self._predict_frames.append(frame)
        return self.act(self._stack(self._predict_frames), training=False)

    def run_epoch(self, train_mode: bool = False) -> EpochResult:
        """
        Run one episode.

        Two orderings below are load-bearing. `begin_episode()` precedes priming,
        because it CLEARS the histories -- priming first and clearing second wipes
        the primed history and leaves every subsequent stacked observation short.
        And `push()` sits inside the training branch, because hoisting it out would
        write greedy evaluation transitions into the replay buffer, which the next
        training epoch would then sample.

        Args:
            train_mode: Whether to store transitions and take gradient steps.

        Returns:
            The episode's reward, length and how it ended.
        """
        env = self.check_environment_or_raise()

        # Seed only the very first reset of a fresh run, so `seed` controls the
        # run's starting point without pinning every episode to the same one.
        if self._needs_seeded_reset:
            observation, _ = env.reset(seed=self.get_hyperparameters().seed)
            self._needs_seeded_reset = False
        else:
            observation, _ = env.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False

        self.begin_episode()
        obs = self._prime(self._frames, self._to_frame(observation))

        while not done:
            env_action, stored_action = self._select_action(obs, training=train_mode)
            next_observation, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            self._frames.append(self._to_frame(next_observation))
            next_obs = self._stack(self._frames)

            if train_mode:
                # `terminated` and `truncated` are kept separate here (not collapsed
                # into `done`, used above only to decide whether the loop stops): a
                # learning target must zero its bootstrap term on a genuine terminal
                # state but NOT on a truncation.
                if self._replay_buffer is not None:
                    self._replay_buffer.push(
                        obs.copy(), stored_action, float(reward), next_obs.copy(), terminated, truncated
                    )

                self._on_training_step()
                self._step_count += 1

                if self._ready_to_update() and self._replay_buffer is not None:
                    self._update(self._replay_buffer.sample(self.get_hyperparameters().batch_size))

                # Synchronisation is on its own period and is deliberately NOT
                # chained to whether a gradient step was taken: at step_modulo > 1
                # the two clocks differ, and coupling them would silently multiply
                # the configured interval.
                self._sync_targets()

            obs = next_obs

        if train_mode:
            self._epoch_count += 1

        return EpochResult(
            reward=float(episode_reward),
            steps_number=episode_length,
            final_state="truncated" if truncated else "terminated",
        )

    # --------------------------------------------------------------------- hooks

    @abstractmethod
    def _build_networks(self) -> None:
        """Construct every parameterised module, eagerly and in a fixed order."""

    @abstractmethod
    def _build_optimizers(self) -> None:
        """Construct the optimizers. Called immediately after `_build_networks()`."""

    @abstractmethod
    def _select_action(
        self, observation: np.ndarray | int, training: bool
    ) -> tuple[int | float | np.ndarray, int | float | np.ndarray]:
        """
        Choose an action, and say how it should be stored.

        Owns ALL exploration, including any warmup phase, so the ancestor never
        samples an action itself.

        Returns:
            `(env_action, stored_action)`. The first drives `env.step()`; the second
            is what the replay buffer holds, in the model's own coordinates. They
            are the same object for a model whose coordinates are the environment's.
        """

    @abstractmethod
    def _update(self, batch: list) -> None:
        """Take one gradient step from a sampled batch."""

    @abstractmethod
    def _networks(self) -> Mapping[str, nn.Module]:
        """Every module whose weights must survive a resume, delayed copies included."""

    @abstractmethod
    def _optimizers(self) -> Mapping[str, optim.Optimizer]:
        """Every optimizer whose state must survive a resume."""

    def _ready_to_update(self) -> bool:
        """Whether this environment step should also be a gradient step."""
        typed_params = self.get_hyperparameters()
        return (
            self._replay_buffer is not None
            and self._step_count % typed_params.step_modulo == 0
            and len(self._replay_buffer) >= typed_params.batch_size
        )

    def _on_training_step(self) -> None:
        """Advance whatever per-step state the model keeps. No-op by default.

        Deliberately not named for exploration: the ancestor makes no claim about
        what a subclass advances here, which is what keeps it compatible with a
        model that has no exploration schedule at all.
        """

    def _target_pairs(self) -> Iterable[tuple[nn.Module, nn.Module]]:
        """`(live, delayed)` module pairs for the default hard-copy. Empty by default."""
        return ()

    def _target_sync_interval(self) -> int | None:
        """Environment steps between hard copies, or None to never hard-copy."""
        return None

    def _sync_targets(self) -> None:
        """Hard-copy each declared pair on its interval. Called once per env step.

        `None` means never, and is the only value special-cased. Any other value
        goes straight into the modulo, deliberately: a subclass that declares an
        interval of `0` gets the `ZeroDivisionError` it would have got before this
        scaffolding existed, and a negative one keeps synchronising on the steps
        Python's modulo says it should. Swallowing either would be a behaviour
        change on an input nothing validates against, and this hook is on the path
        a bit-identity requirement covers.
        """
        interval = self._target_sync_interval()
        if interval is None or self._step_count % interval != 0:
            return
        for live, delayed in self._target_pairs():
            delayed.load_state_dict(live.state_dict())

    def _extra_state(self) -> dict:
        """Subclass-specific non-module state, plus anything the shapes depend on.

        NOT the random streams and NOT the counters: the ancestor owns and exports
        those itself, and a subclass emitting them too would write the same state
        under two keys.
        """
        return {}

    def _load_extra_state(self, model_data: dict) -> None:
        """Restore what `_extra_state()` wrote."""

    def _migrate_parameter_keys(self, state_dict: dict, network: str) -> dict:
        """
        Rename parameter keys read from a checkpoint older than the current layout.

        Identity by default. A model whose module layout has changed since a format
        it still reads must override this: `state_dict` keys are attribute paths, so
        re-nesting a module renames every one of them, and `load_state_dict` raises
        on a key it did not expect. The weights are right; the names are not.
        """
        return state_dict

    # ---------------------------------------------------------------- checkpoint

    @staticmethod
    def _encode_state_dict(payload: dict) -> str:
        """Base64-encode a torch-serialisable payload, for embedding in JSON.

        This replaces a `tensor.tolist()` + `json.dump` encoding that measured
        3.95 s / 140.9 MB on a stacked CarRacing-shaped network against
        0.23 s / 11.7 MB here: a tensor is written as its own compact binary buffer
        rather than a nested Python list of floats re-parsed by the JSON encoder.
        """
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _decode_state_dict(encoded: str) -> dict:
        """Decode a payload produced by `_encode_state_dict`.

        `weights_only=True` is non-negotiable: `torch.load` with
        `weights_only=False` runs arbitrary pickled code on load, which would turn
        loading a shared `model.json` (the normal `hercule play` path) into an
        arbitrary-code-execution vector.
        """
        buffer = io.BytesIO(base64.b64decode(encoded))
        return torch.load(buffer, weights_only=True)

    def _export(self) -> dict:
        """
        Assemble the checkpoint.

        Carries everything a resume needs: every network including the delayed
        copies -- in their own right, so a resumed run keeps their lag instead of
        restarting it at zero -- every optimizer's state, all three random streams,
        the counters, and whatever `_extra_state()` adds.

        Out of scope, deliberately: the replay buffer's CONTENTS. Storing every
        transition would cost gigabytes per checkpoint, which is the entire reason a
        *replay* buffer exists. The consequence is that a resumed off-policy run
        restarts with an EMPTY buffer and re-fills it from scratch: a real
        discontinuity in the learning curve, accepted here rather than left unsaid.
        """
        networks = self._networks()
        if not networks:
            return {}

        return {
            "format_version": self._CHECKPOINT_FORMAT_VERSION,
            "networks_b64": {name: self._encode_state_dict(module.state_dict()) for name, module in networks.items()},
            "optimizer_state_b64": {
                name: self._encode_state_dict(optimizer.state_dict()) for name, optimizer in self._optimizers().items()
            },
            # `random.getstate()` (a tuple of ints, safe under `weights_only=True`)
            # and `_rng.bit_generator.state` (a dict of ints/str, likewise safe) --
            # NEVER the legacy `np.random.get_state()` tuple, which embeds an
            # ndarray and raises `UnpicklingError` under `weights_only=True`.
            "rng_state_b64": self._encode_state_dict(
                {
                    "torch": torch.get_rng_state(),
                    "python_random": random.getstate(),
                    "numpy_generator": self._rng.bit_generator.state,
                }
            ),
            "epoch_count": self._epoch_count,
            "step_count": self._step_count,
            **self._extra_state(),
        }

    def _import(self, model_data: dict) -> None:
        """
        Restore a checkpoint, at any format version this model has ever written.

        Dispatches on the payload: version 3 reads a name-keyed optimizer mapping,
        version 2 a single bare optimizer state, and the pre-006 form a list-encoded
        online network only. Parameter keys from either older form go through
        `_migrate_parameter_keys()` first.
        """
        self._load_extra_state(model_data)

        if "networks_b64" in model_data:
            version = int(model_data.get("format_version", 2))
            self._import_networks(model_data["networks_b64"], version)
            self._import_optimizers(model_data.get("optimizer_state_b64"), version)
            self._import_rng(model_data.get("rng_state_b64"))
            # A resumed run must not re-issue the seeded first reset: the RNG state
            # just restored is what should drive the next env.reset(), not a fresh
            # `seed=`.
            self._needs_seeded_reset = False
        elif "q_network_state_dict" in model_data:
            self._import_legacy(model_data["q_network_state_dict"])
            self._needs_seeded_reset = False

        if "epoch_count" in model_data:
            self._epoch_count = model_data["epoch_count"]
        if "step_count" in model_data:
            self._step_count = model_data["step_count"]

        logger.info("Note: Call configure() with environment before using loaded model")

    def _import_networks(self, encoded: dict, version: int) -> None:
        """Load each module named by `_networks()` that the payload carries."""
        for name, module in self._networks().items():
            if name not in encoded:
                continue
            state_dict = self._decode_state_dict(encoded[name])
            if version < self._CHECKPOINT_FORMAT_VERSION:
                state_dict = self._migrate_parameter_keys(state_dict, name)
            module.load_state_dict(state_dict)

    def _import_optimizers(self, encoded: dict | str | None, version: int) -> None:
        """Load optimizer state from either payload shape.

        Version 3 stores a name-keyed mapping; version 2 stored a single bare
        state, because the only model writing it had exactly one optimizer.
        """
        if encoded is None:
            return
        optimizers = self._optimizers()
        if isinstance(encoded, str):
            if len(optimizers) == 1:
                next(iter(optimizers.values())).load_state_dict(self._decode_state_dict(encoded))
            else:
                logger.warning(
                    "Checkpoint carries one unnamed optimizer state but this model has "
                    f"{len(optimizers)}; optimizer state not restored."
                )
            return
        for name, optimizer in optimizers.items():
            if name in encoded:
                optimizer.load_state_dict(self._decode_state_dict(encoded[name]))

    def _import_rng(self, encoded: str | None) -> None:
        """Restore all three random streams."""
        if encoded is None:
            return
        rng_state = self._decode_state_dict(encoded)
        torch.set_rng_state(rng_state["torch"])
        random.setstate(rng_state["python_random"])
        self._rng.bit_generator.state = rng_state["numpy_generator"]

    def _import_legacy(self, encoded_state_dict: dict) -> None:
        """Load the pre-006 format: one list-encoded network, nothing else.

        Its keys are the old attribute paths too, so they go through the same
        migration a version-2 payload does. A legacy branch that loaded them
        verbatim would fail against the current modules, which is the opposite of
        the compatibility it exists to provide.
        """
        state_dict = {
            key: torch.tensor(value, dtype=torch.float32) if isinstance(value, list) else value
            for key, value in encoded_state_dict.items()
        }
        self._load_legacy_networks(self._migrate_parameter_keys(state_dict, "online"))

    def _load_legacy_networks(self, state_dict: dict) -> None:
        """Place a single migrated legacy state dict. Subclass-specific by nature."""


__all__ = ["Encoder", "ExperienceReplayBuffer", "OffPolicyReplayModel"]
