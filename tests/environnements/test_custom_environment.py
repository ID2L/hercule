"""Tests for custom (non-registry) Gymnasium environments and the action discretization wrapper."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError

from hercule.config import ActionDiscretizationConfig, EnvironmentConfig, load_config_from_yaml
from hercule.controller import CancellationToken, play_interactive
from hercule.environnements import (
    DiscretizeActionWrapper,
    EnvironmentFactory,
    load_environment,
    register_custom_environment,
    save_environment,
)
from hercule.supervisor import Supervisor


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CONFIG_PATH = FIXTURES_DIR / "custom_env_config.yaml"
ENV_ID = "HerculeTestLineEnv-v0"


class TestCustomEnvironmentConfig:
    def test_custom_fields_survive_variant_expansion(self):
        config = load_config_from_yaml(CONFIG_PATH)
        environments = config.get_environment_configs()

        # 2 step_size variants + 1 discretized environment
        assert len(environments) == 3
        for environment in environments:
            assert environment.entry_point == "line_env:LineEnv"
            assert environment.python_path == [(FIXTURES_DIR / "custom_env").resolve()]
        assert environments[2].action_discretization == ActionDiscretizationConfig(bins=3)

    def test_discretization_is_part_of_the_directory_signature(self):
        config = load_config_from_yaml(CONFIG_PATH)
        signatures = [environment.get_hyperparameters_signature() for environment in config.get_environment_configs()]

        assert signatures[2] == "max_epi_ste_20__act_dis_3"
        assert len(set(signatures)) == 3

    def test_entry_point_must_name_module_and_class(self):
        with pytest.raises(ValidationError):
            EnvironmentConfig(name="X-v0", entry_point="line_env")

    def test_bins_must_be_at_least_two(self):
        with pytest.raises(ValidationError):
            ActionDiscretizationConfig(bins=[5, 1])


class TestDiscretizeActionWrapper:
    def test_maps_indices_to_the_grid_of_each_dimension(self):
        env = gym.make("MountainCarContinuous-v0")  # Box(1,) action in [-1, 1]
        wrapped = DiscretizeActionWrapper(env, 5)

        assert wrapped.action_space == gym.spaces.Discrete(5)
        assert [float(wrapped.action(i)[0]) for i in range(5)] == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_cartesian_product_over_dimensions(self):
        class TwoDimensionalEnv(gym.Env):
            action_space = gym.spaces.Box(low=np.array([0.0, -2.0]), high=np.array([1.0, 2.0]), dtype=np.float64)
            observation_space = gym.spaces.Discrete(1)

        wrapped = DiscretizeActionWrapper(TwoDimensionalEnv(), [2, 3])

        assert wrapped.action_space == gym.spaces.Discrete(6)
        # Last dimension varies fastest
        assert [wrapped.action(i).tolist() for i in range(6)] == [
            [0.0, -2.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, -2.0],
            [1.0, 0.0],
            [1.0, 2.0],
        ]

    def test_rejects_discrete_action_space(self):
        with pytest.raises(ValueError, match="Box action space"):
            DiscretizeActionWrapper(gym.make("FrozenLake-v1"), 3)

    def test_rejects_bins_not_matching_dimensions(self):
        with pytest.raises(ValueError, match="entries"):
            DiscretizeActionWrapper(gym.make("MountainCarContinuous-v0"), [3, 3])


class TestCustomEnvironmentRegistration:
    def test_register_and_reject_conflicting_entry_point(self):
        register_custom_environment(ENV_ID, "line_env:LineEnv", [FIXTURES_DIR / "custom_env"])
        register_custom_environment(ENV_ID, "line_env:LineEnv", [FIXTURES_DIR / "custom_env"])  # idempotent

        with pytest.raises(ValueError, match="already registered"):
            register_custom_environment(ENV_ID, "other_module:LineEnv")

    def test_save_and_load_round_trip_keeps_custom_entry_point_and_wrapper(self, temp_test_dir):
        environment_config = load_config_from_yaml(CONFIG_PATH).get_environment_configs()[2]
        env = EnvironmentFactory().get_or_create_from_config(environment_config)
        path = temp_test_dir / "environment.json"

        assert save_environment(env, path, environment_config)
        loaded = load_environment(path)

        assert loaded.action_space == gym.spaces.Discrete(3)
        assert loaded.spec is not None and loaded.spec.max_episode_steps == 20
        assert isinstance(loaded.unwrapped.action_space, gym.spaces.Box)


class TestCustomEnvironmentPipeline:
    def test_learn_test_and_play_on_custom_environment(self, temp_test_dir):
        config = load_config_from_yaml(CONFIG_PATH)
        config.base_output_dir = temp_test_dir
        supervisor = Supervisor(config)
        supervisor.execute_learn_phase()
        supervisor.execute_test_phase()

        run_dirs = sorted(path.parent for path in temp_test_dir.rglob("run_info.json"))
        assert len(run_dirs) == 3

        discretized = next(d for d in run_dirs if "act_dis_3" in str(d))
        token = CancellationToken()
        token.cancel()  # play_interactive loops until cancelled: stop at the first episode boundary
        result = play_interactive(discretized / "model.json", discretized / "environment.json", token, "human")
        assert result.total_episodes == 0
        assert np.isfinite(result.average_reward)
