"""Tests for hyperparameter list expansion functionality."""

from pathlib import Path

import pytest

from hercule.config import EnvironmentConfig, HyperParameter, ModelConfig, load_config_from_yaml


class TestHyperparameterExpansion:
    """Test cases for hyperparameter list expansion."""

    def test_expand_model_config_with_list(self):
        """Test that a model config with a list expands into multiple variants."""
        model = ModelConfig(
            name="test_model",
            hyperparameters=[
                HyperParameter(key="learning_rate", value=0.2),
                HyperParameter(key="epsilon", value=[0.1, 0.2, 0.3]),
            ],
        )

        variants = model.expand_variants()

        assert len(variants) == 3
        assert all(v.name == "test_model" for v in variants)
        assert all(len(v.hyperparameters) == 2 for v in variants)

        # Check first variant
        assert variants[0].get_hyperparameters_dict()["learning_rate"] == 0.2
        assert variants[0].get_hyperparameters_dict()["epsilon"] == 0.1

        # Check second variant
        assert variants[1].get_hyperparameters_dict()["learning_rate"] == 0.2
        assert variants[1].get_hyperparameters_dict()["epsilon"] == 0.2

        # Check third variant
        assert variants[2].get_hyperparameters_dict()["learning_rate"] == 0.2
        assert variants[2].get_hyperparameters_dict()["epsilon"] == 0.3

    def test_expand_model_config_with_multiple_lists(self):
        """Test cartesian product when multiple hyperparameters have lists."""
        model = ModelConfig(
            name="test_model",
            hyperparameters=[
                HyperParameter(key="learning_rate", value=[0.1, 0.2]),
                HyperParameter(key="epsilon", value=[0.5, 0.7]),
            ],
        )

        variants = model.expand_variants()

        assert len(variants) == 4  # 2 * 2 = 4 combinations

        # Check all combinations exist
        combinations = {(v.get_hyperparameters_dict()["learning_rate"], v.get_hyperparameters_dict()["epsilon"]) for v in variants}
        expected_combinations = {(0.1, 0.5), (0.1, 0.7), (0.2, 0.5), (0.2, 0.7)}
        assert combinations == expected_combinations

    def test_expand_model_config_without_lists(self):
        """Test that a model config without lists returns single variant."""
        model = ModelConfig(
            name="test_model",
            hyperparameters=[
                HyperParameter(key="learning_rate", value=0.2),
                HyperParameter(key="epsilon", value=0.1),
            ],
        )

        variants = model.expand_variants()

        assert len(variants) == 1
        assert variants[0].name == "test_model"
        assert variants[0].get_hyperparameters_dict()["learning_rate"] == 0.2
        assert variants[0].get_hyperparameters_dict()["epsilon"] == 0.1

    def test_expand_environment_config_with_list(self):
        """Test that an environment config with a list expands into multiple variants."""
        env = EnvironmentConfig(
            name="test_env",
            hyperparameters=[
                HyperParameter(key="param1", value="fixed"),
                HyperParameter(key="param2", value=[1, 2, 3]),
            ],
        )

        variants = env.expand_variants()

        assert len(variants) == 3
        assert all(v.name == "test_env" for v in variants)

        # Check values
        assert variants[0].get_hyperparameters_dict()["param2"] == 1
        assert variants[1].get_hyperparameters_dict()["param2"] == 2
        assert variants[2].get_hyperparameters_dict()["param2"] == 3

    def test_load_yaml_config_with_list_expansion(self, temp_test_dir):
        """Test loading a YAML config with list values gets automatically expanded."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "config_with_lists.yaml"
        config_file = temp_test_dir / "config_with_lists.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        # Should have 3 model variants (one for each epsilon value)
        assert len(config.models) == 3

        # All should be simple_sarsa
        assert all(m.name == "simple_sarsa" for m in config.models)

        # Check epsilon values
        epsilon_values = [m.get_hyperparameters_dict()["epsilon"] for m in config.models]
        assert set(epsilon_values) == {0.1, 0.2, 0.3}

        # All should have learning_rate=0.2 and discount_factor=0.8
        for model in config.models:
            assert model.get_hyperparameters_dict()["learning_rate"] == 0.2
            assert model.get_hyperparameters_dict()["discount_factor"] == 0.8

    def test_expand_hercule_config_with_multiple_expansions(self):
        """Test that HerculeConfig expands both models and environments."""
        from hercule.config import HerculeConfig

        model1 = ModelConfig(
            name="model1",
            hyperparameters=[HyperParameter(key="param", value=[1, 2])],
        )
        model2 = ModelConfig(
            name="model2",
            hyperparameters=[HyperParameter(key="param", value=10)],
        )
        env1 = EnvironmentConfig(
            name="env1",
            hyperparameters=[HyperParameter(key="env_param", value=["a", "b"])],
        )

        config = HerculeConfig(
            name="test",
            models=[model1, model2],
            environments=[env1],
        )

        expanded = config.expand_variants()

        # model1 expands to 2, model2 stays 1, so 3 models total
        assert len(expanded.models) == 3
        # env1 expands to 2
        assert len(expanded.environments) == 2

        # Check model names
        model_names = [m.name for m in expanded.models]
        assert model_names.count("model1") == 2
        assert model_names.count("model2") == 1


