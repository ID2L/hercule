"""Tests for YAML configuration loading functionality."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hercule.config import load_config_from_yaml


class TestHerculeConfigYAML:
    """Test cases for YAML configuration loading."""

    def test_load_simple_yaml_config(self, temp_test_dir):
        """Test loading a simple YAML configuration."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        assert config.name == "simple_test"
        assert config.environments == ["CartPole-v1"]
        assert len(config.models) == 1
        assert config.models[0].name == "test_model"
        assert config.learn_max_epoch == 100

    def test_load_yaml_config_with_models(self, temp_test_dir):
        """Test loading a YAML configuration with models and hyperparameters."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "models_config.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        assert config.name == "simple_test"
        assert config.environments == ["CartPole-v1"]
        assert len(config.models) == 1
        assert config.models[0].name == "test_model"
        assert len(config.models[0].hyperparameters) == 2
        assert config.models[0].hyperparameters[0].key == "learning_rate"
        assert config.models[0].hyperparameters[0].value == pytest.approx(0.01)

    def test_load_yaml_config_with_evaluation(self, temp_test_dir):
        """Test loading a YAML configuration with evaluation settings."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "complex_config.yaml"
        config_file = temp_test_dir / "evaluation_config.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        assert config.name == "complex_test"
        assert config.evaluation is not None
        assert config.evaluation.num_episodes == 25
        assert config.evaluation.render is True

    def test_load_complex_yaml_config(self, temp_test_dir):
        """Test loading a complex YAML configuration with all features."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "multi_config.yaml"
        config_file = temp_test_dir / "complex_config.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        # Test basic properties
        assert config.name == "multi_test"
        assert len(config.environments) == 2
        assert config.environments[0] == "CartPole-v1"
        assert config.environments[1].name == "LunarLander-v2"
        assert config.learn_max_epoch == 1000
        assert str(config.base_output_dir) == "multi_outputs"

        # Test models
        # advanced_model has 3 lists with 2 values each, so 2*2*2=8 variants
        # plus 1 simple_model = 9 total models
        assert len(config.models) == 9
        assert config.models[0].name == "simple_model"
        # All variants should be expanded (no lists in hyperparameters)
        for model in config.models:
            assert all(not isinstance(hp.value, list) for hp in model.hyperparameters)
        assert len(config.models[0].hyperparameters) == 2
        # All advanced_model variants should have 6 hyperparameters
        advanced_models = [m for m in config.models if m.name == "advanced_model"]
        assert len(advanced_models) == 8
        assert all(len(m.hyperparameters) == 6 for m in advanced_models)

        # Test evaluation
        assert config.evaluation is not None
        assert config.evaluation.num_episodes == 50
        assert config.evaluation.render is False

    def test_load_yaml_config_file_not_found(self, temp_test_dir):
        """Test that loading a non-existent YAML file raises FileNotFoundError."""
        non_existent_file = temp_test_dir / "non_existent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config_from_yaml(non_existent_file)

    def test_load_yaml_config_invalid_yaml(self, temp_test_dir):
        """Test that loading invalid YAML raises yaml.YAMLError."""
        invalid_yaml_content = """
name: invalid_config
environments:
  - CartPole-v1
max_iterations: 1000
invalid: [unclosed: bracket
"""
        config_file = temp_test_dir / "invalid_config.yaml"
        config_file.write_text(invalid_yaml_content)

        with pytest.raises(yaml.YAMLError):
            load_config_from_yaml(config_file)

    def test_load_yaml_config_validation_error(self, temp_test_dir):
        """Test that loading YAML with invalid data raises ValidationError."""
        invalid_config_content = """
name: invalid_config
environments: []  # Empty environments list should fail validation
learn_max_epoch: 1000
"""
        config_file = temp_test_dir / "validation_error_config.yaml"
        config_file.write_text(invalid_config_content)

        with pytest.raises(ValidationError):
            load_config_from_yaml(config_file)

    def test_load_yaml_config_with_string_path(self, temp_test_dir):
        """Test that load_config_from_yaml accepts string paths."""
        yaml_content = """
name: string_path_test
environments:
  - CartPole-v1
learn_max_epoch: 500
"""
        config_file = temp_test_dir / "string_path_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(str(config_file))

        assert config.name == "string_path_test"
        assert config.environments == ["CartPole-v1"]
        assert config.learn_max_epoch == 500

    def test_load_yaml_config_default_values(self, temp_test_dir):
        """Test that YAML config uses default values when not specified."""
        yaml_content = """
name: default_test
environments:
  - CartPole-v1
"""
        config_file = temp_test_dir / "default_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        # Should use default values
        assert config.learn_max_epoch == 1000
        assert str(config.base_output_dir) == "outputs"
        assert config.evaluation is None
        assert config.models == []

    def test_load_yaml_config_with_environment_config(self, temp_test_dir):
        """Test loading YAML with environment configuration objects."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "multi_config.yaml"
        config_file = temp_test_dir / "env_config.yaml"
        config_file.write_text(fixture_path.read_text())

        config = load_config_from_yaml(config_file)

        assert len(config.environments) == 2
        # First environment should be a string
        assert config.environments[0] == "CartPole-v1"
        # Second environment should be an EnvironmentConfig object
        from hercule.config import EnvironmentConfig

        assert isinstance(config.environments[1], EnvironmentConfig)
        assert config.environments[1].name == "LunarLander-v2"
        assert len(config.environments[1].hyperparameters) == 2
