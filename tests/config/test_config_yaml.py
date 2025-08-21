"""Tests for YAML configuration loading functionality."""

import pytest
import yaml
from pydantic import ValidationError

from hercule.config import load_config_from_yaml


class TestHerculeConfigYAML:
    """Test cases for YAML configuration loading."""

    def test_load_simple_yaml_config(self, temp_test_dir):
        """Test loading a simple YAML configuration."""
        yaml_content = """
name: test_yaml_config
environments:
  - CartPole-v1
  - LunarLander-v2
max_iterations: 1500
output_dir: yaml_outputs
"""
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        assert config.name == "test_yaml_config"
        assert config.environments == ["CartPole-v1", "LunarLander-v2"]
        assert config.max_iterations == 1500
        assert str(config.output_dir) == "yaml_outputs"

    def test_load_yaml_config_with_models(self, temp_test_dir):
        """Test loading a YAML configuration with models and hyperparameters."""
        yaml_content = """
name: yaml_with_models
environments:
  - CartPole-v1
models:
  - name: test_model
    hyperparameters:
      - key: learning_rate
        value: 0.01
      - key: epsilon
        value: 0.1
max_iterations: 2000
"""
        config_file = temp_test_dir / "models_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        assert config.name == "yaml_with_models"
        assert config.environments == ["CartPole-v1"]
        assert len(config.models) == 1
        assert config.models[0].name == "test_model"
        assert len(config.models[0].hyperparameters) == 2
        assert config.models[0].hyperparameters[0].key == "learning_rate"
        assert config.models[0].hyperparameters[0].value == pytest.approx(0.01)

    def test_load_yaml_config_with_evaluation(self, temp_test_dir):
        """Test loading a YAML configuration with evaluation settings."""
        yaml_content = """
name: yaml_with_evaluation
environments:
  - CartPole-v1
max_iterations: 1000
evaluation:
  num_episodes: 25
  render: true
"""
        config_file = temp_test_dir / "evaluation_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        assert config.name == "yaml_with_evaluation"
        assert config.evaluation is not None
        assert config.evaluation.num_episodes == 25
        assert config.evaluation.render is True

    def test_load_complex_yaml_config(self, temp_test_dir):
        """Test loading a complex YAML configuration with all features."""
        yaml_content = """
name: complex_yaml_config
environments:
  - CartPole-v1
  - LunarLander-v2
models:
  - name: qlearning_model
    hyperparameters:
      - key: learning_rate
        value: 0.001
      - key: discount_factor
        value: 0.95
  - name: sarsa_model
    hyperparameters:
      - key: learning_rate
        value: 0.01
      - key: epsilon
        value: 0.1
max_iterations: 3000
output_dir: complex_outputs
evaluation:
  num_episodes: 50
  render: false
"""
        config_file = temp_test_dir / "complex_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        # Test basic properties
        assert config.name == "complex_yaml_config"
        assert config.environments == ["CartPole-v1", "LunarLander-v2"]
        assert config.max_iterations == 3000
        assert str(config.output_dir) == "complex_outputs"

        # Test models
        assert len(config.models) == 2
        assert config.models[0].name == "qlearning_model"
        assert config.models[1].name == "sarsa_model"
        assert len(config.models[0].hyperparameters) == 2
        assert len(config.models[1].hyperparameters) == 2

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
max_iterations: 1000
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
max_iterations: 500
"""
        config_file = temp_test_dir / "string_path_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(str(config_file))

        assert config.name == "string_path_test"
        assert config.environments == ["CartPole-v1"]
        assert config.max_iterations == 500

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
        assert config.max_iterations == 1000
        assert str(config.output_dir) == "outputs"
        assert config.evaluation is None
        assert config.models == []

    def test_load_yaml_config_with_environment_config(self, temp_test_dir):
        """Test loading YAML with environment configuration objects."""
        yaml_content = """
name: env_config_test
environments:
  - name: CartPole-v1
    hyperparameters:
      - key: max_steps
        value: 500
  - LunarLander-v2
max_iterations: 1000
"""
        config_file = temp_test_dir / "env_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config_from_yaml(config_file)

        assert len(config.environments) == 2
        # First environment should be an EnvironmentConfig object
        from hercule.config import EnvironmentConfig

        assert isinstance(config.environments[0], EnvironmentConfig)
        assert config.environments[0].name == "CartPole-v1"
        assert len(config.environments[0].hyperparameters) == 1
        # Second environment should be a string
        assert config.environments[1] == "LunarLander-v2"
