"""Basic CLI tests for configuration loading and summary saving."""

from pathlib import Path

import click
import pytest

from hercule.cli.main import cli
from hercule.config import load_config_from_yaml


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_loads_simple_config(self, temp_test_dir, change_to_temp_dir):
        """Test that CLI can load a simple configuration file."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(fixture_path.read_text())

        # Test loading the config directly
        config = load_config_from_yaml(config_file)

        assert config.name == "simple_test"
        assert config.environments == ["CartPole-v1"]
        assert len(config.models) == 1
        assert config.models[0].name == "test_model"
        assert config.learn_max_epoch == 100

    def test_cli_loads_complex_config(self, temp_test_dir, change_to_temp_dir):
        """Test that CLI can load a complex configuration file."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "complex_config.yaml"
        config_file = temp_test_dir / "complex_config.yaml"
        config_file.write_text(fixture_path.read_text())

        # Test loading the config directly
        config = load_config_from_yaml(config_file)

        assert config.name == "complex_test"
        assert len(config.environments) == 1
        assert config.environments[0].name == "FrozenLake-v1"
        assert len(config.environments[0].hyperparameters) == 3
        assert len(config.models) == 1
        assert config.models[0].name == "complex_model"
        assert config.learn_max_epoch == 500
        assert config.evaluation is not None
        assert config.evaluation.num_episodes == 25

    def test_cli_loads_multi_config(self, temp_test_dir, change_to_temp_dir):
        """Test that CLI can load a multi-model, multi-environment configuration."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "multi_config.yaml"
        config_file = temp_test_dir / "multi_config.yaml"
        config_file.write_text(fixture_path.read_text())

        # Test loading the config directly
        config = load_config_from_yaml(config_file)

        assert config.name == "multi_test"
        assert len(config.environments) == 2
        assert config.environments[0] == "CartPole-v1"
        assert config.environments[1].name == "LunarLander-v2"
        assert len(config.models) == 2
        assert config.models[0].name == "simple_model"
        assert config.models[1].name == "advanced_model"
        assert config.learn_max_epoch == 1000

    def test_cli_handles_missing_config_file(self, temp_test_dir, change_to_temp_dir):
        """Test that CLI handles missing configuration file gracefully."""
        non_existent_file = temp_test_dir / "non_existent.yaml"

        # Run CLI with non-existent file - should raise BadParameter
        with pytest.raises(click.exceptions.BadParameter):
            cli.main(["learn", str(non_existent_file)], standalone_mode=False)

    def test_cli_handles_invalid_config_file(self, temp_test_dir, change_to_temp_dir):
        """Test that CLI handles invalid configuration file gracefully."""
        # Create invalid YAML file
        invalid_config = temp_test_dir / "invalid_config.yaml"
        invalid_config.write_text("""
name: invalid_config
environments:
  - CartPole-v1
invalid: [unclosed: bracket
""")

        # Run CLI with invalid file
        result = cli.main(["learn", str(invalid_config)], standalone_mode=False)

        # Should return non-zero exit code
        assert result != 0
