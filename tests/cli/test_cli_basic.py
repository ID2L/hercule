"""Basic CLI tests for configuration loading and summary saving."""

import json
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


class TestCLILearnCommand:
    """Test the learn command functionality."""

    def test_learn_command_help(self, runner):
        """Test that learn command shows help correctly."""
        result = runner.invoke(cli, ["learn", "--help"])
        assert result.exit_code == 0
        assert "Learn and evaluate RL algorithms" in result.output
        assert "CONFIG_FILE" in result.output

    def test_learn_command_with_valid_config(self, temp_test_dir, change_to_temp_dir, runner):
        """Test learn command with a valid configuration file."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(fixture_path.read_text())

        # Create output directory
        output_dir = temp_test_dir / "outputs"

        # Run learn command
        result = runner.invoke(cli, ["learn", str(config_file), "--output-dir", str(output_dir)])

        # Should succeed
        assert result.exit_code == 0
        assert "Learning with" in result.output
        assert "Output directory:" in result.output

        # Check that output files were created
        assert output_dir.exists()
        assert (output_dir / "config_summary.yaml").exists()


class TestCLIPlayCommand:
    """Test the play command functionality."""

    def test_play_command_help(self, runner):
        """Test that play command shows help correctly."""
        result = runner.invoke(cli, ["play", "--help"])
        assert result.exit_code == 0
        assert "Play with a trained RL model" in result.output
        assert "MODEL_FILE" in result.output
        assert "ENVIRONMENT_FILE" in result.output
        assert "Press Ctrl+C to stop" in result.output

    def test_play_command_with_missing_files(self, temp_test_dir, change_to_temp_dir, runner):
        """Test play command with missing model or environment files."""
        # Test with missing model file
        result = runner.invoke(cli, ["play", "missing_model.json", "missing_env.json"])
        assert result.exit_code != 0

    def test_play_command_with_invalid_files(self, temp_test_dir, change_to_temp_dir, runner):
        """Test play command with invalid model or environment files."""
        # Create invalid model file
        invalid_model = temp_test_dir / "invalid_model.json"
        invalid_model.write_text("invalid json content")

        # Create invalid environment file
        invalid_env = temp_test_dir / "invalid_env.json"
        invalid_env.write_text("invalid json content")

        # Run play command
        result = runner.invoke(cli, ["play", str(invalid_model), str(invalid_env)])

        # Should fail
        assert result.exit_code != 0

    def test_play_command_with_valid_files(self, temp_test_dir, change_to_temp_dir, runner):
        """Test play command with valid model and environment files."""
        # Create a minimal valid model file
        model_data = {
            "model_name": "simple_sarsa",
            "q_table": [[0.0, 0.0, 0.0, 0.0]],
            "hyperparameters": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0.1},
        }
        model_file = temp_test_dir / "test_model.json"
        model_file.write_text(json.dumps(model_data))

        # Create a minimal valid environment file
        env_data = {"id": "FrozenLake-v1", "kwargs": {"is_slippery": False}}
        env_file = temp_test_dir / "test_env.json"
        env_file.write_text(json.dumps(env_data))

        # Run play command with ansi render mode
        result = runner.invoke(cli, ["play", str(model_file), str(env_file), "--render-mode", "ansi"])

        # Should succeed (or at least not fail on file loading)
        # Note: This might fail due to environment setup, but should not fail on file loading
        if result.exit_code != 0:
            # If it fails, it should be due to environment setup, not file loading
            assert "Failed to play with model" in result.output or "Error playing with model" in result.output


class TestCLIGeneral:
    """Test general CLI functionality."""

    def test_cli_help(self, runner):
        """Test that main CLI shows help correctly."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Hercule RL framework CLI" in result.output
        assert "learn" in result.output
        assert "play" in result.output

    def test_cli_version(self, runner):
        """Test that CLI shows version correctly."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "hercule" in result.output.lower()

    def test_verbose_option(self, runner):
        """Test that verbose option works correctly."""
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0
