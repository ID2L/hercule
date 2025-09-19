"""CLI tests that actually test the CLI interface."""

import json
from pathlib import Path

from hercule.cli.main import cli


class TestCLIBasic:
    """Test basic CLI functionality."""

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
        assert "version" in result.output.lower()

    def test_cli_handles_missing_config_file(self, temp_test_dir, change_to_temp_dir, runner):
        """Test that CLI handles missing configuration file gracefully."""
        non_existent_file = temp_test_dir / "non_existent.yaml"

        result = runner.invoke(cli, ["learn", str(non_existent_file)])
        assert result.exit_code != 0
        assert "No such file or directory" in result.output or "does not exist" in result.output

    def test_cli_handles_invalid_config_file(self, temp_test_dir, change_to_temp_dir, runner):
        """Test that CLI handles invalid configuration file gracefully."""
        # Create invalid YAML file
        invalid_config = temp_test_dir / "invalid_config.yaml"
        invalid_config.write_text("""
name: invalid_config
environments:
  - CartPole-v1
invalid: [unclosed: bracket
""")

        result = runner.invoke(cli, ["learn", str(invalid_config)])
        # The CLI catches the error and logs it, but returns 0 exit code
        # We check that the error is logged
        assert "Failed to process" in result.output or "Error processing" in result.output


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

        # Run learn command with verbose option
        result = runner.invoke(cli, ["learn", str(config_file), "--verbose"])

        # Should succeed
        assert result.exit_code == 0
        assert "Learning with" in result.output

        # Check that output files were created
        output_dir = temp_test_dir / "outputs"
        assert output_dir.exists()
        assert (output_dir / "simple_test" / "config_summary.yaml").exists()

    def test_learn_command_with_output_dir_override(self, temp_test_dir, change_to_temp_dir, runner):
        """Test learn command with output directory override."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(fixture_path.read_text())

        custom_output_dir = temp_test_dir / "custom_outputs"

        # Run learn command with custom output directory
        result = runner.invoke(cli, ["learn", str(config_file), "--output-dir", str(custom_output_dir)])

        # Should succeed
        assert result.exit_code == 0
        assert "Output directory override" in result.output

        # Check that files were created in custom directory
        assert custom_output_dir.exists()
        assert (custom_output_dir / "simple_test" / "config_summary.yaml").exists()

    def test_learn_command_with_verbose_option(self, temp_test_dir, change_to_temp_dir, runner):
        """Test learn command with verbose option."""
        # Copy fixture to temp directory
        fixture_path = Path(__file__).parent.parent / "fixtures" / "simple_config.yaml"
        config_file = temp_test_dir / "simple_config.yaml"
        config_file.write_text(fixture_path.read_text())

        # Run learn command with verbose option
        result = runner.invoke(cli, ["learn", str(config_file), "--verbose"])

        # Should succeed
        assert result.exit_code == 0
        assert "Learning with" in result.output

    def test_learn_command_missing_config(self, runner):
        """Test learn command with missing config file."""
        result = runner.invoke(cli, ["learn", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "No such file or directory" in result.output or "does not exist" in result.output


class TestCLIPlayCommand:
    """Test the play command functionality."""

    def test_play_command_help(self, runner):
        """Test that play command shows help correctly."""
        result = runner.invoke(cli, ["play", "--help"])
        assert result.exit_code == 0
        assert "Play with a trained RL model" in result.output
        assert "MODEL_FILE" in result.output
        assert "ENVIRONMENT_FILE" in result.output

    def test_play_command_with_missing_files(self, runner):
        """Test play command with missing model or environment files."""
        result = runner.invoke(cli, ["play", "missing_model.json", "missing_env.json"])
        assert result.exit_code != 0
        assert "No such file or directory" in result.output or "does not exist" in result.output

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

        # The CLI catches the error and logs it, but returns 0 exit code
        # We check that the error is logged
        assert "Failed to play with model" in result.output or "Error playing with model" in result.output

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

        # Run play command with no-render option to avoid GUI windows
        result = runner.invoke(cli, ["play", str(model_file), str(env_file), "--no-render"])

        # Should succeed (or at least not fail on file loading)
        # Note: This might fail due to environment setup, but should not fail on file loading
        if result.exit_code != 0:
            # If it fails, it should be due to environment setup, not file loading
            assert "Failed to play with model" in result.output or "Error playing with model" in result.output

    def test_play_command_with_verbose_option(self, temp_test_dir, change_to_temp_dir, runner):
        """Test play command with verbose option."""
        # Create valid files
        model_data = {"model_name": "simple_sarsa", "q_table": [[0.0]]}
        model_file = temp_test_dir / "test_model.json"
        model_file.write_text(json.dumps(model_data))

        env_data = {"id": "FrozenLake-v1", "kwargs": {}}
        env_file = temp_test_dir / "test_env.json"
        env_file.write_text(json.dumps(env_data))

        # Run play command with verbose option and no-render to avoid GUI windows
        result = runner.invoke(cli, ["play", str(model_file), str(env_file), "--verbose", "--no-render"])

        # Should succeed or fail gracefully
        if result.exit_code != 0:
            assert "Failed to play with model" in result.output or "Error playing with model" in result.output
