"""Tests for environment save and load functions."""

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import pytest

from hercule.environnements import load_environment, save_environment


class TestSaveLoadEnvironment:
    """Test cases for save_environment and load_environment functions."""

    def test_save_environment_success(self):
        """Test successful environment save."""
        # Create a simple environment
        env = gym.make("CartPole-v1")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Save environment
            result = save_environment(env, tmp_path)

            # Check result
            assert result is True

            # Check file exists and contains valid JSON
            assert tmp_path.exists()

            with open(tmp_path) as f:
                saved_data = json.load(f)

            # Check that required keys are present
            assert "id" in saved_data
            assert saved_data["id"] == "CartPole-v1"

            # Check that kwargs is present (even if empty)
            assert "kwargs" in saved_data

        finally:
            # Cleanup
            env.close()
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_environment_success(self):
        """Test successful environment load."""
        # Create a simple environment and save it
        original_env = gym.make("CartPole-v1")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Save environment
            save_result = save_environment(original_env, tmp_path)
            assert save_result is True

            # Load environment
            loaded_env = load_environment(tmp_path)

            # Check that loaded environment is valid
            assert loaded_env is not None
            assert isinstance(loaded_env, gym.Env)

            # Check that it's the same type of environment
            assert loaded_env.spec is not None
            assert loaded_env.spec.id == "CartPole-v1"

            # Check that spaces are the same
            assert loaded_env.observation_space == original_env.observation_space
            assert loaded_env.action_space == original_env.action_space

        finally:
            # Cleanup
            original_env.close()
            if "loaded_env" in locals():
                loaded_env.close()
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_environment_with_kwargs(self):
        """Test loading environment with custom kwargs."""
        # Create environment with custom parameters
        original_env = gym.make("CartPole-v1", render_mode="rgb_array")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Save environment
            save_result = save_environment(original_env, tmp_path)
            assert save_result is True

            # Load environment
            loaded_env = load_environment(tmp_path)

            # Check that loaded environment has the same kwargs
            assert loaded_env.spec is not None
            assert loaded_env.spec.id == "CartPole-v1"
            assert loaded_env.spec.kwargs.get("render_mode") == "rgb_array"

        finally:
            # Cleanup
            original_env.close()
            if "loaded_env" in locals():
                loaded_env.close()
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_environment_invalid_file(self):
        """Test loading environment from invalid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Write invalid JSON
            with open(tmp_path, "w") as f:
                f.write("invalid json content")

            # Should raise exception
            with pytest.raises(json.JSONDecodeError):
                load_environment(tmp_path)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
