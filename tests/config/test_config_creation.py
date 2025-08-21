"""Tests for HerculeConfig creation and display functionality."""

from pathlib import Path

from hercule.config import HerculeConfig, HyperParameter, ModelConfig, RunConfig


class TestHerculeConfigCreation:
    """Test cases for HerculeConfig creation."""

    def test_create_default_config(self, temp_test_dir):
        """Test creating a default configuration."""
        config = HerculeConfig()

        assert config.name == "hercule_run"
        assert config.environments == ["CartPole-v1"]
        assert config.models == []
        assert config.max_iterations == 1000
        assert str(config.output_dir) == "outputs"
        assert config.evaluation is None

    def test_create_custom_config(self, temp_test_dir):
        """Test creating a configuration with custom values."""
        config = HerculeConfig(
            name="test_config", environments=["LunarLander-v2"], max_iterations=500, output_dir=Path("custom_outputs")
        )

        assert config.name == "test_config"
        assert config.environments == ["LunarLander-v2"]
        assert config.max_iterations == 500
        assert str(config.output_dir) == "custom_outputs"

    def test_create_config_with_models(self, temp_test_dir):
        """Test creating a configuration with models."""
        model_config = ModelConfig(
            name="test_model",
            hyperparameters=[HyperParameter(key="learning_rate", value=0.01), HyperParameter(key="epsilon", value=0.1)],
        )

        config = HerculeConfig(name="test_with_models", models=[model_config])

        assert len(config.models) == 1
        assert config.models[0].name == "test_model"
        assert len(config.models[0].hyperparameters) == 2

    def test_create_config_with_evaluation(self, temp_test_dir):
        """Test creating a configuration with evaluation settings."""
        evaluation_config = RunConfig(num_episodes=20, render=True)

        config = HerculeConfig(name="test_with_evaluation", evaluation=evaluation_config)

        assert config.evaluation is not None
        assert config.evaluation.num_episodes == 20
        assert config.evaluation.render is True


class TestHerculeConfigDisplay:
    """Test cases for HerculeConfig string representation."""

    def test_str_default_config(self, temp_test_dir):
        """Test string representation of default configuration."""
        config = HerculeConfig()
        config_str = str(config)

        # Should be valid JSON with indentation
        assert '"name": "hercule_run"' in config_str
        assert '"CartPole-v1"' in config_str
        assert '"max_iterations": 1000' in config_str
        assert '"output_dir": "outputs"' in config_str
        assert '"evaluation": null' in config_str

        # Should have proper indentation (2 spaces)
        lines = config_str.split("\n")
        assert any(line.startswith('  "') for line in lines)

    def test_str_custom_config(self, temp_test_dir):
        """Test string representation of custom configuration."""
        config = HerculeConfig(name="custom_test", environments=["LunarLander-v2"], max_iterations=500)
        config_str = str(config)

        assert '"name": "custom_test"' in config_str
        assert '"LunarLander-v2"' in config_str
        assert '"max_iterations": 500' in config_str

    def test_str_config_with_models(self, temp_test_dir):
        """Test string representation with models."""
        model_config = ModelConfig(name="test_model", hyperparameters=[HyperParameter(key="learning_rate", value=0.01)])

        config = HerculeConfig(name="test_with_models", models=[model_config])
        config_str = str(config)

        assert '"models": [' in config_str
        assert '"name": "test_model"' in config_str
        assert '"key": "learning_rate"' in config_str
        assert '"value": 0.01' in config_str

    def test_str_config_with_evaluation(self, temp_test_dir):
        """Test string representation with evaluation."""
        evaluation_config = RunConfig(num_episodes=15, render=True)

        config = HerculeConfig(name="test_with_evaluation", evaluation=evaluation_config)
        config_str = str(config)

        assert '"evaluation": {' in config_str
        assert '"num_episodes": 15' in config_str
        assert '"render": true' in config_str

    def test_repr_config(self, temp_test_dir):
        """Test repr representation of configuration."""
        config = HerculeConfig(name="test_repr")
        config_repr = repr(config)

        assert config_repr.startswith("HerculeConfig(")
        assert config_repr.endswith(")")
        assert '"name": "test_repr"' in config_repr


class TestHerculeConfigRecreation:
    """Test cases for recreating config from its representation."""

    def test_recreate_from_str(self, temp_test_dir):
        """Test that a config can be recreated from its string representation."""
        original_config = HerculeConfig(
            name="recreation_test", environments=["CartPole-v1", "LunarLander-v2"], max_iterations=750
        )

        config_str = str(original_config)

        # The string should be valid JSON that can be parsed
        import json

        config_dict = json.loads(config_str)

        # Recreate the config from the parsed dictionary
        recreated_config = HerculeConfig(**config_dict)

        assert recreated_config.name == original_config.name
        assert recreated_config.environments == original_config.environments
        assert recreated_config.max_iterations == original_config.max_iterations
        assert str(recreated_config.output_dir) == str(original_config.output_dir)

    def test_recreate_from_repr_simple(self, temp_test_dir):
        """Test that a config can be recreated from its repr representation using eval()."""
        original_config = HerculeConfig(name="repr_test", environments=["CartPole-v1"], max_iterations=500)

        # Get the repr string
        repr_str = repr(original_config)

        # Recreate the config by evaluating the repr string
        recreated_config = eval(repr_str)

        # Test that the recreated config is identical
        assert recreated_config.name == original_config.name
        assert recreated_config.environments == original_config.environments
        assert recreated_config.max_iterations == original_config.max_iterations
        assert str(recreated_config.output_dir) == str(original_config.output_dir)

    def test_recreate_from_repr_complex(self, temp_test_dir):
        """Test recreating a complex configuration from repr using eval()."""
        model_config = ModelConfig(
            name="complex_model",
            hyperparameters=[
                HyperParameter(key="learning_rate", value=0.001),
                HyperParameter(key="batch_size", value=32),
            ],
        )

        evaluation_config = RunConfig(num_episodes=25, render=False)

        original_config = HerculeConfig(
            name="complex_repr_test",
            environments=["CartPole-v1"],
            models=[model_config],
            max_iterations=2000,
            evaluation=evaluation_config,
        )

        # Get the repr string
        repr_str = repr(original_config)

        # Recreate the config by evaluating the repr string
        recreated_config = eval(repr_str)

        # Test basic properties
        assert recreated_config.name == original_config.name
        assert recreated_config.max_iterations == original_config.max_iterations

        # Test models
        assert len(recreated_config.models) == 1
        assert recreated_config.models[0].name == "complex_model"
        assert len(recreated_config.models[0].hyperparameters) == 2

        # Test evaluation
        assert recreated_config.evaluation is not None
        assert recreated_config.evaluation.num_episodes == 25
        assert recreated_config.evaluation.render is False

    def test_recreate_complex_config(self, temp_test_dir):
        """Test recreating a complex configuration with models and evaluation."""
        model_config = ModelConfig(
            name="complex_model",
            hyperparameters=[
                HyperParameter(key="learning_rate", value=0.001),
                HyperParameter(key="batch_size", value=32),
            ],
        )

        evaluation_config = RunConfig(num_episodes=25, render=False)

        original_config = HerculeConfig(
            name="complex_test",
            environments=["CartPole-v1"],
            models=[model_config],
            max_iterations=2000,
            evaluation=evaluation_config,
        )

        config_str = str(original_config)
        import json

        config_dict = json.loads(config_str)

        recreated_config = HerculeConfig(**config_dict)

        # Test basic properties
        assert recreated_config.name == original_config.name
        assert recreated_config.max_iterations == original_config.max_iterations

        # Test models
        assert len(recreated_config.models) == 1
        assert recreated_config.models[0].name == "complex_model"
        assert len(recreated_config.models[0].hyperparameters) == 2

        # Test evaluation
        assert recreated_config.evaluation is not None
        assert recreated_config.evaluation.num_episodes == 25
        assert recreated_config.evaluation.render is False
