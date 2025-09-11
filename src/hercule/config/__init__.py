"""Configuration module for Hercule reinforcement learning framework."""

import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# Type alias for hyperparameter values
ParameterValue = str | int | float | bool | None | list[str] | list[int] | list[float] | list[bool]


class HyperParameter(BaseModel):
    """Represents a hyperparameter with key-value pair."""

    key: str = Field(..., description="Hyperparameter name")
    value: ParameterValue = Field(..., description="Hyperparameter value")


class BaseConfig(BaseModel):
    """Base configuration class for models and environments."""

    name: str = Field(..., description="Component name")
    hyperparameters: list[HyperParameter] = Field(
        default_factory=list, description="List of hyperparameters for this component"
    )

    def get_hyperparameters_dict(self) -> dict[str, ParameterValue]:
        """Get hyperparameters as a dictionary."""
        return {hp.key: hp.value for hp in self.hyperparameters}

    def get_hyperparameters_signature(self) -> str:
        """
        Generate a short signature from hyperparameters.

        Returns a string composed of the first three letters of each word in the hyperparameter name
        (split on non-alphanumeric characters) followed by its value, sorted alphabetically by
        parameter name, separated by "_".
        """
        if not self.hyperparameters:
            return "default"

        # Sort hyperparameters alphabetically by key
        sorted_params = sorted(self.hyperparameters, key=lambda hp: hp.key)

        signature_parts = []
        for hp in sorted_params:
            # Split parameter name on non-alphanumeric characters
            words = re.split(r"[^a-zA-Z0-9]", hp.key)
            # Filter out empty strings and get first 3 letters of each word
            param_prefix = ("_").join([word[:3] for word in words if word])

            # Convert value to string and add to signature
            signature_parts.append(f"{param_prefix}_{hp.value}")

        return "__".join(signature_parts)


class ModelConfig(BaseConfig):
    """Configuration for a specific model."""

    name: str = Field(..., description="Model name (subdirectory name in src/hercule/models)")


class EnvironmentConfig(BaseConfig):
    """Configuration for a specific environment."""

    name: str = Field(..., description="Environment name (Gymnasium environment ID)")


class RunConfig(BaseModel):
    """Configuration for model evaluation after training."""

    num_episodes: int = Field(default=10, ge=1, description="Number of episodes to run during evaluation")
    render: bool = Field(default=False, description="Whether to render the environment during evaluation")


class HerculeConfig(BaseModel):
    """Main configuration class for Hercule framework."""

    name: str = Field(
        default="hercule_run",
        description="Configuration name for identifying the run and organizing output directories",
    )
    environments: list[str | EnvironmentConfig] = Field(
        default_factory=lambda: ["CartPole-v1"],
        description="List of Gymnasium environments to test (can be names or configurations)",
    )
    models: list[ModelConfig] = Field(
        default_factory=list, description="List of models to test with their configurations"
    )
    learn_max_epoch: int = Field(default=1000, ge=1, description="Maximum number of learning iterations")
    test_epoch: int = Field(default=100, ge=1, description="Number of testing iterations")
    output_dir: Path = Field(default=Path("outputs"), description="Directory for saving results and models")
    save_every_n_epoch: int = Field(default=None, ge=1, description="Number of epochs to save the model")
    evaluation: RunConfig | None = Field(default=None, description="Evaluation configuration after training")

    @field_validator("environments")
    @classmethod
    def validate_environments(cls, v: list[str | EnvironmentConfig]) -> list[str | EnvironmentConfig]:
        """Validate that environments list is not empty."""
        if not v:
            raise ValueError("At least one environment must be specified")
        return v

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v: str | Path) -> Path:
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v

    def get_environment_configs(self) -> list[EnvironmentConfig]:
        """
        Get all environments as EnvironmentConfig objects.

        Returns:
            List of EnvironmentConfig objects (strings are converted to configs without hyperparameters)
        """
        configs = []
        for env in self.environments:
            if isinstance(env, str):
                configs.append(EnvironmentConfig(name=env))
            else:
                configs.append(env)
        return configs

    def get_environment_names(self) -> list[str]:
        """Get list of environment names."""
        return [env.name if isinstance(env, EnvironmentConfig) else env for env in self.environments]

    def get_model_names(self) -> list[str]:
        """Get list of model names."""
        return [model.name for model in self.models]

    def get_hyperparameters_for_model(self, model_name: str) -> dict[str, ParameterValue]:
        """Get hyperparameters for a specific model as a dictionary."""
        for model in self.models:
            if model.name == model_name:
                return model.get_hyperparameters_dict()
        return {}

    def get_hyperparameters_for_environment(self, env_name: str) -> dict[str, ParameterValue]:
        """Get hyperparameters for a specific environment as a dictionary."""
        for env in self.environments:
            if isinstance(env, EnvironmentConfig) and env.name == env_name:
                return env.get_hyperparameters_dict()
            elif isinstance(env, str) and env == env_name:
                return {}
        return {}

    def get_evaluation_config(self) -> dict[str, ParameterValue] | None:
        """Get evaluation configuration as a dictionary."""
        if self.evaluation is None:
            return None
        return {
            "num_episodes": self.evaluation.num_episodes,
            "render": self.evaluation.render,
        }

    def __str__(self) -> str:
        """Return a formatted string representation of the configuration."""
        config_dict = self.model_dump()

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        config_dict = convert_paths(config_dict)
        return json.dumps(config_dict, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        """Return a detailed string representation of the configuration."""
        config_dict = self.model_dump()

        # Convert Path objects to strings and format for Python eval
        def convert_for_python(obj):
            if isinstance(obj, dict):
                return "{" + ", ".join(f'"{k}": {convert_for_python(v)}' for k, v in obj.items()) + "}"
            elif isinstance(obj, list):
                return "[" + ", ".join(convert_for_python(item) for item in obj) + "]"
            elif isinstance(obj, Path):
                return f'Path("{str(obj)}")'
            elif obj is None:
                return "None"
            elif isinstance(obj, bool):
                return str(obj)
            elif isinstance(obj, str):
                return f'"{obj}"'
            else:
                return str(obj)

        config_str = convert_for_python(config_dict)
        return f"HerculeConfig(**{config_str})"

    def get_directory_for(self, model: ModelConfig, environment: EnvironmentConfig) -> Path:
        """
        Get the directory for a specific model and environment.

        The directory path includes signatures of hyperparameters to ensure uniqueness:
        output_dir/name/environment_name[env_signature]/model_name[model_signature]
        """
        # Get hyperparameter signatures
        env_signature = environment.get_hyperparameters_signature()
        model_signature = model.get_hyperparameters_signature()

        return self.output_dir / self.name / environment.name / env_signature / model.name / model_signature


def load_config_from_yaml(config_path: Path | str) -> HerculeConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        HerculeConfig: Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If configuration validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    return HerculeConfig(**config_data)


def create_default_config() -> HerculeConfig:
    """
    Create a default configuration.

    Returns:
        HerculeConfig: Default configuration instance
    """
    return HerculeConfig()
