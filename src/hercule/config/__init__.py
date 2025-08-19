"""Configuration module for Hercule reinforcement learning framework."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# Type alias for hyperparameter values
ParameterValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]


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


class ModelConfig(BaseConfig):
    """Configuration for a specific model."""

    name: str = Field(..., description="Model name (subdirectory name in src/hercule/models)")


class EnvironmentConfig(BaseConfig):
    """Configuration for a specific environment."""

    name: str = Field(..., description="Environment name (Gymnasium environment ID)")


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
    max_iterations: int = Field(default=1000, ge=1, description="Maximum number of learning iterations")
    output_dir: Path = Field(default=Path("outputs"), description="Directory for saving results and models")

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


def save_config_to_yaml(config: HerculeConfig, config_path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration instance to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and handle Path objects
    config_dict = config.dict()
    config_dict["output_dir"] = str(config_dict["output_dir"])

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config_dict, file, default_flow_style=False, indent=2)
