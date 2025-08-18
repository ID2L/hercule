#!/usr/bin/env python3
"""Test script to display Hercule configuration."""

import logging
from pathlib import Path

from src.hercule.config import load_config_from_yaml
from src.hercule.run import BoxSpaceInfo, DiscreteSpaceInfo, EnvironmentManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def display_config():
    """Load and display the configuration from config_example.yaml."""
    config_path = Path("simple_games.yaml")

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return

    try:
        # Load configuration using the config module function
        config = load_config_from_yaml(config_path)
        logger.info("✓ Configuration loaded successfully from config_example.yaml")

        # Display general settings
        print("\n" + "=" * 60)
        print("HERCULE CONFIGURATION")
        print("=" * 60)

        print("\nGeneral Settings:")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Output directory: {config.output_dir}")

        # Display environments using config methods and environment manager
        env_names = config.get_environment_names()
        env_configs = config.get_environment_configs()
        print(f"\nEnvironments ({len(env_names)} configured):")

        # Use EnvironmentManager to get detailed environment info
        with EnvironmentManager(config) as env_manager:
            for i, env_config in enumerate(env_configs, 1):
                print(f"  {i}. {env_config.name}")

                # Get environment info using the run module function
                try:
                    env_info = env_manager.get_environment_info(env_config.name)
                    print("     Environment Info:")

                    # Display observation space info
                    obs_space = env_info.observation_space
                    print(f"       - Observation space: {obs_space.type}")
                    if obs_space.shape:
                        print(f"         Shape: {obs_space.shape}")
                    if isinstance(obs_space, BoxSpaceInfo):
                        if obs_space.low is not None:
                            print(f"         Low: {obs_space.low}")
                        if obs_space.high is not None:
                            print(f"         High: {obs_space.high}")
                    elif isinstance(obs_space, DiscreteSpaceInfo):
                        if obs_space.n is not None:
                            print(f"         States: {obs_space.n}")

                    # Display action space info
                    action_space = env_info.action_space
                    print(f"       - Action space: {action_space.type}")
                    if action_space.shape:
                        print(f"         Shape: {action_space.shape}")
                    if isinstance(action_space, BoxSpaceInfo):
                        if action_space.low is not None:
                            print(f"         Low: {action_space.low}")
                        if action_space.high is not None:
                            print(f"         High: {action_space.high}")
                    elif isinstance(action_space, DiscreteSpaceInfo):
                        if action_space.n is not None:
                            print(f"         Actions: {action_space.n}")

                    if env_info.max_episode_steps:
                        print(f"       - Max episode steps: {env_info.max_episode_steps}")

                except Exception as e:
                    print(f"     ⚠️  Could not load environment info: {e}")

                # Use the config method to get hyperparameters
                hyperparams = config.get_hyperparameters_for_environment(env_config.name)
                if hyperparams:
                    print("     Configuration Hyperparameters:")
                    for key, value in hyperparams.items():
                        value_type = type(value).__name__
                        if isinstance(value, list) and value:
                            element_type = type(value[0]).__name__ if value else "unknown"
                            print(f"       - {key}: {value} (list of {element_type})")
                        else:
                            print(f"       - {key}: {value} ({value_type})")
                else:
                    print("     No configuration hyperparameters")
                print()  # Empty line for readability

        # Display models using config methods
        model_names = config.get_model_names()
        print(f"\nModels ({len(config.models)} configured):")

        for i, model in enumerate(config.models, 1):
            print(f"  {i}. {model.name}")
            # Use the config method to get hyperparameters
            hyperparams = config.get_hyperparameters_for_model(model.name)
            if hyperparams:
                print("     Hyperparameters:")
                for key, value in hyperparams.items():
                    value_type = type(value).__name__
                    if isinstance(value, list) and value:
                        element_type = type(value[0]).__name__ if value else "unknown"
                        print(f"       - {key}: {value} (list of {element_type})")
                    else:
                        print(f"       - {key}: {value} ({value_type})")
            else:
                print("     No hyperparameters")

        # Display summary using config methods
        print("\nSummary:")
        print(f"  Total environments: {len(env_names)}")
        print(f"  Environment names: {', '.join(env_names)}")
        print(f"  Total models: {len(model_names)}")
        print(f"  Model names: {', '.join(model_names)}")

        print("\n" + "=" * 60)
        logger.info("✓ Configuration display completed successfully")

    except Exception as e:
        logger.error(f"✗ Failed to load or display configuration: {e}")
        raise


def main():
    """Display configuration from config_example.yaml."""
    logger.info("Hercule Configuration Display")
    display_config()


if __name__ == "__main__":
    main()
