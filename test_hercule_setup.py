#!/usr/bin/env python3
"""Test script to display Hercule configuration and run training."""

import json
import logging
from datetime import datetime
from pathlib import Path

from hercule.config import load_config_from_yaml
from hercule.environnements import BoxSpaceInfo, DiscreteSpaceInfo, EnvironmentManager
from hercule.models.dummy import DummyModel
from hercule.run import TrainingRunner, create_output_directory


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

        # Display detailed environment information
        print("\n" + "=" * 60)
        print("DETAILED ENVIRONMENT INFORMATION")
        print("=" * 60)

        # Get detailed info for each created environment
        for i, env_config in enumerate(env_configs, 1):
            print(f"\n{i}. Environment: {env_config.name}")
            print("-" * 40)

            try:
                # Get the created environment instance
                env = env_manager.load_environment(env_config.name)
                env_info = env_manager.get_environment_info(env_config.name)

                # Display basic info
                print(f"   Name: {env_info.name}")
                print(f"   Observation Space: {env_info.observation_space.type}")
                if env_info.observation_space.shape:
                    print(f"   Observation Shape: {env_info.observation_space.shape}")
                print(f"   Action Space: {env_info.action_space.type}")

                if isinstance(env_info.observation_space, DiscreteSpaceInfo):
                    if env_info.observation_space.n:
                        print(f"   Observation States: {env_info.observation_space.n}")
                elif isinstance(env_info.observation_space, BoxSpaceInfo):
                    if env_info.observation_space.low:
                        print(f"   Observation Low: {env_info.observation_space.low}")
                    if env_info.observation_space.high:
                        print(f"   Observation High: {env_info.observation_space.high}")

                if isinstance(env_info.action_space, DiscreteSpaceInfo):
                    if env_info.action_space.n:
                        print(f"   Action Count: {env_info.action_space.n}")
                elif isinstance(env_info.action_space, BoxSpaceInfo):
                    if env_info.action_space.low:
                        print(f"   Action Low: {env_info.action_space.low}")
                    if env_info.action_space.high:
                        print(f"   Action High: {env_info.action_space.high}")

                # Display spec information if available
                if env_info.spec_info:
                    print(f"   Spec ID: {env_info.spec_info.id}")
                    if env_info.spec_info.entry_point:
                        print(f"   Entry Point: {env_info.spec_info.entry_point}")
                    if env_info.spec_info.reward_threshold is not None:
                        print(f"   Reward Threshold: {env_info.spec_info.reward_threshold}")
                    print(f"   Max Episode Steps: {env_info.spec_info.max_episode_steps}")
                    print(f"   Nondeterministic: {env_info.spec_info.nondeterministic}")
                    print(f"   Order Enforce: {env_info.spec_info.order_enforce}")
                    print(f"   Auto Reset: {env_info.spec_info.autoreset}")

                    # Display spec kwargs (actual environment hyperparameters)
                    if env_info.spec_info.kwargs:
                        print("   Spec Kwargs:")
                        for key, value in env_info.spec_info.kwargs.items():
                            print(f"     - {key}: {value} ({type(value).__name__})")

                # Get hyperparameters from the environment inspector
                from hercule.environnements import EnvironmentInspector

                inspector = EnvironmentInspector()
                env_hyperparams = inspector.get_environment_hyperparameters(env)
                if env_hyperparams:
                    print("   Environment Hyperparameters (from .spec):")
                    for key, value in env_hyperparams.items():
                        value_type = type(value).__name__
                        if isinstance(value, list) and value:
                            element_type = type(value[0]).__name__ if value else "unknown"
                            print(f"     - {key}: {value} (list of {element_type})")
                        else:
                            print(f"     - {key}: {value} ({value_type})")

                # Configuration hyperparameters
                config_hyperparams = config.get_hyperparameters_for_environment(env_config.name)
                if config_hyperparams:
                    print("   Configuration Hyperparameters:")
                    for key, value in config_hyperparams.items():
                        value_type = type(value).__name__
                        if isinstance(value, list) and value:
                            element_type = type(value[0]).__name__ if value else "unknown"
                            print(f"     - {key}: {value} (list of {element_type})")
                        else:
                            print(f"     - {key}: {value} ({value_type})")

            except Exception as e:
                print(f"   ⚠️  Error getting detailed info: {e}")

        print("\n" + "=" * 60)
        logger.info("✓ Configuration and environment analysis completed successfully")

    except Exception as e:
        logger.error(f"✗ Failed to load or display configuration: {e}")
        raise


def run_training_tests(config_path: Path):
    """Run training tests with DummyModel and save results to outputs."""
    try:
        # Load configuration
        config = load_config_from_yaml(config_path)
        logger.info("✓ Configuration loaded for training tests")

        # Create output directory structure
        output_dir = create_output_directory(config)
        logger.info(f"✓ Output directory created: {output_dir}")

        # Initialize DummyModel
        dummy_model = DummyModel()

        print("\n" + "=" * 60)
        print("RUNNING TRAINING TESTS")
        print("=" * 60)

        # Run training for each environment with DummyModel
        all_results = []

        with TrainingRunner(config) as runner:
            # Validate configuration first
            if not runner.validate_configuration():
                logger.error("✗ Environment validation failed")
                return

            env_names = config.get_environment_names()
            model_configs = {model.name: config.get_hyperparameters_for_model(model.name) for model in config.models}

            for env_name in env_names:
                print(f"\n🎯 Running training on environment: {env_name}")
                print("-" * 40)

                # Use DummyModel hyperparameters from config if available
                dummy_hyperparams = model_configs.get("dummy", {"seed": 42})

                # Run training
                result = runner.run_single_training(
                    model=dummy_model, environment_name=env_name, model_name="dummy", hyperparameters=dummy_hyperparams
                )

                if result.success:
                    print("✅ Training completed successfully")
                    print(f"   Mean reward: {result.metrics.get('mean_reward', 'N/A'):.2f}")
                    print(f"   Episodes run: {result.metrics.get('episodes_run', 'N/A')}")
                    print(f"   Mean episode length: {result.metrics.get('mean_episode_length', 'N/A'):.1f}")
                else:
                    print(f"❌ Training failed: {result.error_message}")

                all_results.append(result)

        # Save results to outputs directory
        save_results_to_outputs(all_results, output_dir)

        print(f"\n✅ Training tests completed. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"✗ Failed to run training tests: {e}")
        raise


def save_results_to_outputs(results: list, output_dir: Path):
    """Save training results to outputs directory."""
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual results
    results_dir = output_dir / "results"
    for result in results:
        filename = f"{result.environment_name}_{result.model_name}_{timestamp}.json"
        result_file = results_dir / filename

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result: {result_file}")

    # Save summary results
    summary_file = results_dir / f"training_summary_{timestamp}.json"
    summary_data = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "successful_runs": sum(1 for r in results if r.success),
        "failed_runs": sum(1 for r in results if not r.success),
        "results": [result.to_dict() for result in results],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved summary: {summary_file}")

    # Save human-readable summary
    summary_txt = results_dir / f"training_summary_{timestamp}.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("HERCULE TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Successful: {sum(1 for r in results if r.success)}\n")
        f.write(f"Failed: {sum(1 for r in results if not r.success)}\n\n")

        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result.environment_name} + {result.model_name}\n")
            if result.success:
                f.write("   ✅ Success\n")
                f.write(f"   Mean reward: {result.metrics.get('mean_reward', 'N/A')}\n")
                f.write(f"   Episodes: {result.metrics.get('episodes_run', 'N/A')}\n")
                f.write(f"   Mean length: {result.metrics.get('mean_episode_length', 'N/A')}\n")
            else:
                f.write(f"   ❌ Failed: {result.error_message}\n")
            f.write("\n")

    logger.info(f"Saved human-readable summary: {summary_txt}")


def main():
    """Display configuration and run training tests."""
    config_path = Path("simple_games.yaml")

    logger.info("Hercule Configuration Display and Training Tests")

    # Display configuration
    display_config()

    # Run training tests
    run_training_tests(config_path)


if __name__ == "__main__":
    main()
