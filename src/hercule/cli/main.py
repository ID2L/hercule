"""Main CLI entry point for Hercule RL framework."""

import json
import logging
from pathlib import Path

import click

from hercule.config import load_config_from_yaml
from hercule.models import get_available_models
from hercule.run import (
    RunManager,
    create_output_directory,
    generate_filename_suffix,
    save_config_summary,
    save_evaluation_results,
)


@click.command()
@click.argument(
    "config_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path), metavar="CONFIG_FILES..."
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="outputs",
    help="Output directory for results and models (default: outputs)",
    show_default=True,
)
@click.option("--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, or -vvv for different levels)")
@click.version_option()
def cli(config_files: list[Path], output_dir: Path, verbose: int) -> None:
    """Hercule RL framework CLI.

    Run reinforcement learning experiments using configuration files.

    CONFIG_FILES: One or more YAML configuration files to process.
    """
    # Configure logging based on verbosity level
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,  # Most verbose, same as DEBUG but could be extended
    }

    log_level = log_levels.get(verbose, logging.DEBUG)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Hercule with {len(config_files)} configuration file(s)")
    logger.info(f"Output directory: {output_dir.absolute()}")

    all_results = []
    all_config_info = []

    for config_file in config_files:
        logger.info(f"Processing configuration: {config_file}")
        click.echo(f"\n🎯 Processing {config_file}")

        try:
            # Load configuration
            config = load_config_from_yaml(config_file)

            # Override output directory if specified via CLI
            if output_dir != Path("outputs"):
                config.output_dir = output_dir

            # Create output directory
            actual_output_dir = create_output_directory(config)
            click.echo(f"📁 Output directory: {actual_output_dir}")

            # Save configuration summary at the root of the project directory
            save_config_summary(config, actual_output_dir)

            # Store configuration info for summary
            config_info = {"config_file": str(config_file), "config": config, "output_dir": actual_output_dir}
            all_config_info.append(config_info)

            # Run training for this configuration
            config_results = run_training_for_config(config, actual_output_dir)
            all_results.extend(config_results)

        except Exception as e:
            logger.error(f"Failed to process {config_file}: {e}")
            click.echo(f"❌ Error processing {config_file}: {e}")
            continue

    # Save combined results if we have any
    if all_results:
        save_combined_results(all_results, all_config_info, output_dir)
        successful_runs = sum(1 for r in all_results if r.success)
        total_runs = len(all_results)
        click.echo(f"\n✅ Hercule execution completed: {successful_runs}/{total_runs} runs successful")
    else:
        click.echo("\n⚠️ No results generated")

    logger.info("Hercule execution completed")


def run_training_for_config(config, output_dir: Path | None = None):
    """Run training and evaluation for all model-environment combinations in a configuration."""
    logger = logging.getLogger(__name__)
    training_results = []
    evaluation_results = []

    # Get all available models
    available_models = get_available_models()
    logger.info(f"Available models: {list(available_models.keys())}")

    # Get evaluation configuration
    evaluation_config = config.get_evaluation_config()
    if evaluation_config:
        logger.info(f"Evaluation configuration: {evaluation_config}")
        click.echo(f"  🎯 Evaluation enabled: {evaluation_config['num_episodes']} episodes")

    with RunManager(config, log_level="INFO") as run_manager:
        # Validate configuration first
        if not run_manager.validate_configuration():
            logger.error("Environment validation failed")
            click.echo("❌ Environment validation failed")
            return training_results

        # Iterate through all model configurations in config
        for model_config in config.models:
            model_name = model_config.name

            # Create model instance
            try:
                model = available_models[model_name]()
                logger.info(f"Created model instance: {model_name}")
            except Exception as e:
                logger.error(f"Failed to create model '{model_name}': {e}")
                click.echo(f"  ❌ Failed to create model '{model_name}': {e}")
                continue

            # Get hyperparameters for this specific model configuration
            hyperparams = model_config.get_hyperparameters_dict()

            # Run training and evaluation for each environment configuration
            for env_config in config.get_environment_configs():
                env_name = env_config.name
                env_hyperparams = env_config.get_hyperparameters_dict()

                # Create a unique environment identifier for display
                env_display_name = env_name
                if env_hyperparams:
                    env_display_name = f"{env_name} ({', '.join([f'{k}={v}' for k, v in env_hyperparams.items()])})"

                click.echo(f"  🏃 Running training + evaluation: {model_name} model on {env_display_name}")

                # Run training and evaluation
                models_dir = output_dir / "models" if output_dir else None
                training_result, evaluation_result = run_manager.run_training_and_evaluation(
                    model=model,
                    environment_name=env_name,
                    model_name=model_name,
                    hyperparameters=hyperparams,
                    evaluation_config=evaluation_config,
                    environment_hyperparameters=env_hyperparams,
                    models_dir=models_dir,
                )

                if training_result.success:
                    click.echo(
                        f"    ✅ Training success - Mean reward: {training_result.metrics.get('mean_reward', 0):.2f}"
                    )
                    logger.info(f"Training successful for {model_name} on {env_name}")
                else:
                    click.echo(f"    ❌ Training failed: {training_result.error_message}")
                    logger.error(f"Training failed for {model_name} on {env_name}")

                if evaluation_result and evaluation_result.success:
                    click.echo(
                        f"    ✅ Evaluation success - Mean reward: {evaluation_result.mean_reward:.2f} ± {evaluation_result.std_reward:.2f}"
                    )
                    logger.info(f"Evaluation successful for {model_name} on {env_name}")
                    evaluation_results.append(evaluation_result)
                elif evaluation_config:
                    click.echo("    ⚠️ Evaluation failed or skipped")
                    logger.warning(f"Evaluation failed for {model_name} on {env_name}")

                training_results.append(training_result)

    # Save evaluation results if any
    if evaluation_results and output_dir:
        try:
            # Use the provided output directory
            eval_output_dir = output_dir / "evaluation"
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate descriptive filename for evaluation results
            if evaluation_results:
                # Use the first result to get hyperparameters for naming
                first_result = evaluation_results[0]
                param_suffix = generate_filename_suffix(
                    {}
                )  # Evaluation results don't have hyperparameters in the same way
                filename = f"evaluation_results{param_suffix}.json"
            else:
                filename = "evaluation_results.json"

            save_evaluation_results(evaluation_results, eval_output_dir, filename)
            click.echo(f"  📊 Evaluation results saved to {eval_output_dir}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    return training_results


def save_combined_results(results, config_info_list, output_dir: Path):
    """Save combined results from all configurations."""
    logger = logging.getLogger(__name__)

    # Save individual results in their respective configuration directories
    for result in results:
        # Find the corresponding config info for this result
        config_info = None
        for info in config_info_list:
            config = info["config"]
            # Check if this result belongs to this configuration based on environments
            if result.environment_name in config.get_environment_names():
                config_info = info
                break

        if config_info:
            config_output_dir = config_info["output_dir"]

            # Generate descriptive filename with hyperparameters
            param_suffix = generate_filename_suffix(result.hyperparameters, result.environment_hyperparameters)
            filename = f"{result.environment_name}_{result.model_name}{param_suffix}.json"

            # Save training results directly in training subdirectory
            training_results_dir = config_output_dir / "training"
            training_results_dir.mkdir(parents=True, exist_ok=True)
            result_file = training_results_dir / filename

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved training result: {result_file}")

    logger.info("All training results saved successfully")
    click.echo("📊 Training results saved")


if __name__ == "__main__":
    cli()
