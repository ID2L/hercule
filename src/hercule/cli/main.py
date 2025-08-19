"""Main CLI entry point for Hercule RL framework."""

import json
import logging
from datetime import datetime
from pathlib import Path

import click

from hercule.config import load_config_from_yaml
from hercule.models.dummy import DummyModel
from hercule.run import TrainingRunner, create_output_directory


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

            # Store configuration info for summary
            config_info = {"config_file": str(config_file), "config": config, "output_dir": actual_output_dir}
            all_config_info.append(config_info)

            # Run training for this configuration
            config_results = run_training_for_config(config)
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


def run_training_for_config(config):
    """Run training for all model-environment combinations in a configuration."""
    logger = logging.getLogger(__name__)
    results = []

    # Initialize DummyModel for now (TODO: support multiple models from config)
    dummy_model = DummyModel(name="dummy")

    with TrainingRunner(config) as runner:
        # Validate configuration first
        if not runner.validate_configuration():
            logger.error("Environment validation failed")
            click.echo("❌ Environment validation failed")
            return results

        env_names = config.get_environment_names()

        # Get model configurations - for now we'll use dummy model
        # TODO: Iterate through all models in config
        model_configs = {model.name: config.get_hyperparameters_for_model(model.name) for model in config.models}
        dummy_hyperparams = model_configs.get("dummy", {"seed": 42})

        for env_name in env_names:
            click.echo(f"  🏃 Running training: dummy model on {env_name}")

            # Run training
            result = runner.run_single_training(
                model=dummy_model, environment_name=env_name, model_name="dummy", hyperparameters=dummy_hyperparams
            )

            if result.success:
                click.echo(f"    ✅ Success - Mean reward: {result.metrics.get('mean_reward', 'N/A'):.2f}")
                logger.info(f"Training successful for dummy on {env_name}")
            else:
                click.echo(f"    ❌ Failed: {result.error_message}")
                logger.error(f"Training failed for dummy on {env_name}: {result.error_message}")

            results.append(result)

    return results


def save_combined_results(results, config_info_list, output_dir: Path):
    """Save combined results from all configurations."""
    logger = logging.getLogger(__name__)

    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            results_dir = config_output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{result.environment_name}_{result.model_name}_{timestamp}.json"
            result_file = results_dir / filename

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved result: {result_file}")

    # Save global summary results in main output directory
    global_results_dir = output_dir / "results"
    global_results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = global_results_dir / f"cli_summary_{timestamp}.json"

    # Prepare configuration data for summary
    configurations = []
    for info in config_info_list:
        config = info["config"]
        config_dict = config.dict()
        # Convert Path objects to strings for JSON serialization
        config_dict["output_dir"] = str(config_dict["output_dir"])
        configurations.append({"config_file": info["config_file"], "configuration": config_dict})

    summary_data = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "successful_runs": sum(1 for r in results if r.success),
        "failed_runs": sum(1 for r in results if not r.success),
        "configurations": configurations,
        "results": [result.to_dict() for result in results],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Global summary saved to {summary_file}")
    click.echo(f"📊 Global summary saved to {global_results_dir}")

    # Also save configuration-specific summaries
    for info in config_info_list:
        config = info["config"]
        config_output_dir = info["output_dir"]
        config_results_dir = config_output_dir / "results"

        # Filter results for this configuration
        config_results = [r for r in results if r.environment_name in config.get_environment_names()]

        if config_results:
            config_summary_file = config_results_dir / f"cli_summary_{timestamp}.json"
            config_dict = config.dict()
            config_dict["output_dir"] = str(config_dict["output_dir"])

            config_summary_data = {
                "timestamp": timestamp,
                "config_file": info["config_file"],
                "configuration": config_dict,
                "total_runs": len(config_results),
                "successful_runs": sum(1 for r in config_results if r.success),
                "failed_runs": sum(1 for r in config_results if not r.success),
                "results": [result.to_dict() for result in config_results],
            }

            with open(config_summary_file, "w", encoding="utf-8") as f:
                json.dump(config_summary_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration summary saved to {config_summary_file}")
            click.echo(f"📊 Configuration '{config.name}' summary saved to {config_results_dir}")


if __name__ == "__main__":
    cli()
