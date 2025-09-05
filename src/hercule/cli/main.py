"""Main CLI entry point for Hercule RL framework."""

import logging
from pathlib import Path

import click

from hercule.config import load_config_from_yaml
from hercule.supervisor import Supervisor


@click.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), metavar="CONFIG_FILE")
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
def cli(config_file: Path, output_dir: Path, verbose: int) -> None:
    """Hercule RL framework CLI.

    Run reinforcement learning experiments using a configuration file.

    CONFIG_FILE: YAML configuration file to process.
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

    logger.info(f"Starting Hercule with configuration file: {config_file}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    click.echo(f"\n🎯 Processing {config_file}")

    try:
        # Load configuration
        config = load_config_from_yaml(config_file)

        # Override output directory if specified via CLI
        if output_dir != Path("outputs"):
            config.output_dir = output_dir

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Output directory: {config.output_dir}")

        # Save configuration summary at the root of the project directory
        config_summary_file = config.output_dir / "config_summary.yaml"
        with open(config_summary_file, "w", encoding="utf-8") as f:
            f.write(str(config))
        click.echo(f"📄 Configuration summary saved to: {config_summary_file}")

        supervisor = Supervisor(config)
        supervisor.execute_learn_phase()

    except Exception as e:
        logger.error(f"Failed to process {config_file}: {e}")
        click.echo(f"❌ Error processing {config_file}: {e}")
        return

    logger.info("Hercule execution completed")


if __name__ == "__main__":
    cli()
