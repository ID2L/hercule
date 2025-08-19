"""Main CLI entry point for Hercule RL framework."""

import logging
from pathlib import Path

import click


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

    for config_file in config_files:
        logger.info(f"Processing configuration: {config_file}")
        # TODO: Add actual configuration processing here
        click.echo(f"Processing {config_file} -> {output_dir}")

    logger.info("Hercule execution completed")


if __name__ == "__main__":
    cli()
