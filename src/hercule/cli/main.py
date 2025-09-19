"""Main CLI entry point for Hercule RL framework."""

import logging
from functools import wraps
from pathlib import Path

import click

from hercule.controller import CancellationToken, play_interactive, run_learning


def configure_logging(verbose: int) -> logging.Logger:
    """Configure logging based on verbosity level."""
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,  # Most verbose, same as DEBUG but could be extended
    }

    log_level = log_levels.get(verbose, logging.DEBUG)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def verbose_option(func):
    """Decorator to add --verbose/-v option to CLI commands."""

    @click.option("--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, or -vvv for different levels)")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx) -> None:
    """Hercule RL framework CLI.

    A reinforcement learning framework for training and playing with RL agents.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), metavar="CONFIG_FILE")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="outputs",
    help="Output directory for results and models (default: outputs)",
    show_default=True,
)
@verbose_option
@click.pass_context
def learn(ctx, config_file: Path, output_dir: Path, verbose: int) -> None:
    """Learn and evaluate RL algorithms using a configuration file.

    This command trains RL models on specified environments and saves metrics and model data.

    CONFIG_FILE: YAML configuration file to process.
    """
    logger = configure_logging(verbose)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Hercule learning with configuration file: {config_file}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    click.echo(f"\n🎯 Learning with {config_file}")

    try:
        # Echo output directory preview (real creation handled by controller)
        if output_dir != Path("outputs"):
            click.echo(f"📁 Output directory override: {output_dir}")

        run_learning(config_file=config_file, output_dir=output_dir if output_dir != Path("outputs") else None)

    except Exception as e:
        logger.error(f"Failed to process {config_file}: {e}")
        click.echo(f"❌ Error processing {config_file}: {e}")
        return

    logger.info("Hercule learning completed")


@cli.command()
@click.argument("model_file", type=click.Path(exists=True, path_type=Path), metavar="MODEL_FILE")
@click.argument("environment_file", type=click.Path(exists=True, path_type=Path), metavar="ENVIRONMENT_FILE")
@click.option(
    "--no-render",
    is_flag=True,
    default=False,
    help="Disable visual rendering (useful for testing and CI/CD)",
    show_default=True,
)
@verbose_option
@click.pass_context
def play(ctx, model_file: Path, environment_file: Path, no_render: bool, verbose: int) -> None:
    """Play with a trained RL model in visual mode.

    This command loads a trained model and environment configuration to play episodes
    with visual rendering. Press Ctrl+C to stop the simulation.

    MODEL_FILE: JSON file containing the trained model data.

    ENVIRONMENT_FILE: JSON file containing the environment configuration.
    """
    logger = configure_logging(verbose)

    logger.info(f"Starting Hercule play with model: {model_file}")
    logger.info(f"Environment file: {environment_file}")

    click.echo(f"\n🎮 Playing with model: {model_file}")
    click.echo(f"🌍 Environment: {environment_file}")
    click.echo("💡 Press Ctrl+C to stop the simulation")

    try:
        cancel = CancellationToken()

        try:
            result = play_interactive(
                model_file=model_file,
                environment_file=environment_file,
                cancel_token=cancel,
                render_mode=None if no_render else "human",
            )
        except KeyboardInterrupt:
            # Redundant safety; controller already handles it, but we keep a clean UX
            click.echo("\n\n⏹️  Simulation stopped by user")
            result = None

        if result is not None:
            if result.total_episodes > 0:
                click.echo(f"📈 Average reward over {result.total_episodes} episodes: {result.average_reward:.2f}")
            else:
                click.echo("📊 No episodes completed")

    except Exception as e:
        logger.error(f"Failed to play with model {model_file}: {e}")
        click.echo(f"❌ Error playing with model: {e}")
        return

    logger.info("Hercule play completed")


if __name__ == "__main__":
    cli()
