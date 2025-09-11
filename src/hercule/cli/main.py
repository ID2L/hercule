"""Main CLI entry point for Hercule RL framework."""

import json
import logging
from pathlib import Path

import click
import gymnasium as gym

from hercule.config import load_config_from_yaml
from hercule.environnements import load_environment
from hercule.models import create_model
from hercule.supervisor import Supervisor


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


@click.group()
@click.option("--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, or -vvv for different levels)")
@click.version_option()
@click.pass_context
def cli(ctx, verbose: int) -> None:
    """Hercule RL framework CLI.

    A reinforcement learning framework for training and playing with RL agents.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    configure_logging(verbose)


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
@click.pass_context
def learn(ctx, config_file: Path, output_dir: Path) -> None:
    """Learn and evaluate RL algorithms using a configuration file.

    This command trains RL models on specified environments and saves metrics and model data.

    CONFIG_FILE: YAML configuration file to process.
    """
    logger = configure_logging(ctx.obj["verbose"])

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Hercule learning with configuration file: {config_file}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    click.echo(f"\n🎯 Learning with {config_file}")

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
        supervisor.execute_test_phase()

    except Exception as e:
        logger.error(f"Failed to process {config_file}: {e}")
        click.echo(f"❌ Error processing {config_file}: {e}")
        return

    logger.info("Hercule learning completed")


@cli.command()
@click.argument("model_file", type=click.Path(exists=True, path_type=Path), metavar="MODEL_FILE")
@click.argument("environment_file", type=click.Path(exists=True, path_type=Path), metavar="ENVIRONMENT_FILE")
@click.option(
    "--render-mode",
    "-r",
    type=click.Choice(["human", "rgb_array", "ansi"]),
    default="human",
    help="Render mode for visualization (default: human)",
    show_default=True,
)
@click.pass_context
def play(ctx, model_file: Path, environment_file: Path, render_mode: str) -> None:
    """Play with a trained RL model in visual mode.

    This command loads a trained model and environment configuration to play episodes
    with visual rendering. Press Ctrl+C to stop the simulation.

    MODEL_FILE: JSON file containing the trained model data.

    ENVIRONMENT_FILE: JSON file containing the environment configuration.
    """
    logger = configure_logging(ctx.obj["verbose"])

    logger.info(f"Starting Hercule play with model: {model_file}")
    logger.info(f"Environment file: {environment_file}")
    logger.info(f"Render mode: {render_mode}")

    click.echo(f"\n🎮 Playing with model: {model_file}")
    click.echo(f"🌍 Environment: {environment_file}")
    click.echo(f"🎨 Render mode: {render_mode}")
    click.echo("💡 Press Ctrl+C to stop the simulation")

    try:
        # Load environment from saved configuration
        environment = load_environment(environment_file)

        # Create environment with render mode
        env_with_render = gym.make(
            environment.spec.id if environment.spec else "Unknown",
            render_mode=render_mode,
            **getattr(environment.spec, "kwargs", {}) if environment.spec else {},
        )

        # Load model
        with open(model_file) as f:
            model_data = json.load(f)

        # Create and configure model
        model_name = model_data.get("model_name", "simple_sarsa")
        model = create_model(model_name)

        # Configure model with environment
        model.configure(env_with_render, {})

        # Load model weights
        model.load_from_dict(model_data)

        # Play episodes until interrupted
        total_reward = 0
        episode_count = 0

        try:
            while True:
                obs, _ = env_with_render.reset()
                episode_reward = 0
                done = False
                episode_count += 1

                click.echo(f"\n🎯 Episode {episode_count}")

                while not done:
                    action = model.predict(obs)
                    obs, reward, terminated, truncated, _ = env_with_render.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                    if render_mode == "human":
                        env_with_render.render()

                total_reward += episode_reward
                click.echo(f"Episode {episode_count} reward: {episode_reward:.2f}")

        except KeyboardInterrupt:
            click.echo("\n\n⏹️  Simulation stopped by user")
            if episode_count > 0:
                avg_reward = total_reward / episode_count
                click.echo(f"📈 Average reward over {episode_count} episodes: {avg_reward:.2f}")
            else:
                click.echo("📊 No episodes completed")

        finally:
            env_with_render.close()

    except Exception as e:
        logger.error(f"Failed to play with model {model_file}: {e}")
        click.echo(f"❌ Error playing with model: {e}")
        return

    logger.info("Hercule play completed")


if __name__ == "__main__":
    cli()
