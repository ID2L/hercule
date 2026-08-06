"""Controller layer exposing business logic decoupled from CLI.

This module provides reusable functions to orchestrate learning runs and
interactive model execution, without any CLI-specific input/output. It can be
imported by alternative frontends (e.g., web API) to trigger the same actions.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym

from hercule.config import HerculeConfig, load_config_from_yaml
from hercule.environnements import load_environment
from hercule.models import RLModel, create_model
from hercule.reports import _sanitize_reason, generate_report
from hercule.supervisor import Supervisor


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hercule.reports import ReportBundle


logger = logging.getLogger(__name__)


class CancellationToken:
    """Simple thread-safe cancellation token.

    A web API or CLI can hold a reference to this token and request graceful
    stop of a long-running operation.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def run_learning(config_file: Path, output_dir: Path | None = None) -> None:
    """Run the learning then testing phases from a YAML configuration file.

    Args:
        config_file: Path to YAML configuration.
        output_dir: Optional override for output directory.
    """
    config: HerculeConfig = load_config_from_yaml(config_file)

    if output_dir is not None:
        config.base_output_dir = output_dir

    # Save configuration summary to the appropriate location
    config.save()

    supervisor = Supervisor(config)
    supervisor.execute_learn_phase()
    supervisor.execute_test_phase()


@dataclass
class PlayResult:
    total_episodes: int
    total_reward: float

    @property
    def average_reward(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_reward / float(self.total_episodes)


def play_interactive(
    model_file: Path,
    environment_file: Path,
    cancel_token: CancellationToken | None = None,
    render_mode: str | None = "human",
) -> PlayResult:
    """Run an interactive simulation of a trained model until cancelled.

    This function is responsible for the environment/model setup and execution
    loop. It returns aggregate metrics so callers can format final messages.

    Args:
        model_file: JSON file containing trained model parameters.
        environment_file: JSON file containing saved environment configuration.
        render_mode: Gymnasium render mode.
        cancel_token: Optional cancellation token for graceful stop.

    Returns:
        PlayResult: aggregate statistics of the session.
    """
    # Load environment description and instantiate render-capable environment
    environment = load_environment(environment_file)
    env_id = environment.spec.id if getattr(environment, "spec", None) else "Unknown"
    kwargs = getattr(getattr(environment, "spec", None), "kwargs", {}) or {}

    env_with_render = gym.make(env_id, render_mode=render_mode, **kwargs)

    try:
        # Load model payload
        with open(model_file, encoding="utf-8") as f:
            model_data = json.load(f)

        # Require model_name to be present - no default fallback
        if "model_name" not in model_data:
            msg = (
                f"Model file {model_file} does not contain 'model_name' field. "
                "Cannot determine which model type to load. Please ensure the model was saved correctly."
            )
            raise ValueError(msg)

        model_name = model_data["model_name"]
        model: RLModel = create_model(model_name)

        # Configure then hydrate weights
        model.configure(env_with_render, {})
        model.load_from_dict(model_data)

        total_reward: float = 0.0
        episode_count: int = 0

        while True:
            # Cancellation check at episode boundary
            if cancel_token is not None and cancel_token.is_cancelled():
                break

            obs, _ = env_with_render.reset()
            episode_reward: float = 0.0
            done = False
            episode_count += 1

            try:
                while not done:
                    # Periodic cancellation check inside episode
                    if cancel_token is not None and cancel_token.is_cancelled():
                        # Stop current episode early and exit gracefully
                        done = True
                        break

                    action = model.predict(obs)
                    obs, reward, terminated, truncated, _ = env_with_render.step(action)
                    episode_reward += float(reward)
                    done = bool(terminated or truncated)

                    if render_mode is not None:
                        env_with_render.render()

            except KeyboardInterrupt:
                # Graceful interruption inside an episode
                break

            total_reward += episode_reward

        return PlayResult(total_episodes=episode_count, total_reward=total_reward)

    except KeyboardInterrupt:
        # Interruption at outer loop level
        return PlayResult(total_episodes=0, total_reward=0.0)
    finally:
        env_with_render.close()


def generate_experiment_report(
    experiment_path: Path,
    output_path: Path | None = None,
    *,
    execute: bool = True,
    render_pdf: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportBundle:
    """Generate a comprehensive report for an experiment.

    This function orchestrates the report generation process, providing a clean
    interface for CLI and other frontends to generate experiment reports.

    Args:
        experiment_path: Path to the experiment directory containing JSON files
        output_path: Optional path where to save the generated report
        execute: Whether generated notebooks should be executed.
        render_pdf: Whether a PDF should be rendered alongside each notebook.
        progress: Optional sink for human-readable progress lines, so the CLI can satisfy a
            progress cadence without `reports/` importing Click.

    Returns:
        ReportBundle describing every artifact produced and every group skipped. A group
        whose PDF was skipped is still a successful group: `pdf=None` with a populated
        `pdf_skip_reason`.

    Raises:
        ValueError: If experiment data cannot be loaded, or no qualifying report group
            was found. This means the *input* is the problem.
        FileNotFoundError: If experiment directory doesn't exist.
        OSError: If a generated artifact could not be written -- most commonly because its
            destination (a `.pdf`/`.ipynb`/`.html`) is open in another program. This means the
            *output* is the problem, never the experiment data, and must not be reported as
            "invalid experiment data" (Defect 3).
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")

    if not experiment_path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_path}")

    logger.info(f"Generating report for experiment: {experiment_path}")

    try:
        bundle = generate_report(
            experiment_path,
            output_path,
            execute=execute,
            render_pdf=render_pdf,
            progress=progress,
        )
        logger.info(f"Report generated successfully: {bundle.report_count} report(s)")
        return bundle

    except FileNotFoundError:
        raise

    except OSError as e:
        # An OSError here means the experiment data loaded fine and only an *output* artifact
        # could not be written -- almost always a locked file (a PDF preview tab, an editor).
        # Kept as a distinct branch from the generic Exception handler below so the CLI can
        # tell "your output file is locked" apart from "your experiment data is invalid"
        # (Defect 3): the previous blanket handler mislabelled every such failure as the
        # latter.
        sanitized = _sanitize_reason(str(e))
        logger.error(f"Cannot write report output for {experiment_path}: {sanitized}")
        raise OSError(
            f"Cannot write report output: {sanitized}. If a generated file (.pdf/.ipynb/.html) "
            "is open in another program (e.g. a preview tab), close it and retry."
        ) from e

    except Exception as e:
        logger.error(f"Failed to generate report for {experiment_path}: {e}")
        raise ValueError(f"Failed to generate report: {e}") from e
