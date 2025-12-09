"""Module for generating experiment reports."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

from hercule.models import create_model, model_file_name
from hercule.models.epoch_result import EpochResult
from hercule.run import Runner, run_info_file_name
from hercule.supervisor import environment_file_name


logger = logging.getLogger(__name__)


def _format_python_value(value: Any, indent: int = 0, current_indent: int = 0) -> str:
    """
    Format a Python value as a Python literal string.

    Converts dictionaries, lists, booleans, None, etc. to valid Python code.
    This is used to generate Python code from template data.

    Args:
        value: The value to format
        indent: Number of spaces for indentation levels
        current_indent: Current indentation level

    Returns:
        String representation of the value as valid Python code
    """
    indent_str = " " * (current_indent * indent)

    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        next_indent = current_indent + 1
        next_indent_str = " " * (next_indent * indent)
        for k, v in value.items():
            key_str = repr(k) if isinstance(k, str) else repr(k)
            val_str = _format_python_value(v, indent, next_indent)
            items.append(f"{next_indent_str}{key_str}: {val_str}")
        return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]" if isinstance(value, list) else "()"
        items = []
        next_indent = current_indent + 1
        next_indent_str = " " * (next_indent * indent)
        if isinstance(value, list):
            bracket_open = "["
            bracket_close = "]"
        else:
            bracket_open = "("
            bracket_close = ")"
            # Add trailing comma for single-element tuples
            if len(value) == 1:
                bracket_close = ",)"
        for item in value:
            item_str = _format_python_value(item, indent, next_indent)
            items.append(f"{next_indent_str}{item_str}")
        return f"{bracket_open}\n" + ",\n".join(items) + f"\n{indent_str}{bracket_close}"
    # Fallback to repr for other types
    return repr(value)


class ExperimentData:
    """Container for experiment data loaded from JSON files."""

    def __init__(self, experiment_path: Path):
        self.experiment_path = experiment_path
        self.environment_data: dict[str, Any] | None = None
        self.model_data: dict[str, Any] | None = None
        self.run_info_data: dict[str, Any] | None = None
        self.learning_metrics: list[EpochResult] = []
        self.testing_metrics: list[EpochResult] = []
        self.runner: Runner | None = None

    def load_data(self) -> bool:
        """Load all experiment data using existing Hercule methods."""
        try:
            # Load environment data using the constant
            env_file = self.experiment_path / environment_file_name
            if env_file.exists():
                with open(env_file, encoding="utf-8") as f:
                    self.environment_data = json.load(f)
            else:
                logger.warning(f"Environment file not found: {env_file}")

            # Load model data using the constant
            model_file = self.experiment_path / model_file_name
            if model_file.exists():
                with open(model_file, encoding="utf-8") as f:
                    self.model_data = json.load(f)
            else:
                logger.warning(f"Model file not found: {model_file}")

            # Load run info data using Runner.load() method
            try:
                self.runner = Runner.load(self.experiment_path)
                if self.runner:
                    self.run_info_data = {
                        "learning_ongoing_epoch": self.runner.learning_ongoing_epoch,
                        "testing_ongoing_epoch": self.runner.testing_ongoing_epoch,
                        "learning_metrics": [metric.model_dump() for metric in self.runner.learning_metrics],
                        "testing_metrics": [metric.model_dump() for metric in self.runner.testing_metrics],
                    }
                    self.learning_metrics = self.runner.learning_metrics
                    self.testing_metrics = self.runner.testing_metrics
                else:
                    logger.warning(f"Failed to load Runner from {self.experiment_path}")
            except Exception as e:
                logger.warning(f"Error loading Runner: {e}")

            return True

        except Exception as e:
            logger.error(f"Error loading experiment data: {e}")
            return False

    def get_learning_rewards(self) -> list[float]:
        """Extract learning rewards from metrics."""
        return [metric.reward for metric in self.learning_metrics]

    def get_learning_steps(self) -> list[int]:
        """Extract learning steps from metrics."""
        return [metric.steps_number for metric in self.learning_metrics]

    def get_testing_rewards(self) -> list[float]:
        """Extract testing rewards from metrics."""
        return [metric.reward for metric in self.testing_metrics]

    def get_testing_steps(self) -> list[int]:
        """Extract testing steps from metrics."""
        return [metric.steps_number for metric in self.testing_metrics]


def generate_report(experiment_path: Path, output_path: Path | None = None) -> Path:
    """
    Generate a Jupyter notebook report for an experiment.

    The generated report is a Python file (.py) in Jupytext format with cell markers (# %%).
    It can be opened directly as a Jupyter notebook using Jupytext or any IDE that supports it.

    Args:
        experiment_path: Path to the experiment directory containing JSON files
        output_path: Path where to save the generated report (default: experiment_path/report.py)

    Returns:
        Path to the generated report file
    """
    if output_path is None:
        output_path = experiment_path / "report.py"

    # Load experiment data
    experiment_data = ExperimentData(experiment_path)
    if not experiment_data.load_data():
        raise ValueError(f"Failed to load experiment data from {experiment_path}")

    # Create template environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    # Add custom filter to format Python values correctly (False/True instead of false/true)
    env.filters["topython"] = lambda v, indent=2: _format_python_value(v, indent=indent)
    template = env.get_template("report_template.py.j2")

    # Prepare template context
    context = {
        "experiment_path": str(experiment_path),
        "environment_data": experiment_data.environment_data,
        "model_data": experiment_data.model_data,
        "run_info_data": experiment_data.run_info_data,
        "learning_rewards": experiment_data.get_learning_rewards(),
        "learning_steps": experiment_data.get_learning_steps(),
        "testing_rewards": experiment_data.get_testing_rewards(),
        "testing_steps": experiment_data.get_testing_steps(),
        "learning_metrics": experiment_data.learning_metrics,
        "testing_metrics": experiment_data.testing_metrics,
    }

    # Generate report
    report_content = template.render(**context)

    # Write report file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Report generated: {output_path}")
    return output_path


def create_learning_plots(experiment_data: ExperimentData) -> dict[str, str]:
    """
    Create learning progress plots and return base64 encoded images.

    Args:
        experiment_data: Loaded experiment data

    Returns:
        Dictionary with plot names and base64 encoded image data
    """
    import base64
    import io

    plots = {}

    # Learning rewards over time
    if experiment_data.learning_metrics:
        plt.figure(figsize=(12, 8))

        # Plot 1: Learning rewards
        plt.subplot(2, 2, 1)
        rewards = experiment_data.get_learning_rewards()
        plt.plot(rewards, alpha=0.7, label="Episode Reward")

        # Add moving average
        window_size = min(50, len(rewards) // 10)
        if window_size > 1:
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f"Moving Average (window={window_size})", linewidth=2)

        plt.title("Learning Progress - Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Learning steps
        plt.subplot(2, 2, 2)
        steps = experiment_data.get_learning_steps()
        plt.plot(steps, alpha=0.7, label="Episode Steps")

        if window_size > 1:
            moving_avg_steps = pd.Series(steps).rolling(window=window_size).mean()
            plt.plot(moving_avg_steps, label=f"Moving Average (window={window_size})", linewidth=2)

        plt.title("Learning Progress - Steps")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=30, alpha=0.7, edgecolor="black")
        plt.title("Reward Distribution (Learning)")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Plot 4: Steps distribution
        plt.subplot(2, 2, 4)
        plt.hist(steps, bins=30, alpha=0.7, edgecolor="black")
        plt.title("Steps Distribution (Learning)")
        plt.xlabel("Steps")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        plots["learning_progress"] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

    # Testing results boxplot
    if experiment_data.testing_metrics:
        plt.figure(figsize=(10, 6))

        testing_rewards = experiment_data.get_testing_rewards()
        testing_steps = experiment_data.get_testing_steps()

        plt.subplot(1, 2, 1)
        plt.boxplot(testing_rewards, labels=["Testing Rewards"])
        plt.title("Final Model Evaluation - Rewards")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(testing_steps, labels=["Testing Steps"])
        plt.title("Final Model Evaluation - Steps")
        plt.ylabel("Steps")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plots["evaluation_boxplot"] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

    return plots
