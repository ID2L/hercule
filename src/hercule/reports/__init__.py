"""Module for generating experiment reports."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

from hercule.environnements import load_environment
from hercule.models import create_model, model_file_name
from hercule.models.epoch_result import EpochResult
from hercule.run import Runner, run_info_file_name
from hercule.supervisor import environment_file_name


logger = logging.getLogger(__name__)

# Maximum depth for recursive search of experiment directories
MAX_DEPTH = 4


def is_valid_experiment_directory(directory: Path) -> bool:
    """
    Check if a directory contains all required experiment files.

    Args:
        directory: Path to the directory to check

    Returns:
        True if the directory contains environment.json, model.json, and run_info.json
    """
    if not directory.is_dir():
        return False

    required_files = [
        environment_file_name,
        model_file_name,
        run_info_file_name,
    ]

    return all((directory / filename).exists() for filename in required_files)


def find_experiment_directories(root_directory: Path, max_depth: int = MAX_DEPTH, current_depth: int = 0) -> list[Path]:
    """
    Recursively find all directories that contain a valid experiment structure.

    Searches up to max_depth levels deep starting from root_directory.

    Args:
        root_directory: Root directory to search in
        max_depth: Maximum depth to search (default: MAX_DEPTH)
        current_depth: Current recursion depth (used internally)

    Returns:
        List of paths to directories containing valid experiment structures
    """
    experiment_dirs: list[Path] = []

    if not root_directory.is_dir():
        return experiment_dirs

    # Check if current directory is a valid experiment directory
    if is_valid_experiment_directory(root_directory):
        experiment_dirs.append(root_directory)
        return experiment_dirs

    # If we've reached max depth, stop searching
    if current_depth >= max_depth:
        return experiment_dirs

    # Recursively search subdirectories
    try:
        for item in root_directory.iterdir():
            if item.is_dir():
                subdir_experiments = find_experiment_directories(item, max_depth, current_depth + 1)
                experiment_dirs.extend(subdir_experiments)
    except PermissionError:
        logger.warning(f"Permission denied accessing {root_directory}")

    return experiment_dirs


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
            key_str = repr(k)
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
                # Load raw JSON to get hyperparameters
                run_info_file = self.experiment_path / run_info_file_name
                hyperparams_from_json: dict[str, Any] = {}
                if run_info_file.exists():
                    with open(run_info_file, encoding="utf-8") as f:
                        run_info_raw = json.load(f)
                        # Get hyperparameters directly from JSON
                        hyperparams_from_json = run_info_raw.get("model_hyperparameters", {})

                self.runner = Runner.load(self.experiment_path)
                if self.runner:
                    # Get hyperparameters from the configured model using get_hyperparameters_dict()
                    hyperparams: dict[str, Any] = {}
                    if len(self.experiment_path.parts) >= 1:
                        model_name = self.experiment_path.parent.name
                        try:
                            # Create model instance
                            model = create_model(model_name)
                            # Load model data from JSON if available
                            if model_file.exists():
                                model.load(self.experiment_path)

                            # Load environment if available to configure the model
                            env = None
                            if self.environment_data and "id" in self.environment_data:
                                try:
                                    env = load_environment(self.experiment_path)
                                except Exception as e:
                                    logger.debug(f"Could not load environment: {e}")

                            # Configure model with hyperparameters from JSON (or use defaults)
                            if (
                                hyperparams_from_json
                                and isinstance(hyperparams_from_json, dict)
                                and len(hyperparams_from_json) > 0
                            ):
                                # Use hyperparameters from JSON
                                if env:
                                    model.configure(env, hyperparams_from_json)
                                else:
                                    # If no environment, we can't fully configure, but we can still get defaults
                                    # Merge provided hyperparameters with defaults
                                    defaults = model.get_default_hyperparameters()
                                    merged = defaults.copy()
                                    merged.update(hyperparams_from_json)
                                    # Store in model's hyperparameters list
                                    from hercule.config import HyperParameter

                                    model.hyperparameters = [HyperParameter(key=k, value=v) for k, v in merged.items()]
                            else:
                                # No hyperparameters in JSON, use defaults
                                if env:
                                    model.configure(env, {})
                                else:
                                    # Store defaults in model's hyperparameters
                                    defaults = model.get_default_hyperparameters()
                                    from hercule.config import HyperParameter

                                    model.hyperparameters = [
                                        HyperParameter(key=k, value=v) for k, v in defaults.items()
                                    ]

                            # Get hyperparameters from the configured model
                            hyperparams = model.get_hyperparameters_dict()
                            logger.debug(f"Retrieved hyperparameters from model {model_name}: {hyperparams}")
                        except Exception as e:
                            logger.warning(f"Could not get hyperparameters from model {model_name}: {e}")
                            # Fallback to JSON or defaults
                            if (
                                hyperparams_from_json
                                and isinstance(hyperparams_from_json, dict)
                                and len(hyperparams_from_json) > 0
                            ):
                                hyperparams = hyperparams_from_json.copy()
                            else:
                                try:
                                    temp_model = create_model(model_name)
                                    hyperparams = temp_model.get_default_hyperparameters()
                                except Exception:
                                    hyperparams = {}

                    self.run_info_data = {
                        "learning_ongoing_epoch": self.runner.learning_ongoing_epoch,
                        "testing_ongoing_epoch": self.runner.testing_ongoing_epoch,
                        "learning_metrics": [metric.model_dump() for metric in self.runner.learning_metrics],
                        "testing_metrics": [metric.model_dump() for metric in self.runner.testing_metrics],
                        "model_hyperparameters": hyperparams,
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


def generate_individual_report(experiment_path: Path, output_path: Path | None = None) -> Path:
    """
    Generate an individual Jupyter notebook report for a single experiment.

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

    logger.info(f"Individual report generated: {output_path}")
    return output_path


def generate_report(experiment_path: Path, output_path: Path | None = None) -> Path:
    """
    Generate a Jupyter notebook report for an experiment or multiple experiments.

    This function automatically detects the type of report to generate:
    - If the given directory contains a valid experiment structure (environment.json,
      model.json, run_info.json), generates an individual report.
    - If the given directory contains subdirectories with valid experiment structures
      (searched up to MAX_DEPTH levels deep), generates a comparative report.

    Args:
        experiment_path: Path to the experiment directory or parent directory
        output_path: Path where to save the generated report
                    (default: experiment_path/report.py for individual,
                     experiment_path/comparative_report.py for comparative)

    Returns:
        Path to the generated report file

    Raises:
        ValueError: If no valid experiment directories are found
        FileNotFoundError: If the experiment_path doesn't exist
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Directory not found: {experiment_path}")

    if not experiment_path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_path}")

    # Check if the root directory itself is a valid experiment directory
    if is_valid_experiment_directory(experiment_path):
        logger.info("Root directory is a valid experiment directory, generating individual report")
        return generate_individual_report(experiment_path, output_path)

    # Search for experiment directories recursively
    logger.info(f"Searching for experiment directories in {experiment_path} (max depth: {MAX_DEPTH})")
    experiment_dirs = find_experiment_directories(experiment_path, max_depth=MAX_DEPTH)

    if not experiment_dirs:
        logger.warning(
            f"No valid experiment directories found in {experiment_path} (searched up to {MAX_DEPTH} levels deep)"
        )
        raise ValueError(
            f"No valid experiment directories found in {experiment_path}. "
            f"A valid experiment directory must contain: {environment_file_name}, "
            f"{model_file_name}, and {run_info_file_name}"
        )

    logger.info(f"Found {len(experiment_dirs)} experiment directory(ies), generating comparative reports")

    # Group experiments by environment + environment parametrization
    # Structure: root/env/env_params/model/model_params/
    # We group by env/env_params (2 levels up from experiment directory)
    experiment_groups: dict[Path, list[Path]] = {}
    for exp_dir in experiment_dirs:
        # Go up 2 levels to get to the environment parametrization level
        # exp_dir is at model_params level, so:
        # parent is model, parent.parent is env_params
        if len(exp_dir.parts) >= 2:
            env_params_dir = exp_dir.parent.parent
            if env_params_dir not in experiment_groups:
                experiment_groups[env_params_dir] = []
            experiment_groups[env_params_dir].append(exp_dir)
        else:
            logger.warning(f"Cannot determine environment parametrization for {exp_dir}, skipping")

    if not experiment_groups:
        raise ValueError("Could not group experiments by environment parametrization")

    logger.info(f"Found {len(experiment_groups)} environment parametrization group(s)")

    # Generate a comparative report for each group
    generated_reports: list[Path] = []
    for env_params_dir, group_experiment_dirs in experiment_groups.items():
        if len(group_experiment_dirs) < 2:
            logger.info(
                f"Skipping {env_params_dir}: only {len(group_experiment_dirs)} experiment(s), "
                "need at least 2 for comparison"
            )
            continue

        # Generate report at the environment parametrization level
        report_path = env_params_dir / "comparative_report.py"

        logger.info(f"Generating comparative report for {len(group_experiment_dirs)} experiments in {env_params_dir}")

        # Load data from all experiment directories in this group
        experiments = []
        for exp_dir in group_experiment_dirs:
            try:
                exp_data = ExperimentData(exp_dir)
                if exp_data.load_data():
                    # Generate a readable name from the directory path relative to env_params_dir
                    relative_path = exp_dir.relative_to(env_params_dir)
                    exp_name = str(relative_path).replace("\\", "/")

                    experiments.append(
                        {
                            "path": str(exp_dir),
                            "name": exp_name,
                            "environment_data": exp_data.environment_data,
                            "model_data": exp_data.model_data,
                            "run_info_data": exp_data.run_info_data,
                            "learning_rewards": exp_data.get_learning_rewards(),
                            "learning_steps": exp_data.get_learning_steps(),
                            "testing_rewards": exp_data.get_testing_rewards(),
                            "testing_steps": exp_data.get_testing_steps(),
                            "model_hyperparameters": exp_data.run_info_data.get("model_hyperparameters", {})
                            if exp_data.run_info_data
                            else {},
                        }
                    )
                    logger.debug(f"Loaded experiment data from: {exp_dir}")
                else:
                    logger.warning(f"Failed to load data from: {exp_dir}")
            except Exception as e:
                logger.error(f"Error loading experiment from {exp_dir}: {e}")

        if len(experiments) < 2:
            logger.warning(f"Not enough valid experiments for comparison in {env_params_dir}, skipping")
            continue

        # Create template environment
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        # Add custom filter to format Python values correctly (False/True instead of false/true)
        env.filters["topython"] = lambda v, indent=2: _format_python_value(v, indent=indent)
        template = env.get_template("comparative_report_template.py.j2")

        # Prepare template context
        context = {
            "root_path": str(env_params_dir),
            "experiments": experiments,
        }

        # Generate report
        report_content = template.render(**context)

        # Write report file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Comparative report generated: {report_path}")
        generated_reports.append(report_path)

    if not generated_reports:
        raise ValueError(
            "No comparative reports could be generated. "
            "Ensure there are at least 2 experiments per environment parametrization group."
        )

    # If output_path was specified, return the first report, otherwise return list
    if output_path:
        return generated_reports[0] if generated_reports else experiment_path / "comparative_report.py"

    # Return the first report as default (backward compatibility)
    return generated_reports[0] if generated_reports else experiment_path / "comparative_report.py"


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
