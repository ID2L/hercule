"""Run module for training execution and result management."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hercule.config import HerculeConfig, ParameterValue
from hercule.environnements import EnvironmentManager, save_environment
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


# Type checking imports
if TYPE_CHECKING:
    from typing import Any

    ModelType = RLModel
else:
    from typing import Any

    ModelType = Any

# Import evaluation module
from .evaluation import (
    EvaluationProtocol,
    EvaluationResult,
    ModelEvaluator,
    save_evaluation_results,
)


logger = logging.getLogger(__name__)


class Runner(BaseModel):
    ongoing_epoch: int = Field(default=0, description="")
    learning_metrics: list[EpochResult] = Field(default=[], description="")
    directory_path: Path = Field(default=Path("."), description="")
    model: RLModel | None = Field(default=None, description="")
    environment: gym.Env | None = Field(default=None, description="")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load(cls, directory_path: Path):
        # La fonction load doit créer une instance de la classe runner à partir du chemin indiqué
        # Pour cela elle doit vérifier si un fichier "run_info.json" existe à l'empacement spécfié
        # Si il n'existe pas, on renvoie None
        # Si il existe, alors on instancie une classe du Runner avec les informations récupérer dans {directory_path}/run_info.json
        # en parsant le contenu du json
        run_info_file = directory_path / "run_info.json"

        if not run_info_file.exists():
            return Runner(ongoing_epoch=0, directory_path=directory_path)

        try:
            with open(run_info_file, encoding="utf-8") as f:
                run_data = json.load(f)
            return Runner(**run_data, directory_path=directory_path)

        except Exception as e:
            logger.error(f"Failed to load Runner from {run_info_file}: {e}")
            raise e

    def __str__(self):
        # la représentation de l'objet en string est un dictionnaire avec les valeurs litérals pour
        # pour _ongoing_epoch, c'est la valeur
        # pour le directory_path, c'est la représentation sous forme de string du chemin
        # pour le model, il faut utliser la méthode load() de RLmodel
        # pour l'environnemnt, demande moi dans le prompt
        representation = {
            "_ongoing_epoch": self.ongoing_epoch,
            "model": str(self.model) if self.model else None,
            "environment": f"gym.Env({self.model.environment.spec.id if self.model.environment and self.model.environment.spec else 'Unknown'})"
            if self.environment
            else None,
        }
        return json.dumps(representation, indent=2, ensure_ascii=False)

    def save(self, directory_path: Path):
        # écrit la représentation dans un fichier json à l'emplacement {directory_path}/run_info.json
        run_info_file = directory_path / "run_info.json"

        # Créer le répertoire s'il n'existe pas
        directory_path.mkdir(parents=True, exist_ok=True)

        # Préparer les données à sauvegarder
        run_data = {
            "directory_path": str(self.directory_path),
            "_ongoing_epoch": self.ongoing_epoch,
            "learning_metrics": [metric.model_dump() for metric in self.learning_metrics],
        }

        # Écrire le fichier run_info.json
        with open(run_info_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Runner state saved to {run_info_file}")

    def configure(self, model: RLModel, environment: gym.Env):
        self.model = model
        self.environment = environment

    def learn(self, max_epoch: int = 1000, save_every_n_epoch: int = 10):
        if self.model is None or self.environment is None:
            raise ValueError("Model and environment must be configured before learning")

        for episode in range(self.ongoing_epoch, max_epoch):
            epoch_result = self.model.run_epoch(train_mode=True)
            self.learning_metrics.append(epoch_result)
            self.ongoing_epoch += 1
            if self.ongoing_epoch % save_every_n_epoch == 0:
                self.save(self.directory_path)
                self.model.save(self.directory_path)


def save_config_summary(config: HerculeConfig, output_dir: Path) -> None:
    """
    Save configuration summary to a text file.

    Args:
        config: Hercule configuration to save
        output_dir: Directory where to save the summary
    """
    summary_file = output_dir / "config_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(str(config))

    logger.info(f"Configuration summary saved to {summary_file}")


def generate_filename_suffix(
    hyperparameters: dict[str, ParameterValue], environment_hyperparameters: dict[str, ParameterValue] | None = None
) -> str:
    """
    Generate a descriptive filename suffix from hyperparameters.

    Args:
        hyperparameters: Dictionary of hyperparameters and their values
        environment_hyperparameters: Dictionary of environment hyperparameters and their values

    Returns:
        String suffix to append to filename
    """
    all_params = {}

    # Add model hyperparameters (with shorter prefixes)
    if hyperparameters:
        for key, value in hyperparameters.items():
            # Use shorter prefixes for common parameters
            if key == "learning_rate":
                all_params["lr"] = value
            elif key == "discount_factor":
                all_params["df"] = value
            elif key == "epsilon":
                all_params["eps"] = value
            elif key == "epsilon_decay":
                all_params["epd"] = value
            elif key == "epsilon_min":
                all_params["epm"] = value
            elif key == "seed":
                all_params["s"] = value
            else:
                all_params[f"m_{key}"] = value

    # Add environment hyperparameters (with shorter prefixes)
    if environment_hyperparameters:
        for key, value in environment_hyperparameters.items():
            if key == "map_name":
                all_params["map"] = value
            elif key == "is_slippery":
                all_params["slip"] = value
            else:
                all_params[f"e_{key}"] = value

    if not all_params:
        return ""

    # Convert hyperparameters to a sorted list of key-value pairs for consistency
    param_pairs = []
    for key, value in sorted(all_params.items()):
        # Skip None values and empty strings
        if value is not None and str(value).strip():
            # Format the value appropriately
            if isinstance(value, float):
                formatted_value = f"{value:.3f}".rstrip("0").rstrip(".")
            else:
                formatted_value = str(value)
            param_pairs.append(f"{key}_{formatted_value}")

    if not param_pairs:
        return ""

    return "_" + "_".join(param_pairs)


class TrainingProtocol(Protocol):
    """Protocol defining the interface for RL models."""

    def train(self, env: gym.Env, config: dict[str, ParameterValue], max_iterations: int) -> dict[str, ParameterValue]:
        """
        Train the model on the given environment.

        Args:
            env: Gymnasium environment
            config: Model hyperparameters
            max_iterations: Maximum number of training iterations

        Returns:
            Training results and metrics
        """
        ...

    def act(self, observation: np.ndarray | int, training: bool = False) -> int | float | np.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation
            training: Whether the model is in training mode

        Returns:
            Selected action (int for discrete, float/array for continuous)
        """
        ...

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        ...


class RunResult:
    """Container for training run results."""

    def __init__(
        self,
        environment_name: str,
        model_name: str,
        hyperparameters: dict[str, ParameterValue],
        metrics: dict[str, ParameterValue],
        success: bool = True,
        error_message: str | None = None,
        environment_hyperparameters: dict[str, ParameterValue] | None = None,
    ) -> None:
        """
        Initialize run result.

        Args:
            environment_name: Name of the environment used
            model_name: Name of the model used
            hyperparameters: Hyperparameters used for training
            metrics: Training metrics and results
            success: Whether the run was successful
            error_message: Error message if run failed
            environment_hyperparameters: Hyperparameters of the environment
        """
        self.environment_name = environment_name
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.success = success
        self.error_message = error_message
        self.environment_hyperparameters = environment_hyperparameters or {}

    def to_dict(self) -> dict[str, ParameterValue | dict[str, ParameterValue] | None]:
        """Convert result to dictionary for serialization."""
        return {
            "environment_name": self.environment_name,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "environment_hyperparameters": self.environment_hyperparameters,
            "metrics": self.metrics,
            "success": self.success,
            "error_message": self.error_message,
        }


class TrainingRunner:
    """Handles training execution across models and environments."""

    def __init__(self, config: HerculeConfig):
        """
        Initialize the training runner.

        Args:
            config: Hercule configuration
        """
        self.config = config
        self.env_manager = EnvironmentManager(config)

    def run(self, model: RLModel, env: gym.Env, maximum_episode_number: int = 1000, train_mode: bool = False):
        episode_counter = 0
        epochs_results: list[EpochResult] = []

        while episode_counter < maximum_episode_number:
            model.run_epoch()
        pass

    def run_single_training_with_env_hyperparams(
        self,
        model: RLModel,
        environment_name: str,
        model_name: str,
        hyperparameters: dict[str, ParameterValue],
        environment_hyperparameters: dict[str, ParameterValue] | None = None,
        models_dir: Path | None = None,
    ) -> tuple[RLModel | None, RunResult]:
        """
        Run training for a single model-environment combination with explicit environment hyperparameters.

        Args:
            model: Model implementing TrainingProtocol
            environment_name: Name of the environment
            model_name: Name of the model
            hyperparameters: Model hyperparameters
            environment_hyperparameters: Environment hyperparameters
            models_dir: Directory to save the trained model

        Returns:
            RunResult containing training model or None (if failed), and metrics
        """
        import logging

        training_logger = logging.getLogger("hercule.training")

        try:
            # Load environment with explicit hyperparameters
            env = self.env_manager.load_environment_with_hyperparams(
                environment_name, environment_hyperparameters or {}
            )
            training_logger.info(f"Environment {environment_name} loaded successfully")

            # Run training
            training_logger.info(f"Starting training with hyperparameters: {hyperparameters}")
            metrics = model.train(env, hyperparameters, self.config.learn_max_epoch)
            training_logger.info(f"Training completed with metrics: {metrics}")

            # Save the trained model if training was successful and models_dir is provided
            if models_dir:
                try:
                    # Generate a descriptive filename for the model
                    param_suffix = generate_filename_suffix(hyperparameters, environment_hyperparameters)
                    model_filename = f"{environment_name}_{model_name}{param_suffix}"
                    model_path = models_dir / model_filename

                    # Save the model with environment and hyperparameter information
                    # if hasattr(model, "save"):
                    #     model.save(
                    #         path=model_path,
                    #         environment_name=environment_name,
                    #         environment_hyperparameters=environment_hyperparameters or {},
                    #         model_hyperparameters=hyperparameters,
                    #     )
                    #     training_logger.info(f"Model saved to {model_path}")
                    # else:
                    #     training_logger.warning(f"Model {model_name} does not implement save method")
                except Exception as e:
                    training_logger.error(f"Failed to save model: {e}")

            return (
                model,
                RunResult(
                    environment_name=environment_name,
                    model_name=model_name,
                    hyperparameters=hyperparameters,
                    metrics=metrics,
                    success=True,
                    environment_hyperparameters=environment_hyperparameters or {},
                ),
            )
        except Exception as e:
            training_logger.error(f"Training failed for {model_name} on {environment_name}: {e}")
            return (
                None,
                RunResult(
                    environment_name=environment_name,
                    model_name=model_name,
                    hyperparameters=hyperparameters,
                    metrics={},
                    success=False,
                    error_message=str(e),
                    environment_hyperparameters=environment_hyperparameters or {},
                ),
            )

    def validate_configuration(self) -> bool:
        """
        Validate that all configured environments can be loaded.

        Returns:
            True if all environments are valid, False otherwise
        """
        try:
            self.env_manager.validate_environments()
            return True
        except ValueError as e:
            logger.error(f"Environment validation failed: {e}")
            return False

    def close(self) -> None:
        """Close all resources."""
        self.env_manager.close_all()

    def __enter__(self) -> "TrainingRunner":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class RunManager:
    """Combines training and evaluation functionality."""

    def __init__(self, config: HerculeConfig, log_level: str = "INFO"):
        """
        Initialize the run manager.

        Args:
            config: Hercule configuration
            log_level: Logging level for evaluation (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = config
        self.env_manager = EnvironmentManager(config)
        self.training_runner = TrainingRunner(config)
        self.evaluator = ModelEvaluator(self.env_manager, log_level)

        # Setup separate logging for training and evaluation
        self._setup_logging()

    def _setup_logging(self):
        """Setup separate logging for training and evaluation."""
        import logging
        from pathlib import Path

        # Create log directories if they don't exist
        log_base_dir = self.config.output_dir / "logs"
        training_log_dir = log_base_dir / "training"
        evaluation_log_dir = log_base_dir / "evaluation"

        training_log_dir.mkdir(parents=True, exist_ok=True)
        evaluation_log_dir.mkdir(parents=True, exist_ok=True)

        # Setup training logger
        training_logger = logging.getLogger("hercule.training")
        training_logger.handlers.clear()
        training_handler = logging.FileHandler(training_log_dir / "training.log", mode="w")
        training_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        training_logger.addHandler(training_handler)
        training_logger.setLevel(logging.INFO)
        training_logger.propagate = False

        # Setup evaluation logger
        evaluation_logger = logging.getLogger("hercule.evaluation")
        evaluation_logger.handlers.clear()
        evaluation_handler = logging.FileHandler(evaluation_log_dir / "evaluation.log", mode="w")
        evaluation_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        evaluation_logger.addHandler(evaluation_handler)
        evaluation_logger.setLevel(logging.INFO)
        evaluation_logger.propagate = False

        # Test that logging is working
        training_logger.info("Training logger initialized")
        evaluation_logger.info("Evaluation logger initialized")

    def run_training_and_evaluation(
        self,
        model: RLModel,
        environment_name: str,
        model_name: str,
        hyperparameters: dict[str, ParameterValue],
        evaluation_config: dict[str, ParameterValue] | None = None,
        environment_hyperparameters: dict[str, ParameterValue] | None = None,
        models_dir: Path | None = None,
    ) -> tuple[RLModel, EvaluationResult | None] | None:
        """
        Run training followed by evaluation.

        Args:
            model: Model implementing TrainingProtocol
            environment_name: Name of the environment
            model_name: Name of the model
            hyperparameters: Model hyperparameters
            evaluation_config: Evaluation configuration (optional)
            environment_hyperparameters: Environment hyperparameters
            models_dir: Directory to save the trained model

        Returns:
            Tuple of (training_result, evaluation_result)
        """
        import logging

        evaluation_logger = logging.getLogger("hercule.evaluation")

        # Run training first
        trained_model = self.training_runner.run_single_training_with_env_hyperparams(
            model=model,
            environment_name=environment_name,
            model_name=model_name,
            hyperparameters=hyperparameters,
            environment_hyperparameters=environment_hyperparameters,
            models_dir=models_dir,
        )

        # Run evaluation if training was successful and evaluation config provided
        evaluation_result = None
        if trained_model[0] is not None and evaluation_config:
            try:
                evaluation_logger.info(f"Starting evaluation for {model_name} on {environment_name}")

                num_episodes = evaluation_config.get("num_episodes", 10)
                max_steps = evaluation_config.get("max_steps_per_episode")
                render = evaluation_config.get("render", False)

                # Convert ParameterValue types to appropriate types
                num_episodes_int = int(num_episodes) if isinstance(num_episodes, int | float) else 10
                max_steps_int = int(max_steps) if isinstance(max_steps, int | float) else None
                render_bool = bool(render) if isinstance(render, bool) else False

                evaluation_result = self.evaluator.evaluate_model_with_hyperparams(
                    model=trained_model[0],
                    environment_name=environment_name,
                    model_name=model_name,
                    environment_hyperparameters=environment_hyperparameters,
                    num_episodes=num_episodes_int,
                    max_steps_per_episode=max_steps_int,
                    render=render_bool,
                )

                if evaluation_result.success:
                    evaluation_logger.info(f"Evaluation completed successfully for {model_name} on {environment_name}")
                else:
                    evaluation_logger.error(
                        f"Evaluation failed for {model_name} on {environment_name}: {evaluation_result.error_message}"
                    )

            except Exception as e:
                evaluation_logger.error(f"Evaluation failed after successful training: {e}")

            return trained_model[0], evaluation_result

    def validate_configuration(self) -> bool:
        """
        Validate that all configured environments can be loaded.

        Returns:
            True if all environments are valid, False otherwise
        """
        return self.training_runner.validate_configuration()

    def close(self) -> None:
        """Close all resources."""
        self.env_manager.close_all()

    def __enter__(self) -> "RunManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def create_output_directory(config: HerculeConfig) -> Path:
    """
    Create output directory structure with timestamp-based subdirectory.

    Args:
        config: Hercule configuration

    Returns:
        Path to the created output directory (timestamp-based)
    """
    from datetime import datetime

    # Create main output directory (directly using config.output_dir)
    base_output_dir = config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_output_dir = base_output_dir / timestamp
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Create simplified subdirectories
    subdirs = ["models", "logs/training", "logs/evaluation", "training", "evaluation"]
    for subdir in subdirs:
        (config_output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory created: {config_output_dir}")
    return config_output_dir


__all__ = [
    "TrainingProtocol",
    "RunResult",
    "TrainingRunner",
    "RunManager",
    "create_output_directory",
    "save_config_summary",
    "generate_filename_suffix",
    "EvaluationProtocol",
    "EvaluationResult",
    "ModelEvaluator",
    "save_evaluation_results",
]
