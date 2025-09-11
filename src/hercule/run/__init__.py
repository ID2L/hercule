"""Run module for training execution and result management."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hercule.config import HerculeConfig, ParameterValue
from hercule.environnements import EnvironmentManager
from hercule.models import RLModel
from hercule.models.epoch_result import EpochResult


# Type checking imports
if TYPE_CHECKING:
    from typing import Any

    ModelType = RLModel
else:
    from typing import Any

    ModelType = Any


logger = logging.getLogger(__name__)


class Runner(BaseModel):
    """
    Runner class for training execution and result management.
    """

    learning_ongoing_epoch: int = Field(default=0, description="")
    testing_ongoing_epoch: int = Field(default=0, description="")
    learning_metrics: list[EpochResult] = Field(default=[], description="")
    testing_metrics: list[EpochResult] = Field(default=[], description="")
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
            return Runner(learning_ongoing_epoch=0, testing_ongoing_epoch=0, directory_path=directory_path)

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
        # Extract environment representation logic
        env_id = "Unknown"
        if self.environment and hasattr(self.environment, "spec") and self.environment.spec:
            env_id = self.environment.spec.id

        representation = {
            "learning_ongoing_epoch": self.learning_ongoing_epoch,
            "testing_ongoing_epoch": self.testing_ongoing_epoch,
            "model": str(self.model) if self.model else None,
            "environment": f"gym.Env({env_id})" if self.environment else None,
        }
        return json.dumps(representation, indent=2, ensure_ascii=False)

    def save(self, directory_path: Path):
        # écrit la représentation dans un fichier json à l'emplacement {directory_path}/run_info.json
        run_info_file = directory_path / "run_info.json"

        # Créer le répertoire s'il n'existe pas
        directory_path.mkdir(parents=True, exist_ok=True)

        # Préparer les données à sauvegarder
        run_data = {
            "learning_ongoing_epoch": self.learning_ongoing_epoch,
            "testing_ongoing_epoch": self.testing_ongoing_epoch,
            "learning_metrics": [metric.model_dump() for metric in self.learning_metrics],
            "testing_metrics": [metric.model_dump() for metric in self.testing_metrics],
        }

        # Écrire le fichier run_info.json
        with open(run_info_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Runner state saved to {run_info_file}")

    def configure(self, model: RLModel, environment: gym.Env):
        self.model = model
        self.environment = environment

    def learn(self, max_epoch: int = 1000, save_every_n_epoch: int = None):
        if self.model is None or self.environment is None:
            raise ValueError("Model and environment must be configured before learning")

        if self.learning_ongoing_epoch == max_epoch:
            logger.info("Max learning epoch already reached")
            return
        for _ in range(self.learning_ongoing_epoch, max_epoch):
            epoch_result = self.model.run_epoch(train_mode=True)
            self.learning_metrics.append(epoch_result)
            self.learning_ongoing_epoch += 1
            if save_every_n_epoch is not None and self.learning_ongoing_epoch % save_every_n_epoch == 0:
                self.save(self.directory_path)
                self.model.save(self.directory_path)

        self.save(self.directory_path)
        self.model.save(self.directory_path)

    def test(self, max_epoch: int = 1000):
        if self.model is None or self.environment is None:
            raise ValueError("Model and environment must be configured before testing")

        for _ in range(self.testing_ongoing_epoch, max_epoch):
            epoch_result = self.model.run_epoch(train_mode=False)
            self.testing_metrics.append(epoch_result)
            self.testing_ongoing_epoch += 1

        self.save(self.directory_path)


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


__all__ = ["TrainingProtocol", "create_output_directory", "save_config_summary", "generate_filename_suffix"]
