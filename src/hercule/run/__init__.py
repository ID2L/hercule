"""Run module for training execution and result management."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hercule.config import HerculeConfig, ParameterValue
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
run_info_file_name: Final = "run_info.json"


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
        run_info_file = directory_path / run_info_file_name

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
        run_info_file = directory_path / run_info_file_name

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


__all__ = ["TrainingProtocol", "save_config_summary"]
