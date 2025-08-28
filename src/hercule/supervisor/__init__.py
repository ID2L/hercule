import logging
from pathlib import Path

from pydantic import BaseModel, Field

from hercule.config import HerculeConfig, load_config_from_yaml
from hercule.environnements import EnvironmentFactory
from hercule.models import create_model
from hercule.run import Runner


logger = logging.getLogger(__name__)


class Supervisor(BaseModel):
    config: HerculeConfig

    def __init__(self, config: HerculeConfig):
        super().__init__(config=config)

    @classmethod
    def create_from_path(cls, path: Path):
        """
        Crée une instance de Supervisor à partir d'un fichier de configuration YAML.
        """
        if not path.is_file():
            raise ValueError(f"Le chemin fourni n'est pas un fichier valide: {path}")

        if path.suffix.lower() not in [".yaml", ".yml"]:
            raise ValueError(f"Le fichier doit être un fichier YAML (.yaml ou .yml): {path}")

        config = load_config_from_yaml(path)
        return cls(config=config)

    def execute_learn_phase(self):
        environment_factory = EnvironmentFactory()
        for environment_config in self.config.get_environment_configs():
            for model_config in self.config.models:
                directory = self.config.get_directory_for(model_config, environment_config)
                logger.info(f"Running learn phase for {model_config.name} on {environment_config.name} in {directory}")

                # Create directory if it doesn't exist
                directory.mkdir(parents=True, exist_ok=True)

                environment = environment_factory.get_or_create_environment(environment_config.name)
                model = create_model(model_config.name)
                model.configure(environment, model_config.get_hyperparameters_dict())
                model.load(directory)

                # Run training
                runner = Runner.load(directory)
                runner.configure(model, environment)

                runner.learn(self.config.learn_max_epoch)
