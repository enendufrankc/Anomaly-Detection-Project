from bodycoteAnomalyDetection.config.configuration import ConfigurationManager

from bodycoteAnomalyDetection.components.model_trainer import ModelTrainer
from src.bodycoteAnomalyDetection.logging import logger



STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()