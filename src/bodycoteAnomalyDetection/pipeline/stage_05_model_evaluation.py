from bodycoteAnomalyDetection.config.configuration import ConfigurationManager
from bodycoteAnomalyDetection.components.model_evaluation import ModelEvaluation
from bodycoteAnomalyDetection.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config=model_eval_config)
        model_evaluator.evaluate()