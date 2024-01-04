from bodycoteAnomalyDetection.config.configuration import ConfigurationManager

from bodycoteAnomalyDetection.components.data_transformation import DataTransformation
from src.bodycoteAnomalyDetection.logging import logger



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.process_all_csv()