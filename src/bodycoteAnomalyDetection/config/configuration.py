from bodycoteAnomalyDetection.constants import *
from bodycoteAnomalyDetection.utils.common import read_yaml, create_directories
from bodycoteAnomalyDetection.entity.config_entity import (DataIngestionConfig,
                                                           DataValidationConfig,
                                                           DataTransformationConfig,
                                                           ModelTrainerConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            learning_rate_range=params.learning_rate_range,
            batch_size_range=params.batch_size_range,
            epochs_range=params.epochs_range,
            encoder_units_1_range=params.encoder_units_1_range,
            encoder_units_2_range=params.encoder_units_2_range,
            max_trials=params.max_trials,
            executions_per_trial=params.executions_per_trial,
            training_epochs=params.training_epochs,
            training_validation_split=params.training_validation_split,
            training_batch_size=params.training_batch_size,
            num_trials=params.num_trials,
            early_stopping_patience=params.early_stopping_patience,
            restore_best_weights=params.restore_best_weights,
            final_epochs=params.final_epochs,
            final_batch_size=params.final_batch_size,
            final_validation_split=params.final_validation_split
        )