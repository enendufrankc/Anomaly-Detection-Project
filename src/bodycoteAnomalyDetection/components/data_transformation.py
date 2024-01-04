# Define all your imports at the beginning
import glob
import os
import pandas as pd
from bodycoteAnomalyDetection.logging import logger
# from datasets import load_dataset, load_from_disk
from bodycoteAnomalyDetection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def data_transformation(self, example_batch):
        try:
            example_batch['TimeStamp'] = pd.to_datetime(example_batch['TimeStamp'], format='%d/%m/%Y %H:%M:%S')
        except ValueError:
            example_batch['TimeStamp'] = pd.to_datetime(example_batch['TimeStamp'], infer_datetime_format=True)
        
        example_batch.drop_duplicates(subset='TimeStamp', keep='last', inplace=True)
        
        example_batch.set_index('TimeStamp', inplace=True)
        example_batch.sort_index(inplace=True)
        
        data_training = example_batch.select_dtypes(include=['float64', 'int64'])

        return example_batch, data_training
    
    def read_csv(self, file_path):
        return pd.read_csv(file_path)

    def convert(self, file_path):
        df = self.read_csv(file_path)
        clean_data, data_training = self.data_transformation(df)
        clean_data.to_csv(os.path.join(self.config.root_dir, 'clean_data.csv'))
        data_training.to_csv(os.path.join(self.config.root_dir, 'data_training.csv'))

    def process_all_csv(self):
        try:
            csv_files = glob.glob(os.path.join(self.config.data_path, '*.csv'))
            for file in csv_files:
                self.convert(file)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise