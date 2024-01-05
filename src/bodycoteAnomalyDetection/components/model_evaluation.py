import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
from bodycoteAnomalyDetection.components import model_trainer
from bodycoteAnomalyDetection.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.trainer = model_trainer.ModelTrainer(config)
        self.model = self.load_model()
        self.data = self.load_data()
        

    def load_model(self):
        return load_model(self.config.model_path)

    def load_data(self):
        df = self.trainer.load_data()
        scaler = joblib.load(self.config.scaler_path)
        X_scaled = scaler.transform(df)
        X_reshaped = self.trainer.data_reshaper(X_scaled)
        return X_reshaped

    def evaluate(self):
        predictions = self.model.predict(self.data)
        data_flat = self.trainer.data_flattening(self.data)
        predictions_flat = self.trainer.data_flattening(predictions)
        mse = mean_squared_error(data_flat, predictions_flat)
        mae = mean_absolute_error(data_flat, predictions_flat)
        r2 = r2_score(data_flat, predictions_flat)

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2
        }
        with open(self.config.metric_file_name, 'w') as metric_file:
            json.dump(metrics, metric_file)