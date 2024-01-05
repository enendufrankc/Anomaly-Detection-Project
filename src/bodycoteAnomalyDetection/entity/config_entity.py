from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    learning_rate_range: list
    batch_size_range: list
    epochs_range: list
    encoder_units_1_range: list
    encoder_units_2_range: list
    max_trials: int
    executions_per_trial: int
    training_epochs: int
    training_validation_split: float
    training_batch_size: int
    num_trials: int
    early_stopping_patience: int
    restore_best_weights: bool
    final_epochs: int
    final_batch_size: int
    final_validation_split: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    scaler_path: Path
    metric_file_name: Path