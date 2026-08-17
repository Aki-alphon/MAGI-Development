from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DataIngestionArtifact:
    train_dir: str
    val_dir: str
    test_dir: str
    class_names: List[str]
    class_distribution: Dict[str, int]
    total_images: int


@dataclass
class DataValidationArtifact:
    is_valid: bool
    message: str
    validation_report_path: str
    corrupt_images: List[str]
    class_distribution: Dict[str, int]


@dataclass
class DataTransformationArtifact:
    normalization_stats_path: str
    transform_config_path: str
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    train_npy_dir: str
    val_npy_dir: str
    test_npy_dir: str


@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    per_class_f1: Dict[str, float]
    confusion_matrix_path: Optional[str] = None


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    tflite_model_path: str
    training_history_path: str
    class_mapping_path: str
    metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    keras_accuracy: float
    tflite_accuracy: float
    tflite_latency_ms: float
    model_size_mb: float
    evaluation_report_path: str


@dataclass
class ModelPusherArtifact:
    deployed_model_path: str
    manifest_path: str
    model_version: str
