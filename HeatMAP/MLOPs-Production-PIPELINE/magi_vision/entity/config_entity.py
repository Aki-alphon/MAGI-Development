import os
from magi_vision.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
    )
    raw_data_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_RAW_DATA_DIR)
    train_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_TRAIN_DIR)
    val_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_VAL_DIR)
    test_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_TEST_DIR)
    train_split: float = DATA_INGESTION_TRAIN_SPLIT
    val_split: float = DATA_INGESTION_VAL_SPLIT
    test_split: float = DATA_INGESTION_TEST_SPLIT


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )
    validation_report_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_REPORT_FILE
    )
    min_samples_per_class: int = DATA_VALIDATION_MIN_SAMPLES_PER_CLASS
    max_imbalance_ratio: float = DATA_VALIDATION_MAX_IMBALANCE_RATIO
    supported_formats: list = None
    min_image_size: tuple = DATA_VALIDATION_MIN_IMAGE_SIZE

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = DATA_VALIDATION_SUPPORTED_FORMATS


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
    )
    transform_config_dir: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_CONFIG_DIR
    )
    normalization_stats_path: str = os.path.join(
        transform_config_dir, DATA_TRANSFORMATION_NORM_STATS_FILE
    )
    image_size: tuple = IMAGE_SIZE
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_grid_size: tuple = CLAHE_GRID_SIZE
    imagenet_mean: list = None
    imagenet_std: list = None
    horizontal_flip: bool = AUGMENTATION_HORIZONTAL_FLIP
    rotation_range: float = AUGMENTATION_ROTATION_RANGE
    brightness_range: tuple = AUGMENTATION_BRIGHTNESS_RANGE
    contrast_range: tuple = AUGMENTATION_CONTRAST_RANGE
    zoom_range: tuple = AUGMENTATION_ZOOM_RANGE
    channel_dropout_rate: float = AUGMENTATION_CHANNEL_DROPOUT_RATE
    noise_std: float = AUGMENTATION_NOISE_STD

    def __post_init__(self):
        if self.imagenet_mean is None:
            self.imagenet_mean = IMAGENET_MEAN
        if self.imagenet_std is None:
            self.imagenet_std = IMAGENET_STD


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
    )
    trained_model_dir: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR
    )
    trained_model_path: str = os.path.join(trained_model_dir, MODEL_TRAINER_MODEL_NAME)
    tflite_model_path: str = os.path.join(trained_model_dir, MODEL_TRAINER_TFLITE_NAME)
    training_history_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_HISTORY_FILE)
    class_mapping_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_CLASS_MAP_FILE)
    input_channels: int = INPUT_CHANNELS
    image_size: tuple = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    class_names: list = None
    phase1_epochs: int = PHASE1_EPOCHS
    phase1_lr: float = PHASE1_LR
    phase2_epochs: int = PHASE2_EPOCHS
    phase2_lr: float = PHASE2_LR
    unfreeze_from_layer: int = UNFREEZE_FROM_LAYER
    dense_units: int = DENSE_UNITS
    dropout_rate: float = DROPOUT_RATE
    batch_size: int = BATCH_SIZE
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_ACCURACY
    model_config_file_path: str = MODEL_CONFIG_FILE_PATH

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = CLASS_NAMES


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
    )
    evaluation_report_path: str = os.path.join(
        model_evaluation_dir, MODEL_EVALUATION_REPORT_FILE
    )
    confusion_matrix_path: str = os.path.join(
        model_evaluation_dir, MODEL_EVALUATION_CONFUSION_MATRIX_FILE
    )
    changed_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD
    tflite_accuracy_retention: float = MODEL_EVALUATION_TFLITE_ACCURACY_RETENTION
    max_latency_ms: float = MODEL_EVALUATION_MAX_LATENCY_MS
    max_model_size_mb: float = MODEL_EVALUATION_MAX_MODEL_SIZE_MB


@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
    )
    tflite_export_dir: str = os.path.join(model_pusher_dir, TFLITE_EXPORT_DIR)
    deploy_path: str = DEPLOY_PATH
    manifest_path: str = os.path.join(model_pusher_dir, DEPLOYMENT_MANIFEST_FILE)


@dataclass
class MAGIPredictorConfig:
    tflite_model_path: str = os.path.join(TFLITE_EXPORT_DIR, MODEL_TRAINER_TFLITE_NAME)
    class_mapping_path: str = os.path.join(TFLITE_EXPORT_DIR, MODEL_TRAINER_CLASS_MAP_FILE)
    normalization_stats_path: str = os.path.join(TFLITE_EXPORT_DIR, DATA_TRANSFORMATION_NORM_STATS_FILE)
