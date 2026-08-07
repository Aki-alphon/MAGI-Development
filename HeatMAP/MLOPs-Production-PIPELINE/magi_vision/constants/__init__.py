import os
from datetime import date

# =============================================================================
# PROJECT IDENTITY
# =============================================================================
PIPELINE_NAME: str = "magi_vision"
ARTIFACT_DIR: str = "artifact"
PROJECT_NAME: str = "MAGI_VISION"

# =============================================================================
# IMAGE & MODEL ARCHITECTURE
# =============================================================================
IMAGE_SIZE: tuple = (224, 224)
INPUT_CHANNELS: int = 6  # R, G, B, ExG, GRVI, L*
NUM_CLASSES: int = 4
CLASS_NAMES: list = ["healthy", "mild_stress", "moderate_stress", "severe_stress"]
HEALTH_SCORES: list = [1.0, 0.7, 0.4, 0.15]

# =============================================================================
# SPECTRAL MASKING PREPROCESSING
# =============================================================================
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_GRID_SIZE: tuple = (8, 8)
IMAGENET_MEAN: list = [0.485, 0.456, 0.406]
IMAGENET_STD: list = [0.229, 0.224, 0.225]

# =============================================================================
# CANOPY GRID-TILE ANALYSIS
# =============================================================================
GRID_ROWS: int = 3
GRID_COLS: int = 4
SOIL_EXG_THRESHOLD: float = 30.0
MIN_VEGETATION_FRACTION: float = 0.3
EMA_ALPHA: float = 0.3
SPATIAL_CELL_SIZE_M: float = 1.0
MIN_OBSERVATIONS_PER_CELL: int = 3

# =============================================================================
# HEALTH THRESHOLDS (for Caspar decision engine)
# =============================================================================
HEALTH_THRESHOLD_HEALTHY: float = 0.8
HEALTH_THRESHOLD_MILD: float = 0.6
HEALTH_THRESHOLD_MODERATE: float = 0.4

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE: int = 32
PHASE1_EPOCHS: int = 15
PHASE2_EPOCHS: int = 30
PHASE1_LR: float = 1e-3
PHASE2_LR: float = 1e-5
UNFREEZE_FROM_LAYER: int = 100
DROPOUT_RATE: float = 0.4
DENSE_UNITS: int = 256
FOCAL_GAMMA: float = 2.0
EARLY_STOPPING_PATIENCE: int = 7
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-7

# =============================================================================
# DATA INGESTION
# =============================================================================
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR: str = "raw_data"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_VAL_DIR: str = "val"
DATA_INGESTION_TEST_DIR: str = "test"
DATA_INGESTION_TRAIN_SPLIT: float = 0.7
DATA_INGESTION_VAL_SPLIT: float = 0.15
DATA_INGESTION_TEST_SPLIT: float = 0.15

# =============================================================================
# DATA VALIDATION
# =============================================================================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE: str = "validation_report.yaml"
DATA_VALIDATION_MIN_SAMPLES_PER_CLASS: int = 50
DATA_VALIDATION_MAX_IMBALANCE_RATIO: float = 10.0
DATA_VALIDATION_SUPPORTED_FORMATS: list = ["jpg", "jpeg", "png", "bmp"]
DATA_VALIDATION_MIN_IMAGE_SIZE: tuple = (64, 64)

# =============================================================================
# DATA TRANSFORMATION
# =============================================================================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_CONFIG_DIR: str = "transform_config"
DATA_TRANSFORMATION_NORM_STATS_FILE: str = "normalization_stats.json"

# Augmentation defaults
AUGMENTATION_HORIZONTAL_FLIP: bool = True
AUGMENTATION_ROTATION_RANGE: float = 15.0
AUGMENTATION_BRIGHTNESS_RANGE: tuple = (0.7, 1.3)
AUGMENTATION_CONTRAST_RANGE: tuple = (0.8, 1.2)
AUGMENTATION_ZOOM_RANGE: tuple = (0.85, 1.15)
AUGMENTATION_CHANNEL_DROPOUT_RATE: float = 0.1
AUGMENTATION_NOISE_STD: float = 0.02

# =============================================================================
# MODEL TRAINER
# =============================================================================
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_MODEL_NAME: str = "melchior.keras"
MODEL_TRAINER_TFLITE_NAME: str = "melchior.tflite"
MODEL_TRAINER_HISTORY_FILE: str = "training_history.json"
MODEL_TRAINER_CLASS_MAP_FILE: str = "class_mapping.json"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.85
MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# =============================================================================
# MODEL EVALUATION
# =============================================================================
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE: str = "evaluation_report.yaml"
MODEL_EVALUATION_CONFUSION_MATRIX_FILE: str = "confusion_matrix.png"
MODEL_EVALUATION_CHANGED_THRESHOLD: float = 0.02
MODEL_EVALUATION_TFLITE_ACCURACY_RETENTION: float = 0.98
MODEL_EVALUATION_MAX_LATENCY_MS: float = 200.0
MODEL_EVALUATION_MAX_MODEL_SIZE_MB: float = 10.0

# =============================================================================
# MODEL PUSHER
# =============================================================================
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
TFLITE_EXPORT_DIR: str = "tflite_export"
DEPLOY_PATH: str = "/opt/magi/models/"
DEPLOYMENT_MANIFEST_FILE: str = "deployment_manifest.json"

# =============================================================================
# SCHEMA & CONFIG FILE PATHS
# =============================================================================
SCHEMA_CONFIG_FILE_PATH: str = os.path.join("config", "schema.yaml")

# =============================================================================
# APPLICATION
# =============================================================================
APP_HOST: str = "0.0.0.0"
APP_PORT: int = 8080
