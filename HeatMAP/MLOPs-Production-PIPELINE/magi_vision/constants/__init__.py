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
INPUT_CHANNELS: int = 8  # R, G, B, ExG, GRVI, VARI, GLI, NGBDI
NUM_CLASSES: int = 4

# Stress progression stage labels (maps to temporal distance from visible disease)
# Replaces generic healthy/mild/moderate/severe
CLASS_NAMES: list = [
    "baseline_healthy",        # >7 days  — normal nitrogen uptake
    "early_nitrogen_stress",   # 3-5 days — chlorophyll synthesis dropping, no visible change
    "active_chlorosis",        # 1-2 days — yellowing beginning on older leaves
    "severe_deficiency",       # Now      — visible disease, tissue damage
]
# Health score weights per class (1.0 = fully healthy, 0.0 = severe)
HEALTH_SCORES: list = [1.0, 0.68, 0.35, 0.05]

# =============================================================================
# SPECTRAL MASKING PREPROCESSING
# =============================================================================
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_GRID_SIZE: tuple = (8, 8)
IMAGENET_MEAN: list = [0.485, 0.456, 0.406]
IMAGENET_STD: list = [0.229, 0.224, 0.225]

# 8-channel stack order:
# 0:R 1:G 2:B — ImageNet-normalised RGB
# 3:ExG       — Excess Green (general vegetation)
# 4:GRVI      — Green-Red Vegetation Index (chlorophyll proxy)
# 5:VARI      — Visible Atmospherically Resistant Index (best for canopy N-stress)
# 6:GLI       — Green Leaf Index (nitrogen proxy, canopy-level)
# 7:NGBDI     — Normalised Green-Blue Difference (earliest N-deficiency RGB signal)

# L* dropped: under-canopy lighting is too variable — adds noise, not signal

# =============================================================================
# CANOPY GRID-TILE ANALYSIS
# =============================================================================
GRID_ROWS: int = 3
GRID_COLS: int = 4
SOIL_EXG_THRESHOLD: float = 10.0   # Lowered from 30.0 — under-canopy has less sky, more vegetation
MIN_VEGETATION_FRACTION: float = 0.2  # Under-canopy: even 20% plant pixels is meaningful
EMA_ALPHA: float = 0.5              # Faster temporal response vs 0.3 (bot moves fast)
SPATIAL_CELL_SIZE_M: float = 1.0
MIN_OBSERVATIONS_PER_CELL: int = 3

# Trend detection for EARLY_WARNING (pre-symptomatic 3-4 day prediction)
TREND_WINDOW_FRAMES: int = 5         # Number of frames for slope calculation
TREND_DECLINE_THRESHOLD: float = -0.015  # Min slope (per frame) to flag declining health
TREND_MIN_CONSECUTIVE: int = 3       # Consecutive declining frames before EARLY_WARNING
EARLY_WARNING_MIN_HEALTH: float = 0.65  # Don’t fire EARLY_WARNING if already in ALERT territory

# =============================================================================
# HEALTH THRESHOLDS (for Lugia decision engine)
# =============================================================================
HEALTH_THRESHOLD_HEALTHY: float    = 0.85  # Above this → IDLE
HEALTH_THRESHOLD_MONITOR: float    = 0.70  # Above this (and stable) → MONITOR
HEALTH_THRESHOLD_MILD: float       = 0.50  # Above this → ALERT (active chlorosis)
HEALTH_THRESHOLD_SEVERE: float     = 0.30  # Below this → ANALYZE

# =============================================================================
# DATASET CONFIGURATION (3-source strategy)
# =============================================================================
# Source A: Cotton stress (class remapping to stress stages)
KAGGLE_COTTON_DATASET: str = "dhamur/cotton-plant-disease"
COTTON_CLASS_REMAP: dict = {
    "Healthy":           "baseline_healthy",
    "Aphids":            "early_nitrogen_stress",  # Aphids target nitrogen-stressed plants
    "Army worm":         "active_chlorosis",
    "Bacterial Blight":  "severe_deficiency",
}
# Source B: Lettuce/leafy veg NPK deficiency
KAGGLE_LEAFY_DATASET: str = "rayhanadev/plant-leaf-health"
LEAFY_CLASS_REMAP: dict = {
    "Healthy":           "baseline_healthy",
    "Nitrogen":          "early_nitrogen_stress",
    "Phosphorus":        "active_chlorosis",    # P deficiency mimics N visually
    "Potassium":         "active_chlorosis",
    "Diseased":          "severe_deficiency",
}
# Source C: EarlyNSD — only source with pre-symptomatic labels
KAGGLE_EARLYNSD_DATASET: str = "gauravduttakiit/earlynsd"
EARLYNSD_CLASSES_FOR_EARLY_STRESS: list = ["early_nitrogen", "nitrogen_early", "N_early"]

MAX_SAMPLES_PER_CLASS: int = 1200   # Reduced from 1500 — smaller dataset → smaller TFRecords
UNDER_CANOPY_SYNTHETIC_COUNT: int = 500  # Reduced from 800 to keep total sample count manageable

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE: int = 64
                               # buffer fill + val set load. 64 halves per-step memory.
                               # GPU util is still excellent at 64 with 8-ch f32 tensors.
PHASE1_EPOCHS: int = 12        # Head-only training; early stopping fires at ~8–10
PHASE2_EPOCHS: int = 20        # Fine-tune; stays well within Colab's ~90-min session limit
PHASE1_LR: float = 1e-3
PHASE2_LR: float = 1e-5
UNFREEZE_FROM_LAYER: int = 100  # Unfreeze top ~55 layers of MobileNetV2 backbone
DROPOUT_RATE: float = 0.4
DENSE_UNITS: int = 256
FOCAL_GAMMA: float = 2.0
EARLY_STOPPING_PATIENCE: int = 6   # Slightly more patience — cosine LR needs room
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-7

# XLA (Accelerated Linear Algebra) JIT compilation — ~15–20% throughput gain on T4
USE_XLA_JIT: bool = True

# tf.data pipeline workers — AUTOTUNE lets TF pick based on available cores
TF_DATA_NUM_WORKERS: int = -1  # -1 = tf.data.AUTOTUNE

# Steps per execution — amortises Python↔C++ overhead per batch.
# 10 is safe because the XLA graph is now compiled ONCE per phase and reused.
# The old value of 5 was a workaround for the per-epoch clear_session() which
# destroyed the compiled graph — that workaround has been removed.
STEPS_PER_EXECUTION: int = 10

# DEPRECATED — no longer used. The per-epoch save/load/clear_session chunking
# strategy has been replaced by a single model.fit() call per phase with Keras
# callbacks. Kept here only to avoid breaking any external imports.
TRAIN_CHUNK_EPOCHS: int = 5

# Prefetch buffer for tf.data.
# Fixed at 2 (not AUTOTUNE) — on free T4 AUTOTUNE pre-loads 8-16 batches of
# 224×224×8 float32 images = up to 1.5 GB system RAM, causing session crashes.
PREFETCH_BUFFER: int = 2

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
MODEL_TRAINER_MODEL_NAME: str = "celebi.keras"
MODEL_TRAINER_TFLITE_NAME: str = "celebi.tflite"
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
