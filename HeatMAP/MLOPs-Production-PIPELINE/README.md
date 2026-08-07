# 🌿 MAGI Vision — Canopy-Level Plant Health Monitoring Pipeline

<div align="center">

**Multispectral Autonomous Ground Intelligence — Melchior Vision Subsystem**

*Continuous canopy scanning from a walking quadruped robot,  
producing spatial health heatmaps for precision agriculture.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat-square)
![TFLite](https://img.shields.io/badge/TFLite-XNNPACK-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=flat-square)
![Platform](https://img.shields.io/badge/Target-Raspberry%20Pi%204B-red?style=flat-square)

</div>

---

## 📋 Overview

MAGI Vision is a production MLOps training pipeline that produces TFLite models for real-time plant health detection on a Raspberry Pi 4B-powered quadruped robot. Unlike traditional leaf-level classification, this system **scans entire crop canopies continuously while the robot walks**, building a spatial health heatmap of the field.

### Key Features

| Feature | Description |
|---|---|
| **6-Channel Spectral Input** | RGB + computed vegetation indices (ExG, GRVI, L*) |
| **Grid-Tile Analysis** | Splits each frame into 4×3 grid, classifies vegetation tiles |
| **Soil Pre-Filter** | Skips soil tiles via Excess Green Index threshold (saves 30-50% compute) |
| **Two-Phase Training** | Frozen head (15 epochs) → fine-tune top layers (30 epochs) |
| **TFLite Export** | float16 quantization + XNNPACK for ~125ms inference on Pi 4B |
| **Temporal Smoothing** | EMA filter suppresses frame-to-frame flicker |
| **Spatial Health Grid** | Accumulates per-tile scores into GPS-tagged field heatmap |
| **FastAPI Interface** | REST API for training trigger, single-image prediction, canopy analysis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAGI VISION PIPELINE                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 1. Data      │───▶│ 2. Data      │───▶│ 3. Data      │       │
│  │ Ingestion    │    │ Validation   │    │ Transform    │       │
│  │              │    │              │    │              │       │
│  │ Download/    │    │ Integrity    │    │ CLAHE + ExG  │       │
│  │ Split dataset│    │ Class balance│    │ GRVI + L*    │       │
│  └──────────────┘    └──────────────┘    │ Norm stats   │       │
│                                          └──────┬───────┘       │
│                                                 │                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐       │
│  │ 6. Model     │◀───│ 5. Model     │◀───│ 4. Model     │       │
│  │ Pusher       │    │ Evaluation   │    │ Trainer      │       │
│  │              │    │              │    │              │       │
│  │ Deploy       │    │ TFLite bench │    │ MobileNetV2  │       │
│  │ TFLite + cfg │    │ Latency/size │    │ 6ch, 2-phase │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  Output: melchior.tflite + class_mapping.json +                  │
│          preprocessing_config.json + deployment_manifest.json    │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline (On Robot)

```
Camera (640×480 @ 30fps)
    │
    ▼ subsample (every 5th frame)
Grid Split (4×3 = 12 tiles)
    │
    ▼ ExG threshold
Vegetation Filter (skip soil tiles)
    │
    ▼ CLAHE + ExG + GRVI + L*
Spectral Masking (6ch tensors)
    │
    ▼ TFLite XNNPACK
Batch Inference (~125ms)
    │
    ▼ α·Xₜ + (1-α)·Sₜ₋₁
Temporal EMA Smoothing
    │
    ▼ GPS + odometry
Spatial Health Grid (1m × 1m cells)
    │
    ▼
Caspar Decision Engine
    IDLE (healthy) / TRACK (slow) / ALERT (flag) / ANALYZE (stop)
```

---

## 📁 Project Structure

```
MLOPs-Production-PIPELINE/
├── magi_vision/                    # Main package
│   ├── components/                 # 6 pipeline stages
│   │   ├── data_ingestion.py       #   Stage 1: Download/split dataset
│   │   ├── data_validation.py      #   Stage 2: Image integrity checks
│   │   ├── data_transformation.py  #   Stage 3: Compute normalization stats
│   │   ├── model_trainer.py        #   Stage 4: Train MobileNetV2 + TFLite
│   │   ├── model_evaluation.py     #   Stage 5: Benchmark TFLite
│   │   └── model_pusher.py         #   Stage 6: Deploy to export dir
│   ├── constants/                  # All configuration constants
│   ├── entity/
│   │   ├── config_entity.py        # Pipeline config dataclasses
│   │   ├── artifact_entity.py      # Pipeline artifact dataclasses
│   │   └── estimator.py            # SpectralPreprocessor, TFLitePredictor, CanopyAnalyzer
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Full training orchestrator
│   │   └── prediction_pipeline.py  # Single image + canopy analysis
│   ├── data_access/
│   │   └── magi_data.py            # Image dataset loader (Kaggle/local)
│   ├── storage/                    # Local model registry
│   ├── exception/                  # MAGIVisionException
│   ├── logger/                     # File + console logging
│   └── utils/
│       └── main_utils.py           # YAML/JSON I/O, image file discovery
├── config/
│   ├── model.yaml                  # Model architecture & training config
│   └── schema.yaml                 # Dataset schema & preprocessing config
├── templates/
│   └── magi.html                   # Web UI for prediction & training
├── app.py                          # FastAPI application
├── demo.py                         # CLI demo script
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container build
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone and enter project
cd MLOPs-Production-PIPELINE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Dataset

**Option A: Local dataset** — Place images in `data/` with class subfolders:

```
data/
├── healthy/
│   ├── img001.jpg
│   └── ...
├── mild_stress/
│   ├── img101.jpg
│   └── ...
├── moderate_stress/
│   └── ...
└── severe_stress/
    └── ...
```

**Option B: Kaggle download** — Set up Kaggle API credentials:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Option C: Custom dataset path**:

```bash
export MAGI_DATASET_PATH=/path/to/your/dataset
```

### 3. Run Training Pipeline

```bash
# Full training pipeline
python demo.py --train

# Or trigger via API
python demo.py --api
# Then POST to http://localhost:8080/train
```

### 4. Run Predictions

```bash
# Single image
python demo.py --predict path/to/image.jpg

# Start API server
python demo.py --api
# Upload image at http://localhost:8080
```

### 5. Docker

```bash
docker build -t magi-vision .
docker run -p 8080:8080 magi-vision
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/health` | API health check + model status |
| `POST` | `/predict` | Single image health classification |
| `POST` | `/predict/canopy` | Canopy-level grid-tile analysis |
| `POST` | `/train` | Trigger full training pipeline |
| `GET` | `/models` | List deployed model files |

### Example: Single Image Prediction

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@plant_image.jpg"
```

**Response:**

```json
{
  "class": "mild_stress",
  "confidence": 0.8234,
  "health_score": 0.6812,
  "class_probabilities": {
    "healthy": 0.1523,
    "mild_stress": 0.8234,
    "moderate_stress": 0.0198,
    "severe_stress": 0.0045
  }
}
```

### Example: Canopy Analysis

```bash
curl -X POST http://localhost:8080/predict/canopy \
  -F "file=@field_frame.jpg"
```

**Response:**

```json
{
  "tiles": [
    {"row": 0, "col": 0, "is_vegetation": true, "health_score": 0.92, "predicted_class": "healthy"},
    {"row": 0, "col": 1, "is_vegetation": false},
    ...
  ],
  "vegetation_coverage": 0.6667,
  "mean_health": 0.6150,
  "min_health": 0.3800,
  "stress_distribution": {"healthy": 3, "mild_stress": 3, "moderate_stress": 2},
  "action": "TRACK",
  "grid_size": "3x4"
}
```

---

## 🧬 Spectral Masking Pipeline

The preprocessing pipeline converts standard RGB camera images into 6-channel tensors:

```
Channel   Name     Source              Purpose
──────    ──────   ──────              ───────
  0       R        Camera (red)        Chlorophyll absorption
  1       G        Camera (green)      Chlorophyll reflectance
  2       B        Camera (blue)       Carotenoid/pigment health
  3       ExG      2G − R − B          Vegetation vs soil separation
  4       GRVI     (G−R)/(G+R+ε)       Chlorophyll concentration proxy
  5       L*       LAB lightness       Illumination-invariant brightness
```

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the LAB L-channel before computing indices, normalizing for lighting variation (sun, shadow, overcast).

---

## 🎯 Model Architecture

```
Input (224, 224, 6) → Conv2D 1×1 (6→3, channel adapter)
                    → MobileNetV2 backbone (ImageNet pretrained)
                    → GlobalAveragePooling2D
                    → Dense(256, relu)
                    → Dropout(0.4)
                    → Dense(4, softmax) → [healthy, mild, moderate, severe]
```

**Training strategy:**
- Phase 1: Freeze backbone, train head (15 epochs, lr=1e-3)
- Phase 2: Unfreeze top 50 layers, fine-tune (30 epochs, lr=1e-5)

**Export:** TFLite float16 → ~6.5MB model, ~125ms batch inference on Pi 4B

---

## 🤖 MAGI Robot Integration

This pipeline produces models for deployment on the MAGI quadruped robot:

| MAGI Core | Process | Feeds |
|---|---|---|
| **Core 0** | Caspar (decision engine) | Receives health scores via ZeroMQ |
| **Core 1** | Melchior (this system) | Continuous canopy health scanning |
| **Core 2** | Balthasar (scene/navigation) | Obstacle detection & path planning |
| **Core 3** | Gait controller | Motor control via ESP32 |

The Melchior health scores drive Caspar's behavior:

| Health Score | Caspar Action | Robot Behavior |
|---|---|---|
| `> 0.8` | IDLE | Normal walk speed (0.3 m/s) |
| `0.5 – 0.8` | TRACK | Slow down (0.15 m/s), increase scan rate |
| `0.2 – 0.5` | ALERT | Flag GPS zone, detailed logging |
| `< 0.2` | ANALYZE | Brief pause for high-resolution capture |

---

## 📊 Outputs

After training, the pipeline produces:

```
tflite_export/
├── melchior.tflite                 # TFLite model (float16, ~6.5MB)
├── class_mapping.json              # {0: "healthy", 1: "mild_stress", ...}
├── preprocessing_config.json       # CLAHE params, normalization stats
├── normalization_stats.json        # ExG/GRVI/L* channel statistics
└── deployment_manifest.json        # Model metadata, accuracy, latency
```

---

## 📝 License

Part of the MAGI project by Aki-ALPHON.
