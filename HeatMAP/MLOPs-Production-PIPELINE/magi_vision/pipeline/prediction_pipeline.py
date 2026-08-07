import os
import sys
import numpy as np
import cv2

from magi_vision.entity.config_entity import MAGIPredictorConfig
from magi_vision.entity.estimator import (
    SpectralPreprocessor,
    TFLitePredictor,
    CanopyAnalyzer,
)
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import read_json_file
from magi_vision.constants import (
    CLASS_NAMES, HEALTH_SCORES, GRID_ROWS, GRID_COLS,
    SOIL_EXG_THRESHOLD, EMA_ALPHA,
)


class MAGIPredictionPipeline:
    """
    MAGI Vision Prediction Pipeline
    ================================
    Provides prediction capability for:
      1. Single image upload (via API)
      2. Canopy-level frame analysis (grid-tile approach)
      
    Loads the deployed TFLite model and preprocessing config.
    """

    def __init__(self, config: MAGIPredictorConfig = None):
        try:
            self.config = config or MAGIPredictorConfig()
            self.model_loaded = False
            self.predictor = None
            self.preprocessor = None
            self.canopy_analyzer = None
            self.class_names = CLASS_NAMES
            self._load_model()
        except Exception as e:
            logging.warning(f"Prediction pipeline init warning: {e}")
            self.model_loaded = False

    def _load_model(self):
        """Load TFLite model and preprocessing config."""
        try:
            # Load normalization stats
            computed_stats = None
            if os.path.exists(self.config.normalization_stats_path):
                computed_stats = read_json_file(
                    self.config.normalization_stats_path
                )

            # Load class mapping
            if os.path.exists(self.config.class_mapping_path):
                mapping = read_json_file(self.config.class_mapping_path)
                self.class_names = [mapping[str(i)] for i in range(len(mapping))]

            # Initialize preprocessor
            self.preprocessor = SpectralPreprocessor(
                computed_stats=computed_stats
            )

            # Load TFLite model
            if os.path.exists(self.config.tflite_model_path):
                self.predictor = TFLitePredictor(
                    model_path=self.config.tflite_model_path
                )
                self.model_loaded = True
                logging.info("Prediction pipeline: model loaded successfully")
            else:
                logging.warning(
                    f"TFLite model not found: {self.config.tflite_model_path}. "
                    f"Train a model first."
                )

            # Initialize canopy analyzer
            self.canopy_analyzer = CanopyAnalyzer(
                preprocessor=self.preprocessor,
                predictor=self.predictor,
                grid_rows=GRID_ROWS,
                grid_cols=GRID_COLS,
                soil_threshold=SOIL_EXG_THRESHOLD,
                class_names=self.class_names,
                health_scores=HEALTH_SCORES,
                ema_alpha=EMA_ALPHA,
            )

        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def predict_single(self, image_bgr: np.ndarray) -> dict:
        """
        Predict health status of a single image.
        Used for API image upload.

        Returns:
            dict with class, confidence, health_score
        """
        if not self.model_loaded:
            return {
                "error": "Model not loaded. Train a model first.",
                "class": "unknown",
                "confidence": 0.0,
                "health_score": 0.0,
            }

        try:
            tensor = self.preprocessor.preprocess(image_bgr)
            probs = self.predictor.predict(tensor)

            predicted_idx = int(np.argmax(probs))
            predicted_class = self.class_names[predicted_idx]
            confidence = float(np.max(probs))
            health_score = sum(
                float(p) * s for p, s in zip(probs, HEALTH_SCORES)
            )

            return {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "health_score": round(health_score, 4),
                "class_probabilities": {
                    name: round(float(prob), 4)
                    for name, prob in zip(self.class_names, probs)
                },
            }

        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def predict_canopy(self, frame_bgr: np.ndarray) -> dict:
        """
        Full canopy-level analysis of a camera frame.
        Uses grid-tile approach with vegetation pre-filter.

        Returns:
            dict with tile results, vegetation coverage,
            health score, stress distribution, action
        """
        if self.canopy_analyzer is None:
            return {"error": "Canopy analyzer not initialized"}

        try:
            result = self.canopy_analyzer.analyze_frame(frame_bgr)

            # Clean up tiles for JSON serialization
            clean_tiles = []
            for tile in result["tiles"]:
                clean_tile = {
                    "row": tile["row"],
                    "col": tile["col"],
                    "is_vegetation": tile.get("is_vegetation", False),
                }
                if tile.get("is_vegetation"):
                    clean_tile["health_score"] = round(
                        tile.get("health_score", 0), 4
                    )
                    clean_tile["predicted_class"] = tile.get(
                        "predicted_class", "unknown"
                    )
                    clean_tile["confidence"] = round(
                        tile.get("confidence", 0), 4
                    )
                clean_tiles.append(clean_tile)

            return {
                "tiles": clean_tiles,
                "vegetation_coverage": round(
                    result["vegetation_coverage"], 4
                ),
                "mean_health": round(result["mean_health"], 4),
                "min_health": round(result["min_health"], 4),
                "stress_distribution": result["stress_distribution"],
                "action": result["action"],
                "confidence": round(result["confidence"], 4),
                "grid_size": f"{GRID_ROWS}x{GRID_COLS}",
            }

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
