import sys
import os
import json
import time
import numpy as np
import cv2

from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.constants import (
    IMAGE_SIZE, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, GRID_ROWS, GRID_COLS,
    SOIL_EXG_THRESHOLD, MIN_VEGETATION_FRACTION,
    CLASS_NAMES, HEALTH_SCORES, EMA_ALPHA, NUM_CLASSES,
)


class SpectralPreprocessor:
    """
    Implements the MAGI Spectral Masking pipeline:
      1. Resize to target size
      2. CLAHE illumination normalization (LAB L-channel)
      3. Compute spectral channels (ExG, GRVI, L*)
      4. Channel normalization (ImageNet for RGB, custom for computed)
    """

    def __init__(
        self,
        image_size: tuple = IMAGE_SIZE,
        clahe_clip: float = CLAHE_CLIP_LIMIT,
        clahe_grid: tuple = CLAHE_GRID_SIZE,
        imagenet_mean: list = None,
        imagenet_std: list = None,
        computed_stats: dict = None,
    ):
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_grid
        )
        self.imagenet_mean = imagenet_mean or IMAGENET_MEAN
        self.imagenet_std = imagenet_std or IMAGENET_STD
        # Stats for computed channels (ExG, GRVI, L*) — from training set
        self.computed_stats = computed_stats or {
            "ExG": {"mean": 0.0, "std": 1.0},
            "GRVI": {"mean": 0.0, "std": 1.0},
            "L_star": {"mean": 0.5, "std": 0.25},
        }

    def apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of LAB color space."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    def compute_spectral_channels(self, image_rgb: np.ndarray) -> dict:
        """
        Compute RGB-derived spectral proxy channels:
          ExG  = 2G - R - B (excess green index)
          GRVI = (G - R) / (G + R + eps) (green-red vegetation index)
          L*   = lightness from LAB color space
        """
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)

        exg = (2.0 * g - r - b) / 255.0
        grvi = (g - r) / (g + r + 1e-6)

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_star = lab[:, :, 0].astype(np.float32) / 255.0

        return {"ExG": exg, "GRVI": grvi, "L_star": l_star}

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Full spectral masking pipeline: BGR input → 6-channel normalized tensor.

        Returns:
            np.ndarray of shape (H, W, 6), dtype float32
        """
        # Step 1: Resize
        resized = cv2.resize(
            image_bgr, self.image_size, interpolation=cv2.INTER_AREA
        )

        # Step 2: CLAHE illumination normalization
        rgb = self.apply_clahe(resized)

        # Step 3: Compute spectral channels
        spectral = self.compute_spectral_channels(rgb)

        # Step 4: Normalize RGB channels (ImageNet)
        rgb_norm = rgb.astype(np.float32) / 255.0
        for i in range(3):
            rgb_norm[:, :, i] = (
                rgb_norm[:, :, i] - self.imagenet_mean[i]
            ) / self.imagenet_std[i]

        # Step 5: Normalize computed channels
        exg_norm = (
            spectral["ExG"] - self.computed_stats["ExG"]["mean"]
        ) / self.computed_stats["ExG"]["std"]
        grvi_norm = (
            spectral["GRVI"] - self.computed_stats["GRVI"]["mean"]
        ) / self.computed_stats["GRVI"]["std"]
        l_norm = (
            spectral["L_star"] - self.computed_stats["L_star"]["mean"]
        ) / self.computed_stats["L_star"]["std"]

        # Step 6: Stack into 6-channel tensor
        tensor = np.stack(
            [
                rgb_norm[:, :, 0],  # R
                rgb_norm[:, :, 1],  # G
                rgb_norm[:, :, 2],  # B
                exg_norm,           # ExG
                grvi_norm,          # GRVI
                l_norm,             # L*
            ],
            axis=-1,
        ).astype(np.float32)

        return tensor

    def compute_excess_green(self, tile_bgr: np.ndarray) -> float:
        """Quick ExG computation for vegetation pre-filter."""
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        return float(np.mean(exg))

    def save_stats(self, path: str):
        """Save computed channel normalization stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.computed_stats, f, indent=2)

    def load_stats(self, path: str):
        """Load computed channel normalization stats."""
        with open(path, "r") as f:
            self.computed_stats = json.load(f)


class TFLitePredictor:
    """
    Loads and runs inference on a TFLite model.
    Designed for batch tile inference on the Pi 4B.
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        try:
            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(
                model_path=model_path, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]["shape"]
            logging.info(
                f"TFLite model loaded: {model_path}, "
                f"input shape: {self.input_shape}"
            )
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on a single preprocessed tile.

        Args:
            input_tensor: shape (H, W, C), dtype float32
        Returns:
            class probabilities: shape (NUM_CLASSES,)
        """
        input_data = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output[0]

    def predict_batch(self, tiles: list) -> list:
        """
        Run inference on a list of preprocessed tiles.
        Returns list of class probability arrays.
        """
        results = []
        for tile in tiles:
            probs = self.predict(tile)
            results.append(probs)
        return results

    def benchmark_latency(self, input_tensor: np.ndarray, runs: int = 50) -> float:
        """Measure average inference latency in milliseconds."""
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self.predict(input_tensor)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        return float(np.mean(times[5:]))  # Skip first 5 warmup runs


class CanopyAnalyzer:
    """
    Full canopy-level analysis pipeline:
      1. Split frame into grid tiles
      2. Filter out soil tiles via ExG threshold
      3. Preprocess vegetation tiles
      4. Run batch inference
      5. Aggregate into frame-level health score
    """

    def __init__(
        self,
        preprocessor: SpectralPreprocessor,
        predictor: TFLitePredictor = None,
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS,
        soil_threshold: float = SOIL_EXG_THRESHOLD,
        min_veg_fraction: float = MIN_VEGETATION_FRACTION,
        class_names: list = None,
        health_scores: list = None,
        ema_alpha: float = EMA_ALPHA,
    ):
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.soil_threshold = soil_threshold
        self.min_veg_fraction = min_veg_fraction
        self.class_names = class_names or CLASS_NAMES
        self.health_scores = health_scores or HEALTH_SCORES
        self.ema_alpha = ema_alpha
        # EMA state for temporal smoothing
        self._smoothed_scores = None

    def split_into_tiles(self, frame_bgr: np.ndarray) -> list:
        """Split a frame into grid_rows × grid_cols tiles."""
        h, w = frame_bgr.shape[:2]
        tile_h = h // self.grid_rows
        tile_w = w // self.grid_cols
        tiles = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y1 = r * tile_h
                y2 = (r + 1) * tile_h if r < self.grid_rows - 1 else h
                x1 = c * tile_w
                x2 = (c + 1) * tile_w if c < self.grid_cols - 1 else w
                tiles.append({
                    "image": frame_bgr[y1:y2, x1:x2],
                    "row": r,
                    "col": c,
                })
        return tiles

    def filter_vegetation(self, tiles: list) -> tuple:
        """
        Separate vegetation tiles from soil tiles using ExG threshold.
        Returns (vegetation_tiles, soil_indices).
        """
        veg_tiles = []
        soil_indices = []
        for i, tile in enumerate(tiles):
            exg = self.preprocessor.compute_excess_green(tile["image"])
            if exg >= self.soil_threshold:
                tile["exg"] = exg
                tile["is_vegetation"] = True
                veg_tiles.append(tile)
            else:
                tile["is_vegetation"] = False
                soil_indices.append(i)
        return veg_tiles, soil_indices

    def compute_health_score(self, class_probs: np.ndarray) -> float:
        """
        Compute weighted health score from class probabilities.
        health = sum(prob_i × health_value_i)
        """
        score = 0.0
        for i, prob in enumerate(class_probs):
            score += float(prob) * self.health_scores[i]
        return score

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Full frame analysis pipeline.

        Returns dict with:
            tiles: list of per-tile results
            vegetation_coverage: fraction of tiles that are vegetation
            mean_health: average health score across vegetation tiles
            min_health: worst tile health score
            stress_distribution: histogram of tile classes
            action: recommended Caspar action
        """
        # Step 1: Split into grid
        tiles = self.split_into_tiles(frame_bgr)

        # Step 2: Filter vegetation
        veg_tiles, soil_indices = self.filter_vegetation(tiles)

        total_tiles = len(tiles)
        veg_count = len(veg_tiles)
        vegetation_coverage = veg_count / total_tiles if total_tiles > 0 else 0.0

        if veg_count == 0:
            return {
                "tiles": tiles,
                "vegetation_coverage": 0.0,
                "mean_health": 1.0,
                "min_health": 1.0,
                "stress_distribution": {},
                "action": "IDLE",
                "confidence": 0.0,
            }

        # Step 3: Preprocess vegetation tiles
        preprocessed = []
        for tile in veg_tiles:
            tensor = self.preprocessor.preprocess(tile["image"])
            preprocessed.append(tensor)

        # Step 4: Batch inference
        if self.predictor is not None:
            predictions = self.predictor.predict_batch(preprocessed)
        else:
            # No model loaded — return placeholder scores
            predictions = [
                np.array([0.7, 0.2, 0.08, 0.02]) for _ in preprocessed
            ]

        # Step 5: Compute per-tile health scores
        health_scores_list = []
        class_counts = {name: 0 for name in self.class_names}

        for tile, probs in zip(veg_tiles, predictions):
            health = self.compute_health_score(probs)
            predicted_class = self.class_names[int(np.argmax(probs))]
            tile["health_score"] = health
            tile["predicted_class"] = predicted_class
            tile["class_probs"] = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
            tile["confidence"] = float(np.max(probs))
            health_scores_list.append(health)
            class_counts[predicted_class] += 1

        # Step 6: Apply EMA temporal smoothing
        current_mean = float(np.mean(health_scores_list))
        if self._smoothed_scores is None:
            self._smoothed_scores = current_mean
        else:
            self._smoothed_scores = (
                self.ema_alpha * current_mean
                + (1 - self.ema_alpha) * self._smoothed_scores
            )

        mean_health = self._smoothed_scores
        min_health = float(np.min(health_scores_list))

        # Step 7: Determine action for Caspar
        if mean_health > 0.8:
            action = "IDLE"
        elif mean_health > 0.5:
            action = "TRACK"
        elif mean_health > 0.2:
            action = "ALERT"
        else:
            action = "ANALYZE"

        # If any single tile is critically low, escalate
        if min_health < 0.2:
            action = "ALERT" if action == "IDLE" else action

        return {
            "tiles": tiles,
            "vegetation_coverage": vegetation_coverage,
            "mean_health": mean_health,
            "min_health": min_health,
            "stress_distribution": class_counts,
            "action": action,
            "confidence": float(
                np.mean([t.get("confidence", 0) for t in veg_tiles])
            ),
        }

    def reset_smoothing(self):
        """Reset EMA temporal smoothing state."""
        self._smoothed_scores = None
