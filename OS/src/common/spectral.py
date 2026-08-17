"""
MAGI OS — Spectral Masking & Canopy Analysis
src/common/spectral.py

Self-contained port of the MLOps pipeline's SpectralPreprocessor
and CanopyAnalyzer.  No dependency on the magi_vision package —
only numpy and OpenCV.

Channels produced per tile:
  0: R     — Camera red channel (ImageNet-normalised)
  1: G     — Camera green channel
  2: B     — Camera blue channel
  3: ExG   — Excess Green Index:  2G − R − B
  4: GRVI  — Green-Red Vegetation Index: (G−R)/(G+R+ε)
  5: VARI  — Visible Atmospherically Resistant Index
  6: GLI   — Green Leaf Index
  7: NGBDI — Normalised Green-Blue Difference
"""

import json
import numpy as np
import cv2


# ── Default constants (must match training config) ───────────────────────────

IMAGE_SIZE   = (224, 224)          # (W, H) — matches MobileNetV2 input
GRID_ROWS    = 3
GRID_COLS    = 4
EMA_ALPHA    = 0.6                 # EMA smoothing weight for current frame

# ExG soil pre-filter threshold (raw pixel units, not normalised)
SOIL_EXG_THRESHOLD = 10.0

# ImageNet normalisation constants (RGB order)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# CLAHE parameters (illumination equalisation)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE  = (8, 8)

# Plant health class order (MUST match training label directory sort order)
CLASS_NAMES   = ["baseline_healthy", "early_nitrogen_stress", "active_chlorosis", "severe_deficiency"]
HEALTH_SCORES = [1.0, 0.68, 0.35, 0.05]


# ─────────────────────────────────────────────────────────────────────────────
# SpectralPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class SpectralPreprocessor:
    """
    Converts a raw BGR camera frame (or tile) into a 6-channel float32 tensor.

    Pipeline:
      1. Resize → IMAGE_SIZE
      2. CLAHE on LAB L-channel  (normalises sun/shadow/overcast)
      3. Compute 5 spectral indices (ExG, GRVI, VARI, GLI, NGBDI)
      4. ImageNet-normalise RGB channels
      5. z-score normalise computed channels (using stats from training set)
      6. Stack into (H, W, 8) tensor
    """

    def __init__(
        self,
        image_size: tuple = IMAGE_SIZE,
        clahe_clip: float = CLAHE_CLIP_LIMIT,
        clahe_grid: tuple = CLAHE_GRID_SIZE,
        imagenet_mean: list = None,
        imagenet_std:  list = None,
        computed_stats: dict = None,
    ):
        self.image_size    = image_size
        self.clahe         = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.imagenet_mean = imagenet_mean or IMAGENET_MEAN
        self.imagenet_std  = imagenet_std  or IMAGENET_STD
        # Channel statistics from training set (fall back to neutral if not provided)
        self.computed_stats = computed_stats or {
            "ExG":   {"mean": 0.0,  "std": 1.0},
            "GRVI":  {"mean": 0.0,  "std": 1.0},
            "VARI":  {"mean": 0.0,  "std": 1.0},
            "GLI":   {"mean": 0.0,  "std": 1.0},
            "NGBDI": {"mean": 0.0,  "std": 1.0},
        }

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Full pipeline: BGR frame/tile → (H, W, 8) float32 tensor.
        """
        # 1. Resize
        resized = cv2.resize(image_bgr, self.image_size, interpolation=cv2.INTER_AREA)

        # 2. CLAHE illumination normalisation (operates on LAB L-channel)
        lab         = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b     = cv2.split(lab)
        l_enhanced  = self.clahe.apply(l)
        lab_enh     = cv2.merge((l_enhanced, a, b))
        rgb         = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)

        # 3. Compute spectral channels
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b_ch = rgb[:, :, 2].astype(np.float32)

        exg   = (2.0 * g - r - b_ch) / 255.0
        grvi  = (g - r) / (g + r + 1e-6)
        vari  = (g - r) / (g + r - b_ch + 1e-6)
        gli   = (2.0 * g - r - b_ch) / (2.0 * g + r + b_ch + 1e-6)
        ngbdi = (g - b_ch) / (g + b_ch + 1e-6)

        # 4. ImageNet-normalise RGB
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_norm = np.empty_like(rgb_f)
        for i in range(3):
            rgb_norm[:, :, i] = (rgb_f[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]

        # 5. z-score normalise computed channels
        def _znorm(arr, key):
            m = self.computed_stats[key]["mean"]
            s = self.computed_stats[key]["std"]
            return (arr - m) / (s + 1e-8)

        # 6. Stack → (H, W, 8)
        tensor = np.stack(
            [rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2],
             _znorm(exg, "ExG"), _znorm(grvi, "GRVI"), _znorm(vari, "VARI"),
             _znorm(gli, "GLI"), _znorm(ngbdi, "NGBDI")],
            axis=-1,
        ).astype(np.float32)

        return tensor

    def compute_excess_green(self, tile_bgr: np.ndarray) -> float:
        """
        Quick ExG computation (raw pixel units) for the soil pre-filter.
        Returns mean(2G − R − B) over the tile.
        """
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        exg = 2.0 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]
        return float(np.mean(exg))

    def load_stats(self, path: str):
        """Load computed channel stats from a JSON file (e.g. normalization_stats.json)."""
        with open(path) as f:
            self.computed_stats = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CanopyAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class CanopyAnalyzer:
    """
    Stateful canopy analysis pipeline designed to run inside Celebi.

    Maintains EMA temporal smoothing state between frames.

    Usage:
        analyzer = CanopyAnalyzer(preprocessor)
        result   = analyzer.analyze_frame(frame_bgr, interpreter)
    """

    def __init__(
        self,
        preprocessor: SpectralPreprocessor,
        grid_rows: int      = GRID_ROWS,
        grid_cols: int      = GRID_COLS,
        soil_threshold: float = SOIL_EXG_THRESHOLD,
        class_names: list   = None,
        health_scores: list = None,
        ema_alpha: float    = EMA_ALPHA,
    ):
        self.preprocessor   = preprocessor
        self.grid_rows      = grid_rows
        self.grid_cols      = grid_cols
        self.soil_threshold = soil_threshold
        self.class_names    = class_names    or CLASS_NAMES
        self.health_scores  = health_scores  or HEALTH_SCORES
        self.ema_alpha      = ema_alpha
        
        self._smoothed               = None   # EMA state
        self._history                = []     # Health score history (for trend)
        self.trend_window            = 5
        self.trend_decline_threshold = -0.015
        self.trend_min_consecutive   = 3
        self.early_warning_min_health= 0.65

    # ── Internal helpers ────────────────────────────────────────────────────

    def _split_tiles(self, frame_bgr: np.ndarray) -> list:
        h, w = frame_bgr.shape[:2]
        tile_h = h // self.grid_rows
        tile_w = w // self.grid_cols
        tiles  = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y1 = r * tile_h
                y2 = (r + 1) * tile_h if r < self.grid_rows - 1 else h
                x1 = c * tile_w
                x2 = (c + 1) * tile_w if c < self.grid_cols - 1 else w
                tiles.append({"row": r, "col": c, "image": frame_bgr[y1:y2, x1:x2]})
        return tiles

    def _health_score(self, probs: np.ndarray) -> float:
        """Weighted health score: sum(prob_i × health_value_i)."""
        return float(sum(p * s for p, s in zip(probs, self.health_scores)))

    def _infer_batch(self, tensors: list, interpreter) -> list:
        """
        Run interpreter.invoke() for each tile tensor.
        Compatible with both real tflite_runtime and the mock stub.
        Returns list of class-probability arrays.
        """
        results = []
        in_idx  = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]
        for t in tensors:
            data = np.expand_dims(t, axis=0).astype(np.float32)
            interpreter.set_tensor(in_idx, data)
            interpreter.invoke()
            probs = interpreter.get_tensor(out_idx)[0]
            results.append(probs)
        return results

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze_frame(self, frame_bgr: np.ndarray, interpreter=None) -> dict:
        """
        Run the full canopy analysis on one camera frame.

        Args:
            frame_bgr:   Raw BGR frame from camera (any resolution).
            interpreter: tflite_runtime.Interpreter (or mock stub).
                         If None, returns placeholder healthy scores.

        Returns dict with:
            tiles               — list of per-tile dicts
            vegetation_coverage — fraction of tiles that are plants
            mean_health         — EMA-smoothed mean health score (0–1)
            min_health          — minimum tile health this frame
            stress_distribution — {class_name: count}
            recommended_action  — "IDLE" | "TRACK" | "ALERT" | "ANALYZE"
            confidence          — mean max-class probability
        """
        tiles = self._split_tiles(frame_bgr)

        # Vegetation filter (soil pre-filter saves 30–50% of inference)
        veg_tiles  = []
        soil_tiles = []
        for tile in tiles:
            exg = self.preprocessor.compute_excess_green(tile["image"])
            if exg >= self.soil_threshold:
                tile["is_vegetation"] = True
                tile["exg"]           = exg
                veg_tiles.append(tile)
            else:
                tile["is_vegetation"] = False
                soil_tiles.append(tile)

        total_tiles = len(tiles)
        veg_count   = len(veg_tiles)
        veg_coverage = veg_count / total_tiles if total_tiles > 0 else 0.0

        if veg_count == 0 or interpreter is None:
            # Nothing to analyse — stay IDLE (healthy assumption)
            for tile in tiles:
                tile.setdefault("predicted_class", "healthy")
                tile.setdefault("health_score", 1.0)
                tile.setdefault("confidence",   0.0)
            return {
                "tiles":               tiles,
                "vegetation_coverage": veg_coverage,
                "mean_health":         1.0,
                "min_health":          1.0,
                "stress_distribution": {},
                "recommended_action":  "IDLE",
                "confidence":          0.0,
            }

        # Preprocess vegetation tiles into 6-channel tensors
        preprocessed = [self.preprocessor.preprocess(t["image"]) for t in veg_tiles]

        # Batch inference
        predictions = self._infer_batch(preprocessed, interpreter)

        # Per-tile results
        health_list  = []
        class_counts = {n: 0 for n in self.class_names}

        for tile, probs in zip(veg_tiles, predictions):
            health            = self._health_score(probs)
            pred_class        = self.class_names[int(np.argmax(probs))]
            tile["predicted_class"] = pred_class
            tile["health_score"]    = health
            tile["confidence"]      = float(np.max(probs))
            health_list.append(health)
            class_counts[pred_class] += 1

        # EMA temporal smoothing
        current_mean = float(np.mean(health_list))
        if self._smoothed is None:
            self._smoothed = current_mean
        else:
            self._smoothed = self.ema_alpha * current_mean + (1 - self.ema_alpha) * self._smoothed

        mean_health = self._smoothed
        min_health  = float(np.min(health_list))

        # Update history for trend analysis
        self._history.append(mean_health)
        if len(self._history) > self.trend_window:
            self._history.pop(0)

        # Trend analysis (slope calculation)
        trend_slope = 0.0
        consecutive_decline = 0
        if len(self._history) == self.trend_window:
            # Simple linear regression slope
            x = np.arange(self.trend_window)
            y = np.array(self._history)
            trend_slope = float(np.polyfit(x, y, 1)[0])
            
            # Count consecutive declines
            for i in range(1, self.trend_window):
                if self._history[i] < self._history[i-1]:
                    consecutive_decline += 1
                else:
                    consecutive_decline = 0

        # Map health score → Lugia action
        if mean_health > 0.8:
            action = "IDLE"
        elif mean_health > 0.5:
            action = "TRACK"
        elif mean_health > 0.2:
            action = "ALERT"
        else:
            action = "ANALYZE"

        # Trend-based EARLY_WARNING override (pre-symptomatic prediction)
        if (action in ["IDLE", "TRACK"] and 
            trend_slope <= self.trend_decline_threshold and 
            consecutive_decline >= self.trend_min_consecutive and 
            mean_health >= self.early_warning_min_health):
            action = "EARLY_WARNING"

        # Single critically-low tile escalates to ALERT
        if min_health < 0.2 and action in ["IDLE", "TRACK", "EARLY_WARNING"]:
            action = "ALERT"

        avg_conf = float(np.mean([t.get("confidence", 0.0) for t in veg_tiles]))

        return {
            "tiles":               tiles,
            "vegetation_coverage": veg_coverage,
            "mean_health":         mean_health,
            "min_health":          min_health,
            "stress_distribution": class_counts,
            "recommended_action":  action,
            "confidence":          avg_conf,
            "trend_slope":         trend_slope,
        }

    def reset_ema(self):
        """Reset temporal smoothing state (call after long gap / model reload)."""
        self._smoothed = None
        self._history = []
