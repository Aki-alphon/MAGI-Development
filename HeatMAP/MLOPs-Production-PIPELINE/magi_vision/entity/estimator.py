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
        # Stats for 5 computed channels (ExG, GRVI, VARI, GLI, NGBDI)
        # from training set — updated for 8-channel stack
        self.computed_stats = computed_stats or {
            "ExG":   {"mean": 0.0,  "std": 1.0},
            "GRVI":  {"mean": 0.0,  "std": 1.0},
            "VARI":  {"mean": 0.0,  "std": 1.0},
            "GLI":   {"mean": 0.0,  "std": 1.0},
            "NGBDI": {"mean": 0.0,  "std": 1.0},
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
        Compute 5 nitrogen-specific RGB-derived spectral proxy channels.

        Channel stack (matches INPUT_CHANNELS=8):
          ExG   = 2G − R − B         (general vegetation index)
          GRVI  = (G−R)/(G+R+ε)      (chlorophyll proxy — standard)
          VARI  = (G−R)/(G+R−B+ε)   (best canopy-level nitrogen sensitivity)
          GLI   = (2G−R−B)/(2G+R+B+ε) (Green Leaf Index — nitrogen proxy)
          NGBDI = (G−B)/(G+B+ε)      (earliest N-deficiency RGB signal)

        L* dropped: under-canopy lighting too variable — adds noise, not signal.
        """
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)

        exg   = np.clip((2.0 * g - r - b) / 255.0, -10.0, 10.0)
        grvi  = np.clip((g - r) / (g + r + 1e-6), -10.0, 10.0)
        vari  = np.clip((g - r) / (g + r - b + 1e-6), -10.0, 10.0)
        gli   = np.clip((2.0 * g - r - b) / (2.0 * g + r + b + 1e-6), -10.0, 10.0)
        ngbdi = np.clip((g - b) / (g + b + 1e-6), -10.0, 10.0)

        return {
            "ExG":   exg,
            "GRVI":  grvi,
            "VARI":  vari,
            "GLI":   gli,
            "NGBDI": ngbdi,
        }

    def _znorm(self, arr: np.ndarray, key: str) -> np.ndarray:
        """Z-score normalise a computed channel using training stats."""
        m = self.computed_stats[key]["mean"]
        s = self.computed_stats[key]["std"]
        return (arr - m) / (s + 1e-8)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Full spectral masking pipeline:
        BGR input → 8-channel normalised tensor [R,G,B,ExG,GRVI,VARI,GLI,NGBDI]

        Returns:
            np.ndarray of shape (H, W, 8), dtype float32
        """
        # Step 1: Resize
        resized = cv2.resize(
            image_bgr, self.image_size, interpolation=cv2.INTER_AREA
        )

        # Step 2: CLAHE illumination normalisation
        rgb = self.apply_clahe(resized)

        # Step 3: Compute 5 nitrogen-specific spectral channels
        spectral = self.compute_spectral_channels(rgb)

        # Step 4: Normalise RGB channels (ImageNet stats)
        rgb_norm = rgb.astype(np.float32) / 255.0
        for i in range(3):
            rgb_norm[:, :, i] = (
                rgb_norm[:, :, i] - self.imagenet_mean[i]
            ) / self.imagenet_std[i]

        # Step 5: Z-score normalise computed channels using training stats
        # Stack into 8-channel tensor [R, G, B, ExG, GRVI, VARI, GLI, NGBDI]
        tensor = np.stack(
            [
                rgb_norm[:, :, 0],              # ch0: R
                rgb_norm[:, :, 1],              # ch1: G
                rgb_norm[:, :, 2],              # ch2: B
                self._znorm(spectral["ExG"],   "ExG"),    # ch3
                self._znorm(spectral["GRVI"],  "GRVI"),   # ch4
                self._znorm(spectral["VARI"],  "VARI"),   # ch5 ← new
                self._znorm(spectral["GLI"],   "GLI"),    # ch6 ← new
                self._znorm(spectral["NGBDI"], "NGBDI"),  # ch7 ← new
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
        """Save computed channel normalization stats (all 5 channels)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.computed_stats, f, indent=2)

    def load_stats(self, path: str):
        """Load computed channel normalization stats."""
        with open(path, "r") as f:
            self.computed_stats = json.load(f)


class UnderCanopyAugmentor:
    """
    Simulates the lighting environment of a bot navigating under the plant canopy
    (cotton rows / leafy vegetable beds).

    Under-canopy characteristics:
      1. Green-tinted ambient light  — sunlight filtered through leaves above
      2. Directional shadow patches  — occlusion from canopy overhead
      3. Perspective tilt            — bot camera at oblique angle to leaf surface
      4. Dappled brightness          — sun flecks through canopy gaps

    Usage:
        aug = UnderCanopyAugmentor(seed=42)
        augmented = aug.augment(image_bgr)           # single image
        batch     = aug.augment_batch(images, n=800) # generate N synthetic images
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def augment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply full under-canopy augmentation pipeline to one image."""
        img = image_bgr.copy().astype(np.float32)
        img = self._green_cast(img)
        img = self._canopy_shadow(img)
        img = self._perspective_tilt(img)
        img = self._dappled_brightness(img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def augment_batch(self, images: list, n: int) -> list:
        """
        Generate n augmented images by sampling from the input list with replacement.
        Returns list of augmented BGR np.ndarray images.
        """
        out = []
        for _ in range(n):
            src = images[int(self.rng.integers(0, len(images)))]
            out.append(self.augment(src))
        return out

    # ── Private augmentation steps ──────────────────────────────────────────

    def _green_cast(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate sunlight filtered through canopy leaves — adds green tint.
        Under cotton canopy: blue channel attenuated most, green boosted slightly.
        """
        # BGR channel scale factors (random per-image variation)
        b_scale = self.rng.uniform(0.75, 0.90)   # blue attenuated by canopy
        g_scale = self.rng.uniform(1.05, 1.18)   # green slightly boosted
        r_scale = self.rng.uniform(0.85, 0.98)   # red slight reduction

        img[:, :, 0] *= b_scale   # B
        img[:, :, 1] *= g_scale   # G
        img[:, :, 2] *= r_scale   # R
        return img

    def _canopy_shadow(self, img: np.ndarray) -> np.ndarray:
        """
        Add directional shadow patch simulating leaf occlusion from above.
        Uses a soft-edged elliptical shadow at a random position.
        """
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)

        # Random shadow ellipse
        cx = int(self.rng.integers(w // 4, 3 * w // 4))
        cy = int(self.rng.integers(h // 4, 3 * h // 4))
        rx = int(self.rng.integers(w // 6, w // 3))
        ry = int(self.rng.integers(h // 6, h // 3))

        # Draw ellipse shadow on mask
        import cv2 as _cv2
        shadow_strength = self.rng.uniform(0.35, 0.65)
        _cv2.ellipse(
            mask,
            (cx, cy), (rx, ry),
            angle=int(self.rng.integers(0, 180)),
            startAngle=0, endAngle=360,
            color=shadow_strength,
            thickness=-1,
        )
        # Blur mask for soft edges
        mask = _cv2.GaussianBlur(mask, (51, 51), 0)
        mask = np.clip(mask, shadow_strength, 1.0)

        img *= mask[:, :, np.newaxis]
        return img

    def _perspective_tilt(self, img: np.ndarray) -> np.ndarray:
        """
        Apply mild perspective warp — simulates oblique camera angle looking
        up/sideways at the leaf surface from under the canopy.
        """
        import cv2 as _cv2
        h, w = img.shape[:2]
        max_shift = int(min(h, w) * 0.08)   # 8% max tilt

        # Four source corners → tilted destination corners
        pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dx1 = int(self.rng.integers(-max_shift, max_shift))
        dy1 = int(self.rng.integers(-max_shift, max_shift))
        dx2 = int(self.rng.integers(-max_shift, max_shift))
        pts_dst = np.float32([
            [dx1,     dy1],
            [w + dx2, 0],
            [w,       h],
            [0,       h],
        ])
        M = _cv2.getPerspectiveTransform(pts_src, pts_dst)
        return _cv2.warpPerspective(img, M, (w, h),
                                    flags=_cv2.INTER_LINEAR,
                                    borderMode=_cv2.BORDER_REFLECT_101)

    def _dappled_brightness(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate sun flecks through canopy gaps — random bright patches.
        """
        h, w = img.shape[:2]
        n_flecks = int(self.rng.integers(0, 4))   # 0–3 sun flecks
        for _ in range(n_flecks):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(0, h))
            radius = int(self.rng.integers(15, 60))
            strength = self.rng.uniform(1.10, 1.35)

            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
            fleck = np.clip(1.0 - dist / (radius * 1.5), 0, 1)
            fleck = fleck * (strength - 1.0) + 1.0
            img *= fleck[:, :, np.newaxis]

        return img


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
            action: recommended Lugia action
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

        # Step 7: Determine action for Lugia
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
