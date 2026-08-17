"""
MAGI OS v2 — MAGI-1 Celebi: Plant Health Heatmap Node
src/magi1/celebi.py

Replaces the original object-detection Celebi with a canopy-level
plant health classification pipeline using the trained 6-channel
MobileNetV2 TFLite model (celebi.tflite).

Pipeline (per frame, every N-th from camera):
  1. Read BGR frame from POSIX shared memory
  2. Split into 4×3 grid (12 tiles)
  3. ExG vegetation pre-filter (skip soil tiles → save ~40% compute)
  4. 6-channel spectral masking (CLAHE, ExG, GRVI, L*)
  5. Batch TFLite inference (float16 quantised, XNNPACK on Pi)
  6. EMA temporal smoothing
  7. Publish CanopyHealthMsg to /canopy_health

Pinned to CPU Core 1.
"""

import os, sys, time, json
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {1})
except (AttributeError, OSError):
    pass

import yaml
import numpy as np
import cv2

from core.lifecycle import LifecycleNode, State
from core.messages  import (
    CanopyHealthMsg, HealthTile, DiagStatus,
    PLANT_CLASS_NAMES, PLANT_HEALTH_SCORES,
)
from core.param_server import ParamClient
from common.spectral   import SpectralPreprocessor, CanopyAnalyzer

# Relative path within the deployment directory for normalisation stats
_STATS_FILENAME = "normalization_stats.json"


class Celebi(LifecycleNode):
    """
    MAGI-1 Celebi — Plant Canopy Health Node.

    Subscribes to:  /sensors (heartbeat / gating)
                    /scene   (wake/sleep gating from Gengar)
    Publishes to:   /canopy_health
    """

    def on_configure(self):
        with open("/opt/magi/config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        self._cfg      = cfg["models"]["magi1"]
        self._cam_cfg  = cfg["camera"]
        self._ipc_cfg  = cfg["ipc"]
        self._params   = ParamClient("magi1")

        # Model params
        self._model_path   = self._cfg["path"]
        self._num_threads  = self._cfg.get("num_threads", 2)
        self._use_xnnpack  = self._cfg.get("use_xnnpack", True)
        # How many frames to skip between inferences (saves CPU when walking)
        self._run_every    = self._cfg.get("subsample_every", 5)

        # Spectral preprocessor + canopy analyser
        norm_stats = self._load_norm_stats()
        self._preprocessor = SpectralPreprocessor(computed_stats=norm_stats)
        self._analyzer     = CanopyAnalyzer(self._preprocessor)

        # TFLite interpreter (loaded after preprocessor — graceful fallback)
        self._interpreter  = None
        self._load_model()

        # Shared memory frame reader
        self._shm          = None
        self._frame_shape  = None
        self._frame_count  = 0
        self._inf_times    = []
        self._load_shm()

        # Gating state (Core 1 wakes only when Gengar sees a stable plant)
        self._last_plant_ts = 0.0
        self._plant_timeout = self._cfg.get("plant_timeout_s", 5.0)
        self._start_active  = self._cfg.get("start_active", False)

        # Publishers / subscribers
        self.create_publisher("/canopy_health")
        self.create_subscription("/sensors", self._on_sensor)
        self.create_subscription("/scene",   self._on_scene)

        self.log.info(
            f"Celebi configured | model={self._model_path} "
            f"| grid={self._analyzer.grid_rows}×{self._analyzer.grid_cols}"
            + (" | start_active=True (Docker mode)" if self._start_active else "")
        )

    # ── Initialisation helpers ───────────────────────────────────────────────

    def _load_norm_stats(self) -> dict:
        """
        Try to load normalization_stats.json from the models directory.
        Falls back to neutral defaults if the file is missing (safe for Docker).
        """
        stats_path = os.path.join(os.path.dirname(self._cfg.get("path", "")), _STATS_FILENAME)
        if os.path.isfile(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                self.log.info(f"Loaded normalisation stats: {stats_path}")
                return stats
            except Exception as e:
                self.log.warning(f"Failed to read norm stats ({e}) — using defaults")
        else:
            self.log.warning(
                f"normalisation_stats.json not found at {stats_path} "
                "— using neutral defaults (OK for Docker testing)"
            )
        return None  # SpectralPreprocessor falls back to neutral defaults

    def _load_model(self):
        """Load TFLite model with XNNPACK delegate where available."""
        path = self._model_path
        try:
            import tflite_runtime.interpreter as tflite
            if self._use_xnnpack:
                try:
                    delegate = tflite.load_delegate(
                        "libXNNPACK.so.1", {"num_threads": self._num_threads}
                    )
                    self._interpreter = tflite.Interpreter(
                        model_path=path, experimental_delegates=[delegate]
                    )
                    self.log.info("TFLite loaded with XNNPACK delegate")
                except Exception:
                    self._interpreter = tflite.Interpreter(
                        model_path=path, num_threads=self._num_threads
                    )
                    self.log.info("TFLite loaded (no XNNPACK — fallback)")
            else:
                self._interpreter = tflite.Interpreter(
                    model_path=path, num_threads=self._num_threads
                )
                self.log.info("TFLite loaded (XNNPACK disabled)")

            self._interpreter.allocate_tensors()
            in_det = self._interpreter.get_input_details()
            self.log.info(f"Model input shape: {in_det[0]['shape']}")
        except Exception as e:
            self.log.error(f"Model load failed: {e} — running in NO-MODEL mode (Docker safe)")
            self._interpreter = None

    def _load_shm(self):
        """Open the POSIX shared memory frame reader."""
        if not self._cam_cfg.get("enabled", False):
            self.log.info("Camera disabled in config — using blank frames (Docker mode)")
            return
        try:
            from common.ipc import SharedFrame
            self._shm = SharedFrame(
                name=self._ipc_cfg["shm_camera"],
                size=self._ipc_cfg["shm_camera_size"],
                create=False,
            )
            self._frame_shape = (
                self._cam_cfg["height"],
                self._cam_cfg["width"],
                3,
            )
            self.log.info(
                f"Camera SHM reader ready: {self._frame_shape}"
            )
        except Exception as e:
            self.log.warning(f"SHM unavailable: {e} — using blank frames")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_activate(self):
        self._frame_count = 0
        self._analyzer.reset_ema()
        # If start_active flag set, keep last_plant_ts current so timeout never fires
        if self._start_active:
            self._last_plant_ts = float("inf") - self._plant_timeout - 1
        self.log.info("Celebi ACTIVE — plant health scanning started")

    def on_deactivate(self):
        self.log.info("Celebi INACTIVE — conserving Core 1 power")

    # ── Subscription callbacks ───────────────────────────────────────────────

    def _on_scene(self, data: dict):
        """
        Gengar scene detection — wake Celebi when a stable plant is visible,
        sleep it when the scene changes to avoid wasting Core 1.
        """
        scene = data.get("scene", "unknown")
        if scene == "stable_plant":
            self._last_plant_ts = time.time()
            if self.state == State.INACTIVE:
                self.log.info("Stable plant scene → activating health scanner")
                self.activate()

    def _on_sensor(self, data: dict):
        """
        Called at sensor hub rate (~10–50 Hz).
        Acts as the processing heartbeat for Celebi.
        """
        if self.state != State.ACTIVE:
            return

        # Auto-sleep if Gengar hasn't seen a stable plant recently
        if time.time() - self._last_plant_ts > self._plant_timeout:
            self.log.info(
                f"No stable plant for {self._plant_timeout:.0f}s → deactivating"
            )
            self.deactivate()
            return

        # Frame subsampling (every N sensor ticks)
        self._frame_count += 1
        if self._frame_count % self._run_every != 0:
            return

        frame = self._get_frame()
        if frame is None:
            return

        t0     = time.monotonic()
        result = self._analyzer.analyze_frame(frame, self._interpreter)
        inf_ms = (time.monotonic() - t0) * 1000

        self._inf_times.append(inf_ms)
        if len(self._inf_times) > 60:
            self._inf_times.pop(0)

        fps = self._frame_count / max(1.0, time.monotonic())

        # Build typed message
        tile_dicts = []
        for t in result["tiles"]:
            tile_dicts.append({
                "row":             t["row"],
                "col":             t["col"],
                "is_vegetation":   t.get("is_vegetation", False),
                "predicted_class": t.get("predicted_class", "unknown"),
                "health_score":    float(t.get("health_score", 1.0)),
                "confidence":      float(t.get("confidence", 0.0)),
            })

        msg = CanopyHealthMsg(
            header              = self.next_header("camera_link"),
            tiles               = tile_dicts,
            grid_rows           = self._analyzer.grid_rows,
            grid_cols           = self._analyzer.grid_cols,
            vegetation_coverage = float(result["vegetation_coverage"]),
            mean_health         = float(result["mean_health"]),
            min_health          = float(result["min_health"]),
            stress_distribution = result["stress_distribution"],
            recommended_action  = result["recommended_action"],
            trend_slope         = float(result.get("trend_slope", 0.0)),
            latency_ms          = inf_ms,
            fps                 = fps,
        )
        self.publish("/canopy_health", msg)

        # Diagnostics heartbeat
        avg_ms = sum(self._inf_times) / max(1, len(self._inf_times))
        veg_pct = result["vegetation_coverage"] * 100.0
        self.publish_diag(
            DiagStatus.OK,
            f"health={result['mean_health']:.2f} action={result['recommended_action']}",
            {
                "mean_health":    round(result["mean_health"], 3),
                "action":         result["recommended_action"],
                "veg_coverage%":  round(veg_pct, 1),
                "inf_ms":         round(avg_ms, 1),
            },
        )

        if result["recommended_action"] != "IDLE":
            self.log.info(
                f"[CANOPY] action={result['recommended_action']} "
                f"health={result['mean_health']:.3f} "
                f"slope={result.get('trend_slope', 0.0):.3f} "
                f"veg={veg_pct:.0f}% "
                f"({inf_ms:.0f}ms)"
            )

    # ── Frame acquisition ────────────────────────────────────────────────────

    def _get_frame(self) -> np.ndarray | None:
        """
        Read the latest camera frame.
        - From POSIX SHM (real Pi) when camera is enabled.
        - Synthetic green gradient frame in Docker/no-camera mode.
        """
        if self._shm is not None:
            try:
                return self._shm.read(self._frame_shape)
            except Exception:
                return None

        # Docker / no-camera: generate a synthetic plant-like frame so the
        # pipeline runs end-to-end and can be validated without hardware.
        w = self._cam_cfg.get("width",  640)
        h = self._cam_cfg.get("height", 480)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Fill with a green gradient (simulates vegetation canopy)
        frame[:, :, 1] = np.linspace(80, 200, w, dtype=np.uint8)
        # Add some random variation to stress patches
        rng = np.random.default_rng(self._frame_count)
        noise = rng.integers(-30, 30, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame


if __name__ == "__main__":
    Celebi(node_id="magi1", cpu_core=1).boot()
