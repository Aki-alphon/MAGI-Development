"""
MAGI OS v2 — MAGI-2 Gengar: Scene Analysis Node
src/magi2/gengar.py

Upgraded to LifecycleNode. Publishes typed SceneMsg to /scene.
Fuses camera classification with IMU motion analysis.
Pinned to CPU Core 2.
"""

import os, sys, time
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {2})
except (AttributeError, OSError):
    pass

import yaml, numpy as np, cv2
from core.lifecycle import LifecycleNode
from core.messages import SceneMsg, MotionState, Vec3, DiagStatus
from core.param_server import ParamClient
from core.transforms import get_tf

SCENE_LABELS = [
    "indoor_normal","indoor_crowded","outdoor_road","stable_plant",
    "restricted_zone","emergency","low_light","obstacle_close","clear_path","unknown"
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Gengar(LifecycleNode):

    def on_configure(self):
        with open("/opt/magi/config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        self._cfg       = cfg["models"]["magi2"]
        self._cam_cfg   = cfg["camera"]
        self._ipc_cfg   = cfg["ipc"]
        self._params    = ParamClient("magi2")
        self._tf        = get_tf()

        sz = self._cfg.get("input_size", [224, 224])
        self._input_w, self._input_h = sz[0], sz[1]
        self._interpreter = None
        self._shm         = None
        self._frame_count = 0
        self._accel_hist  = []
        self._inf_times   = []
        self._last_imu    = None

        self._load_model()
        self._load_shm()
        self.create_publisher("/scene")
        self.create_subscription("/sensors", self._on_sensor)
        self.log.info("Gengar configured")

    def _load_model(self):
        path    = self._cfg["path"]
        threads = self._cfg.get("num_threads", 1)
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=path, num_threads=threads)
            self._interpreter.allocate_tensors()
            self._in_idx  = self._interpreter.get_input_details()[0]["index"]
            self._out_idx = self._interpreter.get_output_details()[0]["index"]
            self.log.info(f"Model loaded: {path}")
        except Exception as e:
            self.log.error(f"Model load failed: {e}")

    def _load_shm(self):
        if not self._cam_cfg.get("enabled", False):
            return
        try:
            from common.ipc import SharedFrame
            self._shm = SharedFrame(
                name=self._ipc_cfg["shm_camera"],
                size=self._ipc_cfg["shm_camera_size"],
                create=False
            )
            self._frame_shape = (self._cam_cfg["height"], self._cam_cfg["width"], 3)
        except Exception as e:
            self.log.warning(f"SHM unavailable: {e}")

    def on_activate(self):
        self._run_every = 10
        self.log.info("Gengar active")

    def _compute_motion(self, imu) -> MotionState:
        if not imu:
            return MotionState()
        accel = np.array(imu.get("accel", [0, 0, 1.0]), dtype=np.float32)
        lin   = accel - np.array([0.0, 0.0, 1.0])
        mag   = float(np.linalg.norm(lin))
        self._accel_hist.append(accel)
        if len(self._accel_hist) > 10:
            self._accel_hist.pop(0)
        jerk = float(np.linalg.norm(self._accel_hist[-1] - self._accel_hist[-2])) \
               if len(self._accel_hist) >= 2 else 0.0

        # Use TF to get IMU frame (could apply rotation correction here)
        tf = self._tf.get_transform("base_link", "imu_link")

        return MotionState(
            motion_mag = round(mag, 4),
            jerk       = round(jerk, 4),
            is_moving  = mag > 0.05,
            velocity   = Vec3(),
        )

    def _on_sensor(self, data: dict):
        self._frame_count += 1
        if data.get("imu"):
            self._last_imu = data["imu"]

        if self._frame_count % self._run_every != 0:
            return

        motion = self._compute_motion(self._last_imu)
        tensor = self._get_frame_tensor()

        t0 = time.monotonic()
        scene, conf, top3, anomaly = self._infer(tensor, motion)
        inf_ms = (time.monotonic() - t0) * 1000
        self._inf_times.append(inf_ms)

        msg = SceneMsg(
            header        = self.next_header("camera_link"),
            scene         = scene,
            scene_id      = SCENE_LABELS.index(scene) if scene in SCENE_LABELS else -1,
            confidence    = conf,
            top3          = top3,
            anomaly_score = anomaly,
            motion        = motion,
            latency_ms    = inf_ms,
        )
        self.publish("/scene", msg)

        avg_ms = sum(self._inf_times) / max(1, len(self._inf_times))
        self.publish_diag(DiagStatus.OK, f"Scene={scene} anomaly={anomaly:.2f}",
                          {"inf_ms": round(avg_ms, 1), "scene": scene, "anomaly": round(anomaly, 3)})

    def _get_frame_tensor(self):
        if self._shm is None:
            return np.zeros((1, self._input_h, self._input_w, 3), dtype=np.float32)
        try:
            frame  = self._shm.read(self._frame_shape)
            frame  = cv2.resize(frame, (self._input_w, self._input_h))
            frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = (frame.astype(np.float32) / 255.0 - _MEAN) / _STD
            return np.expand_dims(tensor, axis=0)
        except Exception:
            return None

    def _infer(self, tensor, motion: MotionState):
        if self._interpreter is None or tensor is None:
            return "unknown", 0.0, [], 0.0
        try:
            self._interpreter.set_tensor(self._in_idx, tensor)
            self._interpreter.invoke()
            probs   = self._interpreter.get_tensor(self._out_idx)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            scene   = SCENE_LABELS[top_idx] if top_idx < len(SCENE_LABELS) else str(top_idx)
            top3    = [{"scene": SCENE_LABELS[i] if i < len(SCENE_LABELS) else str(i),
                        "confidence": float(round(probs[i], 4))}
                       for i in np.argsort(probs)[::-1][:3]]
            entropy  = float(-np.sum(probs * np.log(probs + 1e-9)))
            anomaly  = min(1.0, (entropy / 3.0) * (1 + motion.motion_mag))
            return scene, round(top_conf, 4), top3, round(anomaly, 4)
        except Exception as e:
            self.log.error(f"Inference error: {e}")
            return "unknown", 0.0, [], 0.0


if __name__ == "__main__":
    Gengar(node_id="magi2", cpu_core=2).boot()
