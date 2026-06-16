"""
MAGI OS v2 — MAGI-1 Melchior: Object Detection Node
src/magi1/melchior.py

Upgraded to LifecycleNode. Publishes typed DetectionMsg to /detections.
Reads camera frames from POSIX shared memory (zero-copy).
Pinned to CPU Core 1.
"""

import os, sys, time
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {1})
except (AttributeError, OSError):
    pass

import yaml, numpy as np, cv2
from core.lifecycle import LifecycleNode, State
from core.messages import DetectionMsg, Detection, BoundingBox, DiagStatus
from core.param_server import ParamClient

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]


class Melchior(LifecycleNode):

    def on_configure(self):
        with open("/opt/magi/config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        self._cfg       = cfg["models"]["magi1"]
        self._cam_cfg   = cfg["camera"]
        self._ipc_cfg   = cfg["ipc"]
        self._params    = ParamClient("magi1")

        sz = self._cfg.get("input_size", [320, 320])
        self._input_w, self._input_h = sz[0], sz[1]
        self._interpreter = None
        self._shm         = None
        self._frame_count = 0
        self._inf_times   = []

        self._load_model()
        self._load_shm()
        self.create_publisher("/detections")
        self.create_subscription("/sensors", self._on_sensor)
        self.create_subscription("/scene", self._on_scene)
        self._last_plant_time = 0.0
        self.log.info("Melchior configured")

    def _load_model(self):
        path = self._cfg["path"]
        threads = self._cfg.get("num_threads", 1)
        try:
            import tflite_runtime.interpreter as tflite
            if self._cfg.get("use_xnnpack", True):
                try:
                    delegate = tflite.load_delegate("libXNNPACK.so.1", {"num_threads": threads})
                    self._interpreter = tflite.Interpreter(model_path=path, experimental_delegates=[delegate])
                except Exception:
                    self._interpreter = tflite.Interpreter(model_path=path, num_threads=threads)
            else:
                self._interpreter = tflite.Interpreter(model_path=path, num_threads=threads)
            self._interpreter.allocate_tensors()
            self._in_idx  = self._interpreter.get_input_details()[0]["index"]
            self._out_det = self._interpreter.get_output_details()
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
            self.log.info("Camera shared memory reader ready")
        except Exception as e:
            self.log.warning(f"SHM unavailable: {e}")

    def on_activate(self):
        self._run_every = 5
        self.log.info(f"Melchior active — {self._input_w}×{self._input_h}")

    def on_deactivate(self):
        self.log.info("Melchior deactivated")

    def _on_scene(self, data: dict):
        """Called on every /scene message — wakes up Core 1 if plant is detected."""
        scene = data.get("scene", "unknown")
        if scene == "stable_plant":
            self._last_plant_time = time.time()
            if self.state == State.INACTIVE:
                self.log.info("Stable plant detected! Activating disease classification on Core 1...")
                self.activate()

    def _on_sensor(self, data: dict):
        """Called on every /sensors message — throttle inference."""
        # Gating check
        if self.state != State.ACTIVE:
            return

        # Timeout check
        if time.time() - self._last_plant_time > 5.0:
            self.log.info("No stable plant detected for 5 seconds. Deactivating Core 1 to save power...")
            self.deactivate()
            return

        self._frame_count += 1
        if self._frame_count % self._run_every != 0:
            return

        # Get live param updates
        self._conf_thresh = float(self._params.get("confidence_threshold", 0.5))

        tensor = self._get_frame_tensor()
        if tensor is None:
            return

        t0 = time.monotonic()
        detections = self._infer(tensor)
        inf_ms = (time.monotonic() - t0) * 1000
        self._inf_times.append(inf_ms)
        if len(self._inf_times) > 50:
            self._inf_times.pop(0)

        msg = DetectionMsg(
            header     = self.next_header("camera_link"),
            detections = detections,
            count      = len(detections),
            fps        = self._frame_count / max(1, time.monotonic()),
            latency_ms = inf_ms,
        )
        self.publish("/detections", msg)

        if detections:
            self.log.info(f"Detected {len(detections)}: {[d.label for d in detections]}")

        # Diagnostics heartbeat
        avg_ms = sum(self._inf_times) / max(1, len(self._inf_times))
        self.publish_diag(DiagStatus.OK, f"Running @ {1000/max(1,avg_ms):.1f} Hz",
                          {"inf_ms": round(avg_ms, 1), "detections": len(detections)})

    def _get_frame_tensor(self):
        if self._shm is None:
            return np.zeros((1, self._input_h, self._input_w, 3), dtype=np.float32)
        try:
            frame = self._shm.read(self._frame_shape)
            frame = cv2.resize(frame, (self._input_w, self._input_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
        except Exception:
            return None

    def _infer(self, tensor) -> list[Detection]:
        if self._interpreter is None:
            return []
        try:
            self._interpreter.set_tensor(self._in_idx, tensor)
            self._interpreter.invoke()
            outs    = self._out_det
            boxes   = self._interpreter.get_tensor(outs[0]["index"])[0]
            classes = self._interpreter.get_tensor(outs[1]["index"])[0]
            scores  = self._interpreter.get_tensor(outs[2]["index"])[0]
            count   = int(self._interpreter.get_tensor(outs[3]["index"])[0]) if len(outs) > 3 else len(scores)
            detections = []
            for i in range(min(count, len(scores))):
                if scores[i] < self._conf_thresh:
                    continue
                cid = int(classes[i])
                b   = boxes[i]
                detections.append(Detection(
                    label      = COCO_LABELS[cid] if cid < len(COCO_LABELS) else str(cid),
                    class_id   = cid,
                    confidence = float(round(scores[i], 4)),
                    bbox       = BoundingBox(b[0], b[1], b[2], b[3]),
                ))
            return detections
        except Exception as e:
            self.log.error(f"Inference error: {e}")
            return []


if __name__ == "__main__":
    node = Melchior(node_id="magi1", cpu_core=1)
    if node.configure():
        node.spin()
    node.shutdown()
