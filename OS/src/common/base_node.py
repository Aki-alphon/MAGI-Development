"""
MAGI OS — Base Node Class
/opt/magi/src/common/base_node.py

All three MAGI inference nodes inherit from this base.
Handles: TFLite loading, CPU pinning, ZMQ IPC, lifecycle.
"""

import os
import time
import signal
import abc
import yaml
import numpy as np
from common.logger import get_logger
from common.ipc import Subscriber, Pusher


class MAGIBaseNode(abc.ABC):
    """
    Abstract base for MAGI-1, MAGI-2, MAGI-3 inference nodes.

    Subclasses must implement:
        preprocess(sensor_data)  → input tensor
        postprocess(output)      → result dict
        on_result(result)        → optional side-effect hook
    """

    def __init__(self, node_id: str, cpu_core: int, config_path: str = "/opt/magi/config/config.yaml"):
        self.node_id = node_id
        self.log = get_logger(node_id)

        # Pin to assigned CPU core
        os.sched_setaffinity(0, {cpu_core})
        self.log.info(f"{node_id} pinned to CPU core {cpu_core}")

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_cfg = self.cfg["models"][node_id.replace("-", "")]
        self.ipc_cfg   = self.cfg["ipc"]

        self._running = True
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

        self._load_model()
        self._setup_ipc()

    # ─── Model Loading ───────────────────────────────────────────────────────

    def _load_model(self):
        """Load TFLite model with optional XNNPACK delegate."""
        model_path = self.model_cfg["path"]
        num_threads = self.model_cfg.get("num_threads", 1)
        use_xnnpack = self.model_cfg.get("use_xnnpack", True)

        self.log.info(f"Loading model: {model_path}")

        try:
            import tflite_runtime.interpreter as tflite

            if use_xnnpack:
                delegate = tflite.load_delegate("libXNNPACK.so.1",
                                                 options={"num_threads": num_threads})
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate],
                    num_threads=num_threads
                )
            else:
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads
                )

            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.log.info(f"Model loaded — input: {self.input_details[0]['shape']}")

        except Exception as e:
            self.log.error(f"Model load failed: {e}")
            self.interpreter = None

    # ─── IPC Setup ──────────────────────────────────────────────────────────

    def _setup_ipc(self):
        """Connect to sensor hub PUB and bind result PUSH socket."""
        self.sub = Subscriber(
            address=self.ipc_cfg["sensor_pub"],
            topics=["sensors", "camera"],
            timeout_ms=500
        )
        result_addr = self.ipc_cfg.get(f"{self.node_id.replace('-','')}_result")
        if result_addr:
            self.pusher = Pusher(result_addr)
        else:
            self.pusher = None

    # ─── Inference ──────────────────────────────────────────────────────────

    def infer(self, input_tensor: np.ndarray) -> list:
        """Run TFLite inference and return raw output tensors."""
        if self.interpreter is None:
            return []
        t0 = time.monotonic()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(d["index"]) for d in self.output_details]
        elapsed_ms = (time.monotonic() - t0) * 1000
        self.log.debug(f"Inference: {elapsed_ms:.1f} ms")
        return outputs

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def _shutdown(self, sig, frame):
        self.log.info(f"{self.node_id} shutting down...")
        self._running = False

    def run(self):
        self.log.info(f"{self.node_id} running...")
        while self._running:
            msg = self.sub.receive()
            if msg is None:
                continue
            topic, data = msg

            try:
                input_tensor = self.preprocess(topic, data)
                if input_tensor is None:
                    continue

                outputs = self.infer(input_tensor)
                result  = self.postprocess(outputs, data)
                result["node"]  = self.node_id
                result["ts"]    = time.time()

                if self.pusher:
                    self.pusher.send(result)

                self.on_result(result)

            except Exception as e:
                self.log.error(f"Processing error: {e}", exc_info=True)

        self.sub.close()
        if self.pusher:
            self.pusher.close()
        self.log.info(f"{self.node_id} stopped.")

    # ─── Abstract Methods ────────────────────────────────────────────────────

    @abc.abstractmethod
    def preprocess(self, topic: str, data: dict) -> np.ndarray | None:
        """Convert raw sensor/camera data to model input tensor."""
        ...

    @abc.abstractmethod
    def postprocess(self, outputs: list, raw_data: dict) -> dict:
        """Convert raw model outputs to structured result dict."""
        ...

    def on_result(self, result: dict):
        """Optional hook — called after every inference. Override if needed."""
        pass
